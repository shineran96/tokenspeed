"""Independent sessions with an optional one-time barrier after prime."""

import argparse
import concurrent.futures
import json
import os
import statistics
import threading
import time
from pathlib import Path

from client_v2 import ROOT, URL, client, collect, quantile, wait_ready, write, workers


def monitor(stop, path):
    with client() as s, path.open("w") as f:
        while not stop.is_set():
            try:
                data = workers()
                f.write(
                    json.dumps(dict(time=time.time(), workers=data.get("workers", [])))
                    + "\n"
                )
                f.flush()
            except Exception as e:
                f.write(json.dumps(dict(time=time.time(), error=repr(e))) + "\n")
                f.flush()
            stop.wait(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    c = args.concurrency
    sessions = json.loads(Path(os.environ["SESSIONS_FILE"]).read_text())[:c]
    prime = int(os.environ["PRIME_TOKENS"])
    chunk = int(os.environ["CHUNK_TOKENS"])
    rounds = int(os.environ["ROUNDS"])
    output = int(os.environ["OUTPUT_TOKENS"])
    assert len(sessions) == c and all(
        len(s["input_ids"]) >= prime + rounds * chunk for s in sessions
    )
    dest = ROOT / "results" / args.tag
    dest.mkdir(parents=True, exist_ok=False)
    lengths = [prime + i * chunk for i in range(rounds + 1)]
    for turn, n in enumerate(lengths):
        (dest / f"turn{turn}-{n}").mkdir()
    wait_ready()
    initial = threading.Barrier(c)
    prime_slots = threading.BoundedSemaphore(8)
    failure = threading.Event()
    stop = threading.Event()
    lock = threading.Lock()
    prime_barrier_enabled = os.environ.get("PRIME_BARRIER", "1")
    if prime_barrier_enabled not in {"0", "1"}:
        raise ValueError("PRIME_BARRIER must be 0 or 1")
    prime_barrier = (
        threading.Barrier(
            c,
            action=lambda: write(
                dest / "prime-barrier.json",
                dict(
                    concurrency=c,
                    released_at=time.time(),
                    note="All prime requests completed and passed cache validation.",
                ),
            ),
        )
        if prime_barrier_enabled == "1"
        else None
    )
    progress = {str(i): dict(completed_turn=-1) for i in range(c)}
    watcher = threading.Thread(target=monitor, args=(stop, dest / "workers.jsonl"))
    watcher.start()

    def run_session_impl(index, session):
        initial.wait(timeout=60)
        previous_end = None
        results = []
        for turn, n in enumerate(lengths):
            if failure.is_set():
                raise RuntimeError(
                    "Another session failed; do not issue further requests"
                )
            path = dest / f"turn{turn}-{n}" / f"request-{index}.json"
            if turn == 0:
                with prime_slots:
                    if failure.is_set():
                        raise RuntimeError("Prime aborted after prior failure")
                    result = collect(index, session["input_ids"][:n], None, path)
            else:
                result = collect(index, session["input_ids"][:n], None, path)
            result.update(
                session_id=session["id"],
                turn=turn,
                previous_request_end=previous_end,
                next_submit_gap_s=(
                    None if previous_end is None else result["start"] - previous_end
                ),
            )
            write(path, result)
            expected = 0 if turn == 0 else n - chunk
            cached = result.get("meta", {}).get("cached_tokens")
            result["expected_cached_tokens"] = expected
            result["cache_full_hit"] = cached == expected
            result["cache_shortfall_tokens"] = expected - cached if isinstance(cached, int) else None
            write(path, result)
            if not result["success"] or not isinstance(cached, int) or not 0 <= cached <= expected:
                failure.set()
                raise RuntimeError(
                    f"Session {index} turn {turn} failed/cache mismatch: {path}"
                )
            # SMG omits acceptance metadata; audit server logs by request ID after each case.
            if c == 1 and turn == 1 and cached < expected - 256:
                raise RuntimeError(f"Prefix cache not verified before the sweep: {cached}/{expected}")
            results.append(result)
            previous_end = result["end"]
            with lock:
                progress[str(index)] = dict(
                    completed_turn=turn,
                    input_tokens=n,
                    ttft_s=result["ttft_s"],
                    tpot_ms=result["tpot_ms"],
                )
                write(
                    ROOT / "status.json",
                    dict(
                        phase="running", concurrency=c, tag=args.tag, sessions=progress
                    ),
                )
            if turn == 0 and prime_barrier is not None:
                prime_barrier.wait(timeout=7200)
        return results

    def run_session(index, session):
        try:
            return run_session_impl(index, session)
        except BaseException:
            failure.set()
            initial.abort()
            if prime_barrier is not None:
                prime_barrier.abort()
            raise

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
            futures = [pool.submit(run_session, i, s) for i, s in enumerate(sessions)]
            for f in concurrent.futures.as_completed(futures):
                f.result()
    finally:
        stop.set()
        watcher.join()
    summaries = []
    for turn, n in enumerate(lengths):
        folder = dest / f"turn{turn}-{n}"
        rs = [json.loads((folder / f"request-{i}.json").read_text()) for i in range(c)]
        summary = dict(
            concurrency=c,
            turn=turn,
            input_tokens=n,
            output_tokens=output,
            successful_requests=c,
            expected_cached_tokens=0 if turn == 0 else n - chunk,
            cached_tokens=[r["meta"]["cached_tokens"] for r in rs],
        )
        for key in ["ttft_s", "tpot_ms", "e2e_s"]:
            values = [r[key] for r in rs]
            summary.update(
                {
                    key + "_mean": statistics.mean(values),
                    key + "_p50": quantile(values, 0.5),
                    key + "_p95": quantile(values, 0.95),
                }
            )
        summary["note"] = (
            "Turn cohort overlaps other turns; its window is not an isolated batch."
        )
        write(folder / "summary.json", summary)
        summaries.append(summary)
    write(dest / "summary.json", summaries)
    write(
        ROOT / "status.json",
        dict(
            phase="concurrency_complete", concurrency=c, tag=args.tag, sessions=progress
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write(ROOT / "status.json", dict(phase="failed", error=repr(e)))
        raise
