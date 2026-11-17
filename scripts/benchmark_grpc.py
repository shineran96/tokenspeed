"""Measure paired PD traffic at explicit per-rank offered concurrency."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import time

import grpc
from google.protobuf.json_format import MessageToDict
from grpc_health.v1 import health_pb2, health_pb2_grpc
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2 as pb
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2_grpc as rpc
from transformers import AutoTokenizer


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def counts_from_loads(state):
    rows = state.get("loads", [])
    running = {int(r.get("dp_rank", 0)): int(r.get("num_running_reqs", 0)) for r in rows}
    waiting = {int(r.get("dp_rank", 0)): int(r.get("num_waiting_reqs", 0)) for r in rows}
    return running, waiting


def request_metrics(requests):
    complete = [r for r in requests if r["first"] is not None and r["count"] > 1]
    if not complete:
        return {}
    tpot = [(r["last"] - r["first"]) * 1000 / (r["count"] - 1) for r in complete]
    ttft = [r["first"] - r["sent"] for r in complete]
    duration = max(r["last"] for r in complete) - min(r["first"] for r in complete)
    return {
        "tpot_p50_ms": statistics.median(tpot),
        "tpot_p90_ms": percentile(tpot, 0.9),
        "ttft_p50_s": statistics.median(ttft),
        "ttft_p90_s": percentile(ttft, 0.9),
        "output_tok_s": sum(r["count"] for r in complete) / duration,
    }


def prepare_inputs(root, output_tokens, tokenizer_path, provenance_path):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    rng = random.Random(20261028)
    prefix = tokenizer.decode(
        tokenizer(tokenizer.decode([rng.randrange(1000, 100000) for _ in range(128864)]))["input_ids"][:128800]
    )
    prefix_ids = tokenizer(prefix)["input_ids"]
    inputs = [
        prefix_ids + tokenizer(
            tokenizer.decode([rng.randrange(1000, 100000) for _ in range(192)]),
            add_special_tokens=False,
        )["input_ids"][:128]
        for _ in range(64)
    ]
    provenance = {
        "seed": 20261028,
        "prompt_lengths": list(map(len, inputs)),
        "input_sha256": [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in inputs],
        "output_tokens": output_tokens,
    }
    previous = json.loads(Path(
        provenance_path
    ).read_text())
    if provenance != previous:
        raise RuntimeError("Inputs differ from the prior benchmark")
    root.joinpath("input-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    root.joinpath("input_ids.json").write_text(json.dumps(inputs) + "\n")
    return [pb.GenerateRequest(
        tokenized=pb.TokenizedInput(input_ids=ids), stream=True,
        sampling_params=pb.SamplingParams(max_new_tokens=output_tokens, temperature=0, ignore_eos=True),
    ).SerializeToString() for ids in inputs]


async def run(args):
    args.output_dir.mkdir(exist_ok=False)
    templates = prepare_inputs(args.output_dir, args.output_tokens, args.tokenizer, args.provenance)
    options = [("grpc.enable_http_proxy", 0)]
    channels = [grpc.aio.insecure_channel(address, options=options)
                for address in (args.decode_address, args.prefill_address)
                for _ in range(8)]
    decode_bank = [rpc.TokenSpeedSchedulerStub(c) for c in channels[:8]]
    prefill_bank = [rpc.TokenSpeedSchedulerStub(c) for c in channels[8:]]
    decode = decode_bank[0]

    async def loads():
        return MessageToDict(await decode.GetLoads(pb.GetLoadsRequest(), timeout=10), preserving_proto_field_name=True)

    async def health():
        for channel in (channels[0], channels[8]):
            response = await health_pb2_grpc.HealthStub(channel).Check(
                health_pb2.HealthCheckRequest(service="tokenspeed.grpc.scheduler.TokenSpeedScheduler"), timeout=10)
            if response.status != health_pb2.HealthCheckResponse.SERVING:
                raise RuntimeError("PD service is not SERVING")

    async def idle():
        for _ in range(120):
            state = await loads()
            running, waiting = counts_from_loads(state)
            if len(running) == 16 and not any(running.values()) and not any(waiting.values()):
                return state
            await asyncio.sleep(1)
        raise RuntimeError("Decode did not become idle; no new requests sent")

    try:
        await health()
        info = MessageToDict(await decode.GetServerInfo(pb.GetServerInfoRequest(), timeout=10), preserving_proto_field_name=True)
        config = info["server_args"]
        if not all(config.get(k) == 16 for k in ("world_size", "data_parallel_size", "ep_size")):
            raise RuntimeError("Expected DP16/EP16")
        if (config["deepep_dispatch_dtype"], config["deepep_combine_dtype"]) != ("nvfp4", "fp8"):
            raise RuntimeError("Expected NVFP4 dispatch / FP8 combine")
        capacity = int(config["max_num_seqs"]) // 16
        args.output_dir.joinpath("server-info.json").write_text(json.dumps(info, indent=2) + "\n")
        await asyncio.gather(*(c.channel_ready() for c in channels))
        for batch in args.per_rank:
            total = batch * 16
            active_target = min(batch, capacity)
            for repeat in range(1, args.repeats + 1):
                await health()
                initial = await idle()
                needed = max(json.loads(args.output_dir.joinpath("input-provenance.json").read_text())["prompt_lengths"]) + args.output_tokens
                if any(int(r["max_total_num_tokens"]) < needed * active_target for r in initial["loads"]):
                    raise RuntimeError("Insufficient decode token capacity")
                root = args.output_dir / f"b{batch}" / f"repeat{repeat}"
                root.mkdir(parents=True)
                errors = []
                samples = []
                requests = [{"index": i, "rank": i % 16, "prompt_index": i % 64,
                             "sent": None, "first": None, "last": None, "count": 0,
                             "token_ids": []} for i in range(total)]
                stamp = time.strftime("%Y%m%d-%H%M%S")
                room = (time.time_ns() % (1 << 61)) & ~15
                log_start = args.decode_log.stat().st_size
                stop = asyncio.Event()

                async def consume(call, index, is_decode):
                    try:
                        async for response in call:
                            if is_decode and response.HasField("chunk") and response.chunk.token_ids:
                                now = time.time()
                                row = requests[index]
                                if row["first"] is None:
                                    row["first"] = now
                                row["last"] = now
                                row["count"] += len(response.chunk.token_ids)
                                row["token_ids"].extend(response.chunk.token_ids)
                        return True
                    except grpc.RpcError as exc:
                        errors.append({"request": index, "decode": is_decode,
                                       "code": exc.code().name, "details": exc.details()})
                        return False

                async def monitor():
                    next_log = 0
                    while not stop.is_set():
                        state = await loads()
                        running, waiting = counts_from_loads(state)
                        sample = {"time": time.time(), "counts": running, "waiting": waiting,
                                  "first_responses": sum(r["first"] is not None for r in requests),
                                  "output_tokens": sum(r["count"] for r in requests),
                                  "valid": len(running) == 16 and set(running.values()) == {active_target}}
                        samples.append(sample)
                        if time.monotonic() >= next_log:
                            print("LOAD", batch, repeat, json.dumps({k: sample[k] for k in ("counts", "waiting", "first_responses")}), flush=True)
                            next_log = time.monotonic() + 10
                        await asyncio.sleep(0.4)

                print("BEGIN", json.dumps({"per_rank_offered": batch, "total": total, "repeat": repeat,
                                            "active_target": active_target, "queue_test": batch > capacity}), flush=True)
                begin = time.time()
                tasks = []
                watcher = asyncio.create_task(monitor())
                try:
                    for offset in range(0, total, 16):
                        paired = []
                        for i in range(offset, offset + 16):
                            request = pb.GenerateRequest.FromString(templates[i % 64])
                            request.request_id = f"w4a8-matrix-{stamp}-b{batch}-r{repeat}-{i}"
                            request.kv_bootstrap_info.CopyFrom(pb.KvBootstrapInfo(
                                bootstrap_host="33.29.38.168", bootstrap_port=8998, bootstrap_room=room + i))
                            request.data_parallel_rank = i % 16
                            requests[i]["sent"] = time.time()
                            tasks.append(asyncio.create_task(consume(decode_bank[i % 8].Generate(request, timeout=args.timeout), i, True)))
                            pre = pb.GenerateRequest()
                            pre.CopyFrom(request)
                            pre.ClearField("data_parallel_rank")
                            pre.sampling_params.max_new_tokens = 1
                            paired.append((i, pre))
                        for i, request in paired:
                            tasks.append(asyncio.create_task(consume(prefill_bank[i % 8].Generate(request, timeout=args.timeout), i, False)))
                        await asyncio.sleep(0)
                    completed = await asyncio.gather(*tasks)
                finally:
                    stop.set()
                    await watcher
                finished = time.time()
                with args.decode_log.open("rb") as stream:
                    stream.seek(log_start)
                    text = stream.read().decode(errors="replace")
                root.joinpath("node1-decode-window.log").write_text(text)
                acceptance = [float(v) for v in re.findall(r"avg_accept_len: ([0-9.]+)", text)]
                valid = [s for s in samples if s["valid"]]
                max_min = max((min(s["counts"].values()) for s in samples if len(s["counts"]) == 16), default=0)
                record = {
                    "per_rank_offered": batch, "total_concurrency": total, "repeat": repeat,
                    "dispatch": "nvfp4", "combine": "fp8", "profiled": False,
                    "scheduler_capacity_per_rank": capacity, "active_target": active_target,
                    "queue_test": batch > capacity, "verified_target_window": len(valid) >= 2,
                    "target_window_samples": len(valid), "max_simultaneous_min_per_rank": max_min,
                    "peak_running_per_rank": {r: max((s["counts"].get(r, 0) for s in samples), default=0) for r in range(16)},
                    "peak_waiting_per_rank": {r: max((s["waiting"].get(r, 0) for s in samples), default=0) for r in range(16)},
                    "all_streams_completed": all(completed), "errors": errors,
                    "ok": sum(r["count"] == args.output_tokens for r in requests),
                    "wall_s": finished - begin, "acceptance_log_mean": statistics.mean(acceptance) if acceptance else None,
                    "output_token_counts": [r["count"] for r in requests],
                    **request_metrics(requests),
                }
                if valid:
                    record["target_window_first_time"] = valid[0]["time"]
                    record["target_window_last_time"] = valid[-1]["time"]
                if record["ok"] != total:
                    errors.append({"code": "INCOMPLETE_OUTPUT", "details": "Unexpected output token count"})
                root.joinpath("requests.json").write_text(json.dumps(requests) + "\n")
                root.joinpath("loads.json").write_text(json.dumps(samples) + "\n")
                root.joinpath("capture.json").write_text(json.dumps(record, indent=2) + "\n")
                with args.output_dir.joinpath("results.jsonl").open("a") as output:
                    output.write(json.dumps(record) + "\n")
                print("RESULT", json.dumps({k: v for k, v in record.items() if k != "output_token_counts"}), flush=True)
                if errors or not all(completed):
                    raise RuntimeError("Paired stream failed; stopping further experiments")
                await idle()
                await asyncio.sleep(3)
        await health()
        args.output_dir.joinpath("final-status.json").write_text(json.dumps({"health": "SERVING", "loads": await idle()}, indent=2) + "\n")
        print("ALL_COMPLETE", flush=True)
    finally:
        await asyncio.gather(*(c.close() for c in channels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--provenance", type=Path, default=Path(__file__).resolve().parents[1]/"data/input-provenance.json")
    parser.add_argument("--per-rank", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--decode-log", type=Path, required=True)
    parser.add_argument("--decode-address", required=True)
    parser.add_argument("--prefill-address", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    if any(b <= 0 for b in args.per_rank) or args.repeats <= 0:
        parser.error("Concurrency and repeats must be positive")
    asyncio.run(run(args))
