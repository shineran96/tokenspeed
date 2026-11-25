# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Audit archived client events and reproduce the staged-decode comparison."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "single-mega": "single-mega-results.tar.gz",
    "dual-mega": "dual-mega-results.tar.gz",
    "humming-ep16": "humming-ep16-results-final.tar.gz",
}


def audit(mode, path):
    batches = {}
    pattern = re.compile(r"(?:^|/)results/c(32|64|128|256)/(formal[123])/(\d+|control)\.json$")
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            match = pattern.search(member.name)
            if not member.isfile() or not match:
                continue
            concurrency, phase, name = match.groups()
            batch = batches.setdefault((int(concurrency), phase), {})
            assert name not in batch, f"Duplicate result: {member.name}"
            batch[name] = json.load(archive.extractfile(member))
    rounds = []
    for concurrency in (32, 64, 128, 256):
        for phase in ("formal1", "formal2", "formal3"):
            batch = batches[(concurrency, phase)]
            assert set(batch) == {"control", *(str(i) for i in range(concurrency))}
            controls = batch["control"]
            gates = [event for event in controls if event["kind"] == "all_transfers_completed"]
            assert len(gates) == 1 and gates[0]["count"] == concurrency
            gate = gates[0]["time"]
            rows = [batch[str(i)] for i in range(concurrency)]
            tpots = []
            for row in rows:
                assert row["success"]
                assert row["meta"]["prompt_tokens"] == 131072
                assert row["meta"]["completion_tokens"] == 2048
                assert row["meta"]["cached_tokens"] == 130816
                events = row["events"]
                assert events[0][0] > gate and events[-1][1] == 2048
                assert all(b[0] >= a[0] and b[1] > a[1] for a, b in zip(events, events[1:]))
                tpot = (events[-1][0] - events[0][0]) * 1000 / (2048 - 1)
                assert math.isclose(tpot, row["tpot_ms"], rel_tol=1e-10)
                tpots.append(tpot)
            first = min(row["events"][0][0] for row in rows)
            last = max(row["events"][-1][0] for row in rows)
            assert min(row["events"][-1][0] for row in rows) > max(row["events"][0][0] for row in rows)
            rounds.append({
                "mode": mode, "concurrency": concurrency, "phase": phase,
                "requests": concurrency, "mean_tpot_ms": statistics.mean(tpots),
                "output_tokens_per_s": concurrency * 2048 / (last - first),
                "first_event_tokens": sorted({row["events"][0][1] for row in rows}),
            })
    return rounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    expected = {row["file"]: row for row in json.loads((ROOT / "data/raw-archives.json").read_text())}
    rows = []
    for mode, name in ARCHIVES.items():
        path = args.raw_dir / name
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(block)
        assert checksum.hexdigest() == expected[name]["sha256"], name
        rows.extend(audit(mode, path))
    with (ROOT / "data/staged-decode-20261124/comparison.csv").open() as stream:
        for expected_row in csv.DictReader(stream):
            concurrency = int(expected_row["concurrency"])
            for mode in ARCHIVES:
                selected = [row for row in rows if row["mode"] == mode and row["concurrency"] == concurrency]
                assert len(selected) == 3
                for key in ("mean_tpot_ms", "output_tokens_per_s"):
                    actual = statistics.mean(row[key] for row in selected)
                    assert math.isclose(actual, float(expected_row[f"{mode}_{key}"]), rel_tol=1e-10), (mode, concurrency, key)
    result = {"status": "passed", "formal_rounds": len(rows), "formal_requests": sum(row["requests"] for row in rows), "rounds": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(rows)} rounds / {result['formal_requests']} requests against comparison.csv")


if __name__ == "__main__":
    main()
