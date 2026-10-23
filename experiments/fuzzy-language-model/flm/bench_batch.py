"""Throughput and peak memory against batch size for the 14M model (`PLAN_M14.md`): ``python -m flm.bench_batch [summary]``; the runs themselves are made by ``outputs/chain_bench.sh``.

One short bf16 run per batch size (about 3M characters, warm-up 5, no learning-rate claim) on the Pile sample, one job at a time, in ascending batch size.  A run is only started if the previous one's peak allocated memory, scaled to the next batch (activations grow with the batch), plus the allocator's overhead and the other applications' share stays under ``LIMIT_MB``: the owner does not want the card to spill into shared memory (everything slows down).  Tokens per second are steps x batch x context / training seconds of the result file.
"""

import json
import sys
from pathlib import Path

LIMIT_MB = 10_000  # total card memory the benchmark may project to (the card has 12,282 MB; about 1,500 MB belong to other applications)
OTHER_MB = 1_500
OVERHEAD = 1.15  # reserved memory over allocated memory
OUT = Path("outputs/v85-bench")


def projected_total(peak_alloc_mb: float, batch: int, next_batch: int) -> float:
    return OTHER_MB + OVERHEAD * peak_alloc_mb * next_batch / batch


def load(batch: int):
    f = next(OUT.glob(f"*bench_b{batch}_s0.json"), None)
    return json.loads(f.read_text()) if f else None


def summary(batches=(16, 24, 32, 40, 48, 64)):
    rows = []
    for b in batches:
        d = load(b)
        if d:
            tps = d["steps"] * b * d["args"]["ctx"] / d["train_seconds"]
            rows.append(
                (
                    b,
                    d["steps"],
                    d["train_seconds"],
                    round(tps),
                    d.get("peak_mem_mb"),
                    d["val_bpc"],
                )
            )
    base = rows[0][3] if rows else None
    print(
        "batch  steps  seconds  tokens/s  vs batch 16  peak MB (allocated)  validation bpc (not comparable: a few steps)"
    )
    for b, st, sec, tps, mem, bpc in rows:
        print(
            f"{b:5d} {st:6d} {sec:8.1f} {tps:9,d} {tps / base:11.2f}x {mem:14,d}  {bpc:.3f}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary()
    elif (
        len(sys.argv) > 3 and sys.argv[1] == "next"
    ):  # next <batch> <next_batch>: exit 1 if the projection exceeds the limit
        d = load(int(sys.argv[2]))
        ok = (
            d is not None
            and projected_total(d["peak_mem_mb"], int(sys.argv[2]), int(sys.argv[3]))
            <= LIMIT_MB
        )
        print("go" if ok else "stop")
        sys.exit(0 if ok else 1)
