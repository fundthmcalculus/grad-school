"""Tables of the inference-precision benchmark (`flm/infer_precision.py`) as markdown: ``python -m flm.infer_precision_report > report/infer_precision_tables.md``.

One block per checkpoint (the 100K F7, the 14M TinyStories model, the 14M Pile model): quality against fp32, GPU speed, CPU speed.  Every number comes from ``report/infer_precision_<name>.json``.
"""

import json
from pathlib import Path

NAMES = [
    ("F7_100K", "F7, 99,882 parameters, TinyStories (300M characters)"),
    ("M14_TinyStories", "14M, 14,067,715 parameters, TinyStories (300M characters)"),
    ("M14_Pile", "14M, 14,067,715 parameters, Pile sample (300M characters)"),
]
FMT = {
    "fp32": "fp32 (reference)",
    "fp16": "fp16",
    "bf16": "bf16",
    "int8": "int8, per row",
    "int8-lazy": "int8 held, rebuilt each pass",
}


def block(name, title):
    f = Path(f"report/infer_precision_{name}.json")
    if not f.exists():
        return f"## {title}\n\n(no report yet)\n"
    d = json.loads(f.read_text())
    fm = d["formats"]
    ref = fm["fp32"]
    out = [f"## {title}", ""]
    out += [
        "### Quality against fp32 (GPU arithmetic, 1M validation characters, 200 windows of 512 pieces, 20 greedy continuations of 64 pieces)",
        "",
        "| format | stored size | bits/char | change | top-1 agrees | KL (bits) | largest logit change | greedy identical | mean first difference |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k, v in fm.items():
        out.append(
            f"| {FMT[k]} | {v['bytes'] / 1e6:.3f} MB ({v['bytes'] / ref['bytes']:.2f}x) | {v['bpc']:.5f} | {v['bpc'] - ref['bpc']:+.5f} | {100 * v['top1_agreement']:.2f}% | {v['kl_bits']:.2e} | {v['max_dlogit']:.2f} | {int(round(v['greedy_identical'] * 20))}/20 | {v['mean_first_difference']:.1f} |"
        )
    out += [
        "",
        f"### GPU speed ({d['device']}; forward-only throughput at 16 x 512, and greedy decoding with batch 1 and no cache)",
        "",
        "| format | pieces/s | against fp32 | ms per piece, decoding | against fp32 |",
        "|---|---|---|---|---|",
    ]
    for k, v in fm.items():
        out.append(
            f"| {FMT[k]} | {v['pieces_per_s']:,} | {v['pieces_per_s'] / ref['pieces_per_s']:.2f}x | {v['ms_per_piece_decode']:.2f} | {v['ms_per_piece_decode'] / ref['ms_per_piece_decode']:.2f}x |"
        )
    cpu = [k for k, v in fm.items() if "cpu_ms_per_piece_decode" in v]
    if cpu:
        out += [
            "",
            f"### CPU speed and quality ({d.get('cpu_threads', 3)} threads; fp16 and int8 are stored as such and computed in fp32 on the CPU, which has no fast fp16 or int8 path for these layers)",
            "",
            "| format | pieces/s (4 x 512) | against fp32 | ms per piece, decoding | against fp32 | bits/char (300K characters) |",
            "|---|---|---|---|---|---|",
        ]
        r0 = fm["fp32"]
        for k in cpu:
            v = fm[k]
            out.append(
                f"| {FMT[k]} | {v.get('cpu_pieces_per_s', '-'):,} | {v.get('cpu_pieces_per_s', 0) / r0.get('cpu_pieces_per_s', 1):.2f}x | {v['cpu_ms_per_piece_decode']:.2f} | {v['cpu_ms_per_piece_decode'] / r0['cpu_ms_per_piece_decode']:.2f}x | {v.get('cpu_bpc_300k', '-')} |"
            )
    return "\n".join(out) + "\n"


def board(out):
    """One row per (model, format) for the dashboard: the reports' numbers and their ratios against fp32."""
    rows = []
    for name, title in NAMES:
        f = Path(f"report/infer_precision_{name}.json")
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        fm = d["formats"]
        r0 = fm["fp32"]
        for k, v in fm.items():
            rows.append(
                dict(
                    id=f"{name}|{k}",
                    model=name,
                    model_label=title,
                    format=k,
                    format_label=FMT[k],
                    bytes=v["bytes"],
                    size_x=round(v["bytes"] / r0["bytes"], 3),
                    bpc=v["bpc"],
                    dbpc=round(v["bpc"] - r0["bpc"], 5),
                    top1=round(100 * v["top1_agreement"], 2),
                    kl=v["kl_bits"],
                    maxd=v["max_dlogit"],
                    greedy=int(round(v["greedy_identical"] * 20)),
                    first_diff=v["mean_first_difference"],
                    gpu_pps=v["pieces_per_s"],
                    gpu_x=round(v["pieces_per_s"] / r0["pieces_per_s"], 3),
                    gpu_ms=v["ms_per_piece_decode"],
                    gpu_ms_x=round(
                        v["ms_per_piece_decode"] / r0["ms_per_piece_decode"], 3
                    ),
                    cpu_pps=v.get("cpu_pieces_per_s"),
                    cpu_x=(
                        round(v["cpu_pieces_per_s"] / r0["cpu_pieces_per_s"], 3)
                        if v.get("cpu_pieces_per_s") and r0.get("cpu_pieces_per_s")
                        else None
                    ),
                    cpu_ms=v.get("cpu_ms_per_piece_decode"),
                    cpu_ms_x=(
                        round(
                            v["cpu_ms_per_piece_decode"]
                            / r0["cpu_ms_per_piece_decode"],
                            3,
                        )
                        if v.get("cpu_ms_per_piece_decode")
                        and r0.get("cpu_ms_per_piece_decode")
                        else None
                    ),
                    cpu_bpc=v.get("cpu_bpc_300k"),
                )
            )
    Path(out).write_text(json.dumps(rows))
    print(len(rows), "rows ->", out)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--board":
        board(sys.argv[2])
        raise SystemExit
    print("# Inference precision: fp32 against fp16 against int8 (tables)\n")
    for n, t in NAMES:
        print(block(n, t))
