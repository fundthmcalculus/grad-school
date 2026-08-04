"""Shared helpers for the experiment-table generators.

Keep the table scripts thin: they load data, fit models over a fixed set of
seeds, and hand (mean, std) cells to the emitters here. Everything about *how a
table renders* lives in this file, so all tables look the same.
"""

from __future__ import annotations

import csv
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager

# Fixed seeds -> every reported number is mean +/- std across these.
#
# Ten, not five. The proposal's tables were originally transcribed from 3-seed
# runs, and several Chapter 4 conclusions that looked decisive there turned out
# to sit inside the seed-to-seed spread once the harness widened it. Ten is the
# standard the chapters are now quoted at; narrowing it changes what the document
# can claim, so treat it as part of the protocol rather than a tuning knob.
# Override with REPRO_SEEDS="0,1,2" for a quick smoke run.
SEEDS = [int(s) for s in
         os.environ.get("REPRO_SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]

# Where emit() writes. `REPRO_OUTPUT_DIR` redirects it, which is what the
# controlled experiments under reproduce/experiments/ use: they run a *stock*
# generator many times over with one variable changed, and each repetition has to
# land somewhere of its own. Without the override every repetition overwrites the
# previous one at the canonical path, so the comparison the experiment exists to
# make is destroyed by the act of making it -- and the loose top-level files are
# gitignored scratch, so nothing would survive to diff afterwards either.
OUTPUT_DIR = os.environ.get("REPRO_OUTPUT_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reproduce", "outputs")

NA = "N/A"          # column that genuinely could not be produced (missing dep/data)


@contextmanager
def timed():
    """`with timed() as t: ...; t.seconds` -> wall-clock seconds of the block."""
    class _T:
        seconds = 0.0
    t = _T()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - start


def agg(values):
    """(mean, std) of a list; std=0 for a single value. Returns (None, None) if empty."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def cell(values, fmt="{:.3f}", na=NA):
    """Render a list of per-seed values as 'mean ± std', or the NA marker."""
    mean, std = agg(values)
    if mean is None:
        return na
    if std and std > 0:
        return f"{fmt.format(mean)} ± {fmt.format(std)}"
    return fmt.format(mean)


def _read(path, default=""):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return default


def machine():
    """Identify the host. Every emitted table records this.

    Timings are not portable between machines OR between runs on one machine, and
    this project has been bitten by both: the same generator at the same commit
    and seeds moved every wall-clock figure by ~45% across two runs, and a full
    suite was found to have run on a laptop while the prose described a
    workstation. A table that does not say where it came from cannot be compared
    with one that does.
    """
    cpu = ""
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    mem_kb = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal"):
            mem_kb = int(line.split()[1])
            break

    # /proc exists under Git Bash but NOT for a native Windows interpreter, which
    # is what `uv run` launches -- so on the very host the proposal calls "the
    # workstation" every table recorded `ram: unknown` and a registry-style CPU
    # string instead of "i9-14900HX". The machine block is the whole point of
    # checklist B1/B2; degrading silently on one platform defeats it. The shell
    # harness already reports these correctly, which is why the gap was invisible
    # in PROVENANCE.txt while being present in all thirteen emitted tables.
    if not mem_kb and sys.platform == "win32":
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                mem_kb = st.ullTotalPhys // 1024
        except Exception:  # noqa: BLE001 -- identity is best-effort, never fatal
            pass
    if not cpu and sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:  # noqa: BLE001
            pass
    gpu = ""
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().replace("\n", "; ")
    except (OSError, subprocess.SubprocessError):
        pass
    if not gpu:
        try:
            for line in subprocess.run(["lspci"], capture_output=True, text=True,
                                       timeout=10).stdout.splitlines():
                if "VGA compatible controller" in line or "3D controller" in line:
                    gpu = line.split(":", 2)[-1].strip()
                    break
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "host": platform.node() or "unknown",
        "os": platform.platform(terse=True),
        "cpu": cpu or platform.processor() or "unknown",
        "cores": f"{os.cpu_count()} logical",
        "ram": f"{mem_kb / 1048576:.1f} GiB" if mem_kb else "unknown",
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "n/a"),
        "gpu": gpu or "none detected",
        "python": platform.python_version(),
    }


def machine_block():
    """The machine stanza appended to every emitted Markdown table."""
    m = machine()
    return ("> **Machine.** " + " · ".join(
        f"{k}: {v}" for k, v in m.items() if v and v != "n/a") +
        "\n>\n> Wall-clock times are machine-dependent; ratios are not. Markdown "
        "tables report normalized ratios where a timing is involved, and the "
        "companion CSV carries the absolute seconds.")


def write_markdown(path, title, header, rows, note="", seeds=None):
    """Write a GitHub-flavored Markdown table.

    Encoding is pinned to UTF-8 rather than left to the platform default. Tables
    carry `±` in every cell and `λ`, `Δ`, `σ` in several headers; on Windows the
    default is cp1252, which encodes `±` but not `λ`, so two of eleven tables
    crashed in `f.write` *after* completing their entire numeric phase -- the
    same shape as the Chapter 5 driver that discarded its own results.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"**{title}**\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(c) for c in r) + " |\n")
        if note:
            f.write(f"\n> {note}\n")
        # `seeds` is an override, not a default, and it exists because the footer used
        # to state `SEEDS` unconditionally -- including on tables the harness does not
        # compute. `table_5_x_ch5_selection.py` does no arithmetic at all: it renders
        # `gated-minimax-selection/outputs/results.json`, whose driver runs at
        # `SEEDS = [0, 1, 2, 3, 4]` -- five restarts, and only for the NERFCM and
        # ConiVAT columns, which are the only cells in that battery carrying a
        # spread. Every other cell is one deterministic pass. So all three Chapter 5
        # tables carried a footer claiming ten seeds over a five-seed run, on a table
        # where most cells have no error bar at all. That is the `PROVENANCE.txt`
        # seed-count bug again -- a file stating a seed list it did not use -- one
        # directory over, and it is worse here because
        f.write(f"\n> Generated by `reproduce/`; "
                f"seeds = {SEEDS if seeds is None else seeds}. "
                f"`{NA}` marks a cell whose method/dataset was unavailable.\n")
        f.write(f"\n{machine_block()}\n")
    print(f"  wrote {path}")


def normalized_worst(values, fmt="{:.1f}×", worst_label="1.0× (worst)"):
    """Render a timing series against its WORST (slowest) entry.

    The slowest arm becomes the 1.0x baseline and every other arm reads as "this
    many times faster than the worst," which is the convenient direction: bigger
    is better, the reference is always the thing you are trying to beat, and no
    cell is a hard-to-read fraction.

    This is also the machine-independent view. Absolute seconds move with
    thermals, governor and host -- this project measured the same generator at
    the same commit and seeds swinging ~45% between runs -- while the ratio
    between two arms measured in the same pass moved under 3%. Ratios are what
    the chapters quote; the CSV keeps the seconds.

    Takes the per-arm MEAN of the raw per-seed times, and divides those. This is
    a ratio of means, not a mean of per-seed ratios -- the two differ, and the
    first is the right one here: a per-seed ratio would pair up timings that were
    taken at different moments in the same run and so carry different thermal
    state, which is the very noise the normalization exists to remove.

    The spread is deliberately not propagated onto the ratio. The seconds and
    their per-seed deviations are preserved in the companion CSV, and a variance
    on a ratio of two noisy means would invite more precision than the underlying
    measurement supports. Read the CSV when the spread matters.

    `None` entries pass through as the NA marker. A series with fewer than two
    real values has nothing to normalize against and returns all NA, rather than
    reporting a lone arm as "1.0x (worst)", which would read as a comparison.
    """
    real = [v for v in values if v]
    if len(real) < 2:
        return [NA] * len(values)
    worst = max(real)
    return [(worst_label if v == worst else fmt.format(worst / v)) if v else NA
            for v in values]


def normalized(values, fmt="{:.2f}×"):
    """Render a timing series as multiples of its own first entry.

    Growth-shape view, for a single arm swept across N: it exhibits the exponent
    (a cubic arm rises ~8x per doubling, a quadratic one ~4x) without inheriting
    the host's thermal state. Use `normalized_worst` for comparing arms against
    each other; use this for watching one arm grow.
    """
    base = next((v for v in values if v), None)
    if not base:
        return [NA] * len(values)
    return [fmt.format(v / base) if v else NA for v in values]


FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")


def save_figure(fig, basename, formats=("png", "eps")):
    """Save one matplotlib figure in every format the document needs.

    PNG is what the Markdown embeds and what a reader sees in the repo. EPS is
    vector, and is what the LaTeX build will want when the dissertation is
    typeset -- generating it now costs nothing and means no figure has to be
    re-run later just to change container format.

    Two EPS caveats worth knowing, both avoidable at authoring time rather than
    export time: the format has no alpha channel, so any transparency is
    flattened, and it has no native support for embedded rasters at arbitrary
    DPI. Keep figures fully vector and opaque and the two outputs agree.

    Returns the paths written, so a caller can name them in its table note.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    written = []
    for ext in formats:
        path = os.path.join(FIGURE_DIR, f"{basename}.{ext}")
        # bbox_inches="tight" keeps the two formats framed identically; without
        # it EPS and PNG crop differently and the figures drift apart.
        fig.savefig(path, format=ext, bbox_inches="tight")
        written.append(path)
        print(f"  wrote {path}")
    return written


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:  # see write_markdown
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}")


def emit(basename, title, header, rows, note="", md_header=None, md_rows=None,
         seeds=None):
    """Write both .md and .csv for one table into reproduce/outputs/.

    `md_header` / `md_rows` let a timing table present a NORMALIZED view in the
    Markdown — the artifact that gets read and quoted into the chapters — while
    the CSV keeps the absolute wall-clock seconds for the record. Ratios survive
    a change of machine; seconds do not, and the chapters should therefore quote
    ratios. Omit both and the two artifacts are identical, which is right for
    every table whose cells are accuracies rather than durations.

    The `is None` tests are deliberate: an explicitly empty `md_rows=[]` means
    "the Markdown view has no rows," and truthiness would silently substitute the
    CSV rows instead — the exact class of quiet fallback this harness exists to
    prevent.
    """
    md = os.path.join(OUTPUT_DIR, f"{basename}.md")
    cv = os.path.join(OUTPUT_DIR, f"{basename}.csv")
    write_markdown(md, title,
                   header if md_header is None else md_header,
                   rows if md_rows is None else md_rows,
                   note, seeds=seeds)
    write_csv(cv, header, rows)


def optional_import(modpath, names):
    """Return the requested attributes, or None for each if the import fails.

    Lets a table skip a baseline (ANFIS, M5, ...) cleanly instead of crashing.
    """
    try:
        mod = __import__(modpath, fromlist=list(names))
        return [getattr(mod, n, None) for n in names]
    except Exception as exc:  # noqa: BLE001 - deliberately broad; report and skip
        print(f"  [skip] {modpath} unavailable ({exc.__class__.__name__}); "
              f"columns {names} -> {NA}")
        return [None] * len(names)
