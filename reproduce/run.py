#!/usr/bin/env python3
"""Orchestrator: run the registered experiments and report what happened.

Walks `manifest.py`, runs each experiment in its own submodule environment, and
prints a summary. Nothing is skipped silently -- an experiment that cannot run
says why, so the difference between "failed" and "not applicable on this
machine" is always visible.

Usage (from repo root):
    python reproduce/run.py --list                 # show the registry
    python reproduce/run.py --tables               # just the proposal tables
    python reproduce/run.py --chapter Ch3          # everything feeding Ch3
    python reproduce/run.py --id table-3-1-reorder # one experiment
    python reproduce/run.py --all                  # everything (long)
    python reproduce/run.py --all --dry-run        # print commands only

Hardware gating: experiments tagged `gpu` or `big-mem` are skipped unless
--include-hardware is passed, since most machines cannot run them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from manifest import EXPERIMENTS, by_id  # noqa: E402

GATED = {"gpu", "big-mem"}


def fmt_row(e, status="", secs=None):
    t = f"{secs:6.1f}s" if secs is not None else ""
    return f"  {e.chapter:<5} {e.id:<34} {e.hardware:<12} {status:<12} {t}"


def cmd_list():
    print(f"\n{len(EXPERIMENTS)} registered experiments\n")
    print(f"  {'CH':<5} {'ID':<34} {'HARDWARE':<12} PRODUCES")
    print("  " + "-" * 86)
    for e in EXPERIMENTS:
        print(f"  {e.chapter:<5} {e.id:<34} {e.hardware:<12} {e.produces}")
    print()


def select(args):
    picked = EXPERIMENTS
    if args.id:
        e = by_id(args.id)
        if e is None:
            sys.exit(f"unknown experiment id: {args.id}\n"
                     f"try: python reproduce/run.py --list")
        picked = [e]
    elif args.chapter:
        want = args.chapter.lower()
        picked = [e for e in EXPERIMENTS if e.chapter.lower() == want]
    elif args.tables:
        picked = [e for e in EXPERIMENTS if e.produces.lower().startswith("table")]
    if not args.include_hardware:
        picked = [e for e in picked if e.hardware not in GATED]
    return picked


def run_one(e, dry_run=False):
    """Run one experiment in its own environment. Returns (status, seconds)."""
    cwd = os.path.join(REPO, e.repo) if e.repo else REPO
    if not os.path.isdir(cwd):
        return "no-repo", None
    if dry_run:
        print(f"      $ (cd {e.repo}) {' '.join(e.command)}")
        return "dry-run", None

    t0 = time.perf_counter()
    try:
        p = subprocess.run(e.command, cwd=cwd, capture_output=True,
                           text=True, timeout=3600)
    except FileNotFoundError:
        return "no-runner", None
    except subprocess.TimeoutExpired:
        return "timeout", time.perf_counter() - t0
    secs = time.perf_counter() - t0

    log_dir = os.path.join(HERE, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{e.id}.log"), "w") as f:
        f.write(f"$ (cd {e.repo}) {' '.join(e.command)}\n\n")
        f.write("---- stdout ----\n" + (p.stdout or ""))
        f.write("\n---- stderr ----\n" + (p.stderr or ""))
    return ("ok" if p.returncode == 0 else f"exit {p.returncode}"), secs


def main():
    ap = argparse.ArgumentParser(description="Run registered reproduction experiments.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="show the registry and exit")
    g.add_argument("--all", action="store_true", help="run everything")
    g.add_argument("--tables", action="store_true", help="run the proposal-table generators")
    g.add_argument("--chapter", metavar="CH", help="run everything feeding a chapter (Ch3, Ch5, ...)")
    g.add_argument("--id", metavar="ID", help="run one experiment by id")
    ap.add_argument("--include-hardware", action="store_true",
                    help="also run gpu / big-mem experiments (skipped by default)")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running")
    args = ap.parse_args()

    if args.list or not (args.all or args.tables or args.chapter or args.id):
        cmd_list()
        return

    picked = select(args)
    if not picked:
        print("nothing selected")
        return

    print(f"\nrunning {len(picked)} experiment(s)"
          f"{' [dry run]' if args.dry_run else ''}\n")
    print(f"  {'CH':<5} {'ID':<34} {'HARDWARE':<12} {'STATUS':<12} TIME")
    print("  " + "-" * 80)

    results = []
    for e in picked:
        status, secs = run_one(e, args.dry_run)
        print(fmt_row(e, status, secs))
        results.append((e, status))

    ok = [r for r in results if r[1] in ("ok", "dry-run")]
    bad = [r for r in results if r[1] not in ("ok", "dry-run")]
    print("\n  " + "-" * 80)
    print(f"  {len(ok)}/{len(results)} succeeded")
    if bad:
        print("\n  did not complete:")
        for e, status in bad:
            print(f"    {e.id:<34} {status}")
        print(f"\n  logs: reproduce/outputs/logs/<id>.log")
    skipped = [e for e in EXPERIMENTS if e.hardware in GATED] if not args.include_hardware else []
    if skipped:
        print(f"\n  {len(skipped)} hardware-gated experiment(s) skipped "
              f"(--include-hardware to attempt): "
              f"{', '.join(e.id for e in skipped)}")
    print()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
