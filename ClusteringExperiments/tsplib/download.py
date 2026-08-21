#!/usr/bin/env python3
"""Fetch the symmetric TSPLIB instances used by the clustering/TSP experiments.

The ``.tsp`` files themselves are no longer tracked (they are standard,
publicly published TSPLIB95 data and add ~9 MB to the repo). This script
restores them from a mirror into this directory, exactly the set the
experiments expect.

Usage, from the repo root or anywhere::

    python ClusteringExperiments/tsplib/download.py            # fetch missing
    python ClusteringExperiments/tsplib/download.py --force    # re-fetch all

The instance list is ``instances.txt`` (one name per line), which stays tracked
so the required set is pinned. Known-optimum tour lengths live in ``solutions``.

Source: https://github.com/mastqe/tsplib — a byte-for-byte mirror of the
Heidelberg TSPLIB95 symmetric-TSP files
(http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/). The Heidelberg host is
HTTP-only and frequently unreachable, so the mirror is the default; pass
``--source heidelberg`` to pull the gzip'd originals instead.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

SOURCES = {
    # name -> (url_template, is_gzip)
    "mirror": (
        "https://raw.githubusercontent.com/mastqe/tsplib/master/{name}.tsp",
        False,
    ),
    "heidelberg": (
        "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{name}.tsp.gz",
        True,
    ),
}


def instance_names() -> list[str]:
    manifest = HERE / "instances.txt"
    if not manifest.exists():
        sys.exit(f"missing {manifest}; cannot know which instances to fetch")
    return [
        line.strip()
        for line in manifest.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def fetch_one(name: str, url_tmpl: str, is_gzip: bool, timeout: int) -> bytes:
    url = url_tmpl.format(name=name)
    with urllib.request.urlopen(
        url, timeout=timeout
    ) as resp:  # noqa: S310 (trusted host)
        raw = resp.read()
    return gzip.decompress(raw) if is_gzip else raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force", action="store_true", help="re-fetch instances already present"
    )
    ap.add_argument("--source", choices=list(SOURCES), default="mirror")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    url_tmpl, is_gzip = SOURCES[args.source]
    names = instance_names()
    fetched = skipped = failed = 0

    for name in names:
        dest = HERE / f"{name}.tsp"
        if dest.exists() and not args.force:
            skipped += 1
            continue
        try:
            data = fetch_one(name, url_tmpl, is_gzip, args.timeout)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAIL {name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        dest.write_bytes(data)
        fetched += 1
        print(f"  ok   {name} ({len(data)} bytes)")

    print(
        f"\n{fetched} fetched, {skipped} already present, {failed} failed "
        f"(of {len(names)} instances) from '{args.source}'."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
