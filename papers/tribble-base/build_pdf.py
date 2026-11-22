#!/usr/bin/env python3
"""Wrapper: render papers/tribble-base/paper.md via the shared converter.

The shared converter lives in papers/tribble-pdf/build_pdf.py; this script is a
thin delegate so the tribble-base paper can build its PDF with a one-liner.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
CONVERTER = os.path.join(REPO_ROOT, "papers", "tribble-pdf", "build_pdf.py")
PAPER = os.path.join(HERE, "paper.md")


def main():
    cmd = [sys.executable, CONVERTER, "--paper", PAPER, *sys.argv[1:]]
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
