#!/bin/bash
cd "$(dirname "$0")/.."
for a in "ts 1e9" "cosmo 30e6" "cosmo 300e6" "cosmo 1e9"; do python -m flm.run_big2m $a; done
