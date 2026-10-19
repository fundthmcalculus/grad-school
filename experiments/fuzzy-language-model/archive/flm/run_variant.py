"""``python -m flm.run_variant <train args>``: flm.train with the RFLM variants of flm.rflm_variants registered."""

import sys

import flm.rflm_variants  # noqa: F401
from flm.train import main

if __name__ == "__main__":
    main(sys.argv[1:])
