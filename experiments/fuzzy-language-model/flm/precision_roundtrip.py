"""Does the invertible stack stay invertible in low precision? ``BIJ=ci FUZZ_FFN_R=156 FUZZ_FFN_R0=109 python -m flm.precision_roundtrip CKPT.pt``

Fraction of the 98 codewords recovered by the nearest-codeword decode after F then G then G^-1 then F^-1 (for ``c``: F then F^-1), with the layers cast to float64, float32, bfloat16 and float16 (the whole round trip in that arithmetic), and the largest absolute error in the codebook.
"""

import sys

import torch

from flm.explain_bijective import BIJ, build


@torch.no_grad()
def main():
    m = build(sys.argv[1], "cuda" if torch.cuda.is_available() else "cpu")
    c0 = m.codebook.detach().clone()
    for name, dt in (
        ("float64", torch.float64),
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
    ):
        mm = build(sys.argv[1], c0.device).to(dt)
        c = mm.codebook.detach().to(dt)
        z = mm.F(c)[0]
        if BIJ == "ci":
            z = mm.G.inverse(mm.G(z))
        back = mm.F.inverse(z)
        rec = (
            (
                torch.cdist(back.float(), c.float()).argmin(-1)
                == torch.arange(c.shape[0], device=c.device)
            )
            .float()
            .mean()
            .item()
        )
        print(
            f"{name}: recovered {rec:.4f}, max |F^-1(F(c)) - c| = {(back - c).abs().max().item():.3g}"
        )


if __name__ == "__main__":
    main()
