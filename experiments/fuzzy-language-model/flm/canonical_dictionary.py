r"""A canonical character dictionary built by construction, 98 symbols x 64 dimensions, no training: ``python -m flm.canonical_dictionary [out.npz]``.

Every coordinate means something, so the geometry can be read and any model that freezes it shares the same space.  Symbols are the data alphabet: 0 end of text, 1 unknown, 2 newline, 3 space, 4-97 the printable characters after space.

Layout (weights in brackets; distances are Euclidean, so a heavier group moves characters further apart):
  0-7    class, one flag each (0.5): special, newline, space, lower-case letter, upper-case letter, digit, punctuation, symbol
  8-33   letter identity, one flag per letter a-z, shared by the two cases (1.0): a and A differ only in class, so they are nearer to each other than to any other letter
  34-42  digit value as a thermometer (0.5): digit v switches on the first v flags, so digits are ordered and near digits are near
  43     vowel (0.5)
  44-51  punctuation and symbol groups (0.5): sentence end . ! ? | pause , ; : | quote " ' ` | opening bracket ( [ { < | closing bracket ) ] } > | joiner - _ / \ | | arithmetic + = * % ^ ~ | other symbol @ # $ &
  52-57  symbol identity, the 6-bit index among the non-alphanumeric printable characters (0.5): keeps every symbol distinct
  58-59  end of text / unknown (1.0)
  60     whitespace (0.5): space and newline
  61-63  kind, one flag each (1.0): letter, digit, punctuation or symbol: puts every letter nearer to every other letter than to any digit or mark
"""

import sys

import numpy as np

from flm.data import decode

W = dict(cls=0.5, letter=1.0, digit=0.5, flag=0.5, code=0.5, special=1.0)
GROUPS = {
    44: ".!?",
    45: ",;:",
    46: "\"'`",
    47: "([{<",
    48: ")]}>",
    49: "-_/" + chr(92) + "|",
    50: "+=*%^~",
    51: "@#$&",
}


def build(W=W, digit_step=None, letter_code=False):
    """digit_step None: thermometer code (distance grows as sqrt of the gap); a number: one linear coordinate, distance = step x gap."""
    C = np.zeros((98, 64), dtype=np.float32)
    symbols = [ch for ch in map(chr, range(33, 127)) if not ch.isalnum()]
    for i in range(98):
        if i == 0:
            C[i, 0], C[i, 58] = W["cls"], W["special"]
        elif i == 1:
            C[i, 0], C[i, 59] = W["cls"], W["special"]
        elif i == 2:
            C[i, 1], C[i, 60] = W["cls"], W["flag"]
        else:
            ch = chr(i - 3 + 32)
            if ch == " ":
                C[i, 2], C[i, 60] = W["cls"], W["flag"]
            elif ch.isalpha():
                C[i, 4 if ch.isupper() else 3] = W["cls"]
                if letter_code:  # 5-bit alphabet position in dims 8-12 instead of one flag per letter
                    k = ord(ch.lower()) - 97
                    for b in range(5):
                        C[i, 8 + b] = W["letter"] * ((k >> b) & 1)
                else:
                    C[i, 8 + ord(ch.lower()) - 97] = W["letter"]
                C[i, 61] = W["letter"]
                if ch.lower() in "aeiou":
                    C[i, 43] = W["flag"]
            elif ch.isdigit():
                C[i, 5], C[i, 62] = W["cls"], W["letter"]
                if digit_step is None:
                    C[i, 34 : 34 + int(ch)] = W["digit"]
                else:
                    C[i, 34] = digit_step * int(ch)
            else:
                grp = [g for g, s in GROUPS.items() if ch in s]
                C[i, 6 if grp and grp[0] <= 48 and grp[0] != 49 else 7] = W["cls"]
                C[i, 63] = W["letter"]
                for g in grp:
                    C[i, g] = W["flag"]
                k = symbols.index(ch)
                for b in range(6):
                    C[i, 52 + b] = W["code"] * ((k >> b) & 1)
    return C


def project(C, k, seed=0):
    """k coordinates: the top principal components of C (scaled by the singular values) for the first 3k/4, then a fixed random projection of the residual for the rest, which keeps near-identical symbols apart."""
    Z = C - C.mean(0)
    u, sv, vt = np.linalg.svd(Z, full_matrices=False)
    kp = k - k // 4
    res = Z - (u[:, :kp] * sv[:kp]) @ vt[:kp]
    R = np.random.default_rng(seed).normal(size=(C.shape[1], k - kp)) / np.sqrt(k - kp)
    return np.concatenate([u[:, :kp] * sv[:kp], res @ R], 1).astype(np.float32)


def distortion(C, P):
    """Correlation of the pairwise distances before and after projection."""
    iu = np.triu_indices(len(C), 1)
    d = lambda M: np.linalg.norm(M[:, None] - M[None], axis=-1)[iu]
    return float(np.corrcoef(d(C), d(P))[0, 1])


def main():
    C = build()
    d = np.linalg.norm(C[:, None] - C[None], axis=-1)
    assert len({tuple(r) for r in C}) == 98, "two symbols share a vector"
    if len(sys.argv) > 1:
        np.savez(sys.argv[1], C=C)
    for ch in ["a", "e", "t", "A", "3", ".", ",", "(", "+"]:
        i = ord(ch) - 32 + 3
        nn = np.argsort(d[i])[1:6]
        print(
            repr(ch),
            "->",
            [decode([j]).replace("\n", "\n") for j in nn],
            np.round(d[i][nn], 2),
        )
    print(
        "a-A",
        round(d[ord("a") - 29, ord("A") - 29], 2),
        "a-b",
        round(d[ord("a") - 29, ord("b") - 29], 2),
        "a-e",
        round(d[ord("a") - 29, ord("e") - 29], 2),
        "3-4",
        round(d[ord("3") - 29, ord("4") - 29], 2),
        "3-9",
        round(d[ord("3") - 29, ord("9") - 29], 2),
    )


if __name__ == "__main__":
    main()
