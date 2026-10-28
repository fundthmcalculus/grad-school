r"""A hand-built character dictionary, 98 symbols x 32 dimensions, every coordinate named: ``python -m flm.canonical_dictionary32 [out.npz]``.

Symbols are the data alphabet: 0 end of text, 1 unknown, 2 newline, 3 space, 4-97 the printable characters after space.  Weights in brackets; distances are Euclidean, so a heavier group moves characters further apart.
  0-4    kind, one flag each (1.0): special, whitespace, letter, digit, mark (punctuation or symbol)
  5      unknown rather than end of text (0.5), 6 newline rather than space (0.5)
  7      upper case (0.5): a and A differ only here
  8      vowel (0.5)
  9-13   letter identity, the alphabet position 0-25 as a 5-bit code (0.5), shared by the two cases
  14     digit value, one linear coordinate (0.25 per step): the distance between two digits is 0.25 x their difference
  15-21  mark group (0.5): sentence end . ! ? | pause , ; : | quote " ' ` | opening bracket ( [ { < | closing bracket ) ] } > | joiner - _ / \ | | arithmetic + = * % ^ ~ | (@ # $ & have no group flag)
  22-27  mark identity, the 6-bit index among the 33 marks (0.25): keeps every mark distinct
  28-31  alphabet band, one flag each (0.5): a-f, g-m, n-s, t-z, so neighbouring letters sit nearer each other
"""

import sys

import numpy as np

from flm.data import decode

W = dict(kind=1.0, flag=0.5, digit=0.25, mark_code=0.25, letter=0.5)
GROUPS = ".!?", ",;:", "\"'`", "([{<", ")]}>", "-_/" + chr(92) + "|", "+=*%^~"
NAMES = (
    ["special", "whitespace", "letter", "digit", "mark", "unknown", "newline", "upper", "vowel"]
    + [f"letter bit {b}" for b in range(5)]
    + ["digit value"]
    + ["sentence end", "pause", "quote", "open", "close", "joiner", "arithmetic"]
    + [f"mark bit {b}" for b in range(6)]
    + ["band a-f", "band g-m", "band n-s", "band t-z"]
)


def build(W=W):
    C = np.zeros((98, 32), dtype=np.float32)
    marks = [ch for ch in map(chr, range(33, 127)) if not ch.isalnum()]
    for i in range(98):
        if i < 2:
            C[i, 0] = W["kind"]
            C[i, 5] = W["flag"] * i
        elif i == 2:
            C[i, 1], C[i, 6] = W["kind"], W["flag"]
        else:
            ch = chr(i - 3 + 32)
            if ch == " ":
                C[i, 1] = W["kind"]
            elif ch.isalpha():
                k = ord(ch.lower()) - 97
                C[i, 2] = W["kind"]
                C[i, 7] = W["flag"] * ch.isupper()
                C[i, 8] = W["flag"] * (ch.lower() in "aeiou")
                for b in range(5):
                    C[i, 9 + b] = W["letter"] * ((k >> b) & 1)
                C[i, 28 + sum(k >= s for s in (6, 13, 19))] = W["flag"]
            elif ch.isdigit():
                C[i, 3] = W["kind"]
                C[i, 14] = W["digit"] * int(ch)
            else:
                C[i, 4] = W["kind"]
                for g, s in enumerate(GROUPS):
                    if ch in s:
                        C[i, 15 + g] = W["flag"]
                k = marks.index(ch)
                for b in range(6):
                    C[i, 22 + b] = W["mark_code"] * ((k >> b) & 1)
    return C


def main():
    C = build()
    assert len({tuple(r) for r in C}) == 98, "two symbols share a vector"
    assert len(NAMES) == 32
    if len(sys.argv) > 1:
        np.savez(sys.argv[1], C=C)
    d = np.linalg.norm(C[:, None] - C[None], axis=-1)
    for ch in ["a", "e", "t", "A", "3", ".", ",", "(", "+"]:
        i = ord(ch) - 29
        nn = np.argsort(d[i])[1:6]
        print(repr(ch), "->", [decode([j]) for j in nn], np.round(d[i][nn], 2))
    print("min distance between two symbols", round(float(d[np.triu_indices(98, 1)].min()), 2))


if __name__ == "__main__":
    main()
