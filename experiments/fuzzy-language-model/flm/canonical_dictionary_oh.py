r"""Named dictionaries whose letters are one flag each, so all 26 letters sit equally far apart: ``python -m flm.canonical_dictionary_oh``  (writes outputs/embspace/oh48.npz and squeeze32.npz).

A frozen dictionary cannot undo a false similarity between letters (a 5-bit letter code cost +0.26 to +0.5 bits frozen), so letters get one flag each.  Symbols are the data alphabet: 0 end of text, 1 unknown, 2 newline, 3 space, 4-97 the printable characters after space.

oh48 (48 dimensions, every coordinate its own meaning):
  0-25   letter flag, one per letter (1.0), shared by the two cases
  26-29  kind (1.0): digit, mark (punctuation or symbol), whitespace, special
  30     unknown rather than end of text (0.5), 31 newline rather than space (0.5)
  32     vowel (0.5)
  33     digit value, one linear coordinate (0.25 per step)
  34-40  mark group (0.5): sentence end . ! ? | pause , ; : | quote " ' ` | opening bracket ( [ { < | closing bracket ) ] } > | joiner - _ / \ | | arithmetic + = * % ^ ~ | (@ # $ & have no group flag)
  41-46  mark identity, the 6-bit index among the 33 marks (0.25)
  47     upper case (0.5)

oh40 (40 dimensions): 0-25 letter flag, 26 space, 27 newline, 28-30 kind (digit, mark, special), 31 unknown, 32 upper case, 33 digit value (linear), 34-39 mark identity (6-bit code at 0.5).  Only the marks (4% of text) rely on a binary code.

squeeze32 (32 dimensions, the floor for one flag per letter: 26 letters plus a 6-bit code):
  0-25   letter flag (1.0 for lower case, 0.6 for upper case: a and A are 0.4 apart, any two letters 1.41)
  26-31  non-letter code (0.5): the 6-bit number 1-47 of the symbol among digits (1-10), space (11), newline (12), end of text (13), unknown (14), then the 33 marks grouped by punctuation group (15-47); zero for letters
"""

import sys

import numpy as np

from flm.data import decode

GROUPS = ".!?", ",;:", "\"'`", "([{<", ")]}>", "-_/" + chr(92) + "|", "+=*%^~"
MARKS = [ch for ch in map(chr, range(33, 127)) if not ch.isalnum()]


def oh48():
    C = np.zeros((98, 48), dtype=np.float32)
    for i in range(98):
        if i < 2:
            C[i, 29] = 1.0
            C[i, 30] = 0.5 * i
            continue
        if i == 2:
            C[i, 28], C[i, 31] = 1.0, 0.5
            continue
        ch = chr(i - 3 + 32)
        if ch == " ":
            C[i, 28] = 1.0
        elif ch.isalpha():
            C[i, ord(ch.lower()) - 97] = 1.0
            C[i, 32] = 0.5 * (ch.lower() in "aeiou")
            C[i, 47] = 0.5 * ch.isupper()
        elif ch.isdigit():
            C[i, 26] = 1.0
            C[i, 33] = 0.25 * int(ch)
        else:
            C[i, 27] = 1.0
            for g, s in enumerate(GROUPS):
                if ch in s:
                    C[i, 34 + g] = 0.5
            k = MARKS.index(ch)
            for b in range(6):
                C[i, 41 + b] = 0.25 * ((k >> b) & 1)
    return C


def squeeze32():
    order = [ch for g in GROUPS for ch in g] + [ch for ch in MARKS if not any(ch in g for g in GROUPS)]
    code = {}
    for d in range(10):
        code[str(d)] = 1 + d
    code.update({" ": 11, "\n": 12, "EOT": 13, "UNK": 14})
    for j, ch in enumerate(order):
        code[ch] = 15 + j
    C = np.zeros((98, 32), dtype=np.float32)
    for i in range(98):
        ch = "EOT" if i == 0 else "UNK" if i == 1 else "\n" if i == 2 else chr(i - 3 + 32)
        if len(ch) == 1 and ch.isalpha():
            C[i, ord(ch.lower()) - 97] = 0.6 if ch.isupper() else 1.0
        else:
            for b in range(6):
                C[i, 26 + b] = 0.5 * ((code[ch] >> b) & 1)
    return C


def oh40():
    """40 dimensions: a flag per letter, equal-distance flags for space and newline (23% of text), a 6-bit code only for the marks."""
    C = np.zeros((98, 40), dtype=np.float32)
    for i in range(98):
        if i < 2:
            C[i, 30] = 1.0
            C[i, 31] = 0.5 * i
            continue
        if i == 2:
            C[i, 27] = 1.0
            continue
        ch = chr(i - 3 + 32)
        if ch == " ":
            C[i, 26] = 1.0
        elif ch.isalpha():
            C[i, ord(ch.lower()) - 97] = 1.0
            C[i, 32] = 0.5 * ch.isupper()
        elif ch.isdigit():
            C[i, 28] = 1.0
            C[i, 33] = 0.25 * int(ch)
        else:
            C[i, 29] = 1.0
            k = MARKS.index(ch)
            for b in range(6):
                C[i, 34 + b] = 0.5 * ((k >> b) & 1)
    return C


def show(C, name):
    d = np.linalg.norm(C[:, None] - C[None], axis=-1)
    assert len({tuple(r) for r in C}) == 98, name
    print(name, C.shape, "min distance", round(float(d[np.triu_indices(98, 1)].min()), 2))
    for ch in ["a", "e", "3", ".", "(", " "]:
        i = ord(ch) - 29
        nn = np.argsort(d[i])[1:5]
        print("  ", repr(ch), "->", [decode([j]) for j in nn], np.round(d[i][nn], 2))


def main():
    for name, f in (("oh48", oh48), ("squeeze32", squeeze32), ("oh40", oh40)):
        C = f()
        np.savez(f"outputs/embspace/{name}.npz", C=C)
        show(C, name)


if __name__ == "__main__":
    main()
