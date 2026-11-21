"""Tests of the mixed-corpus builder (`flm/prep_mix.py`) on synthetic 98-symbol streams: ``pytest experiments/fuzzy-language-model/test_mix.py``"""

import numpy as np

from flm.prep_mix import EOT, docs_of, interleave


def stream(lengths, fill):
    parts = [
        np.concatenate([np.full(n, fill, np.uint8), [EOT]]).astype(np.uint8)
        for n in lengths
    ]
    return np.concatenate(parts)


def test_docs_of_cuts_at_document_boundaries():
    s = stream([5, 7, 3], 9)
    d = docs_of(s, len(s) - 2)  # the last document is cut: dropped
    assert [len(x) for x in d] == [6, 8]
    assert all(x[-1] == EOT for x in d)


def test_interleave_is_balanced_by_characters():
    a = docs_of(stream([10] * 40, 5), 10**6)
    b = docs_of(stream([30] * 14, 6), 10**6)  # longer documents
    m = interleave(a, b)
    assert len(m) == sum(map(len, a)) + sum(map(len, b))
    assert (
        abs(int((m == 5).sum()) - int((m == 6).sum())) <= 30
    )  # within one long document
    # after every whole document the two sources are within one (long) document of each other
    diff, worst = 0, 0
    for d in docs_of(m, len(m)):
        diff += len(d) if d[0] == 5 else -len(d)
        worst = max(worst, abs(diff))
    assert worst <= 31


def test_interleave_is_deterministic_and_keeps_order():
    a = docs_of(stream([4, 5, 6], 5), 10**6)
    b = docs_of(stream([3, 3, 3], 6), 10**6)
    assert np.array_equal(interleave(a, b), interleave(a, b))
    m = interleave(a, b)
    assert [len(x) for x in docs_of(m, len(m)) if x[0] == 5] == [5, 6, 7]
