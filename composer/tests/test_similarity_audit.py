"""The copying screen distinguishes exact rhythmic matches from interval matches."""

from tools.similarity_audit import windows


def test_transposition_and_rhythm_sensitivity():
    original = [(i * 2, p) for i, p in enumerate([60, 62, 65, 64, 67, 69, 66, 72])]
    transposed = [(t, p + 5) for t, p in original]
    stretched = [(t * 2, p) for t, p in original]
    assert list(windows(original, 8)) == list(windows(transposed, 8))
    assert list(windows(original, 8)) != list(windows(stretched, 8))
    assert list(windows(original, 8, False)) == list(windows(stretched, 8, False))
    assert not list(windows(original, 16))
