"""mulberry32, bit-identical to the browser's implementation."""
M = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & M


class Rng:
    def __init__(self, seed: int) -> None:
        self.a = seed & M

    def random(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & M
        a = self.a
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & M) ^ t
        return ((t ^ (t >> 14)) & M) / 4294967296.0

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()
