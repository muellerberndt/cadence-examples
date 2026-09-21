"""The worm's brain: its connectome as a masked TemporalPatchNet, used strictly
through the library's documented calls.

The live state sits at the start of the stretch of experience the brain has
not yet committed. Every tick, `imagine` reads privately what the brain's
command neurons are doing now, from that state through the stretch. When an
outcome arrives (food, pain, a treat, a poke), `observe` learns the stretch
with targets saying what the command neurons should have been doing on the way
there, and carries the live state to the present. Quiet stretches are
committed with `advance`, which carries the state without learning.

Readouts are the command interneurons themselves: `forward` averages AVB and
PVC, `reverse` averages AVA, AVD and AVE. The body reads them directly."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cadence.experimental import PartitionedTemporalPatchNet

from .connectome import params, wiring


@dataclass
class Lesson:
    kind: str
    updated: bool
    reason: str
    delta: dict | None
    inputs: np.ndarray
    target: np.ndarray
    free: np.ndarray
    plus: np.ndarray | None
    minus: np.ndarray | None


class WormBrain:
    def __init__(self, seed: int = 0, lesion: tuple[str, ...] = (), frozen: bool = False,
                 p: dict | None = None) -> None:
        self.p = p or params()
        w = wiring(self.p, lesion)
        self.names, self.inputs, self.outputs = w.names, w.inputs, w.outputs
        self.H = len(w.names)
        index = {n: i for i, n in enumerate(self.names)}
        C = np.zeros_like(w.C)
        for k, readout in enumerate(self.outputs):
            cells = self.p["outputs"][readout]
            for c in cells:
                C[k, index[c]] = 1.0 / len(cells)
        self.net = PartitionedTemporalPatchNet(len(self.inputs), self.H, len(self.outputs),
                                               masks=w.masks, seed=seed)
        self.net.set_parameters({"A": w.A, "B": w.B, "C": C})
        self.frozen = frozen
        self.stretch: list[np.ndarray] = []
        self.path = None
        self.lessons = self.rejected = self.halvings = 0

    # ---- living ------------------------------------------------------------------
    def sense(self, u: np.ndarray) -> np.ndarray:
        """Add this tick's senses to the uncommitted stretch and read the command
        neurons privately (the live state is not moved)."""
        self.stretch.append(np.asarray(u, float))
        T = self.p["window"]
        if len(self.stretch) > 2 * T:
            self.net.advance(np.stack(self.stretch[:T])[None])
            self.stretch = self.stretch[T:]
        self.path = self.net.imagine(np.stack(self.stretch)[None])
        return self.path.output[0, -1]

    def activity(self) -> np.ndarray:
        return np.tanh(self.path.hidden[0, -1])

    def _target(self, kinds: tuple[str, ...]) -> np.ndarray:
        """Correct only the last `teach` ticks before the outcome: the command
        neuron that should have led there goes to one, its rival to zero.
        Earlier ticks get the brain's own free prediction, so they carry no
        correction at all."""
        y = self.path.output[0].copy()
        k = min(self.p["teach"], len(y))
        drive = {"food": "forward", "pain": "reverse"}
        for kind in kinds:
            want = self.outputs.index(drive[kind])
            y[-k:, want] = self.p["teach_level"]
            if len(kinds) == 1:
                y[-k:, 1 - want] = 0.0
        return y

    def learn(self, *kinds: str) -> Lesson:
        """The stretch led to these outcomes: learn what the command neurons
        should have been doing on the way, then carry the state to the present."""
        u = np.stack(self.stretch)[None]
        y = self._target(kinds)[None]
        return self._observe("+".join(kinds), u, y)

    def growth(self, A: np.ndarray | None = None) -> float:
        """Recurrent growth rate: (|A^64 x| / |x|)^(1/64) from a fixed start.
        Below one, activity dies out without input. Same function in web/brain.js."""
        A = self.net.parameters()["A"] if A is None else A
        x = np.linspace(1.0, 2.0, self.H)
        x /= np.linalg.norm(x)
        log = 0.0
        for _ in range(64):
            x = A @ x
            n = float(np.linalg.norm(x))
            if n == 0.0:
                return 0.0
            log += np.log(n)
            x /= n
        return float(np.exp(log / 64))

    def _stabilize(self, before: dict) -> int:
        """Homeostasis: if a lesson made the brain's recurrence grow past
        `stability`, keep only half the change, repeatedly, via set_parameters."""
        halvings = 0
        after = self.net.parameters()
        while self.growth(after["A"]) > self.p["stability"] and halvings < 12:
            after = {k: before[k] + 0.5 * (after[k] - before[k]) for k in after}
            halvings += 1
        if halvings:
            self.net.set_parameters(after)
        return halvings

    def _observe(self, kind: str, u: np.ndarray, y: np.ndarray) -> Lesson:
        before = self.net.parameters()
        r = self.net.observe(u, y, beta=self.p["beta"], rate=0.0 if self.frozen else self.p["rate"],
                             backtrack=not self.frozen)
        self.halvings = self._stabilize(before) if r.updated and not self.frozen else 0
        self.stretch = []
        if r.updated and not self.frozen:
            self.lessons += 1
        elif not r.updated:
            self.rejected += 1
        return Lesson(kind, bool(r.updated), r.reason, r.delta, u[0], y[0], r.free.hidden[0],
                      None if r.plus is None else r.plus.hidden[0],
                      None if r.minus is None else r.minus.hidden[0])

    # ---- before birth --------------------------------------------------------------
    def born(self) -> list[Lesson]:
        """Prenatal lessons give the reflex every worm is born with: when the
        nociceptors fire, reverse. Same rule, same brain, before the life."""
        T, out = self.p["window"], []
        k = self.inputs.index("pain")
        for _ in range(self.p["prenatal_lessons"]):
            self.net.reset()
            u = np.zeros((1, T, len(self.inputs)))
            u[0, :, k] = 1.0
            y = np.zeros((1, T, len(self.outputs)))
            y[0, 2:, self.outputs.index("reverse")] = self.p["teach_level"]
            before = self.net.parameters()
            r = self.net.observe(u, y, beta=self.p["beta"], rate=self.p["rate"], backtrack=True)
            if r.updated:
                self._stabilize(before)
            out.append(r.updated)
        self.net.reset()
        self.stretch = []
        return out

    def probe(self, sense: str) -> np.ndarray:
        """What one smell alone does to the command neurons now, from rest."""
        T = self.p["window"]
        u = np.zeros((1, T, len(self.inputs)))
        u[0, :, self.inputs.index(sense)] = 1.0
        return self.net.imagine(u, state=np.zeros((1, self.H))).output[0, -1]
