"""The worm's brain: the connectome compiled into one Cadence 0.50 brain, used
strictly through the library's documented calls.

Every neuron is one processing patch, and its declared inputs are exactly its
synapses: nothing else can grow. The neurons are declared in order of synaptic
distance from the sensory cells, so every synapse running down that order is a
live state connection inside the tick's one joint settlement, while every
synapse running back up it, and each neuron's own persistence term, arrives
through the previous tick's settled states. Signals flow from the senses
toward the commands within a tick; feedback takes a tick; and a lesson's
correction reaches back through the live chain into the weights of every
synapse on it. Two readout patches, `forward drive` and `reverse drive`, read
the live states of the command interneurons in the same settlement, and the
body reads them through the brain's two declared outputs.

Every tick, `sense` feeds the previous settled states back in with the current
senses and commits one `step`. When an outcome arrives (food, pain, a treat, a
poke), `observe_batch` learns the ticks that led there under constructed
command targets (`source="estimate"`): the whole proposal must qualify, the
parameter anchor bounds how far one lesson moves the weights, and a refused
solve changes nothing. The live state is preserved; life continues."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cadence import Cortex

from .connectome import params, wiring


@dataclass
class Lesson:
    kind: str
    updated: bool
    reason: str
    rows: int
    event_id: int | None
    energy: float | None
    stationarity: float | None
    sweeps: int | None
    # applied is the retained parameter change of an accepted lesson, per edge and per patch.
    applied: dict | None = None


class WormBrain:
    def __init__(self, seed: int = 0, lesion: tuple[str, ...] = (), frozen: bool = False,
                 p: dict | None = None) -> None:
        self.p = p or params()
        w = wiring(self.p, lesion)
        self.names, self.senses, self.readouts = w.names, w.senses, w.readouts
        self.order, self.pos = w.order, w.pos
        self.H = len(w.names)
        layout = Cortex(seed=seed, tolerance=self.p["tolerance"])
        prev = [layout.input(f"~{n}", shape=(1,)) for n in w.names]
        sensors = {s: layout.input(s, shape=(1,)) for s in self.senses}
        index = {n: i for i, n in enumerate(w.names)}
        cells: dict[int, object] = {}
        for i in w.order:
            live = set(w.state_partners[i])
            sources = tuple(sensors[s] for s in w.cell_senses[i])
            sources += tuple(cells[j] if j in live else prev[j]
                             for j in sorted(live | set(w.prev_partners[i])))
            cells[i] = layout.column(w.names[i], patches=1, inputs=sources)
        for readout in self.readouts:
            group = layout.column(f"{readout} drive", patches=1,
                                  inputs=tuple(cells[index[c]] for c in self.p["outputs"][readout]))
            layout.output(readout, shape=(1,), reads=group)
        self.brain = layout.build()
        self.frozen = frozen
        self.stretch: list[dict] = []
        self.readout = np.zeros(len(self.readouts))
        self.lessons = self.rejected = self.refusals = 0

    # ---- living ------------------------------------------------------------------
    def _inputs(self, u, state=None) -> dict:
        state = self.brain.state if state is None else state
        inputs = {f"~{n}": (state[self.pos[i]],) for i, n in enumerate(self.names)}
        inputs.update({s: (float(u[k]),) for k, s in enumerate(self.senses)})
        return inputs

    def sense(self, u) -> np.ndarray:
        """Feed the previous settled states back in with this tick's senses,
        commit one step, and read the command drives. Each committed tick joins
        the stretch a lesson can reach, together with what the drives freely
        did. A refused step keeps the previous state and readout; nothing
        partial is retained."""
        inputs = self._inputs(u)
        r = self.brain.step(inputs)
        if r["accepted"]:
            self.stretch.append((inputs, r["outputs"]))
            if len(self.stretch) > self.p["window"]:
                self.stretch.pop(0)
            self.readout = np.array([r["outputs"][k][0] for k in self.readouts])
        else:
            self.refusals += 1
        return self.readout

    def activity(self) -> np.ndarray:
        """Each neuron's settled state, in connectome order."""
        state = self.brain.state
        return np.array([state[self.pos[i]] for i in range(self.H)])

    def _targets(self, kinds: tuple[str, ...], free: dict) -> dict:
        """Correct only the last `teach` ticks before the outcome: the command
        drive that should have led there is asked to reach `teach_level`, its
        rival to rest. Earlier ticks keep the brain's own free prediction as
        their target, so they carry no correction at all — and the drives'
        biases cannot absorb the outcome, because the same batch restates what
        the drives freely did before it."""
        drive = {"food": "forward", "pain": "reverse"}
        targets = {k: (free[k][0],) for k in self.readouts}
        for kind in kinds:
            targets[drive[kind]] = (self.p["teach_level"],)
        if len(kinds) == 1:
            (want,) = kinds
            rival = "reverse" if drive[want] == "forward" else "forward"
            targets[rival] = (0.0,)
        return targets

    def learn(self, *kinds: str) -> Lesson:
        """The recent ticks led to these outcomes: learn them as one batch of
        constructed-target experiences, preserving the live state."""
        k = min(self.p["teach"], len(self.stretch))
        rows = [(inputs, {r: (free[r][0],) for r in self.readouts})
                for inputs, free in self.stretch[:len(self.stretch) - k]]
        rows += [(inputs, self._targets(kinds, free))
                 for inputs, free in self.stretch[len(self.stretch) - k:]]
        return self._admit("+".join(kinds), rows)

    def _admit(self, kind: str, rows: list) -> Lesson:
        """One outcome, one proposal: the whole batch qualifies or nothing is
        kept. The stretch is spent either way; an admitted lesson's applied
        change is the retained parameter movement."""
        self.stretch = []
        if self.frozen or not rows:
            self.rejected += 1
            return Lesson(kind, False, "frozen" if self.frozen else "no_experience",
                          len(rows), None, None, None, None)
        before_w, before_b = self.brain.weights, self.brain.biases
        r = self.brain.observe_batch(rows, source="estimate")
        if r["accepted"]:
            self.lessons += 1
            applied = {
                "weights": tuple(a - b for a, b in zip(self.brain.weights, before_w)),
                "biases": tuple(a - b for a, b in zip(self.brain.biases, before_b)),
            }
            return Lesson(kind, True, r["reason"], len(rows), r["event_id"],
                          r["energy"], r["stationarity"], r["sweeps"], applied)
        self.rejected += 1
        return Lesson(kind, False, r["reason"], len(rows), r["event_id"],
                      r.get("energy"), r.get("stationarity"), r.get("sweeps"))

    # ---- before birth --------------------------------------------------------------
    def born(self) -> list[Lesson]:
        """Prenatal lessons give the reflex every worm is born with: when the
        nociceptors fire, reverse; when nothing happens, rest. Each episode is
        lived first (quiet, then a sting long enough to spread through the
        wiring), then taught as one batch that must fit both of its contexts:
        the quiet ticks are asked for rest, the last ticks of the sting for the
        withdrawal. A drive's bias can satisfy neither alone, so the sting has
        to be read out of the wiring. Frozen worms are taught too; their freeze
        is on the life, not on what they hatch with."""
        frozen, self.frozen = self.frozen, False
        quiet = np.zeros(len(self.senses))
        sting = np.zeros(len(self.senses))
        sting[self.senses.index("pain")] = 1.0
        stung = self.p["teach"] + 2
        rest = {r: (0.0,) for r in self.readouts}
        out = []
        for _ in range(self.p["prenatal_lessons"]):
            for _ in range(self.p["window"]):
                self.sense(quiet)
            for _ in range(stung):
                self.sense(sting)
            rows = [(inputs, rest) for inputs, _ in self.stretch[:-stung]]
            rows += [(inputs, {r: (free[r][0],) for r in self.readouts})
                     for inputs, free in self.stretch[-stung:-self.p["teach"]]]
            rows += [(inputs, self._targets(("pain",), free))
                     for inputs, free in self.stretch[-self.p["teach"]:]]
            out.append(self._admit("pain", rows))
        for _ in range(self.p["window"]):
            self.sense(quiet)
        self.frozen = frozen
        self.stretch = []
        self.lessons = self.rejected = self.refusals = 0
        return out

    def probe(self, sense: str) -> np.ndarray:
        """What one sense alone does to the command drives now, from rest: a
        pure rollout of settles that never touches the live state."""
        u = np.zeros(len(self.senses))
        u[self.senses.index(sense)] = 1.0
        state = (0.0,) * self.H
        outputs = {k: (0.0,) for k in self.readouts}
        for _ in range(self.p["probe_ticks"]):
            r = self.brain.settle(self._inputs(u, state))
            if not r["qualified"]:
                break
            state, outputs = r["state"][:self.H], r["outputs"]
        return np.array([outputs[k][0] for k in self.readouts])
