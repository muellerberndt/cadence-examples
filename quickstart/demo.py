"""The three quickstart brains behind a local page: ``python -m quickstart.demo stream``, ``decide``, ``body`` (from the examples root).

Each demo runs the quickstart's own computation, with the same seeds, and streams what the
page draws: every neuron and synapse of the brain in the shipped viewer, its activity moment
by moment, how far each neuron still moves as the brain settles and how far the teaching pull
displaces it from the free equilibrium, the synapses' last change from the detuning contrast,
the numbers that define the equilibrium, and the learning curves as they are measured. The
page runs on your machine; nothing is hosted and nothing is trained ahead of time. A run takes
seconds to a minute.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlparse

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadence.brain import Brain  # noqa: E402
from cadence.connectome import Connectome  # noqa: E402
from cadence.genome import Genome, Projection, develop  # noqa: E402
from cadence.learning import Learner, LearnerConfig, learning_neuron_model  # noqa: E402
from cadence.record_patch import RecordPatchNet  # noqa: E402
from cadence.recording import SettlementRecord, record_settlements  # noqa: E402
from cadence.regions import Region, cortex, motor_cortex  # noqa: E402
from cadence.temporal import TemporalPatchNet, contrast_asymmetry  # noqa: E402
from viewer.atlas import atlas_of, brain_scan_script, build_atlas  # noqa: E402

__all__ = ["BodyDemo", "DecideDemo", "Demo", "StreamDemo", "demos", "main", "serve"]


class Demo:
    """A run that reports: events, frames of brain activity and heat, stats and curves."""

    name = "demo"
    title = "A Cadence brain"
    task = ""
    note = ""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: dict[str, Any] = {
            "name": self.name,
            "title": self.title,
            "task": self.task,
            "note": self.note,
            "phase": "start",
            "progress": 0.0,
            "status": "",
            "curves": {},
            "stats": {},
            "result": None,
            "weights_revision": 0,
            "done": False,
        }
        self.frames: list[dict[str, Any]] = []
        self.stop = threading.Event()
        self._previous: np.ndarray | None = None
        self._change: np.ndarray | None = None

    # -- what the page reads

    def report(self, phase: str, progress: float, status: str = "") -> None:
        with self.lock:
            self.events["phase"], self.events["progress"] = phase, progress
            if status:
                self.events["status"] = status

    def point(self, curve: str, x: float, y: float) -> None:
        with self.lock:
            self.events["curves"].setdefault(curve, []).append([round(x, 4), round(y, 6)])

    def stat(self, **values: float | int | str) -> None:
        with self.lock:
            for key, value in values.items():
                self.events["stats"][key] = round(value, 6) if isinstance(value, float) else value

    def frame(
        self, activity: np.ndarray, view: dict[str, Any], heat: np.ndarray | None = None
    ) -> None:
        item: dict[str, Any] = {"activity": [round(float(v), 3) for v in activity], "view": view}
        if heat is not None:
            item["heat"] = [round(float(v), 3) for v in heat]
        with self.lock:
            self.frames.append(item)

    def bump_weights(self) -> None:
        """The synapses changed: remember the change since the last bump for the page."""
        current = np.abs(self.weights())
        with self.lock:
            self._change = np.abs(current - self._previous) if self._previous is not None else None
            self._previous = current
            self.events["weights_revision"] += 1

    def finish(self, result: dict[str, Any]) -> None:
        with self.lock:
            self.events["result"], self.events["done"] = result, True
            self.events["phase"], self.events["progress"] = "done", 1.0

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self.lock:
            view = dict(self.events)
            view["curves"] = {k: list(v) for k, v in self.events["curves"].items()}
            view["stats"] = dict(self.events["stats"])
            view["frames"] = self.frames[since : since + 300]
            view["first"] = since
            view["total"] = len(self.frames)
            return view

    def weights_json(self, what: str = "weights") -> dict[str, Any]:
        with self.lock:
            revision = self.events["weights_revision"]
            change = self._change
        values = np.abs(self.weights()) if what != "change" or change is None else change
        return {"weights": [round(float(w), 5) for w in values], "revision": revision, "what": what}

    # -- what a demo supplies

    def connectome(self) -> Connectome:
        raise NotImplementedError

    def weights(self) -> np.ndarray:
        raise NotImplementedError

    def roles(self) -> dict[str, str]:
        return {}

    def atlas_json(self) -> str:
        self._connectome = self.connectome()
        atlas = build_atlas(self._connectome, np.abs(self.weights()), roles=self.roles(), seed=0)
        return atlas.to_json()

    def run(self) -> dict[str, Any]:
        raise NotImplementedError


def _dense(
    pre: list[int],
    post: list[int],
    weights: list[float],
    table: np.ndarray,
    rows: range,
    cols: range,
) -> None:
    """Add every entry of ``table[j, i]`` as a synapse from ``cols[i]`` to ``rows[j]``."""
    for j, row in enumerate(rows):
        for i, col in enumerate(cols):
            if row != col:
                pre.append(col)
                post.append(row)
                weights.append(float(table[j, i]))


def _ordered(connectome: Connectome, table: dict[tuple[int, int], float]) -> np.ndarray:
    return np.array(
        [table[(int(p), int(q))] for p, q in zip(connectome.pre, connectome.post, strict=True)]
    )


class StreamDemo(Demo):
    """The record patch: a stream by day, written once; a night in which the slow weights
    learn from the store's own dreams; then the weights alone know the rule."""

    name = "stream"
    title = "A record patch learns a stream, remembers in one shot, and sleeps"
    task = (
        "Three streams of eight symbols, each symbol a number from 0 to 11, play one moment at "
        "a time. At every moment the brain has to say one of five outcomes: the current symbol "
        "plus the previous one, modulo 5. It is never told that rule; after each moment it is "
        "shown the right outcome. By day it stores what it saw once; by night, with the streams "
        "closed, it learns the rule from its own memory; afterwards it says every outcome right "
        "with its memory erased."
    )
    note = (
        "Twelve symbols come in three streams of eight; the outcome of each moment is decided "
        "by the last two symbols. The context path is the patch's equilibrium: at every moment "
        "it satisfies its gate equation exactly, so the seam defect is zero and the reading is "
        "settled before the records are read. By day every outcome is written once into the "
        "records, with the slow weights at rate zero. At night the slow weights learn the "
        "store's own completions with the stream closed: the teaching loss pulls on the "
        "outcome ports and the adjoint scan carries that pull back along the settled path, "
        "which for this quadratic path equals the centered detuning contrast. At dawn the "
        "dreams are written back. Afterwards the weights alone, with an empty store, know the "
        "rule."
    )

    def __init__(self, *, day_passes: int = 8, night_passes: int = 240) -> None:
        super().__init__()
        self.day_passes, self.night_passes = day_passes, night_passes
        rng = np.random.default_rng(21)
        self.net = RecordPatchNet(
            12, 12, 5, seed=4, cells=2048, active=16, record_rate=1.0, groups=(5,), slowest=8.0
        )
        self.symbols = rng.integers(12, size=(3, 8))
        self.heard = np.eye(12)[self.symbols]
        self.outcome = np.eye(5)[(self.symbols + np.roll(self.symbols, 1, axis=1)) % 5]
        self.cells = self.net.records.cells

    # neurons: senses 0-11, context 12-23, cells 24.., outcomes at the end
    def _index(self) -> tuple[range, range, range, range]:
        senses = range(0, 12)
        context = range(12, 24)
        cells = range(24, 24 + self.cells)
        outcomes = range(24 + self.cells, 24 + self.cells + 5)
        return senses, context, cells, outcomes

    def _table(self) -> dict[tuple[int, int], float]:
        senses, context, cells, outcomes = self._index()
        parameters = self.net.parameters()
        table: dict[tuple[int, int], float] = {}
        b, c_map = parameters["B"], parameters["C"]
        for j, row in enumerate(context):
            for i, col in enumerate(senses):
                table[(col, row)] = float(b[j, i])
        projection = self.net.records.projection
        for r, source in enumerate(list(senses) + list(context)):
            for c, cell in enumerate(cells):
                table[(source, cell)] = float(projection[r, c])
        store = self.net.records.tables["y"]
        for c, cell in enumerate(cells):
            for o, outcome in enumerate(outcomes):
                table[(cell, outcome)] = float(store[c, o])
        for o, outcome in enumerate(outcomes):
            for j, col in enumerate(context):
                table[(col, outcome)] = float(c_map[o, j])
        return table

    def connectome(self) -> Connectome:
        senses, context, cells, outcomes = self._index()
        table = self._table()
        pre = [p for p, _ in table]
        post = [q for _, q in table]
        populations = {
            "senses": list(senses),
            "context": list(context),
            "records": list(cells),
            "outcome": list(outcomes),
        }
        n = 24 + self.cells + 5
        return Connectome.from_synapses(
            n, pre=pre, post=post, populations=populations, label="stream"
        )

    def weights(self) -> np.ndarray:
        return _ordered(self._connectome, self._table())

    def roles(self) -> dict[str, str]:
        return {
            "senses": "sensory",
            "context": "association",
            "records": "memory",
            "outcome": "motor",
        }

    # -- measurements

    def _accuracy(self, net: RecordPatchNet) -> float:
        net.reset()
        path = net.imagine(self.heard, state=np.zeros((3, 12)))
        return float(np.mean(path.output.argmax(-1) == self.outcome.argmax(-1)))

    def _slow_loss(self) -> float:
        """The cross-entropy of the slow weights alone on the stream, the night's teaching pull."""
        alone = self._slow_only()
        alone.reset()
        logits = alone.imagine(self.heard, state=np.zeros((3, 12))).output
        logits = logits - logits.max(axis=-1, keepdims=True)
        log_soft = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        return float(-np.mean(np.sum(self.outcome * log_soft, axis=-1)))

    def _slow_only(self) -> RecordPatchNet:
        alone = RecordPatchNet.restore(self.net.snapshot())
        alone.records.tables["y"][:] = 0.0
        return alone

    def _frames(self, label: str, x: float, slow: bool = False) -> None:
        """One frame per moment of the three streams: senses, context, the code, the outcome;
        the heat on the outcome ports is the teaching pull, softmax minus target."""
        net = self._slow_only() if slow else self.net
        net.reset()
        path = net.imagine(self.heard, state=np.zeros((3, 12)))
        readings = self.net._readings(self.heard, path.hidden).reshape(24, -1)
        codes = self.net.records.code(readings, valued=False)[0].reshape(3, 8, -1)
        prediction = path.output.argmax(-1)
        for stream in range(3):
            for t in range(8):
                logits = path.output[stream, t] - path.output[stream, t].max()
                soft = np.exp(logits) / np.exp(logits).sum()
                activity = np.concatenate(
                    [
                        self.heard[stream, t],
                        np.abs(path.hidden[stream, t]),
                        codes[stream, t] > 0,
                        soft,
                    ]
                )
                heat = np.zeros_like(activity)
                heat[-5:] = np.abs(soft - self.outcome[stream, t])
                self.frame(
                    activity,
                    {
                        "kind": label,
                        "x": x,
                        "stream": stream,
                        "moment": t,
                        "symbol": int(self.symbols[stream, t]),
                        "target": int(self.outcome[stream, t].argmax()),
                        "predicted": int(prediction[stream, t]),
                        "slow": slow,
                    },
                    heat,
                )

    def run(self) -> dict[str, Any]:
        self.report("day", 0.0, "the day: one write per moment, slow weights at rate zero")
        for day in range(self.day_passes):
            self.net.reset()
            observed = self.net.observe(self.heard, self.outcome, rate=0.0)
            self.point("records", day + 1, self._accuracy(self.net))
            self.point("slow", day + 1, self._accuracy(self._slow_only()))
            self.point("slow_loss", day + 1, self._slow_loss())
            self.stat(
                pass_=day + 1,
                writes=self.net.readback().writes,
                record_entries=self.net.readback().record_entries,
                slow_loss=float(observed.prediction.slow_loss or 0.0),
                loss=float(observed.prediction.loss or 0.0),
                seam_defect=0.0,
            )
            self._frames("day", day + 1)
            self.bump_weights()
            self.report("day", (day + 1) / self.day_passes)
        awake = self._accuracy(self.net)
        self.report("night", 0.0, "the night: the slow weights learn the store's own dreams")
        dream = self.net.dream(self.heard)
        admitted = 0
        for night in range(self.night_passes):
            self.net.reset()
            observed = self.net.observe(self.heard, dream, rate=8.0, backtrack=True, write=False)
            admitted += int(observed.updated)
            if night % 8 == 7 or night == self.night_passes - 1:
                x = self.day_passes + night + 1
                self.point("slow", x, self._accuracy(self._slow_only()))
                self.point("records", x, self._accuracy(self.net))
                self.point("slow_loss", x, self._slow_loss())
                self.stat(
                    pass_=x,
                    updates=admitted,
                    slow_loss=float(observed.final_loss or observed.initial_loss or 0.0),
                    accepted_rate=float(observed.accepted_rate or 0.0),
                    seam_defect=0.0,
                )
                self._frames("night", x, slow=True)
                self.bump_weights()
                self.report("night", (night + 1) / self.night_passes)
        dawn = self.net.sleep([self.heard], passes=0, rate=8.0, dawn_passes=2)
        alone = self._accuracy(self._slow_only())
        self.stat(
            dawn_writes=int(dawn["dawn_writes"]), record_entries=self.net.readback().record_entries
        )
        self._frames("dawn", self.day_passes + self.night_passes + 1, slow=True)
        self.bump_weights()
        result = {
            "awake_after_day": awake,
            "slow_alone_after_night": alone,
            "night_updates": admitted,
            "dawn_writes": dawn["dawn_writes"],
            "record_entries": self.net.readback().record_entries,
        }
        self.finish(result)
        return result


class DecideDemo(Demo):
    """A settling brain of regions: five senses, a cortex, three actions; a learner over the
    motor neurons nudges the chosen action and moves the synapses on the contrast."""

    name = "decide"
    title = "A settling brain decides"
    task = (
        "Five senses, one at a time, and three possible actions. Sense k asks for action k mod "
        "3: sense 0 wants action 0, sense 1 action 1, sense 2 action 2, sense 3 action 0, sense "
        "4 action 1. The brain answers with its most active motor neuron and is then shown the "
        "action that was wanted. It has to learn to choose the right action for all five senses."
    )
    note = (
        "A genome names three regions and two projections; develop lays them out as one "
        "connectome; a Brain settles it. Settling is the equilibrium: every neuron's potential "
        "moves until no neuron moves, and the state it stops in is the brain's own answer. "
        "Learning is equilibrium detuning: the same brain settles once more with the chosen "
        "action nudged, and the synapses move on the contrast of the two equilibria. Nothing "
        "is propagated backward; the difference between two settled states is the signal. "
        "Sense k asks for action k mod 3."
    )

    def __init__(self, *, steps: int = 80) -> None:
        super().__init__()
        self.steps = steps
        genome = Genome(
            regions=(Region("senses", 5), cortex(24), motor_cortex(3, lateral=-0.5)),
            projections=(
                Projection("senses", "association", reciprocal=False),
                Projection("association", "motor"),
            ),
        )
        self.connectome_ = develop(genome, seed=0)
        self.brain = Brain(self.connectome_, learning_neuron_model())
        self.senses = list(self.connectome_.populations["senses"])
        self.learner = Learner(
            self.brain, self.connectome_.populations["motor/actions"], LearnerConfig(eta=1.0)
        )
        self.drive = np.zeros((5, self.connectome_.n))
        self.drive[np.arange(5), self.senses] = 1.0
        self.labels = np.arange(5) % 3

    def connectome(self) -> Connectome:
        return self.connectome_

    def weights(self) -> np.ndarray:
        # The learner replaces its brain on every update; read the current one.
        return np.asarray(self.learner.brain.weights, dtype=float)

    def atlas_json(self) -> str:
        self._connectome = self.connectome_
        return atlas_of(self.learner.brain, seed=0).to_json()

    def run(self) -> dict[str, Any]:
        self.report("calibrate", 0.0, "the gain that puts the free motor activity in its range")
        gain = self.learner.calibrate(self.drive)
        self.bump_weights()
        self.report("learn", 0.0, "a free settle, a nudged settle, one update, eighty times")
        for step in range(self.steps):
            records: list[SettlementRecord] = []
            with record_settlements(records.append, label="step"):
                _, report = self.learner.step(self.drive, self.labels)
            row = step % 5
            chosen = int(self.learner.predict(self.drive)[row])
            free = records[0]
            movement = np.abs(np.diff(free.activation, axis=0))
            common = {
                "step": step + 1,
                "sense": row,
                "target": int(self.labels[row]),
                "chosen": chosen,
                "of": int(free.steps),
            }
            for s in range(0, free.steps + 1, 2):
                heat = movement[s - 1, row] if s else np.zeros(self.connectome_.n)
                self.frame(
                    np.abs(free.activation[s, row]),
                    {"kind": "free", "settle": s, **common},
                    heat,
                )
            if len(records) > 1:
                nudged = records[1]
                displacement = np.abs(nudged.activation[-1, row] - free.activation[-1, row])
                self.frame(
                    np.abs(nudged.activation[-1, row]),
                    {"kind": "nudged", "settle": int(nudged.steps), **common},
                    displacement,
                )
                nudged_steps = int(nudged.steps)
                displaced = float(displacement.max())
            else:
                nudged_steps, displaced = 0, 0.0
            before = np.abs(self._previous) if self._previous is not None else None
            self.bump_weights()
            contrast = (
                float(np.linalg.norm(np.abs(self.weights()) - before))
                if before is not None
                else 0.0
            )
            accuracy = self.learner.accuracy(self.drive, self.labels)
            self.point("accuracy", step + 1, accuracy)
            self.point("contrast", step + 1, contrast)
            self.stat(
                step=step + 1,
                settle_steps=int(free.steps),
                last_movement=float(movement[-1].max()) if len(movement) else 0.0,
                nudged_steps=nudged_steps,
                displacement=displaced,
                contrast=contrast,
                accuracy=accuracy,
            )
            self.report("learn", (step + 1) / self.steps)
        accuracy = self.learner.accuracy(self.drive, self.labels)
        result = {"accuracy": accuracy, "gain": gain, "updates": self.learner.updates}
        self.finish(result)
        return result


def _actual_step(position: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.asarray(0.85 * position + 0.2 * np.tanh(2 * action) - 0.05 * position**3)


def _actual_paths(
    rng: np.random.Generator, batch: int, horizon: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    position = rng.uniform(-0.6, 0.6, batch)
    actions = rng.uniform(-1.0, 1.0, (batch, horizon))
    inputs = np.zeros((batch, horizon, 3))
    inputs[:, 0, 0] = position
    inputs[:, 0, 1] = 1.0
    inputs[:, :, 2] = actions
    measured = np.empty((batch, horizon, 1))
    for t in range(horizon):
        position = _actual_step(position, actions[:, t])
        measured[:, t, 0] = position
    return inputs, measured


class BodyDemo(Demo):
    """The temporal patch learns how a small body moves, then plans an action toward a goal,
    executes the first, reads what happened, and plans again; a shove arrives halfway."""

    name = "body"
    title = "A temporal patch learns a consequence and plans"
    task = (
        "A body on a line: one measured position, and one push between minus one and plus one "
        "per step; the body moves by a rule it is never shown. First the brain has to learn how "
        "pushes move the body, from 256 batches of random pushes and the positions they "
        "produced, well enough to predict six steps ahead on paths it has never seen. Then it "
        "has to drive the body to the goal position 0.35 and hold it there: it plans six pushes "
        "ahead under its own learned model, executes the first, and replans from the measured "
        "position, through an unannounced shove halfway."
    )
    note = (
        "A body with one measured position and one bounded action; only executed actions and "
        "measured positions reach the learner. The free path is the equilibrium of the patch's "
        "energy: the causal recurrence with zero defect at every seam. Teaching detunes it: the "
        "same path is settled twice more with the observed positions pulling by plus and minus "
        "beta, and the synapses move on the difference of the two settled paths, checked for "
        "symmetry around the free one. Planning uses the same contrast on the action port with "
        "the synapses frozen. Halfway through the run an unannounced disturbance moves the body."
    )

    def __init__(self, *, batches: int = 256, decisions: int = 12) -> None:
        super().__init__()
        self.batches, self.decisions = batches, decisions
        self.net = TemporalPatchNet(3, 16, 1, seed=709, initial_radius=0.8)
        self.hidden = 16

    # neurons: inputs 0-2, hidden 3-18, output 19
    def _table(self) -> dict[tuple[int, int], float]:
        parameters = self.net.parameters()
        a, b, c = parameters["A"], parameters["B"], parameters["C"]
        table: dict[tuple[int, int], float] = {}
        for j in range(self.hidden):
            for i in range(3):
                table[(i, 3 + j)] = float(b[j, i])
            for k in range(self.hidden):
                if k != j:
                    table[(3 + k, 3 + j)] = float(a[j, k])
            table[(3 + j, 19)] = float(c[0, j])
        return table

    def connectome(self) -> Connectome:
        table = self._table()
        populations = {"senses": [0, 1, 2], "body": list(range(3, 19)), "prediction": [19]}
        return Connectome.from_synapses(
            20,
            pre=[p for p, _ in table],
            post=[q for _, q in table],
            populations=populations,
            label="body",
        )

    def weights(self) -> np.ndarray:
        return _ordered(self._connectome, self._table())

    def roles(self) -> dict[str, str]:
        return {"senses": "sensory", "body": "association", "prediction": "vision"}

    def _activity(self, inputs: np.ndarray, hidden: np.ndarray, output: np.ndarray) -> np.ndarray:
        return np.concatenate([np.abs(inputs), np.abs(np.tanh(hidden)), np.abs(output)])

    def run(self) -> dict[str, Any]:
        rng = np.random.default_rng(10709)
        test_inputs, test_measured = _actual_paths(np.random.default_rng(20709), 128)
        self.report("learn", 0.0, "learning the consequences of actions from executed paths")
        forecast = float("nan")
        for batch in range(self.batches):
            executed, measured = _actual_paths(rng, 8)
            self.net.reset()
            learned = self.net.observe(executed, measured, beta=0.01, rate=1.0)
            if batch % 8 == 7 or batch == self.batches - 1:
                prediction = self.net.imagine(test_inputs, state=np.zeros((128, self.hidden)))
                forecast = float(np.mean((prediction.output - test_measured) ** 2))
                self.point("forecast", batch + 1, forecast)
                free, plus, minus = learned.free, learned.plus, learned.minus
                update = (
                    float(np.sqrt(sum(float(np.sum(v * v)) for v in learned.delta.values())))
                    if learned.delta
                    else 0.0
                )
                self.point("update", batch + 1, update)
                asymmetry = (
                    contrast_asymmetry(free, plus, minus)
                    if plus is not None and minus is not None
                    else float("nan")
                )
                self.stat(
                    batch=batch + 1,
                    forecast_mse=forecast,
                    residual=float(free.residual or 0.0),
                    energy=float(free.energy or 0.0),
                    plus_iterations=int(plus.iterations) if plus is not None else 0,
                    minus_iterations=int(minus.iterations) if minus is not None else 0,
                    asymmetry=float(asymmetry),
                    beta=float(learned.beta or 0.0),
                    halvings=int(learned.contrast_halvings),
                    update=update,
                    reason=learned.reason,
                )
                for t in range(executed.shape[1]):
                    heat = np.zeros(20)
                    if plus is not None:
                        heat[3:19] = np.abs(np.tanh(plus.hidden[0, t]) - np.tanh(free.hidden[0, t]))
                    self.frame(
                        self._activity(executed[0, t], free.hidden[0, t], free.output[0, t]),
                        {
                            "kind": "learn",
                            "batch": batch + 1,
                            "moment": t,
                            "measured": float(measured[0, t, 0]),
                            "predicted": float(free.output[0, t, 0]),
                        },
                        heat,
                    )
                self.bump_weights()
                self.report("learn", (batch + 1) / self.batches)
        self.report("act", 0.0, "planning under the frozen model, acting, reading the result")
        position, goal = -0.45, 0.35
        zero = position
        measured_positions: list[float] = []
        for t in range(self.decisions):
            if t == self.decisions // 2:
                position -= 0.25
                zero -= 0.25
            inputs = np.zeros((1, 6, 3))
            inputs[0, 0, :2] = [position, 1.0]
            proposal = self.net.plan(
                inputs,
                goal=np.full((1, 6, 1), goal),
                controls=np.array([False, False, True]),
                bounds=(-1.0, 1.0),
                state=np.zeros((1, self.hidden)),
                beta=0.01,
                rate=8.0,
                max_steps=32,
            )
            action = float(proposal.inputs[0, 0, 2])
            predicted = [float(v) for v in proposal.prediction.output[0, :, 0]]
            planned = [float(v) for v in proposal.inputs[0, :, 2]]
            position = float(_actual_step(np.array([position]), np.array([action]))[0])
            zero = float(_actual_step(np.array([zero]), np.array([0.0]))[0])
            measured_positions.append(position)
            self.point("position", t + 1, position)
            self.point("without", t + 1, zero)
            self.stat(
                decision=t + 1,
                plan_iterations=proposal.iterations,
                plan_cost=float(proposal.cost),
                plan_initial_cost=float(proposal.initial_cost),
                projected_residual=float(proposal.projected_residual or 0.0),
                beta=float(proposal.beta),
                halvings=int(proposal.contrast_halvings),
                reason=proposal.reason,
                position=position,
            )
            for s in range(6):
                self.frame(
                    self._activity(
                        proposal.inputs[0, s],
                        proposal.prediction.hidden[0, s],
                        proposal.prediction.output[0, s],
                    ),
                    {
                        "kind": "act",
                        "decision": t + 1,
                        "moment": s,
                        "position": position,
                        "goal": goal,
                        "without": zero,
                        "predicted": predicted,
                        "planned": planned,
                        "iterations": proposal.iterations,
                        "shove": t == self.decisions // 2,
                    },
                )
            self.report("act", (t + 1) / self.decisions)
        late = float(np.mean((np.array(measured_positions[-3:]) - goal) ** 2))
        result = {"forecast_mse": forecast, "late_goal_mse": late, "goal": goal}
        self.finish(result)
        return result


demos: dict[str, type[Demo]] = {"stream": StreamDemo, "decide": DecideDemo, "body": BodyDemo}


def _page(demo: Demo) -> str:
    page = resources.files("quickstart").joinpath("demo_page.html").read_text(encoding="utf-8")
    config = {"name": demo.name, "title": demo.title, "task": demo.task, "note": demo.note}
    return (
        page.replace("__SCRIPT__", brain_scan_script().replace("export ", ""))
        .replace("__ATLAS__", demo.atlas_json())
        .replace("__DEMO__", json.dumps(config))
    )


def serve(demo: Demo, port: int = 8766, open_browser: bool = True) -> ThreadingHTTPServer:
    """Serve the page and the demo's state on localhost; the run starts at once."""
    page = _page(demo).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str = "application/json") -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/":
                self._send(page, "text/html; charset=utf-8")
            elif url.path == "/state":
                since = int(query.get("since", ["0"])[0])
                self._send(json.dumps(demo.snapshot(since), separators=(",", ":")).encode())
            elif url.path == "/weights":
                what = query.get("what", ["weights"])[0]
                self._send(json.dumps(demo.weights_json(what), separators=(",", ":")).encode())
            else:
                self.send_error(404)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return None

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=demo.run, daemon=True).start()
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{server.server_address[1]}/")
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A quickstart brain, live in your browser.")
    parser.add_argument("demo", choices=sorted(demos), help="stream, decide or body")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    demo = demos[args.demo]()
    started = time.time()
    server = serve(demo, port=args.port, open_browser=not args.no_browser)
    print(f"{demo.title}: http://127.0.0.1:{server.server_address[1]}/  (Ctrl-C stops it)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        demo.stop.set()
        server.server_close()
        print(f"served for {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
