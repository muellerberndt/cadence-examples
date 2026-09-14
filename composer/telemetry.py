"""Measure every owner and seam. Rehearsal traces use the composer's actual states."""

import base64

import numpy as np

from .topology import topology


class Recorder:
    def __init__(
        self, learner, drive, *, row=0, modulators=None, origin="actual rehearsal"
    ):
        self.engine = learner.engine
        self.drive = drive[row]
        self.row = row
        self.graph = topology(self.engine)
        self.selected = np.asarray(self.graph["owner_ids"])
        self.frames = []
        self.populations = []
        self.errors = []
        self.iterations = []
        self.previous = None
        self.modulators = modulators or {}
        self.origin = origin

    def observe(self, iteration, state):
        engine = self.engine
        w = engine.wiring
        s = state.activation[self.row]
        v = state.v[self.row]
        inbox = np.bincount(w.post, weights=engine.weights * s[w.pre], minlength=w.n)
        error = inbox + self.drive + engine.bias - v
        if engine.rule.adaptation:
            error -= engine.rule.adaptation.strength * state.adaptation[self.row]
        repair = np.zeros(w.n) if self.previous is None else s - self.previous
        self.frames.append(
            {
                "activation": s[self.selected].tolist(),
                "mismatch": error[self.selected].tolist(),
                "repair": repair[self.selected].tolist(),
            }
        )
        self.populations.append(
            {
                name: {
                    "mean": float(s[list(ids)].mean()),
                    "rms": float(np.sqrt(np.mean(s[list(ids)] ** 2))),
                    "mismatch": float(np.sqrt(np.mean(error[list(ids)] ** 2))),
                    "repair": float(np.sqrt(np.mean(repair[list(ids)] ** 2))),
                }
                for name, ids in w.sets.items()
            }
        )
        self.errors.append(float(np.max(np.abs(error))))
        self.iterations.append(iteration)
        self.previous = s.copy()

    def finish(self, *, packed=False):
        g = self.graph
        result = {
            "frames": self.frames,
            "owner_ids": g["owner_ids"],
            "populations": self.populations,
            "equation_error": self.errors,
            "regions": g["regions"],
            "edges": [],
            "topology": g,
            "total_owners": g["owners"],
            "total_seams": g["seams"],
            "modulators": self.modulators,
            "steps": self.iterations[-1],
            "iterations": self.iterations,
            "origin": self.origin,
            "display": "All owners and all seams. Changes are between recorded iterations; model units, not EEG.",
        }
        if packed:
            array = np.asarray(
                [
                    [f[k] for k in ["activation", "repair", "mismatch"]]
                    for f in self.frames
                ],
                dtype="<f4",
            )
            result["encoded_frames"] = {
                "shape": list(array.shape),
                "format": "float32le",
                "data": base64.b64encode(array.tobytes()).decode("ascii"),
            }
            del result["frames"]
        return result


def capture(
    learner,
    drive,
    *,
    steps=512,
    state=None,
    modulators=None,
    include_release=True,
    packed=False,
):
    recorder = Recorder(
        learner, drive, modulators=modulators, origin="isolated diagnostic replay"
    )
    engine = learner.engine
    current = engine.settle_batch(drive, steps=0) if state is None else state
    for tick in range(steps + 1):
        if tick:
            current = engine.settle_batch(drive, steps=1, state=current)
        error = engine.residual(drive, current)
        done = tick == steps or tick >= 32 and np.all(error < 1e-5)
        if tick <= 16 or tick % 16 == 0 or done:
            recorder.observe(tick, current)
        if done:
            break
    result = recorder.finish(packed=packed)
    if include_release:
        result["release"] = capture(
            learner,
            np.zeros_like(drive),
            steps=512,
            state=current,
            modulators=modulators,
            include_release=False,
            packed=packed,
        )
        result["release"]["released"] = True
    return result
