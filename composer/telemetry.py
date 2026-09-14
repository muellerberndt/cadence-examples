"""Measure every neuron and synapse. Rehearsal traces use the composer's actual states."""

import base64

import numpy as np

from .topology import topology


class Recorder:
    def __init__(
        self, learner, drive, *, row=0, modulators=None, origin="actual rehearsal"
    ):
        self.brain = learner.brain
        self.drive = drive[row]
        self.row = row
        self.graph = topology(self.brain)
        self.selected = np.asarray(self.graph["neuron_ids"])
        self.frames = []
        self.populations = []
        self.errors = []
        self.iterations = []
        self.previous = None
        self.modulators = modulators or {}
        self.origin = origin

    def observe(self, iteration, state):
        brain = self.brain
        w = brain.connectome
        s = state.activation[self.row]
        v = state.v[self.row]
        synaptic_input = np.bincount(
            w.post, weights=brain.weights * s[w.pre], minlength=w.n
        )
        error = synaptic_input + self.drive + brain.bias - v
        if brain.neuron_model.adaptation:
            error -= brain.neuron_model.adaptation.strength * state.adaptation[self.row]
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
                for name, ids in w.populations.items()
            }
        )
        self.errors.append(float(np.max(np.abs(error))))
        self.iterations.append(iteration)
        self.previous = s.copy()

    def finish(self, *, packed=False):
        g = self.graph
        result = {
            "frames": self.frames,
            "neuron_ids": g["neuron_ids"],
            "populations": self.populations,
            "equation_error": self.errors,
            "regions": g["regions"],
            "edges": [],
            "topology": g,
            "total_neurons": g["neurons"],
            "total_synapses": g["synapses"],
            "modulators": self.modulators,
            "steps": self.iterations[-1],
            "iterations": self.iterations,
            "origin": self.origin,
            "display": "All neurons and all synapses. Changes are between recorded iterations; model units, not EEG.",
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
    brain = learner.brain
    current = brain.settle_batch(drive, steps=0) if state is None else state
    for tick in range(steps + 1):
        if tick:
            current = brain.settle_batch(drive, steps=1, state=current)
        error = brain.residual(drive, current)
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
