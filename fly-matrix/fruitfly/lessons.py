"""The lesson's setup on a sub-net: the operating points a connectome does not carry.

Three declared, generic steps (cadence: ``Brain(log_gain=...)``, ``naive_efficacy``,
``calibrate_bias``), each with its reason and its receipt (``receipts/lessons_setup.json``):

- the class gains of the dictionary (``fruitfly.brain.CLASS_LOG_GAIN``: the antennal lobe's
  local neurons at a twentieth of their measured output, selected by protocol on the Kenyon cell
  code, ``tools/class_gains.py``);
- the plastic seam started naive: every Kenyon-cell-to-MBON class at the same weight, because a
  specimen's counts at its memory site are that specimen's memories (on the measured counts the
  naive fly avoided the fruit odour and approached the yeast before any lesson);
- the two output cells calibrated to one half, jointly, over the situations the fly decides in
  (hovering over a fruit: that fruit's odour full on its receptor class, the other's at its plume
  level, the flight tone), so a nudge has a slope on each.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from cadence import Brain, Connectome, NeuronModel, calibrate_bias, naive_efficacy, seam_report

from .brain import GAIN, log_gain_for

__all__ = ["LessonSetup", "setup_lessons", "OUTPUTS", "SEAM", "DECISION_SENSES", "decision_drives"]

OUTPUTS = ("mbon:MBON11:right", "mbon:MBON05:left")  # approach, avoid (Aso et al. 2014)
SEAM = ("kc", "mbon")  # the plastic synapses: Kenyon cells onto every mushroom body output neuron
ODOURS = ("decaying_fruit", "yeasty")
OTHER_LEVEL = 0.16  # the other fruit's smell at a fruit (the page's plume, 25 cm away)
# what the fly senses while it hovers over a fruit and decides, the page's senses.js at that moment
DECISION_SENSES = {"haltere:left": 0.5, "haltere:right": 0.5, "ocelli:left": 0.71, "ocelli:right": 0.71,
                   "lptc:hs:left": 0.5, "lptc:vs:left": 0.5, "lptc:hs:right": 0.5, "lptc:vs:right": 0.5}
READOUT_TARGET = 0.5


def decision_drives(connectome: Connectome, amplitude: float, level: float = 1.0) -> np.ndarray:
    """Two rows of stimulus drive: hovering over the banana, hovering over the bread."""
    P = connectome.populations
    rows = np.zeros((2, connectome.n))
    for k, own in enumerate(ODOURS):
        for name, value in DECISION_SENSES.items():
            rows[k, list(P.get(name, ()))] = amplitude * value
        for j, odour in enumerate(ODOURS):
            for side in ("left", "right"):
                rows[k, list(P.get(f"orn:{odour}:{side}", ()))] = amplitude * (level if j == k else OTHER_LEVEL * level)
    return rows


@dataclass(frozen=True)
class LessonSetup:
    log_gain: np.ndarray
    efficacy: np.ndarray
    bias: np.ndarray
    plastic: np.ndarray  # bool per synapse
    outputs: tuple[int, ...]
    report: dict[str, Any]

    def brain(self, connectome: Connectome, *, backend: str = "cpu", gain: float = GAIN) -> Brain:
        return Brain(connectome, NeuronModel(gain=gain), log_gain=self.log_gain, efficacy=self.efficacy, bias=self.bias, backend=backend)


def setup_lessons(connectome: Connectome, *, backend: str = "cpu", gain: float = GAIN, naive: bool = True, calibrate: bool = True, steps: int = 100, tolerance: float | None = 1e-3) -> LessonSetup:
    """The class gains, the naive seam and the calibrated readouts on this sub-net, with the report."""
    P = connectome.populations
    for name in OUTPUTS:
        if len(P.get(name, ())) != 1:
            raise RuntimeError(f"{name} must be exactly one neuron in the sub-net, found {len(P.get(name, ()))}")
    outputs = tuple(int(P[name][0]) for name in OUTPUTS)
    pre = np.zeros(connectome.n, bool); pre[list(P[SEAM[0]])] = True
    post = np.zeros(connectome.n, bool); post[list(P[SEAM[1]])] = True
    plastic = pre[connectome.pre] & post[connectome.post]
    log_gain = log_gain_for(connectome)
    efficacy = naive_efficacy(connectome, plastic) if naive else np.array(connectome.sign, float)
    model = NeuronModel(gain=gain)
    base = Brain(connectome, model, log_gain=log_gain, efficacy=efficacy, backend=backend)
    drives = decision_drives(connectome, model.stimulus_amplitude)
    bias = calibrate_bias(base, drives, {outputs: READOUT_TARGET}, per_neuron=True, steps=steps, tolerance=tolerance) if calibrate else np.zeros(connectome.n)
    brain = base.with_parameters(bias=bias) if backend == "cpu" else Brain(connectome, model, log_gain=log_gain, efficacy=efficacy, bias=bias, backend=backend)
    state = brain.settle_batch(drives, steps=steps, tolerance=tolerance).activation
    seam = seam_report(connectome, SEAM[0], SEAM[1])
    report = {
        "outputs": {name: int(i) for name, i in zip(OUTPUTS, outputs, strict=True)},
        "seam": {"classes": seam["classes"], "synapses": seam["synapses"], "coverage_pre": seam["coverage_pre"], "median_count": seam["median_count"],
                 "classes_onto_outputs": {name: seam["classes_per_post"][i] for name, i in zip(OUTPUTS, outputs, strict=True)}},
        "naive_seam": bool(naive), "readout_target": READOUT_TARGET if calibrate else None,
        "readout_bias": {name: float(bias[i]) for name, i in zip(OUTPUTS, outputs, strict=True)},
        "naive_outputs": {odour: {name: float(state[k, i]) for name, i in zip(OUTPUTS, outputs, strict=True)} for k, odour in enumerate(ODOURS)},
        "decision_senses": DECISION_SENSES, "other_level": OTHER_LEVEL,
    }
    return LessonSetup(log_gain=log_gain, efficacy=efficacy, bias=bias, plastic=plastic, outputs=outputs, report=report)
