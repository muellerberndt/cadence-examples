"""Compare allowed damping values on held-out drives, retaining fixed-point and time checks."""

import copy
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.brain import drives
from composer.compose import Composer
from composer.music import parse_prompt

ROOT = Path(__file__).resolve().parents[1]
c = Composer()
brain = c.performer
rows = []
states = []
data = np.load(ROOT / "data/classical/validation.npz")
ids = np.random.default_rng(73).choice(len(data["labels"]), 16, replace=False)
context, extra = data["context"][ids], data["extra"][ids]
drive = drives(brain, context, extra)
for j, e in enumerate(extra):
    brief = replace(
        parse_prompt("calm piano"),
        mode=int(e[36:38].argmax()),
        style=int(e[38:41].argmax()),
        arousal=int(e[41:44].argmax()),
    )
    position = int(e[12:28].argmax() + 16 * e[28:36].argmax())
    brain.expect(drive[j : j + 1], brief, context[j : j + 1], np.array([position]))
brain.brain.settle_batch(drive[:1], steps=2)  # compile outside the timed interval
for dt in [1, 0.85, 0.65, 0.5, 0.3]:
    variant = copy.copy(brain.brain)
    variant.neuron_model = replace(variant.neuron_model, dt=dt)
    started = time.monotonic()
    state = variant.settle_batch(drive, steps=2048, tolerance=1e-9)
    elapsed = time.monotonic() - started
    rows.append(
        {
            "dt": dt,
            "steps": state.steps,
            "seconds": elapsed,
            "equation_error": float(variant.residual(drive, state).max()),
        }
    )
    states.append(state.activation)
    print(json.dumps(rows[-1]), flush=True)
reference = states[0]
for row, state in zip(rows, states):
    row["max_difference_from_dt1"] = float(np.max(np.abs(state - reference)))
report = {
    "checkpoint_sha256": hashlib.sha256(c.path.read_bytes()).hexdigest(),
    "rows": rows,
    "examples": 16,
    "split": "validation",
    "rule": "Only dt changes; weights, biases, drives and equation-error checks remain fixed.",
}
(ROOT / "runs/settling-probe.json").write_text(json.dumps(report, indent=2))
