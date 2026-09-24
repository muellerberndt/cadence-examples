"""Export a cadence Brain for the browser engine, and record what the library does so the
engine can be held to it.

    from engine.export import write_payload, settle_cases, record_lessons

    write_payload(brain, "web/data/brain.json")                      # the brain, by receiving neuron
    settle_cases(brain, {"touch": {"input": 1.0}}, ["output"], out="tests/parity_cases.json")
    record_lessons(learner, critic, drives, rewards, dones, out="tests/lessons.json")

The payload holds the synapses in the library's order (sorted by receiving neuron, then
sender), the weights the library settles with (gain, count, sign and any per-neuron gain
folded in), the synapse counts and signs (so a learned efficacy can replace a weight), every
population by name, and the neuron model. `parity.mjs` replays the cases and the lessons.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

import cadence
from cadence.plasticity import ActorCritic, ActorCriticConfig

FORMAT = "cadence.brain-payload/1"


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def payload_of(brain: Any, *, members: np.ndarray | None = None, extra: dict | None = None) -> dict:
    """The brain as the engine reads it. ``members`` are this brain's indices in a larger one."""
    C = brain.connectome
    model = brain.neuron_model
    order = np.lexsort((C.pre, C.post))  # by receiving neuron, then sender: the library's order
    if not np.array_equal(order, np.arange(C.synapses)):
        raise ValueError("the connectome's synapses must be sorted by post, then pre (cadence.Connectome sorts them)")
    row_ptr = np.zeros(C.n + 1, dtype=np.int32)
    np.cumsum(np.bincount(C.post, minlength=C.n), out=row_ptr[1:])
    gain_pre = float(model.gain) * np.asarray(C.count, dtype=np.float64) * np.exp(np.asarray(brain.log_gain, dtype=np.float64)[C.pre])
    weights = np.asarray(brain.weights, dtype=np.float64)
    if not np.allclose(gain_pre * np.asarray(brain.efficacy, dtype=np.float64), weights, rtol=0, atol=1e-12):
        raise ValueError("the weights are not gain * count * exp(log_gain[pre]) * efficacy; the engine cannot reproduce this brain")
    sign = np.asarray(C.sign, dtype=np.float64)
    integral_sign = bool(np.all(sign == np.round(sign)) and np.abs(sign).max() <= 127)
    arrays = {
        "row_ptr": _b64(row_ptr), "pre": _b64(C.pre.astype(np.int32)), "weight": _b64(weights),
        "count": _b64(np.minimum(C.count, 65535).astype(np.uint16)),
        "sign": _b64(sign.astype(np.int8) if integral_sign else sign), "sign_dtype": "int8" if integral_sign else "float64",
        "bias": _b64(np.asarray(brain.bias, dtype=np.float64)),
    }
    # the two arrays the engine can derive are left out when it can (a brain without per-neuron gains, no lesson yet)
    if np.any(np.asarray(brain.log_gain) != 0.0) or np.any(C.count > 65535):
        arrays["gain_pre"] = _b64(gain_pre)  # the factor a synapse's efficacy is multiplied by: gain * count * exp(log_gain[pre])
    if not np.array_equal(np.asarray(brain.efficacy, dtype=np.float64), sign):
        arrays["efficacy"] = _b64(np.asarray(brain.efficacy, dtype=np.float64))  # signed, the connectome's sign until a lesson moves it
    if members is not None:
        arrays["members"] = _b64(np.asarray(members, dtype=np.int32))
    return {
        "format": FORMAT, "library": {"version": cadence.__version__},
        "n": int(C.n), "edges": int(C.synapses), "synapses": int(C.count.sum()),
        "model": {**model.to_dict(), "gain": float(model.gain)},
        "populations": {k: [int(i) for i in v] for k, v in C.populations.items()},
        "arrays": arrays, **(extra or {}),
    }


def write_payload(brain: Any, out: str | Path, **kw: Any) -> Path:
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload_of(brain, **kw), separators=(",", ":")))
    return out


def dense_drive(brain: Any, stimulus: dict[str, float]) -> np.ndarray:
    """A stimulus by population name and level, as the dense drive the library settles under (levels combine by max)."""
    C = brain.connectome
    levels = np.zeros(C.n)
    for name, level in stimulus.items():
        idx = list(C.populations[name])
        levels[idx] = np.maximum(levels[idx], level)
    return brain.stimulus_levels(levels)


def settle_cases(brain: Any, stimuli: dict[str, dict[str, float]], readouts: list[str], *, steps: int = 40, out: str | Path | None = None) -> dict:
    """The library's per-step population means under each stimulus, for parity.mjs."""
    C = brain.connectome
    readouts = [r for r in readouts if r in C.populations]
    cases = []
    for name, stimulus in stimuli.items():
        state = brain.settle_batch(dense_drive(brain, stimulus)[None, :], steps=steps, trajectory=True)
        per_step = [[float(state.trajectory[t, 0, list(C.populations[r])].mean()) for r in readouts] for t in range(state.trajectory.shape[0])]
        cases.append({"name": name, "stimulus": stimulus, "readouts": readouts, "per_step_means": per_step, "final_active": int((state.activation[0] >= 0.5).sum())})
    result = {"format": "cadence.settle-cases/1", "steps": steps, "cases": cases}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(result, indent=1))
    return result


class _Replay:
    """A stand-in for the actor-critic's random generator that records every uniform it hands out."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed); self.log: list[float] = []

    def random(self, shape: Any = None) -> np.ndarray:
        u = self.rng.random(shape)
        self.log.append(float(np.ravel(u)[0]))
        return u


def record_lessons(learner: Any, critic: Any, drives: list[np.ndarray], rewards: list[float], dones: list[bool], *, config: ActorCriticConfig | None = None, seed: int = 0, out: str | Path | None = None, phases: bool = False) -> dict:
    """Run the library's actor-critic for one stream over recorded observations and write what it did:
    the uniform draws, the actions, the dopamine, and the plastic efficacies and critic at the end.
    ``drives`` are dense stimulus levels (before the amplitude), one per decision plus one for the
    state after the last."""
    from cadence.learning import SCALE_CAP

    config = config or ActorCriticConfig()
    outside = []
    if learner.reciprocal: outside.append("reciprocal=True (the engine gives every synapse its own efficacy)")
    if learner.tie_groups is not None: outside.append("tie_groups")
    if learner.synapse_rate is not None: outside.append("synapse_rate")
    if learner.slot_count != 1: outside.append("slots")
    if learner.config.nudge != "cross_entropy" or not learner.config.centered: outside.append("a quadratic or uncentred nudge")
    if learner.config.decay or learner.config.normalize or learner.config.momentum: outside.append("decay, normalize or momentum")
    if config.normalize or config.momentum or config.dopamine_center or config.dopamine_floor: outside.append("normalize, momentum or dopamine centring")
    if not config.critic_normalize: outside.append("critic_normalize=False")
    if outside:
        raise ValueError("the browser engine does not implement: " + ", ".join(outside))
    ac = ActorCritic(learner, critic, config, seed=seed)
    ac.rng = _Replay(seed)
    brain = learner.brain
    amp = brain.neuron_model.stimulus_amplitude
    plastic = np.flatnonzero(learner.plastic_synapses)
    decisions = []
    for t, (levels, reward, done) in enumerate(zip(drives[:-1], rewards, dones, strict=True)):
        action = int(ac.act(amp * np.asarray(levels)[None, :])[0])
        u = ac.rng.log[-1]
        record = {"drive": [[int(i), float(levels[i])] for i in np.flatnonzero(levels)], "u": u, "action": action, "reward": float(reward), "done": bool(done)}
        if phases:  # the settled phases themselves, for a small brain: what the engine must reproduce step for step
            kind, plus, minus, _ = ac._pending
            free = ac.state
            record["phases"] = {"free": [float(x) for x in free.activation[0]], "free_steps": int(free.steps),
                                "plus": [float(x) for x in plus.activation[0]], "plus_steps": int(plus.steps),
                                "minus": [float(x) for x in minus.activation[0]], "minus_steps": int(minus.steps)}
        report = ac.learn(np.array([float(reward)]), np.array([bool(done)]), amp * np.asarray(drives[t + 1])[None, :])
        record.update({"dopamine": float(report["dopamine"]), "value": float(report["value"]), "saturation": float(report.get("saturation", float("nan")))})
        if phases:
            record["phases"]["efficacy"] = [float(x) for x in learner.brain.efficacy[plastic]]
        decisions.append(record)
    lcfg = learner.config
    recording = {
        "format": "cadence.lessons/1", "library": cadence.__version__,
        "config": {"outputs": [int(i) for i in learner.output_index], "plastic": [int(e) for e in plastic], "critic": [int(i) for i in ac.critic_index],
                   "plasticNeurons": [int(i) for i in np.flatnonzero(learner.plastic_neurons)], "etaBias": config.eta_bias,
                   "beta": lcfg.beta, "temperature": lcfg.temperature, "freeSteps": lcfg.free_steps, "nudgedSteps": lcfg.nudged_steps, "tolerance": lcfg.tolerance,
                   "gamma": config.gamma, "lam": config.lam, "eta": config.eta, "etaCritic": config.eta_critic, "dopamineCap": config.dopamine_cap,
                   "cap": float(getattr(lcfg, "scale_cap", SCALE_CAP)), "criticNormalize": config.critic_normalize},
        "next_drive": [[int(i), float(drives[-1][i])] for i in np.flatnonzero(drives[-1])],
        "decisions": decisions,
        "efficacy": [float(x) for x in learner.brain.efficacy[plastic]],
        "bias": [float(x) for x in np.asarray(learner.brain.bias)[np.flatnonzero(learner.plastic_neurons)]],
        "w_critic": [float(x) for x in ac.w_critic], "b_critic": float(ac.b_critic),
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(recording, separators=(",", ":")))
    return recording
