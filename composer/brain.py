"""One Cadence patch graph: auditory history, harmony, rhythm, phrase and note output."""

from dataclasses import dataclass

import cadence as cd
import numpy as np
from cadence.constitution import Constitution, Projection, Region, grow

from .encoding import EVENT_SIZE, INPUTS, SIZES, features


@dataclass
class Design:
    harmony: int = 96
    rhythm: int = 48
    phrase: int = 48
    seed: int = 17


def build(design=None, backend="cpu", device=None):
    design = design or Design()
    regions = (
        Region("auditory_history", INPUTS),
        Region("harmony", design.harmony),
        Region("rhythm", design.rhythm),
        Region("phrase_memory", design.phrase),
        Region("note_intention", EVENT_SIZE),
    )
    projections = []
    for name in ["harmony", "rhythm", "phrase_memory"]:
        projections += [
            Projection("auditory_history", name, scale=0.6, symmetric=False),
            Projection(name, "note_intention", scale=0.6),
        ]
    projections += [
        Projection("auditory_history", "note_intention", scale=0.5, symmetric=False),
        Projection("harmony", "phrase_memory", density=0.2, scale=0.2),
        Projection("rhythm", "phrase_memory", density=0.2, scale=0.2),
    ]
    wiring = grow(
        Constitution(regions, tuple(projections), label="composer"), seed=design.seed
    )
    engine = cd.Settlement(
        wiring,
        cd.learning_rule(dt=1, leak=0.1),
        backend=backend,
        device=device,
        precision="float32" if backend == "torch" else None,
    )
    inputs = np.array(wiring.sets["auditory_history"])
    trainable = np.ones(wiring.n, dtype=bool)
    trainable[inputs] = False
    config = cd.LearnerConfig(
        beta=0.3,
        eta=0.2,
        eta_bias=0.02,
        free_steps=24,
        nudged_steps=12,
        centered=True,
        tolerance=1e-4,
        temperature=0.2,
        normalize=0.98,
        momentum=0.8,
        normalize_floor=0.01,
        decay=1e-6,
    )
    learner = cd.Learner(
        engine,
        wiring.sets["note_intention"],
        config,
        trainable_owners=trainable,
        slots=SIZES,
    )
    return learner


def drives(learner, context, extra):
    x = features(context, extra)
    out = np.zeros((len(x), learner.engine.wiring.n), dtype=np.float32)
    out[:, :INPUTS] = x * 2.5
    return out


def probabilities(learner, context, extra, *, trace=False):
    drive = drives(learner, context, extra)
    if trace:
        state = learner.engine.settle_batch(
            drive, steps=64, tolerance=1e-5, trajectory=True
        )
    else:
        state = learner.free(drive)
    activation = state.activation[:, learner.output_index]
    result = []
    start = 0
    for width in SIZES:
        logits = activation[:, start : start + width] / learner.config.temperature
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        result.append(p)
        start += width
    return result, state


def describe(learner):
    w = learner.engine.wiring
    base = getattr(learner, "base", learner)
    return {
        "owners": w.n,
        "directed_seams": w.edges,
        "trainable_parameters": learner.parameters(),
        "regions": {k: len(v) for k, v in w.sets.items()},
        "learning": "Cadence centered free/nudged local contrast; no backpropagation graph",
        "inputs": len(w.sets.get("auditory_history", w.sets.get("heard_events", ()))),
        "output_slots": base.slot_sizes.tolist(),
    }


def settle_checked(
    learner, drive, *, tolerance=1e-5, budget=512, observer=None, warm=None
):
    """Continue the existing rule until its equations agree, or report the finite cap."""
    engine = learner.engine
    state = warm
    steps = 0
    error = None
    if observer is not None:
        if state is None:
            state = engine.settle_batch(drive, steps=0)
        observer(0, state)
    while steps < budget:
        count = min(
            1
            if observer is not None and steps < 12
            else 4
            if observer is not None and steps < 32
            else 32,
            budget - steps,
        )
        state = engine.settle_batch(drive, steps=count, state=state, tolerance=0)
        steps += count
        if observer is not None:
            observer(steps, state)
        error = engine.residual(drive, state)
        if steps >= 32 and np.all(error <= tolerance):
            break
    return state, {
        "steps": steps,
        "residual": float(error.max()),
        "tolerance": tolerance,
        "converged": bool(np.all(error <= tolerance)),
    }
