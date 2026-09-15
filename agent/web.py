"""Export a living agent for a browser page: parameters, ports, learning state, atlas.

The page continues the same life in JavaScript with the same transaction (the engine is
a port of ``brain.py``); a parity harness feeds both the same events. The replay ring is
exported as a bounded sample (the page keeps its own ring from there), which is recorded
in the export. An agent with a records world head exports a ``records`` block (additive,
the format version stays): its config and seed (the page regenerates the expansion from
them), the reading port indices, the habituation mean in float64, every records table at
``records_precision`` (float32 by default: the tables are the bulk of the export) and the
write count; ``records_sidecar`` names a second file with the mean and the tables in
float64 for the parity harness. The pending decision carries its code.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import cadence as cd
import numpy as np

from .brain import COLORS, ROLES, Agent

RECORDS_FORMAT = "cadence-experience-web-records/1"  # the float64 sidecar of the records tables
RECORDS_COLOR = (96, 165, 250)  # the records cortex on a page: not a connectome region, so not in COLORS


def _b64(array: np.ndarray, dtype: str = "<f4") -> dict[str, Any]:
    data = np.ascontiguousarray(np.asarray(array).astype(dtype))
    return {"dtype": data.dtype.str.lstrip("<>|="), "shape": list(data.shape), "b64": base64.b64encode(data.tobytes()).decode("ascii")}


def efficacy_digest(agent: Agent) -> str:
    return hashlib.sha256(np.ascontiguousarray(agent.brain.efficacy).tobytes()).hexdigest()


def records_export(agent: Agent, precision: str = "<f4") -> dict[str, Any] | None:
    """The records head as a snapshot block, or None without one: the config and seed that
    regenerate the expansion, the reading port, the input pathways (index arrays into the
    reading) with their running norms, the habituation mean (float64), every records table in
    field order at ``precision`` ("<f4" or "<f8"), and the write count."""
    if agent.records is None or agent.config.records is None:
        return None
    r = agent.records
    return {
        "config": agent.config.records.to_dict(),
        "seed": int(agent.seed),
        "inputs": int(len(agent.reading)),
        "reading": [int(i) for i in agent.reading],
        "fields": [f.name for f in r.fields],
        "blocks": [[int(i) for i in b] for b in getattr(r, "blocks", [])],
        "block_norm": _b64(getattr(r, "block_norm", np.ones(0)), "<f8"),
        "mean": _b64(r.mean, "<f8"),
        "tables": {f.name: _b64(r.records[f.name], precision) for f in r.fields},
        "precision": "float64" if precision == "<f8" else "float32",
        "writes": int(r.writes),
        "seen": int(r.seen),
        "color": list(RECORDS_COLOR),
    }


def records_sidecar(agent: Agent, path: str | Path) -> Path | None:
    """The mean and every records table in float64, with the snapshot's efficacy digest so a
    harness can match the two files; None (nothing written) without a records head."""
    block = records_export(agent, "<f8")
    if block is None:
        return None
    payload = {"format": RECORDS_FORMAT, "snapshot_efficacy_sha256": efficacy_digest(agent), "parameter_version": int(agent.parameter_version), "writes": block["writes"], "seen": block["seen"], "mean": block["mean"], "block_norm": block["block_norm"], "tables": block["tables"]}
    path = Path(path)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path


def export(agent: Agent, path: str | Path, *, ring_sample: int = 4096, atlas_seed: int = 0, extra: dict[str, Any] | None = None, records_precision: str = "<f4", records_sidecar_path: str | Path | None = None) -> Path:
    if agent.streams != 1:
        raise ValueError("the page runs one stream")
    c = agent.connectome
    p = agent.ports
    model = agent.brain.neuron_model
    atlas = cd.atlas_of(agent.brain, roles=ROLES, seed=atlas_seed)
    atlas_dict = atlas.to_dict()
    for region in atlas_dict["regions"]:
        if region["name"] in COLORS:
            region["color"] = list(COLORS[region["name"]])
    rng = np.random.default_rng(atlas_seed)
    ring = agent.ring
    if len(ring) > ring_sample:
        keep = sorted(rng.choice(len(ring), size=ring_sample, replace=False).tolist())
        ring = [ring[k] for k in keep]
    fields = agent.config.graph.prediction
    ring_targets = []
    for _, _, _, targets in ring:
        entry = {}
        for f in fields:
            if f.name in targets:
                entry[f.name] = {"target": np.asarray(targets[f.name][0]).tolist(), "known": np.asarray(targets[f.name][1]).astype(int).tolist()}
        ring_targets.append(entry)
    payload: dict[str, Any] = {
        "format": "cadence-experience-web/1",
        "config": agent.config.to_dict(),
        "seed": agent.seed,
        "neuron": {"dt": model.dt, "slope": model.slope, "threshold": model.threshold, "leak": model.leak, "rest": model.rest_emission, "amplitude": model.stimulus_amplitude, "gain": model.gain},
        "connectome": {"n": int(c.n), "synapses": int(c.synapses), "pre": _b64(c.pre, "<u4"), "post": _b64(c.post, "<u4"), "count": _b64(c.count, "<f8")},
        "efficacy": _b64(agent.brain.efficacy, "<f8"),
        "bias": _b64(agent.brain.bias, "<f8"),
        "log_gain": _b64(agent.brain.log_gain, "<f8"),
        "ports": {
            "observation": p.observation.tolist(), "goal": p.goal.tolist(), "action": p.action.tolist(), "perception": p.perception.tolist(),
            "workspace": p.workspace.tolist(), "context": p.context.tolist(), "recall": p.recall.tolist(), "dynamics": p.dynamics.tolist(),
            "prediction": p.prediction.tolist(), "motor": p.motor.tolist(), "intention": p.intention.tolist(),
            "observation_fields": {k: v.tolist() for k, v in p.observation_fields.items()},
            "observation_flags": {k: v.tolist() for k, v in p.observation_flags.items()},
            "prediction_fields": {k: v.tolist() for k, v in p.prediction_fields.items()},
            "action_slots": [s.tolist() for s in p.action_slots], "motor_slots": [s.tolist() for s in p.motor_slots],
            "sources": p.sources.tolist(),
        },
        "plasticity": {
            "world_edges": np.flatnonzero(agent.world_edges).tolist(), "motor_edges": np.flatnonzero(agent.motor_edges).tolist(),
            "world_neurons": np.flatnonzero(agent.world.plastic_neurons).tolist() if agent.world is not None else [],
            "motor_neurons": np.flatnonzero(agent.motor.plastic_neurons).tolist(),
            "reverse": _b64(agent.motor.reverse, "<i4"),
            "synapse_rate": None if agent.motor.synapse_rate is None else _b64(agent.motor.synapse_rate, "<f8"),
        },
        "world_state": None if agent.world is None else {
            "velocity": _b64(agent.world.velocity, "<f8"), "velocity_bias": _b64(agent.world.velocity_bias, "<f8"),
            "second_moment": _b64(agent.world.second_moment, "<f8"), "second_moment_bias": _b64(agent.world.second_moment_bias, "<f8"),
            "contrast_updates": int(agent.world.contrast_updates), "updates": int(agent.world.updates),
        },
        "motor_state": {
            "velocity": _b64(agent.motor.velocity, "<f8"), "velocity_bias": _b64(agent.motor.velocity_bias, "<f8"),
            "contrast_updates": int(agent.motor.contrast_updates), "updates": int(agent.motor.updates),
        },
        "critic": {"w": agent.w_critic.tolist(), "b": float(agent.b_critic)},
        "eligibility": {"edges": _b64(agent.elig[0], "<f8"), "bias": _b64(agent.elig_bias[0], "<f8"), "critic": agent.trace_critic[0].tolist()},
        "context": {"trace": agent.context.trace[0].tolist() if len(agent.context.trace) else [], "last": agent.context.last[0].tolist() if len(agent.context.last) else [], "cold": bool(agent.context.cold[0]) if len(agent.context.cold) else True},
        "rng": {"action": int(agent.rng[0].state), "replay": int(agent.replay_rng.state)},
        "cursors": {"last_event": int(agent.last_event[0]), "episode": int(agent.episode[0]), "next_decision_id": int(agent.next_decision_id), "parameter_version": int(agent.parameter_version), "context_version": int(agent.context_version)},
        "warm": None if agent.warm is None else {"v": _b64(agent.warm.v[0], "<f8"), "a": _b64(agent.warm.adaptation[0], "<f8")},
        "pending": None if agent.pending[0] is None else {
            "decision": agent.pending[0].decision.to_dict(), "value": float(agent.pending[0].value),
            "drive": _b64(agent.pending[0].drive, "<f8"), "state_v": _b64(agent.pending[0].state_v, "<f8"), "state_a": _b64(agent.pending[0].state_a, "<f8"),
            "workspace": _b64(agent.pending[0].workspace, "<f8"),
            "score_edges": None if agent.pending[0].score_edges is None else _b64(agent.pending[0].score_edges, "<f8"),
            "score_bias": None if agent.pending[0].score_bias is None else _b64(agent.pending[0].score_bias, "<f8"),
            "code": None if agent.pending[0].code is None else _b64(agent.pending[0].code, "<f8"),
        },
        "ring": {
            "stored": len(agent.ring), "exported": len(ring), "at": int(agent.ring_at), "precision": "float64 drives and warm potentials",
            "drive": _b64(np.stack([t[0] for t in ring]), "<f8") if ring else None,
            "v": _b64(np.stack([t[1] for t in ring]), "<f8") if ring else None,
            "a": _b64(np.stack([t[2] for t in ring]), "<f8") if ring else None,
            "targets": ring_targets,
        },
        "records": records_export(agent, records_precision),
        "ledger": agent.ledger.to_dict(),
        "atlas": atlas_dict,
        "extra": extra or {},
    }
    if records_sidecar_path is not None and payload["records"] is not None:
        sidecar = records_sidecar(agent, records_sidecar_path)
        payload["records"]["sidecar"] = None if sidecar is None else sidecar.name
    path = Path(path)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path
