"""The same life on the NumPy records and on the torch records: the same decisions.

The records backend is a switch in ``RecordsConfig``. These tests run one life twice, once
with the records on NumPy and once with them on a torch device, and compare what a receipt
would report: every committed action, every prediction, every record after every write. The
device here is the torch CPU, so the test runs wherever torch is installed; the CUDA numbers
are in ``NOTES`` of the benchmark, and the deviation a device's reduction order introduces is
measured there as well.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from agent.brain import Agent, AgentConfig
from agent.tests.worlds import RingWorld

torch = pytest.importorskip("torch")

CONFIG = json.loads((Path(__file__).resolve().parent / "config.json").read_text())
TOLERANCE = 1e-9


def config(backend: str) -> AgentConfig:
    base = copy.deepcopy(CONFIG["agent"])
    base["graph"] = dict(
        base["graph"],
        observation=[{"name": "room", "kind": "categorical", "width": 4}],
        prediction=[
            {"name": "next_room", "kind": "categorical", "width": 4, "source": "room"},
            {"name": "reward", "kind": "continuous", "width": 1, "lo": 0.0, "hi": 1.0, "source": "reward"},
        ],
    )
    base["records"] = {
        "granules": 600, "active": 20, "rate": 0.2, "reward_rate": 1.0, "habituation": 1e-3,
        "bias_scale": 0.3, "context": False, "normalize_blocks": True, "block_rate": 0.01,
        "task_sets": False, "pathways": "ports", "fan_in": 0, "backend": backend,
    }
    base["world"] = dict(base["world"], free_steps=40, nudged_steps=40, chunk=8)
    return AgentConfig.from_dict(base)


def live(backend: str, episodes: int = 6, seed: int = 0) -> dict:
    agent = Agent(config(backend), streams=1, seed=seed)
    world = RingWorld(seed=seed)
    actions, probabilities, predictions = [], [], []
    for _ in range(episodes):
        moment = world.reset()
        decision = agent.step(moment)
        while decision is not None:
            actions.append(decision.action)
            probabilities.append(decision.log_probability)
            predictions.append(np.concatenate([np.atleast_1d(v) for v in decision.prediction.values()]))
            moment = world.act(decision.decision_id, decision.action)
            decision = agent.step(moment)
    return {
        "actions": actions,
        "log_probabilities": np.array(probabilities),
        "predictions": np.array(predictions),
        "records": {name: np.asarray(table) for name, table in agent.records.records.items()},
        "mean": np.asarray(agent.records.mean),
        "block_norm": np.asarray(agent.records.block_norm),
        "writes": agent.records.writes,
        "seen": agent.records.seen,
        "hash": agent.state_hash(),
    }


def test_a_life_decides_the_same_on_both_records_backends() -> None:
    host, device = live("cpu"), live("torch")
    assert host["actions"] == device["actions"]
    assert len(host["actions"]) > 50
    assert np.abs(host["log_probabilities"] - device["log_probabilities"]).max() < TOLERANCE
    assert np.abs(host["predictions"] - device["predictions"]).max() < TOLERANCE


def test_the_records_agree_after_every_write() -> None:
    host, device = live("cpu"), live("torch")
    assert host["writes"] == device["writes"] > 0
    assert host["seen"] == device["seen"] > 0
    for name, table in host["records"].items():
        assert np.abs(table - device["records"][name]).max() < TOLERANCE
    assert np.abs(host["mean"] - device["mean"]).max() < TOLERANCE
    assert np.abs(host["block_norm"] - device["block_norm"]).max() < TOLERANCE


def test_the_head_follows_the_agent_backend_unless_the_config_pins_it() -> None:
    base = copy.deepcopy(CONFIG["agent"])
    base["records"] = {"granules": 200, "active": 8, "backend": ""}
    agent = Agent(AgentConfig.from_dict(base), streams=1, seed=0)
    assert agent.records.backend == "cpu" and agent.records.cortex.backend == "cpu"
    pinned = dict(base, records=dict(base["records"], backend="torch"))
    agent = Agent(AgentConfig.from_dict(pinned), streams=1, seed=0)
    assert agent.records.backend == "torch" and agent.records.cortex.backend == "torch"


def test_a_checkpoint_round_trips_through_the_device_records(tmp_path) -> None:
    agent = Agent(config("torch"), streams=1, seed=0)
    world = RingWorld(seed=0)
    moment = world.reset()
    decision = agent.step(moment)
    for _ in range(20):
        if decision is None:
            break
        moment = world.act(decision.decision_id, decision.action)
        decision = agent.step(moment)
    before = {name: np.asarray(t).copy() for name, t in agent.records.records.items()}
    path = agent.save(tmp_path / "life.npz")
    restored = Agent.load(path)
    for name, table in before.items():
        assert np.abs(table - np.asarray(restored.records.records[name])).max() == 0.0
    assert np.abs(np.asarray(agent.records.mean) - np.asarray(restored.records.mean)).max() == 0.0
