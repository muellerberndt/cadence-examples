"""Pong keeps its local update, device execution and exported artifacts consistent."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def pong_train(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "04_pong"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("pong_train_test", folder / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reward_update_matches_explicit_adam_contrasts(pong_train, monkeypatch):
    m = pong_train
    monkeypatch.setattr(m, "BATCH", 4)
    policy, reference = m.PatchPolicy(4, 9), m.PatchPolicy(4, 9)
    learner = reference.learner
    rng = np.random.default_rng(13)
    frames = rng.random((8, m.INPUTS))
    actions = rng.integers(0, 3, 8)
    advantage = rng.normal(size=8)
    moments = [np.zeros(reference.wiring.edges), np.zeros(reference.wiring.n)]
    seconds = [v.copy() for v in moments]
    count = 0
    for eta in (m.ETA, m.ETA / 2):
        policy.update(frames, actions, advantage, eta)
        order = reference.rng.permutation(len(actions))
        for start in range(0, len(order), m.BATCH):
            idx = order[start : start + m.BATCH]
            drive = reference.drive(frames[idx])
            target = learner.targets(actions[idx])
            free = learner.free(drive)
            plus = learner.nudged(drive, free, target, weight=advantage[idx])
            minus = learner.nudged(drive, free, target, sign=-1, weight=advantage[idx])
            count += 1
            changes = []
            for i, contrast in enumerate(learner.contrast(free, plus, minus)):
                moments[i] = m.MOMENTUM * moments[i] + (1 - m.MOMENTUM) * contrast
                seconds[i] = m.RMS * seconds[i] + (1 - m.RMS) * contrast**2
                changes.append(eta * (moments[i] / (1 - m.MOMENTUM**count)) /
                               (np.sqrt(seconds[i] / (1 - m.RMS**count)) + m.FLOOR))
            learner.apply(*changes)
        np.testing.assert_allclose(policy.learner.engine.dense(), learner.engine.dense(), atol=1e-12)
        np.testing.assert_allclose(policy.learner.engine.bias, learner.engine.bias, atol=1e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_both_learners_update_on_selected_device(pong_train, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    m = pong_train
    frames = m.Pong(8, seed=4).observation()
    actions = np.arange(8) % 3
    for cls in (m.PatchPolicy, m.TorchPolicy):
        policy = cls(4, 0, device=device)
        before = policy.probabilities(frames)
        policy.update(frames, actions, np.linspace(-1, 1, 8), m.ETA)
        after = policy.probabilities(frames)
        assert np.isfinite(after).all()
        np.testing.assert_allclose(after.sum(axis=1), 1, atol=1e-6)
        assert np.max(np.abs(before - after)) > 1e-6
        if cls is m.TorchPolicy:
            assert next(policy.net.parameters()).device.type == device
        elif device != "cpu":
            assert str(policy.learner.engine._torch.device).split(":")[0] == device


def test_training_outputs_follow_receipt_directory(pong_train, tmp_path, monkeypatch):
    m = pong_train
    for name, value in {"HIDDEN": 4, "ENVS": 4, "HORIZON": 4, "BATCH": 8,
                        "IMITATION_STEPS": 2, "IMITATION_EPOCHS": 1, "EVAL_POINTS": 4}.items():
        monkeypatch.setattr(m, name, value)
    # A sentinel makes accidental writes into the source directory observable.
    source = tmp_path / "source"
    source.mkdir()
    sentinel = source / "net.json"
    sentinel.write_text("preserve shipped weights")
    monkeypatch.setattr(m, "HERE", source)
    output = tmp_path / "candidate" / "receipt.json"
    body = m.run(0, output, 1)
    assert sentinel.read_text() == "preserve shipped weights"
    assert output.exists()
    assert (output.parent / "net.json").exists()
    net = json.loads((output.parent / "net_imitation.json").read_text())
    assert net["temperature"] == 0.1
    assert body["final"]["points"] == sum(body["final"][k] for k in ("wins", "losses", "draws"))
    assert m.cd.Receipt.verify(output, sources=m.SOURCES, check=m.check)[0]


def test_cli_rejects_empty_training_run(pong_train, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--iterations", "0"])
    with pytest.raises(SystemExit) as error:
        pong_train.main()
    assert error.value.code == 2
