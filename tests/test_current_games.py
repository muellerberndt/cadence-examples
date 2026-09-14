"""Causal memory, retention, and episode boundaries of the current game examples."""

import importlib.util
from pathlib import Path

import cadence as cd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_game(name, monkeypatch):
    folder = ROOT / name
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        name + "_current", folder / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_frame_uses_past_activity_and_resets_only_finished_streams(monkeypatch):
    m = load_game("04_pong", monkeypatch)
    policy = m.PatchPolicy(4, 0)
    frames = m.Pong(2, seed=1).frames()
    assert len(policy.wiring.sets["input"]) == 192
    policy.act(frames)
    memory = policy.trace.trace.copy()
    assert memory.any()
    drive = policy.drive(frames)
    np.testing.assert_allclose(drive[:, policy.wiring.sets["afterglow"]], memory)
    policy.reset(2, rows=np.array([True, False]))
    assert not policy.trace.trace[0].any()
    np.testing.assert_array_equal(policy.trace.trace[1], memory[1])


def test_replay_cannot_advance_temporal_memory(monkeypatch):
    m = load_game("04_pong", monkeypatch)
    policy = m.PatchPolicy(4, 0)
    frames = m.Pong(2, seed=3).frames()
    policy.act(frames)
    recorded = policy.last_drive.copy()
    memory = policy.trace.trace.copy()
    policy.update(recorded, np.array([0, 2]), np.array([1.0, -1.0]), m.ETA)
    np.testing.assert_array_equal(policy.trace.trace, memory)
    # Recorded clamps are independent of subsequently changed or reset traces.
    policy.reset(2)
    np.testing.assert_array_equal(policy.drive(recorded), recorded)


def test_residual_memory_corrects_without_erasing_other_keys(monkeypatch):
    m = load_game("02_recall", monkeypatch)
    memory = m.Memory()
    memory.write(0, 1)
    memory.write(1, 2)
    for _ in range(30):
        memory.write(0, 1)
    memory.write(0, 3)
    assert memory.query(0) == 3
    assert memory.query(1) == 2


def test_live_lessons_persist_and_reject_invalid_payloads(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from learning_server import Lessons

    folder = tmp_path / "04_pong"
    folder.mkdir()
    wiring = cd.layered(2, 3, 2, density=1, seed=0)
    learner = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule(dt=1)), wiring.sets["output"]
    )
    learner.save(folder / "learner.npz")
    (folder / "net.json").write_text('{"n": 7}')
    x = np.zeros((4, wiring.n))
    x[:, 0] = 1
    y = np.array([0, 0, 0, 0])
    np.savez(folder / "rehearsal.npz", x=x, y=y)
    live = Lessons("pong", root=tmp_path)
    episode = {
        "drives": x.tolist(),
        "targets": y.tolist(),
        "actions": y.tolist(),
        "rewards": [0, 0, 0, 1],
    }
    result = live.teach(episode)
    assert result["games"] == 1
    assert live.path.exists()
    assert (live.directory / "episode-000001.npz").exists()
    restored = Lessons("pong", root=tmp_path)
    assert restored.games == 1
    np.testing.assert_array_equal(
        restored.learner.engine.dense(), live.learner.engine.dense()
    )
    with pytest.raises(ValueError, match="drives"):
        live.teach({**episode, "drives": [[float("nan")] * wiring.n] * 4})
    with pytest.raises(ValueError, match="targets"):
        live.teach({**episode, "targets": [-1] * 4})
    assert live.games == 1


@pytest.mark.parametrize("name", ["03_connect_four", "04_pong"])
def test_shipped_checkpoint_matches_browser_model_and_rehearsal(name):
    import json

    folder = ROOT / name
    net = json.loads((folder / "net.json").read_text())
    learner = cd.Learner.load(folder / "learner.npz", backend="cpu")
    assert learner.engine.wiring.n == net["n"]
    np.testing.assert_array_equal(learner.output_index, net["sets"]["output"])
    np.testing.assert_allclose(
        learner.engine.dense().ravel(), net["W"], atol=5.001e-6, rtol=0
    )
    np.testing.assert_allclose(learner.engine.bias, net["bias"], atol=5.001e-6, rtol=0)
    with np.load(folder / "rehearsal.npz", allow_pickle=False) as replay:
        assert replay["x"].shape == (len(replay["y"]), net["n"])
        assert len(replay["y"]) >= 1 and np.isfinite(replay["x"]).all()
        assert not replay["x"][:, net["sets"]["output"]].any()
