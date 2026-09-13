"""A reflected board must never become its own validation or test example."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

RUNG = Path(__file__).resolve().parents[1] / "03_connect_four"
sys.path.insert(0, str(RUNG))
spec = importlib.util.spec_from_file_location("connect_four_train", RUNG / "train.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reflection_groups_do_not_cross_split():
    rng = np.random.default_rng(4)
    originals = rng.integers(0, 2, size=(30, 84), dtype=np.uint8)
    boards = np.concatenate([originals, module.mirror_planes(originals), originals[:4]])
    train, test = module.grouped_split(boards, 0.8, 2)
    assert len(train) and len(test)
    keys = lambda rows: {min(a.tobytes(), b.tobytes()) for a, b in zip(rows, module.mirror_planes(rows))}
    assert not keys(boards[train]) & keys(boards[test])
    assert len(train) + len(test) == len(boards)


def test_validation_groups_remain_disjoint_after_augmentation():
    boards = np.eye(84, dtype=np.uint8)
    outer_train, outer_test = module.grouped_split(boards, 0.8, 0)
    augmented = np.concatenate([boards[outer_train], module.mirror_planes(boards[outer_train])])
    fit, validation = module.grouped_split(augmented, 0.8, 1)
    assert set(fit).isdisjoint(validation)
    for i in fit:
        assert not any(np.array_equal(augmented[j], module.mirror_planes(augmented[i:i + 1])[0]) for j in validation)
    assert len(outer_test)
