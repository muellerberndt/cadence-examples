"""Independent lookup checks and receipt mutation tests for the public example."""
import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "05_memory/train.py"
spec = importlib.util.spec_from_file_location("memory_example", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_readers_use_latest_values_and_ignore_correlation():
    ids, vals, truth = module.episodes(19, 6, 128)
    for correlation in (0.0, 0.5, 0.99):
        bank = module.keys(correlation)
        for reader in (module.dictionary_read, module.exact_attention_read):
            assert np.array_equal(reader(ids, vals, bank).argmax(-1), truth)


def test_explicit_revision_replaces_only_its_key():
    ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 2, 2]])
    vals = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 4, 6]])
    expected = np.array([[0, 1, 6, 3, 4, 5, 6, 7]])
    for reader in (module.dictionary_read, module.exact_attention_read):
        assert np.array_equal(reader(ids, vals, module.keys(0.5)).argmax(-1), expected)


def test_delta_rewrites_orthogonal_keys_without_forgetting_other_keys():
    ids, vals, truth = module.episodes(13, 4, 128)
    result, _ = module.memory_read(ids, vals, module.keys(0), "delta")
    assert np.array_equal(result.argmax(-1), truth)
    additive, _ = module.memory_read(ids, vals, module.keys(0), "hebb")
    assert np.any(additive.argmax(-1) != truth)


def complete_body():
    import hashlib
    rows = []
    for mode, correlation, length in module.CASES:
        ids, vals, truth = module.episodes(30000 + length, 2, length, balanced=mode == "balanced_first_pass")
        result = module.dictionary_read(ids, vals, module.keys(correlation))
        arms = {arm: module.readings(result, truth, ids, 0) for arm in module.ARMS}
        rows.append({"mode": mode, "length": length, "key_correlation": correlation,
                     "length_extrapolation": length > max(module.TRAIN_LENGTHS),
                         "key_correlation_extrapolation": correlation not in module.CORRELATIONS,
                     "episode_sha256": hashlib.sha256(ids.tobytes() + vals.tobytes()).hexdigest(),
                     "truth": truth.tolist(), "latest_key": ids[:, -1].tolist(), "arms": arms})
    return {"task": {"seeds": 1, "trials_per_condition": 2,
                     "lengths": list(module.LENGTHS), "correlations": list(module.CORRELATIONS)},
            "runs": [{"seed": 0, "rows": rows}]}


def test_receipt_check_rejects_wrong_accuracy():
    body = complete_body()
    assert module.check(body) is None
    body["runs"][0]["rows"][0]["arms"]["delta"]["accuracy"] = 0
    assert "accuracy differs" in module.check(body)


def test_receipt_check_rejects_omitted_outcomes():
    body = complete_body()
    body["runs"].clear()
    assert "seed schedule differs" in module.check(body)
    body = complete_body()
    body["runs"][0]["rows"].pop()
    assert "condition schedule differs" in module.check(body)


def test_receipt_check_rejects_rewritten_truth_or_episode_digest():
    body = complete_body()
    row = body["runs"][0]["rows"][0]
    row["truth"][0][0] = (row["truth"][0][0] + 1) % module.VALUES
    assert "seeded episodes" in module.check(body)
    body = complete_body()
    body["runs"][0]["rows"][0]["episode_sha256"] = "0" * 64
    assert "episode digest differs" in module.check(body)
