"""Independent fixed-point checks and adversarial receipt mutations."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "06_interventions/train.py"
spec = importlib.util.spec_from_file_location("interventions_example", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def body_fixture():
    candidates = []
    for width in module.WIDTHS:
        parameters = (module.N * module.N + 3 * module.N + 1) * width
        parameters += (width + 1) * width + (width + 1) * module.N
        candidates.append({"width": width, "parameters": parameters, "steps": 2,
                           "validation": [{"step": 1, "validation_mse": .2},
                                          {"step": 2, "validation_mse": .1}]})
    training = {"candidates": candidates, "selected_width": 128, "selected_step": 2,
                "selected_validation_mse": .1, "seconds": 0, "preparation_seconds": 0,
                "parameters": candidates[0]["parameters"], "training_examples": 8192,
                "batch": 128, "steps_per_candidate": 2}
    rows = []
    for condition in module.CONDITIONS:
        _, w, drive, mask = module.evaluation(0, condition, 2)
        target = module.propagate(w, drive, mask, 128)
        arms = {}
        for arm in module.ARMS:
            timing = {"sequential_seconds": 0}
            if arm.startswith("cadence"):
                timing.update(binding_seconds=0, initialization_seconds=0,
                              median_query_seconds=0, steps=[128, 128], mean_steps=128)
            elif arm != "newton":
                timing["batch_seconds"] = 0
                if arm != "mlp":
                    timing["steps"] = {"one_propagation": 2, "unrolled8": 8, "unrolled32": 32}[arm]
            predicted = target if arm in ("cadence", "cadence_warm", "mlp", "newton") else module.propagate(w, drive, mask, timing["steps"])
            arms[arm] = module.measurements(predicted, target, w, drive, mask, timing)
        rows.append({"condition": condition, "target": target.tolist(), "arms": arms,
                     "data_sha256": hashlib.sha256(w.tobytes() + drive.tobytes() + mask.tobytes()).hexdigest(),
                     "max_incoming_absolute_sum": float(np.abs(w).sum(axis=1).max())})
    return {"task": {"owners": module.N, "conditions": list(module.CONDITIONS),
                     "trials": 2, "seeds": 1, "training_steps": 2,
                     "contraction_bound": module.CONTRACTION},
            "rule": module.RULE.to_dict(), "boundary": module.BOUNDARY,
            "environment": {"device": "cpu", "precision": "float64 for every arm"},
            "runs": [{"seed": 0, "mlp_training": training, "rows": rows}]}


def test_cadence_and_newton_satisfy_same_independent_equation_after_ablation():
    w = module.graphs(91, 2)
    # Reuse both graphs with changed masks so the warm path must discard ablated activity.
    w = np.concatenate([w, w])
    drive = np.random.default_rng(3).normal(size=(4, module.N))
    mask = np.ones_like(drive)
    mask[2:, [0, 2, 4]] = 0
    target, _ = module.newton_arm(w, drive, mask)
    for warm in (False, True):
        predicted, _ = module.cadence_arm(w, drive, mask, warm=warm)
        np.testing.assert_allclose(predicted, target, atol=5e-10, rtol=0)
        assert module.residual(predicted, w, drive, mask).max() < 1e-10
        assert np.array_equal(predicted[mask == 0], np.zeros((mask == 0).sum()))


def test_residual_bound_covers_error_of_early_stopping():
    _, w, drive, mask = module.evaluation(8, "more_ablations", 8)
    exact, _ = module.newton_arm(w, drive, mask)
    partial = module.propagate(w, drive, mask, 4)
    bound = module.residual(partial, w, drive, mask) / (1 - module.CONTRACTION)
    assert np.all(np.max(np.abs(exact - partial), axis=1) <= bound + 1e-12)


def test_complete_fixture_verifies():
    assert module.check(body_fixture()) is None


@pytest.mark.parametrize("change", [
    lambda b: b["runs"].clear(),
    lambda b: b["runs"][0]["rows"].pop(),
    lambda b: b["task"].update(contraction_bound=.99),
    lambda b: b["rule"].update(threshold=1.5),
    lambda b: b["runs"][0]["rows"][0].update(data_sha256="0" * 64),
    lambda b: b["runs"][0]["rows"][0].update(max_incoming_absolute_sum=.5),
    lambda b: b["runs"][0]["rows"][0]["target"][0].__setitem__(0, float("nan")),
    lambda b: b["runs"][0]["rows"][0]["arms"]["mlp"].update(mse=1),
    lambda b: b["runs"][0]["rows"][0]["arms"]["cadence"].update(mean_steps=1),
    lambda b: b["runs"][0]["rows"][0]["arms"]["mlp"].update(batch_seconds=-1),
    lambda b: b["runs"][0]["mlp_training"].update(selected_step=1),
    lambda b: b["runs"][0]["mlp_training"]["candidates"].pop(),
    lambda b: b["environment"].update(precision="float32"),
    lambda b: b["boundary"].update(supplied_dynamics_not_learned_physics=False),
])
def test_receipt_rejects_mutated_evidence(change):
    body = copy.deepcopy(body_fixture())
    change(body)
    assert module.check(body) is not None


def test_receipt_replays_fixed_depth_controls():
    body = body_fixture()
    row = body["runs"][0]["rows"][0]
    _, w, drive, mask = module.evaluation(0, row["condition"], 2)
    wrong = module.propagate(w, drive, mask, 2)
    arm = row["arms"]["unrolled32"]
    arm.update(module.measurements(wrong, np.asarray(row["target"]), w, drive, mask, {}))
    assert "declared recurrence depth" in module.check(body)


def test_residual_recheck_allows_libm_roundoff_but_not_changed_precision():
    body = body_fixture()
    arm = body["runs"][0]["rows"][0]["arms"]["cadence"]
    arm["max_residual"] += 1e-15
    arm["mean_residual"] += 1e-15
    arm["max_error_bound"] += 4e-15
    assert module.check(body) is None
    arm["max_residual"] += 1e-10
    assert "max_residual does not recompute" in module.check(body)
