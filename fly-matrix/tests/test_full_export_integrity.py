"""Adversarial checks of full-export verification, using a tiny independent graph."""
from __future__ import annotations

import base64
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from cadence import Brain, Connectome, NeuronModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from brain_payload import payload_of
from verify_full_export import (CLASS_LOG_GAIN, DECISION_SENSES, OTHER_LEVEL, OUTPUTS,
                                READOUT_TARGETS, main, verify_payload)


def encode(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def fixture(overflow=True):
    C = Connectome.from_synapses(6, pre=[0, 1, 3], post=[2, 3, 4],
                                count=[8, 70000 if overflow else 7, 1], sign=[1, -1, 1],
                                populations={"kc": (0, 1), "mbon": (2, 3), "ln": (1,),
                                             OUTPUTS[0]: (2,), OUTPUTS[1]: (3,),
                                             "weak_only": (4,), "isolated": (5,)})
    log_gain = np.array([0., -3, 0, 0, 0, 0])
    efficacy = np.array(C.sign)
    efficacy[:2] = C.sign[:2] * C.count[:2].mean() / C.count[:2]
    brain = Brain(C, NeuronModel(), log_gain=log_gain, efficacy=efficacy)
    manifest = {"fixture_sha256": "tiny-test-fixture", "neurons": 6,
                "synapse_classes": 3, "synapses": int(C.count.sum())}
    selection = {"mode": "full_retained", "neurons": 6, "classes": 3,
                 "all_retained_neurons": True, "all_retained_edges": True}
    report = {
        "outputs": dict(zip(OUTPUTS, (2, 3))), "naive_seam": True,
        "readout_targets": dict(zip(OUTPUTS, READOUT_TARGETS)),
        "readout_bias": dict.fromkeys(OUTPUTS, 0.), "decision_senses": DECISION_SENSES,
        "other_level": OTHER_LEVEL,
        "seam": {"classes": 2, "synapses": float(C.count[:2].sum()), "coverage_pre": 1.,
                 "median_count": float(np.median(C.count[:2])), "classes_onto_outputs": dict.fromkeys(OUTPUTS, 1)},
    }
    payload = payload_of(brain, members=np.arange(6), extra={
        "manifest": manifest["fixture_sha256"], "whole": {"neurons": 6, "edges": 3},
        "selection": selection, "recruitment": None,
        "class_log_gain": CLASS_LOG_GAIN, "lessons_setup": report})
    lessons = {"config": {}, "setup": copy.deepcopy(report)}
    setup = {"tool": "tools/export_web.py", "fixture": manifest["fixture_sha256"],
             "gain": .02, "class_log_gain": CLASS_LOG_GAIN, "selection": selection,
             "recruitment": None, **copy.deepcopy(report)}
    return payload, C, manifest, lessons, setup


@pytest.mark.parametrize("overflow", [False, True])
def test_valid_complete_export_and_large_counts(overflow):
    result = verify_payload(*fixture(overflow))
    assert result["isolated_neurons_retained"] == 1
    assert result["edge_classes_below_five_contacts"] == 1
    assert result["counts_exceeding_uint16"] == int(overflow)
    assert result["gain_pre_present"] == overflow
    assert max(result["maximum_absolute_errors"].values()) <= 1e-12


def corrupt(payload, case):
    A = payload["arrays"]
    if case == "isolated_member":
        A["members"] = encode(np.array([0, 1, 2, 3, 4, 4], dtype="<i4"))
    elif case == "weak_edge_missing":
        A["row_ptr"] = encode(np.array([0, 0, 0, 1, 2, 2, 2], dtype="<i4"))
    elif case == "edge_rewired":
        A["pre"] = encode(np.array([0, 1, 2], dtype="<i4"))
    elif case == "count":
        A["count"] = encode(np.array([8, 65535, 2], dtype="<u2"))
    elif case == "sign":
        A["sign"] = encode(np.array([1, 1, 1], dtype="i1"))
    elif case == "population_same_size_wrong_cell":
        payload["populations"]["isolated"] = [4]
    elif case == "overflow_factor_missing":
        del A["gain_pre"]
    elif case == "overflow_factor_truncated":
        A["gain_pre"] = encode(.02 * np.array([8, 65535, 1]) * np.exp([0, -3, 0]))
    elif case == "weight":
        weights = np.frombuffer(base64.b64decode(A["weight"]), dtype="<f8").copy()
        weights[2] = 0
        A["weight"] = encode(weights)
    elif case == "unreported_bias":
        A["bias"] = encode(np.array([1., 0, 0, 0, 0, 0]))
    elif case == "ambiguous_efficacy":
        A["efficacy_index"] = encode(np.array([0], dtype="<i4"))
        A["efficacy_value"] = encode(np.array([1.]))
    elif case == "nonfinite_factor":
        A["gain_pre"] = encode(np.array([1., np.nan, 1.]))
    elif case == "malformed_base64":
        A["members"] += "!"
    elif case == "selection":
        payload["selection"]["mode"] = "recruited_subset"
    else:
        raise AssertionError(case)


@pytest.mark.parametrize("case", [
    "isolated_member", "weak_edge_missing", "edge_rewired", "count", "sign",
    "population_same_size_wrong_cell", "overflow_factor_missing", "overflow_factor_truncated",
    "weight", "unreported_bias", "ambiguous_efficacy", "nonfinite_factor", "malformed_base64", "selection",
])
def test_corrupt_exports_fail(case):
    args = fixture()
    corrupt(args[0], case)
    with pytest.raises(ValueError):
        verify_payload(*args)


def test_lesson_record_drift_fails():
    args = fixture()
    args[3]["setup"]["readout_bias"][OUTPUTS[0]] = .1
    with pytest.raises(ValueError, match="lessons file"):
        verify_payload(*args)


def test_existing_receipt_is_not_overwritten(tmp_path):
    receipt = tmp_path / "already.json"
    receipt.write_bytes(b"historical receipt")
    with pytest.raises(SystemExit):
        main(["--receipt", str(receipt)])
    assert receipt.read_bytes() == b"historical receipt"
