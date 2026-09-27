#!/usr/bin/env python3
"""Independently compare the full browser payload with the pinned retained fixture.

This does not fetch data, regenerate the exporter, rerun calibration, or establish
settlement/behaviour. It verifies every retained member, directed edge, count,
sign and population; the declared class gains and initial naive seam; and the
effective weights that the browser reconstructs. Calibration biases are checked
against both exported setup records, not independently recalibrated.

Run with the Cadence Python environment. --receipt creates a small source-bound
receipt exclusively; existing receipts are never overwritten. Historical
brain.json / lessons.json and the frozen 60,000-neuron assays are not consumed.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cadence  # noqa: E402
from cadence import NeuronModel  # noqa: E402
from fruitfly.banc import DEFAULT_ROOT, FIXTURE_NAME, MANIFEST_PATH  # noqa: E402
from fruitfly.brain import CLASS_LOG_GAIN, GAIN, load_fly  # noqa: E402
from fruitfly.lessons import DECISION_SENSES, OTHER_LEVEL, OUTPUTS, READOUT_TARGETS  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite JSON number {value} in {path}")

    result = json.loads(path.read_text(), object_pairs_hook=object_pairs, parse_constant=nonfinite)
    require(isinstance(result, dict), f"expected JSON object in {path}")
    return result


def decode(arrays: dict, name: str, dtype: str, size: int | None = None) -> np.ndarray:
    require(isinstance(arrays.get(name), str), f"missing or malformed array {name}")
    try:
        raw = base64.b64decode(arrays[name], validate=True)
        values = np.frombuffer(raw, dtype=np.dtype(dtype))
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"malformed {name}: {exc}") from exc
    if size is not None:
        require(values.size == size, f"{name}: length {values.size}, expected {size}")
    require(bool(np.isfinite(values).all()), f"{name}: nonfinite value")
    return values


def same_array(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    require(actual.shape == expected.shape, f"{name}: shape mismatch")
    if not np.array_equal(actual, expected):
        index = int(np.flatnonzero(actual != expected)[0])
        raise ValueError(f"{name}: first mismatch at {index}: {actual[index]} != {expected[index]}")


def close_array(actual: np.ndarray, expected: np.ndarray, name: str) -> float:
    require(actual.shape == expected.shape, f"{name}: shape mismatch")
    error = float(np.max(np.abs(actual - expected), initial=0.0))
    require(bool(np.allclose(actual, expected, rtol=0.0, atol=1e-12)),
            f"{name}: absolute error {error} exceeds 1e-12")
    return error


def integer(value: Any, expected: int, name: str) -> None:
    require(type(value) is int and value == expected, f"{name}: expected integer {expected}, got {value!r}")


def verify_payload(payload: dict, connectome: Any, manifest: dict, lessons: dict,
                   setup_receipt: dict, *, expected_gain: float = GAIN) -> dict:
    """Pure comparison of decoded export and reference arrays; never calls exporter."""
    C = connectome
    n, edges, contacts = C.n, C.synapses, int(C.count.sum())
    require(payload.get("format") == "cadence.brain-payload/1", "unknown payload format")
    require(sys.byteorder == "little", "this export format uses the producer's little-endian arrays")
    for key, expected in (("n", n), ("edges", edges), ("synapses", contacts)):
        integer(payload.get(key), expected, key)
    for key, expected in (("neurons", n), ("synapse_classes", edges), ("synapses", contacts)):
        integer(manifest.get(key), expected, f"fixture manifest {key}")
    require(payload.get("manifest") == manifest["fixture_sha256"], "payload fixture digest differs")
    require(payload.get("whole") == {"neurons": n, "edges": edges}, "whole-graph metadata differs")
    selection = {"mode": "full_retained", "neurons": n, "classes": edges,
                 "all_retained_neurons": True, "all_retained_edges": True}
    require(payload.get("selection") == selection, "full-retained selection metadata differs")
    require("recruitment" in payload and payload["recruitment"] is None, "full export must not recruit")
    require(payload.get("library", {}).get("version") == cadence.__version__, "Cadence version differs")
    require(payload.get("model") == NeuronModel(gain=expected_gain).to_dict(), "neuron model differs")
    require(payload.get("class_log_gain") == CLASS_LOG_GAIN, "class-gain dictionary differs")

    A = payload.get("arrays")
    require(isinstance(A, dict), "missing arrays object")
    allowed = {"row_ptr", "pre", "weight", "count", "sign", "sign_dtype", "bias", "members",
               "gain_pre", "log_gain", "efficacy", "efficacy_index", "efficacy_value"}
    require(not (set(A) - allowed), f"unrecognized arrays: {set(A) - allowed}")
    same_array(decode(A, "members", "<i4", n), np.arange(n), "retained members")
    row_ptr = decode(A, "row_ptr", "<i4", n + 1)
    expected_rows = np.concatenate(([0], np.cumsum(np.bincount(C.post, minlength=n))))
    same_array(row_ptr, expected_rows, "CSR row pointers / receiving endpoints")
    pre = decode(A, "pre", "<i4", edges)
    same_array(pre, C.pre, "sending endpoints")
    count = decode(A, "count", "<u2", edges)
    same_array(count, np.minimum(C.count, 65535), "stored contact counts")
    sign_dtype = A.get("sign_dtype")
    require(sign_dtype in ("int8", "float64"), "unknown sign dtype")
    sign = decode(A, "sign", "i1" if sign_dtype == "int8" else "<f8", edges)
    same_array(sign, C.sign, "transmitter signs")
    populations = payload.get("populations")
    require(isinstance(populations, dict), "missing populations")
    require(set(populations) == set(C.populations), "population names differ")
    for name, expected in C.populations.items():
        actual = populations[name]
        require(isinstance(actual, list) and all(type(i) is int for i in actual), f"{name}: malformed population")
        same_array(np.asarray(actual, dtype=np.int64), np.asarray(expected, dtype=np.int64), f"population {name}")

    # Recompute the declared initial transformation directly from anatomical data.
    # This does not use payload_of, naive_efficacy or log_gain_for.
    log_gain = np.zeros(n)
    for name, value in CLASS_LOG_GAIN.items():
        log_gain[list(C.populations.get(name, ()))] = value
    if "log_gain" in A:
        same_array(decode(A, "log_gain", "<f8", n), log_gain, "per-neuron log gain")
    expected_factor = expected_gain * C.count * np.exp(log_gain[C.pre])
    overflow = int(np.count_nonzero(C.count > 65535))
    if "gain_pre" in A:
        factor = decode(A, "gain_pre", "<f8", edges)
    else:
        require(overflow == 0, "contact count exceeds 65535 but gain_pre is missing")
        require("log_gain" in A or not np.any(log_gain), "nonzero class gains missing from payload")
        browser_log_gain = decode(A, "log_gain", "<f8", n) if "log_gain" in A else np.zeros(n)
        factor = expected_gain * count * np.exp(browser_log_gain[pre])
    factor_error = close_array(factor, expected_factor, "effective presynaptic factors")

    has_sparse = "efficacy_index" in A or "efficacy_value" in A
    require(not (has_sparse and "efficacy" in A), "ambiguous sparse and dense efficacy")
    if "efficacy" in A:
        efficacy = decode(A, "efficacy", "<f8", edges)
    else:
        efficacy = np.array(sign, dtype=float)
        if has_sparse:
            indices = decode(A, "efficacy_index", "<i4")
            values = decode(A, "efficacy_value", "<f8", len(indices))
            require(bool(np.all((indices >= 0) & (indices < edges))), "efficacy index out of bounds")
            require(bool(np.all(np.diff(indices) > 0)), "efficacy indices must be unique and sorted")
            efficacy[indices] = values
    is_kc, is_mbon = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    is_kc[list(C.populations["kc"])] = True
    is_mbon[list(C.populations["mbon"])] = True
    plastic = is_kc[C.pre] & is_mbon[C.post]
    expected_efficacy = np.array(C.sign, dtype=float)
    if plastic.any():
        expected_efficacy[plastic] *= C.count[plastic].mean() / C.count[plastic]
    efficacy_error = close_array(efficacy, expected_efficacy, "initial naive-seam efficacy")
    weight = decode(A, "weight", "<f8", edges)
    weight_error = close_array(weight, expected_factor * expected_efficacy, "fixture-derived effective weights")
    reconstruction_error = close_array(weight, factor * efficacy, "browser weight reconstruction")

    report = payload.get("lessons_setup")
    require(isinstance(report, dict), "missing lesson setup")
    require(lessons == {"config": {}, "setup": report}, "lessons file and embedded setup differ")
    require(setup_receipt == {"tool": "tools/export_web.py", "fixture": manifest["fixture_sha256"],
                            "gain": expected_gain, "class_log_gain": CLASS_LOG_GAIN,
                            "selection": selection, "recruitment": None, **report},
            "setup receipt and embedded setup differ")
    outputs = {}
    for name in OUTPUTS:
        require(len(C.populations.get(name, ())) == 1, f"{name}: expected one output cell")
        outputs[name] = int(C.populations[name][0])
    require(report.get("outputs") == outputs, "lesson output identities differ")
    require(report.get("naive_seam") is True, "initial seam is not declared naive")
    require(report.get("readout_targets") == dict(zip(OUTPUTS, READOUT_TARGETS)), "readout targets differ")
    require(report.get("decision_senses") == DECISION_SENSES and report.get("other_level") == OTHER_LEVEL,
            "declared calibration inputs differ")
    expected_seam = {
        "classes": int(plastic.sum()), "synapses": float(C.count[plastic].sum()),
        "coverage_pre": len(np.unique(C.pre[plastic])) / len(C.populations["kc"]),
        "median_count": float(np.median(C.count[plastic])) if plastic.any() else 0.0,
        "classes_onto_outputs": {name: int(np.count_nonzero(plastic & (C.post == i))) for name, i in outputs.items()},
    }
    require(report.get("seam") == expected_seam, "plastic seam report differs from fixture")
    biases = report.get("readout_bias")
    require(isinstance(biases, dict) and set(biases) == set(outputs), "readout bias keys differ")
    expected_bias = np.zeros(n)
    for name, i in outputs.items():
        require(type(biases[name]) in (int, float) and np.isfinite(biases[name]), "invalid calibrated bias")
        expected_bias[i] = biases[name]
    same_array(decode(A, "bias", "<f8", n), expected_bias, "calibrated bias / zero elsewhere")
    return {
        "neurons": n, "edge_classes": edges, "represented_contacts": contacts,
        "population_count": len(populations), "members_exact": True, "all_directed_edges_exact": True,
        "stored_counts_and_signs_exact": True, "populations_exact": True,
        "isolated_neurons_retained": int(np.count_nonzero(np.bincount(C.pre, minlength=n) + np.bincount(C.post, minlength=n) == 0)),
        "edge_classes_below_five_contacts": int(np.count_nonzero(C.count < 5)),
        "edge_classes_below_six_contacts": int(np.count_nonzero(C.count < 6)),
        "counts_exceeding_uint16": overflow, "largest_contact_count": int(C.count.max(initial=0)),
        "gain_pre_present": "gain_pre" in A, "model": payload["model"], "class_log_gain": CLASS_LOG_GAIN,
        "initial_plastic_edge_classes": int(plastic.sum()), "output_indices": outputs,
        "maximum_absolute_errors": {"factor": factor_error, "efficacy": efficacy_error,
                                    "weight_from_fixture": weight_error, "weight_browser_reconstruction": reconstruction_error},
        "absolute_tolerance": 1e-12, "relative_tolerance": 0,
        "setup_records_agree": True,
        "limitations": ["retained fixture only; excluded raw-release objects and subthreshold non-seam edges remain excluded",
                        "class gains, naive seam and calibrated output biases are supplied model transformations",
                        "bias calibration and setup/preflight outcomes are bound and compared, not rerun",
                        "no dynamics parity, convergence, biological fidelity, learning or behaviour claim"],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", type=Path, default=ROOT / "web/data/brain_full.json")
    ap.add_argument("--lessons", type=Path, default=ROOT / "web/data/lessons_full.json")
    ap.add_argument("--setup", type=Path, default=ROOT / "receipts/lessons_setup_full.json")
    ap.add_argument("--fixture-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--gain", type=float, default=GAIN, help="expected global gain, declared explicitly if different")
    ap.add_argument("--receipt", type=Path, help="create a new source-bound receipt; refuses overwrite")
    args = ap.parse_args(argv)
    if args.receipt and args.receipt.exists():
        ap.error(f"receipt already exists: {args.receipt}")
    fixture = args.fixture_root / FIXTURE_NAME
    require(fixture.is_file(), f"pinned fixture missing: {fixture}; this verifier never fetches or builds it")
    library_root = Path(cadence.__file__).resolve().parent
    files = {"payload": args.payload, "lessons": args.lessons, "setup": args.setup,
             "fixture": fixture, "manifest": MANIFEST_PATH, "verifier": Path(__file__).resolve()}
    for relative in ("tools/export_web.py", "tools/brain_payload.py", "fruitfly/banc.py", "fruitfly/brain.py", "fruitfly/lessons.py"):
        files[relative] = ROOT / relative
    for name in ("__init__.py", "connectome.py", "neuron.py", "brain.py", "learning.py"):
        files[f"cadence/{name}"] = library_root / name
    hashes = {label: digest(path) for label, path in files.items()}
    manifest = read_json(MANIFEST_PATH)
    require(hashes["fixture"] == manifest["fixture_sha256"], "pinned fixture hash mismatch")
    fly = load_fly(args.fixture_root)
    result = verify_payload(read_json(args.payload), fly.connectome, manifest,
                            read_json(args.lessons), read_json(args.setup), expected_gain=args.gain)
    require(hashes == {label: digest(path) for label, path in files.items()}, "input/source changed during verification")
    body = {"schema": "cadence.full-export-integrity/1", "verified": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "cadence": cadence.__version__, "byteorder": sys.byteorder},
            "files": {label: {"path": str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path.resolve()),
                              "bytes": path.stat().st_size, "sha256": hashes[label]} for label, path in files.items()},
            "result": result}
    body_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    receipt = {"body": body, "sha256": body_hash}
    if args.receipt:
        with args.receipt.open("x") as output:
            output.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"verified": True, "receipt": str(args.receipt) if args.receipt else None,
                      "sha256": body_hash, "result": result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
