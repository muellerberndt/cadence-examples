"""Full browser export preserves retained topology and historical subset artifacts."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadence import Connectome

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import export_web as exporter  # noqa: E402


def retained_graph():
    # 0..3 are named seeds; 4 has only a below-recruitment-threshold edge;
    # 5 is disconnected. A larger recruitment budget still cannot include both.
    populations = {"haltere": (0,), "kc": (0, 1), "mbon": (2, 3), "ln": (1,),
                   "mbon:MBON11:right": (2,), "mbon:MBON05:left": (3,),
                   "weak_only": (4,), "isolated": (5,)}
    for name in ("haltere:left", "haltere:right", "ocelli", "wing_sense:right",
                 "vis:LC4", "vis:LPLC2", "orn:decaying_fruit:left",
                 "orn:decaying_fruit:right", "grn:sugar:labellum"):
        populations[name] = (0,)
    return Connectome.from_synapses(6, pre=[0, 1, 3], post=[2, 3, 4],
                                   count=[8, 7, 1], sign=[1, -1, 1], populations=populations)


def decode(payload, name, dtype):
    return np.frombuffer(base64.b64decode(payload["arrays"][name]), dtype=dtype)


def test_full_selection_bypasses_recruitment_and_preserves_weak_and_isolated_cells(monkeypatch):
    whole = retained_graph()
    # Establish that the fixture really distinguishes recruitment from full export.
    subset = exporter.recruit(whole, budget=whole.n, hops=3, min_count=6)
    assert list(subset.members) == [0, 1, 2, 3]

    def forbidden(*args, **kwargs):
        pytest.fail("full selection must bypass recruitment")

    monkeypatch.setattr(exporter, "recruit", forbidden)
    selected, members, metadata, recruitment = exporter.select_connectome(whole)
    assert selected is whole
    np.testing.assert_array_equal(members, np.arange(6))
    assert selected.populations["isolated"] == (5,)
    assert selected.count[-1] == 1
    assert metadata == {"mode": "full_retained", "neurons": 6, "classes": 3,
                        "all_retained_neurons": True, "all_retained_edges": True}
    assert recruitment is None


@pytest.mark.parametrize("flags", [[], ["--full"]])
def test_full_cli_exports_all_edges_and_stages_without_touching_legacy_files(tmp_path, monkeypatch, flags):
    whole = retained_graph()
    monkeypatch.setattr(exporter, "load_fly", lambda: SimpleNamespace(connectome=whole))
    manifest = tmp_path / "fixture_manifest.json"
    manifest.write_text(json.dumps({"fixture_sha256": "tiny-retained-fixture"}))
    monkeypatch.setattr(exporter, "MANIFEST_PATH", manifest)
    setup_calls = []
    original_setup = exporter.setup_lessons

    def record_setup(connectome, **kwargs):
        setup_calls.append((connectome, kwargs))
        return original_setup(connectome, **kwargs)

    monkeypatch.setattr(exporter, "setup_lessons", record_setup)
    artifact_root = tmp_path / "staged"
    legacy = ["web/data/brain.json", "web/data/lessons.json",
              "receipts/lessons_setup.json", "tests/parity_cases.json"]
    for name in legacy:
        path = artifact_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"historical fixture must remain byte-identical\n")

    exporter.main([*flags, "--gain", "0.02", "--output-dir", str(artifact_root)])

    assert len(setup_calls) == 1
    assert setup_calls[0][0] is whole
    assert setup_calls[0][1] == {"gain": .02}  # Keep the existing calibration procedure/defaults.
    payload = json.loads((artifact_root / "web/data/brain_full.json").read_text())
    assert payload["n"] == whole.n
    assert payload["edges"] == whole.synapses
    assert payload["synapses"] == int(whole.count.sum())
    assert payload["manifest"] == "tiny-retained-fixture"
    assert payload["selection"]["mode"] == "full_retained"
    assert payload["recruitment"] is None
    np.testing.assert_array_equal(decode(payload, "members", np.int32), np.arange(whole.n))
    np.testing.assert_array_equal(decode(payload, "pre", np.int32), whole.pre)
    np.testing.assert_array_equal(decode(payload, "count", np.uint16), whole.count)
    row_ptr = decode(payload, "row_ptr", np.int32)
    np.testing.assert_array_equal(np.repeat(np.arange(whole.n), np.diff(row_ptr)), whole.post)
    assert payload["populations"]["isolated"] == [5]
    assert payload["populations"]["weak_only"] == [4]
    np.testing.assert_array_equal(decode(payload, "log_gain", np.float64), [0, -3, 0, 0, 0, 0])
    assert payload["lessons_setup"]["naive_seam"] is True
    assert payload["lessons_setup"]["readout_targets"] == {"mbon:MBON11:right": .6, "mbon:MBON05:left": .4}
    assert np.count_nonzero(decode(payload, "bias", np.float64)) == 2
    setup = json.loads((artifact_root / "receipts/lessons_setup_full.json").read_text())
    assert setup["selection"] == payload["selection"]
    assert setup["recruitment"] is None and "subnet" not in setup
    assert (artifact_root / "web/data/lessons_full.json").is_file()
    parity = json.loads((artifact_root / "tests/parity_full_cases.json").read_text())
    assert len(parity["cases"]) == 6
    for name in legacy:
        assert (artifact_root / name).read_bytes() == b"historical fixture must remain byte-identical\n"


def test_explicit_budget_retains_legacy_recruitment_defaults():
    args = exporter.parse_args(["--budget", "6"])
    whole = retained_graph()
    selected, members, metadata, recruitment = exporter.select_connectome(
        whole, budget=args.budget, hops=args.hops, min_count=args.min_count)
    assert selected.n == 4 and selected.synapses == 2
    np.testing.assert_array_equal(members, [0, 1, 2, 3])
    assert metadata["mode"] == "recruited_subset"
    assert not metadata["all_retained_neurons"] and not metadata["all_retained_edges"]
    assert recruitment == {"budget": 6, "hops": 3, "min_count": 6., "hop_counts": [4]}


def test_budget_export_keeps_legacy_names_and_reports_actual_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "load_fly", lambda: SimpleNamespace(connectome=retained_graph()))
    manifest = tmp_path / "fixture_manifest.json"
    manifest.write_text(json.dumps({"fixture_sha256": "tiny-retained-fixture"}))
    monkeypatch.setattr(exporter, "MANIFEST_PATH", manifest)
    exporter.main(["--budget", "6", "--gain", "0.02", "--output-dir", str(tmp_path)])
    payload = json.loads((tmp_path / "web/data/brain.json").read_text())
    assert payload["n"] == 4 and payload["whole"]["neurons"] == 6
    assert payload["selection"]["mode"] == "recruited_subset"
    assert payload["recruitment"]["budget"] == 6
    assert "isolated" not in payload["populations"] and "weak_only" not in payload["populations"]
    setup = json.loads((tmp_path / "receipts/lessons_setup.json").read_text())
    assert setup["subnet"] == {"neurons": 4, "classes": 2, "budget": 6, "hops": 3, "min_count": 6.}
    assert (tmp_path / "web/data/lessons.json").is_file()
    assert (tmp_path / "tests/parity_cases.json").is_file()
    assert not (tmp_path / "web/data/brain_full.json").exists()


@pytest.mark.parametrize("flags", [
    ["--full", "--budget", "6"], ["--hops", "3"], ["--min-count", "6"],
    ["--budget", "0"], ["--budget", "6", "--hops", "-1"],
    ["--budget", "6", "--min-count", "nan"],
])
def test_invalid_or_conflicting_selection_arguments_fail_before_export(flags):
    with pytest.raises(SystemExit) as raised:
        exporter.parse_args(flags)
    assert raised.value.code == 2
