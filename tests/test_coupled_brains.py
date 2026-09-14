"""Every live controller is independently checked with Python Cadence."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import cadence as cd

ROOT = Path(__file__).resolve().parents[1]


def test_all_six_joint_equations_and_trajectories_match_cadence():
    samples = json.loads(
        subprocess.check_output(
            [
                "node",
                "--input-type=module",
                "-e",
                "import {samples} from './tools/coupled_brain_benchmark.mjs'; console.log(JSON.stringify(samples()));",
            ],
            cwd=ROOT,
            text=True,
        )
    )
    assert set(samples) == {"arm", "mouse", "worm", "fly", "memory", "game"}
    for name, s in samples.items():
        assert "blocks" not in s, name
        pre, post, weights = zip(*s["edges"])
        engine = cd.Settlement(
            cd.Wiring.from_edges(len(s["state"]), pre=pre, post=post, sign=weights),
            cd.GradedRule(
                gain=1, slope=2, threshold=0, leak=1, dt=s["dt"], clamp_amplitude=1
            ),
        )
        state = cd.SettledState(
            v=np.array(s["initialPotential"]),
            activation=np.array(s["initialState"]),
            adaptation=np.zeros(len(s["state"])),
            steps=0,
        )
        final = engine.settle(
            s["drive"],
            state=state,
            mask=np.array(s["mask"]),
            steps=s["steps"],
            tolerance=0,
        )
        np.testing.assert_allclose(
            final.activation, s["state"], atol=1e-12, rtol=0, err_msg=name
        )
        residual = float(
            engine.residual(np.array(s["drive"]), final, mask=np.array(s["mask"]))[0]
        )
        assert residual <= s["equilibrium"]["tolerance"] + 1e-14, (name, residual)


def test_joint_receipt_binds_every_controller_and_records_causal_links():
    receipt = json.loads((ROOT / "showcase/coupled_evidence.json").read_text())
    for name, digest in receipt["sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert len(receipt["results"]) == 6
    for row in receipt["results"]:
        assert row["active_region_links"] > 0
        assert row["changed_regions_when_links_cut"]
        assert row["residual"] <= row["tolerance"]
