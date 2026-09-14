"""Freeze test predictions, lesions, composition checks and source/checkpoint custody."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cadence as cd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.brain import drives, settle_checked
from composer.compose import Composer
from composer.encoding import OFFSETS, SIZES
from tools.train import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best/brain.npz")
    p.add_argument("--suite", action="store_true")
    a = p.parse_args()
    path = ROOT / a.checkpoint
    b = cd.Learner.load(path, backend="cpu", precision="float64")
    test = dict(np.load(ROOT / "data/classical/test.npz"))
    measured = evaluate(b, test, count=8192)
    ids = np.random.default_rng(19).choice(len(test["labels"]), 128, replace=False)
    drive = drives(b, test["context"][ids], test["extra"][ids])
    intact, settling = settle_checked(b, drive)
    lesions = {}
    for name in ["harmony", "rhythm", "phrase_memory"]:
        mask = np.ones(b.brain.connectome.n)
        mask[list(b.brain.connectome.populations[name])] = 0
        altered = b.brain.settle_batch(drive, steps=512, mask=mask, tolerance=0)
        lesions[name] = {
            "mean_absolute_output_change": float(
                np.abs(
                    altered.activation[:, b.output_index]
                    - intact.activation[:, b.output_index]
                ).mean()
            ),
            "changed_argmax_by_slot": {
                name: float(
                    (
                        altered.activation[
                            :, b.output_index[offset : offset + width]
                        ].argmax(1)
                        != intact.activation[
                            :, b.output_index[offset : offset + width]
                        ].argmax(1)
                    ).mean()
                )
                for name, offset, width in zip(
                    ["melody", "duration", "chord", "bass"], OFFSETS, SIZES
                )
            },
        }
    result = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "test_prediction": measured,
        "settling": settling,
        "region_lesions": lesions,
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "composer").glob("*.py"))
        },
        "compositions": [],
    }
    if a.suite:
        composer = Composer(path)
        for prompt in [
            "A sad gentle piano piece",
            "A bright heroic orchestral theme",
            "A calm baroque piano miniature",
        ]:
            for seed in [17, 29]:
                r = composer.compose(prompt, seed=seed, variants=6)
                row = {
                    k: r[k]
                    for k in [
                        "id",
                        "brief",
                        "seed",
                        "before",
                        "after",
                        "audio_after",
                        "seconds",
                        "brain",
                    ]
                }
                row["revision_accepted"] = r["revision"]["accepted"]
                row["settling_runs"] = sum(len(p["settling"]) for p in r["phrases"])
                row["capped_settling_runs"] = sum(
                    not s["converged"] for p in r["phrases"] for s in p["settling"]
                )
                row["motif"] = r["motif"]
                result["compositions"].append(row)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in [
                                "id",
                                "before",
                                "after",
                                "revision_accepted",
                                "seconds",
                            ]
                        }
                    ),
                    flush=True,
                )
    out = ROOT / "runs/assessment.json"
    out.write_text(json.dumps(result, indent=2))
    print(
        json.dumps({"test": measured, "lesions": lesions, "settling": settling}),
        flush=True,
    )


if __name__ == "__main__":
    main()
