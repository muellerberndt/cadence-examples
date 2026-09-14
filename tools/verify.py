"""Verify every required source, digest, schedule, arithmetic and control boundary."""

import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark import hashes
from build_composites import sources

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(body, expected_sources, composite=False):
    original = dict(body)
    digest = original.pop("digest", None)
    require(
        digest
        == hashlib.sha256(
            json.dumps(original, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "evidence digest differs",
    )
    require(
        body["sources"] == expected_sources,
        "required producer, data or Cadence source bytes differ",
    )
    if composite:
        require(
            len(body["mouse"]) == 36 and len(body["arm"]) == 18,
            "incomplete body schedule",
        )
        require(
            {(r["seed"], r["condition"]) for r in body["mouse"]}
            == {
                (s, c)
                for s in range(13, 124, 10)
                for c in ("new_maze", "changed_corridor", "moved_goal")
            },
            "mouse conditions differ",
        )
        require(
            {(r["image"], r["disturbance_tick"], r["feedback"]) for r in body["arm"]}
            == {
                (i, t, f)
                for i in ("flower", "leaf", "spiral")
                for t in (60, 120, 240)
                for f in (True, False)
            },
            "arm conditions differ",
        )
        require(
            all(
                r["success"] and not r["collision"] and r["bfs_replanner_success"]
                for r in body["mouse"]
            ),
            "navigation failed",
        )
        require(
            max(body["python_javascript_error"].values()) < 1e-10,
            "composite parity failed",
        )
        require(
            all(
                body["tasks"][k]
                for k in (
                    "unfamiliar_before",
                    "new_task_after_one_write",
                    "revised_task",
                    "earlier_tasks_retained",
                )
            ),
            "task learning failed",
        )
        require(
            all(r["success"] for r in body["tasks"]["transfer"]),
            "learned task transfer failed",
        )
        for row in body["arm"]:
            require(
                row["targets"] > 0
                and abs(row["coverage"] - row["covered"] / row["targets"]) < 1e-12,
                "coverage arithmetic differs",
            )
        require(
            np.mean([r["coverage"] for r in body["arm"] if r["feedback"]]) > 0.95,
            "feedback drawing regression",
        )
    else:
        require(
            body["schedule"]["seeds"] == [7, 8, 9]
            and len(body["worm"]) == 3
            and len(body["memory"]) == 9,
            "benchmark schedule differs",
        )
        require(
            {(r["seed"], r["correlation"]) for r in body["memory"]}
            == {(s, c) for s in (7, 8, 9) for c in (0, 0.5, 0.9)},
            "memory conditions differ",
        )
        for row in body["memory"]:
            require(
                row["writes"] == 128 and row["queries"] == 996,
                "memory query budget differs",
            )
            require(
                set(row["arms"])
                == {"cadence", "additive", "dictionary", "mlp1", "mlp10", "mlp100"},
                "missing memory control",
            )
            for arm in row["arms"].values():
                require(
                    0 <= arm["correct"] <= row["queries"]
                    and abs(arm["accuracy"] - arm["correct"] / row["queries"]) < 1e-12,
                    "accuracy arithmetic differs",
                )
            if row["correlation"] == 0:
                require(
                    row["arms"]["cadence"]["accuracy"] == 1,
                    "distinct-key memory failed",
                )
        for run in body["worm"]:
            require(
                run["training_examples"] == 2048 and run["steps_per_candidate"] == 1200,
                "training budget differs",
            )
            require(
                {r["condition"] for r in run["rows"]}
                == {"intact", "new_mixtures", "many_lesions"},
                "missing worm condition",
            )
            for row in run["rows"]:
                require(
                    row["trials"] == 128
                    and set(row["arms"]) == {"cadence", "mlp", "unrolled32"},
                    "missing worm control",
                )
                require(
                    row["arms"]["cadence"]["max_residual"] < 1e-8
                    and row["arms"]["cadence"]["mse"] < 1e-15,
                    "circuit conformance failed",
                )
    return True


def main():
    verify(json.loads((EVIDENCE / "evidence.json").read_text()), hashes())
    verify(
        json.loads((EVIDENCE / "composite_evidence.json").read_text()),
        sources(),
        composite=True,
    )
    print(
        "Both shared evidence packages: source binding, digest, schedules and arithmetic pass."
    )


if __name__ == "__main__":
    main()
