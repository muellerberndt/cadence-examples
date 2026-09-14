"""Run composite-body trials and verify their numerical kernels against Cadence."""

import hashlib
import json
import subprocess
from pathlib import Path

import cadence as cd
import numpy as np
from embodied import maze_settlement, motor_settlement

HERE = Path(__file__).resolve().parent


def sources():
    files = [
        HERE / n
        for n in (
            "engine.js",
            "embodied.js",
            "embodied.py",
            "embodied_benchmark.mjs",
            "build_composites.py",
        )
    ]
    return {
        str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    } | {
        "cadence/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(cd.__file__).parent.glob("*.py"))
    }


def main():
    body = json.loads(
        subprocess.check_output(
            ["node", str(HERE / "embodied_benchmark.mjs")], text=True
        )
    )
    parity = body.pop("parity")
    errors = {
        "maze": float(
            np.max(np.abs(maze_settlement(parity["world"]) - parity["maze_state"]))
        ),
        "motor": float(
            np.max(
                np.abs(
                    motor_settlement(parity["q"], parity["target"])
                    - parity["motor_state"]
                )
            )
        ),
    }
    assert max(errors.values()) < 1e-10, errors
    assert all(r["success"] and not r["collision"] for r in body["mouse"])
    body.update(
        schema="cadence.composite-showcase/v1",
        sources=sources(),
        python_javascript_error=errors,
    )
    body["digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (HERE / "composite_evidence.json").write_text(
        json.dumps(body, separators=(",", ":")) + "\n"
    )
    print("Composite parity:", errors)
    print("Mouse:", len(body["mouse"]), "/", len(body["mouse"]), "successful scenarios")
    for mode in (True, False):
        print(
            "Arm feedback",
            mode,
            "mean coverage",
            np.mean([r["coverage"] for r in body["arm"] if r["feedback"] == mode]),
        )


if __name__ == "__main__":
    main()
