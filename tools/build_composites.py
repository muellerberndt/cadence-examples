"""Run composite-body trials and verify their numerical kernels against Cadence."""

import hashlib
import json
import subprocess
from pathlib import Path

import cadence as cd
import numpy as np
from embodied import motor_settling

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sources():
    files = [
        ROOT / n
        for n in (
            "shared/engine.js",
            "shared/embodied.js",
            "tools/embodied.py",
            "tools/embodied_benchmark.mjs",
            "tools/build_composites.py",
        )
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    } | {
        "cadence/" + p.relative_to(Path(cd.__file__).parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(cd.__file__).parent.rglob("*.py"))
    }


def main():
    body = json.loads(
        subprocess.check_output(
            ["node", str(HERE / "embodied_benchmark.mjs")], text=True
        )
    )
    parity = body.pop("parity")
    errors = {
        "motor": float(
            np.max(
                np.abs(
                    motor_settling(parity["q"], parity["target"])
                    - parity["motor_state"]
                )
            )
        ),
    }
    assert max(errors.values()) < 1e-10, errors
    body.update(
        schema="cadence.composite-showcase/v1",
        sources=sources(),
        python_javascript_error=errors,
    )
    body["digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (ROOT / "evidence/composite_evidence.json").write_text(
        json.dumps(body, separators=(",", ":")) + "\n"
    )
    print("Composite parity:", errors)
    for mode in (True, False):
        print(
            "Arm feedback",
            mode,
            "mean coverage",
            np.mean([r["coverage"] for r in body["arm"] if r["feedback"] == mode]),
        )


if __name__ == "__main__":
    main()
