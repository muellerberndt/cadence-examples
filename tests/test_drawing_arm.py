"""Behavioral regressions on the actual ink bitmap and sensory/motor ablations."""

from pathlib import Path
import subprocess


def test_continuous_drawing_and_ink_repair():
    subprocess.run(
        ["node", "--test", "eye-arm/brain.test.mjs"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=120,
    )
