"""Rebuild every model, receipt and web page from the pinned installed Cadence.

Run after installing requirements-reproduce.txt and torch. No cloud is needed.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("torch") is None:
    raise SystemExit("Full model reproduction needs torch: python -m pip install torch")
commands = [
    [sys.executable, "tools/benchmark.py", "--seeds", "3", "--steps", "1200", "--output", "evidence/evidence.json"],
    [sys.executable, "tools/build_composites.py"],
    ["node", "tools/habitat_benchmark.mjs", "--write"],
    *[["node", path] for path in [
        "tools/nervous_system_benchmark.mjs", "tools/coupled_brain_benchmark.mjs",
        "tools/adaptation_benchmark.mjs",
        "connect-four/benchmark.mjs", "benchmarks/memory/benchmark.mjs",
        "benchmarks/memory/history_benchmark.mjs", "benchmarks/memory/consolidation_benchmark.mjs"]],
    [sys.executable, "tools/update_results.py"],
    [sys.executable, "tools/build_showcase.py"],
    [sys.executable, "tools/verify.py"],
]
for command in commands:
    print("Running", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
