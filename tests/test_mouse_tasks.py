from pathlib import Path
import subprocess


def test_sequential_tasks_and_shared_replanning():
    subprocess.run(["node", "--test", "mouse/brain.test.mjs"],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=120)
