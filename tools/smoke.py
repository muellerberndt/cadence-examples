"""Run every rung end to end at a tiny budget and verify the receipt it writes; the files in the repo stay untouched.

Run:  python tools/smoke.py            (all rungs; a few minutes)
      python tools/smoke.py 04_pong 07_music

Each rung's script is imported unchanged (its receipt binds it, so it is never edited for a test) and its budget
constants are replaced on the module before ``run`` is called with an output in a temporary directory. What is
checked: the run completes, the receipt it wrote verifies against the rung's sources and its own ``check``. A rung
that writes side files next to its script (a net, a sample) has them put back afterwards: tracked files are
restored from git and files the run created are removed, so a smoke run leaves the checkout as it found it.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import cadence as cd

ROOT = Path(__file__).resolve().parents[1]


def untracked(name: str) -> set[str]:
    """Paths under a rung that git does not track (ignored files are left alone)."""
    out = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", name], cwd=ROOT, capture_output=True, text=True, check=False)
    return set(out.stdout.split())


def restore(name: str, before: set[str]) -> None:
    """Put a rung's directory back: tracked files as committed, files the run created removed."""
    subprocess.run(["git", "checkout", "--quiet", "--", name], cwd=ROOT, check=False)
    for path in sorted(untracked(name) - before, reverse=True):
        p = ROOT / path
        if p.is_file():
            p.unlink()


def load_rung(name: str):  # type: ignore[no-untyped-def]
    """Import a rung's train.py with its directory first on the path (its sibling modules import lazily)."""
    for k in list(sys.modules):
        if k in ("train", "pong", "dataset", "connect4"):
            del sys.modules[k]
    sys.path[:] = [p for p in sys.path if not p.startswith(str(ROOT))]
    sys.path.insert(0, str(ROOT / name))
    return importlib.import_module("train")


def positions_for_smoke(m) -> None:  # type: ignore[no-untyped-def]
    """Connect Four trains on a generated dataset that is not in the repo; when it is missing (a fresh checkout, the CI), build a small one."""
    import dataset

    if not dataset.DATA.exists():
        x, y, meta = dataset.build(games=60, seed=0)
        dataset.DATA.parent.mkdir(exist_ok=True)
        np.savez_compressed(dataset.DATA, x=x, y=y, meta=np.array(meta, dtype=object))
        print(f"  built a smoke dataset of {meta['positions']} positions from 60 games")


BUDGETS = {
    "01_digits": lambda m: (setattr(m, "GRID", m.GRID[:1]), setattr(m, "VALIDATION_SEEDS", 1), setattr(m, "SCHEDULE", {**m.SCHEDULE, "epochs": 2})),
    "02_recall": lambda m: (setattr(m, "LENGTHS", (4, 8)), setattr(m, "TRIALS", 3), setattr(m, "TRAIN_STEPS", 5)),
    "03_connect_four": lambda m: (positions_for_smoke(m), setattr(m, "GRID", [{"hidden": 16}]), setattr(m, "SCHEDULE", {**m.SCHEDULE, "epochs": 1}), setattr(m, "GAMES_PER_OPPONENT", 4), setattr(m, "CONTROL_EPOCHS", 1)),
    "04_pong": lambda m: None,
}
ARGS = {
    "01_digits": lambda m, out: m.run(0, out, 1),
    "02_recall": lambda m, out: m.run(0, out, 1),
    "03_connect_four": lambda m, out: m.run(0, out),
    "04_pong": lambda m, out: m.run(0, out, 2),
}


def main() -> int:
    names = sys.argv[1:] or list(BUDGETS)
    failed = []
    for name in names:
        t0 = time.perf_counter()
        before = untracked(name)
        module = load_rung(name)
        BUDGETS[name](module)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "receipt.json"
            try:
                ARGS[name](module, out)
                ok, message = cd.Receipt.verify(out, sources=module.SOURCES, check=getattr(module, "check", None))
            except Exception as error:  # noqa: BLE001
                ok, message = False, f"{type(error).__name__}: {error}"
            finally:
                restore(name, before)
        print(f"{'ok  ' if ok else 'FAIL'} {name:16s} {time.perf_counter() - t0:6.0f}s  {message}", flush=True)
        if not ok:
            failed.append(name)
    print("all rungs verified" if not failed else f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
