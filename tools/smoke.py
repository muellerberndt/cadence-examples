"""Run every rung end to end at a tiny budget and verify the receipt it writes; the files in the repo stay untouched.

Run:  python tools/smoke.py            (all rungs; a few minutes)
      python tools/smoke.py 04_pong 07_music

Each rung is copied to a temporary directory and imported there; its budget
constants are replaced on the module before ``run`` is called with an output in a temporary directory. What is
checked: the run completes, the receipt it wrote verifies against the rung's sources and its own ``check``. Side files and generated data remain in that temporary directory. No checkout file is rewritten or restored.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cadence as cd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_rung(name: str, root: Path = ROOT):  # type: ignore[no-untyped-def]
    """Import a rung's train.py with its directory first on the path (its sibling modules import lazily)."""
    for k in list(sys.modules):
        if k in ("train", "pong", "dataset", "connect4"):
            del sys.modules[k]
    sys.path[:] = [p for p in sys.path if not p.startswith(str(ROOT))]
    sys.path.insert(0, str(root / name))
    return importlib.import_module("train")


def positions_for_smoke(m) -> None:  # type: ignore[no-untyped-def]
    """Connect Four trains on a generated dataset that is not in the repo; when it is missing (a fresh checkout, the CI), build a small one."""
    import dataset

    if dataset.DATA.exists():
        x, y, meta = dataset.load()
        if len(y) > 2000:
            # This path is already an isolated copy. Bound smoke cost on a research checkout.
            take = np.random.default_rng(0).choice(len(y), 2000, replace=False)
            x, y = x[take], y[take]
            meta = {**meta, "positions": len(y), "digest": dataset.digest_of(x, y),
                    "smoke_subset": True, "label_histogram": np.bincount(y, minlength=7).tolist(),
                    "mean_plies": float(x.sum(axis=1).mean())}
            np.savez_compressed(dataset.DATA, x=x, y=y, meta=np.array(meta, dtype=object))
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
    "05_memory": lambda m: None,
    "06_interventions": lambda m: None,
}
ARGS = {
    "01_digits": lambda m, out: m.run(0, out, 1),
    "02_recall": lambda m, out: m.run(0, out, 1),
    "03_connect_four": lambda m, out: m.run(0, out),
    "04_pong": lambda m, out: m.run(0, out, 2),
    "05_memory": lambda m, out: m.run(out, steps=5, seeds=1, trials=3),
    "06_interventions": lambda m, out: m.run(out, steps=5, seeds=1, trials=3),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rungs", nargs="*", help="numbered example directories; default: all six")
    names = parser.parse_args().rungs or list(BUDGETS)
    unknown = [name for name in names if name not in BUDGETS]
    if unknown:
        parser.error(f"unknown rung(s): {', '.join(unknown)}; choose from {', '.join(BUDGETS)}")
    failed = []
    for name in names:
        t0 = time.perf_counter()
        original_path = sys.path.copy()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
                module = load_rung(name, root)
                BUDGETS[name](module)
                out = root / "receipt.json"
                ARGS[name](module, out)
                sources = module.SOURCES if hasattr(module, "SOURCES") else module.sources()
                ok, message = cd.Receipt.verify(out, sources=sources, check=getattr(module, "check", None))
        except Exception as error:  # noqa: BLE001
            ok, message = False, f"{type(error).__name__}: {error}"
        finally:
            sys.path[:] = original_path
        print(f"{'ok  ' if ok else 'FAIL'} {name:16s} {time.perf_counter() - t0:6.0f}s  {message}", flush=True)
        if not ok:
            failed.append(name)
    print("all rungs verified" if not failed else f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
