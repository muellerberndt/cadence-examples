"""Run every rung end to end at a tiny budget and verify the receipt it writes; the receipts in the repo stay untouched.

Run:  python tools/smoke.py            (all rungs; a few minutes)
      python tools/smoke.py 04_pong 10_parrot

Each rung's script is imported unchanged (its receipt binds it, so it is never edited for a test) and its budget
constants are replaced on the module before ``run`` is called with an output in a temporary directory. What is
checked: the run completes, the receipt it wrote verifies against the rung's sources and its own ``check``.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
import time
from pathlib import Path

import cadence as cd

ROOT = Path(__file__).resolve().parents[1]


def load_rung(name: str):  # type: ignore[no-untyped-def]
    """Import a rung's train.py with its directory first on the path (its sibling modules import lazily)."""
    for k in list(sys.modules):
        if k in ("train", "arm", "pong", "brain", "syrinx", "board"):
            del sys.modules[k]
    sys.path[:] = [p for p in sys.path if not p.startswith(str(ROOT))]
    sys.path.insert(0, str(ROOT / name))
    return importlib.import_module("train")


BUDGETS = {
    "01_digits": lambda m: (setattr(m, "GRID", m.GRID[:1]), setattr(m, "VALIDATION_SEEDS", 1), setattr(m, "SCHEDULE", {**m.SCHEDULE, "epochs": 2})),
    "02_images": lambda m: (setattr(m, "GRID", [{"hidden": 32}]), setattr(m, "SCHEDULE", {**m.SCHEDULE, "epochs": 1}), setattr(m, "VALIDATION", 500)),
    "03_connect_four": lambda m: (setattr(m, "GRID", [{"hidden": 16}]), setattr(m, "SCHEDULE", {**m.SCHEDULE, "epochs": 1}), setattr(m, "GAMES_PER_OPPONENT", 4), setattr(m, "CONTROL_EPOCHS", 1)),
    "04_pong": lambda m: None,
    "05_text": lambda m: (setattr(m, "GRID", [{"hidden": 32}]), setattr(m, "SCHEDULE", {**m.SCHEDULE, "epochs": 1}), setattr(m, "TRAIN_CHARS", 20_000), setattr(m, "VALIDATION_CHARS", 5_000), setattr(m, "TEST_CHARS", 5_000), setattr(m, "GRID_CHARS", 10_000), setattr(m, "GRID_EPOCHS", 1)),
    "06_sign": lambda m: (setattr(m, "DEMOS_PER_SHAPE", 40), setattr(m, "HELD_OUT_PER_SHAPE", 4), setattr(m, "GRID", [{"hidden": 16}]), setattr(m, "GRID_EPOCHS", 1), setattr(m, "GRID_ROWS", 4000), setattr(m, "SCHEDULE", {**m.SCHEDULE, "epochs": 1})),
    "07_music": lambda m: (setattr(m, "GRID", [{"hidden": 32}]), setattr(m, "GRID_EPOCHS", 1), setattr(m, "SCHEDULE", {**m.SCHEDULE, "epochs": 1}), setattr(m, "TRANSPOSITIONS", [0])),
    "08_cartpole": lambda m: None,
    "09_celegans": lambda m: setattr(m, "UPDATES", 30),
    "10_parrot": lambda m: None,
}
ARGS = {
    "01_digits": lambda m, out: m.run(0, out, 1),
    "02_images": lambda m, out: m.run(0, out, "cpu"),
    "03_connect_four": lambda m, out: m.run(0, out),
    "04_pong": lambda m, out: m.run(0, out, 2),
    "05_text": lambda m, out: m.run(0, out, "cpu"),
    "06_sign": lambda m, out: m.run(0, out, "cpu"),
    "07_music": lambda m, out: m.run(0, out, "cpu"),
    "08_cartpole": lambda m, out: m.run(0, out, 2),
    "09_celegans": lambda m, out: m.run(out, 1, 1),
    "10_parrot": lambda m, out: m.run(0, 1.0, out, "_smoke"),
}


def main() -> int:
    names = sys.argv[1:] or list(BUDGETS)
    failed = []
    for name in names:
        t0 = time.perf_counter()
        module = load_rung(name)
        BUDGETS[name](module)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "receipt.json"
            try:
                ARGS[name](module, out)
                ok, message = cd.Receipt.verify(out, sources=module.SOURCES, check=getattr(module, "check", None))
            except Exception as error:  # noqa: BLE001
                ok, message = False, f"{type(error).__name__}: {error}"
        print(f"{'ok  ' if ok else 'FAIL'} {name:16s} {time.perf_counter() - t0:6.0f}s  {message}", flush=True)
        if not ok:
            failed.append(name)
    if "10_parrot" in names:  # the parrot's tagged side files are the smoke's; remove them
        for p in (ROOT / "10_parrot").glob("*_smoke*"):
            p.unlink()
        for p in (ROOT / "10_parrot" / "imitations").glob("*_smoke*"):
            p.unlink()
    print("all rungs verified" if not failed else f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
