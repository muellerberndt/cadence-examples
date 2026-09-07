"""06 sign: show the net a sign, and it writes what it sees with a two-joint arm.

Run:  python train.py                      (about an hour on an M-series Mac)
      python build_page.py                 (embeds net.json into index.html; open it and watch it write)
      python train.py --verify receipt.json

The net sees pixels only: the sign and what it has drawn so far, through a 7x7 window
around its pen. It picks one of nine pen moves per step, and the arm's two joints follow
the pen by inverse kinematics. It learns by imitating a teacher that heads for the first
stroke cell not yet inked, from
demonstrations that include recoveries from random slips, with the same free/nudged rule
as every other rung (nine classes). Then it writes fresh signs on its own and is scored
by the overlap of its drawing with the sign and with the teacher's drawing, next to an
MLP of the same shape trained on the same demonstrations.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd
from arm import ACTIONS, G, INPUTS, K, SHAPES, STEPS, Episode, iou, make_episode, move, observation, teacher_action, write

HERE = Path(__file__).resolve().parent
SOURCES = [("06_sign/train.py", Path(__file__).resolve()), ("06_sign/arm.py", HERE / "arm.py")]
DEMOS_PER_SHAPE, HELD_OUT_PER_SHAPE = 360, 40
RECOVERY_NOISE = 0.15  # half the demonstrations include random slips, labelled by the teacher
GRID = [{"hidden": 64}, {"hidden": 128}]
GRID_EPOCHS, GRID_ROWS = 3, 60_000
SCHEDULE = {"epochs": 8, "batch": 128, "decay": 0.8, "eta": 3.0}
CONFIG = cd.LearnerConfig(beta=0.1, eta_bias=0.03, temperature=0.1, tolerance=3e-3, nudged_steps=12, free_steps=60)
READOUT_TOLERANCE = 1e-4


def demonstrations(seed: int) -> tuple[np.ndarray, np.ndarray, list[Episode], list[Episode]]:
    """Teacher traces on random signs (half with slips), plus held-out episodes for writing."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for shape in SHAPES:
        for k in range(DEMOS_PER_SHAPE):
            episode = make_episode(rng, shape)
            noise = RECOVERY_NOISE if k % 2 else 0.0
            _, trace, _ = write(None, episode, rng, noise)
            xs.extend(obs for obs, _ in trace)
            ys.extend(label for _, label in trace)
    held_out = [make_episode(rng, shape) for shape in SHAPES for _ in range(HELD_OUT_PER_SHAPE)]
    validation = [make_episode(rng, shape) for shape in SHAPES for _ in range(HELD_OUT_PER_SHAPE // 2)]
    return np.asarray(xs, dtype=np.uint8), np.asarray(ys, dtype=np.int64), validation, held_out


class SignNet:
    def __init__(self, hidden: int, seed: int, backend: str) -> None:
        self.wiring = cd.layered(INPUTS, hidden, ACTIONS, density=1.0, seed=seed)
        self.learner = cd.Learner(cd.Settlement(self.wiring, cd.learning_rule(dt=1.0), backend=backend), self.wiring.sets["output"], CONFIG)  # type: ignore[arg-type]
        self.seed = seed
        self.steps: list[tuple[float, float]] = []

    def drive(self, x: np.ndarray) -> np.ndarray:
        return self.learner.engine.clamp_levels(np.pad(np.asarray(x, dtype=float), ((0, 0), (0, self.wiring.n - INPUTS))))

    def fit(self, x: np.ndarray, y: np.ndarray, *, epochs: int, batch: int, decay: float, eta: float, log: str) -> float:
        rng = np.random.default_rng(self.seed)
        learner = self.learner
        t0 = time.perf_counter()
        for epoch in range(epochs):
            learner.config = dataclasses.replace(learner.config, eta=eta * decay**epoch, eta_bias=CONFIG.eta_bias * decay**epoch)
            order = rng.permutation(len(y))
            for s in range(0, len(order), batch):
                idx = order[s : s + batch]
                _, report = learner.step(self.drive(x[idx]), y[idx])
                self.steps.append((report["free_steps"], report["nudged_steps"]))
            print(f"{log} epoch {epoch + 1}/{epochs} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return time.perf_counter() - t0

    def act(self, x: np.ndarray) -> np.ndarray:
        learner = self.learner
        learning = learner.config
        learner.config = dataclasses.replace(learning, tolerance=READOUT_TOLERANCE)
        try:
            return learner.predict(self.drive(x))
        finally:
            learner.config = learning

    def parameters(self) -> int:
        return self.learner.parameters()


class TorchMLP:
    def __init__(self, hidden: int, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(INPUTS, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, ACTIONS))
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray, *, epochs: int, batch: int, **_: Any) -> float:
        torch = self.torch
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        rng = np.random.default_rng(self.seed)
        t0 = time.perf_counter()
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for s in range(0, len(order), batch):
                idx = order[s : s + batch]
                loss = torch.nn.functional.cross_entropy(self.net(torch.tensor(x[idx], dtype=torch.float32)), torch.tensor(y[idx]))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return time.perf_counter() - t0

    def act(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.net(self.torch.tensor(np.asarray(x, dtype=np.float32))).argmax(dim=1).numpy()

    def parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


def write_many(policy: Any, episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray, float]:
    """Every episode steps in lockstep, one batched policy call per step; returns canvases, teacher canvases, agreement."""
    n = len(episodes)
    pens = [e.stroke[0] for e in episodes]
    canvas = np.zeros((n, G, G))
    for i, pen in enumerate(pens):
        canvas[i][pen] = 1.0
    agree = 0
    for _ in range(STEPS):
        obs = np.stack([observation(e.sign, canvas[i], pens[i]) for i, e in enumerate(episodes)])
        labels = np.array([teacher_action(canvas[i], pens[i], e.stroke) for i, e in enumerate(episodes)])
        actions = policy.act(obs)
        agree += int((actions == labels).sum())
        for i in range(n):
            pens[i] = move(pens[i], int(actions[i]))
            canvas[i][pens[i]] = 1.0
    teacher = np.stack([write(None, e)[0] for e in episodes])
    return canvas, teacher, agree / (n * STEPS)


def evaluate(policy: Any, episodes: list[Episode]) -> dict[str, Any]:
    canvas, teacher, agreement = write_many(policy, episodes)
    per_shape: dict[str, dict[str, float]] = {}
    for shape in SHAPES:
        idx = [i for i, e in enumerate(episodes) if e.shape == shape]
        per_shape[shape] = {"iou_sign": float(np.mean([iou(canvas[i], episodes[i].sign) for i in idx])), "iou_teacher": float(np.mean([iou(canvas[i], teacher[i]) for i in idx]))}
    return {
        "episodes": len(episodes),
        "teacher_agreement": agreement,
        "iou_sign": float(np.mean([iou(canvas[i], episodes[i].sign) for i in range(len(episodes))])),
        "iou_teacher": float(np.mean([iou(canvas[i], teacher[i]) for i in range(len(episodes))])),
        "teacher_iou_sign": float(np.mean([iou(teacher[i], episodes[i].sign) for i in range(len(episodes))])),
        "per_shape": per_shape,
    }


def export(net: SignNet, path: Path, meta: dict[str, Any]) -> None:
    engine = net.learner.engine
    rule = engine.rule
    payload = {
        "n": net.wiring.n, "sets": {k: [int(i) for i in v] for k, v in net.wiring.sets.items()},
        "W": [round(float(w), 5) for w in engine.dense().ravel()], "bias": [round(float(v), 5) for v in engine.bias],
        "rule": {"slope": rule.slope, "threshold": rule.threshold, "leak": rule.leak, "dt": rule.dt, "clamp": rule.clamp_amplitude, "rest": rule.rest_emission},
        "arm": {"G": G, "K": K, "steps": STEPS, "lo": 0.25, "side": 1.15, "shapes": list(SHAPES)},
        "meta": meta,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def run(seed: int, out: Path, backend: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    x, y, validation, held_out = demonstrations(seed)
    print(f"{len(y)} demonstration rows from {DEMOS_PER_SHAPE * len(SHAPES)} signs ({time.perf_counter() - t0:.0f}s); {len(validation)} validation and {len(held_out)} held-out signs; backend {backend}", flush=True)

    table = []
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(y), GRID_ROWS, replace=False)
    for candidate in GRID:
        net = SignNet(candidate["hidden"], seed, backend)
        seconds = net.fit(x[subset], y[subset], epochs=GRID_EPOCHS, batch=SCHEDULE["batch"], decay=SCHEDULE["decay"], eta=SCHEDULE["eta"], log=f"validation hidden {candidate['hidden']}")
        result = evaluate(net, validation)
        table.append({**candidate, "parameters": net.parameters(), "seconds": seconds, "validation_iou_teacher": result["iou_teacher"], "validation_agreement": result["teacher_agreement"]})
        print(f"validation: hidden {candidate['hidden']} -> IoU vs teacher {result['iou_teacher']:.3f}, agreement {result['teacher_agreement']:.3f} ({net.parameters()} parameters, {seconds:.0f}s)", flush=True)
    best = max(table, key=lambda r: (r["validation_iou_teacher"], -r["parameters"]))
    selected = {"hidden": best["hidden"]}

    net = SignNet(selected["hidden"], seed, backend)
    seconds = net.fit(x, y, log=f"selected hidden {selected['hidden']}", **SCHEDULE)
    steps = np.asarray(net.steps).mean(axis=0)
    result = evaluate(net, held_out)
    print(f"selected hidden {selected['hidden']}: held-out IoU vs sign {result['iou_sign']:.3f}, vs teacher {result['iou_teacher']:.3f}, agreement {result['teacher_agreement']:.3f} (teacher vs sign {result['teacher_iou_sign']:.3f}); {seconds:.0f}s, {net.parameters()} parameters", flush=True)

    mlp = TorchMLP(selected["hidden"], seed)
    mlp_seconds = mlp.fit(x, y, **SCHEDULE)
    mlp_result = evaluate(mlp, held_out)
    print(f"baseline MLP: IoU vs sign {mlp_result['iou_sign']:.3f}, vs teacher {mlp_result['iou_teacher']:.3f}, agreement {mlp_result['teacher_agreement']:.3f} ({mlp.parameters()} parameters, {mlp_seconds:.0f}s)", flush=True)

    engine = net.learner.engine
    cpu_engine = cd.Settlement(net.wiring, engine.rule, edge_scale=engine.edge_scale, log_gain=engine.log_gain, bias=engine.bias)
    conformance = cd.conformance(cpu_engine, net.drive(x[:1])[0], steps=60)
    # a few held-out writings for the receipt and the page, as compact strings
    canvases, teacher_canvases, _ = write_many(net, held_out[:12])
    gallery = [{"shape": e.shape, "sign": "".join("#" if v else "." for v in e.sign.ravel()), "drawn": "".join("#" if v else "." for v in canvases[i].ravel()), "iou_sign": iou(canvases[i], e.sign)} for i, e in enumerate(held_out[:12])]
    for row in gallery[:3]:
        print(f"{row['shape']:6s} sign / drawn (IoU {row['iou_sign']:.2f}):")
        for r in range(G - 1, -1, -1):
            print("   " + row["sign"][r * G : (r + 1) * G] + "    " + row["drawn"][r * G : (r + 1) * G])
    export(net, HERE / "net.json", {"iou_sign": result["iou_sign"], "iou_teacher": result["iou_teacher"], "parameters": net.parameters()})

    body = {
        "task": {"grid": G, "steps": STEPS, "shapes": list(SHAPES), "demonstrations": int(len(y)), "demos_per_shape": DEMOS_PER_SHAPE, "recovery_noise": RECOVERY_NOISE, "held_out_signs": len(held_out), "validation_signs": len(validation), "seed": seed},
        "grid_search": GRID, "grid_epochs": GRID_EPOCHS, "grid_rows": GRID_ROWS, "schedule": SCHEDULE, "config": CONFIG.to_dict(), "readout_tolerance": READOUT_TOLERANCE,
        "training_backend": {"name": backend, "device": cd.available_backends().get(backend)},
        "validation": table, "selected": selected,
        "wiring": net.wiring.summary(), "rule": engine.rule.to_dict(), "learner": net.learner.to_dict(),
        "training_seconds": seconds, "mean_free_steps": float(steps[0]), "mean_nudged_steps": float(steps[1]),
        "held_out": result, "gallery": gallery, "conformance": conformance,
        "comparison": [{"model": f"MLP {INPUTS}-{selected['hidden']}-{ACTIONS}, Adam", "parameters": mlp.parameters(), "epochs": SCHEDULE["epochs"], "seconds": mlp_seconds, "held_out": mlp_result}],
        "boundary": {"learning_rule": "free/nudged contrastive Hebbian, centered, owner-local", "goal_enters_only_through_the_nudge": True, "teacher": "heads for the first stroke cell not yet inked; the net imitates it from a 7x7 pixel window around its pen; the arm's joints follow the pen by inverse kinematics", "selection_on_validation_signs_only": True, "held_out_signs_written_once": True, "baseline_trained_on_the_same_demonstrations": True},
    }
    receipt = cd.Receipt.build("cadence-examples/06-sign/v1", body, sources=SOURCES)
    receipt.write(out)
    print(f"conformance {conformance['max_abs_deviation']:.1e}; receipt {out} ({receipt.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    best = max(body["validation"], key=lambda r: (r["validation_iou_teacher"], -r["parameters"]))
    if {"hidden": best["hidden"]} != body["selected"]:
        return "selected configuration is not the best on validation"
    for row in body["gallery"]:
        drawn = np.array([c == "#" for c in row["drawn"]]).reshape(G, G)
        sign = np.array([c == "#" for c in row["sign"]]).reshape(G, G)
        if abs(iou(drawn.astype(float), sign.astype(float)) - row["iou_sign"]) > 1e-9:
            return "a gallery IoU does not follow from its canvases"
    if not body["conformance"]["ledger"]["clean"]:
        return "reference ledger is not clean"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="cpu")  # the net is small; the CPU backend is fastest here
    parser.add_argument("--output", type=Path, default=HERE / "receipt.json")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=SOURCES, check=check)
        print(message)
        return 0 if ok else 1
    run(args.seed, args.output, args.backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
