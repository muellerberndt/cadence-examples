"""A known feedback circuit answers new interventions without fitting a surrogate.

Run: python train.py --steps 2000 --seeds 3 --trials 128
Every arm receives identical weights, drives and ablation masks. No biological claim.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import cadence as cd

HERE = Path(__file__).resolve().parent
N = 16
CONTRACTION = 0.75
CONDITIONS = ("new_drives", "new_graphs", "more_ablations", "strong_drive", "small_changes")
ARMS = ("cadence", "cadence_warm", "mlp", "one_propagation", "unrolled8", "unrolled32", "newton")
WIDTHS = (128, 256)
RULE = cd.GradedRule(gain=1, slope=2, threshold=0, leak=1, dt=1, clamp_amplitude=1)
BOUNDARY = {"supplied_dynamics_not_learned_physics": True,
    "complete_weights_drives_and_masks_given_to_all_arms": True,
    "mlp_gets_exact_first_propagation_and_output_constraints": True,
    "no_biological_or_universal_learning_claim": True,
    "fixed_unrolling_is_also_a_feedforward_computational_graph": True,
    "direct_solver_and_recurrence_are_strong_controls": True,
    "graph_binding_and_training_costs_reported_separately": True,
    "batch_and_sequential_timings_are_distinct": True}


def sources():
    package = Path(cd.__file__).resolve().parent
    return [("06_interventions/train.py", Path(__file__).resolve())] + [
        (f"cadence/{path.name}", path) for path in sorted(package.glob("*.py"))]


def graphs(seed, count):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(count, N, N)) * (rng.random((count, N, N)) < .18)
    w[:, np.arange(N), np.arange(N)] = 0
    w *= CONTRACTION / np.maximum(np.abs(w).sum(axis=1), 1e-12)[:, None, :]
    return w


def sample(seed, bank, count, remove=1, amplitude=.6, small_changes=False):
    rng = np.random.default_rng(seed)
    ids = rng.integers(len(bank), size=count)
    drive = rng.normal(scale=amplitude, size=(count, N)) * (rng.random((count, N)) < .25)
    mask = np.ones((count, N))
    for i in range(count):
        mask[i, rng.choice(N, rng.integers(remove + 1), replace=False)] = 0
        if small_changes and i % 16:
            ids[i], drive[i], mask[i] = ids[i - 1], drive[i - 1].copy(), mask[i - 1].copy()
            drive[i, rng.integers(N)] += rng.normal(scale=.02)
    return ids, bank[ids], drive, mask


def evaluation(seed, condition, trials):
    train_bank, test_bank = graphs(seed + 1000, 32), graphs(seed + 1100, 16)
    bank = train_bank if condition == "new_drives" else test_bank
    return sample(seed + 4000 + CONDITIONS.index(condition), bank, trials,
                  remove=3 if condition in ("more_ablations", "strong_drive") else 1,
                  amplitude=2 if condition == "strong_drive" else .6,
                  small_changes=condition == "small_changes")


def propagate(w, drive, mask, steps):
    """Fixed-depth, weight-tied recurrence; also an explicitly unrolled feedforward graph."""
    state = np.zeros_like(drive)
    for _ in range(steps):
        state = mask * np.tanh(np.einsum("bi,bij->bj", state, w) + drive)
    return state


def residual(state, w, drive, mask):
    return np.max(np.abs(state - mask * np.tanh(np.einsum("bi,bij->bj", state, w) + drive)), axis=1)


def features(w, drive, mask):
    import torch
    # A strong baseline gets the exact first propagation, known activation and mask.
    message = np.einsum("bi,bij->bj", mask * np.tanh(drive), w)
    array = np.concatenate([w.reshape(len(w), -1), message, drive, mask], axis=1)
    return torch.tensor(array, dtype=torch.float64)


def model(width, seed):
    import torch
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(N * N + 3 * N, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, N)).double()


def predict(net, x):
    import torch
    return x[:, -N:] * torch.tanh(x[:, -2 * N:-N] + x[:, -3 * N:-2 * N] + net(x))


def train_mlp(seed, steps):
    import torch
    torch.set_num_threads(1)
    preparation_started = time.perf_counter()
    bank = graphs(seed + 1000, 32)
    _, w, drive, mask = sample(seed + 2000, bank, 8192)
    x = features(w, drive, mask)
    target = torch.tensor(propagate(w, drive, mask, 128), dtype=torch.float64)
    _, vw, vd, vm = sample(seed + 3000, graphs(seed + 1200, 8), 512)
    vx = features(vw, vd, vm)
    vy = torch.tensor(propagate(vw, vd, vm, 128), dtype=torch.float64)
    preparation_seconds = time.perf_counter() - preparation_started
    best = None
    candidates = []
    started = time.perf_counter()
    for width in WIDTHS:
        net = model(width, seed)
        optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        rng = np.random.default_rng(seed + 2100)
        history = []
        for step in range(1, steps + 1):
            rows = torch.tensor(rng.integers(len(x), size=128))
            loss = ((predict(net, x[rows]) - target[rows]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step in {max(1, steps // 2), steps}:
                with torch.inference_mode():
                    mse = float(((predict(net, vx) - vy) ** 2).mean())
                history.append({"step": step, "validation_mse": mse})
                if best is None or mse < best[0]:
                    best = (mse, width, step, {k: v.detach().clone() for k, v in net.state_dict().items()})
        candidates.append({"width": width, "parameters": sum(p.numel() for p in net.parameters()),
                           "steps": steps, "validation": history})
    selected = model(best[1], seed)
    selected.load_state_dict(best[3])
    selected.eval()
    return selected, {"candidates": candidates, "selected_width": best[1], "selected_step": best[2],
                       "selected_validation_mse": best[0], "seconds": time.perf_counter() - started,
                       "preparation_seconds": preparation_seconds,
                       "parameters": sum(p.numel() for p in selected.parameters()),
                       "training_examples": 8192, "batch": 128, "steps_per_candidate": steps,
                       "baseline_prior": "exact first propagation, direct-drive skip, tanh, hard output mask"}


def engine(w):
    pre, post = np.nonzero(w)
    wiring = cd.Wiring.from_edges(N, pre=pre, post=post, sign=w[pre, post])
    return cd.Settlement(wiring, RULE)


def cadence_arm(w, drive, mask, warm=False):
    bindings, states = {}, {}
    started = time.perf_counter()
    for wi in w:
        key = wi.tobytes()
        if key not in bindings:
            bindings[key] = engine(wi)
    binding_seconds = time.perf_counter() - started
    # Initialization, including any first-use JIT, is charged separately from solving.
    started = time.perf_counter()
    for binding in bindings.values():
        binding.settle(np.zeros(N), steps=1)
    initialization_seconds = time.perf_counter() - started
    outputs, steps, latencies = [], [], []
    started = time.perf_counter()
    for wi, di, mi in zip(w, drive, mask, strict=True):
        began = time.perf_counter()
        key = wi.tobytes()
        state = bindings[key].settle(di, mask=mi, state=states.get(key) if warm else None,
                                     steps=128, tolerance=1e-11)
        latencies.append(time.perf_counter() - began)
        outputs.append(state.activation)
        steps.append(state.steps)
        if warm:
            states[key] = state
    result = np.asarray(outputs)
    elapsed = time.perf_counter() - started
    return result, {"binding_seconds": binding_seconds,
        "initialization_seconds": initialization_seconds,
        "sequential_seconds": elapsed, "median_query_seconds": float(np.median(latencies)),
        "steps": steps, "mean_steps": float(np.mean(steps))}


def newton_arm(w, drive, mask):
    from scipy.optimize import root
    out = []
    started = time.perf_counter()
    for wi, di, mi in zip(w, drive, mask, strict=True):
        fun = lambda value: value - mi * np.tanh(value @ wi + di)
        jac = lambda value: np.eye(N) - (mi * (1 - np.tanh(value @ wi + di) ** 2))[:, None] * wi.T
        solved = root(fun, np.zeros(N), jac=jac, tol=1e-11)
        if np.max(np.abs(fun(solved.x))) > 1e-9:
            raise RuntimeError("independent Newton solver did not meet its residual contract")
        out.append(solved.x)
    return np.asarray(out), {"sequential_seconds": time.perf_counter() - started}


def measurements(prediction, target, w, drive, mask, timing):
    rows = residual(prediction, w, drive, mask)
    return {"prediction": prediction.tolist(), "mse": float(((prediction - target) ** 2).mean()),
            "max_residual": float(rows.max()), "mean_residual": float(rows.mean()),
            "max_error_bound": float(rows.max() / (1 - CONTRACTION)),
            "positive_threshold_agreement": float(((prediction > .2) == (target > .2)).mean()), **timing}


def run(output, steps=2000, seeds=3, trials=128):
    import scipy
    import torch
    from cadence.settle import _FUSED
    from cadence.timing import environment
    from cadence.receipts import source_manifest
    frozen_sources = source_manifest(sources())
    runs = []
    for seed in range(seeds):
        net, training = train_mlp(seed, steps)
        rows = []
        for condition in CONDITIONS:
            _, w, drive, mask = evaluation(seed, condition, trials)
            target = propagate(w, drive, mask, 128)
            arms = {}
            for name in ARMS:
                if name.startswith("cadence"):
                    prediction, timing = cadence_arm(w, drive, mask, warm=name == "cadence_warm")
                elif name == "newton":
                    prediction, timing = newton_arm(w, drive, mask)
                elif name == "mlp":
                    with torch.inference_mode():
                        predict(net, features(w[:1], drive[:1], mask[:1]))
                        started = time.perf_counter()
                        prediction = predict(net, features(w, drive, mask)).numpy()
                        batch_seconds = time.perf_counter() - started
                        started = time.perf_counter()
                        for i in range(len(w)):
                            predict(net, features(w[i:i + 1], drive[i:i + 1], mask[i:i + 1]))
                        timing = {"batch_seconds": batch_seconds, "sequential_seconds": time.perf_counter() - started}
                else:
                    count = {"one_propagation": 2, "unrolled8": 8, "unrolled32": 32}[name]
                    started = time.perf_counter()
                    prediction = propagate(w, drive, mask, count)
                    batch_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    for i in range(len(w)):
                        propagate(w[i:i + 1], drive[i:i + 1], mask[i:i + 1], count)
                    timing = {"steps": count, "batch_seconds": batch_seconds,
                              "sequential_seconds": time.perf_counter() - started}
                arms[name] = measurements(prediction, target, w, drive, mask, timing)
            rows.append({"condition": condition, "target": target.tolist(), "arms": arms,
                         "data_sha256": hashlib.sha256(w.tobytes() + drive.tobytes() + mask.tobytes()).hexdigest(),
                         "max_incoming_absolute_sum": float(np.abs(w).sum(axis=1).max())})
            print(f"seed {seed}, {condition}: " + ", ".join(f"{arm} MSE {arms[arm]['mse']:.3g}" for arm in ARMS), flush=True)
        runs.append({"seed": seed, "mlp_training": training, "rows": rows})
    body = {"task": {"owners": N, "conditions": list(CONDITIONS), "trials": trials, "seeds": seeds,
                     "training_steps": steps, "contraction_bound": CONTRACTION},
            "runs": runs, "rule": RULE.to_dict(),
            "environment": {**environment(), "scipy": scipy.__version__, "cadence": cd.__version__,
                "precision": "float64 for every arm", "device": "cpu", "fused_cpu": _FUSED,
                "torch_num_threads": torch.get_num_threads()},
            "boundary": BOUNDARY}
    assert frozen_sources == source_manifest(sources()), "source changed during the run"
    cd.Receipt.build("cadence-examples/06-interventions/v1", body, sources=sources()).write(output)
    return body


def check(body):
    task = body["task"]
    if (body["boundary"] != BOUNDARY or body["environment"]["device"] != "cpu"
            or body["environment"]["precision"] != "float64 for every arm"):
        return "protocol boundary differs"
    if (task["conditions"] != list(CONDITIONS) or task["owners"] != N
            or task["contraction_bound"] != CONTRACTION or body["rule"] != RULE.to_dict()
            or min(task["trials"], task["seeds"], task["training_steps"]) < 1):
        return "invalid task schedule"
    if [run["seed"] for run in body["runs"]] != list(range(task["seeds"])):
        return "seed schedule differs"
    for run in body["runs"]:
        training = run["mlp_training"]
        candidates = training["candidates"]
        if ([c["width"] for c in candidates] != list(WIDTHS)
                or training["steps_per_candidate"] != task["training_steps"]
                or training["training_examples"] != 8192 or training["batch"] != 128):
            return "training budget differs"
        validations = []
        for candidate in candidates:
            width = candidate["width"]
            parameters = (N * N + 3 * N + 1) * width + (width + 1) * width + (width + 1) * N
            if candidate["parameters"] != parameters or candidate["steps"] != task["training_steps"]:
                return "training candidate differs"
            checkpoints = candidate["validation"]
            expected_steps = sorted({max(1, task["training_steps"] // 2), task["training_steps"]})
            if [c["step"] for c in checkpoints] != expected_steps:
                return "validation schedule differs"
            for checkpoint in checkpoints:
                value = checkpoint["validation_mse"]
                if not np.isfinite(value) or value < 0:
                    return "invalid validation measurement"
                validations.append((value, width, checkpoint["step"], parameters))
        selected = min(validations, key=lambda value: value[0])
        if selected != (training["selected_validation_mse"], training["selected_width"],
                        training["selected_step"], training["parameters"]):
            return "validation selection differs"
        if any(not np.isfinite(training[k]) or training[k] < 0 for k in ("seconds", "preparation_seconds")):
            return "invalid training timing"
        if [row["condition"] for row in run["rows"]] != list(CONDITIONS):
            return "condition schedule differs"
        for row in run["rows"]:
            _, w, drive, mask = evaluation(run["seed"], row["condition"], task["trials"])
            if hashlib.sha256(w.tobytes() + drive.tobytes() + mask.tobytes()).hexdigest() != row["data_sha256"]:
                return "data digest differs"
            incoming = np.abs(w).sum(axis=1).max()
            if incoming > CONTRACTION + 1e-12 or row["max_incoming_absolute_sum"] != incoming:
                return "circuit is not within the declared contraction bound"
            target = np.asarray(row["target"])
            if (target.shape != drive.shape or not np.isfinite(target).all()
                    or residual(target, w, drive, mask).max() > 1e-12):
                return "target violates the independent fixed-point equation"
            if set(row["arms"]) != set(ARMS):
                return "comparison arm missing"
            for name, arm in row["arms"].items():
                times = ["sequential_seconds"]
                if name.startswith("cadence"):
                    times += ["binding_seconds", "initialization_seconds", "median_query_seconds"]
                    taken = np.asarray(arm["steps"])
                    if (taken.shape != (task["trials"],) or not np.isfinite(taken).all()
                            or np.any(taken < 1) or np.any(taken > 128) or np.any(taken != taken.astype(int))
                            or arm["mean_steps"] != float(taken.mean())):
                        return f"{name}: invalid step counts"
                elif name != "newton":
                    times.append("batch_seconds")
                    expected_steps = {"one_propagation": 2, "unrolled8": 8, "unrolled32": 32}
                    if name in expected_steps and arm["steps"] != expected_steps[name]:
                        return f"{name}: invalid step count"
                if any(not np.isfinite(arm[k]) or arm[k] < 0 for k in times):
                    return f"{name}: invalid timing"
                prediction = np.asarray(arm["prediction"])
                if prediction.shape != target.shape or not np.isfinite(prediction).all():
                    return f"{name}: invalid predictions"
                depths = {"one_propagation": 2, "unrolled8": 8, "unrolled32": 32}
                if name in depths and not np.allclose(
                        prediction, propagate(w, drive, mask, depths[name]), rtol=1e-12, atol=1e-14):
                    return f"{name}: prediction differs from declared recurrence depth"
                expected = measurements(prediction, target, w, drive, mask, {})
                for metric in ("mse", "max_residual", "mean_residual", "max_error_bound", "positive_threshold_agreement"):
                    # Different libm/vector kernels can move tanh/dot by a few ulps.
                    # Allow this absolute roundoff when rechecking tiny residuals;
                    # the separate solver contract still rejects residual > 1e-9.
                    allowance = {"max_residual": 2e-14, "mean_residual": 2e-14,
                                 "max_error_bound": 8e-14}.get(metric, 1e-25)
                    if not np.isclose(arm[metric], expected[metric], rtol=1e-10, atol=allowance):
                        return f"{name}: {metric} does not recompute"
                if name in ("cadence", "cadence_warm", "newton") and expected["max_residual"] > 1e-9:
                    return f"{name}: solver failed its residual contract"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--output", type=Path, default=HERE / "receipt.json")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=sources(), check=check)
        print(message)
        return int(not ok)
    if min(args.steps, args.seeds, args.trials) < 1:
        parser.error("budgets must be positive")
    run(args.output, args.steps, args.seeds, args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
