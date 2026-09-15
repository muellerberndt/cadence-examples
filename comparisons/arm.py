"""S01 comparison: the arm's life with a conventional world model in place of the records head.

    python comparisons/arm.py --seeds 0 --pilot --workers 2 --out runs/comparisons/arm/pilot
    python comparisons/arm.py --seeds 10 11 12 13 14 --workers 2 --out runs/comparisons/arm/acceptance

Each arm runs the S01 life at the stage's own budgets: motor babbling, reaching with the
stage's beam search over the learned model, held-out reaching on a copy that keeps
learning with exploration off, and the copier on fresh moving targets from the same
checkpoint. The world model is an online perceptron or an online transformer, with the
replay ring off or on. Everything else is the stage's: the arm, the sensors, the planner,
the seeds, the held-out targets and the paths.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != HERE]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from agent.life import Moment, seed_for
from agent.receipt import Receipt, peak_rss_mb
from arm import run as stage
from arm.brain import PREDICTED, ModelController, ModelPlanner, graph_spec
from arm.env import Arm
from comparisons.common import (
    ARMS,
    by_arm,
    guarded,
    network_of,
    ratio,
    run_jobs,
    stage_sources,
    timings,
    write_summaries,
)
from comparisons.networks import Reading

STAGE = "S01-comparison"


class Tally:
    """A controller with its predictions scored: every stored prediction is compared with the
    outcome the next moment shows, against predicting zero for the same field."""

    def __init__(self, controller: ModelController) -> None:
        self.controller = controller
        self.model: dict[str, list[float]] = {n: [] for n in PREDICTED}
        self.zero: dict[str, list[float]] = {n: [] for n in PREDICTED}
        self.pending: dict[str, np.ndarray] | None = None

    def step(self, moment: Moment):
        if moment.has_feedback and self.pending is not None:
            for name in PREDICTED:
                self.model[name].append(float(np.mean((np.asarray(self.pending[name]) - moment.observation[name]) ** 2)))
                self.zero[name].append(float(np.mean(np.asarray(moment.observation[name]) ** 2)))
        self.pending = None
        decision = self.controller.step(moment)
        if decision is not None and self.controller.last_prediction:
            self.pending = {n: np.asarray(v).copy() for n, v in self.controller.last_prediction.items()}
        return decision


def budgets(stage_config: dict, pilot: bool) -> dict[str, int]:
    """The stage's own budgets, divided the way the stage divides them for a pilot."""
    div = 10 if pilot else 1
    out = {k: max(1, v // div) for k, v in stage_config["budget"].items()}
    out["heldout_targets"] = max(8, out["heldout_targets"])
    out["copier_paths"] = max(3, out["copier_paths"])
    out["copier_decisions"] = stage_config["budget"]["copier_decisions"]
    return out


def split_of(stage_config: dict, seeds, pilot: bool) -> str:
    return "development" if pilot or all(s in stage_config["seeds"]["development"] for s in seeds) else "acceptance"


def probe(controller: ModelController, arm, targets, seed: int, split: str, name: str) -> float:
    """Held-out success of a copy with learning off: the curve never teaches the life."""
    reader = copy.deepcopy(controller)
    reader.learning = False
    reader.epsilon = 0.0
    return stage.evaluate_reaching(reader, arm, targets, None, seed, split, [], is_agent=False, name=f"{name}-curve", mode="frozen")["success"]


def run_arm(seed: int, arm_name: str, config: dict, stage_config: dict, pilot: bool) -> dict:
    """One arm of one seed: the life, the curve, held-out reaching and the copier."""
    import torch

    torch.set_num_threads(int(config.get("torch_threads", 1)))
    started = time.time()
    B = budgets(stage_config, pilot)
    split = split_of(stage_config, [seed], pilot)
    settings = config["arm"]
    reading = Reading.of(graph_spec(stage_config), **settings["reading"])
    network = network_of(arm_name, reading, settings[reading_kind(arm_name)], capacity=config["replay"]["capacity"], seed=seed)
    planner = ModelPlanner(**stage_config["planner"])
    controller = ModelController(network, planner, seed=seed, epsilon=1.0, replay=0, capacity=stage_config["replay"]["capacity"])
    arm = stage.arm_config(stage_config)
    env = Arm(arm, seed=seed_for(stage.STAGE, split, seed, 0, 0), life_id=arm_name)
    targets = stage.heldout_targets(seed, B["heldout_targets"], arm)
    interval = max(1, settings["curve"]["interval"] // (10 if pilot else 1))
    curve_targets = targets[: max(8, settings["curve"]["targets"] // (10 if pilot else 1))]
    life_latencies: list[float] = []
    curve: list[dict] = []
    moment = None
    done = 0
    babbling = {"prediction_mse": {n: [] for n in PREDICTED}, "zero_mse": {n: [] for n in PREDICTED}}
    reaching = {"success": []}
    for phase, decisions, epsilon in (
        ("babbling", B["babbling_decisions"], 1.0),
        ("reaching", B["reaching_decisions"], stage_config["actor"]["epsilon"]),
    ):
        controller.epsilon = epsilon
        left = decisions
        while left > 0:
            chunk = min(left, interval - done % interval)
            moment, outcomes = stage.run_life(controller, env, chunk, None, seed, split, phase, life_latencies, is_agent=False, m=moment)
            left -= chunk
            done += chunk
            if phase == "babbling":
                for name in PREDICTED:
                    babbling["prediction_mse"][name] += outcomes["prediction_mse"][name]
                    babbling["zero_mse"][name] += outcomes["zero_mse"][name]
            else:
                reaching["success"] += outcomes["success"]
            if done % interval == 0:
                curve.append({"decisions": done, "success": probe(controller, arm, curve_targets, seed, split, arm_name)})
    # held-out reaching and the copier, each on its own copy of the reaching checkpoint
    heldout_latencies: list[float] = []
    adapting = copy.deepcopy(controller)
    adapting.epsilon = 0.0
    tally = Tally(adapting)
    heldout = stage.evaluate_reaching(tally, arm, targets, None, seed, split, heldout_latencies, is_agent=False, name=arm_name, mode="adapting")
    copier_latencies: list[float] = []
    copier_controller = copy.deepcopy(controller)
    copier_controller.epsilon = 0.0
    copier = stage.evaluate_copier(copier_controller, arm, seed, B["copier_paths"], B["copier_decisions"], None, split, copier_latencies, is_agent=False, name=arm_name)
    return {
        "seed": seed,
        "arm": arm_name,
        "split": split,
        "budget": B,
        "network": network.describe(),
        "counts": {**network.counts(), "real_decisions": done, "controller_decisions": controller.decisions},
        "heldout": {k: v for k, v in heldout.items() if k != "per_target"},
        "heldout_prediction_ratio": {n: ratio(tally.model[n], tally.zero[n]) for n in PREDICTED},
        "babbling_prediction_ratio": {
            n: ratio(babbling["prediction_mse"][n][-500:], babbling["zero_mse"][n][-500:]) for n in PREDICTED
        },
        "babbling": {
            n: {
                "model_mse_last500": float(np.mean(babbling["prediction_mse"][n][-500:])),
                "zero_mse_last500": float(np.mean(babbling["zero_mse"][n][-500:])),
            }
            for n in PREDICTED
        },
        "reaching_training": {
            "episodes": len(reaching["success"]),
            "success_last20": float(np.mean(reaching["success"][-20:])) if reaching["success"] else float("nan"),
        },
        "copier": copier,
        "curve": curve,
        "latency_ms": {
            "life": timings(life_latencies),
            "heldout": timings(heldout_latencies),
            "copier": timings(copier_latencies),
            "all": timings(life_latencies + heldout_latencies + copier_latencies),
        },
        "wall_seconds": time.time() - started,
    }


def reading_kind(arm_name: str) -> str:
    return "transformer" if arm_name.startswith("transformer") else "mlp"


def _worker(payload):
    """One (seed, arm) job in a fresh process: the arguments travel with the job."""
    (seed, arm_name), args = payload
    return guarded(lambda job: run_arm(job[0], job[1], args["config"], args["stage_config"], args["pilot"]), (seed, arm_name))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--stage-config", type=Path, default=ROOT / "arm" / "config.json")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="jobs run in this many processes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    stage_config = json.loads(args.stage_config.read_text())
    split = split_of(stage_config, args.seeds, args.pilot)
    run_id = f"{STAGE}-{split}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    receipt = Receipt(STAGE, run_id, config, HERE, args.out)
    receipt.body["sources"]["stage_under_test"] = stage_sources(ROOT / "arm")
    receipt.body["supplied_components"] = [
        "the S01 stage: the arm, its sensors, the beam search over the learned model, the held-out targets, the moving paths and every budget",
        "the replay ring schedule",
    ]
    receipt.body["learned_components"] = ["the world model of each arm: an online perceptron or an online transformer"]
    jobs = [(seed, arm_name) for seed in args.seeds for arm_name in args.arms]
    receipt.body["seeds"] = {"scheduled": args.seeds, "arms": args.arms, "split": split, "completed": [], "failed": []}
    worker_args = {"config": config, "stage_config": stage_config, "pilot": args.pilot}
    results: list[dict] = []
    for job, outcome in run_jobs(jobs, args.workers, lambda j: run_arm(j[0], j[1], config, stage_config, args.pilot), _worker, worker_args):
        if isinstance(outcome, dict):
            results.append(outcome)
            receipt.body["seeds"]["completed"].append(list(job))
            gain = outcome["babbling_prediction_ratio"]["dd_hand"]
            print(f"seed {job[0]} {job[1]}: heldout {outcome['heldout']['success']:.2f} copier {outcome['copier']['rmse']:.3f} "
                  f"dd_hand ratio {'none' if gain is None else format(gain, '.2f')} {outcome['latency_ms']['all']['mean']:.1f} ms/decision")
        else:
            receipt.body["seeds"]["failed"].append(list(job))
            receipt.body["failures"].append({"seed": job[0], "arm": job[1], "error": outcome})
            print(f"seed {job[0]} {job[1]}: failed")
        receipt.body["metrics"]["per_arm"] = results
        receipt.write()
    if results:
        receipt.body["networks"] = {r["arm"]: r["network"] for r in results}
        receipt.body["metrics"].update({
            "heldout_success": by_arm(results, lambda r: r["heldout"]["success"]),
            "heldout_success_first50": by_arm(results, lambda r: r["heldout"]["success_first50"]),
            "copier_rmse": by_arm(results, lambda r: r["copier"]["rmse"]),
            "babbling_prediction_ratio": {
                name: by_arm(results, lambda r, n=name: r["babbling_prediction_ratio"][n]) for name in PREDICTED
            },
            "heldout_prediction_ratio": {
                name: by_arm(results, lambda r, n=name: r["heldout_prediction_ratio"][n]) for name in PREDICTED
            },
            "reaching_training_success_last20": by_arm(results, lambda r: r["reaching_training"]["success_last20"]),
            "curve": [{"seed": r["seed"], "arm": r["arm"], "points": r["curve"]} for r in results],
        })
        receipt.body["counts"] = {f"seed{r['seed']}-{r['arm']}": r["counts"] for r in results}
        receipt.body["resources"] = {
            "latency_ms": by_arm(results, lambda r: r["latency_ms"]["all"]["mean"]),
            "latency_ms_detail": [{"seed": r["seed"], "arm": r["arm"], **r["latency_ms"]} for r in results],
            "wall_seconds": by_arm(results, lambda r: r["wall_seconds"]),
            "peak_rss_mb": peak_rss_mb(),
            "parameters": {r["arm"]: r["network"]["parameters"] for r in results},
        }
        receipt.body["artifacts"] = {"summaries": write_summaries(args.out, results)}
    complete = not receipt.body["seeds"]["failed"] and len(results) == len(jobs)
    receipt.predicate("runs_complete", float(len(results)), float(len(jobs)), "every scheduled seed and arm produced a result", complete)
    path = receipt.finish(complete)
    print("status", receipt.body["status"], "->", path)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
