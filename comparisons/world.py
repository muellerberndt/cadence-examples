"""S02 comparison: the remembered world with a conventional world model behind the stage's stores.

    python comparisons/world.py --seeds 0 --pilot --workers 2 --out runs/comparisons/world/pilot
    python comparisons/world.py --seeds 10 11 12 13 14 --workers 2 --out runs/comparisons/world/acceptance

Each arm runs the S02 life at the stage's own budgets: exploration, remembered requests at
delays 8, 16 and 32, visible and invisible corrections, the cue at delays 8 and 32, doors
that stop opening, word grounding, and then the held-out suites on a copy with learning
off. The four stores, the witnessing that writes them, the A* search over imagined
consequences and the scoring are the candidate's own, taken from ``world.brain``. The
world model is an online perceptron or an online transformer, with the replay ring off or
on, in place of the records head.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != HERE]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from agent.brain import Agent
from agent.life import Decision, Moment, seed_for
from agent.receipt import Receipt, peak_rss_mb
from comparisons.common import (
    ARMS,
    by_arm,
    guarded,
    network_of,
    run_jobs,
    stage_sources,
    timings,
    write_summaries,
)
from comparisons.networks import Reading, targets_from_moment
from world import run as stage
from world.brain import Brain, WorldPlanner, graph_spec
from world.env import World, WorldConfig
from world.tasks import Curriculum

STAGE = "S02-comparison"
GATED = ("dx", "dy", "outcome", "here_object", "carrying")


class Scorer:
    """The candidate's own scoring of a stored prediction against the moment that follows,
    without its brain: the same targets, the same correctness and the same errors."""

    score = Agent.score
    _outcome_targets = Agent._outcome_targets
    _continuous_target = Agent._continuous_target

    def __init__(self, spec) -> None:
        self.config = SimpleNamespace(graph=spec)


class Imagination:
    """The network as the stage's search reads it. The search imagines observations only,
    so the reads of the stores at the real moment join every imagined one, which is how the
    candidate's own imagination carries its recall."""

    def __init__(self, network, reads: np.ndarray) -> None:
        self.network = network
        self.reads = reads

    def predict_batch(self, observations, actions, *, observed=None, goal=None, valued=True):
        return self.network.predict_batch(observations, actions, observed=observed, goal=goal, reads=self.reads)


class NetworkBrain(Brain):
    """The candidate's stores, witnessing and search with a network as the world model."""

    def __init__(self, config: dict, seed: int, network, *, learning: bool = True) -> None:
        from world.brain import CueStore, MapStore, PlaceStore, WordStore

        self.config = config
        self.network = network
        self.fields = graph_spec(config).prediction
        self.agent = Scorer(graph_spec(config))
        self.places, self.cue, self.words, self.map = PlaceStore(), CueStore(), WordStore(), MapStore()
        self.planner = WorldPlanner(**config.get("planner", {}), seed=seed)
        self.visited: set[tuple[int, int]] = set()
        self.inspected: set[tuple[int, int]] = set()
        self.last_moment: Moment | None = None
        self.last_decision: Decision | None = None
        self.nodes = 0
        self.plan: list[int] = []
        self.plan_cells: list[tuple[int, int]] = []
        self.plan_key: tuple | None = None
        self.searches = 0
        self.reused = 0
        self.delegated = 0
        self.erase_places_at_request = False
        self.learning = learning
        self.epsilon = 0.0
        self.rng = _generator(seed)
        self.pending: tuple | None = None
        self.decisions = 0
        self.scored: dict[str, list[float]] = {}

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = float(epsilon)

    def step(self, m: Moment) -> Decision | None:
        self.witness(m)
        if m.has_feedback and self.pending is not None and self.learning:
            observation, action, observed, goal, reads = self.pending
            self.network.learn(observation, action, targets_from_moment(self.fields, m), observed=observed, goal=goal, reads=reads)
        self.pending = None
        self.last_moment = m
        if m.terminated or m.truncated:
            self.last_decision = None
            return None
        observation = {k: np.asarray(v) for k, v in m.observation.items()}
        goal = np.asarray(m.goal)
        reads = self._recall(m, 0)
        legal = np.flatnonzero(m.action_mask)
        probabilities = np.zeros(len(m.action_mask))
        budget: dict[str, int] = {}
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            action = int(legal[int(self.rng.random() * len(legal))])
            controller = "exploration"
            prediction = self.network.predict(observation, action, observed=m.observed, goal=goal, reads=reads)
            probabilities[legal] = 1.0 / len(legal)
        else:
            controller = "planner"
            action, prediction, budget = self._plan(Imagination(self.network, reads), 0, m, None, None)
            probabilities[int(action)] = 1.0
        decision = Decision(
            decision_id=self.decisions,
            event_id=m.event_id,
            stream=0,
            parameter_version=self.network.updates,
            context_version=0,
            action=int(action),
            action_mask=m.action_mask,
            probabilities=probabilities,
            log_probability=float(np.log(max(probabilities[int(action)], 1e-300))),
            controller=controller,
            prediction=prediction,
            budget=budget,
        )
        self.decisions += 1
        self.pending = (observation, int(action), m.observed, goal, reads)
        self.last_decision = decision
        return decision

    def frozen(self) -> NetworkBrain:
        """An isolated copy with learning off: the stores, the plan and the model are copies."""
        network = self.network
        self.network = None
        try:
            out = copy.deepcopy(self)
        finally:
            self.network = network
        out.network = network.copy(ring=False)
        out.learning = False
        out.epsilon = 0.0
        out.pending = None
        out.plan, out.plan_cells, out.plan_key = [], [], None
        return out

    def counts(self) -> dict[str, int]:
        return {
            **self.network.counts(),
            "decisions": self.decisions,
            "searches": self.searches,
            "reused_plans": self.reused,
            "expansions": self.nodes,
            "place_writes": self.places.writes,
            "word_writes": self.words.writes,
        }


def _generator(seed: int):
    from agent.brain import Mulberry32

    return Mulberry32(seed * 1000)


def budgets(stage_config: dict, pilot: bool) -> tuple[dict[str, int], int]:
    """The stage's own budgets and its held-out count, divided as the stage divides them."""
    div = 10 if pilot else 1
    B = {k: max(2, v // div) for k, v in stage_config["budget"].items()}
    B["explore_steps"] = max(200, B["explore_steps"])
    return B, max(10, B["heldout_episodes"] // div)


def split_of(stage_config: dict, seeds, pilot: bool) -> str:
    return "development" if pilot or all(s in stage_config["seeds"]["development"] for s in seeds) else "acceptance"


def live(life, world: World, cur: Curriculum, B: dict, stage_config: dict, meta: dict, latencies: list, counter: list) -> dict:
    """The S02 life in the stage's order: explore, requests, corrections, cue, doors, words."""
    result: dict = {"consequences": stage.run_explore(life, world, cur, B["explore_steps"], None, meta, latencies, counter)}
    result["remember"] = {}
    for delay in (8, 16, 32):
        succeeded = stage.run_remember(life, world, cur, B["remember_episodes"] // 3, delay, None, meta, latencies, counter)
        result["remember"][str(delay)] = float(np.mean(succeeded))
    result["correction_visible"] = float(np.mean(stage.run_remember(life, world, cur, B["remember_episodes"] // 4, 16, None, meta, latencies, counter, move="visible")))
    result["correction_invisible"] = float(np.mean(stage.run_remember(life, world, cur, B["remember_episodes"] // 4, 16, None, meta, latencies, counter, move="invisible")))
    result["cue"] = {}
    for delay in (8, 32):
        correct = stage.run_cue(life, world, cur, B["cue_episodes"] // 2, delay, None, meta, latencies, counter, decision_epsilon=stage_config["actor"]["epsilon"])
        result["cue"][str(delay)] = {"all": float(np.mean(correct)), "last_third": float(np.mean(correct[-max(1, len(correct) // 3):]))}
    before = stage.run_remember(life, world, cur, B["door_trials"] // 2, 8, None, meta, latencies, counter)
    cur.lock_doors()
    after = stage.run_remember(life, world, cur, B["door_trials"], 8, None, meta, latencies, counter)
    result["door"] = {
        "before": float(np.mean(before)),
        "first_quarter_after": float(np.mean(after[: max(1, len(after) // 4)])),
        "last_quarter_after": float(np.mean(after[-max(1, len(after) // 4):])),
    }
    words = stage.run_words(life, world, cur, B["word_episodes"], None, meta, latencies, counter)
    result["words_training_last_quarter"] = float(np.mean(words[-max(1, len(words) // 4):]))
    return result


def heldout(controller, name: str, seed: int, stage_config: dict, world: World, cur: Curriculum, H: int, latencies: list) -> dict:
    """The stage's held-out suites on a copy, in a world with the same map and the same rules."""
    wcfg = WorldConfig(**stage_config["world_config"])
    world_h = World(wcfg, seed=seed_for(stage.STAGE, "heldout", seed, 2000, 0), life_id=f"heldout-{name}")
    world_h._edges = dict(world.state["edges"])
    world_h._doors = set(world._doors)
    world_h._colors = world._colors.copy()
    world_h._word_of = world._word_of.copy()
    world_h.set_door_rule(world.state["door_rule"])
    cur_h = Curriculum(world_h, seed=seed_for(stage.STAGE, "heldout", seed, 2000, 1))
    cur_h.cue_rule = cur.cue_rule
    life = stage.Life(controller, is_agent=True)
    counter = [10**6]
    meta = {"seed": seed, "split": "heldout", "life": f"heldout-{name}"}
    return {
        "remember_32": float(np.mean(stage.run_remember(life, world_h, cur_h, H, 32, None, meta, latencies, counter))),
        "correction_visible": float(np.mean(stage.run_remember(life, world_h, cur_h, H // 2, 16, None, meta, latencies, counter, move="visible"))),
        "cue_8": float(np.mean(stage.run_cue(life, world_h, cur_h, H // 2, 8, None, meta, latencies, counter))),
        "cue_32": float(np.mean(stage.run_cue(life, world_h, cur_h, H // 2, 32, None, meta, latencies, counter))),
        "words": float(np.mean(stage.run_words(life, world_h, cur_h, H // 2, None, meta, latencies, counter, teach=False))),
        "consequences": stage.run_explore(life, world_h, cur_h, 300, None, meta, latencies, counter),
    }


def consequence_accuracy(consequences: dict) -> float | None:
    """The five gated fields of the stage's own predicate, averaged."""
    values = [consequences[n] for n in GATED if consequences.get(n) is not None]
    return float(np.mean(values)) if values else None


def run_arm(seed: int, arm_name: str, config: dict, stage_config: dict, pilot: bool) -> dict:
    """One arm of one seed: the life through the curriculum, then the held-out suites."""
    import torch

    torch.set_num_threads(int(config.get("torch_threads", 1)))
    started = time.time()
    B, H = budgets(stage_config, pilot)
    split = split_of(stage_config, [seed], pilot)
    settings = config["world"]
    reading = Reading.of(graph_spec(stage_config), **settings["reading"])
    kind = "transformer" if arm_name.startswith("transformer") else "mlp"
    network = network_of(arm_name, reading, settings[kind], capacity=config["replay"]["capacity"], seed=seed)
    wcfg = WorldConfig(**stage_config["world_config"])
    world = World(wcfg, seed=seed_for(stage.STAGE, split, seed, 0, 0), life_id=arm_name)
    cur = Curriculum(world, seed=seed_for(stage.STAGE, split, seed, 0, 1))
    brain = NetworkBrain(stage_config, seed, network)
    life = stage.Life(brain, is_agent=True)
    latencies: list[float] = []
    counter = [0]
    meta = {"seed": seed, "split": split, "life": arm_name}
    result = live(life, world, cur, B, stage_config, meta, latencies, counter)
    result["real_steps"] = counter[0]
    result["counts"] = brain.counts()
    heldout_latencies: list[float] = []
    frozen = brain.frozen()
    result["heldout"] = heldout(frozen, arm_name, seed, stage_config, world, cur, H, heldout_latencies)
    result["heldout_consequence_accuracy"] = consequence_accuracy(result["heldout"]["consequences"])
    result["consequence_accuracy"] = consequence_accuracy(result["consequences"])
    result.update(
        seed=seed,
        arm=arm_name,
        split=split,
        budget={**B, "heldout_episodes": H},
        network=network.describe(),
        latency_ms={"life": timings(latencies), "heldout": timings(heldout_latencies), "all": timings(latencies + heldout_latencies)},
        wall_seconds=time.time() - started,
    )
    return result


def _worker(payload):
    """One (seed, arm) job in a fresh process: the arguments travel with the job."""
    (seed, arm_name), args = payload
    return guarded(lambda job: run_arm(job[0], job[1], args["config"], args["stage_config"], args["pilot"]), (seed, arm_name))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--stage-config", type=Path, default=ROOT / "world" / "config.json")
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
    receipt.body["sources"]["stage_under_test"] = stage_sources(ROOT / "world")
    receipt.body["supplied_components"] = [
        "the S02 stage: the world, its tasks and rewards, the four declared stores and their write rules, the A* search over imagined consequences, the curriculum, the held-out suites and every budget",
        "the replay ring schedule",
    ]
    receipt.body["learned_components"] = [
        "the world model of each arm: an online perceptron or an online transformer",
        "the contents of the four stores, which are witnessed evidence",
    ]
    jobs = [(seed, arm_name) for seed in args.seeds for arm_name in args.arms]
    receipt.body["seeds"] = {"scheduled": args.seeds, "arms": args.arms, "split": split, "completed": [], "failed": []}
    worker_args = {"config": config, "stage_config": stage_config, "pilot": args.pilot}
    results: list[dict] = []
    for job, outcome in run_jobs(jobs, args.workers, lambda j: run_arm(j[0], j[1], config, stage_config, args.pilot), _worker, worker_args):
        if isinstance(outcome, dict):
            results.append(outcome)
            receipt.body["seeds"]["completed"].append(list(job))
            accuracy = outcome["heldout_consequence_accuracy"]
            print(f"seed {job[0]} {job[1]}: consequences {'none' if accuracy is None else format(accuracy, '.2f')} "
                  f"request32 {outcome['heldout']['remember_32']:.2f} cue8 {outcome['heldout']['cue_8']:.2f} "
                  f"words {outcome['heldout']['words']:.2f} door {outcome['door']['last_quarter_after']:.2f} "
                  f"{outcome['latency_ms']['all']['mean']:.1f} ms/decision")
        else:
            receipt.body["seeds"]["failed"].append(list(job))
            receipt.body["failures"].append({"seed": job[0], "arm": job[1], "error": outcome})
            print(f"seed {job[0]} {job[1]}: failed")
        receipt.body["metrics"]["per_arm"] = results
        receipt.write()
    if results:
        receipt.body["networks"] = {r["arm"]: r["network"] for r in results}
        receipt.body["metrics"].update({
            "consequence_accuracy": by_arm(results, lambda r: r["heldout_consequence_accuracy"]),
            "consequence_accuracy_per_field": {
                name: by_arm(results, lambda r, n=name: r["heldout"]["consequences"][n]) for name in GATED
            },
            "request_success_32": by_arm(results, lambda r: r["heldout"]["remember_32"]),
            "correction_visible": by_arm(results, lambda r: r["heldout"]["correction_visible"]),
            "cue_delay8": by_arm(results, lambda r: r["heldout"]["cue_8"]),
            "cue_delay32": by_arm(results, lambda r: r["heldout"]["cue_32"]),
            "grounding": by_arm(results, lambda r: r["heldout"]["words"]),
            "door_recovery": by_arm(results, lambda r: r["door"]["last_quarter_after"]),
            "request_success_training": {
                delay: by_arm(results, lambda r, d=delay: r["remember"][d]) for delay in ("8", "16", "32")
            },
        })
        receipt.body["counts"] = {f"seed{r['seed']}-{r['arm']}": {**r["counts"], "real_steps": r["real_steps"]} for r in results}
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
