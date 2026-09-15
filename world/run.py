"""S02 runner: one remembered-world life per seed, its curriculum, gates and controls.

    python run.py --config config.json --seeds 0 1 --pilot --out runs/pilot
    python run.py --config config.json --seeds 10 11 12 13 14 --out runs/acceptance

Curriculum in one life: free exploration; remembered requests at delays 8, 16 and 32;
visible and invisible object moves; the cue task at delays 8 and 32; a locked-door
change mid-life; word grounding. Held-out episodes run on isolated frozen copies.
Controls: erased place records, erased word associations, shuffled action pairing,
reward-shifted cue pairing, the identical architecture frozen from birth, random
actions, the privileged exact planner, and the tabular model with the same stores and
search. ``--pilot`` divides the episode budgets by ten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from agent.brain import ReplayConfig
from agent.life import Moment, seed_for
from agent.receipt import EventLog, Receipt, peak_rss_mb
from world.baselines import (
    PrivilegedPlanner,
    RandomController,
    TabularController,
)
from world.brain import CONSEQUENCES, Brain, WorldPlanner
from world.env import (
    ACTIONS,
    DIRECTIONS,
    PASSAGE,
    World,
    WorldConfig,
)
from world.tasks import Curriculum, Episode

STAGE = "S02"


class Life:
    """A runnable life: any controller with ``step``; the candidate also carries its agent."""

    def __init__(self, controller, *, is_agent: bool, shuffle: bool = False, shift_cue: bool = False, seed: int = 0) -> None:
        self.controller, self.is_agent = controller, is_agent
        self.shuffle, self.shift_cue = shuffle, shift_cue
        self.rng = np.random.default_rng(seed_for(STAGE, "control", seed, 0, 0))

    def step(self, m: Moment):
        return self.controller.step(m)

    def act(self, world: World, d, count: int) -> Moment:
        if self.is_agent:
            if not self.shuffle:
                return world.act(d.decision_id, d.action)
            executed = int(self.rng.integers(len(ACTIONS)))
            m = world.act(d.decision_id, executed)
            return Moment(**{**m.__dict__, "executed": d.action})
        return world.act(count + 10**7, int(d))

    def action_of(self, d) -> int:
        return int(d.action) if self.is_agent else int(d)


def close_episode(life: Life, world: World, m: Moment, log: EventLog | None, meta: dict, latencies: list[float]) -> None:
    """The task ends the wander: the agent receives its last moment as truncated and closes the
    episode on its side (the world has no pending decision). The request then starts a new episode."""
    m = Moment(**{**m.__dict__, "truncated": True, "final_observation": m.observation})
    t0 = time.perf_counter()
    d = life.step(m)
    latencies.append(time.perf_counter() - t0)
    if d is not None:
        raise RuntimeError("a truncated moment must close the episode")
    if log is not None:
        log.write({**meta, "episode": m.episode_id, "event": m.event_id, "tick": m.tick, "feedback_for": m.feedback_for, "reward": m.reward,
                   "terminated": False, "truncated": True, "cell": [int(m.observation["x"].argmax()), int(m.observation["y"].argmax())],
                   "goal": int(np.asarray(m.goal).argmax()), "decision": None})


def finish_episode(life: Life, world: World, m: Moment, log: EventLog | None, meta: dict, latencies: list[float], counter: list[int]) -> tuple[Moment, dict]:
    """Step until the episode closes; returns the closing moment and what happened."""
    steps, reward = 0, 0.0
    scores = {n: [] for n in CONSEQUENCES}
    while True:
        t0 = time.perf_counter()
        d = life.step(m)
        latencies.append(time.perf_counter() - t0)
        if log is not None:
            log.write({**meta, "episode": m.episode_id, "event": m.event_id, "tick": m.tick, "feedback_for": m.feedback_for, "reward": m.reward,
                       "terminated": m.terminated, "truncated": m.truncated, "cell": [int(m.observation["x"].argmax()), int(m.observation["y"].argmax())],
                       "goal": int(np.asarray(m.goal).argmax()), "decision": None if d is None else (d.to_dict() if life.is_agent else {"action": int(d)})})
        if d is None:
            return m, {"steps": steps, "reward": reward, "terminated": m.terminated, "scores": {n: float(np.mean(v)) if v else float("nan") for n, v in scores.items()}}
        counter[0] += 1
        m2 = life.act(world, d, counter[0])
        if life.is_agent and d.prediction:
            # the object and colour of the next cell are scored only when knowable: the agent did
            # not move (no move, or a visible wall or closed door), or the destination was in view
            name = ACTIONS[int(d.action)]
            knowable = name not in DIRECTIONS or PASSAGE[int(m.observation[f"passage_{name}"].argmax())] != "open" or bool(m.observed[f"{name}_object"].all())
            for n, value in life.controller.agent.score(d.prediction, m2).items() if hasattr(life.controller, "agent") else []:
                if n.endswith("/correct") and not np.isnan(value):
                    field_name = n.split("/")[0].replace("next_", "")
                    if field_name in scores and (knowable or field_name not in ("here_object", "here_color")):
                        scores[field_name].append(value)
        steps += 1
        reward += m2.reward
        m = m2


def run_explore(life: Life, world: World, cur: Curriculum, steps: int, log, meta, latencies, counter) -> dict:
    """Free exploration until ``steps`` real steps; returns the consequence accuracy of the last 300."""
    done = 0
    scores = {n: [] for n in CONSEQUENCES}
    if life.is_agent and hasattr(life.controller, "set_epsilon"):
        life.controller.set_epsilon(1.0)
    if not life.is_agent and hasattr(life.controller, "epsilon"):
        life.controller.epsilon = 1.0
    while done < steps:
        cur.explore()
        m = world.reset()
        m, outcome = finish_episode(life, world, m, log, {**meta, "phase": "explore"}, latencies, counter)
        done += outcome["steps"]
        for n in CONSEQUENCES:
            if outcome["scores"][n] == outcome["scores"][n]:
                scores[n].append(outcome["scores"][n])
    if life.is_agent and hasattr(life.controller, "set_epsilon"):
        life.controller.set_epsilon(0.0)
    if not life.is_agent and hasattr(life.controller, "epsilon"):
        life.controller.epsilon = 0.0
    return {n: float(np.mean(scores[n][-20:])) if scores[n] else None for n in CONSEQUENCES}


def run_remember(life: Life, world: World, cur: Curriculum, episodes: int, delay: int, log, meta, latencies, counter, *, move: str | None = None, hold_cue: bool = False) -> list[float]:
    """Remembered requests: see the object at the start, wander ``delay`` steps, then be asked."""
    successes = []
    for _ in range(episodes):
        episode = cur.remember(delay)
        m = world.reset(agent=episode.target_cell)
        # the delay: exploration steps
        if hasattr(life.controller, "set_epsilon"):
            life.controller.set_epsilon(1.0)
        elif hasattr(life.controller, "epsilon"):
            life.controller.epsilon = 1.0
        closed = False
        moved = False
        for t in range(delay):
            d = life.step(m)
            if log is not None:
                log.write({**meta, "phase": f"remember-{delay}{'-' + move if move else ''}", "episode": m.episode_id, "event": m.event_id, "tick": m.tick, "feedback_for": m.feedback_for, "reward": m.reward,
                           "terminated": m.terminated, "truncated": m.truncated, "cell": [int(m.observation["x"].argmax()), int(m.observation["y"].argmax())], "goal": int(np.asarray(m.goal).argmax()),
                           "decision": None if d is None else (d.to_dict() if life.is_agent else {"action": int(d)})})
            if d is None:
                closed = True
                break
            counter[0] += 1
            m = life.act(world, d, counter[0])
            if move == "visible" and not moved and t >= delay // 2 and world.object_at(world.state["agent"]) is None:
                # onto the agent's own empty cell, in view before its next decision: the first
                # step from the middle of the delay on where the agent stands on an empty cell
                cur.move_object(episode, visible=True)
                obs, observed = world.observation()
                m = Moment(**{**m.__dict__, "observation": obs, "observed": observed})
                moved = True
            if move == "invisible" and t == delay // 2:
                cur.move_object(episode, visible=False)
        if hasattr(life.controller, "set_epsilon"):
            life.controller.set_epsilon(0.0)
        elif hasattr(life.controller, "epsilon"):
            life.controller.epsilon = 0.0
        if closed:
            successes.append(0.0)
            continue
        phase = {**meta, "phase": f"remember-{delay}{'-' + move if move else ''}"}
        close_episode(life, world, m, log, phase, latencies)
        if getattr(life.controller, "erase_places_at_request", False):
            life.controller.places.erase()  # the erasure control: the records of this history are gone when the request comes
        m = cur.request(episode)
        m, outcome = finish_episode(life, world, m, log, {**phase, "request": True, "target": list(episode.target_cell)}, latencies, counter)
        successes.append(float(outcome["terminated"]))
    return successes


def run_cue(life: Life, world: World, cur: Curriculum, episodes: int, delay: int, log, meta, latencies, counter, *, shift: bool = False, decision_epsilon: float = 0.0) -> list[float]:
    """Cue at the junction, wander, then the two-choice decision at the junction in a new episode.
    ``decision_epsilon``: the exploration of the decision (the life's own during training, none held out)."""
    correct = []
    junction = world.config.junction
    for _ in range(episodes):
        episode = cur.cue(delay)
        m = world.reset(agent=junction)
        if hasattr(life.controller, "set_epsilon"):
            life.controller.set_epsilon(1.0)
        elif hasattr(life.controller, "epsilon"):
            life.controller.epsilon = 1.0
        closed = False
        for t in range(delay):
            d = life.step(m)
            if d is None:
                closed = True
                break
            counter[0] += 1
            if t == 0:
                cur.hide_cue()
            m = life.act(world, d, counter[0])
        if hasattr(life.controller, "set_epsilon"):
            life.controller.set_epsilon(decision_epsilon)
        elif hasattr(life.controller, "epsilon"):
            life.controller.epsilon = decision_epsilon
        if closed:
            correct.append(0.0)  # the wander ran out the clock: a wasted episode, counted as wrong
            continue
        phase = {**meta, "phase": f"cue-{delay}"}
        close_episode(life, world, m, log, phase, latencies)
        m = cur.arm_cue(episode, cur_rewarded_shift(cur, episode) if shift else None)
        m, outcome = finish_episode(life, world, m, log, phase, latencies, counter)
        correct.append(float(outcome["reward"] >= 1.0))
    return correct


def cur_rewarded_shift(cur: Curriculum, episode: Episode):
    """The reward-shifted control: the room paid does not follow the cue (drawn independently)."""
    from world.tasks import ROOMS

    return ROOMS[int(cur.rng.integers(2))]


def run_words(life: Life, world: World, cur: Curriculum, episodes: int, log, meta, latencies, counter, *, teach_steps: int = 24, teach: bool = True) -> list[float]:
    """Each episode: ``teach_steps`` of exploration (under joint attention when ``teach``), then
    the word as a request. Held-out suites run with ``teach`` off: the word must already be known."""
    successes = []
    for _ in range(episodes):
        episode = cur.word(teach=teach)
        m = world.reset(agent=episode.old_cell)
        if hasattr(life.controller, "set_epsilon"):
            life.controller.set_epsilon(1.0)
        elif hasattr(life.controller, "epsilon"):
            life.controller.epsilon = 1.0
        closed = False
        for _t in range(teach_steps):
            d = life.step(m)
            if d is None:
                closed = True
                break
            counter[0] += 1
            m = life.act(world, d, counter[0])
        if hasattr(life.controller, "set_epsilon"):
            life.controller.set_epsilon(0.0)
        elif hasattr(life.controller, "epsilon"):
            life.controller.epsilon = 0.0
        if closed:
            successes.append(0.0)
            continue
        close_episode(life, world, m, log, {**meta, "phase": "word"}, latencies)
        m = cur.name_request(episode)
        m, outcome = finish_episode(life, world, m, log, {**meta, "phase": "word", "request": True, "target": list(episode.target_cell)}, latencies, counter)
        successes.append(float(outcome["terminated"]))
    return successes


def run_seed(seed: int, split: str, config: dict, out: Path, pilot: bool) -> dict:
    div = 10 if pilot else 1
    B = {k: max(2, v // div) for k, v in config["budget"].items()}
    B["explore_steps"] = max(200, B["explore_steps"])
    log = EventLog(out / f"events_seed{seed}.jsonl")
    latencies: list[float] = []
    counter = [0]
    t0 = time.time()
    result: dict = {"seed": seed, "split": split, "budget": B}
    wcfg = WorldConfig(**config["world_config"])
    world = World(wcfg, seed=seed_for(STAGE, split, seed, 0, 0), life_id="candidate")
    cur = Curriculum(world, seed=seed_for(STAGE, split, seed, 0, 1))
    brain = Brain(config, seed)
    brain.agent.config = replace(brain.agent.config, replay=ReplayConfig(**config["replay"]))
    life = Life(brain, is_agent=True)
    meta = {"seed": seed, "split": split, "life": "candidate"}
    # 1. explore
    result["consequences"] = run_explore(life, world, cur, B["explore_steps"], log, meta, latencies, counter)
    # 2. remember at delays 8, 16, 32
    result["remember"] = {}
    for delay in (8, 16, 32):
        succ = run_remember(life, world, cur, B["remember_episodes"] // 3, delay, log, meta, latencies, counter)
        result["remember"][str(delay)] = float(np.mean(succ))
    # 3. corrections
    result["correction_visible"] = float(np.mean(run_remember(life, world, cur, B["remember_episodes"] // 4, 16, log, meta, latencies, counter, move="visible")))
    result["correction_invisible"] = float(np.mean(run_remember(life, world, cur, B["remember_episodes"] // 4, 16, log, meta, latencies, counter, move="invisible")))
    # 4. cue at delay 8 then 32
    result["cue"] = {}
    for delay in (8, 32):
        correct = run_cue(life, world, cur, B["cue_episodes"] // 2, delay, log, meta, latencies, counter, decision_epsilon=config["actor"]["epsilon"])
        result["cue"][str(delay)] = {"all": float(np.mean(correct)), "last_third": float(np.mean(correct[-max(1, len(correct) // 3):]))}
    # 5. locked doors mid-life: navigation success before, right after, after 100 trials
    before = run_remember(life, world, cur, B["door_trials"] // 2, 8, log, meta, latencies, counter)
    cur.lock_doors()
    after = run_remember(life, world, cur, B["door_trials"], 8, log, meta, latencies, counter)
    result["door"] = {"before": float(np.mean(before)), "first_quarter_after": float(np.mean(after[: max(1, len(after) // 4)])), "last_quarter_after": float(np.mean(after[-max(1, len(after) // 4):]))}
    # 6. words
    words = run_words(life, world, cur, B["word_episodes"], log, meta, latencies, counter)
    result["words_training_last_quarter"] = float(np.mean(words[-max(1, len(words) // 4):]))
    result["real_steps"] = counter[0]
    result["ledger"] = brain.agent.ledger.to_dict()
    result["parameters"] = brain.agent.parameters()
    result["stores"] = {"place_writes": brain.places.writes, "word_writes": brain.words.writes}
    brain.agent.save(out / f"world_seed{seed}.npz")
    # held-out suites on frozen copies, with interventions on copies
    H = max(10, B["heldout_episodes"] // div)

    def heldout(controller_factory, name: str, *, is_agent: bool) -> dict:
        world_h = World(wcfg, seed=seed_for(STAGE, "heldout", seed, 2000, 0), life_id=f"heldout-{name}")
        world_h._edges, world_h._doors, world_h._colors, world_h._word_of = dict(world.state["edges"]), set(world._doors), world._colors.copy(), world._word_of.copy()
        world_h.set_door_rule(world.state["door_rule"])
        cur_h = Curriculum(world_h, seed=seed_for(STAGE, "heldout", seed, 2000, 1))
        cur_h.cue_rule = cur.cue_rule
        c = controller_factory()
        lh = Life(c, is_agent=is_agent)
        cnt = [10**6]
        lat: list[float] = []
        m_ = {"seed": seed, "split": "heldout", "life": f"heldout-{name}"}
        out_h = {
            "remember_32": float(np.mean(run_remember(lh, world_h, cur_h, H, 32, log, m_, lat, cnt))),
            "correction_visible": float(np.mean(run_remember(lh, world_h, cur_h, H // 2, 16, log, m_, lat, cnt, move="visible"))),
            "cue_8": float(np.mean(run_cue(lh, world_h, cur_h, H // 2, 8, log, m_, lat, cnt))),
            "cue_32": float(np.mean(run_cue(lh, world_h, cur_h, H // 2, 32, log, m_, lat, cnt))),
            "words": float(np.mean(run_words(lh, world_h, cur_h, H // 2, log, m_, lat, cnt, teach=False))),
            "consequences": run_explore(lh, world_h, cur_h, 300, log, m_, lat, cnt),
        }
        return out_h

    result["heldout"] = heldout(lambda: brain.frozen(), "candidate", is_agent=True)

    def erased_places():
        f = brain.frozen(); f.places.erase(); f.erase_places_at_request = True; return f

    def erased_words():
        f = brain.frozen(); f.words.erase(); return f

    controls = {
        "erased_places": heldout(erased_places, "erased-places", is_agent=True),
        "erased_words": heldout(erased_words, "erased-words", is_agent=True),
        "random": heldout(lambda: RandomController(seed), "random", is_agent=False),
    }
    # the privileged planner reads the held-out world it runs in: build it inside the factory via a closure over a fresh world
    privileged_world = {}

    def privileged():
        return _PrivilegedProxy(privileged_world, cur.cue_rule)

    controls["privileged"] = heldout(privileged, "privileged", is_agent=False)
    # tabular model with the same stores and search, learning through the same curriculum on its own life
    world_t = World(wcfg, seed=seed_for(STAGE, split, seed, 0, 0), life_id="tabular")
    cur_t = Curriculum(world_t, seed=seed_for(STAGE, split, seed, 0, 1))
    tab = TabularController(seed, WorldPlanner(**config.get("planner", {})))
    lt = Life(tab, is_agent=False)
    mt = {"seed": seed, "split": split, "life": "tabular"}
    ct = [2 * 10**6]
    run_explore(lt, world_t, cur_t, B["explore_steps"], log, mt, latencies, ct)
    for delay in (8, 16, 32):
        run_remember(lt, world_t, cur_t, B["remember_episodes"] // 3, delay, log, mt, latencies, ct)
    run_cue(lt, world_t, cur_t, B["cue_episodes"] // 2, 8, log, mt, latencies, ct, decision_epsilon=config["actor"]["epsilon"])
    run_words(lt, world_t, cur_t, B["word_episodes"], log, mt, latencies, ct)
    tab.learning = False
    controls["tabular"] = heldout(lambda: tab, "tabular", is_agent=False)
    # born frozen: the same architecture with learning off, run through the same curriculum
    frozen_life = Brain(config, seed, learning=False)
    world_f = World(wcfg, seed=seed_for(STAGE, split, seed, 0, 0), life_id="born-frozen")
    cur_f = Curriculum(world_f, seed=seed_for(STAGE, split, seed, 0, 1))
    lf = Life(frozen_life, is_agent=True)
    mf = {"seed": seed, "split": split, "life": "born-frozen"}
    cf = [3 * 10**6]
    run_explore(lf, world_f, cur_f, B["explore_steps"] // 4, log, mf, latencies, cf)
    controls["born_frozen"] = heldout(lambda: frozen_life.frozen(), "born-frozen", is_agent=True)
    # shuffled action pairing: the world head learns from mismatched action labels
    shuffled_brain = Brain(config, seed)
    shuffled_brain.agent.config = replace(shuffled_brain.agent.config, replay=ReplayConfig(**config["replay"]))
    world_s = World(wcfg, seed=seed_for(STAGE, split, seed, 0, 0), life_id="shuffled")
    cur_s = Curriculum(world_s, seed=seed_for(STAGE, split, seed, 0, 1))
    ls = Life(shuffled_brain, is_agent=True, shuffle=True, seed=seed)
    ms_ = {"seed": seed, "split": split, "life": "shuffled"}
    cs = [4 * 10**6]
    run_explore(ls, world_s, cur_s, B["explore_steps"], log, ms_, latencies, cs)
    controls["shuffled_pairing"] = heldout(lambda: shuffled_brain.frozen(), "shuffled", is_agent=True)
    result["controls"] = controls
    result["latency_ms"] = {"p50": float(np.percentile(latencies, 50) * 1000), "p95": float(np.percentile(latencies, 95) * 1000), "events": len(latencies)}
    result["wall_seconds"] = time.time() - t0
    result["events"] = log.close()
    return result


class _PrivilegedProxy:
    """The upper reference must read the world it is evaluated in; the harness binds it lazily."""

    def __init__(self, holder: dict, cue_rule: int) -> None:
        self.holder, self.cue_rule = holder, cue_rule
        self.planner = None

    def step(self, m: Moment):
        if self.planner is None or self.planner.world.life_id != m.life_id:
            self.planner = PrivilegedPlanner(_WORLDS[m.life_id], self.cue_rule)
        return self.planner.step(m)


_WORLDS: dict[str, World] = {}

_original_world_init = World.__post_init__


def _registering_init(self) -> None:
    _original_world_init(self)
    _WORLDS[self.life_id] = self


World.__post_init__ = _registering_init  # evaluation instrumentation: worlds register by life id


def _run_all(seeds, workers, run):
    """Run the seeds in order, or in ``workers`` processes; yields (seed, result dict | error text)."""
    import traceback

    def guarded(seed):
        try:
            return run(seed)
        except Exception:  # noqa: BLE001  a failed trial is reported, never hidden
            return traceback.format_exc()

    if workers <= 1 or len(seeds) <= 1:
        for seed in seeds:
            yield seed, guarded(seed)
        return
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(workers, len(seeds))) as pool:
        outcomes = list(pool.map(_worker, [(seed, _WORKER_ARGS) for seed in seeds]))
    for seed, outcome in zip(seeds, outcomes, strict=True):
        yield seed, outcome


_WORKER_ARGS: dict = {}


def _worker(job):
    """One seed in a fresh process: the arguments travel with the job (spawned workers start empty)."""
    seed, args = job
    import traceback

    try:
        return run_seed(seed, args['split'], args['config'], Path(args['out']), args['pilot'])
    except Exception:  # noqa: BLE001
        return traceback.format_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="seeds run in this many processes")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    split = "development" if args.pilot or all(s in config["seeds"]["development"] for s in args.seeds) else "acceptance"
    run_id = f"{STAGE}-{split}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    receipt = Receipt(STAGE, run_id, config, HERE, args.out)
    receipt.body["supplied_components"] = ["world dynamics, tasks and reward rules", "categorical sensor encoding with missingness flags", "declared stores with categorical keys (object-place, cue, word co-occurrence) and their write rules", "best-first search over the learned model and one-step reward greed", "replay ring schedule", "settling schedule and rejection guards"]
    receipt.body["learned_components"] = ["workspace and dynamics synapses and the eleven prediction heads (ten consequences and reward)", "the contents of the three stores (witnessed evidence)"]
    receipt.body["seeds"] = {"scheduled": args.seeds, "split": split, "completed": [], "failed": []}
    _WORKER_ARGS.update(split=split, config=config, out=str(args.out), pilot=args.pilot)
    results = []
    for seed, outcome in _run_all(args.seeds, args.workers, lambda seed: run_seed(seed, split, config, args.out, args.pilot)):
        if isinstance(outcome, dict):
            results.append(outcome)
            receipt.body["seeds"]["completed"].append(seed)
        else:  # a failed trial still gets a receipt
            receipt.body["seeds"]["failed"].append(seed)
            receipt.body["failures"].append({"seed": seed, "error": outcome})
        receipt.body["metrics"]["per_seed"] = results
        receipt.write()
    g = config["gates"]
    if results:
        H = [r["heldout"] for r in results]
        cons = [float(np.mean([h["consequences"][n] for n in ("dx", "dy", "outcome", "here_object", "carrying") if h["consequences"][n] is not None])) for h in H]
        req = [h["remember_32"] for h in H]
        erased = [r["controls"]["erased_places"]["remember_32"] for r in results]
        corr = [h["correction_visible"] for h in H]
        cue8 = [h["cue_8"] for h in H]
        words = [h["words"] for h in H]
        erased_w = [r["controls"]["erased_words"]["words"] for r in results]
        door_after = [r["door"]["last_quarter_after"] for r in results]
        shuffled = [r["controls"]["shuffled_pairing"]["remember_32"] for r in results]
        receipt.body["metrics"].update({"consequence_accuracy": cons, "request_success_32": req, "correction_visible": corr, "cue_delay8": cue8, "grounding": words, "door_recovery": door_after})
        receipt.body["controls"] = {k: [r["controls"][k] for r in results] for k in results[0]["controls"]}
        receipt.body["counts"] = {"real_steps_per_seed": [r["real_steps"] for r in results], "events_logged": [r["events"]["events"] for r in results], "imagined_per_seed": [r["ledger"]["imagined"] for r in results], "replay_writes": [r["ledger"]["replay_writes"] for r in results]}
        receipt.body["numerics"] = {"ledger": [r["ledger"] for r in results]}
        receipt.body["resources"] = {"latency_ms": [r["latency_ms"] for r in results], "wall_seconds": [r["wall_seconds"] for r in results], "peak_rss_mb": peak_rss_mb(), "parameters": results[0]["parameters"]}
        receipt.body["artifacts"] = {"events": [r["events"] for r in results], "checkpoints": [f"world_seed{r['seed']}.npz" for r in results]}
        receipt.predicate("consequence_accuracy", float(np.mean(cons)), g["consequence_accuracy"], "mean over seeds of five visible consequences", np.mean(cons) >= g["consequence_accuracy"])
        receipt.predicate("request_success_delay32", float(np.mean(req)), g["request_success"], "mean over seeds", np.mean(req) >= g["request_success"])
        receipt.predicate("erased_places_loss", float(np.mean(req) - np.mean(erased)), g["erasure_loss_points"], "mean success difference", np.mean(req) - np.mean(erased) >= g["erasure_loss_points"])
        receipt.predicate("correction_visible", float(np.mean(corr)), g["correction_success"], "mean over seeds", np.mean(corr) >= g["correction_success"])
        receipt.predicate("door_recovery", float(np.mean(door_after)), g["door_recovery"], "mean over seeds, last quarter of 100 trials", np.mean(door_after) >= g["door_recovery"])
        receipt.predicate("shuffled_pairing_gap", float(np.mean(req) - np.mean(shuffled)), g["shuffled_gap_points"], "mean success difference", np.mean(req) - np.mean(shuffled) >= g["shuffled_gap_points"])
        receipt.predicate("grounding", float(np.mean(words)), g["grounding"], "mean over seeds", np.mean(words) >= g["grounding"])
        receipt.predicate("erased_words_ceiling", float(np.mean(erased_w)), g["erased_words_ceiling"], "mean over seeds", np.mean(erased_w) <= g["erased_words_ceiling"])
        receipt.predicate("cue_delay8", float(np.mean(cue8)), g["cue_delay8"], "mean over seeds", np.mean(cue8) >= g["cue_delay8"])
        receipt.predicate("steps_cap", float(np.max([r["real_steps"] for r in results])), config["budget"]["steps_cap"], "max over seeds", np.max([r["real_steps"] for r in results]) <= config["budget"]["steps_cap"])
    complete = not receipt.body["seeds"]["failed"] and set(receipt.body["seeds"]["completed"]) == set(args.seeds) and (args.pilot or split == "development" or set(args.seeds) == set(config["seeds"]["acceptance"]))
    path = receipt.finish(complete)
    for p in receipt.body["acceptance"]["predicates"]:
        print(f"{'PASS' if p['passed'] else 'FAIL'} {p['name']}: {p['value']} vs {p['threshold']} ({p['aggregation']})")
    for r in results:
        print(f"seed {r['seed']}: heldout {json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r['heldout'].items() if k != 'consequences'})} controls " + " ".join(f"{k}:req32={v['remember_32']:.2f}" for k, v in r["controls"].items()))
    print("status", receipt.body["status"], "->", path)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
