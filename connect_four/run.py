"""S03 runner: one Connect Four life per seed, its schedule, gates and controls.

    python connect_four/run.py --seeds 0 --pilot --out runs/connect_four/pilot
    python connect_four/run.py --seeds 0 1 --workers 2 --out runs/connect_four/dev
    python connect_four/run.py --seeds 10 11 12 13 14 --workers 2 --out runs/connect_four/acceptance
    python connect_four/run.py --controls-only --seeds 0 --out runs/connect_four/controls

Schedule per seed (``brain_interface.md``): random legal games against the random opponent
with epsilon 1; games against the mixture (a snapshot of this life when the mixture starts and
every ``snapshot_every`` games); a checkpoint; on frozen copies the validity and terminal gates,
the tactical suite and paired games against random, one-ply and minimax-256; the no-planning
copy (the snapshot policy) on the tactical suite and the paired games; the corrupted copy on
the tactical suite and the validity gate; games against the changed mixture without a reset; a
second checkpoint and the paired games again on a frozen copy; the identical architecture frozen
from birth. ``--pilot`` divides the game budgets, the paired games and the snapshot interval by
ten. ``--controls-only`` calibrates the evaluator with the supplied controls and no candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from agent.life import seed_for
from agent.receipt import EventLog, Receipt, peak_rss_mb, sha256_file, sha256_json
from connect_four.brain import Brain
from connect_four.controls import MinimaxPolicy, RandomPolicy
from connect_four.env import STAGE, GameConfig, World, cells_of
from connect_four.evaluate import (
    KINDS,
    SIDES,
    TERMINAL_CLASSES,
    heldout_moves,
    judge,
    paired_games,
    relative_string,
    tactical_suite,
    terminal_boards,
)
from connect_four.opponents import OnePlyOpponent, OpponentMixture, RandomOpponent

PAIRED = ("random", "one_ply", "minimax_256")
OLD_OPPONENTS = ("random", "one_ply")
RUNNER_SUPPLIED = [
    "the curriculum: random legal games, the opponent mixture with snapshots of this life, the changed mixture",
    "the opponents (random, one-ply, minimax at 256 nodes) and the independent evaluator with its held-out suites",
]


def budgets(config: dict, pilot: bool) -> dict:
    """The game budgets of one life; the pilot divides the game budgets, the paired games and
    the snapshot interval by ten, the diagnostic suites keep their size."""
    out = dict(config["budget"])
    if pilot:
        for name in ("explore_games", "mixture_games", "adaptation_games", "games_cap", "paired_games", "snapshot_every", "curve_block"):
            out[name] = max(1, int(out[name]) // 10)
    return out


def checkpoint_schedule(config: dict, pilot: bool) -> tuple[set[int], set[str]]:
    """The declared checkpoints of a life: game counts (divided by ten in the pilot) and the ends
    of phases; the ends of the mixture and of the adaptation are always saved."""
    section = config.get("checkpoints", {})
    games = {max(1, int(g) // (10 if pilot else 1)) for g in section.get("games", [])}
    return games, set(section.get("phases", [])) | {"mixture", "adaptation"}


def opponent_factory(name: str, game: GameConfig) -> Callable[[int], object]:
    if name == "random":
        return RandomOpponent
    if name == "one_ply":
        return lambda s: OnePlyOpponent(s, game)
    if name.startswith("minimax_"):
        return lambda s: MinimaxPolicy(int(name.split("_")[1]), s, game)
    raise ValueError(f"unknown opponent {name!r}")


# -- games


def play_brain(brain: Brain, world: World, first: str, opponent, latencies: list[float], decided: set[bytes] | None = None) -> dict:
    """One game of ``brain`` in ``world``: every moment goes to ``step``; decision moments are timed."""
    m = world.reset(first=first, opponent=opponent)
    first_event = m.event_id
    controllers: list[str] = []
    expansions = depth = 0
    while True:
        timed = m.any_legal
        t0 = time.perf_counter()
        d = brain.step(m)
        if timed:
            latencies.append(time.perf_counter() - t0)
        if d is None:
            if timed:
                raise RuntimeError("a decision moment produced no decision")
            break
        if decided is not None:
            decided.add(world.position().key())
        controllers.append(d.controller[0])
        expansions += int(d.budget.get("expansions", 0))
        depth += int(d.budget.get("depth", 0))
        mid = world.act(d.decision_id, d.action)
        if brain.step(mid) is not None:
            raise RuntimeError("the intermediate moment produced a decision")
        if mid.terminated:
            break
        m = world.reply()
    state = world.state()
    return {"moves": state["moves"], "outcome": int(world.outcome()), "controllers": "".join(controllers), "expansions": expansions,
            "depth_sum": depth, "events": [first_event, state["event"] - 1]}


def play_policy(policy, world: World, first: str, opponent) -> dict:
    """One game of a one-shot policy ``(cells, legal) -> column`` for the candidate's side."""
    m = world.reset(first=first, opponent=opponent)
    count = 0
    while not m.terminated:
        column = int(policy(cells_of(m.observation, world.config), np.asarray(m.action_mask)))
        mid = world.act(count, column)
        count += 1
        if mid.terminated:
            break
        m = world.reply()
    return {"moves": world.state()["moves"], "outcome": int(world.outcome())}


def score_of(outcomes: list[int]) -> float:
    return float(np.mean((np.asarray(outcomes, float) + 1) / 2)) if outcomes else float("nan")


# -- evaluation


def suite_digest(suite) -> str:
    return hashlib.sha256(b"".join(case.position.key() for case in suite)).hexdigest()


def tactical_brain(brain: Brain, game: GameConfig, suite, log: EventLog, meta: dict, latencies: list[float]) -> dict:
    """The tactical suite on a frozen copy: start at each position, decide, close the probe."""
    world = World(game, life_id=f"{meta['life']}-tactical-{meta['seed']}")
    brain.new_world()
    hits: dict[tuple[str, str], list[bool]] = {(k, s): [] for k in KINDS for s in SIDES}
    for i, case in enumerate(suite):
        m = world.start_from(case.position)
        t0 = time.perf_counter()
        d = brain.step(m)
        latencies.append(time.perf_counter() - t0)
        hits[(case.kind, case.side)].append(judge(case, d.action))
        mid = world.act(d.decision_id, d.action)
        brain.step(mid)
        if not mid.terminated:
            brain.step(world.truncate())
        log.write({**meta, "phase": "tactical", "index": i, "board": relative_string(case.position.cells), "to_move": case.position.to_move,
                   "kind": case.kind, "column": int(d.action), "controller": d.controller})
    return tactical_summary(hits)


def tactical_policy(policy, game: GameConfig, suite, log: EventLog, meta: dict) -> dict:
    hits: dict[tuple[str, str], list[bool]] = {(k, s): [] for k in KINDS for s in SIDES}
    for i, case in enumerate(suite):
        board = case.position.board(game)
        column = int(policy(board.relative(case.candidate), board.legal_mask()))
        hits[(case.kind, case.side)].append(judge(case, column))
        log.write({**meta, "phase": "tactical", "index": i, "board": relative_string(case.position.cells), "to_move": case.position.to_move,
                   "kind": case.kind, "column": column})
    return tactical_summary(hits)


def tactical_summary(hits: dict[tuple[str, str], list[bool]]) -> dict:
    every = [h for v in hits.values() for h in v]
    return {"success": float(np.mean(every)), "cases": len(every), "strata": {f"{k}-{s}": float(np.mean(v)) for (k, s), v in hits.items() if v}}


def paired_brain(brain: Brain, game: GameConfig, config: dict, seed: int, games: int, log: EventLog, meta: dict, latencies: list[float], opponents=PAIRED) -> dict:
    out = {}
    for name in opponents:
        world = World(game, life_id=f"{meta['life']}-{meta['phase']}-{name}-{seed}")
        brain.new_world()

        def play(k: int, first: str, opponent, name=name, world=world) -> int:
            record = play_brain(brain, world, first, opponent, latencies)
            log.write({**meta, "opponent": name, "k": k, "first": first, "moves": record["moves"], "outcome": record["outcome"]})
            return record["outcome"]

        out[name] = paired_games(play, games, opponent_factory(name, game), seed, int(config["evaluation"]["paired_environments"].get(name, 2003)))
    return out


def paired_policy(policy, game: GameConfig, config: dict, seed: int, games: int, log: EventLog, meta: dict, opponents=PAIRED) -> dict:
    out = {}
    for name in opponents:
        world = World(game, life_id=f"{meta['life']}-{meta['phase']}-{name}-{seed}")

        def play(k: int, first: str, opponent, name=name, world=world) -> int:
            record = play_policy(policy, world, first, opponent)
            log.write({**meta, "opponent": name, "k": k, "first": first, "moves": record["moves"], "outcome": record["outcome"]})
            return record["outcome"]

        out[name] = paired_games(play, games, opponent_factory(name, game), seed, int(config["evaluation"]["paired_environments"].get(name, 2003)))
    return out


def validity_gate(brain: Brain, game: GameConfig, moves) -> dict:
    """Exact next boards on held-out single moves, the changed cell separately, per stratum."""
    exact, changed = [], []
    strata: dict[str, list[bool]] = {}
    for move in moves:
        obs, goal = move.inputs(game)
        predicted = np.asarray(brain.predict_board(obs, move.column, goal=goal))
        expected = np.asarray(move.expected)
        ok = bool(np.array_equal(predicted, expected))
        where = move.changed(game)
        exact.append(ok)
        changed.append(bool(np.array_equal(predicted[where], expected[where])))
        strata.setdefault(f"h{move.height(game)}-{'candidate' if move.mover_is_candidate else 'opponent'}", []).append(ok)
    return {"exact": float(np.mean(exact)), "changed_cell": float(np.mean(changed)), "moves": len(exact),
            "strata": {k: float(np.mean(v)) for k, v in sorted(strata.items())}}


def terminal_gate(brain: Brain, game: GameConfig, cases) -> dict:
    predicted = np.array([int(np.argmax(brain.predict_terminal(*case.inputs(game)[:1], goal=case.inputs(game)[1]))) for case in cases])
    labels = np.array([case.label for case in cases])
    recall = {name: float(np.mean(predicted[labels == k] == k)) for k, name in enumerate(TERMINAL_CLASSES) if (labels == k).any()}
    return {"balanced_accuracy": float(np.mean(list(recall.values()))), "recall": recall, "boards": len(cases)}


# -- one life


def train(brain: Brain, world: World, games: int, opponent_for: Callable[[int], tuple[str, object]], phase: str, counter: dict, log: EventLog,
          meta: dict, latencies: list[float], decided: set[bytes] | None, curves: list[dict], block: int, before_game: Callable[[int], None] | None = None,
          after_game: Callable[[int], None] | None = None) -> None:
    rows: list[dict] = []
    for g in range(games):
        if before_game is not None:
            before_game(g)
        name, opponent = opponent_for(g)
        first = "candidate" if counter["games"] % 2 == 0 else "opponent"
        record = play_brain(brain, world, first, opponent, latencies, decided)
        log.write({**meta, "phase": phase, "game": counter["games"], "first": first, "opponent": name, **record})
        counter["games"] += 1
        counter["decisions"] += len(record["controllers"])
        if after_game is not None:
            after_game(counter["games"])
        rows.append({"opponent": name, **record})
        if len(rows) == block or g == games - 1:
            outcomes = [r["outcome"] for r in rows]
            by = {}
            for r in rows:
                by.setdefault(r["opponent"], []).append(r["outcome"])
            decisions = sum(len(r["controllers"]) for r in rows)
            planned = sum(r["controllers"].count("p") for r in rows)
            state = brain.state()
            curves.append({"phase": phase, "games": counter["games"], "score": score_of(outcomes), "by_opponent": {k: score_of(v) for k, v in by.items()},
                           "decisions": decisions, "planned": planned, "expansions_per_planned": sum(r["expansions"] for r in rows) / max(1, planned),
                           "depth_per_planned": sum(r["depth_sum"] for r in rows) / max(1, planned), "validity": state["validity"], "extended": state["extended"],
                           "invalid": state["planner"]["invalid"]})
            rows = []


def run_seed(seed: int, split: str, config: dict, out: Path, pilot: bool) -> dict:
    t_start = time.time()
    game = GameConfig(**config["game"])
    B = budgets(config, pilot)
    log = EventLog(out / f"events_seed{seed}.jsonl")
    latencies: list[float] = []
    result: dict = {"seed": seed, "split": split, "kind": "candidate", "budget": B}
    brain = Brain(config, seed)
    world = World(game, life_id=f"candidate-{seed}")
    decided: set[bytes] = set()
    curves: list[dict] = []
    counter = {"games": 0, "decisions": 0}
    meta = {"seed": seed, "life": "candidate"}
    training = config["training"]
    scheduled, phase_ends = checkpoint_schedule(config, pilot)
    checkpoints: list[dict] = []

    def save(label: str) -> None:
        """A checkpoint between games, named by the life's game count; the saved agent has epsilon 0."""
        games = counter["games"]
        for entry in checkpoints:
            if entry["games"] == games:
                entry["labels"].append(label)
                return
        epsilon = brain.agent.config.actor.epsilon
        brain.set_epsilon(0.0)
        files = brain.save(out / f"brain_seed{seed}_games{games:06d}")
        brain.set_epsilon(epsilon)
        checkpoints.append({"seed": seed, "games": games, "labels": [label], "files": [{"path": f.name, "sha256": sha256_file(f), "bytes": f.stat().st_size} for f in files]})

    def scheduled_save(games: int) -> None:
        if games in scheduled:
            save("scheduled")

    # 1. random legal games against the random opponent
    brain.set_epsilon(training["explore_epsilon"])
    explorer = RandomOpponent(seed_for(STAGE, split, seed, 0, 0))
    train(brain, world, B["explore_games"], lambda g: ("random", explorer), "explore", counter, log, meta, latencies, decided, curves, B["curve_block"], after_game=scheduled_save)
    result["after_explore"] = brain.state()
    if "explore" in phase_ends:
        save("after_explore")

    # 2. the mixture, with snapshots of this life
    brain.set_epsilon(training["epsilon"])
    mixture = OpponentMixture(config["mixture"], {"random": RandomOpponent(seed_for(STAGE, split, seed, 0, 1)), "one_ply": OnePlyOpponent(seed_for(STAGE, split, seed, 0, 2), game)},
                              seed_for(STAGE, split, seed, 0, 3), pool=B["snapshot_pool"])

    def snapshot(g: int) -> None:
        if g % B["snapshot_every"] == 0:
            mixture.add_snapshot(brain.snapshot_policy(), f"snapshot_{mixture.added}")

    train(brain, world, B["mixture_games"], lambda g: mixture.new_game(), "mixture", counter, log, meta, latencies, decided, curves, B["curve_block"], snapshot, scheduled_save)
    result["mixture_draws"] = dict(mixture.draws)
    result["snapshots_added"] = mixture.added

    # 3. checkpoint
    brain.set_epsilon(0.0)
    save("after_mixture")
    result["after_mixture"] = brain.state()
    result["ledger_mixture"] = brain.agent.ledger.to_dict()

    # 4. frozen copies
    moves = heldout_moves(seed, game, n=config["evaluation"]["heldout_moves"])
    cases = terminal_boards(seed, game, n=config["evaluation"]["terminal_boards"])
    suite = tactical_suite(seed, game, n=config["evaluation"]["tactical_positions"], exclude=decided)
    result["suites"] = {"tactical_sha256": suite_digest(suite), "heldout_moves_sha256": hashlib.sha256(b"".join(m.position.key() + bytes([m.column, m.candidate]) for m in moves)).hexdigest(),
                        "terminal_sha256": hashlib.sha256(b"".join(c.position.key() + bytes([c.candidate]) for c in cases)).hexdigest(), "excluded_positions": len(decided)}
    copy = brain.frozen()
    result["validity"] = validity_gate(copy, game, moves)
    result["terminal"] = terminal_gate(copy, game, cases)
    result["tactical"] = tactical_brain(copy, game, suite, log, {**meta, "life": "candidate"}, latencies)
    result["paired_mixture"] = paired_brain(copy, game, config, seed, B["paired_games"], log, {**meta, "life": "candidate", "phase": "paired-mixture"}, latencies)
    result["planner_eval"] = copy.planner.stats()
    no_planning = brain.snapshot_policy()
    result["no_planning"] = {"tactical": tactical_policy(no_planning, game, suite, log, {**meta, "life": "no-planning"}),
                             "paired": paired_policy(no_planning, game, config, seed, B["paired_games"], log, {**meta, "life": "no-planning", "phase": "paired-mixture"})}
    corrupted = brain.frozen()
    corrupted.corrupt_dynamics()
    result["corrupted"] = {"tactical": tactical_brain(corrupted, game, suite, log, {**meta, "life": "corrupted"}, []), "validity": validity_gate(corrupted, game, moves),
                           "planner": corrupted.planner.stats()}
    result["planning_gain"] = result["paired_mixture"]["one_ply"]["score"] - result["no_planning"]["paired"]["one_ply"]["score"]
    result["corrupted_loss"] = result["tactical"]["success"] - result["corrupted"]["tactical"]["success"]

    # 5. the changed mixture, the same life
    brain.set_epsilon(training["epsilon"])
    changed = OpponentMixture(config["adaptation_mixture"], {"random": RandomOpponent(seed_for(STAGE, split, seed, 0, 4)), "one_ply": OnePlyOpponent(seed_for(STAGE, split, seed, 0, 5), game),
                                                            "minimax_256": MinimaxPolicy(256, seed_for(STAGE, split, seed, 0, 6), game)}, seed_for(STAGE, split, seed, 0, 7), pool=B["snapshot_pool"])
    train(brain, world, B["adaptation_games"], lambda g: changed.new_game(), "adaptation", counter, log, meta, latencies, None, curves, B["curve_block"], after_game=scheduled_save)
    result["adaptation_draws"] = dict(changed.draws)
    brain.set_epsilon(0.0)
    save("final")
    after = brain.frozen()
    result["paired_adaptation"] = paired_brain(after, game, config, seed, B["paired_games"], log, {**meta, "life": "candidate", "phase": "paired-adaptation"}, latencies)
    result["old_loss"] = {name: result["paired_mixture"][name]["score"] - result["paired_adaptation"][name]["score"] for name in OLD_OPPONENTS}
    result["minimax_256_before_after"] = [result["paired_mixture"]["minimax_256"]["score"], result["paired_adaptation"]["minimax_256"]["score"]]
    result["after_adaptation"] = brain.state()
    result["games"] = {"explore": B["explore_games"], "mixture": B["mixture_games"], "adaptation": B["adaptation_games"], "total": counter["games"]}
    result["decisions"] = counter["decisions"]
    result["curves"] = curves
    result["ledger"] = brain.agent.ledger.to_dict()
    result["parameters"] = brain.parameters()
    result["latency_ms"] = {"p50": float(np.percentile(latencies, 50) * 1000), "p95": float(np.percentile(latencies, 95) * 1000),
                            "max": float(np.max(latencies) * 1000), "decisions": len(latencies)}

    # 6. the identical architecture frozen from birth, after a quarter of the exploration games
    born = Brain(config, seed, learning=False)
    born_world = World(game, life_id=f"born-frozen-{seed}")
    born.set_epsilon(training["explore_epsilon"])
    born_games = int(B["explore_games"] * config["controls"]["born_frozen_explore_fraction"])
    born_counter = {"games": 0, "decisions": 0}
    born_opponent = RandomOpponent(seed_for(STAGE, split, seed, 0, 0))  # the exploration stream the candidate met, from its start
    train(born, born_world, born_games, lambda g: ("random", born_opponent), "explore", born_counter, log, {**meta, "life": "born-frozen"}, [], None, [], max(1, born_games))
    born.set_epsilon(0.0)
    born_copy = born.frozen()
    result["born_frozen"] = {"games": born_games, "tactical": tactical_brain(born_copy, game, suite, log, {**meta, "life": "born-frozen"}, []),
                             "paired": paired_brain(born_copy, game, config, seed, B["paired_games"], log, {**meta, "life": "born-frozen", "phase": "paired-mixture"}, [], OLD_OPPONENTS),
                             "validity": validity_gate(born_copy, game, moves)}
    result["checkpoints"] = checkpoints
    result["wall_seconds"] = time.time() - t_start
    result["events"] = log.close()
    return result


def run_controls(seed: int, split: str, config: dict, out: Path, pilot: bool) -> dict:
    """The supplied controls on the same suites and paired schedule, without a candidate."""
    t_start = time.time()
    game = GameConfig(**config["game"])
    B = budgets(config, pilot)
    log = EventLog(out / f"events_seed{seed}.jsonl")
    suite = tactical_suite(seed, game, n=config["evaluation"]["tactical_positions"])
    result: dict = {"seed": seed, "split": split, "kind": "controls", "budget": B, "suites": {"tactical_sha256": suite_digest(suite)}, "controls": {}}
    policies = {"random": lambda: RandomPolicy(seed_for(STAGE, "control", seed, 0, 0)), "one_ply": lambda: OnePlyOpponent(seed_for(STAGE, "control", seed, 0, 1), game)}
    for budget in config["controls"]["minimax_budgets"]:
        policies[f"minimax_{budget}"] = lambda budget=budget: MinimaxPolicy(int(budget), seed_for(STAGE, "control", seed, 0, int(budget)), game)
    for name, make in policies.items():
        policy = make()
        meta = {"seed": seed, "life": name}
        entry = {"tactical": tactical_policy(policy, game, suite, log, meta),
                 "paired": paired_policy(policy, game, config, seed, B["paired_games"], log, {**meta, "phase": "paired-mixture"}, OLD_OPPONENTS)}
        if hasattr(policy, "stats"):
            entry["search"] = policy.stats()
        result["controls"][name] = entry
    result["wall_seconds"] = time.time() - t_start
    result["events"] = log.close()
    return result


# -- the run


def outcomes_of(jobs: list[tuple], workers: int):
    """One outcome per job, in order, in this process or in at most ``workers`` of them."""
    if workers <= 1 or len(jobs) <= 1:
        yield from map(run_guarded, jobs)
        return
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        yield from pool.map(run_guarded, jobs)


def run_guarded(job: tuple) -> dict | str:
    """One seed in this or a fresh process; a failed trial returns its traceback."""
    seed, split, config, out, pilot, controls = job
    try:
        return (run_controls if controls else run_seed)(seed, split, config, Path(out), pilot)
    except Exception:  # noqa: BLE001  a failed trial is reported, never hidden
        return traceback.format_exc()


def candidate_predicates(receipt: Receipt, results: list[dict], config: dict, B: dict) -> None:
    g = config["gates"]
    seeds = [r["seed"] for r in results]

    def per(values) -> dict:
        return {"per_seed": dict(zip([str(s) for s in seeds], [float(v) for v in values], strict=True))}

    validity = [r["validity"]["exact"] for r in results]
    terminal = [r["terminal"]["balanced_accuracy"] for r in results]
    tactical = [r["tactical"]["success"] for r in results]
    random_scores = [r["paired_mixture"]["random"]["score"] for r in results]
    one_ply = [r["paired_mixture"]["one_ply"]["score"] for r in results]
    gain = [r["planning_gain"] for r in results]
    corrupted = [r["corrupted_loss"] for r in results]
    old = {name: float(np.mean([r["old_loss"][name] for r in results])) for name in OLD_OPPONENTS}
    worst = max(old, key=old.get)
    p95 = [r["latency_ms"]["p95"] for r in results]
    total = [r["games"]["total"] for r in results]
    receipt.predicate("next_board_validity", float(np.mean(validity)), g["next_board_validity"], "mean over seeds of exact next boards on held-out single moves", np.mean(validity) >= g["next_board_validity"], **per(validity))
    receipt.predicate("terminal_prediction", float(np.mean(terminal)), g["terminal_prediction"], "mean over seeds of balanced accuracy over four classes", np.mean(terminal) >= g["terminal_prediction"], **per(terminal))
    receipt.predicate("tactical_success", float(np.mean(tactical)), g["tactical_success"], "mean over seeds, judged by the evaluator", np.mean(tactical) >= g["tactical_success"], **per(tactical))
    receipt.predicate("paired_random", float(np.mean(random_scores)), g["paired_random"], "mean over seeds of the paired score", np.mean(random_scores) >= g["paired_random"], **per(random_scores))
    receipt.predicate("paired_random_seed_floor", float(np.min(random_scores)), g["paired_random_seed_floor"], "minimum over seeds", np.min(random_scores) >= g["paired_random_seed_floor"], **per(random_scores))
    receipt.predicate("paired_one_ply", float(np.mean(one_ply)), g["paired_one_ply"], "mean over seeds of the paired score", np.mean(one_ply) >= g["paired_one_ply"], **per(one_ply))
    receipt.predicate("paired_one_ply_seed_floor", float(np.min(one_ply)), g["paired_one_ply_seed_floor"], "minimum over seeds", np.min(one_ply) >= g["paired_one_ply_seed_floor"], **per(one_ply))
    receipt.predicate("planning_gain", float(np.mean(gain)), g["planning_gain"], "mean over seeds of search minus the snapshot policy, one-ply paired schedule", np.mean(gain) >= g["planning_gain"], **per(gain))
    receipt.predicate("corrupted_dynamics_loss", float(np.mean(corrupted)), g["corrupted_dynamics_loss"], "mean over seeds of tactical success minus the corrupted copy's", np.mean(corrupted) >= g["corrupted_dynamics_loss"], **per(corrupted))
    receipt.predicate("continual_old_loss", old[worst], g["continual_old_loss"], "the larger mean over seeds of the paired-score loss on random and on one-ply after the adaptation games", old[worst] <= g["continual_old_loss"], opponent=worst, losses=old)
    receipt.predicate("latency_p95_ms", float(np.max(p95)), g["latency_p95_ms"], "maximum over seeds of the p95 decision latency, training and evaluation", np.max(p95) <= g["latency_p95_ms"], **per(p95))
    receipt.predicate("games_cap", float(np.max(total)), B["games_cap"], "maximum over seeds of the games of one life", np.max(total) <= B["games_cap"], **per(total))


def controls_predicates(receipt: Receipt, results: list[dict], config: dict) -> None:
    g = config["gates"]
    minimax = [min(r["controls"][f"minimax_{b}"]["tactical"]["success"] for b in config["controls"]["minimax_budgets"]) for r in results]
    random_tactical = [r["controls"]["random"]["tactical"]["success"] for r in results]
    one_ply = [r["controls"]["one_ply"]["paired"]["random"]["score"] for r in results]
    band = [abs(r["controls"]["random"]["paired"]["random"]["score"] - 0.5) for r in results]
    receipt.predicate("calibration_control_tactical", float(np.min(minimax)), g["calibration_control_tactical"], "minimum over seeds and minimax budgets of tactical success", np.min(minimax) >= g["calibration_control_tactical"])
    receipt.predicate("calibration_random_tactical_ceiling", float(np.max(random_tactical)), g["calibration_random_tactical_ceiling"], "maximum over seeds of the random policy's tactical success", np.max(random_tactical) <= g["calibration_random_tactical_ceiling"])
    receipt.predicate("calibration_one_ply_beats_random", float(np.min(one_ply)), g["calibration_one_ply_beats_random"], "minimum over seeds of the one-ply paired score against random", np.min(one_ply) >= g["calibration_one_ply_beats_random"])
    receipt.predicate("calibration_random_vs_random_band", float(np.max(band)), g["calibration_random_vs_random_band"], "maximum over seeds of the distance of random against random from 0.5", np.max(band) <= g["calibration_random_vs_random_band"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="seeds run in this many processes")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    split = "development" if args.pilot or all(s in config["seeds"]["development"] for s in args.seeds) else "acceptance"
    kind = "controls" if args.controls_only else "candidate"
    run_id = f"{STAGE}-{kind}-{split}{'-pilot' if args.pilot else ''}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    receipt = Receipt(STAGE, run_id, config, HERE, args.out)
    B = budgets(config, args.pilot)
    receipt.body["run_kind"] = kind
    receipt.body["pilot"] = bool(args.pilot)
    receipt.body["supplied_components"] = (["the supplied-rules controls: random, one-ply, minimax at the matched node budgets"] if args.controls_only else list(Brain.SUPPLIED)) + RUNNER_SUPPLIED
    receipt.body["learned_components"] = [] if args.controls_only else list(Brain.LEARNED)
    receipt.body["seeds"] = {"scheduled": args.seeds, "split": split, "completed": [], "failed": []}
    receipt.body["acceptance"]["spec_sha256"] = sha256_json({"gates": config["gates"], "budget": config["budget"], "evaluation": config["evaluation"], "kind": kind, "pilot": args.pilot})
    jobs = [(seed, split, config, str(args.out), args.pilot, args.controls_only) for seed in args.seeds]
    results: list[dict] = []
    for job, outcome in zip(jobs, outcomes_of(jobs, args.workers), strict=True):
        seed = job[0]
        if isinstance(outcome, dict):
            results.append(outcome)
            receipt.body["seeds"]["completed"].append(seed)
        else:
            receipt.body["seeds"]["failed"].append(seed)
            receipt.body["failures"].append({"seed": seed, "error": outcome})
            print(outcome, file=sys.stderr)
        receipt.body["metrics"]["per_seed"] = results
        receipt.write()
    if results:
        receipt.body["data_manifest_sha256"] = sha256_json([r["suites"] for r in results])
        receipt.body["artifacts"] = {"events": [r["events"] for r in results], "checkpoints": [c for r in results for c in r.get("checkpoints", [])]}
        receipt.body["resources"] = {"wall_seconds_per_seed": [r["wall_seconds"] for r in results], "peak_rss_mb": peak_rss_mb(), "workers": args.workers}
        if args.controls_only:
            receipt.body["controls"] = {r["seed"]: r["controls"] for r in results}
            controls_predicates(receipt, results, config)
        else:
            receipt.body["metrics"].update({
                "next_board_validity": [r["validity"]["exact"] for r in results], "changed_cell_accuracy": [r["validity"]["changed_cell"] for r in results],
                "terminal_prediction": [r["terminal"]["balanced_accuracy"] for r in results], "tactical_success": [r["tactical"]["success"] for r in results],
                "paired_mixture": [{k: v["score"] for k, v in r["paired_mixture"].items()} for r in results],
                "paired_adaptation": [{k: v["score"] for k, v in r["paired_adaptation"].items()} for r in results],
                "planning_gain": [r["planning_gain"] for r in results], "corrupted_dynamics_loss": [r["corrupted_loss"] for r in results],
            })
            receipt.body["controls"] = {"no_planning": [r["no_planning"] for r in results], "corrupted": [r["corrupted"] for r in results], "born_frozen": [r["born_frozen"] for r in results]}
            receipt.body["counts"] = {"games_per_seed": [r["games"] for r in results], "decisions_per_seed": [r["decisions"] for r in results],
                                      "real_transitions_per_seed": [r["after_adaptation"]["transitions"] for r in results],
                                      "imagined_transitions_per_seed": [r["after_adaptation"]["planner"]["nodes"] for r in results],
                                      "record_writes_per_seed": [{k: r["after_adaptation"][k] for k in ("drop_writes", "line_writes", "value_writes")} for r in results],
                                      "replay_writes": [r["ledger"]["replay_writes"] for r in results], "events_logged": [r["events"]["events"] for r in results]}
            receipt.body["numerics"] = {"ledger": [r["ledger"] for r in results]}
            receipt.body["resources"].update({"latency_ms": [r["latency_ms"] for r in results], "parameters": results[0]["parameters"]})
            candidate_predicates(receipt, results, config, B)
    complete = not receipt.body["seeds"]["failed"] and set(receipt.body["seeds"]["completed"]) == set(args.seeds) and (
        args.pilot or split == "development" or args.controls_only or set(args.seeds) == set(config["seeds"]["acceptance"]))
    path = receipt.finish(complete)
    for p in receipt.body["acceptance"]["predicates"]:
        print(f"{'PASS' if p['passed'] else 'FAIL'} {p['name']}: {p['value']} vs {p['threshold']} ({p['aggregation']})")
    print("status", receipt.body["status"], "passed", receipt.body["acceptance"]["passed"], "->", path)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
