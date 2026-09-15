"""S04 runner: one artist life per seed, its progression, its gates, its controls and its baselines.

    python run.py --config config.json --seeds 0 1 --pilot --out runs/pilot
    python run.py --config config.json --seeds 10 11 12 13 14 --out runs/acceptance

The life follows the packet's progression: scribble, then a single requested segment with one
committed intention, then stroke goals with replanning, then multi-stroke goals, then a changed
motor gain and the return to the original body, then a canvas whose ink lands offset. Every real
event goes to ``events.jsonl``; held-out measurements use isolated copies of the life. Controls:
a random scribbler, the supplied path oracle as the upper reference, an online MLP world model
under the same search, the identical architecture frozen from birth, shuffled action-outcome
pairing, a corrupted body and ink model under the intact search, and the artist with the canvas
channels removed. ``--pilot`` divides the budgets by ten.
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

from artist.brain import ArtistLife
from artist.controls import MLPArtist, OracleArtist, RandomScribbler
from artist.env import (
    DEVELOPMENT_FAMILIES,
    FAMILIES,
    PREDICTED,
    SIMPLE,
    STAGE,
    Arm,
    CanvasConfig,
    CanvasWorld,
    body_config,
    drawing,
)
from artist.paths import ensure

ensure()

from agent.brain import Agent
from agent.life import seed_for
from agent.receipt import EventLog, Receipt, peak_rss_mb, sha256_file


def geometry(config: dict, **over) -> CanvasConfig:
    c = dict(config["canvas"])
    c["centre"] = tuple(c["centre"])
    c["pen_offset"] = tuple(c.get("pen_offset", (0.0, 0.0)))
    c.update(over)
    return CanvasConfig(**c)


def build_world(config: dict, geom: CanvasConfig, seed: int, life_id: str, *, gain: float = 1.0) -> CanvasWorld:
    body = Arm(body_config(lengths=tuple(config["body"]["lengths"]), gain=gain, substeps=config["body"]["substeps"]), seed=seed_for(STAGE, "body", seed, 0, 0) % (2**32))
    return CanvasWorld(geom, body, seed=seed, life_id=life_id)


def drawings_for(split: str, seed: int, count: int, geom: CanvasConfig, families: tuple[str, ...] | None = None, offset: int = 0) -> list:
    return [drawing(split, offset + k, seed, geom, families=families) for k in range(count)]


# -- one drawing


def draw(life, world: CanvasWorld, target, log: EventLog | None, meta: dict, latencies: list[float], *, scores: dict | None = None, log_predictions: bool = False) -> dict:
    """One drawing from reset to close; returns its measures and the decisions it took."""
    m = world.reset(target)
    life.begin(world.view())
    pen_down = 0
    while True:
        t0 = time.perf_counter()
        decision = life.step(m, world.view())
        latencies.append(time.perf_counter() - t0)
        record = {
            **meta, "episode": int(m.episode_id), "event": int(m.event_id), "tick": int(m.tick),
            "feedback_for": m.feedback_for, "reward": round(float(m.reward), 6), "chamfer": round(world.discrepancy, 6),
            "terminated": bool(m.terminated), "truncated": bool(m.truncated),
            "action": None if decision is None else int(decision.action),
            "controller": None if decision is None else decision.controller,
        }
        if decision is None:
            record["final"] = {k: round(float(v), 6) for k, v in world.measure().items()}
            if log is not None:
                log.write(record)
            break
        prediction = {k: np.asarray(v, float) for k, v in decision.prediction.items()}
        m = life.feedback(world, decision)
        if prediction and scores is not None:
            for name in PREDICTED:
                if name in prediction:
                    scores["model"][name].append(float(np.mean((prediction[name] - np.asarray(m.observation[name], float)) ** 2)))
                    scores["zero"][name].append(float(np.mean(np.square(np.asarray(m.observation[name], float)))))
        if log_predictions and prediction:
            record["predicted"] = {name: [round(float(x), 6) for x in np.asarray(prediction[name], float)] for name in ("dd_hand", "mark") if name in prediction}
            record["observed"] = {name: [round(float(x), 6) for x in np.asarray(m.observation[name], float)] for name in ("dd_hand", "mark")}
        if log is not None:
            log.write(record)
        pen_down += int(decision.action) % 2
    measure = world.measure()
    return {"family": target.family, "drawing": target.index, "decisions": int(m.tick), "pen_down": pen_down, **{k: float(v) for k, v in measure.items()}}


def run_drawings(life, world: CanvasWorld, targets: list, log: EventLog | None, meta: dict, latencies: list[float], *, scores: dict | None = None, counter: list[int] | None = None, schedule: dict | None = None) -> list[dict]:
    """Every target in order, each from a fresh attachment; ``schedule`` saves the life every
    ``every`` decisions, so a run keeps the intermediate states a page can replay."""
    out = []
    for target in targets:
        life.detach()
        out.append(draw(life, world, target, log, {**meta, "drawing": target.index, "family": target.family}, latencies, scores=scores))
        if counter is not None:
            counter[0] += out[-1]["decisions"]
            if schedule is not None and counter[0] >= schedule["next"]:
                path = Path(schedule["prefix"] + f"_{counter[0]:06d}.npz")
                life.agent.save(path)
                schedule["saved"].append(path)
                schedule["next"] += schedule["every"]
    return out


def checkpoint_schedule(out: Path, seed: int, phase: str, steps: list[int], every: int, saved: list[Path]) -> dict:
    """Save the life every ``every`` decisions of a phase, named by the decisions it has lived."""
    return {"every": every, "next": steps[0] + every, "prefix": str(out / f"artist_seed{seed}_{phase}"), "saved": saved}


def empty_scores() -> dict:
    return {"model": {n: [] for n in PREDICTED}, "zero": {n: [] for n in PREDICTED}}


def gains(scores: dict, tail: int = 1000) -> dict:
    out = {}
    for name in PREDICTED:
        model, zero = scores["model"][name][-tail:], scores["zero"][name][-tail:]
        if not model:
            out[name] = {"model_mse": float("nan"), "zero_mse": float("nan"), "gain": float("nan"), "count": 0}
            continue
        m, z = float(np.mean(model)), float(np.mean(zero))
        out[name] = {"model_mse": m, "zero_mse": z, "gain": 1 - m / z if z > 0 else float("nan"), "count": len(model)}
    return out


# -- phases


def scribble(life, world: CanvasWorld, config: dict, seed: int, split: str, decisions: int, episode_length: int, log: EventLog | None, meta: dict, latencies: list[float], scores: dict, mark_log: int) -> int:
    """Bounded random choices on fresh canvases: the ink and body consequences are witnessed."""
    life.planning = False
    life.set_epsilon(1.0)
    geom = world.config
    world.config = replace(geom, max_decisions=episode_length, patience=10**9)
    done, index = 0, 0
    while done < decisions:
        target = drawing("development", 10**5 + index, seed, world.config)
        index += 1
        life.detach()
        result = draw(life, world, target, log, {**meta, "drawing": target.index, "family": target.family}, latencies, scores=scores, log_predictions=done >= decisions - mark_log)
        done += result["decisions"]
    world.config = geom
    life.detach()
    life.planning = True
    life.set_epsilon(config["actor"]["epsilon"])
    return done


def evaluate(life, world: CanvasWorld, targets: list, log: EventLog | None, meta: dict, latencies: list[float], *, mode: str = "adapting", readback: bool | None = None, agent: Agent | None = None, scores: dict | None = None, replan_every: int | None = None) -> dict:
    """Held-out drawings on an isolated copy: ``frozen`` never learns, ``adapting`` keeps learning.

    ``replan_every`` overrides the slow level's interval on the copy: a very large value commits
    one intention per drawing, which is the removal of replanning."""
    copy = life.replica(frozen=mode == "frozen", readback=readback, agent=agent) if hasattr(life, "replica") else life
    if replan_every is not None:
        copy.replan_every = replan_every
    results = run_drawings(copy, world, targets, log, {**meta, "mode": mode}, latencies, scores=scores)
    return summarise(results, mode)


LESION_DRAWINGS = 10  # a lesion is read on this many drawings, before the adapting copy repairs it


def summarise(results: list[dict], mode: str = "") -> dict:
    families = sorted({r["family"] for r in results})
    per_family = {f: {
        "chamfer": float(np.mean([r["chamfer"] for r in results if r["family"] == f])),
        "f1": float(np.mean([r["f1"] for r in results if r["family"] == f])),
        "count": int(sum(1 for r in results if r["family"] == f)),
    } for f in families}
    simple = [r for r in results if r["family"] in SIMPLE]
    return {
        "mode": mode,
        "drawings": len(results),
        "chamfer": float(np.mean([r["chamfer"] for r in results])),
        "f1": float(np.mean([r["f1"] for r in results])),
        "precision": float(np.mean([r["precision"] for r in results])),
        "recall": float(np.mean([r["recall"] for r in results])),
        "decisions": float(np.mean([r["decisions"] for r in results])),
        "max_decisions": int(max(r["decisions"] for r in results)) if results else 0,
        "chamfer_simple": float(np.mean([r["chamfer"] for r in simple])) if simple else float("nan"),
        "f1_simple": float(np.mean([r["f1"] for r in simple])) if simple else float("nan"),
        "per_family": per_family,
        "f1_first": float(np.mean([r["f1"] for r in results[:LESION_DRAWINGS]])) if results else float("nan"),
        "chamfer_first": float(np.mean([r["chamfer"] for r in results[:LESION_DRAWINGS]])) if results else float("nan"),
        "first_drawings": [r["drawing"] for r in results[:LESION_DRAWINGS]],
        "per_drawing": [{"drawing": r["drawing"], "family": r["family"], "chamfer": r["chamfer"], "f1": r["f1"], "decisions": r["decisions"]} for r in results],
    }


def paired_first(heldout: dict, control: dict) -> dict:
    """The candidate's own score on the drawings a control saw first: the paired lesion baseline."""
    ids = set(control.get("first_drawings", []))
    rows = [r for r in heldout["per_drawing"] if r["drawing"] in ids]
    if not rows:
        return control
    return {**control, "candidate_f1_first": float(np.mean([r["f1"] for r in rows])), "candidate_chamfer_first": float(np.mean([r["chamfer"] for r in rows]))}


def corrupt(path: Path, seed: int) -> Agent:
    """The trained checkpoint with its records permuted across cells and its dynamics scrambled.

    The placeholder planner satisfies the controller at load time; ``replica`` binds the real one."""
    agent = Agent.load(path, planner=lambda *_args, **_kwargs: (0, {}, {}))
    rng = np.random.default_rng(seed_for(STAGE, "corrupt", seed, 0, 0) % (2**32))
    efficacy = agent.brain.efficacy.copy()
    touching = np.isin(agent.connectome.post, agent.ports.dynamics) | np.isin(agent.connectome.pre, agent.ports.dynamics)
    efficacy[touching] = rng.permutation(efficacy[touching])
    restored = agent.brain.with_parameters(efficacy=efficacy)
    for head in agent.heads:
        head.brain = restored
    if agent.records is not None:
        order = rng.permutation(agent.records.cortex.cells)
        for name in list(agent.records.records):
            agent.records.records[name] = agent.records.records[name][order].copy()
    return agent


# -- one seed


def run_seed(seed: int, split: str, config: dict, out: Path, pilot: bool) -> dict:
    div = 10 if pilot else 1
    B = {k: max(1, v // div) for k, v in config["budget"].items()}
    B["scribble_decisions"] = max(2000, B["scribble_decisions"])
    B["scribble_episode"] = config["budget"]["scribble_episode"]
    B["heldout_drawings"] = max(10, B["heldout_drawings"])
    B["control_drawings"] = max(5, B["control_drawings"])
    B["curve_drawings"] = max(4, B["curve_drawings"])
    B["mark_log"] = config["budget"]["mark_log"] // div
    B["body_steps_cap"] = config["budget"]["body_steps_cap"]
    geom = geometry(config)
    substeps = int(config["body"]["substeps"])
    log = EventLog(out / f"events_seed{seed}.jsonl")
    latencies: list[float] = []
    result: dict = {"seed": seed, "split": split, "budget": B}
    started = time.time()
    meta = {"seed": seed, "split": split}
    checkpoints: list[Path] = []
    steps = [0]  # decisions of the candidate's own life

    life = ArtistLife(config, seed)
    world = build_world(config, geom, seed, "candidate")
    heldout = drawings_for("heldout", seed, B["heldout_drawings"], geom, FAMILIES)
    controls_set = heldout[: B["control_drawings"]]

    # 1. scribble: motor to mark and the body's consequences
    scores = empty_scores()
    steps[0] += scribble(life, world, config, seed, split, B["scribble_decisions"], B["scribble_episode"], log, {**meta, "phase": "scribble", "life": "candidate"}, latencies, scores, B["mark_log"])
    result["scribble"] = gains(scores, tail=B["mark_log"])
    result["scribble_window"] = B["mark_log"]  # the decisions the gains and the log cover
    path = out / f"artist_seed{seed}_scribble.npz"
    life.agent.save(path)
    checkpoints.append(path)

    # 2. a single requested segment: one committed intention per drawing
    curve: list[dict] = []
    life.replan_every = 10**6
    result["segment_phase"] = summarise(run_drawings(life, world, drawings_for("development", seed, B["segment_drawings"], geom, ("segment",)), log, {**meta, "phase": "segment", "life": "candidate"}, latencies, counter=steps, schedule=checkpoint_schedule(out, seed, "segment", steps, B["checkpoint_every"], checkpoints)), "training")
    life.replan_every = int(config["intention_interval"])

    # 3. stroke goals with replanning, 4. multi-stroke goals
    for phase, count, families in (("stroke", B["stroke_drawings"], SIMPLE), ("multi", B["multi_drawings"], DEVELOPMENT_FAMILIES)):
        results = []
        phase_schedule = checkpoint_schedule(out, seed, phase, steps, B["checkpoint_every"], checkpoints)
        targets = drawings_for("development", seed, count, geom, families, offset=1000 if phase == "stroke" else 2000)
        while targets:
            chunk, targets = targets[: max(1, B["curve_drawings"])], targets[max(1, B["curve_drawings"]) :]
            results += run_drawings(life, world, chunk, log, {**meta, "phase": phase, "life": "candidate"}, latencies, counter=steps, schedule=phase_schedule)
            if steps[0] // max(1, B["curve_interval"]) > len(curve):
                probe = evaluate(life, world, heldout[: B["curve_drawings"]], None, {**meta, "phase": "curve", "life": "curve"}, [], mode="frozen")
                curve.append({"decisions": steps[0], "f1": probe["f1"], "chamfer": probe["chamfer"]})
        result[f"{phase}_phase"] = summarise(results, "training")
    result["curve"] = curve
    trained = out / f"artist_seed{seed}_trained.npz"
    life.agent.save(trained)
    checkpoints.append(trained)

    # held-out suites on isolated copies
    result["heldout_frozen"] = evaluate(life, world, controls_set, log, {**meta, "phase": "heldout", "life": "candidate"}, latencies, mode="frozen")
    heldout_scores = empty_scores()
    result["heldout"] = evaluate(life, world, heldout, log, {**meta, "phase": "heldout", "life": "candidate"}, latencies, mode="adapting", scores=heldout_scores)
    result["heldout_gains"] = gains(heldout_scores, tail=2000)
    subset = {d.index for d in controls_set}
    rows = [r for r in result["heldout"]["per_drawing"] if r["drawing"] in subset]
    result["heldout_subset"] = {"drawings": len(rows), "f1": float(np.mean([r["f1"] for r in rows])), "chamfer": float(np.mean([r["chamfer"] for r in rows]))}

    # 5. a changed motor gain, then the return to the original body
    changed = build_world(config, geom, seed, "changed", gain=float(config["body"]["gain_change"]))
    run_drawings(life, changed, drawings_for("development", seed, B["perturb_drawings"], geom, DEVELOPMENT_FAMILIES, offset=3000), log, {**meta, "phase": "changed-body", "life": "candidate"}, latencies, counter=steps)
    result["changed_body"] = evaluate(life, changed, controls_set, log, {**meta, "phase": "heldout", "life": "changed"}, latencies)
    run_drawings(life, world, drawings_for("development", seed, B["return_drawings"], geom, DEVELOPMENT_FAMILIES, offset=4000), log, {**meta, "phase": "returned-body", "life": "candidate"}, latencies, counter=steps)
    result["returned_body"] = evaluate(life, world, controls_set, log, {**meta, "phase": "heldout", "life": "returned"}, latencies)

    # 6. the canvas perturbation: the ink lands offset from the pen
    offset_geometry = geometry(config, pen_offset=tuple(config["body"]["canvas_offset"]))
    offset_world = build_world(config, offset_geometry, seed, "offset")
    result["offset_before"] = evaluate(life, offset_world, controls_set, log, {**meta, "phase": "heldout", "life": "offset-before"}, latencies)
    run_drawings(life, offset_world, drawings_for("development", seed, B["offset_drawings"], geom, DEVELOPMENT_FAMILIES, offset=5000), log, {**meta, "phase": "offset-canvas", "life": "candidate"}, latencies, counter=steps)
    result["offset_after"] = evaluate(life, offset_world, controls_set, log, {**meta, "phase": "heldout", "life": "offset-after"}, latencies)
    result["offset_no_readback"] = evaluate(life, offset_world, controls_set, log, {**meta, "phase": "heldout", "life": "offset-no-readback"}, latencies, readback=False)
    final = out / f"artist_seed{seed}_final.npz"
    life.agent.save(final)
    checkpoints.append(final)

    result["ledger"] = life.agent.ledger.to_dict()
    result["parameters"] = life.agent.parameters()
    result["checkpoints"] = [{"path": p.name, "sha256": sha256_file(p)} for p in checkpoints]
    result["body_steps"] = int(steps[0] * substeps)

    # -- controls and baselines
    controls: dict = {}
    born = ArtistLife(config, seed, learning=False)
    born_world = build_world(config, geom, seed, "born-frozen")
    scribble(born, born_world, config, seed, split, max(500, B["scribble_decisions"] // 5), B["scribble_episode"], log, {**meta, "phase": "scribble", "life": "born-frozen"}, latencies, empty_scores(), 0)
    controls["born_frozen"] = evaluate(born, born_world, controls_set, log, {**meta, "phase": "heldout", "life": "born-frozen"}, latencies, mode="frozen")

    shuffled = ArtistLife(config, seed, shuffle_actions=True)
    shuffled_world = build_world(config, geom, seed, "shuffled")
    shuffle_scores = empty_scores()
    scribble(shuffled, shuffled_world, config, seed, split, B["scribble_decisions"], B["scribble_episode"], log, {**meta, "phase": "scribble", "life": "shuffled"}, latencies, shuffle_scores, 0)
    run_drawings(shuffled, shuffled_world, drawings_for("development", seed, B["stroke_drawings"], geom, SIMPLE, offset=1000), log, {**meta, "phase": "stroke", "life": "shuffled"}, latencies)
    shuffled.shuffle = False  # the pairing was broken while learning; the test drives the body honestly
    controls["shuffled_pairing"] = evaluate(shuffled, shuffled_world, controls_set, log, {**meta, "phase": "heldout", "life": "shuffled"}, latencies)
    controls["shuffled_pairing"]["scribble_gains"] = gains(shuffle_scores)

    controls["corrupted_model"] = evaluate(life, world, controls_set, log, {**meta, "phase": "heldout", "life": "corrupted"}, latencies, agent=corrupt(trained, seed))
    controls["single_intention"] = evaluate(life, world, controls_set, log, {**meta, "phase": "heldout", "life": "single-intention"}, latencies, replan_every=10**6)
    controls["readback_removed"] = {**result["offset_no_readback"], "paired_with": "offset_after"}

    random_world = build_world(config, geom, seed, "random")
    controls["random"] = summarise(run_drawings(RandomScribbler(seed_for(STAGE, "random", seed, 0, 0) % (2**32)), random_world, controls_set, log, {**meta, "phase": "heldout", "life": "random", "mode": "supplied"}, latencies), "supplied")

    oracle = OracleArtist(config)
    oracle_world = build_world(config, geom, seed, "oracle")
    oracle_results = []
    for target in controls_set:
        oracle.plan_path(target)
        oracle_results.append(draw(oracle, oracle_world, target, log, {**meta, "phase": "heldout", "life": "oracle", "mode": "supplied", "drawing": target.index, "family": target.family}, latencies))
    controls["oracle"] = summarise(oracle_results, "supplied")

    mlp = MLPArtist(config, seed, replay=config["baselines"].get("mlp_replay", 0))
    mlp_world = build_world(config, geom, seed, "mlp")
    mlp_scores = empty_scores()
    scribble(mlp, mlp_world, config, seed, split, B["scribble_decisions"], B["scribble_episode"], log, {**meta, "phase": "scribble", "life": "mlp"}, latencies, mlp_scores, 0)
    run_drawings(mlp, mlp_world, drawings_for("development", seed, B["stroke_drawings"], geom, SIMPLE, offset=1000), log, {**meta, "phase": "stroke", "life": "mlp"}, latencies)
    run_drawings(mlp, mlp_world, drawings_for("development", seed, B["multi_drawings"], geom, DEVELOPMENT_FAMILIES, offset=2000), log, {**meta, "phase": "multi", "life": "mlp"}, latencies)
    controls["online_mlp"] = summarise(run_drawings(mlp, mlp_world, controls_set, log, {**meta, "phase": "heldout", "life": "mlp", "mode": "adapting"}, latencies), "adapting")
    controls["online_mlp"]["scribble_gains"] = gains(mlp_scores)
    controls["online_mlp"]["parameters"] = mlp.model.parameters()
    controls["online_mlp"]["replay_per_transition"] = mlp.replay

    result["controls"] = {name: paired_first(result["heldout"], control) for name, control in controls.items()}
    result["latency_ms"] = {"p50": float(np.percentile(latencies, 50) * 1000), "p95": float(np.percentile(latencies, 95) * 1000), "events": len(latencies)}
    result["wall_seconds"] = time.time() - started
    result["events"] = log.close()
    (out / f"result_seed{seed}.json").write_text(json.dumps(result, sort_keys=True, indent=1, allow_nan=False, default=float) + "\n")
    return result


# -- the schedule


def _run_all(seeds, workers, run):
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
    """One seed in a fresh process: the arguments travel with the job."""
    seed, args = job
    import traceback

    try:
        return run_seed(seed, args["split"], args["config"], Path(args["out"]), args["pilot"])
    except Exception:  # noqa: BLE001
        return traceback.format_exc()


def predicates(receipt: Receipt, results: list[dict], config: dict) -> None:
    g = config["gates"]
    mean = lambda values: float(np.mean(values))
    heldout = [r["heldout"] for r in results]
    paired = [r["heldout_subset"] for r in results]  # the candidate on the drawings the controls saw
    frozen = [r["controls"]["born_frozen"] for r in results]
    chamfer_simple = [h["chamfer_simple"] for h in heldout]
    f1_simple = [h["f1_simple"] for h in heldout]
    family_f1 = [min(v["f1"] for v in h["per_family"].values()) for h in heldout]
    receipt.predicate("chamfer_simple", mean(chamfer_simple), g["chamfer_simple"], "mean over seeds of the simple families", mean(chamfer_simple) <= g["chamfer_simple"])
    receipt.predicate("f1_simple", mean(f1_simple), g["f1_simple"], "mean over seeds of the simple families", mean(f1_simple) >= g["f1_simple"])
    receipt.predicate("f1_each_family", mean(family_f1), g["f1_family"], "mean over seeds of the weakest family", mean(family_f1) >= g["f1_family"])
    born = mean([f["chamfer"] for f in frozen])
    candidate = mean([h["chamfer"] for h in paired])
    improvement = 1 - candidate / born if born > 0 else float("nan")
    receipt.predicate("chamfer_vs_born_frozen", improvement, g["chamfer_vs_frozen"], "relative improvement over the born-frozen model with the same search", improvement >= g["chamfer_vs_frozen"])
    corrupted = mean([r["controls"]["corrupted_model"]["f1_first"] for r in results])
    intact = mean([r["controls"]["corrupted_model"]["candidate_f1_first"] for r in results])
    loss = 1 - corrupted / intact if intact > 0 else float("nan")
    receipt.predicate("corrupted_model_f1_loss", loss, g["corrupt_f1_loss"], f"relative F1 lost when the learned model is permuted, over the first {LESION_DRAWINGS} drawings of the adapting copy, before it repairs itself", loss >= g["corrupt_f1_loss"], over_all_drawings=1 - mean([r["controls"]["corrupted_model"]["f1"] for r in results]) / mean([h["f1"] for h in paired]))
    recover = mean([r["changed_body"]["f1"] for r in results])
    retain = mean([r["returned_body"]["f1"] for r in results])
    receipt.predicate("changed_gain_recovery_f1", recover, g["recover_f1"], "mean over seeds after the motor-gain change", recover >= g["recover_f1"])
    receipt.predicate("returned_body_retention_f1", retain, g["retain_f1"], "mean over seeds after the return", retain >= g["retain_f1"])
    with_readback = mean([r["offset_after"]["f1"] for r in results])
    without = mean([r["offset_no_readback"]["f1"] for r in results])
    readback_loss = 1 - without / with_readback if with_readback > 0 else float("nan")
    receipt.predicate("readback_removal_loss", readback_loss, g["readback_f1_loss"], "relative F1 lost when new canvas observations are removed", readback_loss >= g["readback_f1_loss"])
    mark_gain = mean([r["heldout_gains"]["mark"]["gain"] for r in results])
    displacement = mean([r["heldout_gains"]["dd_hand"]["gain"] for r in results])
    receipt.predicate("ink_prediction_gain", mark_gain, g["mark_gain"], "mean over seeds: 1 - model/zero MSE of the ink window on held-out drawings", mark_gain >= g["mark_gain"], scribble=mean([r["scribble"]["mark"]["gain"] for r in results]))
    receipt.predicate("displacement_prediction_gain", displacement, g["displacement_gain"], "mean over seeds: 1 - model/zero MSE of the hand acceleration on held-out drawings", displacement >= g["displacement_gain"], scribble=mean([r["scribble"]["dd_hand"]["gain"] for r in results]))
    latency = float(np.max([r["latency_ms"]["p95"] for r in results]))
    receipt.predicate("latency_p95_ms", latency, g["latency_p95_ms"], "max over seeds", latency <= g["latency_p95_ms"])
    per_drawing = float(np.max([h["max_decisions"] for h in heldout])) * config["body"]["substeps"]
    receipt.predicate("steps_per_drawing", per_drawing, g["steps_per_drawing"], "max over seeds of the held-out drawings", per_drawing <= g["steps_per_drawing"])
    body_steps = float(np.max([r["body_steps"] for r in results]))
    receipt.predicate("body_steps_cap", body_steps, config["budget"]["body_steps_cap"], "max over seeds", body_steps <= config["budget"]["body_steps_cap"])


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
    receipt.body["supplied_components"] = [
        "the S01 arm body, its integrator, sensors and bounds (imported, not copied)",
        "stroke rasterisation, the canvas geometry and the pen-centred windows",
        "the discrepancy function, the foreground F1 and the capped planning objective",
        "the two-level search: a polar grid of candidate intentions and the eighteen-choice torque search, with the S01 velocity-field objective and the S01 composition of an imagined body state",
        "the intention encoding written into the goal port",
        "the target families, their splits and the drawing budget",
    ]
    receipt.body["learned_components"] = [
        "the records cortex: the ink the next decision leaves in the window around the pen, and the S01 body consequences (hand acceleration, velocity change, angle change)",
        "the workspace, dynamics and prediction synapses of the settled graph",
    ]
    receipt.body["seeds"] = {"scheduled": args.seeds, "split": split, "completed": [], "failed": []}
    _WORKER_ARGS.update(split=split, config=config, out=str(args.out), pilot=args.pilot)
    results: list[dict] = []
    for seed, outcome in _run_all(args.seeds, args.workers, lambda seed: run_seed(seed, split, config, args.out, args.pilot)):
        if isinstance(outcome, dict):
            results.append(outcome)
            receipt.body["seeds"]["completed"].append(seed)
        else:
            saved = args.out / f"result_seed{seed}.json"
            if saved.exists():  # the seed finished and its process died afterwards
                results.append(json.loads(saved.read_text()))
                receipt.body["seeds"]["completed"].append(seed)
                receipt.body["failures"].append({"seed": seed, "error": outcome, "recovered_from": saved.name})
            else:
                receipt.body["seeds"]["failed"].append(seed)
                receipt.body["failures"].append({"seed": seed, "error": outcome})
        receipt.body["metrics"]["per_seed"] = results
        receipt.write()
    if results:
        pick = lambda key, field: [r[key][field] for r in results]
        control = lambda key, field="f1": [r["controls"][key][field] for r in results]
        receipt.body["metrics"].update({
            "heldout_f1": pick("heldout", "f1"), "heldout_chamfer": pick("heldout", "chamfer"),
            "heldout_frozen_f1": pick("heldout_frozen", "f1"), "heldout_frozen_chamfer": pick("heldout_frozen", "chamfer"),
            "f1_simple": pick("heldout", "f1_simple"), "chamfer_simple": pick("heldout", "chamfer_simple"),
            "per_family": [r["heldout"]["per_family"] for r in results],
            "changed_body_f1": pick("changed_body", "f1"), "returned_body_f1": pick("returned_body", "f1"),
            "offset_before_f1": pick("offset_before", "f1"), "offset_after_f1": pick("offset_after", "f1"),
            "offset_no_readback_f1": pick("offset_no_readback", "f1"),
            "scribble_gains": [r["scribble"] for r in results],
            "heldout_gains": [r["heldout_gains"] for r in results],
            "curve": [{"seed": r["seed"], "points": r["curve"]} for r in results],
        })
        receipt.body["controls"] = {key: {"f1": control(key), "chamfer": control(key, "chamfer")} for key in results[0]["controls"]}
        receipt.body["counts"] = {
            "body_steps_per_seed": [r["body_steps"] for r in results],
            "events_logged": [r["events"]["events"] for r in results],
            "imagined_per_seed": [r["ledger"]["imagined"] for r in results],
            "record_writes": [r["ledger"]["record_writes"] for r in results],
            "real_transitions": [r["ledger"]["real_transitions"] for r in results],
        }
        receipt.body["numerics"] = {"ledger": [r["ledger"] for r in results]}
        receipt.body["resources"] = {
            "latency_ms": [r["latency_ms"] for r in results], "wall_seconds": [r["wall_seconds"] for r in results],
            "peak_rss_mb": peak_rss_mb(), "workers": args.workers, "parameters": results[0]["parameters"],
            "mlp_parameters": results[0]["controls"]["online_mlp"]["parameters"],
        }
        receipt.body["artifacts"] = {"events": [r["events"] for r in results], "checkpoints": [c for r in results for c in r["checkpoints"]]}
        predicates(receipt, results, config)
    complete = not receipt.body["seeds"]["failed"] and set(receipt.body["seeds"]["completed"]) == set(args.seeds) and (args.pilot or split == "development" or set(args.seeds) == set(config["seeds"]["acceptance"]))
    path = receipt.finish(complete)
    for p in receipt.body["acceptance"]["predicates"]:
        print(f"{'PASS' if p['passed'] else 'FAIL'} {p['name']}: {p['value']} vs {p['threshold']} ({p['aggregation']})")
    for r in results:
        h = r["heldout"]
        print(f"seed {r['seed']}: f1 {h['f1']:.2f} chamfer {h['chamfer']:.4f} simple f1 {h['f1_simple']:.2f} families " + " ".join(f"{k}={v['f1']:.2f}" for k, v in h["per_family"].items()))
        print("  controls " + " ".join(f"{k}={v['f1']:.2f}" for k, v in r["controls"].items()))
    print("status", receipt.body["status"], "->", path)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
