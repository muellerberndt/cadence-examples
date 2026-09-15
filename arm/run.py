"""S01 runner: one arm life per seed, its gates, its controls and its baselines.

    python run.py --config config.json --seeds 0 1 --pilot --out runs/pilot
    python run.py --config config.json --seeds 10 11 12 13 14 --out runs/acceptance

Life: motor babbling (random torques), reaching with the learned-model planner, the arm
copier on fresh moving targets, then a changed body without any reset and the return to
the original body. Every real event goes to ``events.jsonl``; held-out measurements use
isolated frozen copies. Controls: random torques, the supplied Jacobian PD controller,
an online MLP forward model with the same planner and replay budget, the identical
architecture frozen from birth, shuffled action-transition pairing, and a corrupted
world model (the dynamics synapses and the records permuted across cells) under the
intact planner. ``--pilot`` divides the budgets by ten.
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

from agent.brain import Agent, ReplayConfig
from agent.life import Moment, seed_for
from agent.receipt import EventLog, Receipt, peak_rss_mb, sha256_file
from arm.brain import (
    PREDICTED,
    JacobianPD,
    ModelController,
    ModelPlanner,
    OnlineMLP,
    RandomController,
    agent_config,
)
from arm.env import Arm, ArmConfig, moving_path

STAGE = "S01"


def arm_config(config: dict, **over) -> ArmConfig:
    a = dict(config["arm"])
    a["lengths"] = tuple(a["lengths"])
    a["target_radius"] = tuple(a["target_radius"])
    a.update(over)
    return ArmConfig(**a)


class CadenceLife:
    """The candidate: the S00 agent with the arm's planner; ``step`` returns the action."""

    def __init__(self, config: dict, seed: int, *, learning: bool = True, replay: bool = True, shuffle_actions: bool = False) -> None:
        planner = ModelPlanner(**config["planner"])
        cfg = agent_config(config, learning=learning)
        if replay and config.get("replay"):
            cfg = replace(cfg, replay=ReplayConfig(**config["replay"]))
        self.agent = Agent(cfg, seed=seed, planner=planner.for_agent())
        self.shuffle = shuffle_actions
        self.shuffle_rng = np.random.default_rng(seed_for(STAGE, "shuffle", seed, 0, 0))
        self.decision = None
        self.last_prediction = None

    def set_epsilon(self, epsilon: float) -> None:
        self.agent.config = replace(self.agent.config, actor=replace(self.agent.config.actor, epsilon=epsilon))

    def step(self, moment: Moment):
        d = self.agent.step(moment)
        self.decision = d
        self.last_prediction = None if d is None else d.prediction
        return d

    def feedback(self, env: Arm, d) -> Moment:
        """Execute the committed action; a shuffled-pairing control feeds back a different executed action's outcome."""
        if not self.shuffle:
            return env.act(d.decision_id, d.action)
        executed = int(self.shuffle_rng.integers(9))
        m = env.act(d.decision_id, executed)
        # the outcome of another torque is reported as this decision's outcome: the pairing is broken
        return Moment(**{**m.__dict__, "executed": d.action})


def run_life(life, env: Arm, decisions: int, log: EventLog | None, seed: int, split: str, phase: str, latencies: list[float], *, is_agent: bool, m: Moment | None = None, save_at: dict[int, Path] | None = None) -> tuple[Moment, dict]:
    """Run ``decisions`` decisions of one life on ``env`` (continuing from ``m``); returns the last moment and per-episode outcomes.
    ``save_at`` maps a decision count to a checkpoint path: the brain is saved when the count is reached."""
    outcomes = {"success": [], "final_distance": [], "decisions": [], "prediction_mse": {n: [] for n in PREDICTED}, "zero_mse": {n: [] for n in PREDICTED}}
    if m is None:
        m = env.reset()
    count = 0
    while count < decisions:
        t0 = time.perf_counter()
        d = life.step(m)
        latencies.append(time.perf_counter() - t0)
        if log is not None:
            record = {"seed": seed, "split": split, "phase": phase, "life": env.life_id, "episode": m.episode_id, "event": m.event_id, "tick": m.tick,
                      "feedback_for": m.feedback_for, "reward": m.reward, "terminated": m.terminated, "truncated": m.truncated,
                      "hand": m.observation["hand"].tolist(), "target": (np.asarray(m.goal) * 2 - 1).tolist(), "goal": np.asarray(m.goal).tolist(),
                      "dd_hand": m.observation["dd_hand"].tolist(), "d_velocity": m.observation["d_velocity"].tolist(), "d_angles": m.observation["d_angles"].tolist(),
                      "decision": None if d is None else (d.to_dict() if is_agent else {"action": int(d), "prediction": {k: np.asarray(v).tolist() for k, v in (life.last_prediction or {}).items()}})}
            log.write(record)
        if d is None:
            outcomes["success"].append(float(m.terminated))
            outcomes["final_distance"].append(float(np.linalg.norm(m.observation["hand"] - (np.asarray(m.goal) * 2 - 1))))
            outcomes["decisions"].append(int(m.tick))
            m = env.reset()
            continue
        action = d.action if is_agent else int(d)
        prediction = d.prediction if is_agent else life.last_prediction
        m = life.feedback(env, d) if hasattr(life, "feedback") else env.act(count + 10**7, action)
        if prediction:
            for n in PREDICTED:
                outcomes["prediction_mse"][n].append(float(np.mean((np.asarray(prediction[n]) - m.observation[n]) ** 2)))
                outcomes["zero_mse"][n].append(float(np.mean(m.observation[n] ** 2)))
        count += 1
        if save_at and count in save_at and is_agent:
            life.agent.save(save_at[count])
    return m, outcomes


def heldout_targets(seed: int, count: int, arm: ArmConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """Held-out (start angles, target) pairs balanced over four quadrants and two radius bands."""
    rng = np.random.default_rng(seed_for(STAGE, "heldout", seed, 2000, 0))
    out = []
    for k in range(count):
        quadrant, band = k % 4, (k // 4) % 2
        r = rng.uniform(*((arm.target_radius[0], 0.575) if band == 0 else (0.575, arm.target_radius[1])))
        a = rng.uniform(quadrant * np.pi / 2, (quadrant + 1) * np.pi / 2) - np.pi
        out.append((rng.uniform(-np.pi, np.pi, 2), np.array([r * np.cos(a), r * np.sin(a)])))
    return out


def evaluate_reaching(controller, arm: ArmConfig, targets, log, seed, split, latencies, *, is_agent: bool, name: str, mode: str = "adapting") -> dict:
    """Success on held-out static targets. ``mode`` "frozen": a read-only copy; "adapting": a copy
    that keeps learning with exploration off, one target after another (the ongoing mode)."""
    if is_agent:
        copy_ = controller.agent.frozen() if mode == "frozen" else controller.agent.clone()
        copy_.config = replace(copy_.config, actor=replace(copy_.config.actor, epsilon=0.0))
    succ, dist, steps = [], [], []
    for k, (theta, target) in enumerate(targets):
        env = Arm(arm, seed=seed_for(STAGE, "heldout", seed, 2000 + k, 1), life_id=f"heldout-{name}-{mode}-{k}")
        if is_agent:
            copy_.abandon(0)
            copy_.new_stream(0)
            life = copy_
        else:
            life = controller
        m = env.reset(theta=theta, target=target)
        while True:
            t0 = time.perf_counter()
            d = life.step(m)
            latencies.append(time.perf_counter() - t0)
            if log is not None:
                log.write({"seed": seed, "split": split, "phase": "heldout", "mode": mode, "life": env.life_id, "episode": m.episode_id, "event": m.event_id, "tick": m.tick,
                           "terminated": m.terminated, "truncated": m.truncated, "hand": m.observation["hand"].tolist(), "target": target.tolist(),
                           "decision": None if d is None else {"action": int(d.action if is_agent else d)}})
            if d is None:
                break
            m = env.act(d.decision_id if is_agent else k * 1000 + m.tick, d.action if is_agent else int(d))
        succ.append(float(m.terminated)); dist.append(float(np.linalg.norm(m.observation["hand"] - target))); steps.append(int(m.tick))
    return {"success": float(np.mean(succ)), "success_first50": float(np.mean(succ[:50])), "final_distance": float(np.mean(dist)), "decisions": float(np.mean(steps)), "per_target": succ, "mode": mode}


def evaluate_copier(controller, arm: ArmConfig, seed: int, paths: int, decisions: int, log, split, latencies, *, is_agent: bool, name: str, approach: int = 50) -> dict:
    """Tracking RMSE on fresh moving targets, per family, through a frozen copy; the arm starts
    at a random configuration, so the first ``approach`` decisions (5 s) are excluded."""
    if is_agent:
        copy_ = controller.agent.clone()  # the adapting copy, exploration off
        copy_.config = replace(copy_.config, actor=replace(copy_.config.actor, epsilon=0.0))
    families = ("line", "circle", "combination")
    rmse = {f: [] for f in families}
    for k in range(paths):
        family = families[k % 3]
        rng = np.random.default_rng(seed_for(STAGE, "copier", seed, 3000 + k, 0))
        path = moving_path(family, rng)
        env = Arm(arm, seed=seed_for(STAGE, "copier", seed, 3000 + k, 1), life_id=f"copier-{name}-{k}", moving_target=path)
        if is_agent:
            copy_.abandon(0)
            copy_.new_stream(0)
            life = copy_
        else:
            life = controller
        m = env.reset()
        errors = []
        for t in range(decisions + approach):
            t0 = time.perf_counter()
            d = life.step(m)
            latencies.append(time.perf_counter() - t0)
            if d is None:
                break
            m = env.act(d.decision_id if is_agent else k * 1000 + t, d.action if is_agent else int(d))
            errors.append(float(np.linalg.norm(m.observation["hand"] - (np.asarray(m.goal) * 2 - 1))))
        rmse[family].append(float(np.sqrt(np.mean(np.square(errors[approach:])))))  # tracking after the approach
    return {"rmse": float(np.mean([v for f in families for v in rmse[f]])), "per_family": {f: float(np.mean(rmse[f])) for f in families}, "approach_decisions": approach}


def run_seed(seed: int, split: str, config: dict, out: Path, pilot: bool) -> dict:
    div = 10 if pilot else 1
    B = {k: max(1, v // div) for k, v in config["budget"].items()}
    B["heldout_targets"] = max(8, B["heldout_targets"])
    B["copier_paths"] = max(3, B["copier_paths"])
    B["copier_decisions"] = config["budget"]["copier_decisions"]
    arm = arm_config(config)
    log = EventLog(out / f"events_seed{seed}.jsonl")
    latencies: list[float] = []
    result: dict = {"seed": seed, "split": split, "budget": B}
    t0 = time.time()
    planner_cfg = config["planner"]
    # -- the candidate life
    life = CadenceLife(config, seed)
    env = Arm(arm, seed=seed_for(STAGE, split, seed, 0, 0), life_id="candidate")
    life.set_epsilon(1.0)
    checkpoints: list[Path] = []

    def schedule(phase: str, total: int, points: tuple[float, ...]) -> dict[int, Path]:
        """Checkpoints at the given fractions of a phase's decisions (the intermediate states a page offers)."""
        saves = {}
        for fraction in points:
            n = max(1, round(total * fraction))
            saves[n] = out / f"arm_seed{seed}_{phase}_{n:05d}.npz"
            checkpoints.append(saves[n])
        return saves

    m, babble = run_life(life, env, B["babbling_decisions"], log, seed, split, "babbling", latencies, is_agent=True, save_at=schedule("babbling", B["babbling_decisions"], (0.1, 0.4, 1.0)))
    result["babbling"] = {n: {"model_mse_last500": float(np.mean(babble["prediction_mse"][n][-500:])), "zero_mse_last500": float(np.mean(babble["zero_mse"][n][-500:]))} for n in PREDICTED}
    life.set_epsilon(config["actor"]["epsilon"])
    m, reach = run_life(life, env, B["reaching_decisions"], log, seed, split, "reaching", latencies, is_agent=True, m=m, save_at=schedule("reaching", B["reaching_decisions"], (0.25, 0.5)))
    result["reaching_training"] = {"episodes": len(reach["success"]), "success_last20": float(np.mean(reach["success"][-20:])) if reach["success"] else float("nan")}
    targets = heldout_targets(seed, B["heldout_targets"], arm)
    result["heldout_frozen"] = evaluate_reaching(life, arm, targets, log, seed, split, latencies, is_agent=True, name="candidate", mode="frozen")
    result["heldout"] = evaluate_reaching(life, arm, targets, log, seed, split, latencies, is_agent=True, name="candidate", mode="adapting")
    life.agent.save(out / f"arm_seed{seed}_reaching.npz")
    checkpoints.append(out / f"arm_seed{seed}_reaching.npz")
    result["copier"] = evaluate_copier(life, arm, seed, B["copier_paths"], B["copier_decisions"], log, split, latencies, is_agent=True, name="candidate")
    # -- adaptation: a longer upper link, no reset; then the original body again
    changed = arm_config(config, lengths=(arm.lengths[0] * 1.15, arm.lengths[1]))
    env.config = changed
    m, _ = run_life(life, env, B["adaptation_decisions"], log, seed, split, "changed-body", latencies, is_agent=True, m=m)
    result["changed_body"] = evaluate_reaching(life, changed, targets, log, seed, split, latencies, is_agent=True, name="changed")
    env.config = arm
    m, _ = run_life(life, env, B["return_decisions"], log, seed, split, "returned-body", latencies, is_agent=True, m=m)
    result["returned_body"] = evaluate_reaching(life, arm, targets, log, seed, split, latencies, is_agent=True, name="returned")
    result["ledger"] = life.agent.ledger.to_dict()
    result["parameters"] = life.agent.parameters()
    life.agent.save(out / f"arm_seed{seed}_final.npz")
    checkpoints.append(out / f"arm_seed{seed}_final.npz")
    result["checkpoints"] = [{"path": p.name, "sha256": sha256_file(p)} for p in checkpoints]
    # -- interventions on the reaching checkpoint
    controls: dict = {}
    frozen_birth = CadenceLife(config, seed, learning=False)
    env_f = Arm(arm, seed=seed_for(STAGE, split, seed, 0, 0), life_id="born-frozen")
    frozen_birth.set_epsilon(1.0)
    run_life(frozen_birth, env_f, B["babbling_decisions"] // 5, log, seed, split, "babbling", latencies, is_agent=True)
    frozen_birth.set_epsilon(config["actor"]["epsilon"])
    controls["born_frozen"] = evaluate_reaching(frozen_birth, arm, targets, None, seed, split, latencies, is_agent=True, name="born-frozen", mode="frozen")
    controls["born_frozen"]["copier_rmse"] = evaluate_copier(frozen_birth, arm, seed, B["copier_paths"], B["copier_decisions"], None, split, latencies, is_agent=True, name="born-frozen")["rmse"]
    shuffled = CadenceLife(config, seed, shuffle_actions=True)
    env_s = Arm(arm, seed=seed_for(STAGE, split, seed, 0, 0), life_id="shuffled")
    shuffled.set_epsilon(1.0)
    m_s, sh = run_life(shuffled, env_s, B["babbling_decisions"], log, seed, split, "babbling", latencies, is_agent=True)
    shuffled.set_epsilon(config["actor"]["epsilon"])
    m_s, _ = run_life(shuffled, env_s, B["reaching_decisions"] // 2, log, seed, split, "reaching", latencies, is_agent=True, m=m_s)
    controls["shuffled_pairing"] = evaluate_reaching(shuffled, arm, targets, None, seed, split, latencies, is_agent=True, name="shuffled")
    controls["shuffled_pairing"]["babbling_model_mse"] = {n: float(np.mean(sh["prediction_mse"][n][-500:])) for n in PREDICTED}
    corrupted = Agent.load(out / f"arm_seed{seed}_reaching.npz", planner=ModelPlanner(**planner_cfg).for_agent())
    rng = np.random.default_rng(seed_for(STAGE, "corrupt", seed, 0, 0))
    dyn = corrupted.ports.dynamics
    efficacy = corrupted.brain.efficacy.copy()
    into = np.isin(corrupted.connectome.post, dyn) | np.isin(corrupted.connectome.pre, dyn)
    efficacy[into] = rng.permutation(efficacy[into])
    restored = corrupted.brain.with_parameters(efficacy=efficacy)
    for head in corrupted.heads:
        head.brain = restored
    wrapper = type("L", (), {})()
    wrapper.agent = corrupted
    if corrupted.records is not None:
        # the world model the planner reads is the records cortex: every table's records are
        # permuted across cells, so a reading reads the records of other readings
        order = rng.permutation(corrupted.records.cortex.cells)
        for name in list(corrupted.records.records):
            corrupted.records.records[name] = corrupted.records.records[name][order].copy()
    controls["corrupted_dynamics"] = evaluate_reaching(wrapper, arm, targets, None, seed, split, latencies, is_agent=True, name="corrupted")
    # -- baselines outside the candidate's imports
    controls["random"] = evaluate_reaching(RandomController(seed), arm, targets, None, seed, split, latencies, is_agent=False, name="random")
    pd = JacobianPD(arm.lengths, **config["baselines"]["pd"])
    controls["jacobian_pd"] = evaluate_reaching(pd, arm, targets, None, seed, split, latencies, is_agent=False, name="pd")
    controls["jacobian_pd"]["copier_rmse"] = evaluate_copier(pd, arm, seed, B["copier_paths"], B["copier_decisions"], None, split, latencies, is_agent=False, name="pd")["rmse"]
    # the MLP twice: under the candidate's replay policy (matched), and with its own replay ring
    # (its best known protocol), so the comparison is read both ways
    for name, per_transition in (("online_mlp", config["replay"]["per_transition"]), ("online_mlp_replay", config["baselines"].get("mlp_replay", 1))):
        if name == "online_mlp_replay" and per_transition == config["replay"]["per_transition"]:
            continue
        mlp = ModelController(OnlineMLP(hidden=config["baselines"]["mlp_hidden"], lr=config["baselines"]["mlp_lr"], seed=seed), ModelPlanner(**planner_cfg), seed=seed, epsilon=1.0, replay=per_transition, capacity=config["replay"]["capacity"])
        env_m = Arm(arm, seed=seed_for(STAGE, split, seed, 0, 0), life_id=name.replace("online_", ""))
        m_m, mb = run_life(mlp, env_m, B["babbling_decisions"], log, seed, split, "babbling", latencies, is_agent=False)
        mlp.epsilon = config["actor"]["epsilon"]
        m_m, _ = run_life(mlp, env_m, B["reaching_decisions"], log, seed, split, "reaching", latencies, is_agent=False, m=m_m)
        mlp.epsilon = 0.0  # the adapting protocol: the baseline keeps learning too
        controls[name] = evaluate_reaching(mlp, arm, targets, None, seed, split, latencies, is_agent=False, name=name.replace("online_", ""))
        controls[name]["copier_rmse"] = evaluate_copier(mlp, arm, seed, B["copier_paths"], B["copier_decisions"], None, split, latencies, is_agent=False, name=name.replace("online_", ""))["rmse"]
        controls[name]["babbling_model_mse"] = {n: float(np.mean(mb["prediction_mse"][n][-500:])) for n in PREDICTED}
        controls[name]["parameters"] = mlp.model.parameters()
        controls[name]["replay_per_transition"] = per_transition
    result["controls"] = controls
    result["latency_ms"] = {"p50": float(np.percentile(latencies, 50) * 1000), "p95": float(np.percentile(latencies, 95) * 1000), "events": len(latencies)}
    result["body_steps"] = (B["babbling_decisions"] + B["reaching_decisions"] + B["adaptation_decisions"] + B["return_decisions"]) * arm.substeps
    result["wall_seconds"] = time.time() - t0
    result["events"] = log.close()
    return result


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
    receipt.body["supplied_components"] = ["arm dynamics, forward kinematics for sensors and rendering, reward rule", "sensor encoding and bounds", "velocity-field planning objective and beam search over the learned model", "replay ring schedule", "settling schedule and rejection guards"]
    receipt.body["learned_components"] = ["workspace and dynamics synapses and the three prediction heads (hand acceleration, velocity change, angle change)"]
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
        success = [r["heldout"]["success"] for r in results]
        gain = [1 - r["babbling"]["dd_hand"]["model_mse_last500"] / r["babbling"]["dd_hand"]["zero_mse_last500"] for r in results]
        copier = [r["copier"]["rmse"] for r in results]
        recover = [r["changed_body"]["success"] for r in results]
        retain = [r["returned_body"]["success"] for r in results]
        c = lambda key, field="success": [r["controls"][key][field] for r in results]
        receipt.body["metrics"].update({"heldout_success": success, "heldout_success_frozen": [r["heldout_frozen"]["success"] for r in results], "displacement_gain": gain, "copier_rmse": copier, "changed_body_success": recover, "returned_body_success": retain})
        receipt.body["controls"] = {k: c(k) for k in results[0]["controls"]}
        receipt.body["controls"]["copier_rmse"] = {k: c(k, "copier_rmse") for k in ("born_frozen", "jacobian_pd", "online_mlp")}
        receipt.body["counts"] = {"body_steps_per_seed": [r["body_steps"] for r in results], "events_logged": [r["events"]["events"] for r in results], "imagined_per_seed": [r["ledger"]["imagined"] for r in results], "replay_writes": [r["ledger"]["replay_writes"] for r in results]}
        receipt.body["numerics"] = {"ledger": [r["ledger"] for r in results]}
        receipt.body["resources"] = {"latency_ms": [r["latency_ms"] for r in results], "wall_seconds": [r["wall_seconds"] for r in results], "peak_rss_mb": peak_rss_mb(), "parameters": results[0]["parameters"], "mlp_parameters": results[0]["controls"]["online_mlp"]["parameters"]}
        receipt.body["artifacts"] = {"events": [r["events"] for r in results], "checkpoints": [c for r in results for c in r["checkpoints"]]}
        receipt.predicate("reaching_success_mean", float(np.mean(success)), g["success_mean"], "mean over seeds", np.mean(success) >= g["success_mean"])
        receipt.predicate("reaching_success_each", float(np.min(success)), g["success_each"], "min over seeds", np.min(success) >= g["success_each"])
        receipt.predicate("displacement_prediction_gain", float(np.mean(gain)), g["displacement_gain"], "mean over seeds: 1 - model/zero MSE", np.mean(gain) >= g["displacement_gain"])
        receipt.predicate("born_frozen_fails_reaching", float(np.max(c("born_frozen"))), g["success_each"], "max over seeds, must be below", np.max(c("born_frozen")) < g["success_each"])
        pd_rmse = float(np.mean(c("jacobian_pd", "copier_rmse")))
        copier_bar = max(g["copier_rmse"], g["copier_vs_pd"] * pd_rmse)
        receipt.predicate("copier_rmse", float(np.mean(copier)), copier_bar, f"mean over seeds; bar = max({g['copier_rmse']}, {g['copier_vs_pd']} x supplied PD {pd_rmse:.3f})", np.mean(copier) <= copier_bar)
        frozen_rmse = c("born_frozen", "copier_rmse")
        receipt.predicate("copier_vs_random_model", float(1 - np.mean(copier) / np.mean(frozen_rmse)), g["copier_vs_random_model"], "relative improvement over the born-frozen model with the same planner", 1 - np.mean(copier) / np.mean(frozen_rmse) >= g["copier_vs_random_model"])
        receipt.predicate("changed_body_recovery", float(np.mean(recover)), g["adapt_recover"], "mean over seeds", np.mean(recover) >= g["adapt_recover"])
        receipt.predicate("returned_body_retention", float(np.mean(retain)), g["retain"], "mean over seeds", np.mean(retain) >= g["retain"])
        first50 = [r["heldout"]["success_first50"] for r in results]
        receipt.predicate("shuffled_pairing_loss", float(np.mean(first50) - np.mean(c("shuffled_pairing", "success_first50"))), g["lesion_loss_points"], "mean success difference on the first fifty targets", np.mean(first50) - np.mean(c("shuffled_pairing", "success_first50")) >= g["lesion_loss_points"])
        receipt.predicate("corrupted_dynamics_loss", float(np.mean(first50) - np.mean(c("corrupted_dynamics", "success_first50"))), g["lesion_loss_points"], "mean success difference on the first fifty targets", np.mean(first50) - np.mean(c("corrupted_dynamics", "success_first50")) >= g["lesion_loss_points"])
        receipt.predicate("latency_p95_ms", float(np.max([r["latency_ms"]["p95"] for r in results])), g["latency_p95_ms"], "max over seeds", np.max([r["latency_ms"]["p95"] for r in results]) <= g["latency_p95_ms"])
        receipt.predicate("body_steps_cap", float(np.max([r["body_steps"] for r in results])), config["budget"]["body_steps_cap"], "max over seeds", np.max([r["body_steps"] for r in results]) <= config["budget"]["body_steps_cap"])
    complete = not receipt.body["seeds"]["failed"] and set(receipt.body["seeds"]["completed"]) == set(args.seeds) and (args.pilot or split == "development" or set(args.seeds) == set(config["seeds"]["acceptance"]))
    path = receipt.finish(complete)
    for p in receipt.body["acceptance"]["predicates"]:
        print(f"{'PASS' if p['passed'] else 'FAIL'} {p['name']}: {p['value']} vs {p['threshold']} ({p['aggregation']})")
    for r in results:
        print(f"seed {r['seed']}: heldout {r['heldout']['success']:.2f} copier {r['copier']['rmse']:.3f} changed {r['changed_body']['success']:.2f} returned {r['returned_body']['success']:.2f} controls " + " ".join(f"{k}={v['success']:.2f}" for k, v in r["controls"].items()))
    print("status", receipt.body["status"], "->", path)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
