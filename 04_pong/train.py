"""04 Pong: a single-frame brain learns a teacher, practises, and revisits its mistakes.

Run python train.py, then python build_page.py. Add --device cuda or --device mps
for GPU execution. Use --output runs/my-pong/receipt.json to keep separate candidates.
Each observation is 192 pixels. Afterglow carries prior input activity; reward
replay uses the causal clamps saved when each action was taken. The teacher can
see simulator velocity; the learned player must infer it from its own trace.
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
from pong import ACTIONS, H, MAX_RALLY, W, Pong

HERE = Path(__file__).resolve().parent
SOURCES = [("04_pong/train.py", Path(__file__).resolve()), ("04_pong/pong.py", HERE / "pong.py")]
SOURCES += [(f"cadence/{p.name}", p) for p in sorted(Path(cd.__file__).parent.glob("*.py"))]

INPUTS = 2 * H * W  # two frames
HIDDEN = 128
ENVS, HORIZON = 64, 64  # parallel games, steps per rollout
GAMMA = 0.5  # short credit horizon: the shaping already says whether each step helped
ITERATIONS = 30
BATCH = 256
ETA, DECAY = 5e-4, 1.0  # the local step per seam, with no anneal: the running RMS below sets the scale
# Each seam steps on a running average of its own contrasts, divided by the running RMS of them,
# both corrected for their short history: the per-parameter step of an adaptive optimiser, kept
# local (a seam reads only its own two endpoints and its own history). The baseline gets the
# same treatment from Adam; without it the plain rule stalled at 78% of balls returned.
MOMENTUM, RMS, FLOOR = 0.9, 0.999, 1e-8
ADVANTAGE_CLIP: float | None = None  # clip normalised advantages to [-clip, clip] so the nudge stays small
CONFIG = cd.LearnerConfig(
    beta=0.1, eta=ETA, eta_bias=0.02, temperature=0.2, tolerance=1e-4, nudged_steps=24, free_steps=100
)
EVAL_POINTS = 1000


def returns_from(rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    """Discounted reward-to-go along the time axis, cut at episode ends. Shapes (T, E)."""
    out = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + GAMMA * running * (~dones[t])
        out[t] = running
    return out


class PatchPolicy:
    """The patch-net policy and its reward-nudged learner."""

    def __init__(self, hidden: int, seed: int, *, device: str = "cpu", memory: bool = True, decay: float = 0.5, focus: float = 0.0) -> None:
        self.wiring = cd.layered(INPUTS, hidden, ACTIONS, density=1.0, seed=seed)
        self.memory = memory
        if memory:
            self.wiring.sets["input"] = tuple(range(H * W))
            self.wiring.sets["afterglow"] = tuple(range(H * W, INPUTS))
        self.learner = cd.Learner(
            cd.Settlement(
                self.wiring, cd.learning_rule(dt=1.0),
                backend="cpu" if device == "cpu" else "torch", device=device,
                precision=None if device == "cpu" else "float32",
            ), self.wiring.sets["output"], CONFIG
        )
        self.rng = np.random.default_rng(seed)
        self.steps: list[float] = []
        self.trace = cd.Afterglow(self.wiring, source="input", decay=decay, focus=focus,
                                 amplitude=self.learner.engine.rule.clamp_amplitude) if memory else None
        self.last_drive: np.ndarray | None = None

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        if self.trace is not None:
            self.trace.reset(batch, rows=rows)

    def observation(self, env: Pong) -> np.ndarray:
        return env.frames() if self.memory else env.observation()

    def drive(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[1] == self.wiring.n:
            return frames  # replay the exact causal clamp recorded during the rollout
        drive = np.zeros((len(frames), self.wiring.n))
        count = H * W if self.memory else INPUTS
        drive[:, :count] = self.learner.engine.clamp_levels(frames[:, :count])
        return self.trace.clamp(drive) if self.trace is not None else drive

    def probabilities(self, frames: np.ndarray) -> np.ndarray:
        self.last_drive = self.drive(frames)
        free = self.learner.free(self.last_drive)
        if self.trace is not None:
            self.trace.update(free)
        s = free.activation[:, self.learner.output_index]
        z = s / self.learner.config.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def act(self, frames: np.ndarray, greedy: bool = False) -> np.ndarray:
        config = self.learner.config
        if greedy:
            self.learner.config = dataclasses.replace(config, tolerance=1e-4)
        try:
            p = self.probabilities(frames)
        finally:
            self.learner.config = config
        if greedy:
            return p.argmax(axis=1)
        return (self.rng.random(len(p))[:, None] < p.cumsum(axis=1)).argmax(axis=1)

    def update(self, frames: np.ndarray, actions: np.ndarray, advantages: np.ndarray, eta: float) -> None:
        """Free phase, the two nudged phases toward and away from the action taken (weighted by its
        advantage), the contrast every seam reads, then the adaptive local step."""
        learner = self.learner
        learner.config = dataclasses.replace(
            learner.config, eta=eta, eta_bias=eta, momentum=MOMENTUM,
            normalize=RMS, normalize_floor=FLOOR,
        )
        order = self.rng.permutation(len(actions))
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            drive = self.drive(frames[idx])
            target = learner.targets(actions[idx])
            free = learner.free(drive)
            toward = learner.nudged(drive, free, target, weight=advantages[idx])
            away = learner.nudged(drive, free, target, sign=-1.0, weight=advantages[idx])
            learner.update(free, toward, away)
            self.steps.append(free.steps)

    def parameters(self) -> int:
        return self.learner.parameters()


class TorchPolicy:
    """The same shape as a plain MLP trained by backprop REINFORCE with Adam: the baseline."""

    def __init__(self, hidden: int, seed: int, lr: float = 1e-3, *, device: str = "cpu") -> None:
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.device = torch.device(device)
        self.net = torch.nn.Sequential(torch.nn.Linear(INPUTS, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, ACTIONS)).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

    def probabilities(self, frames: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.torch.softmax(self.net(self.torch.tensor(frames, dtype=self.torch.float32, device=self.device)), dim=1).cpu().numpy()

    def act(self, frames: np.ndarray, greedy: bool = False) -> np.ndarray:
        p = self.probabilities(frames)
        if greedy:
            return p.argmax(axis=1)
        return (self.rng.random(len(p))[:, None] < p.cumsum(axis=1)).argmax(axis=1)

    def update(self, frames: np.ndarray, actions: np.ndarray, advantages: np.ndarray, eta: float) -> None:
        torch = self.torch
        order = self.rng.permutation(len(actions))
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            logp = torch.log_softmax(self.net(torch.tensor(frames[idx], dtype=torch.float32, device=self.device)), dim=1)
            chosen = logp[torch.arange(len(idx), device=self.device), torch.tensor(actions[idx], device=self.device)]
            loss = -(chosen * torch.tensor(advantages[idx], dtype=torch.float32, device=self.device)).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


def rollout(policy: Any, env: Pong) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    frames, actions, rewards, dones = [], [], [], []
    hits = misses = 0
    for _ in range(HORIZON):
        f = policy.observation(env) if isinstance(policy, PatchPolicy) else env.observation()
        a = policy.act(f)
        r, d, info = env.step(a)
        frames.append(policy.last_drive.copy() if isinstance(policy, PatchPolicy) else f)
        if isinstance(policy, PatchPolicy):
            policy.reset(env.envs, rows=d)
        actions.append(a)
        rewards.append(r)
        dones.append(d)
        hits += int(info["hit"].sum())
        misses += int(info["miss"].sum())
    returns = returns_from(np.asarray(rewards), np.asarray(dones))
    stats = {"hits": hits, "misses": misses, "hit_rate": hits / max(hits + misses, 1), "mean_reward": float(np.mean(rewards))}
    return np.concatenate(frames), np.concatenate(actions), returns.ravel(), stats


def train(policy: Any, seed: int, iterations: int, log: str) -> tuple[list[dict[str, float]], float]:
    env = Pong(ENVS, seed=seed)
    history = []
    if isinstance(policy, PatchPolicy):
        policy.reset(env.envs)
    eta = ETA
    t0 = time.perf_counter()
    for it in range(1, iterations + 1):
        frames, actions, returns, stats = rollout(policy, env)
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        if ADVANTAGE_CLIP is not None:
            advantages = np.clip(advantages, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)
        policy.update(frames, actions, advantages, eta)
        eta *= DECAY
        history.append({"iteration": it, **stats, "seconds": time.perf_counter() - t0})
        if it % 10 == 0 or it == 1:
            print(f"{log} iteration {it:3d}: hit rate {stats['hit_rate']:.2f} ({stats['hits']} hits, {stats['misses']} misses), {time.perf_counter() - t0:.0f}s", flush=True)
    return history, time.perf_counter() - t0


def evaluate(policy: Any, seed: int, points: int, *, opponent_skill: float = 0.7) -> dict[str, float]:
    """Greedy play until ``points`` points have ended: hits per point and rally length."""
    env = Pong(ENVS, seed=seed, opponent_skill=opponent_skill)
    hits = misses = ended = wins = 0
    if hasattr(policy, "reset"):
        policy.reset(env.envs)
    steps = 0
    while ended < points:
        obs = policy.observation(env) if hasattr(policy, "observation") else env.observation()
        r, d, info = env.step(policy.act(obs, greedy=True))
        if hasattr(policy, "reset"):
            policy.reset(env.envs, rows=d)
        hits += int(info["hit"].sum())
        misses += int(info["miss"].sum())
        wins += int(info["opponent_miss"].sum())
        ended += int(d.sum())
        steps += ENVS
    return {"points": ended, "wins": wins, "losses": misses, "draws": ended - wins - misses, "win_rate": wins / ended, "opponent_skill": opponent_skill, "hits": hits, "misses": misses, "hits_per_point": hits / ended, "return_rate": hits / max(hits + misses, 1)}


IMITATION_STEPS, IMITATION_EPOCHS = 800, 12


def teacher(env: Pong) -> np.ndarray:
    """Return at an edge of the paddle to keep the ball diagonal, rather than drawing flat rallies.

    The teacher can see simulator velocity; the learned policy must infer it from its trace.
    """
    row = np.clip(env.ball_r + env.ball_dr, 0, H - 1)
    target = row + np.where(row < H // 2, 1, -1)
    return (np.sign(target - (env.right + 1)) + 1).astype(int)


def imitation(seed: int, *, device: str = "cpu", decay: float = 0.5, focus: float = 0.0) -> tuple[PatchPolicy, dict[str, Any]]:
    """The same net taught by the scripted tracker: rollouts of the tracker with random slips, labels from the tracker."""
    env = Pong(ENVS, seed=seed + 7)
    rng = np.random.default_rng(seed)
    frames, labels = [], []
    policy = PatchPolicy(HIDDEN, seed, device=device, decay=decay, focus=focus)
    for _ in range(IMITATION_STEPS):
        obs = policy.observation(env)
        want = teacher(env)
        policy.probabilities(obs)
        frames.append(policy.last_drive.copy())
        labels.append(want)
        _, done, _ = env.step(np.where(rng.random(ENVS) < 0.3, rng.integers(0, ACTIONS, ENVS), want))
        policy.reset(env.envs, rows=done)
    x, y = np.concatenate(frames), np.concatenate(labels)
    policy.rehearsal = (x[::max(1, len(x) // 512)].copy(), y[::max(1, len(y) // 512)].copy())
    learner = policy.learner
    learner.config = dataclasses.replace(learner.config, temperature=0.1)
    t0 = time.perf_counter()
    for epoch in range(IMITATION_EPOCHS):
        learner.config = dataclasses.replace(learner.config, eta=3.0 * 0.8**epoch, eta_bias=0.03 * 0.8**epoch)
        order = rng.permutation(len(y))
        for s in range(0, len(order), 64):
            idx = order[s : s + 64]
            learner.step(policy.drive(x[idx]), y[idx])
    seconds = time.perf_counter() - t0
    return policy, {"rows": int(len(y)), "epochs": IMITATION_EPOCHS, "seconds": seconds, "parameters": policy.parameters()}


def practice(policy: PatchPolicy, seed: int, iterations: int) -> dict:
    """Explore after imitation, then coach mistakes on the states actually encountered.

    Replay clamps include only the trace available when the action was taken.
    Validation selects a retained checkpoint; test opponents never train the model.
    """
    initial = evaluate(policy, 200, 500)
    best_score, best_engine = initial["win_rate"], policy.learner.engine
    history = []
    coached = 0
    env = Pong(ENVS, seed=seed)
    policy.reset(ENVS)
    t0 = time.perf_counter()
    for iteration in range(1, iterations + 1):
        drives, actions, labels, rewards, dones = [], [], [], [], []
        for _ in range(HORIZON):
            labels.append(teacher(env))
            action = policy.act(policy.observation(env))
            drives.append(policy.last_drive.copy())
            reward, done, info = env.step(action)
            rewards.append(reward + 2 * info["opponent_miss"])
            dones.append(done)
            actions.append(action)
            policy.reset(ENVS, rows=done)
        x, a, y = np.concatenate(drives), np.concatenate(actions), np.concatenate(labels)
        reward = np.asarray(rewards)
        returns = np.zeros_like(reward)
        running = np.zeros(ENVS)
        for t in reversed(range(HORIZON)):
            running = reward[t] + .97 * running * (~np.asarray(dones[t]))
            returns[t] = running
        advantage = returns.ravel()
        advantage = np.clip((advantage - advantage.mean()) / (advantage.std() + 1e-8), -2, 2)
        learner = policy.learner
        order = policy.rng.permutation(len(a))
        for start in range(0, len(a), BATCH):
            ix = order[start:start + BATCH]
            learner.config = dataclasses.replace(learner.config, eta=.003, eta_bias=.00003,
                                                  momentum=0, normalize=0)
            free, _ = learner.step(x[ix], a[ix], weight=advantage[ix])
            wrong = free.free.activation[:, learner.output_index].argmax(axis=1) != y[ix]
            if wrong.any():
                learner.config = dataclasses.replace(learner.config, eta=.1, eta_bias=.001)
                learner.step(x[ix], y[ix])
                coached += int(wrong.sum())
        if hasattr(policy, "rehearsal"):
            learner.step(*policy.rehearsal)
        if iteration % 10 == 0 or iteration == iterations:
            result = evaluate(policy, 200, 500)
            keep = result["win_rate"] >= best_score
            if keep:
                best_score, best_engine = result["win_rate"], learner.engine
            history.append({"iteration": iteration, "validation": result, "retained": keep,
                            "coached_rows": coached})
            print(f"practice {iteration}: validation win rate {result['win_rate']:.3f}, retained {keep}", flush=True)
            env = Pong(ENVS, seed=seed + iteration)
            policy.reset(ENVS)
    policy.learner.engine = best_engine
    return {"initial_validation": initial, "history": history, "iterations": iterations,
            "coached_rows": coached, "seconds": time.perf_counter() - t0,
            "reward": "return/miss and distance shaping, plus 2 for opponent miss; discount .97",
            "selection": "best validation win rate on seed 200; no test feedback"}


def export(policy: PatchPolicy, path: Path, meta: dict) -> None:
    engine = policy.learner.engine
    rule = engine.rule
    payload = {
        "n": policy.wiring.n,
        "sets": {k: [int(i) for i in v] for k, v in policy.wiring.sets.items()},
        "W": [round(float(w), 5) for w in engine.dense().ravel()],
        "bias": [round(float(b), 5) for b in engine.bias],
        "rule": {"slope": rule.slope, "threshold": rule.threshold, "leak": rule.leak, "dt": rule.dt, "clamp": rule.clamp_amplitude, "rest": rule.rest_emission},
        "temperature": policy.learner.config.temperature,
        "field": {"H": H, "W": W, "frames": 1 if policy.memory else 2},
        "trace": policy.trace.to_dict() if policy.trace is not None else None,
        "meta": meta,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def run(seed: int, out: Path, iterations: int, *, device: str = "cpu") -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    patch = PatchPolicy(HIDDEN, seed, device=device)
    untrained = evaluate(patch, seed + 100, 200)
    patch, imitation_meta = imitation(seed, device=device)
    imitation_final = evaluate(patch, seed + 100, EVAL_POINTS)
    export(patch, out.parent / "net_imitation.json", {**imitation_final,
           "parameters": patch.parameters(), "learned_from": "a teacher's diagonal returns"})
    practiced = practice(patch, seed + 1000, iterations)
    final = evaluate(patch, seed + 100, EVAL_POINTS)
    tests = [evaluate(patch, s, EVAL_POINTS, opponent_skill=skill)
             for skill in (.7, 1.0) for s in (101, 102, 103)]
    seconds = imitation_meta["seconds"] + practiced["seconds"]
    history = []
    baseline = TorchPolicy(HIDDEN, seed, device=device)
    base_history, base_seconds = train(baseline, seed, iterations, "MLP reward-only reference")
    base_final = evaluate(baseline, seed + 100, EVAL_POINTS)
    conformance = cd.conformance(patch.learner.engine, patch.drive(Pong(1, seed=1).frames())[0], steps=100)
    export(patch, out.parent / "net.json", {**final, "parameters": patch.parameters(),
           "learned_from": "imitation, practice, and corrective teaching"})
    patch.learner.save(out.parent / "learner.npz")
    np.savez_compressed(out.parent / "rehearsal.npz", x=patch.rehearsal[0], y=patch.rehearsal[1])
    print(f"staged paddle: {final['win_rate']:.3f} wins, {final['return_rate']:.3f} returns", flush=True)
    body = {
        "memory": {"incoming_frames": 1, "trace": patch.trace.to_dict(), "replay": "frozen causal clamps; no gradient through time"},
        "execution": {"device": device, "backend": patch.learner.engine.backend,
                      "precision": patch.learner.engine.precision, "environment_device": "cpu"},
        "environment": {"H": H, "W": W, "max_rally": MAX_RALLY, "envs": ENVS, "horizon": HORIZON, "gamma": GAMMA, "iterations": iterations, "batch": BATCH, "eta": ETA, "decay": DECAY},
        "local_step": {"momentum": 0, "normalize": 0, "reward_eta": .003, "reward_eta_bias": .00003, "teacher_eta": .1, "teacher_eta_bias": .001, "practice_discount": .97},
        "config": CONFIG.to_dict(),
        "wiring": patch.wiring.summary(),
        "rule": patch.learner.engine.rule.to_dict(),
        "learner": patch.learner.to_dict(),
        "untrained": untrained,
        "practice": practiced, "held_out_play": tests,
        "history": history,
        "training_seconds": seconds,
        "mean_free_steps": float(np.mean(patch.steps)) if patch.steps else 0.0,
        "final": final,
        "imitation": {**imitation_meta, "final": imitation_final},
        "conformance": conformance,
        "comparison": [{"model": f"MLP {INPUTS}-{HIDDEN}-{ACTIONS}, REINFORCE with Adam", "parameters": baseline.parameters(), "history": base_history, "training_seconds": base_seconds, "final": base_final}],
        "boundary": {
            "learning_rule": "free/nudged contrastive Hebbian, centered, owner-local; plain local steps, advantage-weighted action nudges and teacher corrections",
            "goal_enters_only_through_the_nudge": True,
            "reward": "practice: +1 return, -1 miss, +2 opponent miss, plus paddle-to-ball distance shaping; reward-only MLP omits opponent-miss bonus and uses environment gamma",
            "opponent": "scripted tracker with skill 0.7, not learned",
            "staged_policy_receives_teacher_labels_baseline_does_not": True,
            "same_play_rollout_budget_after_imitation": True,
            "baseline_observation": "two explicit frames; patch sees one frame and keeps its own input-activity trace",
            "evaluation_is_greedy_play_on_fresh_seeds": True,
            "imitation_net_is_a_second_policy_for_the_page_and_is_not_learned_from_reward": True,
        },
    }
    receipt = cd.Receipt.build("cadence-examples/04-pong/v1", body, sources=SOURCES)
    receipt.write(out)
    print(f"conformance {conformance['max_abs_deviation']:.1e}; receipt {out} ({receipt.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    for name, row in (("final", body["final"]), ("untrained", body["untrained"]), ("imitation", body["imitation"]["final"])):
        if abs(row["hits"] / row["points"] - row["hits_per_point"]) > 1e-9:
            return f"{name}: hits per point does not follow from the counts"
    for row in body["history"]:
        if abs(row["hits"] / max(row["hits"] + row["misses"], 1) - row["hit_rate"]) > 1e-9:
            return f"iteration {row['iteration']}: hit rate does not follow from the counts"
    if not body["conformance"]["ledger"]["clean"]:
        return "reference ledger is not clean"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--output", type=Path, default=HERE / "receipt.json")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu",
                        help="training device for both learners (GPU uses float32; game simulation stays on CPU)")
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=SOURCES, check=check)
        print(message)
        return 0 if ok else 1
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    run(args.seed, args.output, args.iterations, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
