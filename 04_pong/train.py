"""04 Pong: a paddle learns from pixels and reward, by nudging toward the actions that paid.

Run:  python train.py                      (a few minutes; then a backprop baseline on the same rollouts)
      python train.py --verify receipt.json

The policy is a patch net: 384 input owners (the pixels of two frames), hidden owners, 3
output owners (up, stay, down). Acting is a free settlement and a draw from the softmax of the output
activations. Learning is the same free/nudged rule as classification, with one change:
the target of the nudge is the action that was taken, and the strength of the nudge is
that action's advantage, so actions that paid are pulled toward and actions that cost are
pushed away. That is the policy gradient, entering through the nudge alone.
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
INPUTS = 2 * H * W  # two frames
HIDDEN = 32
ENVS, HORIZON = 64, 64  # parallel games, steps per rollout
GAMMA = 0.5  # short credit horizon: the shaping already says whether each step helped
ITERATIONS = 300
BATCH = 256
ETA, DECAY = 2.0, 0.99
ADVANTAGE_CLIP: float | None = None  # clip normalised advantages to [-clip, clip] so the nudge stays small
CONFIG = cd.LearnerConfig(
    beta=0.1, eta=ETA, eta_bias=0.02, temperature=0.2, tolerance=3e-3, nudged_steps=12, free_steps=100
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

    def __init__(self, hidden: int, seed: int) -> None:
        self.wiring = cd.layered(INPUTS, hidden, ACTIONS, density=1.0, seed=seed)
        self.learner = cd.Learner(
            cd.Settlement(self.wiring, cd.learning_rule(dt=1.0)), self.wiring.sets["output"], CONFIG
        )
        self.rng = np.random.default_rng(seed)
        self.steps: list[float] = []

    def drive(self, frames: np.ndarray) -> np.ndarray:
        return self.learner.engine.clamp_levels(np.pad(frames, ((0, 0), (0, self.wiring.n - INPUTS))))

    def probabilities(self, frames: np.ndarray) -> np.ndarray:
        s = self.learner.free(self.drive(frames)).activation[:, self.learner.output_index]
        z = s / self.learner.config.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def act(self, frames: np.ndarray, greedy: bool = False) -> np.ndarray:
        p = self.probabilities(frames)
        if greedy:
            return p.argmax(axis=1)
        return (self.rng.random(len(p))[:, None] < p.cumsum(axis=1)).argmax(axis=1)

    def update(self, frames: np.ndarray, actions: np.ndarray, advantages: np.ndarray, eta: float) -> None:
        learner = self.learner
        learner.config = dataclasses.replace(learner.config, eta=eta, eta_bias=CONFIG.eta_bias * eta / ETA)
        order = self.rng.permutation(len(actions))
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            _, report = learner.step(self.drive(frames[idx]), actions[idx], weight=advantages[idx])
            self.steps.append(report["free_steps"])

    def parameters(self) -> int:
        return self.learner.parameters()


class TorchPolicy:
    """The same shape as a plain MLP trained by backprop REINFORCE with Adam: the baseline."""

    def __init__(self, hidden: int, seed: int, lr: float = 1e-3) -> None:
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(INPUTS, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, ACTIONS))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

    def probabilities(self, frames: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.torch.softmax(self.net(self.torch.tensor(frames, dtype=self.torch.float32)), dim=1).numpy()

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
            logp = torch.log_softmax(self.net(torch.tensor(frames[idx], dtype=torch.float32)), dim=1)
            chosen = logp[torch.arange(len(idx)), torch.tensor(actions[idx])]
            loss = -(chosen * torch.tensor(advantages[idx], dtype=torch.float32)).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


def rollout(policy: Any, env: Pong) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    frames, actions, rewards, dones = [], [], [], []
    hits = misses = 0
    for _ in range(HORIZON):
        f = env.observation()
        a = policy.act(f)
        r, d, info = env.step(a)
        frames.append(f)
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


def evaluate(policy: Any, seed: int, points: int) -> dict[str, float]:
    """Greedy play until ``points`` points have ended: hits per point and rally length."""
    env = Pong(ENVS, seed=seed)
    hits = misses = ended = 0
    steps = 0
    while ended < points:
        r, d, info = env.step(policy.act(env.observation(), greedy=True))
        hits += int(info["hit"].sum())
        misses += int(info["miss"].sum())
        ended += int(d.sum())
        steps += ENVS
    return {"points": ended, "hits": hits, "misses": misses, "hits_per_point": hits / ended, "return_rate": hits / max(hits + misses, 1)}


IMITATION_STEPS, IMITATION_EPOCHS = 400, 8


def imitation(seed: int) -> tuple[PatchPolicy, dict[str, Any]]:
    """The same net taught by the scripted tracker: rollouts of the tracker with random slips, labels from the tracker."""
    from pong import track_policy

    env = Pong(ENVS, seed=seed + 7)
    rng = np.random.default_rng(seed)
    frames, labels = [], []
    for _ in range(IMITATION_STEPS):
        obs = env.observation()
        want = track_policy(env)
        frames.append(obs)
        labels.append(want)
        env.step(np.where(rng.random(ENVS) < 0.3, rng.integers(0, ACTIONS, ENVS), want))
    x, y = np.concatenate(frames), np.concatenate(labels)
    policy = PatchPolicy(HIDDEN, seed)
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


def export(policy: PatchPolicy, path: Path, meta: dict) -> None:
    engine = policy.learner.engine
    rule = engine.rule
    payload = {
        "n": policy.wiring.n,
        "sets": {k: [int(i) for i in v] for k, v in policy.wiring.sets.items()},
        "W": [round(float(w), 5) for w in engine.dense().ravel()],
        "bias": [round(float(b), 5) for b in engine.bias],
        "rule": {"slope": rule.slope, "threshold": rule.threshold, "leak": rule.leak, "dt": rule.dt, "clamp": rule.clamp_amplitude, "rest": rule.rest_emission},
        "temperature": CONFIG.temperature,
        "field": {"H": H, "W": W, "frames": 2},
        "meta": meta,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def run(seed: int, out: Path, iterations: int) -> dict[str, Any]:
    patch = PatchPolicy(HIDDEN, seed)
    untrained = evaluate(patch, seed + 100, 200)
    print(f"untrained: {untrained['hits_per_point']:.2f} hits per point")
    history, seconds = train(patch, seed, iterations, "patch net")
    final = evaluate(patch, seed + 100, EVAL_POINTS)
    print(f"patch net: {final['hits_per_point']:.2f} hits per point, return rate {final['return_rate']:.2f} ({patch.parameters()} parameters, {seconds:.0f}s)")

    baseline = TorchPolicy(HIDDEN, seed)
    base_history, base_seconds = train(baseline, seed, iterations, "backprop")
    base_final = evaluate(baseline, seed + 100, EVAL_POINTS)
    print(f"backprop: {base_final['hits_per_point']:.2f} hits per point, return rate {base_final['return_rate']:.2f} ({baseline.parameters()} parameters, {base_seconds:.0f}s)")

    conformance = cd.conformance(patch.learner.engine, patch.drive(Pong(1, seed=1).observation())[0], steps=100)
    export(patch, HERE / "net.json", {"hits_per_point": final["hits_per_point"], "return_rate": final["return_rate"], "parameters": patch.parameters(), "learned_from": "reward"})

    imitator, imitation_meta = imitation(seed)
    imitation_final = evaluate(imitator, seed + 100, EVAL_POINTS)
    print(f"imitation of the tracker: {imitation_final['hits_per_point']:.2f} hits per point, return rate {imitation_final['return_rate']:.2f} ({imitation_meta['seconds']:.0f}s)", flush=True)
    export(imitator, HERE / "net_imitation.json", {"hits_per_point": imitation_final["hits_per_point"], "return_rate": imitation_final["return_rate"], "parameters": imitator.parameters(), "learned_from": "the scripted tracker's actions"})
    body = {
        "environment": {"H": H, "W": W, "max_rally": MAX_RALLY, "envs": ENVS, "horizon": HORIZON, "gamma": GAMMA, "iterations": iterations, "batch": BATCH, "eta": ETA, "decay": DECAY},
        "config": CONFIG.to_dict(),
        "wiring": patch.wiring.summary(),
        "rule": patch.learner.engine.rule.to_dict(),
        "learner": patch.learner.to_dict(),
        "untrained": untrained,
        "history": history,
        "training_seconds": seconds,
        "mean_free_steps": float(np.mean(patch.steps)),
        "final": final,
        "imitation": {**imitation_meta, "final": imitation_final},
        "conformance": conformance,
        "comparison": [{"model": f"MLP {INPUTS}-{HIDDEN}-{ACTIONS}, REINFORCE with Adam", "parameters": baseline.parameters(), "history": base_history, "training_seconds": base_seconds, "final": base_final}],
        "boundary": {
            "learning_rule": "free/nudged contrastive Hebbian, centered, owner-local; nudge target = action taken, nudge weight = normalised advantage",
            "goal_enters_only_through_the_nudge": True,
            "reward": "+1 return, -1 miss, plus the change in paddle-to-ball row distance each step (potential-based shaping)",
            "opponent": "scripted tracker with skill 0.7, not learned",
            "same_rollout_budget_for_the_baseline": True,
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
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=SOURCES, check=check)
        print(message)
        return 0 if ok else 1
    run(args.seed, args.output, args.iterations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
