"""08 cart-pole: a patch net learns to balance a pole from reward, with advantage-weighted nudges.

Run:  python train.py                      (a few minutes)
      python train.py --verify receipt.json

The classic control task: a cart on a track, a pole hinged on top, push left or right each
step, keep the pole up. The state is four numbers (cart position and velocity, pole angle
and angular velocity); each is clamped onto a small population of input owners as a
soft one-hot over bins, so the net sees a place code rather than raw floats. Two output
owners are the two pushes. Learning is the reward-nudged rule of the Pong example: the
nudge's target is the action taken, its weight the action's advantage (the discounted
return, normalised). The receipt records episode length over training and on fresh
evaluation episodes, next to the same net trained by REINFORCE with Adam on the same
rollouts.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
SOURCES = [("08_cartpole/train.py", Path(__file__).resolve())]
BINS = 12  # owners per state variable
INPUTS = 4 * BINS
ACTIONS = 2
HIDDEN = 32
ENVS, HORIZON, MAX_STEPS = 64, 100, 500
GAMMA = 0.97
ITERATIONS = 120
BATCH = 256
ETA, DECAY = 2.0, 0.99
CONFIG = cd.LearnerConfig(beta=0.1, eta=ETA, eta_bias=0.02, temperature=0.2, tolerance=3e-3, nudged_steps=12, free_steps=100)
EVAL_EPISODES = 200
RANGES = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.21, 0.21], [-3.0, 3.0]])


class CartPole:
    """The standard cart-pole (Barto, Sutton, Anderson 1983) as arrays over many episodes."""

    def __init__(self, envs: int, seed: int) -> None:
        self.envs = envs
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros((envs, 4))
        self.age = np.zeros(envs, dtype=int)
        self.reset(np.ones(envs, dtype=bool))

    def reset(self, which: np.ndarray) -> None:
        k = int(which.sum())
        if k:
            self.state[which] = self.rng.uniform(-0.05, 0.05, size=(k, 4))
            self.age[which] = 0

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gravity, mass_cart, mass_pole, length, force_mag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
        x, x_dot, theta, theta_dot = self.state.T
        force = np.where(action == 1, force_mag, -force_mag)
        cos, sin = np.cos(theta), np.sin(theta)
        total = mass_cart + mass_pole
        temp = (force + mass_pole * length * theta_dot**2 * sin) / total
        theta_acc = (gravity * sin - cos * temp) / (length * (4.0 / 3.0 - mass_pole * cos**2 / total))
        x_acc = temp - mass_pole * length * theta_acc * cos / total
        self.state = np.stack([x + tau * x_dot, x_dot + tau * x_acc, theta + tau * theta_dot, theta_dot + tau * theta_acc], axis=1)
        self.age += 1
        failed = (np.abs(self.state[:, 0]) > 2.4) | (np.abs(self.state[:, 2]) > 0.2095)
        done = failed | (self.age >= MAX_STEPS)
        reward = np.where(failed, -1.0, 1.0 / MAX_STEPS * 10)  # a small reward for every upright step, a penalty for falling
        lengths = self.age.copy()
        self.reset(done)
        return reward, done, lengths


def place_code(state: np.ndarray) -> np.ndarray:
    """Each variable as a soft one-hot over BINS owners: a bump centred on its value."""
    out = np.zeros((len(state), 4, BINS))
    centres = np.linspace(0.0, 1.0, BINS)
    for k in range(4):
        u = np.clip((state[:, k] - RANGES[k, 0]) / (RANGES[k, 1] - RANGES[k, 0]), 0.0, 1.0)
        out[:, k, :] = np.exp(-((u[:, None] - centres[None, :]) ** 2) / (2 * (1.0 / BINS) ** 2))
    return out.reshape(len(state), INPUTS)


def returns_from(rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + GAMMA * running * (~dones[t])
        out[t] = running
    return out


class PatchPolicy:
    def __init__(self, seed: int) -> None:
        self.wiring = cd.layered(INPUTS, HIDDEN, ACTIONS, density=1.0, seed=seed)
        self.learner = cd.Learner(cd.Settlement(self.wiring, cd.learning_rule(dt=1.0)), self.wiring.sets["output"], CONFIG)
        self.rng = np.random.default_rng(seed)

    def drive(self, obs: np.ndarray) -> np.ndarray:
        return self.learner.engine.clamp_levels(np.pad(obs, ((0, 0), (0, self.wiring.n - INPUTS))))

    def probabilities(self, obs: np.ndarray) -> np.ndarray:
        s = self.learner.free(self.drive(obs)).activation[:, self.learner.output_index]
        z = s / self.learner.config.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def act(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        p = self.probabilities(obs)
        return p.argmax(axis=1) if greedy else (self.rng.random(len(p))[:, None] < p.cumsum(axis=1)).argmax(axis=1)

    def update(self, obs: np.ndarray, actions: np.ndarray, advantages: np.ndarray, eta: float) -> None:
        learner = self.learner
        learner.config = dataclasses.replace(learner.config, eta=eta, eta_bias=CONFIG.eta_bias * eta / ETA)
        order = self.rng.permutation(len(actions))
        for s in range(0, len(order), BATCH):
            idx = order[s : s + BATCH]
            learner.step(self.drive(obs[idx]), actions[idx], weight=advantages[idx])

    def parameters(self) -> int:
        return self.learner.parameters()


class TorchPolicy:
    def __init__(self, seed: int, lr: float = 1e-3) -> None:
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(INPUTS, HIDDEN), torch.nn.Tanh(), torch.nn.Linear(HIDDEN, ACTIONS))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

    def probabilities(self, obs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.torch.softmax(self.net(self.torch.tensor(obs, dtype=self.torch.float32)), dim=1).numpy()

    def act(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        p = self.probabilities(obs)
        return p.argmax(axis=1) if greedy else (self.rng.random(len(p))[:, None] < p.cumsum(axis=1)).argmax(axis=1)

    def update(self, obs: np.ndarray, actions: np.ndarray, advantages: np.ndarray, eta: float) -> None:
        torch = self.torch
        order = self.rng.permutation(len(actions))
        for s in range(0, len(order), BATCH):
            idx = order[s : s + BATCH]
            logp = torch.log_softmax(self.net(torch.tensor(obs[idx], dtype=torch.float32)), dim=1)
            chosen = logp[torch.arange(len(idx)), torch.tensor(actions[idx])]
            loss = -(chosen * torch.tensor(advantages[idx], dtype=torch.float32)).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


def rollout(policy: Any, env: CartPole) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    obs_list, actions, rewards, dones, finished = [], [], [], [], []
    for _ in range(HORIZON):
        obs = place_code(env.state)
        a = policy.act(obs)
        r, d, lengths = env.step(a)
        obs_list.append(obs)
        actions.append(a)
        rewards.append(r)
        dones.append(d)
        finished.extend(lengths[d].tolist())
    returns = returns_from(np.asarray(rewards), np.asarray(dones))
    stats = {"episodes_finished": len(finished), "mean_length": float(np.mean(finished)) if finished else float(HORIZON)}
    return np.concatenate(obs_list), np.concatenate(actions), returns.ravel(), stats


def train(policy: Any, seed: int, iterations: int, log: str) -> tuple[list[dict[str, float]], float]:
    env = CartPole(ENVS, seed)
    history = []
    eta = ETA
    t0 = time.perf_counter()
    for it in range(1, iterations + 1):
        obs, actions, returns, stats = rollout(policy, env)
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy.update(obs, actions, advantages, eta)
        eta *= DECAY
        history.append({"iteration": it, **stats, "seconds": time.perf_counter() - t0})
        if it % 10 == 0 or it == 1:
            print(f"{log} iteration {it:3d}: mean episode length {stats['mean_length']:.0f} over {stats['episodes_finished']} finished ({time.perf_counter() - t0:.0f}s)", flush=True)
    return history, time.perf_counter() - t0


def evaluate(policy: Any, seed: int, episodes: int) -> dict[str, float]:
    """Greedy play from fresh starts until ``episodes`` have ended; the mean length, out of MAX_STEPS."""
    env = CartPole(ENVS, seed)
    lengths: list[int] = []
    while len(lengths) < episodes:
        _, d, ages = env.step(policy.act(place_code(env.state), greedy=True))
        lengths.extend(ages[d].tolist())
    lengths = lengths[:episodes]
    return {"episodes": episodes, "mean_length": float(np.mean(lengths)), "solved_fraction": float(np.mean(np.asarray(lengths) >= MAX_STEPS)), "max_steps": MAX_STEPS}


def run(seed: int, out: Path, iterations: int) -> dict[str, Any]:
    patch = PatchPolicy(seed)
    untrained = evaluate(patch, seed + 100, 100)
    print(f"untrained: mean episode length {untrained['mean_length']:.0f}", flush=True)
    history, seconds = train(patch, seed, iterations, "patch net")
    final = evaluate(patch, seed + 100, EVAL_EPISODES)
    print(f"patch net: mean episode length {final['mean_length']:.0f}, solved {final['solved_fraction']:.2f} ({patch.parameters()} parameters, {seconds:.0f}s)", flush=True)
    baseline = TorchPolicy(seed)
    base_history, base_seconds = train(baseline, seed, iterations, "backprop")
    base_final = evaluate(baseline, seed + 100, EVAL_EPISODES)
    print(f"backprop: mean episode length {base_final['mean_length']:.0f}, solved {base_final['solved_fraction']:.2f} ({baseline.parameters()} parameters, {base_seconds:.0f}s)", flush=True)
    conformance = cd.conformance(patch.learner.engine, patch.drive(place_code(CartPole(1, 1).state))[0], steps=100)
    body = {
        "environment": {"bins": BINS, "envs": ENVS, "horizon": HORIZON, "max_steps": MAX_STEPS, "gamma": GAMMA, "iterations": iterations, "batch": BATCH, "eta": ETA, "decay": DECAY, "ranges": RANGES.tolist()},
        "config": CONFIG.to_dict(), "wiring": patch.wiring.summary(), "rule": patch.learner.engine.rule.to_dict(), "learner": patch.learner.to_dict(),
        "untrained": untrained, "history": history, "training_seconds": seconds, "final": final, "conformance": conformance,
        "comparison": [{"model": f"MLP {INPUTS}-{HIDDEN}-{ACTIONS}, REINFORCE with Adam", "parameters": baseline.parameters(), "history": base_history, "training_seconds": base_seconds, "final": base_final}],
        "boundary": {"learning_rule": "free/nudged contrastive Hebbian, centered, owner-local; nudge target = action taken, weight = normalised advantage", "goal_enters_only_through_the_nudge": True, "reward": "+0.02 per upright step, -1 on falling; episodes capped at 500 steps", "observation": "place code, 12 owners per state variable", "same_rollout_budget_for_the_baseline": True, "evaluation_is_greedy_play_on_fresh_seeds": True},
    }
    receipt = cd.Receipt.build("cadence-examples/08-cartpole/v1", body, sources=SOURCES)
    receipt.write(out)
    print(f"conformance {conformance['max_abs_deviation']:.1e}; receipt {out} ({receipt.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    if body["final"]["episodes"] != EVAL_EPISODES:
        return "evaluation did not run the declared number of episodes"
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
