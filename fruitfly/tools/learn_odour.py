#!/usr/bin/env python3
"""The fly learns which odour means sugar: the library's actor-critic on the measured wiring.

    python tools/learn_odour.py --kind connectome --seed 0 --out runs/learn_odour/connectome_s0.json
    python tools/learn_odour.py --kind shuffled --seed 0 ...      (postsynaptic endpoints permuted)
    python tools/learn_odour.py --kind frozen --seed 0 ...        (the measured wiring, no plasticity)
    python tools/learn_odour.py --kind mlp --seed 0 ...           (an online A2C on the same senses)

The brain is the olfactory sub-net of the BANC connectome (receptor neurons, antennal lobe
projection neurons, Kenyon cells, dopaminergic neurons, mushroom body output neurons and what
two hops reach). The senses are the two odour classes of the release on their receptor neurons.
The actions are read from two mushroom body output neurons: MBON11 (gamma1pedc>alpha/beta,
GABAergic) stands for approach and MBON05 (gamma4>gamma1gamma2, glutamatergic) for avoidance,
by the transmitter rule of Aso et al. 2014 (every MBON whose activation repels the fly is
glutamatergic, every MBON whose activation attracts it is GABAergic or cholinergic; both cells
are in that study's activation screen; Perisse et al. 2016 for MBON11). The arena supplies the turn
toward or away from the odour smelled most; the brain chooses which. The critic reads the
Kenyon cells. Plastic synapses are those from Kenyon cells onto mushroom body output neurons
(``--plastic kc>mbon``, the site of the animal's olfactory memory), or every synapse onto an
MBON, or every synapse. Reward is sugar at a source, nothing at the other, nothing at the time
limit. ``--swap-at`` moves the sugar to the other odour partway through. Evaluation is held out
on fresh arenas, drawn and greedy, with lesions; the direction of the synaptic change is
reported per Kenyon cell class and target, to compare with the animal's depression of the
odour's drive onto the avoidance neurons after sugar.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cadence as cd  # noqa: E402
import cadence.learning  # noqa: E402
from cadence import Brain, NeuronModel  # noqa: E402
from cadence.learning import Learner, LearnerConfig  # noqa: E402
from cadence.plasticity import ActorCritic, ActorCriticConfig  # noqa: E402
from cadence.protocol import shuffled  # noqa: E402
from fruitfly.arena import ACTIONS, ODOURS, Arena, ArenaConfig  # noqa: E402
from fruitfly.banc import MANIFEST_PATH  # noqa: E402
from fruitfly.brain import load_fly  # noqa: E402
from fruitfly.subnet import recruit  # noqa: E402

SEEDS = ("orn:decaying_fruit", "orn:yeasty", "orn:aversive", "orn:fruity", "pn", "kc", "apl", "dan:pam", "dan:ppl1", "mbon", "mbon:MBON11", "mbon:MBON05")
# One cell of each valence: the approach cell is MBON11 (gamma1pedc>alpha/beta, GABAergic) and
# the avoidance cell MBON05 (gamma4>gamma1gamma2, glutamatergic), each the copy with the most
# Kenyon cell input in the release (the right MBON11 receives 1,554 synapses from Kenyon cells,
# the left 262; the left MBON05 219, the right 38).
OUTPUTS = ("mbon:MBON11:right", "mbon:MBON05:left")  # approach, avoid
OUTPUT_ACTION = (0, 1)  # the arena action each output neuron stands for
LESIONS = {"none": (), "fruit ORNs": ("orn:decaying_fruit",), "yeast ORNs": ("orn:yeasty",), "projection neurons": ("pn",), "Kenyon cells": ("kc",), "APL": ("apl",), "dopaminergic neurons": ("dan",)}
PROBE_LEVEL = 0.5  # the odour drive on both sides when an odour alone is probed


class Curve:
    def __init__(self) -> None:
        self.rows: list[tuple[int, str, int]] = []

    def add(self, t: int, info: dict) -> None:
        for key in ("food", "empty", "timeout"):
            for i in np.flatnonzero(info[key]):
                self.rows.append((t, key, int(info["steps"][i])))

    def windows(self, size: int) -> list[dict]:
        out = []
        if not self.rows:
            return out
        for start in range(0, self.rows[-1][0] + 1, size):
            block = [r for r in self.rows if start <= r[0] < start + size]
            if not block:
                continue
            k = len(block); food = [r for r in block if r[1] == "food"]
            out.append({"decision": start + size, "searches": k, "food": len(food) / k, "empty": sum(r[1] == "empty" for r in block) / k, "timeout": sum(r[1] == "timeout" for r in block) / k,
                        "steps_to_food": float(np.mean([r[2] for r in food])) if food else float("nan")})
        return out


class FlyAgent:
    """The olfactory sub-net as an actor: odours onto the receptor neurons, approach or avoidance from four MBONs."""

    def __init__(self, kind: str, seed: int, gain: float, plastic: str, cfg: LearnerConfig, budget: int, hops: int, min_count: float, backend: str = "torch") -> None:
        fly = load_fly()
        sub = recruit(fly.connectome, budget=budget, hops=hops, min_count=min_count, seeds=SEEDS)
        C = sub.connectome if kind != "shuffled" else shuffled(sub.connectome, seed)
        self.sub, self.C, self.kind, self.fly = sub, C, kind, fly
        self.brain = Brain(C, NeuronModel(gain=gain), backend=backend)
        P = C.populations
        for name in OUTPUTS:
            if len(P.get(name, ())) != 1:
                raise RuntimeError(f"{name} must be exactly one neuron in the sub-net, found {len(P.get(name, ()))}")
        self.outputs = [int(P[name][0]) for name in OUTPUTS]
        kc = np.zeros(C.n, dtype=bool); kc[list(P["kc"])] = True
        mb = np.zeros(C.n, dtype=bool); mb[list(P["mbon"])] = True
        if plastic == "kc>mbon":
            mask = kc[C.pre] & mb[C.post]
        elif plastic == "mb":
            mask = mb[C.post]
        elif plastic == "all":
            mask = np.ones(C.synapses, dtype=bool)
        else:
            raise ValueError(plastic)
        if kind == "frozen":
            mask = np.zeros(C.synapses, dtype=bool)
        self.plastic = mask
        self.kc, self.mb = kc, mb
        self.learner = Learner(self.brain, self.outputs, cfg, reciprocal=False, plastic_synapses=mask, plastic_neurons=np.zeros(C.n, dtype=bool))
        self.critic = np.array(P["kc"])
        self.sense_sets = [[list(P[f"orn:{o}:{side}"]) for side in ("left", "right")] for o in ODOURS]
        self.amp = self.brain.neuron_model.stimulus_amplitude
        self.summary = {"kind": kind, "seed": seed, "gain": gain, "plastic": plastic, "plastic_synapses": int(mask.sum()), "neurons": int(C.n), "synapse_classes": int(C.synapses), "outputs": list(OUTPUTS), "output_action": list(OUTPUT_ACTION),
                        "output_ids": [int(sub.members[i]) for i in self.outputs], "actions": list(ACTIONS), "critic": "kc", "critic_neurons": int(len(self.critic)), "seeds": list(SEEDS), "budget": budget, "hops": hops, "min_count": min_count}
        self.kc_class = self.classify_kenyon_cells()
        self.tonic = 0.0

    def balance(self, approach: float = 0.5) -> float:
        """A declared tonic drive on the avoidance cell so that, before any lesson, the naive fly approaches
        a smell with probability ``approach`` on average over the two odours (0.5: indifferent). The
        animal's naive attraction to food odours is the reason to set it above one half: a fly that
        never reaches a source is never rewarded. Found by bisection on the bias of that one neuron; the
        same number is a constant input on the page."""
        cfg = self.learner.config
        drives = np.concatenate([self.odour_drive(k) for k in range(2)])
        base = self.learner.brain.bias.copy()
        lo, hi = 0.0, 4.0
        for _ in range(16):  # the avoidance cell inhibits the approach cell directly, so both are read at each candidate
            mid = 0.5 * (lo + hi); b = base.copy(); b[self.outputs[1]] += mid
            self.learner.brain = self.learner.brain.with_parameters(bias=b)
            act = self.learner.brain.settle_batch(drives, steps=cfg.free_steps, tolerance=cfg.tolerance).activation
            p_approach = float(self.probabilities(act)[:, np.asarray(OUTPUT_ACTION) == 0].sum(axis=1).mean())
            if p_approach > approach:
                lo = mid
            else:
                hi = mid
        self.tonic = 0.5 * (lo + hi); b = base.copy(); b[self.outputs[1]] += self.tonic
        self.learner.brain = self.learner.brain.with_parameters(bias=b); self.brain = self.learner.brain
        self.summary["tonic_avoid"] = self.tonic; self.summary["balance_to"] = approach
        return self.tonic

    def drive(self, obs: np.ndarray) -> np.ndarray:
        d = np.zeros((len(obs), self.C.n))
        for k in range(2):
            for j, idx in enumerate(self.sense_sets[k]):
                d[:, idx] = self.amp * obs[:, 3 * k + j][:, None]
        return d

    def odour_drive(self, k: int, level: float = PROBE_LEVEL) -> np.ndarray:
        obs = np.zeros((1, 6)); obs[0, 3 * k] = obs[0, 3 * k + 1] = level; obs[0, 3 * k + 2] = level
        return self.drive(obs)

    def classify_kenyon_cells(self) -> np.ndarray:
        """Each Kenyon cell by the odour that drives it on the wiring before any lesson: 0 fruit only, 1 yeast only, 2 both, 3 neither."""
        cfg = self.learner.config
        on = []
        for k in range(2):
            st = self.learner.brain.settle_batch(self.odour_drive(k), steps=cfg.free_steps, tolerance=cfg.tolerance)
            on.append(st.activation[0] > 0.05)
        cls = np.full(self.C.n, 3, dtype=int)
        cls[on[0] & ~on[1]] = 0; cls[on[1] & ~on[0]] = 1; cls[on[0] & on[1]] = 2
        return cls

    def probabilities(self, state_activation: np.ndarray) -> np.ndarray:
        z = state_activation[:, self.outputs] / self.learner.config.temperature
        z = z - z.max(axis=1, keepdims=True); p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        return p

    def probe(self) -> dict:
        """The approach probability and the four outputs under each odour alone."""
        cfg = self.learner.config; out = {}
        for k, odour in enumerate(ODOURS):
            st = self.learner.brain.settle_batch(self.odour_drive(k), steps=cfg.free_steps, tolerance=cfg.tolerance)
            p = self.probabilities(st.activation)[0]
            out[odour] = {"approach": float(p[np.asarray(OUTPUT_ACTION) == 0].sum()), "outputs": [float(x) for x in st.activation[0, self.outputs]], "kc_active": int((st.activation[0] > 0.05)[self.kc].sum())}
        return out

    def policy(self, lesion: tuple[str, ...], greedy: bool, rng: np.random.Generator):
        mask = np.ones(self.C.n)
        for name in lesion:
            mask[list(self.C.populations[name])] = 0.0
        cfg = self.learner.config

        def act(obs: np.ndarray) -> np.ndarray:
            state = self.learner.brain.settle_batch(self.drive(obs), steps=cfg.free_steps, mask=mask, tolerance=cfg.tolerance)
            p = self.probabilities(state.activation)
            choice = np.argmax(p, axis=1) if greedy else np.minimum((p.cumsum(axis=1) < rng.random(len(p))[:, None]).sum(axis=1), len(OUTPUTS) - 1)
            return np.asarray(OUTPUT_ACTION)[choice]
        return act

    def change_by_class(self) -> dict:
        """Mean efficacy change of the plastic synapses by Kenyon cell class and target neuron."""
        change = self.learner.brain.efficacy - self.C.sign
        names = ("fruit KCs", "yeast KCs", "both", "neither")
        out: dict[str, Any] = {}
        for j, target in enumerate(OUTPUTS):
            row = {}
            for c, name in enumerate(names):
                sel = self.plastic & (self.C.post == self.outputs[j]) & (self.kc_class[self.C.pre] == c)
                row[name] = {"synapses": int(sel.sum()), "mean_change": float(change[sel].mean()) if sel.any() else None}
            out[target] = row
        others = self.plastic & ~np.isin(self.C.post, self.outputs)
        out["other MBONs"] = {"synapses": int(others.sum()), "mean_abs_change": float(np.abs(change[others]).mean()) if others.any() else None}
        return out


def rollout(policy: Any, meaning: int, seed: int, searches: int, batch: int, arena_cfg: ArenaConfig) -> dict:
    arena = Arena(batch, seed=seed, meaning=meaning, config=arena_cfg)
    obs = arena.observe(); tally = {"food": 0, "empty": 0, "timeout": 0}; steps: list[int] = []
    while sum(tally.values()) < searches:
        obs, _, _, info = arena.step(policy(obs))
        for key in tally:
            tally[key] += int(info[key].sum())
        steps += info["steps"][info["food"]].tolist()
    total = sum(tally.values())
    return {**{k: v / total for k, v in tally.items()}, "searches": total, "steps_to_food": float(np.mean(steps)) if steps else float("nan")}


def evaluate(agent: FlyAgent, meaning: int, args, lesions: bool) -> dict:
    out = {}
    for name, lesion in (LESIONS.items() if lesions else [("none", ())]):
        if any(n not in agent.C.populations for n in lesion):
            continue
        rng = np.random.default_rng(args.seed + 99)
        out[name] = {"drawn": rollout(agent.policy(lesion, False, rng), meaning, 10_000 + args.seed, args.eval, args.eval_batch, args.arena),
                     "greedy": rollout(agent.policy(lesion, True, rng), meaning, 20_000 + args.seed, args.eval, args.eval_batch, args.arena)}
    return out


def run_fly(args) -> dict:
    cadence.learning.SCALE_CAP = args.scale_cap  # the magnitude a plastic synapse may not exceed (the library's default is 8)
    cfg = LearnerConfig(beta=args.beta, temperature=args.temperature, tolerance=args.tolerance, free_steps=args.free_steps, nudged_steps=args.nudged_steps)
    agent = FlyAgent(args.kind, args.seed, args.gain, args.plastic, cfg, args.budget, args.hops, args.min_count, args.backend)
    ac = ActorCritic(agent.learner, agent.critic, ActorCriticConfig(gamma=args.gamma, lam=args.lam, eta=args.eta, eta_bias=0.0, eta_critic=args.eta_critic), seed=args.seed)
    if args.balance:
        print(f"[{args.kind} s{args.seed}] tonic drive on the avoidance cell: {agent.balance(args.balance_to):.3f} (naive approach {args.balance_to})", flush=True)
    print(f"[{args.kind} s{args.seed}] sub-net {agent.C.n} neurons, {agent.C.synapses} classes, plastic {int(agent.plastic.sum())} ({args.plastic}), gain {args.gain}, KC classes fruit/yeast/both/neither {[int((agent.kc_class[agent.kc] == c).sum()) for c in range(4)]}", flush=True)
    naive_probe = agent.probe()
    naive = evaluate(agent, 0, args, lesions=False)["none"]
    print(f"[{args.kind} s{args.seed}] naive: drawn food {naive['drawn']['food']:.2f} empty {naive['drawn']['empty']:.2f} timeout {naive['drawn']['timeout']:.2f} | greedy food {naive['greedy']['food']:.2f} | p(approach) fruit {naive_probe[ODOURS[0]]['approach']:.2f} yeast {naive_probe[ODOURS[1]]['approach']:.2f}", flush=True)
    arena = Arena(args.batch, seed=args.seed, meaning=0, config=args.arena)
    obs = arena.observe(); curve = Curve(); t0 = time.perf_counter(); swapped_at = None
    pref = {0: [], 1: []}; pref_curve = []
    frozen = args.kind == "frozen"
    frozen_policy = agent.policy((), False, np.random.default_rng(args.seed + 7)) if frozen else None
    report: dict = {}
    for t in range(args.decisions):
        if args.swap_at and t == args.swap_at:
            arena.meaning, swapped_at = 1, t
            print(f"[{args.kind} s{args.seed}] the sugar moves to the {ODOURS[1]} source", flush=True)
        smelled = arena.smelled()
        if frozen:
            action = frozen_policy(obs)
        else:
            choice = ac.act(agent.drive(obs))
            p = agent.probabilities(ac.state.activation)
            for k in (0, 1):
                rows = smelled == k
                if rows.any():
                    pref[k].append(float(p[rows][:, np.asarray(OUTPUT_ACTION) == 0].sum(axis=1).mean()))
            action = np.asarray(OUTPUT_ACTION)[choice]
        obs, reward, finished, info = arena.step(action)
        curve.add(t, info)
        reward = reward + args.shaping * info["approach"]
        if not frozen:
            report = ac.learn(reward, finished, agent.drive(obs))
        if (t + 1) % args.report == 0:
            ws = curve.windows(args.report); w = ws[-1] if ws else {}
            pa = {k: (float(np.mean(v)) if v else float("nan")) for k, v in pref.items()}; pref_curve.append({"decision": t + 1, "approach_fruit": pa[0], "approach_yeast": pa[1]}); pref = {0: [], 1: []}
            print(f"[{args.kind} s{args.seed}] decision {t + 1}: food {w.get('food', float('nan')):.2f} empty {w.get('empty', float('nan')):.2f} timeout {w.get('timeout', float('nan')):.2f} steps {w.get('steps_to_food', float('nan')):.1f} | p(approach) fruit {pa[0]:.2f} yeast {pa[1]:.2f} | delta {report.get('delta', float('nan')):.3f} value {report.get('value', float('nan')):.2f} | {time.perf_counter() - t0:.0f}s ({(time.perf_counter() - t0) / (t + 1):.2f} s/decision)", flush=True)
    wall = time.perf_counter() - t0
    held = evaluate(agent, arena.meaning, args, lesions=not frozen)
    change = agent.learner.brain.efficacy - agent.C.sign
    result = {"summary": agent.summary, "naive": naive, "naive_probe": naive_probe, "curve": curve.windows(args.report), "preference": pref_curve, "wall_seconds": wall, "held_out": held, "probe": agent.probe(), "swapped_at": swapped_at,
              "final_meaning": ODOURS[arena.meaning], "change_by_class": agent.change_by_class() if not frozen else None,
              "mean_abs_scale_change": float(np.abs(change[agent.plastic]).mean()) if agent.plastic.any() else 0.0, "max_abs_scale_change": float(np.abs(change).max()), "synapses_moved": int((np.abs(change) > 1e-6).sum())}
    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.checkpoint, efficacy=agent.learner.brain.efficacy, members=agent.sub.members, w_critic=ac.w_critic, b_critic=ac.b_critic, plastic=np.flatnonzero(agent.plastic), outputs=np.array(agent.outputs), critic=agent.critic)
    return result


def run_mlp(args) -> dict:
    import torch
    torch.manual_seed(args.seed)
    net = torch.nn.Sequential(torch.nn.Linear(6, args.hidden), torch.nn.Tanh(), torch.nn.Linear(args.hidden, 3))
    opt = torch.optim.Adam(net.parameters(), lr=args.mlp_lr)
    arena = Arena(args.batch, seed=args.seed, meaning=0, config=args.arena); obs = arena.observe(); curve = Curve(); rng = np.random.default_rng(args.seed)

    def policy(o, greedy=False):
        with torch.no_grad():
            logits = net(torch.tensor(o, dtype=torch.float32))[:, :2]
        p = torch.softmax(logits, dim=1).numpy()
        return np.argmax(p, axis=1) if greedy else np.minimum((p.cumsum(axis=1) < rng.random(len(p))[:, None]).sum(axis=1), 1)
    naive = rollout(policy, 0, 10_000 + args.seed, args.eval, args.eval_batch, args.arena); t0 = time.perf_counter(); swapped_at = None
    for t in range(args.decisions):
        if args.swap_at and t == args.swap_at:
            arena.meaning, swapped_at = 1, t
        x = torch.tensor(obs, dtype=torch.float32); out = net(x); dist = torch.distributions.Categorical(logits=out[:, :2]); a = dist.sample()
        obs, reward, finished, info = arena.step(a.numpy()); curve.add(t, info); reward = reward + args.shaping * info["approach"]
        with torch.no_grad():
            v_next = net(torch.tensor(obs, dtype=torch.float32))[:, 2]
        target = torch.tensor(reward, dtype=torch.float32) + args.gamma * v_next * torch.tensor(~finished, dtype=torch.float32)
        adv = target - out[:, 2]; loss = -(dist.log_prob(a) * adv.detach()).mean() + 0.5 * (adv ** 2).mean() - 0.01 * dist.entropy().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (t + 1) % args.report == 0:
            w = curve.windows(args.report)[-1]
            print(f"[mlp s{args.seed}] decision {t + 1}: food {w['food']:.2f} empty {w['empty']:.2f} timeout {w['timeout']:.2f} steps {w['steps_to_food']:.1f} {time.perf_counter() - t0:.0f}s", flush=True)
    held = {"none": {"drawn": rollout(policy, arena.meaning, 10_000 + args.seed, args.eval, args.eval_batch, args.arena), "greedy": rollout(lambda o: policy(o, True), arena.meaning, 20_000 + args.seed, args.eval, args.eval_batch, args.arena)}}
    return {"summary": {"kind": "mlp", "seed": args.seed, "hidden": args.hidden, "lr": args.mlp_lr, "parameters": int(sum(p.numel() for p in net.parameters())), "actions": list(ACTIONS)}, "naive": {"drawn": naive}, "curve": curve.windows(args.report), "wall_seconds": time.perf_counter() - t0, "held_out": held, "swapped_at": swapped_at, "final_meaning": ODOURS[arena.meaning]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["connectome", "shuffled", "frozen", "mlp"], default="connectome"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decisions", type=int, default=4000); p.add_argument("--batch", type=int, default=16); p.add_argument("--gain", type=float, default=None)
    p.add_argument("--plastic", default="kc>mbon", help="kc>mbon, mb or all"); p.add_argument("--budget", type=int, default=12000); p.add_argument("--hops", type=int, default=2); p.add_argument("--min-count", type=float, default=5.0); p.add_argument("--backend", default="torch")
    p.add_argument("--beta", type=float, default=0.1); p.add_argument("--temperature", type=float, default=0.05); p.add_argument("--tolerance", type=float, default=1e-3); p.add_argument("--free-steps", type=int, default=40); p.add_argument("--nudged-steps", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.95); p.add_argument("--lam", type=float, default=0.9); p.add_argument("--eta", type=float, default=10.0); p.add_argument("--eta-critic", type=float, default=0.5)
    p.add_argument("--scale-cap", type=float, default=3.0); p.add_argument("--balance", action="store_true", help="a declared tonic drive on the avoidance cell setting the naive approach probability"); p.add_argument("--balance-to", type=float, default=0.8); p.add_argument("--shaping", type=float, default=0.0); p.add_argument("--swap-at", type=int, default=0)
    p.add_argument("--eval", type=int, default=128); p.add_argument("--eval-batch", type=int, default=64); p.add_argument("--report", type=int, default=250)
    p.add_argument("--task", choices=["tmaze", "valence", "steer"], default="tmaze"); p.add_argument("--hidden", type=int, default=32); p.add_argument("--mlp-lr", type=float, default=3e-3); p.add_argument("--checkpoint", default=""); p.add_argument("--out", default="")
    args = p.parse_args()
    args.arena = ArenaConfig(task=args.task, limit=40 if args.task == "tmaze" else 120)
    if args.gain is None:
        args.gain = float(json.loads((ROOT / "receipts" / "g3_instinct_facts.json").read_text())["body"]["connectome"]["gain"])
    result = run_mlp(args) if args.kind == "mlp" else run_fly(args)
    result["args"] = {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in vars(args).items()}; result["cadence"] = cd.__version__
    result["fixture_sha256"] = json.loads(MANIFEST_PATH.read_text())["fixture_sha256"]
    for lesion, r in result["held_out"].items():
        print(f"held out [{lesion}]: drawn food {r['drawn']['food']:.3f} empty {r['drawn']['empty']:.3f} | greedy food {r['greedy']['food']:.3f} empty {r['greedy']['empty']:.3f} steps {r['greedy']['steps_to_food']:.1f}", flush=True)
    if "probe" in result:
        for odour, pr in result["probe"].items():
            print(f"probe [{odour}]: p(approach) {pr['approach']:.3f} outputs {[round(x, 3) for x in pr['outputs']]}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
