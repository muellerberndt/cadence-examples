#!/usr/bin/env python3
"""A small brain, its payload, its settle cases and a recorded run of the library's actor-critic,
for the engine's parity test (and as the worked example of export.py).

    python engine/make_fixture.py --out /tmp/engine_fixture
    node engine/parity.mjs /tmp/engine_fixture/brain.json /tmp/engine_fixture/cases.json /tmp/engine_fixture/lessons.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cadence as cd  # noqa: E402
from engine.export import record_lessons, settle_cases, write_payload  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/engine_fixture"); ap.add_argument("--decisions", type=int, default=24); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    connectome = cd.layered(4, 24, 2, density=0.6, seed=args.seed)  # populations input, hidden, output
    brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0))
    write_payload(brain, out / "brain.json")
    settle_cases(brain, {"first pair": {"input": 0.6}, "half": {"output": 0.3}}, ["input", "hidden", "output"], steps=30, out=out / "cases.json")
    # a contextual bandit, one stream: action k pays in context k
    rng = np.random.default_rng(args.seed + 1)
    learner = cd.Learner(brain, connectome.populations["output"], cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12, free_steps=60), reciprocal=False)
    drives, contexts = [], []
    for _ in range(args.decisions + 1):
        k = int(rng.integers(0, 2)); levels = np.zeros(connectome.n); levels[[2 * k, 2 * k + 1]] = 1.0
        drives.append(levels); contexts.append(k)
    # the rewards depend on the actions the library takes, so the recording runs the library and pays action == context
    rewards, dones = [], []
    from cadence.plasticity import ActorCritic, ActorCriticConfig
    from engine.export import _Replay
    probe = ActorCritic(cd.Learner(cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)), connectome.populations["output"], learner.config, reciprocal=False), connectome.populations["hidden"], ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, eta_critic=0.3), seed=args.seed)
    probe.rng = _Replay(args.seed)
    amp = brain.neuron_model.stimulus_amplitude
    for t in range(args.decisions):
        a = int(probe.act(amp * drives[t][None, :])[0]); r = 1.0 if a == contexts[t] else 0.0; d = (t % 6) == 5
        rewards.append(r); dones.append(d); probe.learn(np.array([r]), np.array([d]), amp * drives[t + 1][None, :])
    rec = record_lessons(learner, connectome.populations["hidden"], drives, rewards, dones, config=ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, eta_critic=0.3), seed=args.seed, out=out / "lessons.json", phases=True)
    print(f"{out}: brain {connectome.n} neurons {connectome.synapses} synapses; {len(rec['decisions'])} decisions recorded, {len(rec['efficacy'])} plastic synapses")


if __name__ == "__main__":
    main()
