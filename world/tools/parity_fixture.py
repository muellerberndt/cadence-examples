"""A fixture for the JavaScript port of S02: a snapshot of a living remembered-world brain,
the moments it was then fed and what it decided, its store contents after every event, and
the world's private state and events for the world port.

    python world/tools/parity_fixture.py --out runs/world/parity

The life is driven by run.py's own loop functions (``run_explore``, ``run_remember``,
``run_cue``, ``run_words``, with ``close_episode`` and the new-episode requests they call),
so the fixture follows the stage's flow as run.py changes; a recording ``Life`` writes what
the brain saw and did, and a recording ``World`` writes every privileged operation of the
curriculum, every reset and act, the moment each produced and the private state after it.

Writes ``snapshot.json`` (the web export after the explore phase, at an episode boundary,
with the stores, the planner state, the brain's bookkeeping, the recall read, the world
state and the stage config under ``extra``; the records tables in float32),
``records_f64.json`` (the records head's mean and tables in float64, for the harness),
``stream.jsonl`` (one line per moment the brain was fed, with the decision, the exploration
rate in effect, the planner counters, the plan, the stores, the recall read, the learning
report and the record-write count), ``world.jsonl`` (one line per world event: the
operations before it, the event, the moment, the state after) and ``final.json``
(parameter arrays, the records head in float64, store contents, counters and tolerances
after the stream).

The stream: one explore episode, ``--remember`` remembered requests at delay 8, ``--cue``
cue episodes at delay 8, ``--words`` word episodes (teaching then the name request). The
world's horizon is ``--horizon`` so the explore episode stays short; the map, the doors and
the walls come from the stage config. ``--max-nodes`` bounds the search so both sides
finish in seconds; the value travels in ``extra.planner``.

``--graph-override`` merges settings over the stage config, so a fixture can carry the
settings the stage config leaves at their defaults and the port is checked on those paths
too:

    python world/tools/parity_fixture.py --out runs/world/parity_gains \\
        --graph-override '{"graph": {"field_gains": {"x": 1.5}, "action_gain": 3.0},
                           "records": {"pathways": "fields", "fan_in": 3}}'
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from agent.brain import ReplayConfig
from agent.life import seed_for
from agent.web import export, records_export
from world import run as runner
from world.brain import Brain
from world.env import SIZE, World, WorldConfig
from world.tasks import Curriculum

WORLD_STATE_FORMAT = "cadence-s02-world/1"


SECTIONS = ("graph", "world", "actor", "replay", "records", "planner", "world_config")


def apply_override(config: dict, override: dict) -> dict:
    """The stage config with ``override`` merged into it, one level deep.

    A mapping of section to settings ({"graph": {"action_gain": 3.0}}) merges into those
    sections; a mapping of no section name is a mapping of graph settings."""
    if not isinstance(override, dict) or any(not isinstance(k, str) for k in override):
        raise SystemExit("--graph-override takes a JSON object")
    if not set(override) & set(SECTIONS):
        override = {"graph": override}
    out = dict(config)
    for section, settings in override.items():
        if section not in SECTIONS:
            raise SystemExit(f"--graph-override names the section {section!r}; the sections are {', '.join(SECTIONS)}")
        if not isinstance(settings, dict):
            raise SystemExit(f"--graph-override: section {section!r} takes a JSON object of settings")
        out[section] = {**(out.get(section) or {}), **settings}
    return out


def moment_record(m) -> dict:
    return {
        "life_id": m.life_id, "episode_id": m.episode_id, "event_id": m.event_id, "tick": m.tick, "dt": m.dt,
        "observation": {k: np.asarray(v).tolist() for k, v in m.observation.items()},
        "observed": {k: np.asarray(v).astype(int).tolist() for k, v in m.observed.items()},
        "action_mask": m.action_mask.astype(int).tolist(), "feedback_for": m.feedback_for, "executed": m.executed,
        "reward": m.reward, "reward_known": m.reward_known, "terminated": m.terminated, "truncated": m.truncated,
        "final_observation": None if m.final_observation is None else {k: np.asarray(v).tolist() for k, v in m.final_observation.items()},
        "goal": None if m.goal is None else np.asarray(m.goal).tolist(),
    }


def cell_record(c):
    return None if c is None else [int(c[0]), int(c[1])]


def reward_rule_record(rule) -> dict | None:
    """The closure a task set, as data: tasks.py `_reach_reward` (target) or `arm_cue` (paid)."""
    if rule is None:
        return None
    free = inspect.getclosurevars(rule).nonlocals
    if "target" in free:
        return {"kind": "reach", "target": cell_record(free["target"])}
    if "paid" in free:
        return {"kind": "cue", "paid": cell_record(free["paid"])}
    raise SystemExit(f"unknown reward rule {rule.__qualname__}: the fixture tool knows _reach_reward and arm_cue")


def word_rule_record(rule) -> dict | None:
    """tasks.py's joint-attention lambda: `word if k in w.visible_objects() else 0`, keyed by its defaults."""
    if rule is None:
        return None
    defaults = rule.__defaults__ or ()
    if len(defaults) != 2:
        raise SystemExit(f"unknown word rule {rule}: the fixture tool knows the joint-attention lambda (k, word)")
    return {"kind": "joint_attention", "object": int(defaults[0]), "word": int(defaults[1])}


def world_state(world: World) -> dict:
    """The world's private state, in the shape web/world.js `World.fromState` loads and `state()` returns."""
    return {
        "format": WORLD_STATE_FORMAT, "size": SIZE,
        "config": {"doors": world.config.doors, "walls": world.config.walls, "horizon": world.config.horizon, "inspect_visible": world.config.inspect_visible, "junction": list(world.config.junction)},
        "seed": int(world.seed), "life_id": world.life_id,
        "edges": [[cell_record(a), cell_record(b), kind] for (a, b), kind in world._edges.items()],
        "doors": [[cell_record(a), cell_record(b)] for (a, b) in world._doors],
        "colors": [int(c) for c in world._colors], "words": [int(w) for w in world._word_of],
        "objects": [[int(k), cell_record(c)] for k, c in world._objects.items()],
        "agent": cell_record(world._agent), "carrying": world._carrying, "outcome": int(world._outcome), "inspected": bool(world._inspected),
        "move": [int(world._move[0]), int(world._move[1])],
        "word": int(world._word), "word_rule": word_rule_record(world._word_rule), "goal": np.asarray(world._goal).tolist(),
        "episode": int(world._episode), "event": int(world._event), "tick": int(world._tick), "pending": world._pending, "decision_action": world._decision_action,
        "door_rule": world._door_rule, "mask": world._mask.astype(int).tolist(), "deadline": world._deadline,
        "reward_rule": reward_rule_record(world._reward_rule),
    }


class RecordingWorld(World):
    """env.py's World, recording every privileged operation, reset and act with the moment and the state after."""

    def __post_init__(self) -> None:
        self.events: list[dict] = []
        self.ops: list[dict] = []
        self.recording = False
        self._last_rule = None
        super().__post_init__()

    def _sync_rule(self) -> None:
        if self._reward_rule is not self._last_rule:  # tasks.py assigns the rule directly
            self.ops.append({"op": "reward_rule", "rule": reward_rule_record(self._reward_rule)})
            self._last_rule = self._reward_rule

    def _op(self, **op) -> None:
        self._sync_rule()
        self.ops.append(op)

    def _record_event(self, event: dict, m) -> None:
        self._sync_rule()
        if self.recording:
            self.events.append({"ops": self.ops, "event": event, "moment": moment_record(m), "state": world_state(self)})
        self.ops = []

    def place_object(self, k, cell) -> None:
        self._op(op="place_object", object=int(k), cell=cell_record(cell))
        super().place_object(k, cell)

    def set_word(self, word) -> None:
        self._op(op="set_word", word=int(word))
        super().set_word(word)

    def set_word_rule(self, rule) -> None:
        self._op(op="set_word_rule", rule=word_rule_record(rule))
        super().set_word_rule(rule)
        if rule is not None:  # the rule as data must agree with the closure it describes
            k, word = word_rule_record(rule)["object"], word_rule_record(rule)["word"]
            if int(rule(self)) != (word if k in self.visible_objects() else 0):
                raise SystemExit("the word rule is not the joint-attention rule the fixture tool describes")

    def set_goal(self, goal) -> None:
        self._op(op="set_goal", goal=np.asarray(goal, float).tolist())
        super().set_goal(goal)

    def set_door_rule(self, rule) -> None:
        self._op(op="set_door_rule", rule=str(rule))
        super().set_door_rule(rule)

    def set_deadline(self, steps) -> None:
        self._op(op="set_deadline", steps=None if steps is None else int(steps))
        super().set_deadline(steps)

    def set_mask(self, legal) -> None:
        self._op(op="set_mask", legal=None if legal is None else np.asarray(legal).astype(int).tolist())
        super().set_mask(legal)

    def reset(self, *, agent=None):
        m = super().reset(agent=agent)
        self._record_event({"op": "reset", "agent": cell_record(self._agent)}, m)
        return m

    def act(self, decision_id, action):
        m = super().act(decision_id, action)
        self._record_event({"op": "act", "decision_id": int(decision_id), "action": int(action)}, m)
        return m


def stores_record(brain: Brain) -> dict:
    p, w, g = brain.places.memory, brain.words.memory, brain.map.memory
    return {
        "places": {"memory": {"strength": p.strength[0].tolist(), "mass": p.mass.tolist(), "writes": int(p.writes)}, "writes": int(brain.places.writes)},
        "cue": {"value": brain.cue.value.tolist()},
        "words": {"memory": {"strength": w.strength[0].tolist(), "mass": w.mass.tolist(), "writes": int(w.writes), "consolidated": w.consolidated.tolist()}, "writes": int(brain.words.writes)},
        "map": {"memory": {"strength": g.strength[0].tolist(), "mass": g.mass.tolist(), "writes": int(g.writes)}, "writes": int(brain.map.writes)},
    }


def planner_record(brain: Brain) -> dict:
    pl = brain.planner
    return {"max_nodes": int(pl.max_nodes), "depth": int(pl.depth), "seed": int(pl.seed), "rng": int(pl.rng.state), "fallbacks": int(pl.fallbacks), "pruned": int(pl.pruned)}


def counters_record(brain: Brain) -> dict:
    return {"searches": int(brain.searches), "reused": int(brain.reused), "nodes": int(brain.nodes), "fallbacks": int(brain.planner.fallbacks), "pruned": int(brain.planner.pruned), "rng": int(brain.planner.rng.state)}


def bookkeeping_record(brain: Brain) -> dict:
    return {
        "visited": [cell_record(c) for c in brain.visited], "inspected": [cell_record(c) for c in brain.inspected],
        "plan": [int(a) for a in brain.plan], "plan_cells": [cell_record(c) for c in brain.plan_cells],
        "plan_key": None if brain.plan_key is None else [int(brain.plan_key[0]), cell_record(brain.plan_key[1])],
        "searches": int(brain.searches), "reused": int(brain.reused), "nodes": int(brain.nodes),
    }


class RecordingLife(runner.Life):
    """run.py's Life, writing one stream line per moment the brain is fed."""

    def __init__(self, brain: Brain) -> None:
        super().__init__(brain, is_agent=True)
        self.stream = None
        self.phase = "explore"
        self.lines = 0

    def step(self, m):
        brain: Brain = self.controller
        epsilon = float(brain.agent.config.actor.epsilon)
        reused_before, searches_before = brain.reused, brain.searches
        d = super().step(m)
        if self.stream is not None:
            agent = brain.agent
            record = {
                "phase": self.phase, "epsilon": epsilon, "moment": moment_record(m), "decision": None if d is None else d.to_dict(),
                "parameter_version": int(agent.parameter_version),
                "efficacy_sha256": hashlib.sha256(np.ascontiguousarray(agent.brain.efficacy).tobytes()).hexdigest(),
                "learning": {k: v for k, v in agent.last_report.get("learning", {}).items()},
                "planner": counters_record(brain), "reused": brain.reused > reused_before, "searched": brain.searches > searches_before,
                "plan": [int(a) for a in brain.plan], "plan_cells": [cell_record(c) for c in brain.plan_cells],
                "plan_key": None if brain.plan_key is None else [int(brain.plan_key[0]), cell_record(brain.plan_key[1])],
                "stores": stores_record(brain), "recall": agent._last_recall[0].tolist(),
                "record_writes": int(agent.ledger.record_writes), "imagined": int(agent.ledger.imagined),
            }
            self.stream.write(json.dumps(record) + "\n")
            self.lines += 1
        return d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE.parent / "config.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--babble", type=int, default=300, help="explore steps before the snapshot (whole episodes, exploration rate 1)")
    parser.add_argument("--horizon", type=int, default=32, help="the fixture world's episode horizon")
    parser.add_argument("--max-nodes", type=int, default=None, help="the search budget (default: the config's)")
    parser.add_argument("--graph-override", type=str, default=None, metavar="JSON", help='config settings merged over the stage config, to cover engine paths the default config leaves at their defaults: a mapping of section to settings ({"graph": {"action_gain": 3.0}, "records": {"fan_in": 3}}), or a flat mapping of graph settings ({"action_gain": 3.0})')
    parser.add_argument("--remember", type=int, default=2, help="remembered requests at delay 8 in the stream")
    parser.add_argument("--cue", type=int, default=2, help="cue episodes at delay 8 in the stream")
    parser.add_argument("--words", type=int, default=1, help="word episodes in the stream")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.max_nodes is not None:
        config = {**config, "planner": {**config.get("planner", {}), "max_nodes": args.max_nodes}}
    if args.graph_override:
        config = apply_override(config, json.loads(args.graph_override))
        print(f"config overridden: graph {json.dumps({k: v for k, v in config['graph'].items() if k in ('field_gains', 'action_gain', 'input_gain')})}, records {json.dumps(config.get('records'))}")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wcfg = WorldConfig(**{**config["world_config"], "horizon": args.horizon})
    world = RecordingWorld(wcfg, seed=seed_for("S02", "parity", args.seed, 0, 0), life_id="parity")
    cur = Curriculum(world, seed=seed_for("S02", "parity", args.seed, 0, 1))
    brain = Brain(config, args.seed)
    brain.agent.config = replace(brain.agent.config, replay=ReplayConfig(**config["replay"]))
    life = RecordingLife(brain)
    latencies: list[float] = []
    counter = [0]
    meta = {"seed": args.seed, "split": "parity", "life": "parity"}
    # the life before the snapshot: exploration at rate 1, whole episodes, as run.py's explore phase
    runner.run_explore(life, world, cur, args.babble, None, meta, latencies, counter)
    world.events = []
    assert brain.agent.pending[0] is None and not world.ops, "the snapshot is taken at an episode boundary"
    extra = {
        "stage": "S02", "boundary": True, "next_moment": None,
        "config": config, "planner": planner_record(brain), "brain": bookkeeping_record(brain), "stores": stores_record(brain),
        "recall": {"last": brain.agent._last_recall[0].tolist(), "gain": config["graph"].get("recall_gain", 2.0), "width": len(brain.agent.ports.recall)},
        "world": world_state(world), "curriculum": {"cue_rule": int(cur.cue_rule), "seed": int(seed_for("S02", "parity", args.seed, 0, 1))},
        "world_seed": int(world.seed), "steps_before": int(counter[0]),
    }
    export(brain.agent, args.out / "snapshot.json", extra=extra, records_sidecar_path=args.out / "records_f64.json")
    babble_seconds = time.time() - t0
    # the stream: run.py's phases on the recorded life
    world.recording = True
    successes: dict[str, list[float]] = {}
    with open(args.out / "stream.jsonl", "w") as f:
        life.stream = f
        life.phase = "explore"
        runner.run_explore(life, world, cur, 1, None, meta, latencies, counter)
        life.phase = "remember-8"
        successes["remember_8"] = runner.run_remember(life, world, cur, args.remember, 8, None, meta, latencies, counter)
        life.phase = "cue-8"
        successes["cue_8"] = runner.run_cue(life, world, cur, args.cue, 8, None, meta, latencies, counter, decision_epsilon=config["actor"]["epsilon"])
        life.phase = "word"
        successes["words"] = runner.run_words(life, world, cur, args.words, None, meta, latencies, counter)
        life.stream = None
    with open(args.out / "world.jsonl", "w") as f:
        f.writelines(json.dumps(event) + "\n" for event in world.events)
    agent = brain.agent
    final = {
        "efficacy": agent.brain.efficacy.tolist(), "bias": agent.brain.bias.tolist(),
        "world_velocity": agent.world.velocity.tolist(), "critic_w": agent.w_critic.tolist(), "critic_b": agent.b_critic,
        "context_trace": agent.context.trace[0].tolist(), "rng": {"action": agent.rng[0].state, "replay": agent.replay_rng.state},
        "parameter_version": agent.parameter_version, "ledger": agent.ledger.to_dict(), "ring_stored": len(agent.ring), "ring_at": agent.ring_at,
        "stores": stores_record(brain), "planner": counters_record(brain), "brain": bookkeeping_record(brain), "recall": agent._last_recall[0].tolist(),
        "world": world_state(world), "curriculum": {"cue_rule": int(cur.cue_rule)}, "successes": successes,
        "records": records_export(agent, "<f8"),
        "stream_lines": life.lines, "world_events": len(world.events), "real_steps": counter[0],
        "seconds": {"before_snapshot": babble_seconds, "stream": time.time() - t0 - babble_seconds},
        "tolerances": {"activation": 1e-9, "efficacy": 1e-8, "prediction": 1e-8, "stores": 1e-12, "recall": 1e-12, "records": 1e-10},
    }
    (args.out / "final.json").write_text(json.dumps(final))
    print(f"fixture written: {args.out}; {counter[0]} real steps ({extra['steps_before']} before the snapshot), stream {life.lines} moments, world {len(world.events)} events; final parameter version {agent.parameter_version}; searches {brain.searches}, reused {brain.reused}, expansions {brain.nodes}; successes {successes}; record writes {agent.ledger.record_writes}, imagined {agent.ledger.imagined}; {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
