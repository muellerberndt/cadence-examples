"""Controls of the remembered world, outside the candidate's imports.

``PrivilegedPlanner``: the exact map and the true object positions (upper reference).
``TabularController``: an observed-only tabular transition model (counts of visible
consequences per (visible state, action)) with the same declared stores and the same
best-first search as the candidate. ``RandomController``: uniform actions.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from agent.life import Moment
from .brain import (
    CONSEQUENCES,
    CueStore,
    MapStore,
    PlaceStore,
    WordStore,
    WorldPlanner,
    observe_from,
)
from .env import ACTIONS, DIRECTIONS, FIELDS, OBJECTS, OUTCOMES, SIZE, World
from .tasks import (
    CUE_WORDS,
    GOAL_CUE,
    GOAL_EXPLORE,
    GOAL_OBJECT,
    GOAL_WORD,
    ROOMS,
    manhattan,
)


class RandomController:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def step(self, m: Moment):
        if m.terminated or m.truncated:
            return None
        legal = np.flatnonzero(m.action_mask)
        return int(legal[self.rng.integers(len(legal))])


class PrivilegedPlanner:
    """Reads the world's private state: the upper reference, never a candidate."""

    def __init__(self, world: World, cue_rule: int) -> None:
        self.world = world
        self.cue_rule = cue_rule
        self.last_word = 0  # the reference holds the cue the same way the candidate's cue store does

    def step(self, m: Moment):
        word = int(m.observation["word"].argmax())
        if word > 0:
            self.last_word = word
        if m.terminated or m.truncated:
            return None
        goal = int(np.asarray(m.goal).argmax())
        state = self.world.state
        here = state["agent"]
        target = None
        if GOAL_OBJECT <= goal < GOAL_OBJECT + OBJECTS:
            target = state["objects"][goal - GOAL_OBJECT]
        elif goal == GOAL_WORD:
            word = int(m.observation["word"].argmax()) - 1
            k = int(np.flatnonzero(state["words"] == word)[0])
            target = state["objects"][k]
        elif goal == GOAL_CUE:
            cue = CUE_WORDS.index(self.last_word) if self.last_word in CUE_WORDS else 0
            target = ROOMS[cue ^ self.cue_rule]
        if target is None or target == here:
            return ACTIONS.index("inspect")
        path = self.world.shortest_path(here, target)
        if path is None:
            path = self.world.shortest_path(here, target, doors_open=True)
            if path is None:
                return ACTIONS.index("inspect")
            # a door blocks the way: interact when standing next to a closed door on the path
            if self.world._passage(here, path[0]) == "door_closed":
                return ACTIONS.index("interact")
        return ACTIONS.index(path[0])


class TabularModel:
    """Counts of each consequence class per (visible state key, action); unseen keys predict persistence."""

    def __init__(self) -> None:
        self.counts: dict[tuple, dict[str, np.ndarray]] = defaultdict(lambda: {n: np.zeros(FIELDS[n]) for n in CONSEQUENCES})
        self.reward: dict[tuple, list[float]] = defaultdict(list)

    @staticmethod
    def key(observation: dict[str, np.ndarray], action: int) -> tuple:
        cue = int(observation["cue"].argmax()) if "cue" in observation else 0
        return tuple(int(observation[n].argmax()) for n in ("x", "y", "passage_north", "passage_east", "passage_south", "passage_west", "here_object", "carrying", "word")) + (cue, int(action))

    def learn(self, observation: dict[str, np.ndarray], action: int, outcome: Moment) -> None:
        k = self.key(observation, action)
        for n in CONSEQUENCES:
            self.counts[k][n][int(outcome.observation[n].argmax())] += 1
        if outcome.reward_known:
            self.reward[k].append(outcome.reward)

    def predict_batch(self, observations, actions, *, observed=None, goal=None):
        out = []
        for obs, a in zip(observations, actions, strict=True):
            k = self.key(obs, a)
            pred = {}
            for n in CONSEQUENCES:
                c = self.counts[k][n] if k in self.counts else None
                if c is None or c.sum() == 0:
                    pred[f"next_{n}"] = np.eye(3)[1] if n in ("dx", "dy") else np.asarray(obs[n], float)
                else:
                    pred[f"next_{n}"] = c / c.sum()
            r = self.reward.get(k, [])
            pred["reward"] = np.array([float(np.mean(r)) if r else 0.0])
            out.append(pred)
        return out


class TabularController:
    """The tabular model behind the candidate's own stores and search."""

    def __init__(self, seed: int, planner: WorldPlanner, epsilon: float = 0.0) -> None:
        self.model = TabularModel()
        self.planner = planner
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.places, self.words, self.cue, self.map = PlaceStore(), WordStore(), CueStore(), MapStore()
        self.inspected: set = set()
        self.pending: tuple[dict, int] | None = None
        self.learning = True

    def witness(self, m: Moment) -> None:
        o = m.observation
        cell = (int(o["x"].argmax()), int(o["y"].argmax()))
        if m.tick == 0 and m.feedback_for is None:
            self.inspected.clear()
        if m.observed["x"].all() and m.observed["passage_north"].all():
            self.map.see(cell, observe_from({k: np.asarray(v) for k, v in o.items()})["passages"])
        word = int(o["word"].argmax())
        self.cue.see(word)
        here = int(o["here_object"].argmax()) - 1
        visible = []
        if here >= 0:
            self.places.see(here, cell); visible.append(here)
        else:
            for k in range(OBJECTS):
                where, known = self.places.recall(k)
                if known >= 0.5 and where == cell:
                    self.places.forget(k)
        if int(o["outcome"].argmax()) == OUTCOMES.index("inspected"):
            self.inspected.add(cell)
            for d, (dx, dy) in DIRECTIONS.items():
                if m.observed[f"{d}_object"].all():
                    k = int(o[f"{d}_object"].argmax()) - 1
                    if k >= 0:
                        self.places.see(k, (cell[0] + dx, cell[1] + dy)); visible.append(k)
        if int(np.asarray(m.goal).argmax()) == GOAL_EXPLORE:
            self.words.scene(word, visible)

    def step(self, m: Moment):
        self.witness(m)
        if m.has_feedback and self.pending is not None and self.learning:
            self.model.learn(*self.pending, m)
        self.pending = None
        if m.terminated or m.truncated:
            return None
        observation = {k: np.asarray(v) for k, v in m.observation.items()}
        observation["cue"] = self.cue.value.copy()  # the same declared store, read by the tabular key
        if self.rng.random() < self.epsilon:
            legal = np.flatnonzero(m.action_mask)
            action = int(legal[self.rng.integers(len(legal))])
        else:
            goal = np.asarray(m.goal)
            which = int(goal.argmax())
            cell = observe_from(observation)["cell"]
            if which == GOAL_CUE:
                action, _, _ = self.planner.greedy_reward(self.model, observation, goal, m.action_mask, m.observed)
            else:
                target = None
                k = None
                if GOAL_OBJECT <= which < GOAL_OBJECT + OBJECTS:
                    k = which - GOAL_OBJECT
                elif which == GOAL_WORD:
                    k, _ = self.words.referent(int(observation["word"].argmax()))
                if which != GOAL_EXPLORE and k is not None:
                    target, _ = self.places.recall(k)
                if target is None or target == cell:
                    candidates = [(x, y) for y in range(SIZE) for x in range(SIZE) if (x, y) not in self.inspected and (x, y) != cell] or [(x, y) for y in range(SIZE) for x in range(SIZE) if (x, y) != cell]
                    target = min(candidates, key=lambda c: (manhattan(cell, c), c))
                action = self.planner.search(self.model, observation, goal, target, m.action_mask, self.map, m.observed)[0]
        self.pending = (observation, action)
        return int(action)
