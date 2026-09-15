"""The remembered world's brain: the S00 agent, three declared stores and a supplied search.

The world head predicts the next visible consequences of an action: position, the four
passages, the object and colour in the cell, what is carried, the outcome, and the reward.
Three stores with supplied categorical keys hold witnessed evidence: object -> place with a
known flag (``PlaceStore``), the episode's cue word (``CueStore``) and word -> referent
co-occurrence (``WordStore``). Their reads enter the recall port every moment, so the
learned heads can condition on them, and the planner reads them explicitly. Decisions are
supplied search over the learned model: a best-first search from the current observation
through imagined consequences toward a target cell, or toward the action with the highest
predicted reward. Unknown targets lead to exploration (the nearest cell not yet inspected).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from typing import Any

import cadence as cd
import numpy as np

from agent.brain import ActorConfig, Agent, AgentConfig, Field, GraphSpec, RecordsConfig, WorldConfig
from agent.life import Moment
from .env import ACTIONS, DIRECTIONS, FIELDS, OBJECTS, OUTCOMES, PASSAGE, SIZE, WORDS
from .tasks import GOAL_CUE, GOAL_EXPLORE, GOAL_OBJECT, GOAL_WORD, manhattan

CONSEQUENCES = ("dx", "dy", "passage_north", "passage_east", "passage_south", "passage_west", "here_object", "here_color", "carrying", "outcome")
CELLS = SIZE * SIZE


def cell_index(x: int, y: int) -> int:
    return y * SIZE + x


def graph_spec(config: dict[str, Any]) -> GraphSpec:
    observation = tuple(Field(name, "categorical", width) for name, width in FIELDS.items())
    prediction = tuple(Field(f"next_{name}", "categorical", FIELDS[name], source=name) for name in CONSEQUENCES)
    prediction += (Field("reward", "continuous", 1, -1.0, 1.0, source="reward"),)
    g = config["graph"]
    return GraphSpec(
        observation=observation, prediction=prediction, action_fields=(len(ACTIONS),), goal=8,
        perceptual=g.get("perceptual", 0), workspace=g["workspace"], dynamics=g["dynamics"], episodic=g["episodic"],
        missing_flags=True, flags_per_field=True, density=g.get("density", 1.0), scale=g.get("scale", 1.0),
        source_scale=g.get("source_scale", 2.0), sensory_to_dynamics=g.get("sensory_to_dynamics", True), bias=g.get("bias", 0.25),
        input_gain=g.get("input_gain", 2.0), readout_init=g.get("readout_init", 0.0), dt=g.get("dt", 0.5), lateral=g.get("lateral", 0.0),
    )


def agent_config(config: dict[str, Any], **over: Any) -> AgentConfig:
    cfg = AgentConfig(graph=graph_spec(config), world=WorldConfig(**config["world"]), actor=ActorConfig(**config["actor"]),
                      records=None if config.get("records") is None else RecordsConfig(**config["records"]),
                      context_decay=config.get("context_decay", 0.8), context_amplitude=config.get("context_amplitude", 1.0), controller="planner")
    return replace(cfg, **over) if over else cfg


# -- stores with supplied keys


class PlaceStore:
    """object -> (place one-hot over 16 cells, known flag); one exposure writes it, a visit that
    finds nothing erases it. Delta-rule fast synapses, one stream."""

    def __init__(self) -> None:
        self.memory = cd.FastSynapses(np.arange(OBJECTS), np.arange(OBJECTS, OBJECTS + CELLS + 1), rule="delta")
        self.memory.reset(1)
        self.writes = 0

    def see(self, k: int, cell: tuple[int, int]) -> None:
        value = np.zeros((1, CELLS + 1))
        value[0, cell_index(*cell)] = 1.0
        value[0, -1] = 1.0
        self.memory.observe(np.eye(OBJECTS)[[k]], value)
        self.writes += 1

    def forget(self, k: int) -> None:
        self.memory.observe(np.eye(OBJECTS)[[k]], np.zeros((1, CELLS + 1)))
        self.writes += 1

    def recall(self, k: int) -> tuple[tuple[int, int] | None, float]:
        read = self.memory.recall(np.eye(OBJECTS)[[k]])[0]
        known = float(read[-1])
        if known < 0.5:
            return None, known
        c = int(np.argmax(read[:-1]))
        return (c % SIZE, c // SIZE), known

    def drive(self, k: int | None) -> np.ndarray:
        if k is None:
            return np.zeros(CELLS + 1)
        return self.memory.recall(np.eye(OBJECTS)[[k]])[0]

    def erase(self) -> None:
        self.memory.reset(1)


class MapStore:
    """cell -> its four passages with a known flag: the witnessed map. Standing in a cell shows
    its passages; one exposure writes them, a later exposure overwrites (a door that changed)."""

    def __init__(self) -> None:
        self.memory = cd.FastSynapses(np.arange(CELLS), np.arange(CELLS, CELLS + 13), rule="delta")
        self.memory.reset(1)
        self.writes = 0

    def see(self, cell: tuple[int, int], passages: list[str]) -> None:
        value = np.zeros((1, 13))
        for d, kind in enumerate(passages):
            value[0, 3 * d + PASSAGE.index(kind)] = 1.0
        value[0, -1] = 1.0
        self.memory.observe(np.eye(CELLS)[[cell_index(*cell)]], value)
        self.writes += 1

    def recall(self, cell: tuple[int, int]) -> tuple[list[str] | None, float]:
        read = self.memory.recall(np.eye(CELLS)[[cell_index(*cell)]])[0]
        known = float(read[-1])
        if known < 0.5:
            return None, known
        return [PASSAGE[int(np.argmax(read[3 * d : 3 * d + 3]))] for d in range(4)], known

    def erase(self) -> None:
        self.memory.reset(1)


class CueStore:
    """The last word seen, held until the next word is seen (across episode boundaries)."""

    def __init__(self) -> None:
        self.value = np.zeros(WORDS + 1)

    def see(self, word: int) -> None:
        if word > 0:
            self.value = np.eye(WORDS + 1)[word]

    def reset(self) -> None:
        self.value = np.zeros(WORDS + 1)


class WordStore:
    """word -> referent by accumulated co-occurrence with visible objects across scenes."""

    def __init__(self, rate: float = 0.25) -> None:
        self.memory = cd.SynapticMemory(np.arange(WORDS + 1), np.arange(WORDS + 1, WORDS + 1 + OBJECTS), decay=1.0, rate=rate)
        self.memory.reset(1)
        self.writes = 0

    def scene(self, word: int, visible: list[int]) -> None:
        if word <= 0 or not visible:
            return
        value = np.zeros((1, OBJECTS))
        value[0, visible] = 1.0
        self.memory.observe(np.eye(WORDS + 1)[[word]], value)
        self.writes += 1

    def referent(self, word: int) -> tuple[int | None, float]:
        read = self.memory.recall(np.eye(WORDS + 1)[[word]])[0]
        order = np.argsort(-read)
        if read[order[0]] <= 0 or read[order[0]] - read[order[1]] < 0.05:
            return None, float(read[order[0]])
        return int(order[0]), float(read[order[0]])

    def drive(self, word: int) -> np.ndarray:
        if word <= 0:
            return np.zeros(OBJECTS)
        return self.memory.recall(np.eye(WORDS + 1)[[word]])[0]

    def erase(self) -> None:
        self.memory.clear()


RECALL_WIDTH = (CELLS + 1) + (WORDS + 1) + OBJECTS  # place read, cue, word read


# -- imagined consequences


def observe_from(obs: dict[str, np.ndarray]) -> dict[str, Any]:
    return {"cell": (int(obs["x"].argmax()), int(obs["y"].argmax())), "passages": [PASSAGE[int(obs[f"passage_{d}"].argmax())] for d in ("north", "east", "south", "west")]}


IMAGINED_OBSERVED = {name: np.ones(width, bool) for name, width in FIELDS.items()}
for _d in ("north", "east", "south", "west"):
    IMAGINED_OBSERVED[f"{_d}_object"] = np.zeros(OBJECTS + 1, bool)  # an imagined cell's neighbours are never in view


def compose(observation: dict[str, np.ndarray], prediction: dict[str, np.ndarray], map_store: MapStore | None = None) -> dict[str, np.ndarray]:
    """The imagined next observation: predicted consequences replace the visible fields, the
    position composes from the predicted displacement, a known cell's passages come from the
    witnessed map (``map_store``), neighbours stay unknown, the word stays, the phase moves on."""
    out = dict(observation)
    for name in CONSEQUENCES:
        p = np.asarray(prediction[f"next_{name}"])
        out[name] = np.eye(len(p))[int(p.argmax())]
    x, y = int(np.asarray(observation["x"]).argmax()), int(np.asarray(observation["y"]).argmax())
    x = min(max(x + int(out["dx"].argmax()) - 1, 0), SIZE - 1)
    y = min(max(y + int(out["dy"].argmax()) - 1, 0), SIZE - 1)
    out["x"], out["y"] = np.eye(SIZE)[x], np.eye(SIZE)[y]
    if map_store is not None:
        passages, _ = map_store.recall((x, y))
        if passages is not None:
            for d, kind in zip(("north", "east", "south", "west"), passages, strict=True):
                out[f"passage_{d}"] = np.eye(3)[PASSAGE.index(kind)]
    for d in ("north", "east", "south", "west"):
        out[f"{d}_object"] = np.eye(OBJECTS + 1)[0]
    out["phase"] = np.array([0.0, 1.0])
    return out


@dataclass
class WorldPlanner:
    """Best-first search over the learned model toward a target cell, or one-step reward greed."""

    max_nodes: int = 128
    depth: int = 8
    seed: int = 0
    fallbacks: int = 0
    pruned: int = 0

    def __post_init__(self) -> None:
        from agent.brain import Mulberry32

        self.rng = Mulberry32((self.seed * 104729 + 7) & 0xFFFFFFFF)

    def search(self, model, observation: dict[str, np.ndarray], goal: np.ndarray, target: tuple[int, int], legal: np.ndarray | None = None, map_store: MapStore | None = None, observed: dict[str, np.ndarray] | None = None) -> tuple[int, dict[str, np.ndarray], int]:
        """Returns (first action, its prediction, nodes expanded, path, cells). Ties and dead ends
        fall back to the action whose imagined cell is nearest the target. ``legal`` restricts the
        moves; ``map_store`` supplies the witnessed passages of known cells after a move;
        ``observed`` is the real moment's flag map (imagined states carry ``IMAGINED_OBSERVED``)."""
        allowed = set(range(len(ACTIONS))) if legal is None else {int(a) for a in np.flatnonzero(legal)}
        start = observe_from(observation)["cell"]
        root_flags = IMAGINED_OBSERVED if observed is None else observed
        if start == target:
            action = ACTIONS.index("inspect") if ACTIONS.index("inspect") in allowed else min(allowed)
            pred = model.predict_batch([observation], [action], observed=[root_flags], goal=goal)[0]
            return action, pred, 1, [action], [start]
        counter = 0
        frontier = [(manhattan(start, target), 0, counter, observation, None, None, 0)]  # (f, g, id, obs, path, first_pred, depth)
        best = (manhattan(start, target), None, None)
        expanded = 0
        closed: set[tuple[int, int]] = set()  # cells already expanded (A* over cells: unit costs, consistent heuristic)
        best_g = {start: 0}
        cache: dict[tuple, dict[str, np.ndarray]] = {}  # imagined consequences of one search, keyed by the visible state and action

        def imagine(obs: dict[str, np.ndarray], moves: list[int], depth: int) -> list[dict[str, np.ndarray]]:
            flags = root_flags if depth == 0 else IMAGINED_OBSERVED
            key = tuple(int(np.asarray(obs[name]).argmax()) for name in FIELDS) + (depth == 0,)
            missing = [a for a in moves if (key, a) not in cache]
            if missing:
                for a, pred in zip(missing, model.predict_batch([obs] * len(missing), missing, observed=[flags] * len(missing), goal=goal), strict=True):
                    cache[(key, a)] = pred
            return [cache[(key, a)] for a in moves]

        while frontier and expanded < self.max_nodes:
            _, g, _, obs, first, first_pred, depth = heapq.heappop(frontier)
            here_cell = observe_from(obs)["cell"]
            if here_cell in closed or depth >= self.depth:
                continue
            closed.add(here_cell)
            moves = [a for a in [ACTIONS.index(d) for d in ("north", "east", "south", "west")] + [ACTIONS.index("interact")] if a in allowed]
            predictions = imagine(obs, moves, depth)
            expanded += 1
            for action, pred in zip(moves, predictions, strict=True):
                # a move through what the state shows as a wall or a closed door is impossible:
                # the candidate is discarded, never repaired into the right next cell
                name = ACTIONS[action]
                if name in DIRECTIONS and PASSAGE[int(np.asarray(obs[f"passage_{name}"]).argmax())] != "open":
                    self.pruned += 1
                    continue
                nxt = compose(obs, pred, map_store if name in DIRECTIONS else None)
                cell = observe_from(nxt)["cell"]
                if name in DIRECTIONS and cell == observe_from(obs)["cell"]:
                    continue  # the model imagines standing still: no progress along this branch
                path = (*first, action) if first is not None else (action,)
                child_pred = pred if first_pred is None else first_pred
                h = manhattan(cell, target)
                if cell == target:
                    return path[0], child_pred, expanded, list(path), self._cells_of(observation, path, cache, model, goal, map_store)
                if h < best[0]:
                    best = (h, path, child_pred)
                if cell in closed or g + 1 >= best_g.get(cell, 10**9):
                    continue
                best_g[cell] = g + 1
                counter += 1
                heapq.heappush(frontier, (g + 1 + h, g + 1, counter, nxt, path, child_pred, depth + 1))
        if best[1] is None:  # no imagined move improves on the start: the model's least bad move
            moves = [a for a in [ACTIONS.index(d) for d in ("north", "east", "south", "west")] if a in allowed] or sorted(allowed)
            predictions = model.predict_batch([observation] * len(moves), moves, observed=[root_flags] * len(moves), goal=goal)
            scored = []
            for a, pred in zip(moves, predictions, strict=True):
                cell = observe_from(compose(observation, pred, map_store))["cell"]
                scored.append((manhattan(cell, target), a, pred))
            closest = min(sc[0] for sc in scored)
            ties = [sc for sc in scored if sc[0] == closest]
            _, action, pred = ties[int(self.rng.random() * len(ties))]
            self.fallbacks += 1
            return action, pred, expanded, [action], []
        return best[1][0], best[2], expanded, list(best[1]), self._cells_of(observation, best[1], cache, model, goal, map_store)

    def _cells_of(self, observation, path, cache, model, goal, map_store=None) -> list[tuple[int, int]]:
        """The imagined cell after each action of ``path`` (from the search's own cache)."""
        cells = []
        obs = observation
        for a in path:
            key = tuple(int(np.asarray(obs[name]).argmax()) for name in FIELDS) + (obs is observation,)
            pred = cache.get((key, a))
            if pred is None:
                pred = model.predict_batch([obs], [a], observed=[IMAGINED_OBSERVED], goal=goal)[0]
            obs = compose(obs, pred, map_store if ACTIONS[a] in DIRECTIONS else None)
            cells.append(observe_from(obs)["cell"])
        return cells

    def greedy_reward(self, model, observation: dict[str, np.ndarray], goal: np.ndarray, legal: np.ndarray | None = None, observed: dict[str, np.ndarray] | None = None) -> tuple[int, dict[str, np.ndarray], int]:
        actions = list(range(len(ACTIONS))) if legal is None else [int(a) for a in np.flatnonzero(legal)]
        flags = IMAGINED_OBSERVED if observed is None else observed
        predictions = model.predict_batch([observation] * len(actions), actions, observed=[flags] * len(actions), goal=goal)
        rewards = [float(np.asarray(p["reward"])[0]) for p in predictions]
        best = int(np.argmax(rewards))
        return actions[best], predictions[best], 1


class Brain:
    """The candidate: agent + stores + planner behind the one ``step`` contract."""

    def __init__(self, config: dict[str, Any], seed: int, *, learning: bool = True) -> None:
        self.config = config
        self.agent = Agent(agent_config(config, learning=learning), seed=seed, planner=self._plan)
        assert len(self.agent.ports.recall) == RECALL_WIDTH, (len(self.agent.ports.recall), RECALL_WIDTH)
        self.places, self.cue, self.words, self.map = PlaceStore(), CueStore(), WordStore(), MapStore()
        self.planner = WorldPlanner(**config.get("planner", {}), seed=seed)
        self.agent.recall_drive = self._recall
        self.visited: set[tuple[int, int]] = set()
        self.inspected: set[tuple[int, int]] = set()
        self.last_moment: Moment | None = None
        self.last_decision = None
        self.nodes = 0
        self.plan: list[int] = []  # the remaining imagined path of the last search
        self.plan_cells: list[tuple[int, int]] = []  # the cell each remaining action should reach
        self.plan_key: tuple | None = None  # (goal, target) the plan was made for
        self.searches = 0
        self.reused = 0
        self.delegated = 0  # cue decisions handed to the actor
        self.erase_places_at_request = False  # the erasure control's flag, read by the runner

    # -- what the moment reveals goes into the stores (supplied bookkeeping of witnessed facts)

    def witness(self, m: Moment) -> None:
        o = m.observation
        cell = (int(o["x"].argmax()), int(o["y"].argmax()))
        if m.tick == 0 and m.feedback_for is None:
            self.visited.clear()
            self.inspected.clear()
        self.visited.add(cell)
        if m.observed["x"].all() and m.observed["passage_north"].all():
            self.map.see(cell, observe_from(o)["passages"])
        word = int(o["word"].argmax())
        self.cue.see(word)
        here = int(o["here_object"].argmax()) - 1
        visible = []
        if here >= 0:
            self.places.see(here, cell)
            visible.append(here)
        else:
            for k in range(OBJECTS):
                where, known = self.places.recall(k)
                if known >= 0.5 and where == cell:
                    self.places.forget(k)  # the remembered place is empty: the record is stale
        if int(o["outcome"].argmax()) == OUTCOMES.index("inspected"):
            self.inspected.add(cell)
            for d, (dx, dy) in DIRECTIONS.items():
                if m.observed[f"{d}_object"].all():
                    k = int(o[f"{d}_object"].argmax()) - 1
                    if k >= 0:
                        self.places.see(k, (cell[0] + dx, cell[1] + dy))
                        visible.append(k)
        if int(np.asarray(m.goal).argmax()) == GOAL_EXPLORE:
            self.words.scene(word, visible)  # a word heard as a request is not a naming scene

    def _requested_object(self, m: Moment) -> int | None:
        goal = int(np.asarray(m.goal).argmax())
        if GOAL_OBJECT <= goal < GOAL_OBJECT + OBJECTS:
            return goal - GOAL_OBJECT
        if goal == GOAL_WORD:
            k, _ = self.words.referent(int(m.observation["word"].argmax()))
            return k
        return None

    def _recall(self, m: Moment, row: int) -> np.ndarray:
        k = self._requested_object(m)
        word = int(m.observation["word"].argmax())
        return np.concatenate([self.places.drive(k), self.cue.value, self.words.drive(word)]) * self.config["graph"].get("recall_gain", 2.0)

    # -- decisions

    def _exploration_target(self, cell: tuple[int, int]) -> tuple[int, int]:
        candidates = [(x, y) for y in range(SIZE) for x in range(SIZE) if (x, y) not in self.inspected]
        if not candidates:
            self.inspected.clear()
            candidates = [(x, y) for y in range(SIZE) for x in range(SIZE) if (x, y) != cell]
        return min(candidates, key=lambda c: (manhattan(cell, c), c))

    def _plan(self, agent: Agent, row: int, m: Moment, drive: np.ndarray, free) -> tuple[int | None, dict[str, np.ndarray], dict[str, int]]:
        observation = {k: np.asarray(v) for k, v in m.observation.items()}
        goal = np.asarray(m.goal)
        which = int(goal.argmax())
        cell = observe_from(observation)["cell"]
        if which == GOAL_CUE:
            self.plan = []
            action, pred, nodes = self.planner.greedy_reward(agent, observation, goal, m.action_mask, m.observed)
        else:
            target = None
            k = self._requested_object(m)
            if which != GOAL_EXPLORE and k is not None:
                target, _ = self.places.recall(k)
            if target is None:
                target = self._exploration_target(cell)
            if cell == target and which != GOAL_EXPLORE:
                target = self._exploration_target(cell)
            key = (which, target)
            # the plan of the previous decision is reused while the world follows the imagined path
            if self.plan and self.plan_key == key and self.plan_cells and cell == self.plan_cells[0]:
                self.plan.pop(0)
                self.plan_cells.pop(0)
            else:
                self.plan, self.plan_cells = [], []
            if self.plan and self.plan_key == key and m.action_mask[self.plan[0]]:
                action = self.plan[0]
                pred = agent.predict_batch([observation], [action], observed=[m.observed], goal=goal)[0]
                nodes = 0
                self.reused += 1
            else:
                action, pred, nodes, path, cells = self.planner.search(agent, observation, goal, target, m.action_mask, self.map, m.observed)
                self.searches += 1
                self.plan, self.plan_cells, self.plan_key = list(path), list(cells), key
        self.nodes += nodes
        return int(action), pred, {"expansions": nodes}

    def step(self, m: Moment):
        self.witness(m)
        d = self.agent.step(m)
        self.last_moment, self.last_decision = m, d
        return d

    def set_epsilon(self, epsilon: float) -> None:
        self.agent.config = replace(self.agent.config, actor=replace(self.agent.config.actor, epsilon=epsilon))

    def frozen(self) -> Brain:
        """An isolated read-only copy with copies of the stores."""
        import copy

        out = copy.copy(self)
        out.agent = self.agent.frozen()
        out.places, out.cue, out.words, out.map = copy.deepcopy(self.places), copy.deepcopy(self.cue), copy.deepcopy(self.words), copy.deepcopy(self.map)
        out.visited, out.inspected = set(self.visited), set(self.inspected)
        out.plan, out.plan_cells, out.plan_key = [], [], None
        out.agent.planner = out._plan
        out.agent.recall_drive = out._recall
        return out
