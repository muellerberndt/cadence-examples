"""A 4x4 world that must be remembered: rooms, passages, doors, movable objects, words.

The simulator keeps the map, the object positions and the door rules private. A moment
exposes only the current cell and what is visible from it: the cell's own coordinates,
its four passages, the object in the cell, the objects in the four adjacent cells after
an ``inspect`` (otherwise their flags are off), what the agent carries, the outcome of
the last action, a cue word and an episode-phase marker. Every field is categorical; the
eighteen fields and their classes are listed in ``FIELDS``. Connectivity, door placement,
object identities, colours and word labels are permuted per life from the life seed.

Actions: north, east, south, west, inspect, interact. Interact picks up the object in
the cell, drops a carried object into an empty cell, or toggles the doors of the cell.
Tasks (the goal port, eight values) are declared by the curriculum in ``tasks.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agent.life import Moment

SIZE = 6
ACTIONS = ("north", "east", "south", "west", "inspect", "interact")
DIRECTIONS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
PASSAGE = ("open", "wall", "door_closed")
OUTCOMES = ("none", "moved", "blocked", "picked", "dropped", "toggled", "inspected")
OBJECTS = 4
COLORS = 4
WORDS = 8
FIELDS = {
    "x": SIZE, "y": SIZE,
    "dx": 3, "dy": 3,  # the displacement of the last action: -1, 0, +1 (as S01 reports its displacement)
    "passage_north": 3, "passage_east": 3, "passage_south": 3, "passage_west": 3,
    "here_object": OBJECTS + 1, "here_color": COLORS + 1,
    "north_object": OBJECTS + 1, "east_object": OBJECTS + 1, "south_object": OBJECTS + 1, "west_object": OBJECTS + 1,
    "carrying": OBJECTS + 1, "outcome": len(OUTCOMES), "word": WORDS + 1, "phase": 2,
}
GOALS = 8


def one_hot(index: int, width: int) -> np.ndarray:
    out = np.zeros(width)
    out[index] = 1.0
    return out


@dataclass
class WorldConfig:
    doors: int = 3  # internal edges that carry a door (closed at first)
    walls: int = 4  # internal edges that are walls
    horizon: int = 64
    inspect_visible: bool = True  # adjacent objects are visible only after an inspect
    junction: tuple[int, int] = (1, 1)  # the cue task's decision cell: its north and west passages stay open


@dataclass
class World:
    """One life's private map and one episode at a time; ``act`` executes a committed decision."""

    config: WorldConfig = field(default_factory=WorldConfig)
    seed: int = 0
    life_id: str = "world"
    _rng: np.random.Generator = field(init=False, repr=False)
    _edges: dict[tuple[tuple[int, int], tuple[int, int]], str] = field(default_factory=dict, init=False)
    _colors: np.ndarray = field(default_factory=lambda: np.arange(COLORS), init=False)
    _word_of: np.ndarray = field(default_factory=lambda: np.arange(WORDS), init=False)
    _objects: dict[int, tuple[int, int] | None] = field(default_factory=dict, init=False)
    _agent: tuple[int, int] = field(default=(0, 0), init=False)
    _carrying: int | None = field(default=None, init=False)
    _outcome: int = field(default=0, init=False)
    _move: tuple[int, int] = field(default=(0, 0), init=False)  # the last action's displacement
    _inspected: bool = field(default=False, init=False)
    _word: int = field(default=0, init=False)
    _word_rule: object = None  # a callable(world) -> word shown this moment (joint attention), or None for the static word
    _goal: np.ndarray = field(default_factory=lambda: np.zeros(GOALS), init=False)
    _episode: int = field(default=-1, init=False)
    _event: int = field(default=0, init=False)
    _tick: int = field(default=0, init=False)
    _pending: int | None = field(default=None, init=False)
    _decision_action: int | None = field(default=None, init=False)
    _door_rule: str = field(default="toggle", init=False)  # or "locked": interact no longer opens doors
    _mask: np.ndarray = field(default_factory=lambda: np.ones(len(ACTIONS), bool), init=False)  # the task's legal actions
    _deadline: int | None = field(default=None, init=False)  # a task's own time limit (tick), truncating earlier than the horizon
    _doors: set = field(default_factory=set, init=False)  # the edges built as doors, whatever their state
    _reward_rule: object = None  # a callable(world, action, moved) -> (reward, terminated), set by a task
    _episode_hook: object = None  # a callable(world) at every reset, set by a task

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._build()

    # -- the private map

    def _build(self) -> None:
        rng = self._rng
        cells = [(x, y) for y in range(SIZE) for x in range(SIZE)]
        edges = []
        for x, y in cells:
            if x + 1 < SIZE:
                edges.append(((x, y), (x + 1, y)))
            if y + 1 < SIZE:
                edges.append(((x, y), (x, y + 1)))
        # a random spanning tree stays open so every cell is reachable through doors at most
        order = rng.permutation(len(edges))
        parent = {c: c for c in cells}

        def root(c):
            while parent[c] != c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            return c

        tree = set()
        for k in order:
            a, b = edges[k]
            if root(a) != root(b):
                parent[root(a)] = root(b)
                tree.add(k)
        rest = [k for k in range(len(edges)) if k not in tree]
        rng.shuffle(rest)
        kinds = {k: "open" for k in range(len(edges))}
        jx, jy = self.config.junction
        protected = {edges.index(((jx - 1, jy), (jx, jy))), edges.index(((jx, jy - 1), (jx, jy)))}
        rest = [k for k in rest if k not in protected]
        for k in rest[: self.config.walls]:
            kinds[k] = "wall"
        candidates = [k for k in range(len(edges)) if kinds[k] == "open" and k not in protected]
        rng.shuffle(candidates)
        for k in candidates[: self.config.doors]:
            kinds[k] = "door_closed"
        self._edges = {edges[k]: kinds[k] for k in range(len(edges))}
        self._doors = {edges[k] for k in candidates[: self.config.doors]}
        self._colors = rng.permutation(COLORS)
        self._word_of = rng.permutation(WORDS)

    def _passage(self, cell: tuple[int, int], direction: str) -> str:
        dx, dy = DIRECTIONS[direction]
        other = (cell[0] + dx, cell[1] + dy)
        if not (0 <= other[0] < SIZE and 0 <= other[1] < SIZE):
            return "wall"
        key = (cell, other) if (cell, other) in self._edges else (other, cell)
        return self._edges[key]

    def _set_passage(self, cell: tuple[int, int], direction: str, kind: str) -> None:
        dx, dy = DIRECTIONS[direction]
        other = (cell[0] + dx, cell[1] + dy)
        key = (cell, other) if (cell, other) in self._edges else (other, cell)
        self._edges[key] = kind

    def object_at(self, cell: tuple[int, int]) -> int | None:
        for k, where in self._objects.items():
            if where == cell:
                return k
        return None

    # -- privileged access for tasks, baselines and evaluation only

    @property
    def state(self) -> dict:
        return {"agent": self._agent, "objects": dict(self._objects), "carrying": self._carrying, "edges": dict(self._edges), "colors": self._colors.copy(), "words": self._word_of.copy(), "door_rule": self._door_rule}

    def place_object(self, k: int, cell: tuple[int, int] | None) -> None:
        self._objects[k] = cell

    def set_word(self, word: int) -> None:
        self._word = int(word)
        self._word_rule = None

    def set_word_rule(self, rule) -> None:
        """Joint attention: the word shown each moment is computed from what is visible."""
        self._word_rule = rule

    def visible_objects(self) -> list[int]:
        """Objects in the cell, plus those in adjacent cells after an inspect."""
        x, y = self._agent
        out = []
        here = self.object_at(self._agent)
        if here is not None:
            out.append(here)
        if self._inspected:
            for direction, (dx, dy) in DIRECTIONS.items():
                cell = (x + dx, y + dy)
                if self._passage(self._agent, direction) != "wall" and 0 <= cell[0] < SIZE and 0 <= cell[1] < SIZE:
                    k = self.object_at(cell)
                    if k is not None:
                        out.append(k)
        return out

    def set_goal(self, goal: np.ndarray) -> None:
        self._goal = np.asarray(goal, float).copy()

    def set_door_rule(self, rule: str) -> None:
        self._door_rule = rule

    def set_deadline(self, steps: int | None) -> None:
        """A task's time limit: ``steps`` more decisions from now, or None for the horizon alone."""
        self._deadline = None if steps is None else self._tick + int(steps)

    def set_mask(self, legal: np.ndarray | None) -> None:
        """The legal actions a task allows (a rule of the task, visible to the agent as the mask)."""
        self._mask = np.ones(len(ACTIONS), bool) if legal is None else np.asarray(legal, bool).copy()

    def random_cell(self) -> tuple[int, int]:
        return (int(self._rng.integers(SIZE)), int(self._rng.integers(SIZE)))

    def shortest_path(self, start: tuple[int, int], goal: tuple[int, int], *, doors_open: bool = False) -> list[str] | None:
        """Privileged planner for the upper reference and for evaluation."""
        from collections import deque

        queue = deque([start])
        came: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                path = []
                while came[cell] is not None:
                    cell, action = came[cell]
                    path.append(action)
                return list(reversed(path))
            for direction, (dx, dy) in DIRECTIONS.items():
                kind = self._passage(cell, direction)
                if kind == "wall" or (kind == "door_closed" and not doors_open):
                    continue
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt not in came:
                    came[nxt] = (cell, direction)
                    queue.append(nxt)
        return None

    # -- episodes

    def reset(self, *, agent: tuple[int, int] | None = None) -> Moment:
        self._episode += 1
        self._tick = 0
        self._pending = None
        self._agent = self.random_cell() if agent is None else agent
        self._carrying = None
        self._outcome = 0
        self._move = (0, 0)
        self._inspected = False
        self._deadline = None
        if self._episode_hook is not None:
            self._episode_hook(self)
        return self._moment(feedback=False, reward=0.0, terminated=False, truncated=False)

    def observation(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        x, y = self._agent
        here = self.object_at(self._agent)
        if self._word_rule is not None:
            self._word = int(self._word_rule(self))
        obs = {
            "x": one_hot(x, SIZE), "y": one_hot(y, SIZE),
            "dx": one_hot(self._move[0] + 1, 3), "dy": one_hot(self._move[1] + 1, 3),
            "passage_north": one_hot(PASSAGE.index(self._passage(self._agent, "north")), 3),
            "passage_east": one_hot(PASSAGE.index(self._passage(self._agent, "east")), 3),
            "passage_south": one_hot(PASSAGE.index(self._passage(self._agent, "south")), 3),
            "passage_west": one_hot(PASSAGE.index(self._passage(self._agent, "west")), 3),
            "here_object": one_hot(0 if here is None else here + 1, OBJECTS + 1),
            "here_color": one_hot(0 if here is None else int(self._colors[here]) + 1, COLORS + 1),
            "carrying": one_hot(0 if self._carrying is None else self._carrying + 1, OBJECTS + 1),
            "outcome": one_hot(self._outcome, len(OUTCOMES)),
            "word": one_hot(self._word, WORDS + 1),
            "phase": one_hot(0 if self._tick == 0 else 1, 2),
        }
        observed = {}
        for direction in ("north", "east", "south", "west"):
            dx, dy = DIRECTIONS[direction]
            cell = (x + dx, y + dy)
            visible = self._inspected and self._passage(self._agent, direction) != "wall" and 0 <= cell[0] < SIZE and 0 <= cell[1] < SIZE
            obj = self.object_at(cell) if visible else None
            obs[f"{direction}_object"] = one_hot(0 if obj is None else obj + 1, OBJECTS + 1)
            observed[f"{direction}_object"] = np.full(OBJECTS + 1, visible)
        return obs, observed

    def _moment(self, *, feedback: bool, reward: float, terminated: bool, truncated: bool) -> Moment:
        obs, observed = self.observation()
        m = Moment(
            life_id=self.life_id, episode_id=self._episode, event_id=self._event, tick=self._tick,
            observation=obs, observed=observed, action_mask=self._mask.copy(),
            feedback_for=self._pending if feedback else None, executed=self._decision_action if feedback else None,
            reward=reward, reward_known=feedback, terminated=terminated, truncated=truncated,
            final_observation=obs if truncated else None, goal=self._goal.copy(),
        )
        self._event += 1
        return m

    def act(self, decision_id: int, action: int) -> Moment:
        if self._pending is not None:
            raise RuntimeError("the previous decision has not been fed back")
        name = ACTIONS[int(action)]
        if not self._mask[int(action)]:
            raise ValueError(f"action {name} is not legal under the task's mask")
        self._pending, self._decision_action = decision_id, int(action)
        self._inspected = False
        self._move = (0, 0)
        moved = False
        if name in DIRECTIONS:
            kind = self._passage(self._agent, name)
            if kind == "open":
                dx, dy = DIRECTIONS[name]
                self._agent = (self._agent[0] + dx, self._agent[1] + dy)
                self._move = (dx, dy)
                moved = True
                self._outcome = OUTCOMES.index("moved")
            else:
                self._outcome = OUTCOMES.index("blocked")
        elif name == "inspect":
            self._inspected = True
            self._outcome = OUTCOMES.index("inspected")
        else:
            here = self.object_at(self._agent)
            if self._carrying is None and here is not None:
                self._objects[here] = None
                self._carrying = here
                self._outcome = OUTCOMES.index("picked")
            elif self._carrying is not None and here is None:
                self._objects[self._carrying] = self._agent
                self._carrying = None
                self._outcome = OUTCOMES.index("dropped")
            else:
                toggled = False
                for direction in DIRECTIONS:
                    kind = self._passage(self._agent, direction)
                    if kind == "door_closed" and self._door_rule == "toggle":
                        self._set_passage(self._agent, direction, "open")
                        toggled = True
                    elif kind == "open" and self._door_rule == "toggle" and self._was_door(direction):
                        self._set_passage(self._agent, direction, "door_closed")
                        toggled = True
                self._outcome = OUTCOMES.index("toggled" if toggled else "none")
        self._tick += 1
        reward, terminated = (0.0, False) if self._reward_rule is None else self._reward_rule(self, name, moved)
        truncated = not terminated and (self._tick >= self.config.horizon or (self._deadline is not None and self._tick >= self._deadline))
        m = self._moment(feedback=True, reward=reward, terminated=terminated, truncated=truncated)
        self._pending = None
        return m

    def _was_door(self, direction: str) -> bool:
        """An open passage that was built as a door."""
        dx, dy = DIRECTIONS[direction]
        other = (self._agent[0] + dx, self._agent[1] + dy)
        key = (self._agent, other) if (self._agent, other) in self._edges else (other, self._agent)
        return key in self._doors
