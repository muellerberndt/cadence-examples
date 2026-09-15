"""The curriculum of the remembered world: task setups, reward rules, held-out suites.

Every task is a way of running episodes of one ``World`` life. Setups use the world's
privileged accessors (placing objects, showing words, changing doors); the agent only
ever receives moments. Goals on the goal port (eight values):

    0 free exploration          1..4 reach object k (k = goal - 1)
    5 reach the object the current word names (grounding)
    6 the cue task: reach the room the cue rewards
    7 fetch: carry the requested object (reserved)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.life import Moment
from .env import ACTIONS, OBJECTS, SIZE, WORDS, World, one_hot

GOAL_EXPLORE, GOAL_OBJECT, GOAL_WORD, GOAL_CUE = 0, 1, 5, 6
REQUEST_DISTANCE = SIZE // 2 + 1  # a request starts at least this far (Manhattan) from its target


def request_deadline(world, target: tuple[int, int]) -> int:
    """Decisions allowed to fulfil a request: twice the world's own shortest path plus two.
    A remembered place is reachable in time; a blind search of sixteen cells mostly is not."""
    path = world.shortest_path(world.state["agent"], target, doors_open=True)
    return 2 * (len(path) if path is not None else 6) + 2
CUE_WORDS = (7, 8)  # the two cue words (1-based in the word field: 0 is "no word")
ROOMS = ((0, 1), (1, 0))  # the two reward rooms of the cue task: west and north of the junction (1, 1)


def goal_vector(index: int) -> np.ndarray:
    return one_hot(index, 8)


@dataclass
class Episode:
    """What one task episode asks, for evaluation: the private answer never reaches the agent."""

    task: str
    target_cell: tuple[int, int] | None = None
    object_id: int | None = None
    delay: int = 0
    cue: int | None = None
    rewarded_room: tuple[int, int] | None = None
    old_cell: tuple[int, int] | None = None


class Curriculum:
    """Runs task episodes on a world; the world's RNG is separate from the agent's."""

    def __init__(self, world: World, seed: int) -> None:
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.cue_rule = int(self.rng.integers(2))  # which cue word rewards which room in this life

    # -- helpers

    def _clear_objects(self) -> None:
        for k in range(OBJECTS):
            self.world.place_object(k, None)

    def _free_cell(self, exclude: set) -> tuple[int, int]:
        while True:
            c = self.world.random_cell()
            if c not in exclude:
                return c

    def _reach_reward(self, target: tuple[int, int], steps: int):
        def rule(world: World, action: str, moved: bool) -> tuple[float, bool]:
            if world.state["agent"] == target:
                return 1.0, True
            return -0.01, False
        return rule

    # -- tasks

    def explore(self):
        """Free exploration: objects scattered, no goal, no reward; time limit only."""
        self._clear_objects()
        self.world.set_mask(None)
        cells = set()
        for k in range(OBJECTS):
            c = self._free_cell(cells)
            cells.add(c)
            self.world.place_object(k, c)
        self.world.set_word(0)
        self.world.set_goal(goal_vector(GOAL_EXPLORE))
        self.world._reward_rule = None
        return Episode("explore")

    def remember(self, delay: int, *, move: str | None = None):
        """See one object once at the start, wander ``delay`` steps with no goal, then be asked for it.

        ``move``: "visible" moves the object while the agent watches (it sees it again at the
        new place); "invisible" moves it while unseen; the request then tests correction."""
        self._clear_objects()
        k = int(self.rng.integers(OBJECTS))
        start = self.world.random_cell()
        self.world.place_object(k, start)  # the agent starts on the object and sees it here
        others = {start}
        for j in range(OBJECTS):
            if j != k:
                c = self._free_cell(others)
                others.add(c)
                self.world.place_object(j, c)
        self.world.set_word(0)
        self.world.set_goal(goal_vector(GOAL_EXPLORE))
        self.world._reward_rule = None
        self.world.set_mask(np.array([1, 1, 1, 1, 1, 0], bool))  # no interact while wandering: the object stays put
        episode = Episode("remember", target_cell=start, object_id=k, delay=delay, old_cell=start)
        self._pending_move = move
        return episode

    def _far_start(self, target: tuple[int, int]) -> tuple[int, int]:
        """A free cell at least ``REQUEST_DISTANCE`` from the target: the request needs a path."""
        objects = {c for c in self.world.state["objects"].values() if c is not None}
        far = [(x, y) for y in range(SIZE) for x in range(SIZE) if (x, y) not in objects and manhattan((x, y), target) >= REQUEST_DISTANCE]
        if not far:
            far = [(x, y) for y in range(SIZE) for x in range(SIZE) if (x, y) != target]
        return far[int(self.rng.integers(len(far)))]

    def request(self, episode: Episode) -> Moment:
        """After the delay the request starts a new episode of the same life: the agent finds
        itself in a distant cell and the goal names the object; reaching its cell within the
        deadline ends the episode. The stores persist across episodes, the context trace does
        not, so only a remembered place is reachable in time."""
        assert episode.object_id is not None
        target = self.world.state["objects"][episode.object_id]
        if target is None:  # carried after all: put it back where it was witnessed
            target = episode.old_cell
            self.world.place_object(episode.object_id, target)
        episode.target_cell = target
        self.world.set_mask(None)
        self.world.set_goal(goal_vector(GOAL_OBJECT + episode.object_id))
        self.world._reward_rule = self._reach_reward(target, 0)
        m = self.world.reset(agent=self._far_start(target))
        self.world.set_deadline(request_deadline(self.world, target))
        return m

    def move_object(self, episode: Episode, *, visible: bool) -> None:
        """Move the remembered object to a new cell. Visibly means onto the agent's own cell,
        which must be empty, so the next moment shows it; invisibly means a free cell the agent
        neither stands on nor saw the object in."""
        assert episode.object_id is not None
        agent = self.world.state["agent"]
        occupied = {c for c in self.world.state["objects"].values() if c is not None}
        if visible:
            if self.world.object_at(agent) is not None:
                raise ValueError("a visible move needs the agent on an empty cell")
            new = agent
        else:
            new = self._free_cell(occupied | {agent, episode.old_cell})
        self.world.place_object(episode.object_id, new)
        episode.target_cell = new

    def cue(self, delay: int):
        """A cue word at the start decides which room pays; the agent is far from both rooms."""
        self._clear_objects()
        cue = int(self.rng.integers(2))
        self.world.set_word(CUE_WORDS[cue])
        rewarded = ROOMS[cue ^ self.cue_rule]
        self.world.set_goal(goal_vector(GOAL_CUE))
        self.world.set_mask(np.array([1, 1, 1, 1, 1, 0], bool))
        self.world._reward_rule = None  # nothing pays during the wander
        return Episode("cue", delay=delay, cue=cue, rewarded_room=rewarded)

    def arm_cue(self, episode: Episode, rewarded: tuple[int, int] | None = None) -> Moment:
        """The plan's two-choice diagnostic, a new episode of the same life at the junction: the
        legal actions are the two room entries; entering a room ends the episode and pays for
        the cued one. The cue store carries the word across the episode boundary."""
        paid = episode.rewarded_room if rewarded is None else rewarded
        legal = np.zeros(len(ACTIONS), bool)
        legal[[ACTIONS.index("north"), ACTIONS.index("west")]] = True
        self.world.set_mask(legal)
        self.world.set_word(0)
        self.world.set_goal(goal_vector(GOAL_CUE))

        def rule(world: World, action: str, moved: bool) -> tuple[float, bool]:
            here = world.state["agent"]
            if here in ROOMS:
                return (1.0 if here == paid else 0.0), True
            return 0.0, False

        self.world._reward_rule = rule
        return self.world.reset(agent=self.world.config.junction)

    def hide_cue(self) -> None:
        self.world.set_word(0)

    def word(self, *, teach: bool = True):
        """Word grounding. Teaching: while the agent explores, the word is said in the moments its
        referent is in view (joint attention), with other objects sometimes in view too, so
        only accumulated co-occurrence isolates the referent. Requesting (``name_request``):
        the word alone names the object to reach."""
        self._clear_objects()
        self.world.set_mask(np.array([1, 1, 1, 1, 1, 0], bool))  # nothing is carried off while the word is taught
        k = int(self.rng.integers(OBJECTS))
        word = int(self.world.state["words"][k]) + 1  # the life's label of object k, 1-based
        cells = set()
        for j in range(OBJECTS):
            c = self._free_cell(cells)
            cells.add(c)
            self.world.place_object(j, c)
        self.world.set_goal(goal_vector(GOAL_EXPLORE))
        self.world._reward_rule = None
        if teach:
            self.world.set_word_rule(lambda w, k=k, word=word: word if k in w.visible_objects() else 0)
        else:
            self.world.set_word(0)
        target = self.world.state["objects"][k]
        # the episode starts on the referent: the first scene is guaranteed, its place witnessed
        return Episode("word", target_cell=target, object_id=k, cue=word, old_cell=target)

    def name_request(self, episode: Episode) -> Moment:
        """The word alone, as a request in a new episode from a distant cell: reach its referent
        within the deadline."""
        assert episode.object_id is not None and episode.cue is not None
        self.world.set_word_rule(None)
        self.world.set_word(episode.cue)
        self.world.set_mask(None)
        self.world.set_goal(goal_vector(GOAL_WORD))
        target = self.world.state["objects"][episode.object_id]
        if target is None:
            target = episode.target_cell
            self.world.place_object(episode.object_id, target)
        episode.target_cell = target
        self.world._reward_rule = self._reach_reward(target, 0)
        m = self.world.reset(agent=self._far_start(target))
        self.world.set_deadline(request_deadline(self.world, target))
        return m

    def lock_doors(self) -> None:
        """A changed consequence: interact no longer opens doors, mid-life."""
        self.world.set_door_rule("locked")


def scripted_wander(world: World, rng: np.random.Generator, steps: int) -> list[int]:
    """Random legal-looking actions for the delay phase (moves only, no interact)."""
    return [int(rng.integers(4)) for _ in range(steps)]


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


__all__ = ["ACTIONS", "GOAL_CUE", "GOAL_EXPLORE", "GOAL_OBJECT", "GOAL_WORD", "ROOMS", "WORDS", "Curriculum", "Episode", "goal_vector", "manhattan", "scripted_wander"]
