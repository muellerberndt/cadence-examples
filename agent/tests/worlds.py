"""Deterministic probe environments of the contract: they expose only ``Moment`` fields.

``DelayedCue``: a two-choice cue at the first moment, blank moments with irrelevant
choices, and a reward ``delay`` decisions later that is 1 when the first choice matched
the cue. ``RingWorld``: four rooms in a ring, two actions (clockwise, counterclockwise),
the next room fully observed; its dynamics can be altered mid-life to test causal
prediction gain. Private state lives in ``_`` attributes; the agent never receives the
environment object, only moments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agent.life import Moment


def one_hot(index: int, width: int) -> np.ndarray:
    out = np.zeros(width)
    out[index] = 1.0
    return out


@dataclass
class DelayedCue:
    """Cue, blanks, reward. Observation ``cue``: [cue A, cue B, blank]; ``phase``: [first, later]."""

    delay: int = 8
    life_id: str = "delayed-cue"
    seed: int = 0
    shuffle: bool = False  # control: the rewarded association is re-drawn per episode
    _rng: np.random.Generator = field(init=False, repr=False)
    _episode: int = field(default=-1, init=False)
    _event: int = field(default=0, init=False)
    _tick: int = field(default=0, init=False)
    _cue: int = field(default=0, init=False)
    _rule: int = field(default=0, init=False)
    _first: int | None = field(default=None, init=False)
    _pending: int | None = field(default=None, init=False)
    _decision_action: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @property
    def action_count(self) -> int:
        return 2

    def reset(self) -> Moment:
        self._episode += 1
        self._tick = 0
        self._cue = int(self._rng.integers(2))
        self._rule = int(self._rng.integers(2)) if self.shuffle else 0
        self._first = None
        self._pending = None
        return self._moment(feedback=False, reward=0.0, terminated=False)

    def _observation(self) -> dict[str, np.ndarray]:
        cue = one_hot(self._cue, 3) if self._tick == 0 else one_hot(2, 3)
        phase = one_hot(0 if self._tick == 0 else 1, 2)
        return {"cue": cue, "phase": phase}

    def _moment(self, *, feedback: bool, reward: float, terminated: bool) -> Moment:
        m = Moment(
            life_id=self.life_id,
            episode_id=self._episode,
            event_id=self._event,
            tick=self._tick,
            observation=self._observation(),
            observed={},
            action_mask=np.ones(2, bool),
            feedback_for=self._pending if feedback else None,
            executed=self._decision_action if feedback else None,
            reward=reward,
            reward_known=feedback,
            terminated=terminated,
        )
        self._event += 1
        return m

    def act(self, decision_id: int, action: int) -> Moment:
        """Execute the committed decision; the next moment carries its feedback."""
        if self._pending is not None:
            raise RuntimeError("the previous decision has not been fed back")
        self._pending, self._decision_action = decision_id, int(action)
        if self._first is None:
            self._first = int(action)
        self._tick += 1
        terminal = self._tick > self.delay
        correct = (self._first ^ self._rule) == self._cue
        reward = float(correct) if terminal else 0.0
        m = self._moment(feedback=True, reward=reward, terminated=terminal)
        self._pending = None
        return m

    @property
    def correct_first(self) -> int:
        """Evaluation only: the first action the current episode rewards."""
        return self._cue ^ self._rule


@dataclass
class RingWorld:
    """Four rooms in a ring; action 0 moves clockwise, 1 counterclockwise. ``altered`` swaps them."""

    rooms: int = 4
    life_id: str = "ring"
    seed: int = 0
    altered: bool = False
    episode_length: int = 16
    _rng: np.random.Generator = field(init=False, repr=False)
    _episode: int = field(default=-1, init=False)
    _event: int = field(default=0, init=False)
    _tick: int = field(default=0, init=False)
    _room: int = field(default=0, init=False)
    _pending: int | None = field(default=None, init=False)
    _decision_action: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> Moment:
        self._episode += 1
        self._tick = 0
        self._room = int(self._rng.integers(self.rooms))
        self._pending = None
        return self._moment(feedback=False, reward=0.0, terminated=False)

    def _moment(self, *, feedback: bool, reward: float, terminated: bool) -> Moment:
        m = Moment(
            life_id=self.life_id,
            episode_id=self._episode,
            event_id=self._event,
            tick=self._tick,
            observation={"room": one_hot(self._room, self.rooms)},
            observed={},
            action_mask=np.ones(2, bool),
            feedback_for=self._pending if feedback else None,
            executed=self._decision_action if feedback else None,
            reward=reward,
            reward_known=feedback,
            terminated=terminated,
        )
        self._event += 1
        return m

    def next_room(self, room: int, action: int) -> int:
        direction = 1 if action == 0 else -1
        if self.altered:
            direction = -direction
        return (room + direction) % self.rooms

    def act(self, decision_id: int, action: int) -> Moment:
        if self._pending is not None:
            raise RuntimeError("the previous decision has not been fed back")
        self._pending, self._decision_action = decision_id, int(action)
        self._room = self.next_room(self._room, int(action))
        self._tick += 1
        m = self._moment(feedback=True, reward=0.0, terminated=self._tick >= self.episode_length)
        self._pending = None
        return m
