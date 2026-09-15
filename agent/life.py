"""Immutable event records of one learning life: moments in, decisions out.

A ``Moment`` is what the environment exposes at one real event: observed values with
their observed flags, feedback for the previously executed decision, reward with a
known flag, termination, legal actions and an optional witnessed demonstration. A
``Decision`` is what the agent commits before the environment executes it. Arrays are
copied and made read-only so that retained records cannot change under a later event.
Nothing here reads environment internals.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

STAGE = "S00"
SPLITS = ("development", "validation", "acceptance", "heldout")


def seed_for(stage: str, split: str, brain_seed: int, environment_id: int, stream_id: int) -> int:
    """One seed per (stage, split, brain seed, environment, stream): a hash, never shared."""
    key = f"{stage}|{split}|{int(brain_seed)}|{int(environment_id)}|{int(stream_id)}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def _frozen(value: Any, dtype: Any = float) -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True)
    if out.dtype.kind == "f" and not np.isfinite(out).all():
        raise ValueError("records must be finite")
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class Demonstration:
    """A witnessed action for a witnessed observation, with its source declared."""

    action: int
    observation: np.ndarray
    source: str
    visibility: str = "public"

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation", _frozen(self.observation))


@dataclass(frozen=True)
class Moment:
    life_id: str
    episode_id: int
    event_id: int
    tick: int
    observation: Mapping[str, np.ndarray]
    observed: Mapping[str, np.ndarray]
    action_mask: np.ndarray
    dt: float = 1.0
    feedback_for: int | None = None
    executed: int | None = None
    reward: float = 0.0
    reward_known: bool = False
    terminated: bool = False
    truncated: bool = False
    final_observation: Mapping[str, np.ndarray] | None = None
    goal: np.ndarray | None = None
    demonstration: Demonstration | None = None
    replay_of: int | None = None  # a replayed observed event carries the ID it replays

    def __post_init__(self) -> None:
        for name, value in (("episode_id", self.episode_id), ("event_id", self.event_id), ("tick", self.tick)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.life_id, str) or not self.life_id:
            raise ValueError("life_id must be a nonempty string")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(self.reward):
            raise ValueError("reward must be finite")
        if (self.feedback_for is None) != (self.executed is None):
            raise ValueError("feedback names both the decision and the executed action, or neither")
        if self.feedback_for is None and self.reward_known:
            raise ValueError("a known reward belongs to an executed decision")
        if self.terminated and self.truncated:
            raise ValueError("an event is terminated or truncated, not both")
        observation = {}
        observed = {}
        for name, value in dict(self.observation).items():
            arr = _frozen(value)
            flag = self.observed.get(name) if self.observed is not None else None
            if flag is None:
                flag = np.ones(arr.shape, bool)
            flag = _frozen(flag, bool)
            if flag.shape != arr.shape:
                raise ValueError(f"observed flags of {name!r} must match the observation's shape")
            observation[name] = arr
            observed[name] = flag
        if set(self.observed or {}) - set(observation):
            raise ValueError("observed flags name unknown observation fields")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "observed", observed)
        mask = _frozen(self.action_mask, bool)
        if mask.ndim != 1 or not mask.size:
            raise ValueError("action_mask must be a nonempty boolean vector")
        object.__setattr__(self, "action_mask", mask)
        if self.goal is not None:
            object.__setattr__(self, "goal", _frozen(self.goal))
        if self.final_observation is not None:
            object.__setattr__(
                self, "final_observation", {k: _frozen(v) for k, v in dict(self.final_observation).items()}
            )

    @property
    def has_feedback(self) -> bool:
        return self.feedback_for is not None

    @property
    def any_legal(self) -> bool:
        return bool(self.action_mask.any())


@dataclass(frozen=True)
class Decision:
    decision_id: int
    event_id: int
    stream: int
    parameter_version: int
    context_version: int
    action: int
    action_mask: np.ndarray
    probabilities: np.ndarray
    log_probability: float
    controller: str  # actor | planner | demonstration | exploration
    prediction: Mapping[str, np.ndarray] = field(default_factory=dict)
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    budget: Mapping[str, int] = field(default_factory=dict)
    intention: int | None = None

    def __post_init__(self) -> None:
        if self.controller not in ("actor", "planner", "demonstration", "exploration"):
            raise ValueError("controller must be actor, planner, demonstration or exploration")
        mask = _frozen(self.action_mask, bool)
        if not mask[self.action]:
            raise ValueError("a decision must choose a legal action")
        object.__setattr__(self, "action_mask", mask)
        object.__setattr__(self, "probabilities", _frozen(self.probabilities))
        object.__setattr__(self, "prediction", {k: _frozen(v) for k, v in dict(self.prediction).items()})
        object.__setattr__(self, "uncertainty", dict(self.uncertainty))
        object.__setattr__(self, "budget", dict(self.budget))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "event_id": self.event_id,
            "stream": self.stream,
            "parameter_version": self.parameter_version,
            "context_version": self.context_version,
            "action": int(self.action),
            "controller": self.controller,
            "log_probability": float(self.log_probability),
            "probabilities": [float(p) for p in self.probabilities],
            "prediction": {k: np.asarray(v).tolist() for k, v in self.prediction.items()},
            "uncertainty": {k: float(v) for k, v in self.uncertainty.items()},
            "budget": {k: int(v) for k, v in self.budget.items()},
            "intention": self.intention,
        }
