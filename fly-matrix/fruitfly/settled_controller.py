"""Residual-gated sensory ports -> connectome -> motor-only actuators.

This controller has no body, world, target, behavioural state machine, teacher or
reward input. Its caller supplies a frozen dictionary of declared sensory-port
levels for each event. These ports may be engineered projections (for example
HS/VS); this interface does not establish biological receptor fidelity. A small
potential/adaptation-equation residual is required before any motor authority is
released. It does not establish stability, uniqueness or competent behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np

from cadence import Brain, BrainState

from .motor import SETTLED_MOTOR_GROUPS, settled_motor


@dataclass(frozen=True)
class MotorFrame:
    actuators: dict
    readouts: dict[str, float]
    converged: bool
    residual: float
    steps: int
    reason: str


class SettledController:
    """Bounded equilibrium solves; an unresolved event returns zero actuation.

    ``sensory_populations`` is an explicit port declaration. Unknown populations
    and ports overlapping annotated motor neurons are rejected at construction.
    Population means are absolute; no resting baseline is subtracted. Only
    converged readouts are exposed to callers. Finite capped neural states may be
    retained as warm starts for a later event, but never supply motor commands.
    """

    def __init__(
        self, brain: Brain, *, sensory_populations: Sequence[str],
        budget: int = 512, chunk: int = 32, tolerance: float = 1e-5,
    ) -> None:
        for name, value, minimum in (("budget", budget, 0), ("chunk", chunk, 1)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        populations = brain.connectome.populations
        motor_members = {
            i for name, indices in populations.items()
            if name == "motor" or name.startswith(("mn:", "power", "steering", "tension", "neck_mn", "leg_mn")) or name == "mn9"
            for i in indices
        }
        self.inputs = {}
        for name in sensory_populations:
            if name not in populations or not populations[name]:
                raise ValueError(f"unknown or empty sensory population {name!r}")
            indices = list(populations[name])
            if motor_members.intersection(indices):
                raise ValueError(f"sensory population {name!r} overlaps motor neurons")
            self.inputs[name] = indices
        self.brain, self.budget, self.chunk, self.tolerance = brain, budget, chunk, float(tolerance)
        self.members = {name: list(populations[name]) for name in SETTLED_MOTOR_GROUPS if name in populations}
        self.state: BrainState | None = None

    def reset(self) -> None:
        self.state = None

    def _failure(self, reason: str, *, residual: float = float("inf"), steps: int = 0) -> MotorFrame:
        return MotorFrame(settled_motor({}), {}, False, residual, steps, reason)

    def step(self, sensory_levels: Mapping[str, float], *, mask: np.ndarray | None = None) -> MotorFrame:
        """Freeze one sensory event, solve its equations, then decode motor neurons.

        Duplicate sensory populations sharing a neuron combine by maximum level,
        matching the existing sensory drive convention. Malformed sensor input,
        numerical failure and an exhausted solve budget all return zero commands.
        ``mask`` is a laboratory lesion only, passed to both solve and residual.
        """
        drive = np.zeros(self.brain.connectome.n)
        for name, level in sensory_levels.items():
            if name not in self.inputs or isinstance(level, bool) or not isinstance(level, Real) or not isfinite(level) or not 0 <= level <= 1:
                self.state = None
                return self._failure("invalid_sensory_input")
            indices = self.inputs[name]
            drive[indices] = np.maximum(drive[indices], self.brain.neuron_model.stimulus_amplitude * level)
        try:
            result = self.brain.equilibrate(
                drive, budget=self.budget, chunk=self.chunk, tolerance=self.tolerance,
                state=self.state, mask=mask,
            )
        except (ValueError, FloatingPointError):
            self.state = None
            return self._failure("solver_error")
        self.state = result.state
        residual = float(np.max(result.residual))
        if not isfinite(residual) or not np.isfinite(result.state.activation).all():
            self.state = None
            return self._failure("nonfinite_state", residual=residual, steps=result.steps)
        if residual > self.tolerance:
            return self._failure("residual_above_tolerance", residual=residual, steps=result.steps)
        readouts = {name: result.state.mean(indices) for name, indices in self.members.items()}
        return MotorFrame(settled_motor(readouts), readouts, True, residual, result.steps, "equilibrium")
