"""Planning through the record patch: the library's planning rule on ``cadence.RecordPatchNet``.

``RecordPatchNet`` on library ``main`` has no ``plan``. This reference applies the rule of
``TemporalPatchNet.plan`` to the record patch: the error is taken at the full prediction (slow
readout plus record read), the sensitivity runs through the slow path (the record read is a
port value outside every gradient, as in learning), declared control ports move by projected
gradient inside their bounds, and a proposal is accepted only when a target-free causal replay
lowers the cost and meets the Armijo condition. ``feedback`` pairs ``(output, input)`` feed a
moment's prediction into the next moment's input, so a path can be rolled out from the present
reading alone; with ``feedback_add`` the prediction is a change added to the moment's input. The
chain rule then runs through that feedback. With ``fixed_read`` the record read of the initial
path is held through every replay: the records correct the prediction where the actor stands, and
a proposal is judged by the slow model from there, never by records at readings nobody witnessed.
Nothing here changes the model.

The JavaScript twin (``sim/patch.js``) implements the same rule; ``sim/parity.js`` checks both
against ``ref/fixture.json`` produced by ``ref/make_fixture.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecordPlan:
    inputs: np.ndarray
    output: np.ndarray
    losses: tuple[float, ...]
    step_sizes: tuple[float, ...]
    replays: int
    reason: str
    converged: bool
    residual: float | None

    @property
    def improved(self) -> bool:
        return self.losses[-1] < self.losses[0]


def plan(
    net,
    inputs: np.ndarray,
    goal: np.ndarray,
    weights: np.ndarray,
    controls,
    lower,
    upper,
    *,
    feedback=(),
    feedback_add: bool = False,
    fixed_read: bool = False,
    state: np.ndarray | None = None,
    rate: float = 1.0,
    max_steps: int = 8,
    max_backtracks: int = 8,
    tolerance: float = 1e-9,
) -> RecordPlan:
    inputs = np.asarray(inputs, dtype=float)
    goal = np.asarray(goal, dtype=float)
    weights = np.asarray(weights, dtype=float)
    T, I = inputs.shape
    H, O = net.hidden, net.outputs
    if goal.shape != (T, O) or weights.shape != (O,):
        raise ValueError("goal is (time, outputs) and weights has one entry per output")
    p = net.parameters()
    B, b, G, C = p["B"], p["b"], p["G"], p["C"]
    live = net.state
    boundary = (np.zeros(H) if live is None else live[0]) if state is None else np.asarray(state, dtype=float)
    is_control = np.zeros(I, dtype=bool)
    is_control[np.asarray(list(controls), dtype=int)] = True
    lower = np.broadcast_to(np.asarray(lower, dtype=float), (I,)); upper = np.broadcast_to(np.asarray(upper, dtype=float), (I,))  # one bound per input port
    feedback = [(int(o), int(i)) for o, i in feedback]

    held: list = [None]  # with fixed_read: the record read of the initial path, held through every replay

    def forward(U: np.ndarray):
        U = U.copy()
        hidden = np.empty((T, H)); gate = np.empty((T, H)); port = np.empty((T, H)); out = np.empty((T, O))
        h = boundary.copy()
        for t in range(T):
            path = net.imagine(U[None, t : t + 1], state=h[None])
            hidden[t] = path.hidden[0, 0]; gate[t] = path.gate[0, 0]
            out[t] = path.output[0, 0] if held[0] is None else path.output[0, 0] - path.read[0, 0] + held[0][t]
            port[t] = np.tanh(B @ U[t] + b)
            h = hidden[t]
            if t + 1 < T:
                for oi, ii in feedback:
                    U[t + 1, ii] = (U[t, ii] if feedback_add else 0.0) + out[t, oi]
        return U, hidden, gate, port, out

    def cost(out: np.ndarray) -> float:
        return float(0.5 * np.mean(weights * (out - goal) ** 2))

    current, hidden, gate, port, out = forward(inputs)
    if fixed_read:
        h0 = boundary.copy(); reads0 = np.empty((T, O))
        for t in range(T):
            path = net.imagine(current[None, t : t + 1], state=h0[None]); reads0[t] = path.read[0, 0]; h0 = path.hidden[0, 0]
        held[0] = reads0
    costs, steps, replays = [cost(out)], [], 1
    if not np.isfinite(costs[0]):
        return RecordPlan(current, out, tuple(costs), (), replays, "nonfinite_initial", False, None)
    reason, converged, residual = "step_cap", False, None
    for iteration in range(max_steps + 1):
        du = np.zeros((T, I)); carried = np.zeros(H)
        for t in range(T - 1, -1, -1):
            d = weights * (out[t] - goal[t]) / (T * O)
            if t + 1 < T:
                for oi, ii in feedback:
                    d[oi] += du[t + 1, ii]
                    if feedback_add:
                        du[t, ii] += du[t + 1, ii]
            gh = C.T @ d + carried
            l, z = gate[t], port[t]
            prev = hidden[t - 1] if t > 0 else boundary
            gp = (1 - l) * gh * (1 - z**2)
            gs = gh * (prev - z) * l * (1 - l)
            du[t] += B.T @ gp + G.T @ gs
            carried = l * gh
        du[:, ~is_control] = 0.0
        if not np.isfinite(du).all():
            reason = "nonfinite_input_gradient"; break
        projected = np.where(is_control, np.clip(current - du, lower, upper), current)
        residual = float(np.abs(current - projected).max())
        if residual <= tolerance:
            converged, reason = True, "projected_stationary"; break
        if iteration == max_steps:
            break
        accepted, step = False, float(rate)
        for _ in range(max_backtracks):
            proposal = np.where(is_control, np.clip(current - step * du, lower, upper), current)
            slope = float(np.sum(du * (proposal - current)))
            if slope < 0:
                trial, th, tg, tp, tout = forward(proposal); replays += 1
                c = cost(tout)
                if np.isfinite(c) and c < costs[-1] and c <= costs[-1] + 1e-4 * slope:
                    current, hidden, gate, port, out = trial, th, tg, tp, tout
                    costs.append(c); steps.append(step); accepted = True
                    break
            step *= 0.5
        if not accepted:
            reason = "no_decreasing_causal_step"; break
    return RecordPlan(current, out, tuple(costs), tuple(steps), replays, reason, converged, residual)
