"""A batched free-settle twin of the library's reference solver, for reading many boards.

Cadence 0.50 settles one query at a time: ``Brain.settle`` starts from the live state and
repairs it by projected gradient steps with a secant step size and Armijo backtracking
(``cadence/_repair.py``). The search reads thousands of positions per move, so this module
solves the same free-query problem for a whole batch of rows at once with NumPy: the same
energy ``sum(e**2)/2 + state_prior*sum(x**2)/2``, the same analytic derivatives through
state and error-readback connections, the same acceptance rule, step estimate, bounds and
qualification threshold, each row carrying its own step size and line search.

The twin is rebuilt from ``brain.graph``, ``brain.weights`` and ``brain.biases`` whenever
the brain has learned, and it is bound to the library by ``parity``: the same readings
through ``brain.settle`` and through the twin, compared output by output. The page's
JavaScript engine plays the same role for the record patch; this is that pattern for the
deep brain. Summation order differs from ``math.fsum``, so agreement is near machine
precision but not bitwise; ``parity`` reports the worst difference found.
"""

from __future__ import annotations

import numpy as np

ARMIJO = 1e-4


class FreeSettleTwin:
    """Batched free queries against one brain's frozen parameters, from the zero state.

    The deep value patch never calls ``step`` or single ``observe``, so its live state
    stays at the build-time zeros and every query legitimately starts from rest, exactly
    as ``brain.settle`` would.
    """

    def __init__(self, brain, output_index: int) -> None:
        graph, config = brain.graph, brain.config
        n, m = graph.n_patches, graph.n_inputs
        if any(x != 0.0 for x in brain.state):
            raise ValueError("the twin settles from rest; this brain holds live state")
        self.n, self.m = n, m
        self.output_index = int(output_index)
        self.state_prior = float(config["state_prior"])
        self.state_bound = float(config["state_bound"])
        self.tolerance = float(config["tolerance"])
        self.budget = int(config["settle_budget"])
        self.step = float(config["step"])
        self.backtracks = int(config["backtracks"])
        self.ceiling = np.ldexp(self.step, min(self.backtracks - 1, 1023))
        self.biases = np.array(brain.biases, dtype=float)
        weights = np.array(brain.weights, dtype=float)
        a_in = np.zeros((n, m))
        a_x = np.zeros((n, n))
        a_e = np.zeros((n, n))
        for w, (kind, source, target) in zip(weights, graph.edges, strict=True):
            {"input": a_in, "state": a_x, "residual": a_e}[kind][target, source] += w
        # Patches grouped by depth over error-readback edges: every error is computed
        # before any patch that reads it, as the library's residual_order guarantees.
        depth = np.zeros(n, dtype=int)
        for target in graph.residual_order:
            sources = np.flatnonzero(a_e[target])
            if len(sources):
                depth[target] = depth[sources].max() + 1
        self.levels = [np.flatnonzero(depth == d) for d in range(depth.max() + 1)]
        self.a_in_t = [a_in[rows].T.copy() for rows in self.levels]
        self.a_x_t = [a_x[rows].T.copy() for rows in self.levels]
        self.a_e_t = [a_e[rows].T.copy() for rows in self.levels]
        self.a_x_rows = [a_x[rows].copy() for rows in self.levels]
        self.a_e_rows = [a_e[rows].copy() for rows in self.levels]

    # ------------------------------------------------------------- the energy

    def _evaluate(self, u: np.ndarray, x: np.ndarray):
        """Energy, predictions and the exact state gradient for each row."""
        p = np.zeros_like(x)
        e = np.zeros_like(x)
        for rows, a_in_t, a_x_t, a_e_t in zip(self.levels, self.a_in_t, self.a_x_t, self.a_e_t, strict=True):
            drive = self.biases[rows] + u @ a_in_t + x @ a_x_t + e @ a_e_t
            p[:, rows] = np.tanh(drive)
            e[:, rows] = x[:, rows] - p[:, rows]
        energy = 0.5 * np.sum(e * e, axis=1) + 0.5 * self.state_prior * np.sum(x * x, axis=1)
        grad = self.state_prior * x
        adj_e = e.copy()
        for rows, a_x_rows, a_e_rows in zip(reversed(self.levels), reversed(self.a_x_rows), reversed(self.a_e_rows), strict=True):
            adj = adj_e[:, rows]
            grad[:, rows] += adj
            h = -adj * (1.0 - p[:, rows] ** 2)
            grad += h @ a_x_rows
            adj_e += h @ a_e_rows
        return energy, p, e, grad

    def _residual(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """The library's projected stationarity measure, per row (its stable form)."""
        projected = np.where(grad >= 0, np.minimum(grad, x + self.state_bound),
                             np.maximum(grad, x - self.state_bound))
        return np.abs(projected).max(axis=1)

    # -------------------------------------------------------------- the solve

    def settle(self, u: np.ndarray, *, budget: int | None = None):
        """Settle every row of ``u`` from rest. Returns settled states, qualification
        and sweep counts; the solve follows ``cadence/_repair.settle`` row by row."""
        u = np.asarray(u, dtype=float)
        count = len(u)
        x = np.zeros((count, self.n))
        energy, p, e, grad = self._evaluate(u, x)
        steps = np.full(count, self.step)
        sweeps = np.zeros(count, dtype=int)
        active = np.ones(count, dtype=bool)
        for _ in range(self.budget if budget is None else int(budget)):
            residual = self._residual(x[active], grad[active])
            settled = np.flatnonzero(active)[residual <= self.tolerance]
            active[settled] = False
            if not active.any():
                break
            rows = np.flatnonzero(active)
            trial = steps[rows].copy()
            accepted = np.zeros(len(rows), dtype=bool)
            for _attempt in range(self.backtracks):
                trying = ~accepted
                r = rows[trying]
                candidate = np.clip(x[r] - trial[trying, None] * grad[r], -self.state_bound, self.state_bound)
                displacement = candidate - x[r]
                slope = np.sum(grad[r] * displacement, axis=1)
                moved = np.any(displacement != 0.0, axis=1)
                c_energy, c_p, c_e, c_grad = self._evaluate(u[r], candidate)
                armijo = c_energy <= energy[r] + ARMIJO * slope
                finishing = np.abs(c_energy - energy[r]) <= 8 * np.spacing(np.abs(energy[r]))
                finishing &= self._residual(candidate, c_grad) <= self.tolerance
                ok = moved & np.isfinite(slope) & (slope < 0) & np.isfinite(c_energy) & (armijo | finishing)
                if ok.any():
                    taken = r[ok]
                    change = c_grad[ok] - grad[taken]
                    distance = np.sum(displacement[ok] ** 2, axis=1)
                    curvature = np.sum(displacement[ok] * change, axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        estimate = distance / curvature
                    safe = (curvature > 0) & np.isfinite(estimate) & (estimate > 0)
                    steps[taken] = np.where(safe, np.minimum(estimate, self.ceiling), self.step)
                    x[taken] = candidate[ok]
                    energy[taken], p[taken], e[taken], grad[taken] = c_energy[ok], c_p[ok], c_e[ok], c_grad[ok]
                    sweeps[taken] += 1
                    accepted[trying] = ok
                if accepted.all():
                    break
                trial[~accepted] *= 0.5
            active[rows[~accepted]] = False  # line search found no descent: refused
        energy, p, e, grad = self._evaluate(u, x)
        qualified = self._residual(x, grad) <= self.tolerance
        return {"state": x, "qualified": qualified, "sweeps": sweeps,
                "outputs": x[:, self.output_index]}


def parity(brain, twin: FreeSettleTwin, u: np.ndarray) -> dict:
    """The same rows through ``brain.settle`` and the twin: the worst output difference,
    the worst settled-state difference and both sides' qualification flags."""
    mine = twin.settle(u)
    worst_output, worst_state, library_qualified = 0.0, 0.0, []
    for k, row in enumerate(np.asarray(u, dtype=float)):
        result = brain.settle({"board": row.tolist()})
        library_qualified.append(bool(result["qualified"]))
        state = np.array(result["state"])
        worst_state = max(worst_state, float(np.max(np.abs(state - mine["state"][k]))))
        worst_output = max(worst_output, abs(state[twin.output_index] - mine["outputs"][k]))
    return {"rows": len(u), "worst_output": worst_output, "worst_state": worst_state,
            "library_qualified": library_qualified,
            "twin_qualified": mine["qualified"].tolist()}
