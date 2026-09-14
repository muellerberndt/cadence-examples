"""Independent Cadence checks for the coupled visual/motor circuit and spatial field."""

import cadence as cd
import numpy as np


def motor_settlement(q, target):
    a, b = q
    tip = np.array(
        [
            0.5 + 0.43 * np.cos(a) + 0.37 * np.cos(a + b),
            0.94 + 0.43 * np.sin(a) + 0.37 * np.sin(a + b),
        ]
    )
    j = np.array(
        [
            [-0.43 * np.sin(a) - 0.37 * np.sin(a + b), -0.37 * np.sin(a + b)],
            [0.43 * np.cos(a) + 0.37 * np.cos(a + b), 0.37 * np.cos(a + b)],
        ]
    )
    edges = []
    for v in range(2):
        for m in range(2):
            edges.extend([(m + 2, v, -j[v, m]), (v, m + 2, 0.9 * j[v, m])])
    pre, post, weights = zip(*edges)
    engine = cd.Settlement(
        cd.Wiring.from_edges(4, pre=pre, post=post, sign=weights),
        cd.GradedRule(gain=1, slope=2, threshold=0, leak=1, dt=0.25, clamp_amplitude=1),
    )
    drive = np.r_[np.asarray(target) - tip, [0, 0]]
    return engine.settle(drive, steps=80, tolerance=0).activation


def maze_settlement(world):
    grid, cols = world["grid"], world["cols"]
    edges = []
    for i, wall in enumerate(grid):
        if wall:
            continue
        x, y = i % cols, i // cols
        ns = [
            b * cols + a
            for a, b in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if 0 <= a < cols and 0 <= b < world["rows"] and not grid[b * cols + a]
        ]
        edges.extend((j, i, 0.985 / len(ns)) for j in ns)
    pre, post, weights = zip(*edges)
    engine = cd.Settlement(
        cd.Wiring.from_edges(len(grid), pre=pre, post=post, sign=weights),
        cd.GradedRule(gain=1, slope=2, threshold=0, leak=1, dt=1, clamp_amplitude=1),
    )
    drive = np.zeros(len(grid))
    drive[world["goal"]] = 0.03
    return engine.settle(
        drive, mask=1 - np.asarray(grid), steps=2400, tolerance=1e-13
    ).activation
