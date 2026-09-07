"""A two-joint arm with a pen, a 16x16 canvas, single-stroke signs, and a teacher that writes them.

The canvas is G x G cells over the square [LO, LO + SIDE]^2 of the plane. The pen sits on a
cell and, each step, moves to one of its eight neighbours or stays: nine actions. The arm,
anchored at the origin with two unit links, follows the pen by inverse kinematics, the way
a spinal reflex follows an intended hand position; its joint angles are what the page
draws. Every cell the pen visits is inked.

A sign is a polyline of one to three straight segments, one of six shapes with random
placement and size, rasterised to an ordered list of cells: its stroke. The teacher writes
a sign by heading for the first stroke cell that is not yet inked, and stays when none is
left. The net sees pixels through a window of K x K cells centred on its pen: the sign and
the canvas so far. That window is what makes the rule the same wherever the pen is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 16
K = 7  # the window around the pen, K x K cells
LO, SIDE = 0.25, 1.15
STEPS = 40
SHAPES = ("bar", "dash", "slash", "L", "V", "Z")
ACTIONS = 9  # (dr, dc) in {-1, 0, +1}^2, index 3 * (dr + 1) + (dc + 1); 4 is stay
INPUTS = 2 * K * K


def centre(r: int, c: int) -> tuple[float, float]:
    return LO + SIDE * (c + 0.5) / G, LO + SIDE * (r + 0.5) / G


def inverse(x: float, y: float) -> tuple[float, float]:
    """Joint angles that put the tip at (x, y), elbow up."""
    r2 = min(max(x * x + y * y, 0.05), 3.95)
    b = float(np.arccos((r2 - 2.0) / 2.0))
    a = float(np.arctan2(y, x) - np.arctan2(np.sin(b), 1.0 + np.cos(b)))
    return a, b


def polyline(shape: str, rng: np.random.Generator) -> np.ndarray:
    size = rng.uniform(0.55, 0.85)
    if shape == "bar":
        points = [(0.5, 0.0), (0.5, 1.0)]
    elif shape == "dash":
        points = [(0.0, 0.5), (1.0, 0.5)]
    elif shape == "slash":
        points = [(0.0, 0.0), (1.0, 1.0)]
    elif shape == "L":
        points = [(0.15, 1.0), (0.15, 0.0), (1.0, 0.0)]
    elif shape == "V":
        points = [(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)]
    else:  # Z
        points = [(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)]
    p = np.asarray(points, dtype=float) * size
    return p + rng.uniform(0.0, 1.0 - size, size=2)


def stroke_of(points: np.ndarray) -> list[tuple[int, int]]:
    """The ordered cells a fine walk along the polyline passes through, consecutive duplicates removed."""
    cells: list[tuple[int, int]] = []
    for (u0, v0), (u1, v1) in zip(points[:-1], points[1:], strict=True):
        for t in np.linspace(0.0, 1.0, 96):
            u, v = u0 + t * (u1 - u0), v0 + t * (v1 - v0)
            rc = (min(int(v * G), G - 1), min(int(u * G), G - 1))
            if not cells or cells[-1] != rc:
                cells.append(rc)
    return cells


@dataclass
class Episode:
    shape: str
    sign: np.ndarray  # (G, G)
    stroke: list[tuple[int, int]]


def make_episode(rng: np.random.Generator, shape: str | None = None) -> Episode:
    shape = shape or str(rng.choice(SHAPES))
    stroke = stroke_of(polyline(shape, rng))
    sign = np.zeros((G, G))
    for r, c in stroke:
        sign[r, c] = 1.0
    return Episode(shape, sign, stroke)


def teacher_action(canvas: np.ndarray, pen: tuple[int, int], stroke: list[tuple[int, int]]) -> int:
    """Head for the first stroke cell not yet inked; stay when the stroke is done."""
    for r, c in stroke:
        if canvas[r, c] == 0.0:
            dr = int(np.sign(r - pen[0]))
            dc = int(np.sign(c - pen[1]))
            return 3 * (dr + 1) + (dc + 1)
    return 4


def window(field: np.ndarray, pen: tuple[int, int]) -> np.ndarray:
    """The K x K cells of ``field`` centred on the pen, zero outside the canvas."""
    h = K // 2
    padded = np.zeros((G + 2 * h, G + 2 * h))
    padded[h : h + G, h : h + G] = field
    return padded[pen[0] : pen[0] + K, pen[1] : pen[1] + K]


def observation(sign: np.ndarray, canvas: np.ndarray, pen: tuple[int, int]) -> np.ndarray:
    """(2 * K * K,): the sign and the canvas so far, seen through the window around the pen."""
    return np.concatenate([window(sign, pen).ravel(), window(canvas, pen).ravel()])


def move(pen: tuple[int, int], action: int) -> tuple[int, int]:
    dr, dc = divmod(int(action), 3)
    return (min(max(pen[0] + dr - 1, 0), G - 1), min(max(pen[1] + dc - 1, 0), G - 1))


def write(policy, episode: Episode, rng: np.random.Generator | None = None, noise: float = 0.0):
    """Drive the pen with ``policy(observation) -> action`` (the teacher when ``policy`` is None).

    Returns the canvas, the trace of (observation, teacher label) at every step, and the pen
    path. With ``noise`` > 0 a random action replaces the chosen one now and then; the labels
    are always the teacher's, so slips become recovery demonstrations.
    """
    pen = episode.stroke[0]
    canvas = np.zeros((G, G))
    canvas[pen] = 1.0
    trace: list[tuple[np.ndarray, int]] = []
    path = [pen]
    for _ in range(STEPS):
        obs = observation(episode.sign, canvas, pen)
        label = teacher_action(canvas, pen, episode.stroke)
        trace.append((obs, label))
        action = label if policy is None else policy(obs)
        if rng is not None and noise > 0 and rng.random() < noise:
            action = int(rng.integers(0, ACTIONS))
        pen = move(pen, action)
        canvas[pen] = 1.0
        path.append(pen)
    return canvas, trace, path


def iou(drawn: np.ndarray, sign: np.ndarray) -> float:
    inter = float(((drawn > 0) & (sign > 0)).sum())
    union = float(((drawn > 0) | (sign > 0)).sum())
    return inter / union if union else 0.0
