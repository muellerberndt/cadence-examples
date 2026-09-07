"""What the trained paddle does as a function of where the ball is relative to it.

Run:  python policy_table.py            (reads net.json)

For every ball row and paddle position, with the ball one column from the paddle and
moving toward it horizontally, settle the exported net and record its action. The table
shows, per offset (ball row minus paddle centre row), how often it chose up, stay, down.
A paddle that tracks goes up for negative offsets, stays at zero, and goes down for
positive ones; the first version of this example chose up or down at random near zero.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from pong import H, PADDLE, W

HERE = Path(__file__).resolve().parent


def main() -> None:
    net = json.loads((HERE / "net.json").read_text())
    n, rule = net["n"], net["rule"]
    weights = np.asarray(net["W"]).reshape(n, n)
    bias = np.asarray(net["bias"])
    inputs, outputs = net["sets"]["input"], net["sets"]["output"]

    def act(v: np.ndarray) -> np.ndarray:
        r = 1 / (1 + np.exp(-rule["slope"] * (v - rule["threshold"]))) - rule["rest"]
        return np.where(r > 0, r / (1 - rule["rest"]), rule["leak"] * r / rule["rest"])

    def settle(drive: np.ndarray) -> np.ndarray:
        v = np.zeros(n)
        s = np.zeros(n)
        for _ in range(100):
            v = v + rule["dt"] * (-v + s @ weights + drive + bias)
            fresh = act(v)
            moved = np.abs(fresh - s).max()
            s = fresh
            if moved < 1e-4:
                break
        return s

    def frame(ball_r: int, ball_c: int, left: int, right: int) -> np.ndarray:
        f = np.zeros(H * W)
        for k in range(PADDLE):
            f[(left + k) * W] = 0.6
            f[(right + k) * W + W - 1] = 0.6
        f[ball_r * W + ball_c] = 1.0
        return f

    table: dict[int, Counter[int]] = {}
    for ball_r in range(H):
        for top in range(H - PADDLE + 1):
            now = frame(ball_r, W - 2, 4, top)  # ball one column from the paddle
            before = frame(ball_r, W - 3, 4, top)  # it came from the left, level
            drive = np.zeros(n)
            drive[inputs[: H * W]] = now * rule["clamp"]
            drive[inputs[H * W :]] = before * rule["clamp"]
            s = settle(drive)
            action = int(np.argmax(s[outputs]))
            table.setdefault(ball_r - (top + PADDLE // 2), Counter())[action] += 1
    print("ball row minus paddle centre -> how often the net chose up / stay / down")
    for offset in sorted(table):
        c = table[offset]
        total = sum(c.values())
        print(f"  {offset:+3d}:  up {c[0]:2d}   stay {c[1]:2d}   down {c[2]:2d}   (of {total})")


if __name__ == "__main__":
    main()
