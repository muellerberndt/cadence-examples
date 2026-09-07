"""A small pixel Pong, vectorised over many games at once.

The field is ``H`` rows by ``W`` columns of pixels. The agent's paddle is on the right
edge, the opponent's on the left; each is ``PADDLE`` pixels tall and moves one row per
step. The ball moves one column per step and -1, 0, or +1 rows; it bounces off the top
and bottom, and off a paddle that covers its row when it reaches that paddle's column,
leaving with a vertical velocity set by where on the paddle it struck. A ball that
reaches a paddle's column uncovered is a miss for that side and the point ends.

The agent sees two frames, the current one and the one before, so the ball's direction
is visible: the field as pixels, ball and paddles lit, twice. Actions are 0 = up,
1 = stay, 2 = down. Rewards: +1 when the agent's paddle returns the ball, -1 when
it misses, and a small shaping term each step for the distance between paddle centre and
ball row, so credit does not have to travel a whole rally. The opponent tracks the ball
with a fixed lag and is not learned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H, W, PADDLE = 12, 16, 3
ACTIONS = 3
SHAPING = 0.05
MAX_RALLY = 400  # a point ends after this many steps even if nobody misses


@dataclass
class Pong:
    envs: int
    seed: int = 0
    opponent_skill: float = 0.7  # probability the opponent moves toward the ball each step

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.ball_r = np.zeros(self.envs, dtype=int)
        self.ball_c = np.zeros(self.envs, dtype=int)
        self.ball_dr = np.zeros(self.envs, dtype=int)
        self.ball_dc = np.zeros(self.envs, dtype=int)
        self.left = np.zeros(self.envs, dtype=int)  # top row of the opponent's paddle
        self.right = np.zeros(self.envs, dtype=int)  # top row of the agent's paddle
        self.age = np.zeros(self.envs, dtype=int)  # steps since the point began
        self.previous = np.zeros((self.envs, H * W))
        self.reset(np.ones(self.envs, dtype=bool))

    def reset(self, which: np.ndarray) -> None:
        k = int(which.sum())
        if not k:
            return
        self.ball_r[which] = self.rng.integers(1, H - 1, k)
        self.ball_c[which] = W // 2
        self.ball_dr[which] = self.rng.choice([-1, 0, 1], k)
        self.ball_dc[which] = self.rng.choice([-1, 1], k)
        self.left[which] = self.rng.integers(0, H - PADDLE + 1, k)
        self.right[which] = self.rng.integers(0, H - PADDLE + 1, k)
        self.age[which] = 0
        self.previous[which] = self.frames()[which]  # a fresh point starts with a still ball

    def observation(self) -> np.ndarray:
        """(envs, 2*H*W): the current frame followed by the previous one."""
        return np.concatenate([self.frames(), self.previous], axis=1)

    def frames(self) -> np.ndarray:
        """(envs, H*W) pixels in [0, 1]: ball 1.0, paddles 0.6."""
        out = np.zeros((self.envs, H, W))
        rows = np.arange(PADDLE)
        idx = np.arange(self.envs)
        out[idx[:, None], self.left[:, None] + rows[None, :], 0] = 0.6
        out[idx[:, None], self.right[:, None] + rows[None, :], W - 1] = 0.6
        out[idx, self.ball_r, self.ball_c] = 1.0
        return out.reshape(self.envs, H * W)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Apply the agent's actions; returns (reward, done, info) and resets finished games."""
        self.previous = self.frames()
        # paddles
        self.right = np.clip(self.right + (action - 1), 0, H - PADDLE)
        centre = self.left + PADDLE // 2
        want = np.sign(self.ball_r - centre)
        move = np.where(self.rng.random(self.envs) < self.opponent_skill, want, 0)
        self.left = np.clip(self.left + move, 0, H - PADDLE)
        # ball
        r = self.ball_r + self.ball_dr
        bounce = (r < 0) | (r >= H)
        self.ball_dr = np.where(bounce, -self.ball_dr, self.ball_dr)
        r = np.clip(r, 0, H - 1)
        c = self.ball_c + self.ball_dc
        reward = np.zeros(self.envs)
        done = np.zeros(self.envs, dtype=bool)
        hits = np.zeros(self.envs, dtype=bool)
        misses = np.zeros(self.envs, dtype=bool)
        for side, paddle, col in (("right", self.right, W - 1), ("left", self.left, 0)):
            at = c == col
            covered = at & (r >= paddle) & (r < paddle + PADDLE)
            if side == "right":
                hits |= covered
                misses |= at & ~covered
            else:
                done |= at & ~covered  # the opponent missed: the point is over, no reward
            where_hit = np.flatnonzero(covered)
            offset = r[where_hit] - paddle[where_hit]  # 0 top, 1 middle, 2 bottom
            self.ball_dc[where_hit] = -self.ball_dc[where_hit]
            self.ball_dr[where_hit] = offset - 1
            c[where_hit] = col + (-1 if col == W - 1 else 1)  # back onto the field
        reward += hits.astype(float) - misses.astype(float)
        done |= misses
        self.age += 1
        done |= self.age >= MAX_RALLY
        self.ball_r, self.ball_c = r, np.clip(c, 0, W - 1)
        # shaping: how far the agent's paddle centre is from the ball's row
        distance = np.abs(self.ball_r - (self.right + PADDLE // 2)) / (H - 1)
        reward -= SHAPING * distance
        info = {"hit": hits, "miss": misses, "opponent_miss": done & ~misses & (self.age < MAX_RALLY)}
        self.reset(done)
        return reward, done, info


def track_policy(env: Pong) -> np.ndarray:
    """The scripted tracker, for reference: move the paddle centre toward the ball row."""
    centre = env.right + PADDLE // 2
    return (np.sign(env.ball_r - centre) + 1).astype(int)
