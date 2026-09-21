"""The plate, its smells, and the worm's body. Mirrored exactly in web/world.js."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .rng import Rng


@dataclass
class Item:
    kind: str                  # "food" or "noxious"
    x: float
    y: float
    odour: str | None          # "A", "B" or None
    amount: float = 1.0        # food is eaten


@dataclass
class Body:
    x: float
    y: float
    heading: float
    trail: list = field(default_factory=list)   # head positions, newest first
    mode: str = "forward"                        # forward | reverse
    timer: float = 0.0
    turn_after: float = 0.0


class World:
    def __init__(self, p: dict, rng: Rng) -> None:
        self.p, self.rng = p, rng
        self.w, self.h = p["plate"]
        self.items: list[Item] = []
        self.t = 0.0
        self.body = Body(self.w / 2, self.h / 2, rng.uniform(-math.pi, math.pi))
        L, n = p["body_length"], 60
        for k in range(n):           # born stretched out behind its heading
            d = 3 * L * k / n
            self.body.trail.append((self.body.x - d * math.cos(self.body.heading),
                                    self.body.y - d * math.sin(self.body.heading)))

    # ---- smells -----------------------------------------------------------------
    def odour(self, x: float, y: float) -> tuple[float, float]:
        s2 = 2 * self.p["odour_sigma"] ** 2
        a = b = 0.0
        for it in self.items:
            if it.odour is None or it.amount <= 0:
                continue
            c = it.amount ** 0.5 * math.exp(-((x - it.x) ** 2 + (y - it.y) ** 2) / s2)
            if it.odour == "A":
                a += c
            else:
                b += c
        return min(a, 1.0), min(b, 1.0)

    def contact(self, kind: str) -> Item | None:
        r = self.p["food_radius"] if kind == "food" else self.p["noxious_radius"]
        for it in self.items:
            if it.kind == kind and it.amount > 0 and math.hypot(self.body.x - it.x, self.body.y - it.y) <= r:
                return it
        return None

    # ---- the body ---------------------------------------------------------------
    def reverse(self, seconds: float, turn: float) -> None:
        self.body.mode, self.body.timer, self.body.turn_after = "reverse", seconds, turn

    def physics(self, dt: float, on_food: bool) -> None:
        b, p = self.body, self.p
        if b.mode == "reverse":
            dist = p["reverse_speed"] * dt
            while dist > 0 and len(b.trail) > 2:
                hx, hy = b.trail[0]
                nx, ny = b.trail[1]
                seg = math.hypot(nx - hx, ny - hy)
                if seg <= dist:
                    b.trail.pop(0)
                    dist -= seg
                else:
                    f = dist / seg
                    b.trail[0] = (hx + f * (nx - hx), hy + f * (ny - hy))
                    dist = 0
            b.x, b.y = b.trail[0]
            b.timer -= dt
            if b.timer <= 0:
                b.mode = "forward"
                b.heading += b.turn_after
                # face along the body's own axis after reversing, then turn
                if len(b.trail) > 3:
                    tx, ty = b.trail[3]
                    b.heading = math.atan2(b.y - ty, b.x - tx) + b.turn_after
        else:
            speed = p["dwell_speed"] if on_food else p["crawl_speed"]
            phase = b.heading + p["swing_amplitude"] * math.sin(2 * math.pi * self.t / p["swing_period"])
            b.x += speed * dt * math.cos(phase)
            b.y += speed * dt * math.sin(phase)
            m = 0.25
            if b.x < m or b.x > self.w - m:
                b.heading = math.pi - b.heading
                b.x = min(max(b.x, m), self.w - m)
            if b.y < m or b.y > self.h - m:
                b.heading = -b.heading
                b.y = min(max(b.y, m), self.h - m)
            b.trail.insert(0, (b.x, b.y))
        # keep three body lengths of trail
        total, keep = 0.0, 1
        for k in range(1, len(b.trail)):
            total += math.hypot(b.trail[k][0] - b.trail[k - 1][0], b.trail[k][1] - b.trail[k - 1][1])
            keep = k + 1
            if total > 3 * p["body_length"]:
                break
        del b.trail[keep:]
        self.t += dt

    def swing_direction(self) -> float:
        return 1.0 if math.cos(2 * math.pi * self.t / self.p["swing_period"]) >= 0 else -1.0
