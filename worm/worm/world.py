"""The plate, its smells, and the worm's body. Mirrored in web/life.js; tests/body.mjs
checks that both lay the same track."""
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
    """The centreline, head first, always exactly one body length. Whichever end
    leads lays the track and the rest follows it; the leading end can only
    change direction at a bounded curvature."""
    x: float
    y: float
    heading: float             # where the head means to go
    theta: float = 0.0         # where it is going: follows heading and head swing, curvature-bounded
    phase: float = 0.0         # of the body wave, advanced by distance travelled
    tail_heading: float = 0.0
    tail_theta: float = 0.0
    tail_turn: float = 0.0
    trail: list = field(default_factory=list)
    mode: str = "forward"      # forward | reverse
    timer: float = 0.0
    turn: float = 0.0          # what is left of a turn under way
    turn_after: float = 0.0    # the omega turn that follows the reversal
    stuck: int = 0             # steps in a row with the nose against the edge


TAU = 2 * math.pi


def wrap(a: float) -> float:
    a = math.fmod(a + math.pi, TAU)
    if a < 0:
        a += TAU
    return a - math.pi


def clamp(v: float, m: float) -> float:
    return max(-m, min(m, v))


class World:
    def __init__(self, p: dict, rng: Rng) -> None:
        self.p, self.rng = p, rng
        self.w, self.h = p["plate"]
        self.items: list[Item] = []
        self.t = 0.0
        heading = rng.uniform(-math.pi, math.pi)
        L = p["body_length"]
        crawl = 1.25 * L
        x0 = self.w / 2 - 0.926 * crawl * math.cos(heading)
        y0 = self.h / 2 - 0.926 * crawl * math.sin(heading)
        self.body = Body(x0, y0, heading, theta=heading)
        for k in range(101):
            self.body.trail.append((x0 - L * k / 100 * math.cos(heading), y0 - L * k / 100 * math.sin(heading)))
        # born crawling: a body length and a quarter of travel gives it its wave
        for _ in range(int(round(crawl / (p["crawl_speed"] * p["physics_step"])))):
            self.physics(p["physics_step"], False)
        self.t = 0.0

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
    def direction(self, end: str, span: float = 0.0) -> float:
        """Direction of the centreline at one end, pointing out of the body; over
        `span` mm it is the body's axis there."""
        tr = self.body.trail
        n = len(tr)
        step, first = (1, 0) if end == "head" else (-1, n - 1)
        k, acc = first, 0.0
        while 0 <= k + step < n:
            acc += math.hypot(tr[k + step][0] - tr[k][0], tr[k + step][1] - tr[k][1])
            k += step
            if acc > max(span, 1e-9):
                break
        return math.atan2(tr[first][1] - tr[k][1], tr[first][0] - tr[k][0])

    def trim(self, end: str) -> None:
        """Keep the centreline exactly one body length, giving up the excess at the trailing end."""
        tr = self.body.trail
        extra = -self.p["body_length"]
        for k in range(1, len(tr)):
            extra += math.hypot(tr[k][0] - tr[k - 1][0], tr[k][1] - tr[k - 1][1])
        while extra > 0 and len(tr) > 2:
            a, b = (len(tr) - 1, len(tr) - 2) if end == "tail" else (0, 1)
            seg = math.hypot(tr[b][0] - tr[a][0], tr[b][1] - tr[a][1])
            if seg <= extra:
                tr.pop() if end == "tail" else tr.pop(0)
                extra -= seg
            else:
                f = extra / seg
                tr[a] = (tr[a][0] + f * (tr[b][0] - tr[a][0]), tr[a][1] + f * (tr[b][1] - tr[a][1]))
                extra = 0.0

    def wall_turn(self, x: float, y: float, heading: float, under_way: float) -> float | None:
        """The plate's edge. Within `wall_margin` of an edge a leading end must head inward, the more so
        the nearer the edge: a heading that points out is mirrored, one that runs along the edge is turned
        in. Returns the turn that does it, or None. `under_way` keeps a head-on turn from changing sides."""
        m = self.p["wall_margin"]
        hx, hy, hit = math.cos(heading), math.sin(heading), False
        for d, nx, ny in ((x, 1, 0), (self.w - x, -1, 0), (y, 0, 1), (self.h - y, 0, -1)):
            if d >= m:
                continue
            need = 0.4 * (1 - max(0.0, d) / m)
            c = hx * nx + hy * ny
            if c >= need:
                continue
            hit = True
            if c < 0:
                hx, hy, c = hx - 2 * c * nx, hy - 2 * c * ny, -c
            if c < need:
                side = 1 if nx * hy - ny * hx >= 0 else -1
                a = math.atan2(ny, nx) + side * math.acos(need)
                hx, hy = math.cos(a), math.sin(a)
        if not hit:
            return None
        turn = wrap(math.atan2(hy, hx) - heading)
        if under_way != 0 and abs(turn) > 2.4 and (turn > 0) != (under_way > 0):
            turn = math.copysign(TAU - abs(turn), under_way)
        return turn

    def inside(self, x: float, y: float) -> bool:
        e = 0.06
        return e <= x <= self.w - e and e <= y <= self.h - e

    def reverse(self, seconds: float, turn: float) -> None:
        """The tail leads for `seconds`, then the head curls through `turn` radians (an omega turn)."""
        b = self.body
        b.mode, b.timer, b.turn_after, b.turn, b.tail_turn = "reverse", seconds, turn, 0.0, 0.0
        b.tail_theta = self.direction("tail")
        b.tail_heading = self.direction("tail", self.p["swing_wavelength"])

    def resume(self) -> None:
        b = self.body
        b.mode = "forward"
        b.theta = self.direction("head")
        b.heading = self.direction("head", self.p["swing_wavelength"])
        b.turn = b.turn_after

    def lead(self, x: float, y: float, heading: float, theta: float, turn: float, ds: float):
        """One leading end, one step: the heading turns (an omega turn, or away from the edge), and the
        direction of travel follows the heading and the body wave with what is left of the bend the body allows."""
        p = self.p
        away = self.wall_turn(x, y, heading, turn)
        if away is not None:
            turn = away
        swing, d = p["swing_amplitude"], 0.0
        if turn != 0:
            d = clamp(turn, p["turn_curvature"] * ds)
            heading = wrap(heading + d)
            turn -= d
            swing *= p["turn_swing"]
        theta = wrap(theta + d + clamp(wrap(heading + swing * math.sin(self.body.phase) - theta - d),
                                       p["max_curvature"] * ds - abs(d)))
        return heading, theta, turn, x + ds * math.cos(theta), y + ds * math.sin(theta)

    def physics(self, dt: float, on_food: bool) -> None:
        b, p, tr = self.body, self.p, self.body.trail
        speed = p["reverse_speed"] if b.mode == "reverse" else p["dwell_speed"] if on_food else p["crawl_speed"]
        ds = speed * dt
        b.phase = math.fmod(b.phase + TAU * ds / p["swing_wavelength"], TAU)
        if b.mode == "reverse":
            tx, ty = tr[-1]
            b.tail_heading, b.tail_theta, b.tail_turn, nx, ny = self.lead(tx, ty, b.tail_heading, b.tail_theta, b.tail_turn, ds)
            if not self.inside(nx, ny):
                b.timer = 0.0            # the tail has met the edge of the plate all the same
            else:
                tr.append((nx, ny))
                self.trim("head")
                b.x, b.y = tr[0]
            b.timer -= dt
            if b.timer <= 0:
                self.resume()
        else:
            b.heading, b.theta, b.turn, nx, ny = self.lead(b.x, b.y, b.heading, b.theta, b.turn, ds)
            if self.inside(nx, ny):
                b.x, b.y, b.stuck = nx, ny, 0
                tr.insert(0, (b.x, b.y))
                self.trim("tail")
            elif b.stuck < 2:            # nose against the edge: back off and turn
                b.stuck += 1
                self.reverse(p["pirouette_reverse"], -2.2)
            else:                        # both ends against edges: slide along it
                e = 0.06
                b.x, b.y = min(max(nx, e), self.w - e), min(max(ny, e), self.h - e)
                tr.insert(0, (b.x, b.y))
                self.trim("tail")
        self.t += dt

    def swing_direction(self) -> float:
        return 1.0 if math.cos(self.body.phase) >= 0 else -1.0
