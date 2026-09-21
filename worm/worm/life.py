"""One life: sense, think, act, and learn from what actually happens.
Mirrored by web/life.js; the browser adds the user's hand."""
from __future__ import annotations

import math

import numpy as np

from .brain import WormBrain
from .connectome import params
from .rng import Rng
from .world import Item, World

EAT_RATE = 0.05          # a drop of bacteria lasts about 20 seconds of feeding


class Life:
    def __init__(self, condition: str = "paired", seed: int = 0, p: dict | None = None) -> None:
        self.p = p or params()
        self.condition, self.seed = condition, seed
        self.rng = Rng(seed * 7919 + 17)
        self.world = World(self.p, self.rng)
        self.brain = WormBrain(seed, lesion=("AWAL", "AWAR") if condition == "lesion_AWA" else (),
                               frozen=condition == "frozen", p=self.p)
        self.prenatal = self.brain.born()
        self.tick_count = 0
        self.prev = {"food": 0.0, "pain": 0.0}
        self.value_prev = 0.0
        self.bias = 0.0
        self.outcome_ticks: list[int] = []
        self.events: list[dict] = []
        self.pending = {"food": 0, "pain": 0}      # user treat / poke, in ticks
        self.readout = np.zeros(2)
        self._setup()

    # ---- the plate --------------------------------------------------------------
    def _free_spot(self, clearance: float = 1.6) -> tuple[float, float]:
        w, h = self.world.w, self.world.h
        for _ in range(200):
            x, y = self.rng.uniform(1.0, w - 1.0), self.rng.uniform(1.0, h - 1.0)
            if all(math.hypot(x - it.x, y - it.y) > 2 * clearance for it in self.world.items) and \
                    math.hypot(x - self.world.body.x, y - self.world.body.y) > clearance:
                return x, y
        return x, y

    def _setup(self) -> None:
        smell = self.condition != "unpaired"
        for _ in range(self.p["food_drops"]):
            self.world.items.append(Item("food", *self._free_spot(), "A" if smell else None))
        for _ in range(self.p["noxious_drops"]):
            self.world.items.append(Item("noxious", *self._free_spot(), "B" if smell else None))
        if not smell:   # the same smells exist, but they predict nothing
            for odour in ("A", "A", "B", "B"):
                self.world.items.append(Item("smell", *self._free_spot(), odour))

    def _respawn(self, item: Item) -> None:
        item.x, item.y = self._free_spot()
        item.amount = 1.0

    # ---- one brain tick -----------------------------------------------------------
    def tick(self) -> dict:
        p, world, brain = self.p, self.world, self.brain
        steps = int(round(p["tick"] / p["physics_step"]))
        for _ in range(steps):
            world.physics(p["physics_step"], world.contact("food") is not None)
        a, b = world.odour(world.body.x, world.body.y)
        eating = world.contact("food")
        hurting = world.contact("noxious")
        food = 1.0 if (eating is not None or self.pending["food"] > 0) else 0.0
        pain = 1.0 if (hurting is not None or self.pending["pain"] > 0) else 0.0
        for k in self.pending:
            self.pending[k] = max(0, self.pending[k] - 1)
        u = np.array([a, b, food, pain])
        y = brain.sense(u)
        self.readout = y
        n = self.tick_count
        food_on, pain_on = food > self.prev["food"], pain > self.prev["pain"]
        self.prev = {"food": food, "pain": pain}
        if eating is not None:
            eating.amount -= EAT_RATE * p["tick"]
            if eating.amount <= 0:
                self._respawn(eating)

        lessons = []
        arrived = tuple(k for k, on in (("food", food_on), ("pain", pain_on)) if on)
        for kind in arrived:
            self.outcome_ticks.append(n)
            self.events.append({"tick": n, "event": kind})
        if arrived:
            lessons.append(brain.learn(*arrived))

        # The body reads the command interneurons: AVB/PVC forward, AVA/AVD/AVE reverse.
        body = world.body
        forward, reverse = float(y[0]), float(y[1])
        value = forward - reverse
        dv = value - self.value_prev
        self.value_prev = value
        if body.mode == "forward":
            self.bias = p["taxis_memory"] * self.bias + (1 - p["taxis_memory"]) * dv * world.swing_direction()
            body.heading += p["taxis"] * self.bias
            if eating is None:
                turn = (p["base_turn"] + p["reverse_gain"] * max(0.0, reverse - forward - p["reverse_threshold"])
                        + p["kinesis"] * max(0.0, -dv))
                if self.rng.random() < turn:
                    side = -1.0 if self.rng.random() < p["ventral_bias"] else 1.0   # omega turns curl ventrally, mostly
                    hold = p["pirouette_reverse"] * (1.0 + 2.0 * min(1.0, max(0.0, reverse)))
                    world.reverse(hold, side * self.rng.uniform(1.5, 3.0))
        self.tick_count += 1
        return {"tick": n, "u": u, "y": y, "lessons": lessons}

    def treat(self) -> None:
        self.pending["food"] = 2

    def poke(self) -> None:
        self.pending["pain"] = 2

    def nearest(self, kind: str) -> float:
        d = [math.hypot(self.world.body.x - it.x, self.world.body.y - it.y)
             for it in self.world.items if it.kind == kind and it.amount > 0]
        return min(d) if d else float("nan")
