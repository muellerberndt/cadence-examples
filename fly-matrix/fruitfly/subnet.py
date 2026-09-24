"""The flight sub-net: the part of the nervous system the browser settles live.

Recruited from declared seed populations along the strongest synapse classes for a few hops,
capped by a budget; the whole-brain view draws every neuron, the sub-net computes. Which
neurons are in it, and how far its settled state deviates from the whole brain's under the
flight stimuli, is measured by ``tools/subnet_closure.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cadence import Connectome

__all__ = ["SEEDS", "recruit", "SubNet"]

SEEDS: tuple[str, ...] = (
    # flight: the senses of rotation and the wing motor system
    "haltere", "wing_sense", "steering", "power", "tension", "mn:haltere", "neck_mn",
    "ocelli", "lptc:hs", "lptc:vs", "dn", "gf", "mn:ttm",
    "jo:C:left", "jo:C:right", "jo:E:left", "jo:E:right",
    # the instincts: looming, landing, odour, taste, the mushroom body
    "vis:LC4", "vis:LPLC2", "dn:landing", "orn:decaying_fruit", "orn:yeasty", "orn:aversive", "orn:fruity",
    "kc", "dan:pam", "dan:ppl1", "mbon", "grn:sugar:labellum", "grn:sugar:front_leg", "grn:bitter:labellum", "mn9", "leg_touch", "thermo",
)


@dataclass(frozen=True)
class SubNet:
    members: np.ndarray  # whole-brain indices in sub-net order
    connectome: Connectome  # the induced sub-connectome, re-indexed
    hops: np.ndarray  # recruitment hop per member (0 = seed)

    @property
    def n(self) -> int:
        return len(self.members)


def recruit(whole: Connectome, *, budget: int = 30000, hops: int = 3, min_count: float = 10.0, seeds: tuple[str, ...] = SEEDS) -> SubNet:
    """Breadth-first over synapse classes at ``min_count`` or more, outward and inward, strongest first."""
    n = whole.n
    inside = np.zeros(n, dtype=bool)
    hop = np.full(n, -1, dtype=np.int32)
    for name in seeds:
        idx = list(whole.populations.get(name, ()))
        inside[idx] = True
        hop[idx] = 0
    strong = whole.count >= min_count
    pre, post, count = whole.pre[strong], whole.post[strong], whole.count[strong]
    for h in range(1, hops + 1):
        if inside.sum() >= budget:
            break
        touching = (inside[pre] & ~inside[post]) | (inside[post] & ~inside[pre])
        candidates = np.where(inside[pre[touching]], post[touching], pre[touching])
        weight = np.bincount(candidates, weights=count[touching], minlength=n)
        order = np.argsort(-weight, kind="stable")
        order = order[weight[order] > 0]
        room = budget - int(inside.sum())
        chosen = order[:room]
        inside[chosen] = True
        hop[chosen] = h
    members = np.flatnonzero(inside)
    index = np.full(n, -1, dtype=np.int64)
    index[members] = np.arange(len(members))
    keep = inside[whole.pre] & inside[whole.post]
    pops = {k: tuple(int(index[i]) for i in v if index[i] >= 0) for k, v in whole.populations.items()}
    pops = {k: v for k, v in pops.items() if v}
    sub = Connectome(len(members), index[whole.pre[keep]], index[whole.post[keep]], whole.count[keep], whole.sign[keep], pops, whole.label + ":flight")
    return SubNet(members=members, connectome=sub, hops=hop[members])
