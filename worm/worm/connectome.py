"""The wiring the worm is born with, read from data/connectome.json."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def params() -> dict:
    return json.loads((DATA / "params.json").read_text())


@dataclass(frozen=True)
class Wiring:
    names: list[str]                          # the 302 neurons, in connectome order
    order: list[int]                          # declaration order: indices by sensory depth
    pos: list[int]                            # each neuron's place in that order
    state_partners: list[tuple[int, ...]]     # per neuron: presynaptic partners read within the tick
    prev_partners: list[tuple[int, ...]]      # per neuron: partners read from the previous tick, and itself
    cell_senses: list[tuple[str, ...]]        # per neuron: the senses it receives
    senses: list[str]                         # sense order, as the brain's input vector has it
    readouts: list[str]                       # readout order: forward, reverse


def load() -> dict:
    return json.loads((DATA / "connectome.json").read_text())


def wiring(p: dict | None = None, lesion: tuple[str, ...] = ()) -> Wiring:
    """Which connections exist: every chemical synapse, every gap junction both
    ways, and each neuron's own state a moment ago. The neurons are declared in
    order of synaptic distance from the sensory cells, and each synapse that
    runs down that order is read live, within the tick's joint settlement,
    while each synapse that runs back up it, and every neuron's persistence
    term, is read from the previous tick. Signals therefore flow from the
    senses toward the commands within a tick, and feedback takes a tick — and
    a lesson's correction can reach back through the live chain to the weights
    of every synapse on it. Cadence 0.50 has no way to install inherited
    synaptic strengths, so existence is all the connectome supplies; every
    weight starts at the library's seeded value and is learned."""
    p = p or params()
    c = load()
    names = [n["name"] for n in c["neurons"]]
    index = {n: i for i, n in enumerate(names)}
    H = len(names)
    forward: list[set[int]] = [set() for _ in range(H)]   # pre -> its postsynaptic cells
    partners: list[set[int]] = [set() for _ in range(H)]  # post -> its presynaptic cells
    for pre, post, _count, _sign, _kind in c["chemical"]:
        forward[pre].add(post)
        partners[post].add(pre)
    for a, b, _count in c["gap"]:
        forward[a].add(b)
        forward[b].add(a)
        partners[a].add(b)
        partners[b].add(a)

    # synaptic distance from the sensory cells; cells no path reaches come last
    depth = [H] * H
    queue = deque()
    for cells in p["inputs"].values():
        for cell in cells:
            if depth[index[cell]] == H:
                depth[index[cell]] = 0
                queue.append(index[cell])
    while queue:
        i = queue.popleft()
        for j in forward[i]:
            if depth[j] > depth[i] + 1:
                depth[j] = depth[i] + 1
                queue.append(j)
    order = sorted(range(H), key=lambda i: (depth[i], i))
    pos = [0] * H
    for place, i in enumerate(order):
        pos[i] = place

    state_partners = [tuple(sorted(j for j in partners[i] if pos[j] < pos[i]))
                      for i in range(H)]
    prev_partners = [tuple(sorted({i, *(j for j in partners[i] if pos[j] > pos[i])}))
                     for i in range(H)]
    cell_senses: list[list[str]] = [[] for _ in names]
    for sense, cells in p["inputs"].items():
        for cell in cells:
            if cell not in lesion:
                cell_senses[index[cell]].append(sense)
    return Wiring(names, order, pos, state_partners, prev_partners,
                  [tuple(s) for s in cell_senses], list(p["inputs"]), list(p["outputs"]))
