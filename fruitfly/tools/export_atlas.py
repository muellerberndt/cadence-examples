#!/usr/bin/env python3
"""Export the BANC connectome as a cadence atlas payload for the whole-brain viewer.

Writes ``web/data/atlas.json`` in the ``cadence.atlas/v1`` format the viewer's
``decodeAtlas`` reads, with the measured soma positions as ``positions3`` (three floats per
neuron in the brain style's frame: centred, the longest extent spanning the unit ball) and
``spacing3`` (the distance to a neuron's neighbours as seen from the front, from the local
density of the projection). The scan
style's two-dimensional positions are the frontal projection scaled into its square.

The frame is the frontal view: the fly's left on the viewer's right, the brain above the
nerve cord, the anterior face toward the camera; in BANC's own micrometre frame that is
``(x, -y, -z)``, a proper rotation.

Regions, one per neuron: the intrinsic neurons by anatomical region and side, and the
functional populations pulled out as their own regions (sensory afferents by body part,
descending and ascending neurons, motor neurons by effector). Neurons the release leaves
without a root position are placed at the synapse-count-weighted centroid of their
positioned partners, or, without one, at the centroid of their region's positioned members,
each with a small seeded jitter; the payload's ``note`` records the counts.

Synapse classes: the strongest ``EDGE_BUDGET`` by synapse count (the viewer's line budget),
weight ``sign * log1p(count)``.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fruitfly.banc import MANIFEST_PATH, load_fixture  # noqa: E402

OUT_PATH = ROOT / "web" / "data" / "atlas.json"
EDGE_BUDGET = 400_000
SEED = 7
GRID = 64  # the spacing grid, as in the viewer's spacing3Of
JITTER_UM = 8.0  # the jitter around a partner centroid, in micrometres
ANATOMY_RADIUS = 1.0  # the longest half-extent in the viewer's frame
SCAN_SPAN = 1.84  # the scan style's square

# the viewer's PALETTE by role
PALETTE: dict[str, tuple[int, int, int]] = {
    "vision": (89, 183, 255), "sensory": (143, 225, 157), "memory": (171, 153, 255),
    "association": (89, 229, 203), "motor": (255, 191, 112), "value": (237, 129, 182),
    "other": (143, 163, 184),
}
REGION_LABEL = {"optic_lobe": "optic lobe", "central_brain": "central brain", "ventral_nerve_cord": "nerve cord"}
INTRINSIC = [f"{REGION_LABEL[r]}, {s}" for r in ("optic_lobe", "central_brain", "ventral_nerve_cord") for s in ("left", "right")]
FUNCTIONAL = [
    ("antenna", "sensory"), ("eye (retina, lamina)", "vision"), ("ocelli", "sensory"), ("halteres", "sensory"),
    ("wings", "sensory"), ("legs", "sensory"), ("other sensory", "sensory"),
    ("descending neurons", "other"), ("ascending neurons", "other"),
    ("wing and haltere motor neurons", "motor"), ("leg motor neurons", "motor"), ("other motor", "motor"),
]
ROLE_OF = {name: "association" for name in INTRINSIC} | dict(FUNCTIONAL)
ORDER = INTRINSIC + [name for name, _ in FUNCTIONAL]


def _b64(array: np.ndarray) -> dict[str, Any]:
    data = np.ascontiguousarray(array)
    return {"dtype": data.dtype.str.lstrip("<>|="), "shape": list(data.shape), "b64": base64.b64encode(data.tobytes()).decode("ascii")}


def _has(column: np.ndarray, needle: str) -> np.ndarray:
    return np.char.find(column.astype(str), needle) >= 0


def functional_groups(fields: dict[str, np.ndarray]) -> np.ndarray:
    """The functional population of every neuron, or '' for an intrinsic neuron."""
    sc, bps, eff = fields["super_class"].astype(str), fields["body_part_sensory"].astype(str), fields["body_part_effector"].astype(str)
    n = len(sc)
    out = np.full(n, "", dtype=object)
    sensory = np.char.startswith(sc, "sensory") | (bps != "")
    rules = [
        ("antenna", _has(bps, "antenna")),
        ("eye (retina, lamina)", np.isin(bps, ["retina", "lamina", "eye"])),
        ("ocelli", _has(bps, "ocell")),
        ("halteres", _has(bps, "haltere")),
        ("wings", _has(bps, "wing")),
        ("legs", _has(bps, "leg")),
    ]
    for name, mask in rules:
        pick = sensory & mask & (out == "")
        out[pick] = name
    out[sensory & (out == "")] = "other sensory"
    free = out == ""
    out[free & (sc == "descending")] = "descending neurons"
    out[free & np.isin(sc, ["ascending", "ascending_visceral_circulatory"])] = "ascending neurons"
    motor = free & (sc == "motor")
    out[motor & np.isin(eff, ["wing", "haltere"])] = "wing and haltere motor neurons"
    out[motor & _has(eff, "leg") & (out == "")] = "leg motor neurons"
    out[(motor | (free & (sc == "visceral_circulatory"))) & (out == "")] = "other motor"
    return out


def fill_positions(pos: np.ndarray, pre: np.ndarray, post: np.ndarray, count: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Positions for the neurons without one: the count-weighted centroid of their positioned
    synaptic partners (two passes, so a neuron whose partners were themselves placed in the
    first pass is placed too) with a small jitter. Returns the positions and the mask of the
    neurons placed this way."""
    pos = pos.copy()
    n = len(pos)
    placed = np.zeros(n, dtype=bool)
    for _ in range(2):
        missing = np.isnan(pos).any(axis=1)
        have = ~missing
        acc = np.zeros((n, 3)); wsum = np.zeros(n)
        for a, b in ((pre, post), (post, pre)):
            m = missing[a] & have[b]
            np.add.at(acc, a[m], pos[b[m]] * count[m, None])
            np.add.at(wsum, a[m], count[m])
        done = missing & (wsum > 0)
        if not done.any():
            break
        pos[done] = acc[done] / wsum[done, None] + rng.normal(0.0, JITTER_UM, size=(int(done.sum()), 3))
        placed |= done
    return pos, placed


def spacing_of(positions: np.ndarray, grid: int = GRID) -> np.ndarray:
    """The distance to a neuron's neighbours as seen from the front: the local density of
    the (x, y) projection on a grid, the cells of a 3 x 3 neighbourhood sharing their area
    among the neurons they hold (the viewer's spacing3Of). What the additive somata sum to
    on the screen is the column density along the view, so the projected spacing is the one
    the viewer's point alphas need."""
    xy = positions[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    cell = max(1e-6, float((hi - lo).max())) / grid
    dims = np.maximum(1, np.ceil((hi - lo) / cell).astype(int))
    at = np.minimum(dims - 1, np.floor((xy - lo) / cell).astype(int))
    counts = np.zeros(tuple(dims), dtype=np.float64)
    np.add.at(counts, (at[:, 0], at[:, 1]), 1.0)
    padded = np.zeros(tuple(dims + 2)); padded[1:-1, 1:-1] = counts
    inside = np.zeros(tuple(dims + 2)); inside[1:-1, 1:-1] = 1.0
    summed = np.zeros(tuple(dims)); cells = np.zeros(tuple(dims))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            summed += padded[1 + dx:dims[0] + 1 + dx, 1 + dy:dims[1] + 1 + dy]
            cells += inside[1 + dx:dims[0] + 1 + dx, 1 + dy:dims[1] + 1 + dy]
    s = summed[at[:, 0], at[:, 1]]; c = cells[at[:, 0], at[:, 1]]
    return np.sqrt(c * cell ** 2 / np.maximum(1.0, s)).astype(np.float32)


def build(edge_budget: int = EDGE_BUDGET, seed: int = SEED) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t0 = time.time()
    neurons, pre, post, count, sign = load_fixture()
    F = neurons.fields
    n = neurons.n
    rng = np.random.default_rng(seed)
    raw = neurons.position_um.astype(np.float64)
    missing = np.isnan(raw).any(axis=1)

    # 1. positions for the neurons without one: partners first
    pos, by_partners = fill_positions(raw, pre, post, count, rng)
    still = np.isnan(pos).any(axis=1)

    # 2. the side, from the annotation or from x against the midline (the fly's left has the larger x)
    side = F["side"].astype(str).copy()
    labelled = ~missing & np.isin(side, ["left", "right"])
    midline = (np.median(raw[labelled & (side == "left"), 0]) + np.median(raw[labelled & (side == "right"), 0])) / 2
    unsided = ~np.isin(side, ["left", "right"])
    positioned = ~np.isnan(pos).any(axis=1)
    side[unsided & positioned] = np.where(pos[unsided & positioned, 0] > midline, "left", "right")
    parity = unsided & ~positioned
    side[parity] = np.where(np.arange(n)[parity] % 2 == 0, "left", "right")

    # 3. the anatomical region, from the annotation or from the nearest region centroid
    region = F["region"].astype(str).copy()
    known = np.isin(region, list(REGION_LABEL))
    centroids = {r: pos[known & positioned & (region == r)].mean(axis=0) for r in REGION_LABEL}
    unknown = ~known
    for i in np.flatnonzero(unknown & positioned):
        region[i] = min(centroids, key=lambda r: float(np.sum((pos[i] - centroids[r]) ** 2)))
    region[unknown & ~positioned] = "central_brain"

    # 4. the declared grouping: functional populations, then intrinsic neurons by region and side
    group = functional_groups(F)
    intrinsic = group == ""
    group[intrinsic] = np.array([f"{REGION_LABEL[r]}, {s}" for r, s in zip(region[intrinsic], side[intrinsic])], dtype=object)
    names = [name for name in ORDER if (group == name).any()]
    index_of = {name: k for k, name in enumerate(names)}
    region_index = np.array([index_of[g] for g in group], dtype=np.uint16)

    # 5. the rest of the unplaced neurons: their region's centroid, jittered by a quarter of its spread
    by_centroid = still.copy()
    for k, name in enumerate(names):
        members = region_index == k
        gap = members & still
        if not gap.any():
            continue
        have = members & ~still
        if not have.any():
            have = ~still
        centre, spread = pos[have].mean(axis=0), pos[have].std(axis=0)
        pos[gap] = centre + rng.normal(0.0, 1.0, size=(int(gap.sum()), 3)) * spread * 0.25
    assert not np.isnan(pos).any()

    # 6. the viewer's frame: frontal view, centred, the longest extent spanning the unit ball
    view = np.stack([pos[:, 0], -pos[:, 1], -pos[:, 2]], axis=1)
    lo, hi = view.min(axis=0), view.max(axis=0)
    positions3 = ((view - (lo + hi) / 2) * (2 * ANATOMY_RADIUS / float((hi - lo).max()))).astype(np.float32)
    spacing3 = spacing_of(positions3.astype(np.float64))
    lo2, hi2 = positions3[:, :2].min(axis=0), positions3[:, :2].max(axis=0)
    positions2 = ((positions3[:, :2] - (lo2 + hi2) / 2) * (SCAN_SPAN / float((hi2 - lo2).max()))).astype(np.float32)

    # 7. the regions: colour by role, centre and extents from the members' bounding box in the scan's square
    regions = []
    table = []
    for k, name in enumerate(names):
        members = region_index == k
        mn, mx = positions2[members].min(axis=0), positions2[members].max(axis=0)
        extent = np.maximum(0.02, (mx - mn) / 2)
        role = ROLE_OF[name]
        regions.append({
            "name": name, "label": name, "role": role, "color": list(PALETTE[role]), "count": int(members.sum()),
            "center": [float(v) for v in (mn + mx) / 2], "radius": float(np.hypot(*extent)), "extent": [float(v) for v in extent], "shape": None,
        })
        table.append({"region": name, "role": role, "count": int(members.sum()), "positioned": int((members & ~missing).sum()),
                      "placed by partners": int((members & by_partners).sum()), "placed at centroid": int((members & by_centroid).sum())})

    # 8. the strongest synapse classes by count, in a deterministic order
    order = np.lexsort((pre, post, -count))[:edge_budget]
    order = np.sort(order)  # back to (post, pre) order, as the fixture is sorted
    weight = (sign[order] * np.log1p(count[order])).astype(np.float32)

    filled_partners, filled_centroid = int(by_partners.sum()), int(by_centroid.sum())
    manifest = json.loads(MANIFEST_PATH.read_text())
    note = (f"BANC release 888 (adult female Drosophila, brain and nerve cord): {n:,} neurons, the strongest "
            f"{len(order):,} of {len(pre):,} synapse classes by synapse count. {int(missing.sum()):,} neurons have no root position "
            f"in the release: {filled_partners:,} placed at the centroid of their positioned synaptic partners, "
            f"{filled_centroid:,} at their region's centroid, each with a small seeded jitter.")
    payload = {
        "format": "cadence.atlas/v1",
        "n": int(n),
        "synapses": int(len(order)),
        "seed": seed,
        "note": note,
        "regions": regions,
        "region": _b64(region_index),
        "positions": _b64(positions2),
        "positions3": _b64(positions3.reshape(-1)),
        "spacing3": _b64(spacing3),
        "pre": _b64(pre[order].astype(np.uint32)),
        "post": _b64(post[order].astype(np.uint32)),
        "weight": _b64(weight),
        "palette": {k: list(v) for k, v in PALETTE.items()},
        "extras": {
            "source": "BANC v888", "fixture_sha256": manifest["fixture_sha256"], "min_synapses": manifest["min_synapses"],
            "synapses_total": int(len(pre)), "synapses_kept": int(len(order)), "weakest_kept_count": int(count[order].min()),
            "frame": "frontal: viewer (x, y, z) = BANC (x, -y, -z) in micrometres, centred, longest extent 2.0",
            "extent_um": [float(v) for v in (raw[~missing].max(axis=0) - raw[~missing].min(axis=0))],
            "midline_x_um": float(midline),
            "unpositioned": int(missing.sum()), "placed_by_partners": filled_partners, "placed_at_centroid": filled_centroid,
            "unsided": int(unsided.sum()), "unsided_by_position": int((unsided & positioned).sum()), "unsided_by_parity": int(parity.sum()),
            "unregioned": int(unknown.sum()),
            "seconds": round(time.time() - t0, 1),
        },
    }
    return payload, table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--edges", type=int, default=EDGE_BUDGET)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    payload, table = build(args.edges, args.seed)
    text = json.dumps(payload, separators=(",", ":"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    width = max(len(row["region"]) for row in table)
    print(f"{'region':<{width}}  {'role':<11} {'count':>7} {'positioned':>10} {'partners':>8} {'centroid':>8}")
    for row in table:
        print(f"{row['region']:<{width}}  {row['role']:<11} {row['count']:>7,} {row['positioned']:>10,} {row['placed by partners']:>8,} {row['placed at centroid']:>8,}")
    x = payload["extras"]
    print(f"\nneurons {payload['n']:,}  synapse classes kept {x['synapses_kept']:,} of {x['synapses_total']:,} (weakest kept: {x['weakest_kept_count']} synapses)")
    print(f"unpositioned {x['unpositioned']:,}: {x['placed_by_partners']:,} by partners, {x['placed_at_centroid']:,} at a region centroid; "
          f"unsided {x['unsided']:,}: {x['unsided_by_position']:,} by x, {x['unsided_by_parity']:,} by parity; unregioned {x['unregioned']}")
    print(f"extent um {[round(v, 1) for v in x['extent_um']]}  midline x {x['midline_x_um']:.1f} um")
    print(f"wrote {args.out} ({len(text) / 1e6:.1f} MB) in {x['seconds']} s")


if __name__ == "__main__":
    main()
