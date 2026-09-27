"""The brain atlas: one integrated layout of a whole connectome for the standard brain view.

A page should show the entire brain, every neuron a point and every synapse a line, laid
out so that connected neurons sit near each other and regions keep their colour, with the
activity and the change of every settling step animated live. The atlas is the layout half
of that standard: regions are placed by a force layout of the region graph (connected
regions attract), neurons inside a region by their synapses (iterated neighbour averaging,
a spectral-style embedding), by a declared sheet shape (a grid, for a retina), or by
supplied coordinates. The atlas exports positions, region roles and colours, every synapse
and quantised settling frames as one compact payload that the shipped ``brain_scan.js``
renders in any browser page. Nothing here changes a brain; it only draws one.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from cadence.brain import Brain
    from cadence.connectome import Connectome
    from cadence.recording import SettlementRecord

__all__ = ["Atlas", "Region", "build_atlas", "role_of", "PALETTE", "brain_scan_script"]

# Region roles and their colours (RGB, 0 to 255): the legend every page shares.
PALETTE: dict[str, tuple[int, int, int]] = {
    "vision": (89, 183, 255),
    "sensory": (143, 225, 157),
    "memory": (171, 153, 255),
    "association": (89, 229, 203),
    "motor": (255, 191, 112),
    "value": (237, 129, 182),
    "other": (143, 163, 184),
}

_ROLES: tuple[tuple[str, str], ...] = (
    (
        "memory",
        r"afterglow|afterimage|context|trace|record|recall|notebook|route|memory|echo|prefrontal|hippocamp",
    ),
    ("value", r"value|reward|critic|dopamine|valence|monitor|salience"),
    ("motor", r"motor|action|actuator|joint|pencil|pen\b|body|efferen|output|slot|cord|muscle"),
    ("vision", r"retina|visual|vision|eye|fovea|periph|pixel|sheet|v1\b|gaze"),
    (
        "sensory",
        r"sensor|sense|input|cue|key|smell|odou?r|touch|whisker|auditory|ear\b|heard|nose|taste|vestib|place",
    ),
    (
        "association",
        r"associat|hidden|cortex|assoc|belt|phrase|harmon|rhythm|melody|timbre|intention|interneuron",
    ),
)


def role_of(name: str, roles: Mapping[str, str] | None = None) -> str:
    """The role a region name declares, by an explicit map or by its wording."""
    if roles and name in roles:
        return roles[name]
    key = name.lower()
    for role, pattern in _ROLES:
        if re.search(pattern, key):
            return role
    return "other"


def _b64(array: np.ndarray) -> dict[str, Any]:
    data = np.ascontiguousarray(array)
    return {
        "dtype": data.dtype.str.lstrip("<>|="),
        "shape": list(data.shape),
        "b64": base64.b64encode(data.tobytes()).decode("ascii"),
    }


@dataclass
class Region:
    """A named group of neurons with a role, a colour, a place and a size on the atlas."""

    name: str
    role: str
    indices: np.ndarray
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.1
    shape: tuple[int, ...] | None = None
    extent: tuple[float, float] = (0.1, 0.1)

    @property
    def color(self) -> tuple[int, int, int]:
        return PALETTE.get(self.role, PALETTE["other"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "color": list(self.color),
            "count": int(len(self.indices)),
            "center": [float(self.center[0]), float(self.center[1])],
            "radius": float(self.radius),
            "extent": [float(self.extent[0]), float(self.extent[1])],
            "shape": None if self.shape is None else list(self.shape),
        }


@dataclass
class Atlas:
    """Positions in the unit square (both axes in [-1, 1]), regions, and every synapse."""

    n: int
    positions: np.ndarray
    region_index: np.ndarray
    regions: list[Region]
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    seed: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def synapses(self) -> int:
        return int(len(self.pre))

    def region_of(self, neuron: int) -> Region:
        return self.regions[int(self.region_index[neuron])]

    def subsample_edges(self, limit: int, seed: int | None = None) -> Atlas:
        """The same atlas with at most ``limit`` synapses, the strongest kept preferentially."""
        if self.synapses <= limit:
            return self
        rng = np.random.default_rng(self.seed if seed is None else seed)
        weight = np.abs(self.weight) + 1e-12
        keep = np.sort(
            rng.choice(self.synapses, size=limit, replace=False, p=weight / weight.sum())
        )
        return Atlas(
            self.n,
            self.positions,
            self.region_index,
            self.regions,
            self.pre[keep],
            self.post[keep],
            self.weight[keep],
            self.seed,
            {**self.extras, "synapses_total": self.synapses},
        )

    def frames(self, activation: np.ndarray, potential: np.ndarray | None = None) -> dict[str, Any]:
        """Quantise recorded settling steps ``(steps, n)`` for replay: activation in [-1, 1]
        to eight bits per neuron and step; potentials likewise over their own range."""
        act = np.asarray(activation, dtype=float)
        if act.ndim != 2 or act.shape[1] != self.n:
            raise ValueError(f"activation must be (steps, {self.n})")
        out: dict[str, Any] = {
            "steps": int(act.shape[0]),
            "n": self.n,
            "activation": _b64(
                np.rint(np.clip((act + 1.0) / 2.0, 0.0, 1.0) * 255).astype(np.uint8)
            ),
        }
        if potential is not None:
            pot = np.asarray(potential, dtype=float)
            if pot.shape != act.shape:
                raise ValueError("potential must match the activation's shape")
            span = float(max(np.max(np.abs(pot)), 1e-12))
            out["potential"] = _b64(
                np.rint(np.clip((pot / span + 1.0) / 2.0, 0.0, 1.0) * 255).astype(np.uint8)
            )
            out["potential_span"] = span
        return out

    def frames_from_record(self, record: SettlementRecord, row: int = 0) -> dict[str, Any]:
        """The frames of one batch row of a settlement recording."""
        return self.frames(record.activation[:, row, :], record.potential[:, row, :])

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "cadence.atlas/v1",
            "n": self.n,
            "synapses": self.synapses,
            "seed": self.seed,
            "regions": [r.to_dict() for r in self.regions],
            "region": _b64(self.region_index.astype(np.uint16)),
            "positions": _b64(self.positions.astype(np.float32)),
            "pre": _b64(self.pre.astype(np.uint32)),
            "post": _b64(self.post.astype(np.uint32)),
            "weight": _b64(self.weight.astype(np.float32)),
            "palette": {k: list(v) for k, v in PALETTE.items()},
            **({"extras": self.extras} if self.extras else {}),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def summary(self) -> dict[str, Any]:
        return {
            "neurons": self.n,
            "synapses": self.synapses,
            "regions": [(r.name, r.role, int(len(r.indices))) for r in self.regions],
        }


def brain_scan_script() -> str:
    """The standard renderer, ``brain_scan.js``, shipped with the library, as source text."""
    return resources.files("viewer").joinpath("brain_scan.js").read_text(encoding="utf-8")


def _partition(
    n: int,
    populations: Mapping[str, Sequence[int]] | None,
    regions: Mapping[str, Sequence[int]] | None,
) -> list[tuple[str, np.ndarray]]:
    """Every neuron in exactly one region: an explicit map, else the coarsest populations."""
    source = regions if regions is not None else (populations or {})
    names = sorted(source.keys(), key=lambda k: (k.count("/"), -len(source[k]), k))
    owner = np.full(n, -1, dtype=np.int64)
    out: list[tuple[str, np.ndarray]] = []
    for name in names:
        members = np.asarray(list(source[name]), dtype=np.int64)
        members = members[(members >= 0) & (members < n)]
        fresh = members[owner[members] < 0]
        if not len(fresh):
            continue
        owner[fresh] = len(out)
        out.append((name, np.sort(fresh)))
    rest = np.flatnonzero(owner < 0)
    if len(rest):
        out.append(("other", rest))
    return out


def _region_layout(
    regions: list[Region],
    mass: np.ndarray,
    n: int,
    rng: np.random.Generator,
    given: Mapping[str, Any] | None = None,
) -> None:
    """Place region centres in the unit square by a force layout of the region graph."""
    k = len(regions)
    sizes = np.array([len(r.indices) for r in regions], dtype=float)
    radius = np.clip(0.58 * np.sqrt(sizes / max(n, 1)), 0.05, 0.5)
    for r, rad in zip(regions, radius, strict=True):
        aspect = 1.0
        if r.shape is not None and len(r.shape) >= 2:
            h, w = float(r.shape[0]), float(r.shape[1])
            aspect = float(np.sqrt(max(w, 1.0) / max(h, 1.0)))
        elif given is not None and r.name in given and len(given[r.name]) > 1:
            pts = np.asarray(given[r.name], dtype=float)
            span = np.maximum(pts.max(axis=0) - pts.min(axis=0), 1e-6)
            aspect = float(np.sqrt(span[0] / span[1]))
        aspect = min(3.0, max(1 / 3, aspect))
        r.extent = (float(rad * aspect), float(rad / aspect))
        r.radius = float(np.hypot(*r.extent))
    bound = np.array([r.radius for r in regions])
    if k == 1:
        regions[0].center = (0.0, 0.0)
        return
    # order around a circle so that strongly connected regions start adjacent
    order = [int(np.argmax(sizes))]
    while len(order) < k:
        remaining = [i for i in range(k) if i not in order]
        pull = [mass[i, order].sum() for i in remaining]
        order.append(remaining[int(np.argmax(pull))])
    centre = np.zeros((k, 2))
    for rank, i in enumerate(order):
        angle = 2 * np.pi * rank / k
        centre[i] = 0.6 * np.array([np.cos(angle), np.sin(angle)])
    strength = mass / max(mass.max(), 1e-12)
    velocity = np.zeros((k, 2))
    for _ in range(400):
        force = -0.03 * centre
        for i in range(k):
            for j in range(i + 1, k):
                d = centre[j] - centre[i]
                dist = float(np.hypot(*d)) + 1e-9
                u = d / dist
                want = bound[i] + bound[j] + 0.10
                if dist < want:
                    push = (want - dist) * 1.5
                    force[i] -= push * u
                    force[j] += push * u
                elif strength[i, j] > 0:
                    pull = (dist - want) * (0.15 + 0.85 * strength[i, j])
                    force[i] += pull * u
                    force[j] -= pull * u
                else:
                    force[i] -= 0.002 / dist * u
                    force[j] += 0.002 / dist * u
        velocity = 0.6 * velocity + 0.08 * force
        centre += velocity
    lo = (centre - bound[:, None]).min(axis=0)
    hi = (centre + bound[:, None]).max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scale = float(min(1.84 / span[0], 1.84 / span[1]))
    centre = (centre - (lo + hi) / 2) * scale
    for r, c, rad in zip(regions, centre, bound, strict=True):
        r.center = (float(c[0]), float(c[1]))
        r.radius = float(rad * scale)
        r.extent = (float(r.extent[0] * scale), float(r.extent[1] * scale))
    _ = rng


def _place_grid(region: Region) -> np.ndarray:
    shape = region.shape or (1, 1)
    h, w = int(shape[0]), int(shape[1])
    channels = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    count = len(region.indices)
    cells = h * w * channels
    if cells != count:
        # fall back to a square-ish grid over the count
        w = int(np.ceil(np.sqrt(count)))
        h = int(np.ceil(count / w))
        channels = 1
    idx = np.arange(count)
    channel = idx % channels
    cell = idx // channels
    row = cell // w
    col = cell % w
    ax, ay = region.extent
    x = (
        region.center[0]
        + ((col + 0.5) / w * 2 - 1) * ax
        + (channel - (channels - 1) / 2) * (0.4 * ax / w)
    )
    y = region.center[1] + (1 - (row + 0.5) / h * 2) * ay
    return np.asarray(np.stack([x, y], axis=1))


def _fit_into(
    points: np.ndarray, region: Region, rng: np.random.Generator, *, spread: bool = False
) -> np.ndarray:
    """Re-centre and scale points into the region's extent, clamping to the ellipse.

    With ``spread`` the cloud is whitened (its two principal axes scaled to equal variance)
    and its radii rank-transformed to a uniform disc: iterated neighbour averaging is a
    rank-one operation that folds a region into a line, and whitening at every iteration
    turns it into a two-dimensional spectral embedding that fills the region evenly.
    """
    if not len(points):
        return points
    extent = np.array(region.extent)
    q = points - points.mean(axis=0)
    if spread and len(points) >= 3:
        q = q + rng.normal(scale=1e-4, size=q.shape) * extent
        cov = q.T @ q / len(points)
        vals, vecs = np.linalg.eigh(cov)
        q = (q @ vecs) / np.sqrt(np.maximum(vals, 1e-18))
        radius = np.hypot(q[:, 0], q[:, 1])
        angle = np.arctan2(q[:, 1], q[:, 0])
        rank = np.argsort(np.argsort(radius, kind="stable"), kind="stable")
        rad = np.sqrt((rank + 0.5) / len(points)) * 0.94
        q = np.stack([rad * np.cos(angle), rad * np.sin(angle)], axis=1) * extent
    else:
        # supplied coordinates keep their shape: one uniform scale fits them into the box
        width = max(float(np.abs(q[:, 0]).max()), 1e-9)
        height = max(float(np.abs(q[:, 1]).max()), 1e-9)
        q = q * (min(extent[0] / width, extent[1] / height) * 0.96)
    q += rng.normal(scale=0.004, size=q.shape) * extent
    return np.asarray(q + np.array(region.center))


def build_atlas(
    connectome: Connectome,
    weights: np.ndarray | None = None,
    *,
    regions: Mapping[str, Sequence[int]] | None = None,
    shapes: Mapping[str, Sequence[int]] | None = None,
    positions: Mapping[str, np.ndarray] | None = None,
    roles: Mapping[str, str] | None = None,
    seed: int = 0,
    iterations: int = 24,
) -> Atlas:
    """Lay out a whole connectome as one atlas.

    ``weights`` are the effective synaptic weights in connectome order (``brain.weights``);
    without them the contact counts stand in. ``regions`` partitions the neurons by name
    (default: the connectome's populations, coarsest first); ``shapes`` declares sheets as
    ``(rows, cols[, channels])`` per region name and places them on a grid; ``positions``
    supplies ``(count, 2)`` coordinates per region name, or under the key ``"*"`` one shared
    ``(n, 2)`` frame for every neuron (an anatomy, kept as given); ``roles`` overrides the
    role a region's name would suggest. Everything else is placed by the synapses.
    """
    n = int(connectome.n)
    pre = np.asarray(connectome.pre, dtype=np.int64)
    post = np.asarray(connectome.post, dtype=np.int64)
    weight = (
        np.asarray(connectome.count, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if weight.shape != pre.shape:
        raise ValueError("weights must have one entry per synapse")
    rng = np.random.default_rng(seed)
    parts = _partition(n, getattr(connectome, "populations", None), regions)
    region_list = [
        Region(
            name,
            role_of(name, roles),
            members,
            shape=None if not shapes or name not in shapes else tuple(int(x) for x in shapes[name]),
        )
        for name, members in parts
    ]
    region_index = np.zeros(n, dtype=np.int64)
    for i, r in enumerate(region_list):
        region_index[r.indices] = i
    k = len(region_list)
    mass = np.zeros((k, k))
    strength = np.abs(weight)
    np.add.at(mass, (region_index[pre], region_index[post]), strength)
    mass = mass + mass.T
    np.fill_diagonal(mass, 0.0)
    if positions and "*" in positions:
        return _anatomical(n, region_list, region_index, pre, post, weight, positions["*"], seed)
    _region_layout(region_list, mass, n, rng, positions)

    pos = np.zeros((n, 2))
    free = np.zeros(n, dtype=bool)
    for r in region_list:
        if r.shape is not None:
            pos[r.indices] = _place_grid(r)
        elif positions and r.name in positions:
            given = np.asarray(positions[r.name], dtype=float)
            if given.shape != (len(r.indices), 2):
                raise ValueError(f"positions for {r.name} must be ({len(r.indices)}, 2)")
            pos[r.indices] = _fit_into(given, r, rng)
        else:
            count = len(r.indices)
            rad = np.sqrt(rng.uniform(0.0, 1.0, count))
            ang = rng.uniform(0.0, 2 * np.pi, count)
            pos[r.indices] = (
                np.array(r.center)
                + np.stack(
                    [rad * np.cos(ang) * r.extent[0], rad * np.sin(ang) * r.extent[1]], axis=1
                )
                * 0.9
            )
            free[r.indices] = True
    if free.any() and len(pre):
        den = np.bincount(post, weights=strength, minlength=n) + np.bincount(
            pre, weights=strength, minlength=n
        )
        has = den > 0
        for step in range(iterations):
            target = np.empty_like(pos)
            for axis in range(2):
                num = np.bincount(post, weights=strength * pos[pre, axis], minlength=n)
                num += np.bincount(pre, weights=strength * pos[post, axis], minlength=n)
                target[:, axis] = np.where(has, num / np.where(has, den, 1.0), pos[:, axis])
            mixed = 0.45 * pos + 0.55 * target
            for r in region_list:
                members = r.indices
                if not free[members[0]] if len(members) else True:
                    continue
                jitter = np.random.default_rng(seed + 1000 + step)
                pos[members] = _fit_into(mixed[members], r, jitter, spread=True)
    extras: dict[str, Any] = {}
    return Atlas(
        n,
        pos.astype(np.float32),
        region_index.astype(np.int32),
        region_list,
        pre.astype(np.int64),
        post.astype(np.int64),
        weight.astype(np.float32),
        seed,
        extras,
    )


def _anatomical(
    n: int,
    region_list: list[Region],
    region_index: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    given: Any,
    seed: int,
) -> Atlas:
    """One shared frame for every neuron (an anatomy), scaled into the square as given."""
    all_pos = np.asarray(given, dtype=float)
    if all_pos.shape != (n, 2):
        raise ValueError(f"positions['*'] must be ({n}, 2)")
    lo, hi = all_pos.min(axis=0), all_pos.max(axis=0)
    scale = 1.84 / max(1e-9, float((hi - lo).max()))
    pos = (all_pos - (lo + hi) / 2) * scale
    for r in region_list:
        members = pos[r.indices]
        mn, mx = members.min(axis=0), members.max(axis=0)
        r.center = (float((mn[0] + mx[0]) / 2), float((mn[1] + mx[1]) / 2))
        r.extent = (max(0.02, float((mx[0] - mn[0]) / 2)), max(0.02, float((mx[1] - mn[1]) / 2)))
        r.radius = float(np.hypot(*r.extent))
    return Atlas(
        n,
        pos.astype(np.float32),
        region_index.astype(np.int32),
        region_list,
        pre.astype(np.int64),
        post.astype(np.int64),
        weight.astype(np.float32),
        seed,
        {},
    )


def atlas_of(brain: Brain, **options: Any) -> Atlas:
    """The atlas of a brain, laid out by its effective synaptic weights."""
    return build_atlas(brain.connectome, brain.weights, **options)


_LEGEND = "".join(
    f'<span><i style="background:rgb({r},{g},{b})"></i>{role}</span>'
    for role, (r, g, b) in PALETTE.items()
)


def _page_template() -> str:
    return resources.files("viewer").joinpath("brain_scan_page.html").read_text(encoding="utf-8")


def _brain_payload(brain: Brain, inputs: Sequence[int] | None) -> dict[str, Any]:
    from cadence.certificate import certificate

    model = brain.neuron_model
    if model.adaptation is not None:
        raise ValueError("a live page settles without adaptation")
    connectome = brain.connectome
    if inputs is None:
        pops = connectome.populations
        candidates = [name for name in pops if role_of(name) in ("vision", "sensory")]
        inputs = list(pops[candidates[0]]) if candidates else list(range(min(connectome.n, 8)))
    return {
        "n": int(connectome.n),
        "W": _b64(brain.dense().astype(np.float32)),
        "bias": _b64(np.asarray(brain.bias, dtype=np.float32)),
        "neuron": {
            "slope": model.slope,
            "threshold": model.threshold,
            "leak": model.leak,
            "dt": model.dt,
            "rest": model.rest_emission,
            "amplitude": model.stimulus_amplitude,
        },
        "inputs": [int(i) for i in inputs],
        "certificate": certificate(brain).to_dict(),
    }


def _page(
    atlas: Atlas,
    frames: Mapping[str, Any] | None,
    brain: Brain | None,
    title: str,
    note: str,
    inputs: Sequence[int] | None,
    limit: int,
) -> str:
    """A self-contained page: the atlas, the renderer, recorded frames and/or a live brain."""
    if brain is not None and brain.connectome.n > limit:
        raise ValueError(
            f"a live page embeds a dense {brain.connectome.n}x{brain.connectome.n} matrix; "
            "raise limit= to allow it"
        )
    payload = {
        "__TITLE__": title,
        "__LEGEND__": _LEGEND,
        "__NOTE__": note,
        "__SCRIPT__": brain_scan_script().replace("export ", ""),
        "__ATLAS__": atlas.to_json(),
        "__FRAMES__": "null" if frames is None else json.dumps(dict(frames), separators=(",", ":")),
        "__BRAIN__": "null"
        if brain is None
        else json.dumps(_brain_payload(brain, inputs), separators=(",", ":")),
    }
    page = _page_template()
    for key, value in payload.items():
        page = page.replace(key, value)
    return page


def _atlas_page(
    self: Atlas,
    *,
    frames: Mapping[str, Any] | None = None,
    brain: Brain | None = None,
    title: str = "Cadence brain scan",
    note: str = (
        "Every neuron is a point coloured by its region, every synapse a line. Brightness is "
        "activation, the hot glow is change, and the particles are messages travelling along "
        "synapses. Detune changes the stimulus and the brain settles into a new equilibrium; "
        "the traces below follow every region."
    ),
    inputs: Sequence[int] | None = None,
    limit: int = 2000,
) -> str:
    return _page(self, frames, brain, title, note, inputs, limit)


Atlas.page = _atlas_page  # type: ignore[attr-defined]
