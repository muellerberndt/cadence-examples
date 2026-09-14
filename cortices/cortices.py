"""A catalogue of basic cortices for cadence brains.

Every builder returns a ``Cortex``: a cadence ``Region`` that also says which of its
populations are clamped (driven from outside the brain) and, for free neurons, the
resting bias it wants. ``assemble_brain`` develops a list of them into one connectome
and returns the mask of plastic neurons and the bias vector that the cortices ask for.

Only the public cadence API is used. A designed catalogue is one way to start a brain;
the core principles are local repair, checked convergence and equilibrium detuning, and
minimal design plus plasticity is the research direction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import cadence as cd
import numpy as np
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region, cortex, motor_cortex, prefrontal_cortex, visual_cortex

__all__ = [
    "Assembled",
    "ClockedRecord",
    "Cortex",
    "WorkingMemory",
    "assemble_brain",
    "association",
    "clamped_populations",
    "clocked_record",
    "condition",
    "conditioning",
    "motor",
    "sensory_sheet",
    "stimulus",
    "tonic_bias",
    "visual_sheet",
    "working_memory",
]

WHOLE = "*"  # in ``Cortex.clamped``: every neuron of the region is clamped
TONIC = 0.5  # the default resting bias of free neurons (see ``association``)


@dataclass(frozen=True)
class Cortex(Region):
    """A region that also records which populations are clamped and its tonic level.

    ``clamped`` lists population names whose activity is set by a stimulus from outside
    the brain (a sense, a trace, a record); ``WHOLE`` stands for the entire region. Those
    neurons get no resting bias and no bias learning. ``tonic`` is the resting bias of the
    region's free neurons; ``None`` takes the assembly default.
    """

    clamped: tuple[str, ...] = ()
    tonic: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for population in self.clamped:
            if population != WHOLE and (
                self.circuit is None or population not in self.circuit.populations
            ):
                raise ValueError(f"clamped population {population!r} is not in {self.name!r}")
        if self.tonic is not None and not np.isfinite(self.tonic):
            raise ValueError("tonic must be finite")

    def free_populations(self) -> tuple[str, ...]:
        """Connectome population names of this region's free neurons: its own name when
        it is blank or has no named populations, else each population not clamped."""
        if WHOLE in self.clamped:
            return ()
        if self.circuit is None or not self.circuit.populations:
            return (self.name,)
        return tuple(
            f"{self.name}/{p}" for p in self.circuit.populations if p not in self.clamped
        )


def _positive(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


# -- senses


def sensory_sheet(n: int, *, name: str = "sensory") -> Cortex:
    """A clamped input population, one neuron per feature.

    The environment writes a feature vector into it as stimulus. It has no synapses of its
    own; project from it with ``reciprocal=False`` so nothing writes back into a sense.
    """
    return Cortex(name, _positive(n, "n"), clamped=(WHOLE,))


def visual_sheet(
    height: int, width: int, *, features: int = 8, field: int = 3, seed: int = 0, name: str = "visual"
) -> Cortex:
    """A retina and feature maps with local, shared receptive fields (``visual_cortex``).

    Population ``input`` holds one neuron per pixel, row by row, and is clamped to the
    picture. Population ``output`` holds ``features`` maps; each neuron sees one ``field``
    by ``field`` window. The receptive-field synapses start random and learn.
    """
    region = visual_cortex(height, width, features=features, field=field, seed=seed, name=name)
    return Cortex(name, circuit=region.circuit, inputs="input", outputs="output", clamped=("input",))


# -- association


def association(n: int, *, tonic: float = TONIC, lateral: float = 0.0, name: str = "association") -> Cortex:
    """A blank cortex of ``n`` neurons with a resting bias of ``tonic`` on every neuron.

    The learning neuron model is a rectified sigmoid that is exactly zero at rest. With
    symmetric random projections about half of a stage receives negative net input, sits at
    rest and carries no contrast, so the learning rule cannot move it. A resting bias of
    about half a unit on the free neurons puts nearly every neuron on the slope of the
    sigmoid. In this catalogue's own check, 0.5 put 96 to 99 percent of an association
    cortex on the slope and gave the best held-out accuracy; 1.0 put every neuron on the
    slope but lost accuracy on a wider cortex. The learner moves the bias from there.

    A negative ``lateral`` adds mutual inhibition among the neurons (``cortex``).
    """
    region = cortex(n, lateral=lateral, name=name)
    return Cortex(name, region.size, circuit=region.circuit, tonic=float(tonic))


# -- motor


def motor(actions: int, *, lateral: float = -0.5, name: str = "motor") -> Cortex:
    """One neuron per action with lateral inhibition (``motor_cortex``), population ``actions``.

    Read the choice as the most active neuron (``Learner.predict``) or as a softmax; with
    several independent choices per moment pass ``slots=`` to the learner and lay the
    actions out slot by slot.
    """
    region = motor_cortex(actions, lateral=lateral, name=name)
    return Cortex(name, circuit=region.circuit, inputs="actions", outputs="actions")


# -- working memory


@dataclass(frozen=True)
class WorkingMemory:
    """A prefrontal population sized to the region it holds, and the trace that feeds it.

    ``region`` is clamped: a ``Trace`` of the held region's activity is written into it as
    stimulus before every settling run. ``projection()`` is the synapse bundle from it back
    into the held region; ``trace(connectome)`` is the trace, built once the connectome
    exists. The trace fades over about ``1 / (1 - decay)`` moments: 0.5 keeps the last two,
    0.85 the last several, 0.98 about the last fifty. ``amplitude`` scales the trace as
    stimulus; a settled cortex is small, so it is set high.
    """

    region: Cortex
    held: str
    decay: float = 0.85
    amplitude: float = 8.0
    scale: float = 12.0

    def projection(self) -> Projection:
        return Projection(self.region.name, self.held, scale=self.scale, reciprocal=False)

    def trace(self, connectome: cd.Connectome) -> cd.Trace:
        return cd.Trace(
            connectome,
            decay=self.decay,
            amplitude=self.amplitude,
            source=self.held,
            target=self.region.name,
        )


def working_memory(
    held: Region, *, decay: float = 0.85, amplitude: float = 8.0, scale: float = 12.0, name: str = "prefrontal"
) -> WorkingMemory:
    """Working memory for ``held``: see ``WorkingMemory``. Add ``.region`` to the regions
    and ``.projection()`` to the projections, then ``.trace(connectome)`` after assembly."""
    if not 0 <= decay < 1:
        raise ValueError("decay lies in [0, 1)")
    region = prefrontal_cortex(held, name=name)
    return WorkingMemory(
        Cortex(name, region.size, clamped=(WHOLE,), tonic=0.0),
        held.name,
        float(decay),
        float(amplitude),
        float(scale),
    )


# -- records addressed by time


@dataclass(frozen=True)
class ClockedRecord:
    """A delta-rule record addressed by a clock or position key.

    ``region`` has two clamped populations: ``clock``, one neuron per beat or position,
    which the environment or a rhythm drives one-hot, and ``recall``, one neuron per value
    feature, into which ``store(connectome).stimulate(drive)`` writes the record's read.
    Read the beat before writing the new item there: the read is then the item from one
    period earlier. ``beats(t)`` is the one-hot key of moment ``t``.
    """

    region: Cortex
    keys: int
    values: int
    decay: float = 1.0

    def beats(self, t: int | Sequence[int] | np.ndarray) -> np.ndarray:
        """One-hot clock keys, ``(batch, keys)``, for moments ``t`` (an int or a vector)."""
        return np.eye(self.keys)[np.atleast_1d(np.asarray(t, dtype=np.int64)) % self.keys]

    def store(self, connectome: cd.Connectome | None = None) -> cd.FastSynapses:
        """The record. Without a connectome its ports are region-local (keys then values);
        with one they are the ``clock`` and ``recall`` populations of the whole brain."""
        if connectome is None:
            pre = np.arange(self.keys)
            post = np.arange(self.keys, self.keys + self.values)
        else:
            pre = np.asarray(connectome.populations[f"{self.region.name}/clock"])
            post = np.asarray(connectome.populations[f"{self.region.name}/recall"])
        return cd.FastSynapses(pre, post, rule="delta", decay=self.decay)


def clocked_record(keys: int, values: int, *, decay: float = 1.0, name: str = "record") -> ClockedRecord:
    """A record of ``values`` features addressed by ``keys`` beats or positions.

    ``decay`` below one gives a graded loss with age. Nothing projects into this region;
    project from its ``recall`` population into a free cortex.
    """
    keys, values = _positive(keys, "keys"), _positive(values, "values")
    circuit = cd.Connectome.from_synapses(
        keys + values,
        pre=[],
        post=[],
        populations={"clock": range(keys), "recall": range(keys, keys + values)},
        label=f"clocked-record:{keys}x{values}",
    )
    region = Cortex(name, circuit=circuit, outputs="recall", clamped=("clock", "recall"))
    return ClockedRecord(region, keys, values, float(decay))


# -- conditioning


def conditioning(classes: int, *, name: str = "context") -> Cortex:
    """A clamped population of one-hot context classes: a mode, a mood, a task.

    The environment sets the class; ``condition`` projects it into every free region so
    the same brain settles differently under each class. This is supplied conditioning,
    not understanding: the brain is told which situation it is in.
    """
    return Cortex(name, _positive(classes, "classes"), clamped=(WHOLE,))


def condition(context: Region, regions: Sequence[Region], *, scale: float = 1.0) -> tuple[Projection, ...]:
    """One-way projections from ``context`` into every free population of ``regions``."""
    out = []
    for region in regions:
        if region.name == context.name:
            continue
        targets = region.free_populations() if isinstance(region, Cortex) else (region.name,)
        out.extend(Projection(context.name, t, scale=scale, reciprocal=False) for t in targets)
    return tuple(out)


# -- assembly


def clamped_populations(regions: Sequence[Region]) -> tuple[str, ...]:
    """Connectome population names that the cortices in ``regions`` declare clamped."""
    out = []
    for region in regions:
        if not isinstance(region, Cortex):
            continue
        for population in region.clamped:
            out.append(region.name if population == WHOLE else f"{region.name}/{population}")
    return tuple(out)


def tonic_bias(connectome: cd.Connectome, populations: Sequence[str], level: float = TONIC) -> np.ndarray:
    """A bias vector at ``level`` on every neuron and zero on the listed clamped populations."""
    if not np.isfinite(level):
        raise ValueError("level must be finite")
    bias = np.full(connectome.n, float(level))
    for name in populations:
        bias[list(connectome.populations[name])] = 0.0
    return bias


class Assembled(NamedTuple):
    connectome: cd.Connectome
    plastic_neurons: np.ndarray  # bool per neuron: False on clamped populations
    bias: np.ndarray  # the tonic bias: zero on clamped populations


def assemble_brain(
    regions: Sequence[Region],
    projections: Sequence[Projection],
    *,
    seed: int = 0,
    tonic: float = TONIC,
    label: str = "catalogue",
) -> Assembled:
    """Develop the regions and projections into one connectome (``Genome`` and ``develop``).

    Returns the connectome, the mask of plastic neurons (False on every clamped population,
    so the learner leaves their bias alone) and the bias vector: ``tonic`` on free neurons,
    a cortex's own ``tonic`` where it set one, zero on clamped populations. Pass the mask
    as ``plastic_neurons=`` to the learner and the bias as ``bias=`` to the brain.
    """
    genome = Genome(tuple(regions), tuple(projections), label=label)
    connectome = develop(genome, seed=seed)
    clamped = clamped_populations(regions)
    bias = tonic_bias(connectome, (), tonic)
    for region in regions:
        if isinstance(region, Cortex) and region.tonic is not None:
            bias[list(connectome.populations[region.name])] = region.tonic
    plastic = np.ones(connectome.n, dtype=bool)
    for name in clamped:
        neurons = list(connectome.populations[name])
        bias[neurons] = 0.0
        plastic[neurons] = False
    return Assembled(connectome, plastic, bias)


def stimulus(connectome: cd.Connectome, values: Mapping[str, np.ndarray | Sequence[float]], batch: int | None = None) -> np.ndarray:
    """A ``(batch, n)`` drive with ``values`` written into the named populations.

    Each value is ``(batch, size)`` or ``(size,)``; the batch is taken from the first
    two-dimensional value, or from ``batch``, or is one.
    """
    arrays = {k: np.asarray(v, dtype=float) for k, v in values.items()}
    if batch is None:
        batch = next((len(a) for a in arrays.values() if a.ndim == 2), 1)
    drive = np.zeros((batch, connectome.n))
    for name, value in arrays.items():
        drive[:, list(connectome.populations[name])] = value
    return drive
