"""The musician: one brain of wired cortices that hears, holds a phrase in mind,
remembers the form, feels a mood and intends the next event.

Every region below is a contiguous range of neurons in one connectome and settles
in one joint equilibrium. The names describe the intended function of each range,
after the parts of a human composer's brain that carry that function; they are
engineering analogues, not anatomical claims. Supplied parts are labeled: the
event encoder, the mood classes, the beat/bar clock, the trace and the record.

Regions (in neuron order)
    ear         the last WINDOW heard events, one-hot                  (clamped input)
    sense       chroma, sounding notes by family, beat, bar, progress  (clamped input)
    mood        seven mood classes of the piece                        (clamped input)
    interval    relative pitch of the last eight steps (version 2)     (clamped input)
    plan        the profile of the bar being written (version 3)       (clamped input)
    bars        the profiles of the eight bars before (version 3)      (clamped input)
    belt        one shared embedding of each heard event (auditory belt; synapses tied
                across the window positions, so one ear hears every position)
    melody      contour cortex (belt <-> melody)
    harmony     tonal cortex (belt, chroma, sounding notes <-> harmony)
    rhythm      timing cortex (belt, beat, bar, progress <-> rhythm)
    timbre      orchestration cortex (belt, sounding notes by family <-> timbre)
    form        coherence of the piece as a whole (version 2): reads the clock, the mood,
                the slow trace, the record and, in version 3, the plan and the bars before
    phrase      association cortex integrating the four cortices, the working memory,
                the form record and the mood
    prefrontal  working memory: a Trace of the phrase cortex from the events before
    recall      form memory: version 2, what the phrase cortex held at this bar of the
                previous sixteen-bar cycle; version 3, the theme record: what it held at
                the bar this bar returns to, addressed by the plan's lag
    piece       the piece so far: a slow Trace of the phrase cortex (version 2)
    intention   motor intention: pitch, sounding duration, time to next attack,
                instrument family and velocity, read as five softmax choices; version 3
                adds the plan head: the eight profile classes of the bar being written

Version 3 is the form of the piece as a learned quantity. The plan head is trained by
teacher forcing on the measured profile of the bar being written; the plan input is
present for half the training rows and absent for the other half, so the head learns to
predict the bar from the mood, the progress, the bars before and the brain's own state,
and the note intention learns to realise a given plan. When composing, the plan of the
whole piece is imagined first at bar resolution (``composer/perform.py``), then clamped
bar by bar while the notes are written.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cadence as cd
import numpy as np
from cadence.stream import FastSynapses, Trace, columns

from .form import HISTORY, PLAN, PLAN_WIDTHS, RECORD, bar_profiles, encode_bars, encode_plan, history_of, record_keys
from .vocabulary import BAR, CYCLE, DELTAS, DURATIONS, EVENT, FAMILIES, OFFSETS, PROGRAMS, SIZES, WINDOW

__all__ = [
    "BAR", "CYCLE", "DELTAS", "DURATIONS", "EVENT", "FAMILIES", "OFFSETS", "PROGRAMS", "SIZES", "WINDOW",
    "Design", "Musician", "Senses", "StreamState", "build", "connectome", "describe", "encode_events",
    "encode_intervals", "encode_mood", "encode_sense", "events_from_tokens", "family", "learner_config",
    "mood_classes", "primed", "regions", "write_score",
]

ROOT = Path(__file__).resolve().parents[1]

# sense: beat 16, bar-in-cycle 16, progress 4, chroma 12, sounding 8 families x 12 classes
SENSE_RAW = 3 + 12 + 96
SENSE = BAR + CYCLE + 4 + 12 + 96
# interval sense (version 2): the pitch step from each heard event to the one before it,
# clipped to two octaves, for the last INTERVALS events; relative pitch is what a melody is
INTERVALS = 8
INTERVAL_BINS = 49
MOOD_WIDTHS = (2, 3, 3, 4, 3, 3, 3)  # mode, tempo, energy, dynamics, register, texture, tension
MOOD_NAMES = ("mode", "tempo", "energy", "dynamics", "register", "texture", "tension")
MOOD = int(sum(MOOD_WIDTHS))
MOOD_OFFSETS = np.cumsum((0,) + MOOD_WIDTHS[:-1])


def family(program, channel):
    if channel == 9:
        return 7
    if 24 <= program < 32:
        return 5
    if 32 <= program < 40:
        return 4
    if 40 <= program < 56:
        return 1
    if 56 <= program < 64:
        return 2
    if 64 <= program < 80:
        return 3
    return 0 if program < 24 else 6


# --- supplied encoders --------------------------------------------------------------


def encode_events(context):
    """``(batch, WINDOW, 5)`` integer events as ``(batch, WINDOW * EVENT)`` one-hot."""
    context = np.asarray(context, dtype=int)
    x = np.zeros((len(context), WINDOW * EVENT), np.float32)
    rows = np.arange(len(context))
    for h in range(WINDOW):
        for k, offset in enumerate(OFFSETS):
            x[rows, h * EVENT + offset + context[:, h, k]] = 1
    return x


def encode_intervals(context):
    """``(batch, WINDOW, 5)`` events as ``(batch, INTERVALS * INTERVAL_BINS)`` one-hot pitch
    steps between consecutive heard events (percussion steps are clipped like any other)."""
    context = np.asarray(context, dtype=int)
    x = np.zeros((len(context), INTERVALS * INTERVAL_BINS), np.float32)
    rows = np.arange(len(context))
    steps = np.clip(np.diff(context[:, -INTERVALS - 1 :, 0], axis=1), -24, 24) + 24
    for h in range(INTERVALS):
        x[rows, h * INTERVAL_BINS + steps[:, h]] = 1
    return x


def encode_sense(raw):
    """``(batch, SENSE_RAW)`` uint8 rows as ``(batch, SENSE)`` sense activations."""
    raw = np.asarray(raw)
    x = np.zeros((len(raw), SENSE), np.float32)
    rows = np.arange(len(raw))
    x[rows, raw[:, 0].astype(int) % BAR] = 1
    x[rows, BAR + raw[:, 1].astype(int) % CYCLE] = 1
    x[rows, BAR + CYCLE + np.clip(raw[:, 2].astype(int), 0, 3)] = 1
    x[:, BAR + CYCLE + 4 :] = raw[:, 3:] / 255.0
    return x


def encode_mood(classes):
    """``(batch, 7)`` mood classes as ``(batch, MOOD)`` one-hot."""
    classes = np.asarray(classes, dtype=int)
    x = np.zeros((len(classes), MOOD), np.float32)
    for col, (offset, width) in enumerate(zip(MOOD_OFFSETS, MOOD_WIDTHS)):
        x[np.arange(len(classes)), offset + np.clip(classes[:, col], 0, width - 1)] = 1
    return x


def mood_classes(notes, tempo_bpm, key, key_mode):
    """Seven piece-level mood classes from a note list ``(start, duration, pitch, velocity, family)``
    with absolute MIDI pitches and the piece's key.

    These are measured features of the score, not emotions: mode, tempo, onset energy,
    dynamic range, register, texture and chromatic tension.
    """
    notes = np.asarray(notes)
    pitched = notes[notes[:, 4] != 7] if len(notes) else notes
    starts = np.unique(notes[:, 0]) if len(notes) else np.zeros(1)
    gaps = np.diff(starts) if len(starts) > 1 else np.array([4])
    median_gap = float(np.median(gaps))
    velocities = notes[:, 3]
    distinct = len(np.unique(velocities))
    mean_velocity = float(velocities.mean()) if len(velocities) else 80.0
    mean_pitch = float(pitched[:, 2].mean()) if len(pitched) else 60.0
    families = len(np.unique(notes[:, 4])) if len(notes) else 1
    scale = [0, 2, 3, 5, 7, 8, 10] if key_mode else [0, 2, 4, 5, 7, 9, 11]
    scale = [(k + int(key)) % 12 for k in scale]
    chromatic = float(np.mean(~np.isin(pitched[:, 2] % 12, scale))) if len(pitched) else 0.0
    return np.array(
        [
            int(key_mode),
            0 if tempo_bpm < 90 else 1 if tempo_bpm <= 130 else 2,
            0 if median_gap >= 4 else 2 if median_gap <= 1 else 1,
            0 if distinct <= 2 else 1 if mean_velocity < 64 else 2 if mean_velocity < 96 else 3,
            0 if mean_pitch < 55 else 1 if mean_pitch <= 67 else 2,
            0 if families <= 1 else 1 if families <= 3 else 2,
            0 if chromatic < 0.1 else 1 if chromatic < 0.25 else 2,
        ],
        dtype=np.uint8,
    )


class Senses:
    """The causal heard-state of one stream: chroma, sounding notes, clock and progress.

    ``observe(token)`` advances the state by the event just heard; ``raw()`` reads the
    sense row that describes the moment before the next event. ``bar`` is the absolute
    bar of that moment.
    """

    def __init__(self, total_steps=None):
        self.chroma = np.zeros(12)
        self.held = []  # (end_step, pitch_class, family)
        self.step = 0
        self.total = total_steps

    @property
    def bar(self):
        return self.step // BAR

    def raw(self):
        out = np.zeros(SENSE_RAW, np.uint8)
        out[0] = self.step % BAR
        out[1] = (self.step // BAR) % CYCLE
        out[2] = 0 if not self.total else min(3, int(4 * self.step / max(1, self.total)))
        out[3:15] = np.round(self.chroma / max(1.0, self.chroma.sum()) * 255).astype(np.uint8)
        sounding = np.zeros((8, 12))
        for end, pc, fam in self.held:
            if end > self.step:
                sounding[fam, pc] = 1
        out[15:] = (sounding.ravel() * 255).astype(np.uint8)
        return out

    def observe(self, token):
        pitch, duration, delta, fam, _velocity = (int(t) for t in token)
        self.step += int(DELTAS[delta])
        self.held = [h for h in self.held if h[0] > self.step]
        self.held.append((self.step + int(DURATIONS[duration]), (pitch + 24) % 12, fam))
        self.chroma *= 0.96
        if fam != 7:
            self.chroma[(pitch + 24) % 12] += 1


# --- the connectome -----------------------------------------------------------------


@dataclass
class Design:
    cortex: int = 2048  # melody and harmony; rhythm and timbre are half
    phrase: int = 2048
    embedding: int = 64
    seed: int = 41
    trace_decay: float = 0.85
    recall_amplitude: float = 1.0
    working_memory: bool = True
    form_memory: bool = True
    belt: bool = True  # False: the ear projects straight into the cortices, no shared embedding
    belt_feedback: bool = True  # the cortices also project back into the belt
    eta: float = 0.005  # 0.16 (the earlier design) kicked every synapse by ~1% per update and froze the net
    tonic: float = 1.0  # resting bias of every free neuron: keeps each stage alive and on the sigmoid's slope
    free_steps: int = 32  # settling budget of the free phase while learning (32 leaves the equilibrium unreached)
    nudged_steps: int = 16
    decay: float = 1e-6  # every update shrinks each plastic synapse and bias by this fraction
    normalize: float = 0.98  # RMS normalisation of the step (0: plain momentum steps)
    version: int = 1  # 2: interval sense, a slow piece trace and a form cortex; 3: the plan and the theme record
    form: int = 0  # neurons of the form cortex (version 2); 0 picks cortex // 2
    piece_decay: float = 0.98  # the slow trace: the piece so far (0.995 in version 3: about seven bars)
    plan_dropout: float = 0.5  # version 3: share of training rows that learn without the plan input

    @property
    def label(self):
        return f"musician{self.version}:c{self.cortex}-p{self.phrase}-e{self.embedding}"

    @property
    def form_size(self):
        return self.form or self.cortex // 2

    @property
    def slots(self):
        """Output slots: the five event choices and, in version 3, the eight plan classes."""
        return tuple(SIZES) + (tuple(PLAN_WIDTHS) if self.version >= 3 else ())

    @property
    def record_keys(self):
        return RECORD if self.version >= 3 else CYCLE


def regions(design):
    d = design
    sizes = [
        ("ear", WINDOW * EVENT),
        ("sense", SENSE),
        ("mood", MOOD),
    ]
    if d.version >= 2:
        sizes.append(("interval", INTERVALS * INTERVAL_BINS))
    if d.version >= 3:
        sizes.append(("plan", PLAN))
        sizes.append(("bars", HISTORY * PLAN))
    if d.belt:
        sizes.append(("belt", WINDOW * d.embedding))
    sizes += [
        ("melody", d.cortex),
        ("harmony", d.cortex),
        ("rhythm", d.cortex // 2),
        ("timbre", d.cortex // 2),
    ]
    if d.version >= 2:
        sizes.append(("form", d.form_size))
    sizes += [
        ("phrase", d.phrase),
        ("prefrontal", d.phrase),
        ("recall", d.phrase),
    ]
    if d.version >= 2:
        sizes.append(("piece", d.phrase))
    sizes.append(("intention", int(sum(d.slots))))
    out, start = {}, 0
    for name, size in sizes:
        out[name] = range(start, start + size)
        start += size
    return out


def connectome(design=None):
    """The musician's connectome and the tie groups of the shared embedding."""
    d = design or Design()
    rng = np.random.default_rng(d.seed)
    pops = regions(d)
    pre, post, sign, tie = [], [], [], []

    def project(a, b, *, scale=1.0, reciprocal=True, density=1.0):
        a, b = np.asarray(pops[a]), np.asarray(pops[b])
        if density < 1.0:
            mask = rng.random((len(a), len(b))) < density
            i, j = np.nonzero(mask)
        else:
            i, j = np.meshgrid(np.arange(len(a)), np.arange(len(b)), indexing="ij")
            i, j = i.ravel(), j.ravel()
        magnitude = rng.uniform(0.0, 1.0, len(i)) * np.sqrt(6.0 / (len(a) + len(b))) * scale
        weights = rng.choice([-1.0, 1.0], len(i)) * magnitude
        pre.append(a[i])
        post.append(b[j])
        sign.append(weights)
        tie.append(np.full(len(i), -1))
        if reciprocal:
            pre.append(b[j])
            post.append(a[i])
            sign.append(weights)
            tie.append(np.full(len(i), -1))

    if d.belt:
        # the ear into the belt: one embedding table shared across the window positions
        ear, belt = np.asarray(pops["ear"]), np.asarray(pops["belt"])
        table = rng.choice([-1.0, 1.0], (EVENT, d.embedding)) * rng.uniform(
            0.0, 1.0, (EVENT, d.embedding)
        ) * np.sqrt(6.0 / (EVENT + d.embedding))
        e, k = np.meshgrid(np.arange(EVENT), np.arange(d.embedding), indexing="ij")
        for h in range(WINDOW):
            pre.append(ear[h * EVENT + e.ravel()])
            post.append(belt[h * d.embedding + k.ravel()])
            sign.append(table.ravel())
            tie.append((e * d.embedding + k).ravel())
    for cortex in ("melody", "harmony", "rhythm", "timbre"):
        if d.belt:
            project("belt", cortex, scale=0.6, reciprocal=d.belt_feedback)
        else:
            project("ear", cortex, scale=0.6, reciprocal=False)
        project("mood", cortex, scale=0.5, reciprocal=False)
        project(cortex, "phrase", scale=0.5)
        project(cortex, "intention", scale=0.5)
    project("sense", "harmony", scale=0.5, reciprocal=False)
    project("sense", "rhythm", scale=0.5, reciprocal=False)
    project("sense", "timbre", scale=0.5, reciprocal=False)
    project("sense", "phrase", scale=0.5, reciprocal=False)
    project("mood", "phrase", scale=0.5, reciprocal=False)
    project("mood", "intention", scale=0.3, reciprocal=False)
    if d.working_memory:
        project("prefrontal", "phrase", scale=0.5, reciprocal=False)
    if d.form_memory:
        project("recall", "phrase", scale=0.5, reciprocal=False)
    if d.version >= 2:
        # relative pitch into the melody and harmony cortices
        project("interval", "melody", scale=0.5, reciprocal=False)
        project("interval", "harmony", scale=0.4, reciprocal=False)
        # the form cortex: coherence of the piece as a whole. It reads the clock and
        # progress, the mood, the slow trace of the piece so far and the form record,
        # and biases the phrase cortex and every specialised cortex from above.
        project("sense", "form", scale=0.5, reciprocal=False)
        project("mood", "form", scale=0.5, reciprocal=False)
        if d.working_memory:
            project("piece", "form", scale=0.5, reciprocal=False)
            project("piece", "phrase", scale=0.4, reciprocal=False)
        if d.form_memory:
            project("recall", "form", scale=0.5, reciprocal=False)
        project("form", "phrase", scale=0.5)
        for cortex in ("melody", "harmony", "rhythm", "timbre"):
            project("form", cortex, scale=0.3, reciprocal=False)
        project("form", "intention", scale=0.3)
    if d.version >= 3:
        # the plan of the bar being written conditions the form, the phrase, every
        # specialised cortex and the intention from above; the bars before enter the form
        project("plan", "form", scale=0.5, reciprocal=False)
        project("plan", "phrase", scale=0.4, reciprocal=False)
        for cortex in ("melody", "harmony", "rhythm", "timbre"):
            project("plan", cortex, scale=0.3, reciprocal=False)
        project("plan", "intention", scale=0.3, reciprocal=False)
        project("bars", "form", scale=0.5, reciprocal=False)
        project("bars", "phrase", scale=0.2, reciprocal=False)
    project("phrase", "intention", scale=0.6)
    n = sum(len(v) for v in pops.values())
    all_pre, all_post, all_tie = np.concatenate(pre), np.concatenate(post), np.concatenate(tie)
    result = cd.Connectome.from_synapses(
        n,
        pre=all_pre,
        post=all_post,
        sign=np.concatenate(sign),
        populations={k: list(v) for k, v in pops.items()},
        label=d.label,
    )
    # ``from_synapses`` sorts by (post, pre); no projection is drawn twice, so the same
    # permutation carries the tie groups (checked, with a slow exact fallback).
    order = np.lexsort((all_pre, all_post))
    if len(result.pre) == len(order) and np.array_equal(result.pre, all_pre[order]) and np.array_equal(result.post, all_post[order]):
        groups = all_tie[order].astype(np.int64)
    else:
        key = {(int(a), int(b)): int(g) for a, b, g in zip(all_pre, all_post, all_tie, strict=True)}
        groups = np.array(
            [key[(int(a), int(b))] for a, b in zip(result.pre, result.post, strict=True)],
            dtype=np.int64,
        )
    return result, groups


def learner_config(eta=0.005, free_steps=32, nudged_steps=16, decay=1e-6, normalize=0.98):
    return cd.LearnerConfig(
        beta=0.3,
        eta=eta,
        eta_bias=eta / 10,
        free_steps=free_steps,
        nudged_steps=nudged_steps,
        centered=True,
        tolerance=0,
        temperature=0.2,
        normalize=normalize,
        momentum=0.8,
        normalize_floor=0.01,
        decay=decay,
    )


CLAMPED = ("ear", "sense", "mood", "prefrontal", "recall", "interval", "piece", "plan", "bars")


def build(design=None, *, backend="cpu", device=None):
    d = design or Design()
    graph, tie = connectome(d)
    plastic = np.ones(graph.n, bool)
    for name in CLAMPED:
        if name in graph.populations:
            plastic[list(graph.populations[name])] = False
    # Clamped ports rest at zero; every free neuron starts with a tonic bias so it sits on
    # the slope of its activation function rather than in the silent half below rest.
    bias = np.where(plastic, d.tonic, 0.0)
    brain = cd.Brain(
        graph,
        cd.learning_neuron_model(dt=1, leak=0.1),
        backend=backend,
        device=device,
        dense_limit=32768,
        precision="float32" if backend == "torch" else None,
        bias=bias,
    )
    learner = cd.Learner(
        brain,
        graph.populations["intention"],
        learner_config(d.eta, d.free_steps, d.nudged_steps, d.decay, d.normalize),
        plastic_neurons=plastic,
        tie_groups=tie,
        slots=d.slots,
    )
    return Musician(learner, d)


def describe(musician):
    w = musician.learner.brain.connectome
    d = musician.design
    stream_state = {
        "prefrontal": "Trace of the phrase cortex, decay %.2f" % d.trace_decay,
        "recall": (
            "theme record: delta-rule record keyed by absolute bar (32 keys), written at each new bar, read at the bar the plan's lag returns to"
            if d.version >= 3
            else "delta-rule record keyed by the bar of a sixteen-bar cycle, written at each new bar"
        ),
    }
    if d.version >= 2:
        stream_state["piece"] = "slow Trace of the phrase cortex, decay %.3f: the piece so far" % d.piece_decay
    return {
        "neurons": int(w.n),
        "directed_synapses": int(w.synapses),
        "trainable_parameters": int(musician.learner.parameters()),
        "regions": {k: len(v) for k, v in w.populations.items()},
        "design": asdict(d),
        "learning": "Cadence centered free/nudged local contrast on one joint equilibrium; no backpropagation graph",
        "stream_state": stream_state,
        "output_slots": list(d.slots),
        **({"plan_head": "the eight profile classes of the bar being written (composer/form.py)"} if d.version >= 3 else {}),
    }


# --- the owned state of a stream ---------------------------------------------------------


@dataclass
class StreamState:
    """Per-row state carried between events: working memory, form record, warm equilibrium.
    ``bar`` is the bar of the last real event (absolute in version 3, in-cycle before)."""

    trace: Trace | None
    memory: FastSynapses | None
    warm: cd.BrainState | None = None
    bar: np.ndarray = field(default_factory=lambda: np.full(0, -1))
    piece: Trace | None = None

    def reset_rows(self, rows):
        rows = np.asarray(rows)
        if not len(rows):
            return
        if self.trace is not None:
            self.trace.reset(len(self.trace.trace), rows)
        if self.piece is not None:
            self.piece.reset(len(self.piece.trace), rows)
        if self.memory is not None:
            self.memory.reset(len(self.memory.strength), rows)
        self.bar[rows] = -1
        if self.warm is not None:
            _zero_rows(self.warm, rows)

    def select(self, row):
        """An isolated one-row copy of this state (for a future that must not write back)."""
        return self.copy_rows([row])

    def copy_rows(self, rows):
        rows = np.asarray(rows)
        out = StreamState(None, None, None, self.bar[rows].copy())
        out.trace = _copy_trace(self.trace, rows)
        out.piece = _copy_trace(self.piece, rows)
        if self.memory is not None:
            out.memory = FastSynapses(
                self.memory.pre, self.memory.post, rule="delta", amplitude=self.memory.amplitude
            )
            out.memory.reset(len(rows))
            out.memory.strength = self.memory.strength[rows].copy()
            out.memory.mass = self.memory.mass[rows].copy()
        if self.warm is not None:
            out.warm = cd.BrainState(
                v=np.array(self.warm.v)[rows].copy(),
                activation=np.array(self.warm.activation)[rows].copy(),
                adaptation=np.array(self.warm.adaptation)[rows].copy(),
                steps=self.warm.steps,
            )
        return out


def _copy_trace(trace, rows):
    if trace is None:
        return None
    out = Trace(trace.connectome, decay=trace.decay, amplitude=trace.amplitude, source=trace.source, target=trace.target)
    out.reset(len(rows))
    out.trace = trace.trace[rows].copy()
    out.last = trace.last[rows].copy()
    out.cold = trace.cold[rows].copy()
    return out


def _carry_rows(state, source, rows):
    """Write the rows of ``source`` (a state copied by ``copy_rows`` from ``rows``) back
    into ``rows`` of ``state``: the traces, the record, the bar and the warm equilibrium."""
    rows = np.asarray(rows)
    for mine, theirs in ((state.trace, source.trace), (state.piece, source.piece)):
        if mine is not None and theirs is not None:
            mine.trace[rows] = theirs.trace
            mine.last[rows] = theirs.last
            mine.cold[rows] = theirs.cold
    if state.memory is not None and source.memory is not None:
        state.memory.strength[rows] = source.memory.strength
        state.memory.mass[rows] = source.memory.mass
    state.bar[rows] = source.bar
    if state.warm is not None and source.warm is not None:
        handle = getattr(state.warm, "device", None)
        if isinstance(handle, dict) and "s" in handle:
            import torch

            index = torch.as_tensor(rows, device=handle["s"].device)
            for key, name in (("s", "activation"), ("v", "v"), ("a", "adaptation")):
                if key in handle and handle[key] is not None:
                    value = getattr(source.warm, name)
                    if value is not None:
                        handle[key][index] = torch.as_tensor(np.asarray(value), device=handle[key].device, dtype=handle[key].dtype)
            for name in ("v", "activation", "adaptation"):
                state.warm.__dict__[name] = None
        else:
            for name in ("v", "activation", "adaptation"):
                mine, theirs = state.warm.__dict__.get(name), getattr(source.warm, name)
                if mine is not None and theirs is not None:
                    mine[rows] = theirs


def _zero_rows(state, rows):
    handle = getattr(state, "device", None)
    if isinstance(handle, dict) and "s" in handle:
        import torch

        index = torch.as_tensor(np.asarray(rows), device=handle["s"].device)
        for key in ("s", "v", "a"):
            if key in handle and handle[key] is not None:
                handle[key][index] = 0
        for name in ("v", "activation", "adaptation"):
            state.__dict__[name] = None
        return
    for name in ("v", "activation", "adaptation"):
        array = state.__dict__.get(name)
        if array is not None:
            array[rows] = 0.0


@dataclass
class Conditioning:
    """What version 3 clamps beside the heard events, one row per stream:

    ``plan``: the ``(batch, 8)`` profile classes of the bar being written (-1 in a row: the
    plan input is absent for that row); ``bars``: ``(batch, HISTORY, 8)`` profiles of the
    bars before (-1 rows: no bar yet); ``bar``: the absolute bar of each row; ``lag``: the
    lag class the theme record reads at (from the plan when composing, from the measured
    profile when learning or listening).
    """

    plan: np.ndarray
    bars: np.ndarray
    bar: np.ndarray
    lag: np.ndarray

    @classmethod
    def absent(cls, batch):
        return cls(
            np.full((batch, len(PLAN_WIDTHS)), -1, int),
            np.full((batch, HISTORY, len(PLAN_WIDTHS)), -1, int),
            np.zeros(batch, int),
            np.zeros(batch, int),
        )

    def rows(self, rows):
        rows = np.asarray(rows)
        return Conditioning(self.plan[rows], self.bars[rows], self.bar[rows], self.lag[rows])


def conditioning_of(piece_tokens, bar, *, plan_row=None, mode=0):
    """One row's conditioning while composing or listening: the measured profiles of the
    bars before ``bar`` from the piece so far, and the plan of the current bar (the planned
    profile when given, else absent, with the lag read from the plan when given and from
    the measured piece otherwise)."""
    profiles = bar_profiles(piece_tokens, mode=mode, bars=bar + 1) if len(piece_tokens) else np.zeros((bar + 1, len(PLAN_WIDTHS)), int)
    history = history_of(profiles, bar)
    if plan_row is not None:
        plan = np.asarray(plan_row, dtype=int)
        lag = int(plan[7])
    else:
        plan = np.full(len(PLAN_WIDTHS), -1, int)
        lag = int(profiles[bar, 7]) if bar < len(profiles) else 0
    return plan, history, lag


class Musician:
    """A trained (or untrained) musician: the learner plus what a stream owns."""

    def __init__(self, learner, design=None):
        self.learner = learner
        self.design = design or Design()
        w = learner.brain.connectome
        self.populations = w.populations
        self.n = w.n
        self.ear = columns(np.asarray(w.populations["ear"]))
        self.sense = columns(np.asarray(w.populations["sense"]))
        self.mood = columns(np.asarray(w.populations["mood"]))
        self.phrase = columns(np.asarray(w.populations["phrase"]))
        self.recall = columns(np.asarray(w.populations["recall"]))
        self.interval = columns(np.asarray(w.populations["interval"])) if "interval" in w.populations else None
        self.plan = columns(np.asarray(w.populations["plan"])) if "plan" in w.populations else None
        self.bars = columns(np.asarray(w.populations["bars"])) if "bars" in w.populations else None
        self.output_index = learner.output_index
        self.slots = tuple(self.design.slots)
        self.slot_offsets = np.cumsum((0,) + self.slots[:-1])
        # settling budget when playing (predict, imagine, review, replay); learning keeps the config's
        self.settle_steps = int(learner.config.free_steps)

    @property
    def planned(self):
        """Whether this brain has the plan input and head (design version 3)."""
        return self.plan is not None

    def free(self, drive, warm=None):
        return self.brain.settle_batch(drive, steps=self.settle_steps, state=warm, tolerance=0)

    # -- construction and custody

    @property
    def brain(self):
        return self.learner.brain

    def save(self, path, *, compressed=False):
        path = Path(path)
        self.learner.save(path, compressed=compressed)
        path.with_suffix(".design.json").write_text(json.dumps(asdict(self.design), indent=2))
        return path

    @classmethod
    def load(cls, path, *, backend="cpu", device=None, precision=None):
        path = Path(path)
        learner = cd.Learner.load(path, backend=backend, device=device, precision=precision)
        design_file = path.with_suffix(".design.json")
        design = Design(**json.loads(design_file.read_text())) if design_file.exists() else Design()
        return cls(learner, design)

    # -- stream state

    def fresh(self, batch):
        w = self.learner.brain.connectome
        trace = None
        if self.design.working_memory:
            trace = Trace(
                w,
                decay=self.design.trace_decay,
                amplitude=1.0,
                source="phrase",
                target="prefrontal",
            )
            trace.reset(batch)
        piece = None
        if self.design.working_memory and "piece" in w.populations:
            piece = Trace(w, decay=self.design.piece_decay, amplitude=1.0, source="phrase", target="piece")
            piece.reset(batch)
        memory = None
        if self.design.form_memory:
            keys = self.design.record_keys
            memory = FastSynapses(
                np.arange(keys),
                np.arange(keys, keys + len(w.populations["phrase"])),
                rule="delta",
                amplitude=self.design.recall_amplitude,
            )
            memory.reset(batch)
        return StreamState(trace, memory, None, np.full(batch, -1), piece)

    def drive(self, context, sense, mood, state=None, conditioning=None):
        """The stimulus of one event: the heard window, the senses, the mood, and the
        working memory and form record written into their ports; in version 3 also the
        plan of the bar and the bars before, with the theme record read at the plan's lag."""
        context = np.asarray(context)
        out = np.zeros((len(context), self.n), np.float64)
        out[:, self.ear] = 2.5 * encode_events(context)
        out[:, self.sense] = 2.5 * encode_sense(sense)
        out[:, self.mood] = 2.5 * encode_mood(mood)
        if self.interval is not None:
            out[:, self.interval] = 2.5 * encode_intervals(context)
        if self.planned:
            if conditioning is None:
                conditioning = Conditioning.absent(len(context))
            out[:, self.plan] = 2.5 * encode_plan(conditioning.plan)
            out[:, self.bars] = 2.5 * encode_bars(conditioning.bars)
        if state is not None:
            if state.trace is not None:
                out = state.trace.stimulate(out)
            if state.piece is not None:
                out = state.piece.stimulate(out)
            if state.memory is not None:
                if self.planned:
                    _, read = record_keys(conditioning.bar, conditioning.lag)
                    out[:, self.recall] = state.memory.recall(read)
                else:
                    out[:, self.recall] = state.memory.recall(_bar_key(sense))
        return out

    def advance(self, state, free, sense, conditioning=None):
        """After a real event's free phase: the trace follows the phrase cortex; a new bar
        writes the phrase activity into the form record at the bar's key."""
        if state.trace is not None:
            state.trace.update(free)
        if state.piece is not None:
            state.piece.update(free)
        if self.planned:
            bar = np.asarray(conditioning.bar if conditioning is not None else np.zeros(len(state.bar), int), dtype=int)
            if state.memory is not None:
                phrase = np.atleast_2d(free.activation)[:, self.phrase]
                write, _ = record_keys(bar, np.zeros(len(bar), int))
                state.memory.observe(write, phrase, write=bar != state.bar)
        else:
            bar = np.asarray(sense)[:, 1].astype(int)
            if state.memory is not None:
                phrase = np.atleast_2d(free.activation)[:, self.phrase]
                state.memory.observe(_bar_key(sense), phrase, write=bar != state.bar)
        state.bar = bar
        state.warm = free

    # -- prediction

    def probabilities(self, activation):
        """One softmax distribution per output slot from an intention activation: the five
        event choices, then (version 3) the eight plan classes."""
        out = []
        for offset, width in zip(self.slot_offsets, self.slots):
            z = activation[:, offset : offset + width] / self.learner.config.temperature
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            out.append(p / p.sum(1, keepdims=True))
        return out

    def predict(self, context, sense, mood, state=None, conditioning=None):
        drive = self.drive(context, sense, mood, state, conditioning)
        free = self.free(drive, None if state is None else state.warm)
        activation = np.atleast_2d(free.activation)[:, self.output_index]
        return self.probabilities(activation), free

    def learn(self, context, sense, mood, labels, state, *, warm=True, weight=None, conditioning=None):
        """One imitation update on a batch of streams; ``weight`` (one number per row) scales
        and signs each row's pull toward its label, the valence of a practised piece. In
        version 3 ``labels`` carries the eight plan classes after the five event choices."""
        drive = self.drive(context, sense, mood, state, conditioning)
        learned, report = self.learner.step(drive, labels, warm=state.warm if warm else None, weight=weight)
        self.advance(state, learned.free, sense, conditioning)
        return learned, report

    # -- imagination: futures rolled forward through the brain's own predictions

    def imagine(
        self,
        history,
        senses,
        mood,
        *,
        futures=8,
        horizon=16,
        rng=None,
        temperature=0.9,
        state=None,
        detune=0.0,
        stop_at=None,
        top=0,
        plan_table=None,
        prime=WINDOW,
    ):
        """Roll ``futures`` continuations of one stream forward, each in its own batch row
        with its own copy of the working memory, form record and warm state.

        ``history`` is the list of committed tokens (the first ``prime`` are the heard
        opening, not part of the piece), ``senses`` the stream's ``Senses`` after those
        tokens, ``state`` the one-row stream state after them. In version 3
        ``plan_table`` is the ``(bars, 8)`` plan of the piece: each row hears the plan of
        the bar it is writing and the measured profiles of the bars it has written.
        Returns the futures (tokens, surprise per event, per-event probabilities) and their
        end states. Nothing here writes the live state.
        """
        rng = rng or np.random.default_rng(0)
        branch = state.copy_rows([0] * futures) if state is not None else self.fresh(futures)
        histories = [list(history) for _ in range(futures)]
        sensed = [_copy_senses(senses) for _ in range(futures)]
        tokens = [[] for _ in range(futures)]
        surprise = [[] for _ in range(futures)]
        moods = np.repeat(np.asarray(mood, dtype=int)[None, :], futures, axis=0)
        mode = int(np.asarray(mood)[0])
        hidden = np.concatenate(
            [np.asarray(self.populations[k]) for k in ("melody", "harmony", "rhythm", "timbre")]
        )
        for _ in range(horizon):
            active = np.array([stop_at is None or s.step < stop_at for s in sensed])
            if not active.any():
                break
            context = np.array([h[-WINDOW:] for h in histories])
            raw = np.array([s.raw() for s in sensed])
            conditioning = self._conditioning(histories, sensed, plan_table, prime, mode) if self.planned else None
            drive = self.drive(context, raw, moods, branch, conditioning)
            if detune:
                drive[:, hidden] += rng.normal(0, detune, (futures, len(hidden)))
            free = self.free(drive, branch.warm)
            probabilities = self.probabilities(
                np.atleast_2d(free.activation)[:, self.output_index]
            )[: len(SIZES)]
            for j in range(futures):
                if not active[j]:
                    continue
                token, nll = [], 0.0
                for p in probabilities:
                    q = p[j] ** (1 / temperature)
                    if top:  # sample among the ``top`` most expected choices only
                        q[np.argsort(q)[:-top]] = 0.0
                    q /= q.sum()
                    choice = int(rng.choice(len(q), p=q))
                    token.append(choice)
                    nll -= float(np.log(max(p[j][choice], 1e-12)))
                if stop_at is not None:
                    remaining = stop_at - sensed[j].step
                    if DELTAS[token[2]] > remaining:
                        token[2] = int(np.argmin(np.abs(DELTAS - remaining)))
                        if DELTAS[token[2]] > remaining:
                            token[2] = max(0, token[2] - 1)
                tokens[j].append(token)
                surprise[j].append(nll / len(SIZES))
                histories[j].append(token)
                sensed[j].observe(token)
            # a row whose future has ended hears nothing more: its working memory, record and
            # warm state stay where its last event left them, exactly as a replay leaves them
            idle = np.flatnonzero(~active)
            kept = branch.copy_rows(idle) if len(idle) else None
            self.advance(branch, free, raw, conditioning)
            if kept is not None:
                _carry_rows(branch, kept, idle)
        return {
            "tokens": tokens,
            "surprise": surprise,
            "senses": sensed,
            "state": branch,
            "histories": histories,
        }

    def _conditioning(self, histories, sensed, plan_table, prime, mode):
        """Version 3 conditioning of every row from its own piece so far and the plan."""
        plans, bars, bar_of, lags = [], [], [], []
        for h, s in zip(histories, sensed):
            bar = int(s.bar)
            row = None
            if plan_table is not None and len(plan_table):
                row = np.asarray(plan_table)[min(bar, len(plan_table) - 1)]
            plan, history, lag = conditioning_of(h[prime:], bar, plan_row=row, mode=mode)
            plans.append(plan)
            bars.append(history)
            bar_of.append(bar)
            lags.append(lag)
        return Conditioning(np.array(plans), np.array(bars), np.array(bar_of), np.array(lags))

    # -- imagining the plan of a whole piece at bar resolution (version 3)

    def imagine_plan(self, prime, mood, bars, *, futures=8, rng=None, temperature=0.9, total_steps=None):
        """Roll ``futures`` plans of ``bars`` bars forward through the plan head alone: at
        each bar the brain hears the opening, the mood, the clock at that bar and the plan
        so far (the bars before), with the plan input absent, and the eight classes of the
        bar are sampled from the plan head. Nothing is written; the stream state is fresh.
        Returns the plans ``(futures, bars, 8)`` and the plan surprise per bar."""
        if not self.planned:
            raise ValueError("this design has no plan head")
        rng = rng or np.random.default_rng(0)
        moods = np.repeat(np.asarray(mood, dtype=int)[None, :], futures, axis=0)
        context = np.repeat(np.asarray(prime)[None, -WINDOW:], futures, axis=0)
        plans = np.full((futures, bars, len(PLAN_WIDTHS)), -1, int)
        surprise = np.zeros((futures, bars))
        state = self.fresh(futures)
        total = total_steps or bars * BAR
        for b in range(bars):
            senses = Senses(total)
            senses.step = b * BAR
            raw = np.repeat(senses.raw()[None, :], futures, axis=0)
            conditioning = Conditioning(
                np.full((futures, len(PLAN_WIDTHS)), -1, int),
                np.array([history_of(plans[j], b) for j in range(futures)]),
                np.full(futures, b, int),
                np.zeros(futures, int),
            )
            probabilities, free = self.predict(context, raw, moods, state, conditioning)
            for j in range(futures):
                nll = 0.0
                for k, p in enumerate(probabilities[len(SIZES) :]):
                    q = p[j] ** (1 / temperature)
                    q /= q.sum()
                    choice = int(rng.choice(len(q), p=q))
                    plans[j, b, k] = choice
                    nll -= float(np.log(max(p[j][choice], 1e-12)))
                surprise[j, b] = nll / len(PLAN_WIDTHS)
            state.warm = free
        return plans, surprise

    # -- listening back: the whole score through a fresh stream

    def review(self, tokens, mood, *, prime=WINDOW, total_steps=None, rewind=True):
        """Hear the score from the start in one fresh stream and measure the surprise of every
        event under the brain's own expectation. Returns per-event surprise, the bar of every
        event, and the stream state snapshots at the start of every bar. The first ``prime``
        events are heard before the clock starts when ``rewind`` is set, as when composing.
        In version 3 the plan input is absent, the bars before are the measured profiles of
        the piece, and ``plan_surprise`` is the plan head's surprise at each event's bar."""
        state = self.fresh(1)
        senses = primed(tokens[:prime], total_steps, rewind=rewind)
        history = [list(t) for t in tokens[:prime]]
        surprise, bars, snapshots, plan_surprise = [], [], {}, []
        moods = np.asarray(mood, dtype=int)[None, :]
        mode = int(np.asarray(mood)[0])
        piece = [list(t) for t in tokens[prime:]]
        profiles = bar_profiles(piece, mode=mode) if self.planned and piece else None
        for token in tokens[prime:]:
            raw = senses.raw()[None, :]
            bar = int(senses.bar)
            if bar not in snapshots:
                snapshots[bar] = (
                    state.copy_rows([0]),
                    _copy_senses(senses),
                    len(history),
                )
            conditioning = None
            if self.planned:
                current = profiles[bar] if bar < len(profiles) else np.zeros(len(PLAN_WIDTHS), int)
                conditioning = Conditioning(
                    np.full((1, len(PLAN_WIDTHS)), -1, int),
                    history_of(profiles, bar)[None, :],
                    np.array([bar]),
                    np.array([int(current[7])]),
                )
            probabilities, free = self.predict(np.array([history[-WINDOW:]]), raw, moods, state, conditioning)
            nll = -np.mean(
                [np.log(max(p[0][int(c)], 1e-12)) for p, c in zip(probabilities[: len(SIZES)], token)]
            )
            surprise.append(float(nll))
            bars.append(bar)
            if self.planned:
                plan_surprise.append(
                    -float(np.mean([np.log(max(p[0][int(c)], 1e-12)) for p, c in zip(probabilities[len(SIZES) :], current)]))
                )
            self.advance(state, free, raw, conditioning)
            history.append(list(token))
            senses.observe(token)
        return {
            "surprise": np.asarray(surprise),
            "bars": np.asarray(bars),
            "snapshots": snapshots,
            "steps": senses.step,
            "plan_surprise": np.asarray(plan_surprise),
            "profiles": profiles,
        }

    # -- exact replay: every settling iteration of one row, for the live view

    def replay(self, history, senses, state, tokens, mood, observer, *, total_steps=None, plan_table=None, prime=WINDOW):
        """Hear ``tokens`` one by one from ``state`` (one row) exactly as ``imagine`` or
        ``review`` settles them, but one iteration at a time, calling
        ``observer(event_index, iteration, activation, repair, mismatch, drive)`` at every
        iteration. Returns the stream state after the tokens and the surprise per event.

        Settling one step at a time with the state carried is the same recurrence as one
        call of ``free_steps`` steps; ``tests/test_musician.py`` checks the equality.
        """
        history = [list(t) for t in history]
        senses = _copy_senses(senses)
        if total_steps is not None:
            senses.total = total_steps
        state = state.copy_rows([0]) if state is not None else self.fresh(1)
        moods = np.asarray(mood, dtype=int)[None, :]
        mode = int(np.asarray(mood)[0])
        brain = self.learner.brain
        steps = self.settle_steps
        surprise = []
        for k, token in enumerate(tokens):
            raw = senses.raw()[None, :]
            conditioning = self._conditioning([history], [senses], plan_table, prime, mode) if self.planned else None
            drive = self.drive(np.array([history[-WINDOW:]]), raw, moods, state, conditioning)
            current = state.warm
            previous = None if current is None else np.array(current.activation[0])
            for iteration in range(1, steps + 1):
                current = brain.settle_batch(drive, steps=1, state=current, tolerance=0)
                activation = np.array(current.activation[0])
                mismatch = _mismatch(brain, drive[0], current, 0)
                repair = activation if previous is None else activation - previous
                observer(k, iteration, activation, repair, mismatch, drive[0])
                previous = activation
            probabilities = self.probabilities(np.atleast_2d(current.activation)[:, self.output_index])[: len(SIZES)]
            nll = -np.mean([np.log(max(p[0][int(c)], 1e-12)) for p, c in zip(probabilities, token)])
            surprise.append(float(nll))
            self.advance(state, current, raw, conditioning)
            history.append(list(token))
            senses.observe(token)
        return state, senses, surprise


def _mismatch(brain, drive, state, row):
    """Every neuron's equation error at this iteration: synaptic input plus drive and bias
    minus its potential (the same quantity the studio's recorder reports)."""
    from cadence.blocks import BlockTransport

    s = np.array(state.activation[row])
    v = np.array(state.v[row])
    blocks = None if brain._blocks is None else BlockTransport(brain.layout, brain._blocks)
    synaptic = brain._synaptic_input(s[None, :], blocks)[0]
    error = synaptic + drive + brain.bias - v
    if brain.neuron_model.adaptation:
        error -= brain.neuron_model.adaptation.strength * np.array(state.adaptation[row])
    return error


def primed(prime, total_steps=None, *, rewind=True):
    """The senses after hearing ``prime`` events; with ``rewind`` the clock returns to the
    first bar and nothing is left sounding, so the piece proper starts at step zero."""
    senses = Senses(total_steps)
    for t in prime:
        senses.observe(t)
    if rewind:
        senses.step = 0
        senses.held = []
    return senses


def _bar_key(sense):
    raw = np.asarray(sense)
    key = np.zeros((len(raw), CYCLE))
    key[np.arange(len(raw)), raw[:, 1].astype(int) % CYCLE] = 1
    return key


def _copy_senses(senses):
    out = Senses(senses.total)
    out.chroma = senses.chroma.copy()
    out.held = list(senses.held)
    out.step = senses.step
    return out


# --- score output -----------------------------------------------------------------------


def events_from_tokens(tokens, key=0):
    """Absolute note events ``(start_step, duration_steps, midi_pitch, velocity, family)``."""
    out, step = [], 0
    for pitch, duration, delta, fam, velocity in tokens:
        step += int(DELTAS[int(delta)])
        midi = int(pitch) + 24 + (0 if int(fam) == 7 else int(key))
        out.append(
            (
                step,
                int(DURATIONS[int(duration)]),
                int(np.clip(midi, 0, 127)),
                int(min(127, int(velocity) * 16 + 8)),
                int(fam),
            )
        )
    return out


def write_score(events, path, *, bpm=100, name="Cadence musician"):
    """A multitrack MIDI file, one channel per instrument family (supplied GM programs)."""
    import mido

    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    midi.tracks.append(meta)
    meta.append(mido.MetaMessage("track_name", name=name[:80]))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    end = max((e[0] + e[1] for e in events), default=BAR) * 120
    meta.append(mido.MetaMessage("end_of_track", time=end))
    for fam, program in enumerate(PROGRAMS):
        mine = [e for e in events if e[4] == fam]
        if not mine:
            continue
        channel = 9 if fam == 7 else min(8, fam)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.Message("program_change", channel=channel, program=program, time=0))
        messages = []
        for start, duration, pitch, velocity, _ in mine:
            messages.append((start * 120, 1, pitch, velocity))
            messages.append(((start + duration) * 120 - 1, 0, pitch, 0))
        previous = 0
        for tick, on, pitch, velocity in sorted(messages, key=lambda m: (m[0], m[1])):
            track.append(
                mido.Message(
                    "note_on" if on else "note_off",
                    channel=channel,
                    note=pitch,
                    velocity=velocity,
                    time=tick - previous,
                )
            )
            previous = tick
        track.append(mido.MetaMessage("end_of_track", time=max(0, end - previous)))
    midi.save(path)
    return path
