"""Form: the piece at bar resolution.

A piece is more than the chain of its events. This module gives the musician the bar as a
unit: a **bar profile** of eight measured classes (how dense, how high, how wide, how loud,
how many instrument families, how chromatic, how long the notes, and which earlier bar it
returns to), the **plan** of a piece as the sequence of its bar profiles, and the
measures a listener applies to a whole plan or a whole draft.

Everything here is supplied measurement, stated as such. What is learned is elsewhere:
the musician's plan head predicts the profile of the bar being written from the mood, the
progress, the profiles of the bars before and its own state (``composer/musician.py``,
design version 3), and the note intention is conditioned on that plan.

Bar profile classes (one-hot widths in ``PLAN_WIDTHS``):

    density   onsets in the bar: 0 · 1-2 · 3-4 · 5-8 · 9-16 · 17+
    register  mean pitch of the pitched notes: none · <48 · 48-59 · 60-71 · 72+ (MIDI)
    range     highest minus lowest pitch: 0-4 · 5-11 · 12-23 · 24+
    dynamics  mean velocity class (0..7): 0-1 · 2-3 · 4-5 · 6-7
    texture   instrument families sounding: 0 · 1 · 2 · 3+
    tension   fraction of pitched notes outside the key: <0.1 · <0.25 · more
    rhythm    the most common duration: short (<=2 steps) · medium (<=8) · long
    lag       which earlier bar this bar returns to, as a distance in bars, or 0 for new
              material: 0 · 1 · 2 · 3 · 4 · 6 · 8 · 12 · 16 · 32

The lag is found by similarity of the bar's melodic bigrams (pitch step and gap between
consecutive notes of the top line, weight 0.7) and its onset positions (0.3) to every earlier bar; the most
similar earlier bar above ``RETURN_THRESHOLD`` is the referent and its distance is rounded
to the nearest lag class. Pitch steps make the match transposition-invariant.
"""

from __future__ import annotations

import numpy as np

from .vocabulary import BAR, DELTAS, DURATIONS

PLAN_NAMES = ("density", "register", "range", "dynamics", "texture", "tension", "rhythm", "lag")
PLAN_WIDTHS = (6, 5, 4, 4, 4, 3, 3, 10)
PLAN_OFFSETS = np.cumsum((0,) + PLAN_WIDTHS[:-1])
PLAN = int(sum(PLAN_WIDTHS))
LAGS = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 32])
HISTORY = 8  # bars of profile history the form cortex hears
RECORD = 32  # keys of the theme record: the last 32 bars, addressed by absolute bar
RETURN_THRESHOLD = 0.5
DENSITY_EDGES = np.array([0, 2, 4, 8, 16])  # class = number of edges the count exceeds
RANGE_EDGES = np.array([4, 11, 23])


def _bar_notes(tokens):
    """Absolute step, duration, pitch (0..72 key-relative), velocity class, family per event."""
    tokens = np.asarray(tokens, dtype=int)
    if not len(tokens):
        return np.zeros((0, 5), int)
    steps = np.cumsum(DELTAS[tokens[:, 2]])
    return np.column_stack((steps, DURATIONS[tokens[:, 1]], tokens[:, 0], tokens[:, 4], tokens[:, 3]))


def _signature(notes, bar_start):
    """The melodic bigrams of a bar's top line and its onset positions, as two sets."""
    onsets = {}
    for step, duration, pitch, velocity, fam in notes:
        if fam == 7:
            continue
        if step not in onsets or pitch > onsets[step]:
            onsets[step] = pitch
    line = sorted(onsets.items())
    bigrams = set()
    for (s0, p0), (s1, p1) in zip(line, line[1:]):
        bigrams.add((int(p1 - p0), int(min(s1 - s0, 16))))
    positions = {int(s - bar_start) for s, *_ in notes}
    return bigrams, positions


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(signature_a, signature_b):
    """Melodic-bigram overlap weighted 0.7 and onset-position overlap 0.3, so a bar with the
    same rhythm but another melody does not reach the return threshold; 0 for empty bars."""
    return 0.7 * _jaccard(signature_a[0], signature_b[0]) + 0.3 * _jaccard(signature_a[1], signature_b[1])


def lag_class(distance):
    """The nearest lag class of a distance in bars (0 is new material)."""
    if distance <= 0:
        return 0
    return int(np.argmin(np.abs(LAGS[1:] - distance))) + 1


def bar_profiles(tokens, *, mode=0, bars=None):
    """``(bars, 8)`` uint8 profile classes of a key-relative token list.

    ``bars`` fixes the number of bars (padding empty bars at the end); otherwise the count
    follows the last note's end. ``mode`` (0 major, 1 minor) selects the key's scale for
    the tension class; pitches are key-relative so the tonic is pitch class 0.
    """
    notes = _bar_notes(tokens)
    if bars is None:
        bars = int((notes[:, 0] + notes[:, 1]).max() // BAR + 1) if len(notes) else 1
    scale = {0, 2, 3, 5, 7, 8, 10} if mode else {0, 2, 4, 5, 7, 9, 11}
    out = np.zeros((bars, len(PLAN_WIDTHS)), np.uint8)
    signatures = []
    which = notes[:, 0] // BAR if len(notes) else np.zeros(0, int)
    for b in range(bars):
        mine = notes[which == b]
        pitched = mine[mine[:, 4] != 7]
        onsets = len(np.unique(mine[:, 0])) if len(mine) else 0
        out[b, 0] = int((onsets > DENSITY_EDGES).sum())
        if len(pitched):
            midi = pitched[:, 2] + 24
            mean_pitch = midi.mean()
            out[b, 1] = 1 if mean_pitch < 48 else 2 if mean_pitch < 60 else 3 if mean_pitch < 72 else 4
            out[b, 2] = int(((midi.max() - midi.min()) > RANGE_EDGES).sum())
            chromatic = sum((p + 24) % 12 not in scale for p in pitched[:, 2]) / len(pitched)
            out[b, 5] = 0 if chromatic < 0.1 else 1 if chromatic < 0.25 else 2
        if len(mine):
            out[b, 3] = int(mine[:, 3].mean() // 2)
            out[b, 4] = min(3, len(np.unique(mine[:, 4])))
            durations = mine[:, 1]
            typical = np.bincount(durations).argmax()
            out[b, 6] = 0 if typical <= 2 else 1 if typical <= 8 else 2
        signature = _signature(mine, b * BAR)
        best, referent = 0.0, -1
        if signature[0] or signature[1]:
            for j in range(b - 1, -1, -1):
                s = similarity(signature, signatures[j])
                if s > best + 1e-9:
                    best, referent = s, j
        out[b, 7] = lag_class(b - referent) if referent >= 0 and best >= RETURN_THRESHOLD else 0
        signatures.append(signature)
    return out


def encode_plan(classes):
    """``(batch, 8)`` classes as ``(batch, PLAN)`` one-hot; a row with any -1 is unknown (zeros)."""
    classes = np.asarray(classes, dtype=int)
    x = np.zeros((len(classes), PLAN), np.float32)
    known = (classes >= 0).all(1)
    rows = np.flatnonzero(known)
    for col, (offset, width) in enumerate(zip(PLAN_OFFSETS, PLAN_WIDTHS)):
        x[rows, offset + np.clip(classes[rows, col], 0, width - 1)] = 1
    return x


def encode_bars(history):
    """``(batch, HISTORY, 8)`` profiles of the bars before (most recent last; -1 rows are
    absent bars) as ``(batch, HISTORY * PLAN)`` one-hot."""
    history = np.asarray(history, dtype=int)
    x = np.zeros((len(history), HISTORY * PLAN), np.float32)
    for h in range(HISTORY):
        x[:, h * PLAN : (h + 1) * PLAN] = encode_plan(history[:, h])
    return x


def history_of(profiles, bar):
    """The ``HISTORY`` profiles before absolute bar ``bar`` from a ``(bars, 8)`` table,
    -1 where the piece had not started."""
    out = np.full((HISTORY, len(PLAN_WIDTHS)), -1, int)
    for h in range(HISTORY):
        b = bar - HISTORY + h
        if 0 <= b < len(profiles):
            out[h] = profiles[b]
    return out


def record_keys(bar, lag):
    """One-hot write key (this bar) and read key (the referent bar, or zeros for lag 0)
    for the theme record, ``bar`` and ``lag`` one integer per row."""
    bar = np.atleast_1d(np.asarray(bar, dtype=int))
    lag = np.atleast_1d(np.asarray(lag, dtype=int))
    write = np.zeros((len(bar), RECORD))
    write[np.arange(len(bar)), bar % RECORD] = 1
    read = np.zeros((len(bar), RECORD))
    distance = LAGS[np.clip(lag, 0, len(LAGS) - 1)]
    hit = (distance > 0) & (distance <= bar)
    read[np.flatnonzero(hit), (bar[hit] - distance[hit]) % RECORD] = 1
    return write, read


# --- what a listener measures about a whole plan or a whole draft --------------------------


def plan_shape(profiles):
    """Supplied whole-piece measures of a ``(bars, 8)`` profile table, each in [0, 1]:

    * contrast: the density and dynamics classes are not all the same (normalised entropy);
    * climax: the loudest, densest bar lies in the second half but not in the last bar;
    * ending: the last bar is quieter and sparser than the piece's peak;
    * return: at least one bar in the second half returns to earlier material.
    """
    profiles = np.asarray(profiles, dtype=int)
    n = len(profiles)
    if n < 2:
        return {"contrast": 0.0, "climax": 0.0, "ending": 0.0, "return": 0.0}

    def entropy(column, width):
        counts = np.bincount(column, minlength=width) / n
        counts = counts[counts > 0]
        return float(-(counts * np.log(counts)).sum() / np.log(width))

    contrast = 0.5 * entropy(profiles[:, 0], PLAN_WIDTHS[0]) + 0.5 * entropy(profiles[:, 3], PLAN_WIDTHS[3])
    energy = profiles[:, 0] + profiles[:, 3]
    peak = int(np.argmax(energy))
    climax = 1.0 if n // 2 <= peak < n - 1 else 0.5 if peak < n - 1 else 0.0
    ending = float(np.clip((energy[peak] - energy[-1]) / max(1, energy[peak]), 0, 1))
    returns = profiles[n // 2 :, 7] > 0
    return {"contrast": contrast, "climax": climax, "ending": ending, "return": float(returns.any())}


def plan_fit(realised, plan):
    """Fraction of bars whose realised density, dynamics, register and texture classes
    equal the plan's (the classes a listener hears most directly)."""
    realised = np.asarray(realised, dtype=int)
    plan = np.asarray(plan, dtype=int)
    n = min(len(realised), len(plan))
    if n == 0:
        return 0.0
    hits = realised[:n][:, [0, 3, 1, 4]] == plan[:n][:, [0, 3, 1, 4]]
    return float(hits.mean())


def return_fidelity(tokens, profiles):
    """Mean similarity between every bar that claims a return and its referent bar (1 when
    no bar claims one); a realised return that does not resemble its referent scores low."""
    notes = _bar_notes(tokens)
    profiles = np.asarray(profiles, dtype=int)
    which = notes[:, 0] // BAR if len(notes) else np.zeros(0, int)
    signatures = [_signature(notes[which == b], b * BAR) for b in range(len(profiles))]
    scores = []
    for b, lag in enumerate(profiles[:, 7]):
        distance = int(LAGS[lag])
        if distance and b - distance >= 0:
            scores.append(similarity(signatures[b], signatures[b - distance]))
    return float(np.mean(scores)) if scores else 1.0
