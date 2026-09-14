"""Listening: how the musician judges an imagined future or a finished draft.

Three measured quantities, all supplied and reported separately:

* **coherence**: the brain's own surprise along the passage, compared with the surprise
  it feels on real music (its held-out validation surprise). Far below is rote
  repetition; far above is noise. The score is the negative distance to that target.
* **mood fit**: the seven mood classes measured on the passage (the same measurement the
  corpus was labeled with) against the requested mood, weighted per class.
* **health**: explicit anti-collapse checks that no listener would forgive: the same
  pitch over and over, long silence, one note per event with no chords in a texture
  that asks for them, or a register far outside the requested one.

A valence for the next rehearsal is the winner's advantage over its rivals.
None of this is a proof of beauty; it is what the musician can measure about itself.
"""

from __future__ import annotations

import numpy as np

from .musician import DELTAS, DURATIONS, MOOD_NAMES, mood_classes

MOOD_WEIGHTS = np.array([1.0, 0.4, 0.8, 0.3, 0.6, 0.8, 0.4])  # mode, tempo, energy, dynamics, register, texture, tension


def measured_mood(tokens, tempo_bpm, mode):
    """The mood classes of a token passage (key-relative pitches; tempo is a supplied plan)."""
    notes, step = [], 0
    for pitch, duration, delta, fam, velocity in tokens:
        step += int(DELTAS[int(delta)])
        notes.append((step, int(DURATIONS[int(duration)]), int(pitch) + 24, int(velocity) * 16 + 8, int(fam)))
    if not notes:
        return np.zeros(7, np.uint8)
    return mood_classes(notes, tempo_bpm, 0, mode)


def mood_fit(tokens, brief, tempo_bpm):
    measured = measured_mood(tokens, tempo_bpm, int(brief[0]))
    hits = (measured == np.asarray(brief)).astype(float)
    return float((hits * MOOD_WEIGHTS).sum() / MOOD_WEIGHTS.sum()), measured


def health(tokens):
    tokens = np.asarray(tokens)
    if len(tokens) < 2:
        return 0.0, ["too few events"]
    issues, penalty = [], 0.0
    pitches = tokens[tokens[:, 3] != 7, 0]
    if len(pitches) > 3:
        top = np.bincount(pitches).max() / len(pitches)
        if top > 0.4:
            issues.append("one pitch dominates")
            penalty += 3 * (top - 0.4)
        same = float((np.diff(pitches) == 0).mean())
        if same > 0.5:
            issues.append("repeated notes")
            penalty += 2 * (same - 0.5)
    gaps = DELTAS[tokens[:, 2]]
    if len(gaps) > 3 and (gaps >= 32).mean() > 0.25:
        issues.append("long silences")
        penalty += 1.0
    if len(np.unique(pitches)) < 3 and len(pitches) > 6:
        issues.append("little pitch variety")
        penalty += 1.0
    return -penalty, issues


class Listener:
    """Scores futures and drafts. ``target_surprise`` is the brain's held-out surprise."""

    def __init__(self, target_surprise, brief, tempo_bpm=100, *, weights=(1.0, 2.0, 1.0)):
        self.target = float(target_surprise)
        self.brief = np.asarray(brief, dtype=int)
        self.tempo = tempo_bpm
        self.weights = weights

    def coherence(self, surprise):
        surprise = np.asarray(surprise, float)
        if not len(surprise):
            return 0.0
        return -abs(float(surprise.mean()) - self.target)

    def score(self, tokens, surprise):
        coherence = self.coherence(surprise)
        fit, measured = mood_fit(tokens, self.brief, self.tempo)
        healthy, issues = health(tokens)
        wc, wf, wh = self.weights
        return {
            "score": wc * coherence + wf * fit + wh * healthy,
            "coherence": coherence,
            "mean_surprise": float(np.mean(surprise)) if len(surprise) else 0.0,
            "mood_fit": fit,
            "measured_mood": {k: int(v) for k, v in zip(MOOD_NAMES, measured)},
            "health": healthy,
            "issues": issues,
        }

    def rank(self, futures):
        """Scores of every future of an ``imagine`` result and the index of the best."""
        scores = [self.score(t, s) for t, s in zip(futures["tokens"], futures["surprise"])]
        return scores, int(np.argmax([s["score"] for s in scores]))
