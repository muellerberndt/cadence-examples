"""Listening: how the musician judges an imagined future, a plan or a finished draft.

Measured quantities, all supplied and reported separately:

* **coherence**: the brain's own surprise along the passage, compared with the surprise
  it feels on real music (its held-out validation surprise). Far below is rote
  repetition; far above is noise. The score is the negative distance to that target.
* **mood fit**: the seven mood classes measured on the passage (the same measurement the
  corpus was labeled with) against the requested mood, weighted per class.
* **health**: explicit anti-collapse checks that no listener would forgive: the same
  pitch over and over, long silence, one note per event with no chords in a texture
  that asks for them, or a register far outside the requested one.
* **plan fit** (version 3): how far the bars a passage realises match their plan.
* **form** (version 3, whole piece): the plan head's own surprise at the realised bars
  against its held-out target; the shape of the realised profiles (contrast, a climax in
  the second half, an ending that falls away, a return to earlier material); and the
  fidelity of every claimed return to the bar it returns to.

A valence for the next rehearsal is the winner's advantage over its rivals.
None of this is a proof of beauty; it is what the musician can measure about itself.
"""

from __future__ import annotations

import numpy as np

from .form import bar_profiles, plan_fit, plan_shape, return_fidelity
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
    """Scores futures, plans and drafts. ``target_surprise`` is the brain's held-out
    surprise on real music; ``target_plan_surprise`` the plan head's (version 3)."""

    def __init__(
        self,
        target_surprise,
        brief,
        tempo_bpm=100,
        *,
        weights=(1.0, 2.0, 1.0),
        plan_weight=1.5,
        target_plan_surprise=None,
        form_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ):
        self.target = float(target_surprise)
        self.brief = np.asarray(brief, dtype=int)
        self.tempo = tempo_bpm
        self.weights = weights
        self.plan_weight = plan_weight
        self.target_plan = None if target_plan_surprise is None else float(target_plan_surprise)
        self.form_weights = form_weights  # plan coherence, contrast, climax, ending, return, return fidelity

    @property
    def mode(self):
        return int(self.brief[0])

    def coherence(self, surprise, target=None):
        surprise = np.asarray(surprise, float)
        if not len(surprise):
            return 0.0
        target = self.target if target is None else target
        return -abs(float(surprise.mean()) - target)

    def score(self, tokens, surprise, *, plan=None, prefix=(), bars_from=0):
        """The local score of a passage. With ``plan`` (a ``(bars, 8)`` table) the passage
        is heard after ``prefix`` (the piece before it) and its realised bars from
        ``bars_from`` are compared with the plan."""
        coherence = self.coherence(surprise)
        fit, measured = mood_fit(tokens, self.brief, self.tempo)
        healthy, issues = health(tokens)
        wc, wf, wh = self.weights
        out = {
            "score": wc * coherence + wf * fit + wh * healthy,
            "coherence": coherence,
            "mean_surprise": float(np.mean(surprise)) if len(surprise) else 0.0,
            "mood_fit": fit,
            "measured_mood": {k: int(v) for k, v in zip(MOOD_NAMES, measured)},
            "health": healthy,
            "issues": issues,
        }
        if plan is not None and len(tokens):
            whole = [list(t) for t in prefix] + [list(t) for t in tokens]
            realised = bar_profiles(whole, mode=self.mode)
            window = slice(bars_from, len(realised))
            out["plan_fit"] = plan_fit(realised[window], np.asarray(plan)[window])
            out["score"] += self.plan_weight * out["plan_fit"]
        return out

    def rank(self, futures, **kwargs):
        """Scores of every future of an ``imagine`` result and the index of the best."""
        scores = [self.score(t, s, **kwargs) for t, s in zip(futures["tokens"], futures["surprise"])]
        return scores, int(np.argmax([s["score"] for s in scores]))

    # -- whole plans and whole pieces (version 3)

    def score_plan(self, plan, plan_surprise):
        """A plan before any note: the plan head's surprise near its target, and the
        shape measures of the planned profiles."""
        shape = plan_shape(plan)
        wc, wk, wm, we, wr, _ = self.form_weights
        coherence = self.coherence(plan_surprise, self.target_plan) if self.target_plan is not None else 0.0
        return {
            "score": wc * coherence + wk * shape["contrast"] + wm * shape["climax"] + we * shape["ending"] + wr * shape["return"],
            "plan_coherence": coherence,
            "mean_plan_surprise": float(np.mean(plan_surprise)) if len(plan_surprise) else 0.0,
            **shape,
        }

    def score_piece(self, tokens, review, *, plan=None):
        """A whole draft after listening back: the local score of the whole, plus (version 3)
        the form measures on the realised bars and the fit to the plan."""
        out = self.score(tokens, review["surprise"])
        profiles = review.get("profiles")
        if profiles is None or not len(tokens):
            return out
        shape = plan_shape(profiles)
        wc, wk, wm, we, wr, wf = self.form_weights
        plan_surprise = review.get("plan_surprise", np.zeros(0))
        coherence = self.coherence(plan_surprise, self.target_plan) if self.target_plan is not None and len(plan_surprise) else 0.0
        fidelity = return_fidelity(tokens, profiles)
        form = wc * coherence + wk * shape["contrast"] + wm * shape["climax"] + we * shape["ending"] + wr * shape["return"] + wf * fidelity
        out.update(
            form=form,
            plan_coherence=coherence,
            mean_plan_surprise=float(np.mean(plan_surprise)) if len(plan_surprise) else 0.0,
            return_fidelity=fidelity,
            **shape,
        )
        if plan is not None:
            out["plan_fit"] = plan_fit(profiles, plan)
            form += self.plan_weight * out["plan_fit"]
        out["score"] += form
        return out
