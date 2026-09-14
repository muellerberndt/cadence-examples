"""The form of a piece: bar profiles, returns, the plan head and the planned composition."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.form import (
    HISTORY,
    LAGS,
    PLAN,
    PLAN_WIDTHS,
    RECORD,
    bar_profiles,
    encode_bars,
    encode_plan,
    history_of,
    plan_fit,
    plan_shape,
    record_keys,
    return_fidelity,
)
from composer.listen import Listener
from composer.musician import SIZES, WINDOW, Conditioning, Design, Senses, build, primed, regions
from composer.perform import compose, prime_tokens

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def bar_of_notes(pitches, *, delta=3, duration=3, family=0, velocity=4, first_delta=0):
    """Four notes on the beats of one bar (deltas 0, 4, 4, 4 steps), pitches key-relative."""
    out = []
    for k, p in enumerate(pitches):
        out.append([int(p), duration, first_delta if k == 0 else 4, family, velocity])
    return out


def test_profiles_measure_the_bar_and_find_returns():
    theme = bar_of_notes([36, 40, 43, 48])  # a rising arpeggio
    other = bar_of_notes([50, 49, 47, 45], velocity=7, family=2, first_delta=4)  # a louder brass descent
    transposed = bar_of_notes([41, 45, 48, 53], first_delta=4)  # the theme a fourth up: same intervals
    tokens = theme + other + transposed
    p = bar_profiles(tokens, mode=0, bars=3)
    assert p.shape == (3, 8)
    assert p[0, 0] == 2  # four onsets exceed the density edges 0 and 2
    assert p[1, 3] > p[0, 3]  # louder
    assert p[1, 4] == 1 and p[0, 4] == 1  # one family each
    assert p[0, 7] == 0 and p[1, 7] == 0  # new material
    assert LAGS[p[2, 7]] == 2  # the third bar returns to the first, two bars back
    assert return_fidelity(tokens, p) > 0.5


def test_profile_encoders_and_record_keys():
    classes = np.array([[1, 2, 3, 0, 1, 2, 1, 4], [-1, 0, 0, 0, 0, 0, 0, 0]])
    x = encode_plan(classes)
    assert x.shape == (2, PLAN) and x[0].sum() == len(PLAN_WIDTHS) and x[1].sum() == 0
    history = np.full((2, HISTORY, 8), -1)
    history[0, -1] = classes[0]
    h = encode_bars(history)
    assert h.shape == (2, HISTORY * PLAN) and h[0].sum() == len(PLAN_WIDTHS) and h[1].sum() == 0
    table = np.arange(5 * 8).reshape(5, 8) % 3
    assert history_of(table, 2).shape == (HISTORY, 8)
    assert (history_of(table, 2)[-1] == table[1]).all() and (history_of(table, 2)[0] == -1).all()
    write, read = record_keys([5, 5, 1], [0, 2, 5])  # lags 0, 2 bars, 6 bars
    assert write.shape == (3, RECORD) and write[0, 5] == 1
    assert read[0].sum() == 0  # new material reads nothing
    assert read[1, 3] == 1  # bar 5 returns to bar 3
    assert read[2].sum() == 0  # six bars before bar 1 does not exist


def test_shape_measures_prefer_a_formed_piece():
    flat = np.zeros((16, 8), int)
    flat[:, 0] = 3
    flat[:, 3] = 2
    formed = flat.copy()
    formed[8:12, 0] = 5
    formed[8:12, 3] = 3  # a climax in the third quarter
    formed[15, 0] = 1
    formed[15, 3] = 1  # an ending that falls away
    formed[12, 7] = 8  # a return to the opening
    a, b = plan_shape(flat), plan_shape(formed)
    assert b["contrast"] > a["contrast"] and b["climax"] > a["climax"]
    assert b["ending"] > a["ending"] and b["return"] > a["return"]
    assert plan_fit(formed, formed) == 1.0 and plan_fit(flat, formed) < 1.0


@pytest.fixture(scope="module")
def small():
    return build(Design(cortex=48, phrase=32, embedding=8, seed=3, belt=False, version=3, tonic=1.0))


def test_version_three_regions_and_head(small):
    pops = small.populations
    assert "plan" in pops and "bars" in pops and len(pops["plan"]) == PLAN and len(pops["bars"]) == HISTORY * PLAN
    assert small.slots == tuple(SIZES) + tuple(PLAN_WIDTHS)
    assert len(pops["intention"]) == sum(small.slots)
    starts = sorted((r.start, r.stop) for r in regions(small.design).values())
    assert all(a[1] == b[0] for a, b in zip(starts, starts[1:]))
    activation = np.random.default_rng(0).normal(size=(2, sum(small.slots)))
    probabilities = small.probabilities(activation)
    assert len(probabilities) == len(small.slots)
    assert np.allclose([p.sum(1) for p in probabilities], 1.0)


def test_learning_with_plan_labels_and_dropout(small):
    rng = np.random.default_rng(1)
    batch = 4
    context = rng.integers(0, 8, size=(batch, WINDOW, 5)) * np.array([9, 1, 1, 1, 1])
    sense = np.zeros((batch, 3 + 12 + 96), np.uint8)
    mood = np.zeros((batch, 7), int)
    labels = np.concatenate([rng.integers(0, 8, size=(batch, 5)), rng.integers(0, 3, size=(batch, 8))], axis=1)
    conditioning = Conditioning(labels[:, 5:].copy(), np.full((batch, HISTORY, 8), -1), np.zeros(batch, int), labels[:, 12])
    conditioning.plan[1] = -1  # one row learns without the plan input
    state = small.fresh(batch)
    before = small.brain.efficacy.copy()
    small.learn(context, sense, mood, labels, state, conditioning=conditioning)
    assert not np.allclose(before, small.brain.efficacy)
    assert (state.bar == 0).all()
    # the theme record was written at bar 0 for every row
    assert state.memory is not None and np.abs(state.memory.strength).sum() > 0


def test_imagined_plans_and_the_planned_composition_are_isolated(small):
    brief = np.array([0, 1, 1, 3, 1, 2, 0])
    prime = prime_tokens(brief)
    plans, surprise = small.imagine_plan(prime, brief, 6, futures=3, rng=np.random.default_rng(0))
    assert plans.shape == (3, 6, 8) and (plans >= 0).all() and surprise.shape == (3, 6)
    for k, width in enumerate(PLAN_WIDTHS):
        assert (plans[:, :, k] < width).all()
    listener = Listener(1.0, brief, target_plan_surprise=1.0)
    scored = listener.score_plan(plans[0], surprise[0])
    assert {"score", "contrast", "climax", "ending", "return", "plan_coherence"} <= set(scored)
    small.settle_steps = 4
    result = compose(small, brief, bars=4, futures=2, horizon=6, edits=1, seed=1, target_plan_surprise=1.0)
    assert result["plan"]["plan"] and len(result["plan"]["plan"]) == 4
    assert len(result["events"]) > 0 and result["profiles"] is not None
    assert "form" in result["final_score"] and "plan_fit" in result["final_score"]
    assert all(e["width"] == 2 for e in result["edits"])
    assert len(result["per_bar_plan_surprise"]) >= 1


def test_review_reports_plan_surprise_and_a_saved_version_three_brain_round_trips(small, tmp_path):
    brief = np.array([1, 0, 0, 1, 0, 0, 0])
    tokens = prime_tokens(brief) + bar_of_notes([36, 40, 43, 48]) * 3
    small.settle_steps = 4
    review = small.review(tokens, brief, total_steps=48)
    assert len(review["plan_surprise"]) == len(review["surprise"]) == len(tokens) - WINDOW
    assert review["profiles"] is not None and review["profiles"].shape[1] == 8
    path = small.save(tmp_path / "brain.npz")
    from composer.musician import Musician

    again = Musician.load(path)
    assert again.design.version == 3 and again.planned and again.slots == small.slots
    again.settle_steps = 4
    other = again.review(tokens, brief, total_steps=48)
    assert np.allclose(other["surprise"], review["surprise"])
