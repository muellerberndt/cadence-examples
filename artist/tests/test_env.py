"""The canvas world: rasterisation, the measurements, the windows and the splits."""

import numpy as np
import pytest

from artist.env import (
    DEVELOPMENT_FAMILIES,
    FAMILIES,
    TORQUES,
    Arm,
    CanvasConfig,
    CanvasWorld,
    body_config,
    bresenham,
    chamfer,
    crop,
    drawing,
    foreground_f1,
    hand_of,
    join_action,
    pixel_of,
    render,
    split_action,
    window_origin,
)


def world(config: CanvasConfig | None = None, seed: int = 0) -> CanvasWorld:
    config = config or CanvasConfig()
    return CanvasWorld(config, Arm(body_config(), seed=seed), seed=seed)


def test_actions_split_and_join_over_nine_torques_and_two_pen_states():
    assert len({join_action(t, p) for t in range(9) for p in (0, 1)}) == 18
    for action in range(18):
        torque, pen = split_action(action)
        assert join_action(torque, pen) == action
    with pytest.raises(ValueError):
        split_action(18)


def test_bresenham_is_one_pixel_wide_and_connected():
    points = bresenham(np.array([2.0, 3.0]), np.array([9.0, 14.0]))
    assert points[0].tolist() == [2, 3] and points[-1].tolist() == [9, 14]
    steps = np.abs(np.diff(points, axis=0)).max(axis=1)
    assert set(np.unique(steps).tolist()) == {1}
    assert len(points) == len(np.unique(points, axis=0))
    single = bresenham(np.array([4.0, 4.0]), np.array([4.0, 4.0]))
    assert single.tolist() == [[4, 4]]


def test_pen_up_leaves_no_ink_and_pen_down_marks_the_swept_path():
    w = world()
    w.reset(drawing("development", 0, 0, w.config))
    before_pixel = w.pen_pixel.copy()
    w.act(0, join_action(8, 0))
    assert w.canvas.sum() == 0
    after_pixel = w.pen_pixel.copy()
    w.act(1, join_action(8, 1))
    expected = bresenham(after_pixel, w.pen_pixel)
    inside = [p for p in expected if 0 <= p[0] < w.config.size and 0 <= p[1] < w.config.size]
    assert w.canvas.sum() == len(inside)
    for point in inside:
        assert w.canvas[point[0], point[1]]
    assert np.linalg.norm(before_pixel - after_pixel) > 0


def test_the_mark_sensor_is_the_window_the_decision_started_in():
    w = world(seed=3)
    w.reset(drawing("development", 4, 0, w.config))
    w.act(0, join_action(8, 1))
    origin = window_origin(w.pen_pixel, w.config.window)
    before = crop(w.canvas, origin, w.config.window).copy()
    m = w.act(1, join_action(8, 1))
    after = crop(w.canvas, origin, w.config.window)
    np.testing.assert_allclose(m.observation["mark"], (after - before).ravel())
    now = window_origin(w.pen_pixel, w.config.window)
    np.testing.assert_allclose(m.observation["canvas_local"], crop(w.canvas, now, w.config.window).ravel())
    np.testing.assert_allclose(m.observation["target_local"], crop(w.target, now, w.config.window).ravel())
    assert m.observation["pen"].tolist() == [1.0]


def test_reward_is_the_discrepancy_reduction_minus_the_torque_cost_and_replay_is_deterministic():
    a, b = world(seed=5), world(seed=5)
    target = drawing("development", 7, 0, a.config)
    ma, mb = a.reset(target), b.reset(target)
    np.testing.assert_array_equal(ma.observation["hand"], mb.observation["hand"])
    before = a.discrepancy
    for decision, action in enumerate((join_action(8, 1), join_action(0, 1), join_action(4, 0), join_action(2, 1))):
        ma, mb = a.act(decision, action), b.act(decision, action)
        torque, _ = split_action(action)
        cost = a.config.torque_cost * float(TORQUES[torque] @ TORQUES[torque])
        assert ma.reward == pytest.approx(before - a.discrepancy - cost)
        assert ma.executed == action and ma.feedback_for == decision and ma.reward_known
        before = a.discrepancy
        np.testing.assert_array_equal(a.canvas, b.canvas)
    assert a.discrepancy == chamfer(a.canvas, a.target)


def test_chamfer_and_f1_on_known_canvases():
    target = np.zeros((32, 32), bool)
    target[10, 5:20] = True
    assert chamfer(target, target) == 0.0
    assert foreground_f1(target, target)["f1"] == 1.0
    empty = np.zeros_like(target)
    assert chamfer(empty, target) == 1.0 and chamfer(target, empty) == 1.0
    assert foreground_f1(empty, target)["f1"] == 0.0
    assert chamfer(empty, empty) == 0.0 and foreground_f1(empty, empty)["f1"] == 1.0
    shifted = np.zeros_like(target)
    shifted[11, 5:20] = True
    assert chamfer(shifted, target) == pytest.approx(1.0 / float(np.hypot(32, 32)))
    assert foreground_f1(shifted, target)["f1"] == 1.0  # one pixel of tolerance
    far = np.zeros_like(target)
    far[25, 5:20] = True
    assert foreground_f1(far, target)["f1"] == 0.0
    assert chamfer(far, target) > chamfer(shifted, target)


def test_coordinates_round_trip_between_pixels_and_the_body():
    config = CanvasConfig()
    for pixel in ([0.0, 0.0], [15.5, 15.5], [31.0, 7.0]):
        np.testing.assert_allclose(pixel_of(hand_of(np.array(pixel), config), config), pixel, atol=1e-9)


def test_targets_are_balanced_by_family_and_split_by_rotation_scale_and_composition():
    config = CanvasConfig()
    development = [drawing("development", k, 0, config) for k in range(40)]
    heldout = [drawing("heldout", k, 0, config) for k in range(50)]
    assert {d.family for d in development} == set(DEVELOPMENT_FAMILIES)
    assert {d.family for d in heldout} == set(FAMILIES)
    assert "composition" not in {d.family for d in development}
    counts = [sum(1 for d in heldout if d.family == f) for f in FAMILIES]
    assert max(counts) - min(counts) <= 1
    assert all(0.55 <= d.scale <= 0.80 for d in development)
    assert all(d.scale < 0.55 or d.scale > 0.80 for d in heldout)
    step = 2 * np.pi / 8
    for d in heldout:
        offsets = np.abs(((d.rotation - np.arange(8) * step + np.pi) % (2 * np.pi)) - np.pi)
        assert offsets.min() > step / 4  # never on a development rotation
    for d in development + heldout:
        points = np.argwhere(d.target)
        assert points.min() >= config.margin and points.max() <= config.size - 1 - config.margin
        radius = np.linalg.norm([hand_of(p, config) for p in points], axis=1)
        assert radius.min() > 0.12 and radius.max() < 0.95  # inside the reach, outside the fold


def test_a_drawing_ends_on_the_budget_or_when_it_stops_improving():
    config = CanvasConfig(max_decisions=6, patience=10**9)
    w = world(config)
    m = w.reset(drawing("development", 1, 0, config))
    for k in range(6):
        m = w.act(k, join_action(4, 0))
    assert m.truncated and not m.terminated and m.final_observation is not None


def test_the_patience_clock_starts_at_the_first_mark():
    """An approach with the pen up never improves the discrepancy and must not spend the clock."""
    config = CanvasConfig(max_decisions=40, patience=3)
    w = world(config, seed=11)
    m = w.reset(drawing("development", 2, 0, config))
    for k in range(12):
        m = w.act(k, join_action(4, 0))  # coasting with the pen up, well past the patience
        assert not m.truncated, "a drawing with no ink is abandoned before it has drawn anything"
    assert w.canvas.sum() == 0
    while w.canvas.sum() == 0:  # put the pen down until it leaves a mark
        m = w.act(w._tick, join_action(8, 1))
        assert not m.truncated
    for k in range(3):
        m = w.act(w._tick, join_action(4, 0))  # the pen is up again: no better discrepancy
    assert m.truncated and not m.terminated


def test_render_draws_every_polyline_of_a_shape():
    paths = [np.array([[4.0, 4.0], [4.0, 10.0]]), np.array([[20.0, 20.0], [26.0, 20.0]])]
    canvas = render(paths, 32)
    assert canvas[4, 4:11].all() and canvas[20:27, 20].all()
    assert canvas.sum() == 14
