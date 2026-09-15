import numpy as np

from arm.env import TORQUES, Arm, ArmConfig, jacobian, moving_path, wrap


def test_integrator_replays_deterministically_and_reports_executed_action():
    a, b = Arm(seed=3), Arm(seed=3)
    ma, mb = a.reset(), b.reset()
    np.testing.assert_array_equal(ma.observation["hand"], mb.observation["hand"])
    for d, action in enumerate((4, 8, 0, 2, 6, 4, 4)):
        ma, mb = a.act(d, action), b.act(d, action)
        assert ma.executed == action and ma.feedback_for == d
        np.testing.assert_array_equal(ma.observation["hand"], mb.observation["hand"])
    assert a.state["theta"].tolist() == b.state["theta"].tolist()


def test_observation_and_displacement_are_aligned_with_the_executed_decision():
    arm = Arm(seed=1)
    m0 = arm.reset()
    hand0 = m0.observation["hand"].copy()
    m1 = arm.act(0, 8)  # both torques +1
    np.testing.assert_allclose(m1.observation["d_hand"], m1.observation["hand"] - hand0, atol=1e-12)
    assert m1.observation["d_velocity"].tolist() == (arm.state["omega"] - np.zeros(2)).tolist()
    assert np.all(m1.observation["d_velocity"] > 0)
    assert m1.dt == 0.1 and m1.reward_known


def test_velocity_clip_is_reported_as_a_flag_and_executed_action_is_the_clipped_one():
    arm = Arm(ArmConfig(gain=100.0), seed=0)
    arm.reset()
    m = arm.act(0, 8)
    assert m.observation["flags"].tolist() == [1.0, 1.0]
    assert np.all(np.abs(arm.state["omega"]) <= 2.0)


def test_targets_lie_in_the_reachable_annulus_and_unreachable_targets_are_rejected():
    arm = Arm(seed=5)
    for _ in range(200):
        r = float(np.linalg.norm(arm.sample_target()))
        assert 0.2 <= r <= 0.95


def test_success_terminates_with_bonus_and_time_limit_truncates_with_final_observation():
    arm = Arm(ArmConfig(horizon=5), seed=2)
    m = arm.reset()
    for k in range(5):
        m = arm.act(k, 4)  # no torque: coasting from rest never reaches
    assert m.truncated and not m.terminated and m.final_observation is not None
    arm = Arm(seed=2)
    m = arm.reset(theta=np.array([0.3, 0.4]))
    arm._target = arm.hand.copy()  # evaluation-only access to place the target on the hand
    for k in range(9):
        m = arm.act(k, 4)
        assert not m.terminated
    m = arm.act(9, 4)
    assert m.terminated and m.reward > 0.9


def test_wrap_and_jacobian_and_paths():
    assert wrap(np.array([np.pi + 0.1]))[0] < 0
    theta = np.array([0.3, -0.7])
    arm = Arm()
    eps = 1e-6
    J = jacobian(theta, arm.config.lengths)
    for j in range(2):
        d = np.zeros(2); d[j] = eps
        fd = (arm.hand_of(theta + d) - arm.hand_of(theta - d)) / (2 * eps)
        np.testing.assert_allclose(J[:, j], fd, atol=1e-6)
    rng = np.random.default_rng(0)
    for kind in ("line", "circle", "combination"):
        path = moving_path(kind, rng)
        points = np.array([path(t) for t in range(200)])
        assert np.all(np.linalg.norm(points, axis=1) <= 1.0 + 1e-9)
        assert np.linalg.norm(points[1] - points[0]) < 0.1
    assert len(TORQUES) == 9 and TORQUES[4].tolist() == [0.0, 0.0]
