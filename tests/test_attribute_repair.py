"""A selected instrument can causally affect pitch through shared local repairs."""

import cadence as cd
import numpy as np

from composer.ensemble import SIZES
from tools.sample_ensemble import draw_event


def test_attribute_completion_uses_links_without_learning():
    n = sum(SIZES)
    instrument = sum(SIZES[:3])
    wiring = cd.Wiring.from_edges(
        n=n, pre=[instrument, 0], post=[0, instrument], sign=[0.6, 0.6]
    )
    brain = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule()),
        np.arange(n),
        cd.LearnerConfig(temperature=0.2, beta=0.3, free_steps=64, tolerance=0),
        slots=SIZES,
    )
    drive = np.ones((1, n)) * 0.03
    state = brain.free(drive)
    before = state.activation.copy()
    weights = brain.engine.weights.copy()
    _, independent, _ = draw_event(brain, state, drive, np.random.default_rng(11))
    token, conditional, _ = draw_event(
        brain, state, drive, np.random.default_rng(11), conditional=True
    )
    assert token[3] == 0
    assert conditional[0][0] > independent[0][0] + 1e-6
    np.testing.assert_array_equal(brain.engine.weights, weights)
    np.testing.assert_array_equal(state.activation, before)
    assert brain.updates == 0
    brain.engine = brain.engine.with_parameters(edge_scale=np.zeros(wiring.edges))
    cut_state = brain.free(drive)
    _, cut_free, _ = draw_event(brain, cut_state, drive, np.random.default_rng(11))
    _, cut_conditioned, _ = draw_event(
        brain, cut_state, drive, np.random.default_rng(11), conditional=True
    )
    np.testing.assert_allclose(cut_free[0], cut_conditioned[0], atol=1e-10)
