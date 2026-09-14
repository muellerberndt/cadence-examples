"""A selected instrument can causally affect pitch through shared local repairs."""

import cadence as cd
import numpy as np

from composer.ensemble import SIZES
from tools.sample_ensemble import draw_event


def test_attribute_completion_uses_links_without_learning():
    n = sum(SIZES)
    instrument = sum(SIZES[:3])
    connectome = cd.Connectome.from_synapses(
        n=n, pre=[instrument, 0], post=[0, instrument], sign=[0.6, 0.6]
    )
    brain = cd.Learner(
        cd.Brain(connectome, cd.learning_neuron_model()),
        np.arange(n),
        cd.LearnerConfig(temperature=0.2, beta=0.3, free_steps=64, tolerance=0),
        slots=SIZES,
    )
    drive = np.ones((1, n)) * 0.03
    state = brain.free(drive)
    before = state.activation.copy()
    weights = brain.brain.weights.copy()
    _, independent, _ = draw_event(brain, state, drive, np.random.default_rng(11))
    token, conditional, _ = draw_event(
        brain, state, drive, np.random.default_rng(11), conditional=True
    )
    assert token[3] == 0
    assert conditional[0][0] > independent[0][0] + 1e-6
    np.testing.assert_array_equal(brain.brain.weights, weights)
    np.testing.assert_array_equal(state.activation, before)
    assert brain.updates == 0
    brain.brain = brain.brain.with_parameters(efficacy=np.zeros(connectome.synapses))
    cut_state = brain.free(drive)
    _, cut_free, _ = draw_event(brain, cut_state, drive, np.random.default_rng(11))
    _, cut_conditioned, _ = draw_event(
        brain, cut_state, drive, np.random.default_rng(11), conditional=True
    )
    np.testing.assert_allclose(cut_free[0], cut_conditioned[0], atol=1e-10)
