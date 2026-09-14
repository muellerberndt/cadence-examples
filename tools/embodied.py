"""Independent Cadence checks for the coupled visual/motor circuit."""

import cadence as cd
import numpy as np


def motor_settling(q, target):
    a, b = q
    tip = np.array(
        [
            0.5 + 0.43 * np.cos(a) + 0.37 * np.cos(a + b),
            0.94 + 0.43 * np.sin(a) + 0.37 * np.sin(a + b),
        ]
    )
    j = np.array(
        [
            [-0.43 * np.sin(a) - 0.37 * np.sin(a + b), -0.37 * np.sin(a + b)],
            [0.43 * np.cos(a) + 0.37 * np.cos(a + b), 0.37 * np.cos(a + b)],
        ]
    )
    edges = []
    for v in range(2):
        for m in range(2):
            edges.extend([(m + 2, v, -j[v, m]), (v, m + 2, 0.9 * j[v, m])])
    pre, post, weights = zip(*edges)
    brain = cd.Brain(
        cd.Connectome.from_synapses(4, pre=pre, post=post, sign=weights),
        cd.NeuronModel(
            gain=1, slope=2, threshold=0, leak=1, dt=0.25, stimulus_amplitude=1
        ),
    )
    drive = np.r_[np.asarray(target) - tip, [0, 0]]
    return brain.settle(drive, steps=80, tolerance=0).activation

