"""The conventional control's held-out metric agrees with an analytic uniform model."""

import numpy as np
import torch

from composer.ensemble import EXTRA_RAW, HISTORY, INPUTS, SIZES
from tools.baseline_ensemble import measure


class Rows:
    count = 256

    def rows(self, ids):
        return {
            "context": np.zeros((len(ids), HISTORY, 5), np.uint8),
            "extra": np.zeros((len(ids), EXTRA_RAW), np.uint8),
            "labels": np.stack([ids % width for width in SIZES], axis=1),
        }


def test_uniform_prediction_metric():
    net = torch.nn.Linear(INPUTS, sum(SIZES))
    with torch.no_grad():
        net.weight.zero_()
        net.bias.zero_()
    result = measure(net, Rows(), "cpu", 256)
    np.testing.assert_allclose(result["nll"], np.log(SIZES), rtol=1e-6)
    assert abs(result["mean_nll"] - np.log(SIZES).mean()) < 1e-6
    assert result["examples"] == 256
    assert result["instrument_counts"] == [32] * 8
    assert result["instrument_accuracy"] == [1] + [0] * 7
