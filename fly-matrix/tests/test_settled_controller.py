"""Causal controls for the sensor -> equilibrium -> motor -> force boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from cadence import Brain, Connectome, NeuronModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fruitfly.body import CONTROL_KEYS, DT, GRAVITY, Flight
from fruitfly.motor import (
    SETTLED_MOTOR_GROUPS,
    settled_motor,
    wing_controls,
)
from fruitfly.settled_controller import SettledController


def brain(*, disconnected=False, dt=0.5):
    connectome = Connectome.from_synapses(
        6, pre=[0, 0, 0, 0, 0], post=[1, 2, 3, 4, 5], count=[2] * 5,
        populations={"sensor": (0,), "power:left": (1,), "power:right": (2,),
                     "mn:wing:b2:left": (3,), "mn9": (4,), "mn:ttm:left": (5,),
                     "motor": (1, 2, 3, 4, 5)},
    )
    return Brain(connectome, NeuronModel(gain=1, threshold=0, dt=dt, stimulus_amplitude=2),
                 efficacy=np.zeros(5) if disconnected else None)


def controller(model=None, **kwargs):
    return SettledController(brain() if model is None else model,
                             sensory_populations=["sensor"], tolerance=1e-10, **kwargs)


def test_silent_neurons_supply_no_active_power_or_auxiliary_motor_command():
    zero = settled_motor({})
    assert zero == {"wings": [0.] * 7, "proboscis": 0., "legs": {"ttm_left": 0., "ttm_right": 0.}}
    assert settled_motor(dict.fromkeys(SETTLED_MOTOR_GROUPS, 0)) == zero
    # Preserve the historical comparator, and expose why it is unsuitable here.
    assert wing_controls({})["amplitude_left"] == 1
    assert wing_controls({})["frequency"] == 180


def test_steering_neurons_cannot_generate_wing_power_without_power_neurons():
    readings = {name: 1. for name in SETTLED_MOTOR_GROUPS if name.startswith("mn:wing:")}
    output = settled_motor(readings)["wings"]
    assert output[0] == output[1] == output[6] == 0


def test_one_side_motor_activation_changes_force_and_roll_through_mechanics():
    silent, active = Flight((2., 1.5, 1.2), 0.), Flight((2., 1.5, 1.2), 0.)
    silent.step(DT, dict(zip(CONTROL_KEYS, settled_motor({})["wings"])))
    active.step(DT, dict(zip(CONTROL_KEYS, settled_motor({"power:left": 1})["wings"])))
    assert active.v[2] > silent.v[2]
    assert active.w[0] > 0
    assert silent.w[0] == 0
    assert silent.v[2] < 0  # Gravity remains even when motor power is absent.


@pytest.mark.parametrize("power", [0, .001, .25, .5, 1])
def test_normalized_power_has_linear_initial_lift_not_fourth_power(power):
    output = settled_motor({"power:left": power, "power:right": power})["wings"]
    flight = Flight((2., 1.5, 1.2), 0.)
    flight.step(DT, dict(zip(CONTROL_KEYS, output)))
    # Recover force from independently implemented body mechanics at zero speed.
    lift_over_weight = 1 + flight.v[2] / (DT * GRAVITY)
    assert lift_over_weight == pytest.approx(power, abs=1e-14)
    assert output[6] == (200 if power > 0 else 0)


def test_single_powered_side_does_not_drive_the_silent_wing():
    output = settled_motor({"power:left": .5, "power:right": 0})["wings"]
    flight = Flight((2., 1.5, 1.2), 0.)
    flight.step(DT, dict(zip(CONTROL_KEYS, output)))
    assert output[1] == 0 and output[6] == 200
    assert 1 + flight.v[2] / (DT * GRAVITY) == pytest.approx(.25, abs=1e-14)
    assert flight.w[0] > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, "1", None])
def test_invalid_motor_activity_is_rejected(bad):
    with pytest.raises(ValueError, match="finite number"):
        settled_motor({"power:left": bad})


def test_motor_decoder_has_no_body_or_goal_dependence():
    reads = {"power:left": .4, "power:right": .6, "mn:wing:b2:left": .7, "mn9": .3}
    expected = settled_motor(reads)
    assert settled_motor({**reads, "target": object(), "heading": float("nan"), "state": object()}) == expected
    assert expected["proboscis"] == .3
    assert settled_motor({"power:left": -1, "mn9": 2})["proboscis"] == 1


def test_sensor_to_motor_authority_requires_synaptic_transport():
    intact = controller().step({"sensor": 1})
    cut = controller(brain(disconnected=True)).step({"sensor": 1})
    assert intact.converged and cut.converged
    assert intact.residual <= 1e-10 and cut.residual <= 1e-10
    assert intact.actuators["wings"][0] > 0 and intact.actuators["proboscis"] > 0
    assert cut.actuators == settled_motor({})


def test_no_sensory_drive_has_no_motor_authority_on_unbiased_circuit():
    model = controller()
    driven = model.step({"sensor": 1})
    assert driven.converged
    # Decay of an old neural state is settled, rather than replaying the old command.
    quiet = model.step({})
    assert quiet.converged
    # A tiny residual can leave positive activation and the fixed carrier on;
    # only the resulting amplitude/force is required to approach zero.
    assert max(quiet.actuators["wings"][:2]) < 1e-4
    passive_limit = Flight((2., 1.5, 1.2), 0.)
    passive_limit.step(DT, dict(zip(CONTROL_KEYS, quiet.actuators["wings"])))
    assert abs(1 + passive_limit.v[2] / (DT * GRAVITY)) < 1e-8
    model.reset()
    assert model.step({}).actuators == settled_motor({})


def test_motor_lesion_is_used_by_settle_and_equation_residual():
    mask = np.array([1, 0, 0, 0, 0, 0])
    result = controller().step({"sensor": 1}, mask=mask)
    assert result.converged and result.residual <= 1e-10
    assert result.actuators == settled_motor({})


def test_small_movement_is_not_an_equation_certificate():
    model = brain(dt=1e-12)
    drive = model.stimulus_vector({0: 1})
    one = model.settle(drive, steps=1)
    assert np.max(abs(one.activation)) < 1e-5
    assert model.residual(drive, one)[0] > 1
    result = controller(model, budget=1, chunk=1).step({"sensor": 1})
    assert not result.converged and result.reason == "residual_above_tolerance"
    assert result.steps == 1 and result.residual > 1
    assert result.readouts == {} and result.actuators == settled_motor({})


def test_new_drive_invalidates_old_equilibrium_even_when_budget_is_zero():
    model = controller()
    assert model.step({"sensor": 1}).converged
    model.budget = 0
    capped = model.step({})
    assert not capped.converged and capped.steps == 0
    assert capped.actuators == settled_motor({})


@pytest.mark.parametrize("levels", [{"sensor": float("nan")}, {"sensor": -1}, {"sensor": 2}, {"power:left": 1}, {"target": 1}])
def test_bad_sensory_input_cannot_reuse_previous_motor_output(levels):
    model = controller()
    assert model.step({"sensor": 1}).actuators["wings"][0] > 0
    result = model.step(levels)
    assert not result.converged and result.reason == "invalid_sensory_input"
    assert result.actuators == settled_motor({}) and model.state is None


def test_ports_cannot_directly_drive_annotated_motor_neurons():
    with pytest.raises(ValueError, match="overlaps motor"):
        SettledController(brain(), sensory_populations=["power:left"])
    with pytest.raises(ValueError, match="unknown or empty"):
        SettledController(brain(), sensory_populations=["missing"])
