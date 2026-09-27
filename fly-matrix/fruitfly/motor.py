"""The muscles: what the wing motor neurons do to the body (declared, with sources).

The body takes six wing controls (per wing a stroke amplitude, a stroke-plane tilt and a
mean stroke shift) and a wingbeat frequency. The power muscles set the amplitude and the
frequency through the thorax's resonance; the steering muscles change the stroke. Each entry
names the physiology it rests on. The activation of a motor neuron group is its mean settled
activation in [0, 1].
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real

__all__ = [
    "GROUPS_PER_SIDE",
    "HOVER",
    "MUSCLES",
    "SETTLED_MOTOR_GROUPS",
    "Muscle",
    "settled_motor",
    "wing_controls",
]


@dataclass(frozen=True)
class Muscle:
    group: str  # population name without the side
    effect: str  # which control and its sign
    source: str


MUSCLES: tuple[Muscle, ...] = (
    Muscle("power", "amplitude and frequency: the asynchronous power muscles set the stroke through the thorax's resonance",
           "Dickinson and Tu 1997 (the function of dipteran flight muscle); Gordon and Dickinson 2006 (power muscle activity scales with aerodynamic power)"),
    Muscle("mn:wing:b1", "amplitude up on that side: b1 fires every stroke and its phase advance raises the stroke amplitude",
           "Heide and Goetz 1996; Tu and Dickinson 1996 (b1 phase and stroke amplitude)"),
    Muscle("mn:wing:b2", "amplitude up and stroke forward on that side during turns", "Heide and Goetz 1996; Lindsay, Sustar and Dickinson 2017"),
    Muscle("mn:wing:b3", "amplitude down on that side", "Lindsay, Sustar and Dickinson 2017 (b3 active with decreasing stroke amplitude)"),
    Muscle("mn:wing:i1", "amplitude down on that side", "Heide and Goetz 1996 (i1 activity with reduced amplitude)"),
    Muscle("mn:wing:i2", "stroke-plane tilt back on that side", "Lindsay, Sustar and Dickinson 2017"),
    Muscle("mn:wing:iii1", "stroke shift forward on that side", "Lindsay, Sustar and Dickinson 2017 (third axillary muscles and the stroke's fore-aft position)"),
    Muscle("mn:wing:iii3", "stroke shift backward and tilt on that side", "Lindsay, Sustar and Dickinson 2017"),
    Muscle("mn:wing:iii4", "stroke shift backward on that side", "Lindsay, Sustar and Dickinson 2017"),
)

GROUPS_PER_SIDE = ("power", "mn:wing:b1", "mn:wing:b2", "mn:wing:b3", "mn:wing:i1", "mn:wing:i2", "mn:wing:iii1", "mn:wing:iii3", "mn:wing:iii4")
HOVER = {"amplitude": 1.0, "tilt": 0.0, "shift": 0.0, "frequency": 200.0}
# Each gain is the largest excursion the hand-written pilot used to recover from the same six
# kicks (tools/reflex_loop.py), so the brain has the control's authority per unit of activation change and no more.
GAIN_AMPLITUDE = 0.02
GAIN_TILT = 0.227  # rad
GAIN_SHIFT = 0.149  # mm


def wing_controls(activation: dict[str, float]) -> dict[str, float]:
    """Legacy hover-assisted control, retained for the historical comparison experiments.

    ``activation`` maps population names with a side (``mn:wing:b1:left`` ...) and ``power:left``
    to values in [0, 1]. Absent groups count as zero. Controls are the hover trim plus the
    declared effects; the body clips them to its ranges.
    """
    g = lambda name: float(activation.get(name, 0.0))
    out: dict[str, float] = {}
    for side in ("left", "right"):
        amplitude = HOVER["amplitude"] + GAIN_AMPLITUDE * (g(f"power:{side}") + g(f"mn:wing:b1:{side}") + g(f"mn:wing:b2:{side}") - g(f"mn:wing:b3:{side}") - g(f"mn:wing:i1:{side}"))
        tilt = HOVER["tilt"] + GAIN_TILT * (g(f"mn:wing:b2:{side}") - g(f"mn:wing:i2:{side}") - 0.5 * g(f"mn:wing:iii3:{side}"))
        shift = HOVER["shift"] + GAIN_SHIFT * (g(f"mn:wing:iii1:{side}") - g(f"mn:wing:iii3:{side}") - g(f"mn:wing:iii4:{side}"))
        out[f"amplitude_{side}"] = amplitude
        out[f"tilt_{side}"] = tilt
        out[f"shift_{side}"] = shift
    out["frequency"] = HOVER["frequency"] * (0.9 + 0.1 * (g("power:left") + g("power:right")))
    return out


SETTLED_MOTOR_GROUPS = tuple(
    f"{group}:{side}" for group in GROUPS_PER_SIDE for side in ("left", "right")
) + ("mn9", "mn:ttm:left", "mn:ttm:right")


def settled_motor(readouts: Mapping[str, float]) -> dict:
    """Absolute motor activity to actuators, with no hover, target or body-state input.

    ``wings`` is [left amplitude, right amplitude, left/right stroke-plane angle
    (radians), left/right stroke shift (metres), shared frequency (Hz)]. Power
    activity is a candidate normalized stroke-force fraction: amplitude is its
    square root, with the existing steering gains modifying that stroke. A
    supplied 200 Hz mechanical carrier operates when either side is powered;
    each silent side has zero amplitude and therefore zero active wing force.
    This assumption is not a fitted biological transfer function or a neural
    oscillator. With no steering, equal activity p gives p times body weight in
    total lift at level rest, rather than applying p to two squared controls.
    Missing populations are silent; finite activities are rectified/clipped to
    [0, 1]. Invalid supplied motor readings are rejected, never replaced by trim.

    MN9 supplies a normalized proboscis extension command. The two TTM outputs
    are normalized extensor activations only: no gait, jump impulse, landing,
    feeding policy or joint geometry is inferred from them. Other muscles still
    need a measured actuator dictionary and mechanics before they can act.
    """
    activity = {}
    for name in SETTLED_MOTOR_GROUPS:
        value = readouts.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"motor readout {name!r} must be a finite number")
        activity[name] = min(1.0, max(0.0, float(value)))

    amplitudes, planes, shifts = [], [], []
    for side in ("left", "right"):
        power = activity[f"power:{side}"]
        b1, b2, b3, i1, i2, iii1, iii3, iii4 = (
            activity[f"mn:wing:{name}:{side}"]
            for name in ("b1", "b2", "b3", "i1", "i2", "iii1", "iii3", "iii4")
        )
        amplitudes.append(min(1.3, max(0.0, sqrt(power) * (1.0 + GAIN_AMPLITUDE * (b1 + b2 - b3 - i1)))))
        planes.append(min(0.5, max(-0.5, GAIN_TILT * (b2 - i2 - 0.5 * iii3))))
        shifts.append(min(0.0005, max(-0.0005, 1e-3 * GAIN_SHIFT * (iii1 - iii3 - iii4))))
    frequency = 200.0 if activity["power:left"] > 0 or activity["power:right"] > 0 else 0.0
    return {
        "wings": amplitudes + planes + shifts + [frequency],
        "proboscis": activity["mn9"],
        "legs": {"ttm_left": activity["mn:ttm:left"], "ttm_right": activity["mn:ttm:right"]},
    }
