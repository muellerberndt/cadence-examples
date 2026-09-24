"""The senses: what the room does to the afferents (declared, with sources).

The afferent neurons are in the volume; their organs are not. Each channel below turns a
physical quantity of the body or the room into a drive level in [0, 1] on a named population,
and says where the convention comes from. Everything downstream of these populations is the
measured wiring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Channel", "CHANNELS", "HALTERE_TONE", "haltere_tone", "haltere_drive", "ocellar_drive", "ocelli_lr", "antenna_drive", "optic_flow_drive", "sense"]


@dataclass(frozen=True)
class Channel:
    id: str
    sets: tuple[str, ...]
    quantity: str
    source: str


CHANNELS: tuple[Channel, ...] = (
    Channel("haltere_left", ("haltere:left",), "the flight tone of the left haltere: its afferents fire once per stroke while the fly flies, and a rotation shifts their timing, which a rate model cannot carry; the drive is a constant HALTERE_TONE in flight and zero on the ground",
            "Fox and Daniel 2008; Yarger and Fox 2016 (haltere afferents are phase-locked to the stroke and encode rotation in spike timing); Nalbach 1993 (the Coriolis mechanics)"),
    Channel("haltere_right", ("haltere:right",), "the same on the right haltere", "as haltere_left"),
    Channel("ocelli_left", ("ocelli:left",), "brightness the left ocellus sees: it looks up and to the left, so the sky fills it when the fly is level and the floor when it rolls left or pitches down",
            "Stange 1981; Krapp 2009 (ocelli as horizon detectors feeding the ocellar projection neurons); the lateral ocelli are lateralized onto the OCG neurons of their side"),
    Channel("ocelli_right", ("ocelli:right",), "the same for the right ocellus", "as ocelli_left"),
    Channel("antenna", ("jo:C:left", "jo:C:right", "jo:E:left", "jo:E:right"), "airspeed through the deflection of the antennae onto the Johnston's organ afferents",
            "Fuller et al. 2014 (antennal mechanosensation of airspeed in flight; JO-C/E neurons respond to sustained deflection)"),
    Channel("optic_flow_left", ("lptc:hs:left", "lptc:vs:left"), "rotational optic flow as the renderer computes it, injected where HS and VS receive it",
            "Krapp and Hengstenberg 1996 (HS: yaw and horizontal flow; VS: roll and pitch); declared convention, the optic lobe is bypassed"),
    Channel("optic_flow_right", ("lptc:hs:right", "lptc:vs:right"), "the same on the right", "as optic_flow_left"),
)

HALTERE_TONE = 0.5  # the afferent drive of a beating haltere, rotation or not
STROKE_PLANE_TILT = math.radians(30.0)  # the haltere's stroke plane is tilted back from vertical
HALTERE_SATURATION = 20.0  # rad/s: the rotation rate at which the afferent drive saturates
OCELLI_GAIN = 2.0  # drive per unit of normalised brightness asymmetry
OCELLUS_ELEVATION = math.radians(45.0)  # each lateral ocellus looks this far above the body axis
OCELLUS_AZIMUTH = math.radians(35.0)  # and this far to its side
AIRSPEED_SATURATION = 1.0  # m/s


def haltere_tone(flying: bool = True) -> tuple[float, float]:
    """The standing haltere drive of flight, the same on both sides: the rate model's haltere."""
    return (HALTERE_TONE, HALTERE_TONE) if flying else (0.0, 0.0)


def haltere_drive(omega_body: np.ndarray) -> tuple[float, float]:
    """Left and right Coriolis load magnitude in [0, 1] from the body angular velocity (roll, pitch, yaw).

    Kept for the page's instrument strip; the brain receives ``haltere_tone`` because a rate
    model cannot carry the timing shift that encodes the rotation's direction.

    The Coriolis force on a haltere beating in a plane tilted by STROKE_PLANE_TILT from the
    vertical has a component from pitch shared by both sides and a component from roll of
    opposite sign on the two sides; yaw enters both at twice the stroke frequency and is added
    as a magnitude. The drive is the magnitude of that force, clipped at HALTERE_SATURATION.
    """
    roll, pitch, yaw = float(omega_body[0]), float(omega_body[1]), float(omega_body[2])
    c, s = math.cos(STROKE_PLANE_TILT), math.sin(STROKE_PLANE_TILT)
    left = abs(pitch * c + roll * s) + 0.5 * abs(yaw)
    right = abs(pitch * c - roll * s) + 0.5 * abs(yaw)
    return min(1.0, left / HALTERE_SATURATION), min(1.0, right / HALTERE_SATURATION)


def ocellar_drive(up_brightness: float, down_brightness: float) -> float:
    """Drive in [0, 1] from the sky-versus-ground brightness asymmetry the ocelli see."""
    total = up_brightness + down_brightness
    if total <= 0:
        return 0.0
    return min(1.0, max(0.0, OCELLI_GAIN * (up_brightness - down_brightness) / total))


def ocelli_lr(rotation: tuple[float, ...]) -> tuple[float, float]:
    """Left and right ocellar drive in [0, 1] from the body-to-world rotation matrix (row major).

    Each lateral ocellus looks along a fixed body direction; the sky is bright above the horizon
    and the floor dark below it, so the drive is the upward component of that direction in the
    world, clipped to [0, 1]. Level flight gives both ocelli the same drive; a roll brightens the
    ocellus on the rising side and darkens the other; a pitch down darkens both.
    """
    ce, se = math.cos(OCELLUS_ELEVATION), math.sin(OCELLUS_ELEVATION)
    ca, sa = math.cos(OCELLUS_AZIMUTH), math.sin(OCELLUS_AZIMUTH)
    out = []
    for side in (1.0, -1.0):
        dx, dy, dz = ce * ca, side * ce * sa, se  # the ocellus direction in the body frame
        up = rotation[6] * dx + rotation[7] * dy + rotation[8] * dz  # its world z component
        out.append(min(1.0, max(0.0, up)))
    return out[0], out[1]


def antenna_drive(airspeed: float) -> float:
    return min(1.0, max(0.0, airspeed / AIRSPEED_SATURATION))


FLOW_REST = 0.5  # the resting level of a graded lobula plate cell; flow in its preferred direction raises it, opposite flow lowers it
FLOW_SATURATION = 20.0  # rad/s


def optic_flow_drive(yaw_rate: float, roll_rate: float, pitch_rate: float, side: str) -> float:
    """HS and VS drive from the rotational optic flow the renderer measures.

    HS and VS are graded cells with a resting potential; the rate model gives them a resting
    level FLOW_REST and moves it with the flow. An HS cell prefers front-to-back flow on its
    side, which a yaw toward the other side produces; a VS cell prefers downward flow, which a
    roll toward its side and a nose-up pitch produce. Both cells of a side share one drive here.
    """
    sign = 1.0 if side == "left" else -1.0
    hs = sign * yaw_rate
    vs = sign * roll_rate - pitch_rate
    level = FLOW_REST + 0.5 * (hs + vs) / FLOW_SATURATION
    return min(1.0, max(0.0, level))


def sense(state: dict) -> dict[str, float]:
    """All channel drives from a body state dict: omega (3,), up/down brightness, airspeed, flow rates."""
    l, r = haltere_tone(bool(state.get("flying", True)))
    ol, orr = ocelli_lr(state["rotation"]) if "rotation" in state else (ocellar_drive(float(state.get("up", 1.0)), float(state.get("down", 1.0))),) * 2
    return {
        "haltere_left": l,
        "haltere_right": r,
        "ocelli_left": ol,
        "ocelli_right": orr,
        "antenna": antenna_drive(float(state.get("airspeed", 0.0))),
        "optic_flow_left": optic_flow_drive(*(float(x) for x in state.get("flow", (0.0, 0.0, 0.0))), "left"),
        "optic_flow_right": optic_flow_drive(*(float(x) for x in state.get("flow", (0.0, 0.0, 0.0))), "right"),
    }
