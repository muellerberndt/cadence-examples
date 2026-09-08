"""The body and the world of the parrot: a syrinx, a cochlea, and the sounds of a household.

Syrinx: the labial oscillator of Amador and Mindlin (a single-mass model of the songbird
syrinx that the physics-of-birdsong literature uses), ``x'' = -eps x - C x^2 x' + beta x'``,
where the tension ``eps`` of the labia sets the pitch and the air-sac pressure ``beta``
turns phonation on through a Hopf bifurcation: below zero the labia are still, above it
they oscillate. The pressure wave enters a vocal tract modelled as one resonance whose
centre the beak and tongue set. Three muscle commands a frame: tension, pressure, tract.

Cochlea: a bank of band-pass filters on a log scale from 200 Hz to 5 kHz, an envelope per
10 ms frame, log-compressed to levels in [0, 1]: what the auditory owners are clamped to.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

SR = 16000  # samples a second
FRAME = 160  # samples a frame: 10 ms
SUB = 8  # integration sub-steps a sample
CHANNELS = 24
F_LO, F_HI = 200.0, 5000.0
PITCH_LO, PITCH_HI = 300.0, 3500.0  # the syrinx's range, mapped from a tension level in [0, 1] on a log scale
TRACT_LO, TRACT_HI = 800.0, 4000.0
C = 1.0e7  # nonlinear damping of the labial model; with BETA_SCALE it sets the saturation amplitude
BETA_SCALE = 1.0e3  # pressure units to 1/s: weak against the labial stiffness, so pitch stays with tension
GAIN = 50.0  # sound units per unit of labial displacement
MIX = 0.6  # share of the radiated sound that passes through the tract resonance; the rest radiates directly


def pitch_of(tension: float) -> float:
    return PITCH_LO * (PITCH_HI / PITCH_LO) ** float(np.clip(tension, 0.0, 1.0))


def tract_of(level: float) -> float:
    return TRACT_LO * (TRACT_HI / TRACT_LO) ** float(np.clip(level, 0.0, 1.0))


def pressure_of(level: float) -> float:
    """Air-sac pressure: below 0.3 the labia are still (beta < 0), above it they sing."""
    return -0.3 + 1.0 * float(np.clip(level, 0.0, 1.0))


class Syrinx:
    """State of the labia and the tract; renders one frame per motor command."""

    def __init__(self) -> None:
        self.x, self.y = 1e-3, 0.0
        self.tract = np.zeros(2)  # resonator state (two-pole)

    def frame(self, tension: float, pressure: float, tract: float) -> np.ndarray:
        eps = (2 * np.pi * pitch_of(tension)) ** 2
        beta = pressure_of(pressure) * BETA_SCALE  # the Hopf onset takes a few ms
        dt = 1.0 / (SR * SUB)
        out = np.empty(FRAME)
        x, y = self.x, self.y
        for i in range(FRAME):
            for _ in range(SUB):
                y += dt * (-eps * x - C * x * x * y + beta * y)
                x += dt * y
            x = float(np.clip(x, -0.05, 0.05))
            out[i] = x
        self.x, self.y = x, y
        # the tract: one resonance at the beak/tongue setting, Q about 4
        f0 = tract_of(tract)
        r = np.exp(-np.pi * f0 / (4.0 * SR))
        a1, a2 = -2 * r * np.cos(2 * np.pi * f0 / SR), r * r
        z1, z2 = self.tract
        wave = np.empty(FRAME)
        for i in range(FRAME):
            w = out[i] - a1 * z1 - a2 * z2
            wave[i] = (1 - r) * (w - z2)
            z2, z1 = z1, w
        self.tract = np.array([z1, z2])
        return np.clip((MIX * wave + (1 - MIX) * out) * GAIN, -1.0, 1.0)

    def render(self, tension: np.ndarray, pressure: np.ndarray, tract: np.ndarray) -> np.ndarray:
        return np.concatenate([self.frame(a, b, c) for a, b, c in zip(tension, pressure, tract, strict=True)])


class Cochlea:
    """Streaming filterbank: call ``frame`` with 160 samples and read 24 levels in [0, 1]."""

    def __init__(self) -> None:
        edges = np.geomspace(F_LO, F_HI, CHANNELS + 1)
        self.sos = [butter(2, [edges[k], edges[k + 1]], btype="band", fs=SR, output="sos") for k in range(CHANNELS)]
        self.zi = [np.zeros_like(sosfilt_zi(s)) for s in self.sos]
        self.centres = np.sqrt(edges[:-1] * edges[1:])

    def frame(self, wave: np.ndarray) -> np.ndarray:
        levels = np.empty(CHANNELS)
        for k, s in enumerate(self.sos):
            band, self.zi[k] = sosfilt(s, wave, zi=self.zi[k])
            levels[k] = float(np.sqrt(np.mean(band * band)))
        return np.clip((np.log10(levels + 1e-5) + 4.0) / 3.5, 0.0, 1.0)  # 1e-4 .. 0.3 rms onto [0, 1]

    def frames(self, wave: np.ndarray) -> np.ndarray:
        return np.stack([self.frame(wave[s : s + FRAME]) for s in range(0, len(wave) - FRAME + 1, FRAME)])


# ---------------------------------------------------------------------------
# The household: tonal sounds a syrinx could in principle produce


def tone(freqs: np.ndarray, amp: np.ndarray) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(freqs) / SR
    return amp * np.sin(phase)


def envelope(n: int, attack: int = 160, release: int = 480) -> np.ndarray:
    e = np.ones(n)
    e[:attack] = np.linspace(0, 1, attack)
    e[-release:] = np.linspace(1, 0, release)
    return e


def doorbell() -> np.ndarray:
    a = tone(np.full(4000, 880.0), 0.5 * envelope(4000))
    b = tone(np.full(5600, 660.0), 0.5 * envelope(5600))
    return np.concatenate([a, np.zeros(400), b])


def ring() -> np.ndarray:
    n = 9600
    t = np.arange(n) / SR
    burst = tone(np.full(n, 1200.0) + 60 * np.sin(2 * np.pi * 20 * t), 0.45 * envelope(n) * (0.7 + 0.3 * np.sin(2 * np.pi * 20 * t)))
    return np.concatenate([burst, np.zeros(3200), burst])


def beeps() -> np.ndarray:
    beep = tone(np.full(1600, 2000.0), 0.5 * envelope(1600, 80, 160))
    gap = np.zeros(1600)
    return np.concatenate([beep, gap, beep, gap, beep])


def whistle() -> np.ndarray:
    n = 9600
    up = np.linspace(1000.0, 3000.0, n // 2)
    return tone(np.concatenate([up, up[::-1]]), 0.4 * envelope(n))


def siren() -> np.ndarray:
    n = 19200
    t = np.arange(n) / SR
    return tone(1050.0 + 450.0 * np.sin(2 * np.pi * t / 1.2 - np.pi / 2), 0.4 * envelope(n))


SOUNDS = {"doorbell": doorbell, "ring": ring, "beeps": beeps, "whistle": whistle, "siren": siren}
FREQUENT = ("doorbell", "ring", "beeps")  # heard many times a day
RARE = ("whistle", "siren")  # heard a couple of times


def waves() -> dict[str, np.ndarray]:
    out = {}
    for name, make in SOUNDS.items():
        w = make()
        out[name] = np.concatenate([w, np.zeros(FRAME - len(w) % FRAME)]) if len(w) % FRAME else w
    return out
