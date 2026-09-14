"""The musician's event vocabulary and clock: shared by the brain, the form and the tools."""

from __future__ import annotations

import numpy as np

WINDOW = 16  # heard events in the ear
DURATIONS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
DELTAS = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
SIZES = (73, 12, 13, 8, 8)  # pitch 24..96 key-relative, duration, delta, family, velocity
OFFSETS = np.cumsum((0,) + SIZES[:-1])
EVENT = int(sum(SIZES))
FAMILIES = ("keys", "strings", "brass", "woodwind", "bass", "plucked", "other", "percussion")
PROGRAMS = (0, 48, 60, 73, 32, 24, 80, 0)
BAR = 16  # sixteenth steps per bar
CYCLE = 16  # bars in the version-2 form record's cycle
