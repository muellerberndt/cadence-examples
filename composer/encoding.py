"""Compact note-event vocabulary. No hold tokens or silent-step accuracy inflation."""

import numpy as np

HISTORY = 4
SIZES = (61, 12, 25, 49)  # melody C2-C7, duration, chord, bass C1-C5
DURATIONS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
OFFSETS = np.cumsum((0,) + SIZES[:-1])
EVENT_SIZE = sum(SIZES)
EXTRA = 48  # past pitch-class memory12 + beat16 + phrase8 + mode2 + style3 + arousal3 + motif4
INPUTS = HISTORY * EVENT_SIZE + EXTRA


def features(context, extra):
    context = np.asarray(context, dtype=int)
    x = np.zeros((len(context), INPUTS), dtype=np.float32)
    rows = np.arange(len(context))
    for h in range(HISTORY):
        for k, offset in enumerate(OFFSETS):
            x[rows, h * EVENT_SIZE + offset + context[:, h, k]] = 1
    x[:, HISTORY * EVENT_SIZE :] = extra
    return x


def music_features(previous, step, mode, style=0, arousal=1):
    extra = np.zeros(EXTRA, dtype=np.float32)
    for e in previous[-32:]:
        extra[(int(e[0]) + 36) % 12] += 1
    extra[:12] /= max(1, extra[:12].sum())
    extra[12 + int(step) % 16] = 1
    extra[28 + (int(step) // 16) % 8] = 1
    extra[36 + int(mode)] = 1
    extra[38 + int(style)] = 1
    extra[41 + int(arousal)] = 1
    for i, e in enumerate(previous[:4]):
        extra[44 + i] = (int(e[0]) - 30) / 30
    return extra


def chord(notes):
    pcs = np.bincount(np.array(notes, dtype=int) % 12, minlength=12)
    if len(set(np.array(notes) % 12)) < 2:
        return 24
    scores = []
    for minor in (0, 1):
        for root in range(12):
            triad = [root, (root + (3 if minor else 4)) % 12, (root + 7) % 12]
            scores.append(
                sum(pcs[i] for i in triad)
                - 0.5 * sum(pcs[i] for i in range(12) if i not in triad)
            )
    return int(np.argmax(scores))


def tonal_center(pitches, durations):
    hist = np.bincount(np.array(pitches) % 12, weights=durations, minlength=12)
    scores = []
    for minor in (0, 1):
        profile = np.zeros(12)
        profile[[0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11]] = 1
        profile[[0, 7]] += 0.3
        scores.extend(float(hist @ np.roll(profile, k)) for k in range(12))
    idx = int(np.argmax(scores))
    return idx % 12, idx // 12
