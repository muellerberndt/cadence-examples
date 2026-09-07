"""Write a chorale continuation with the trained net, drawing pitches from its calibrated probabilities.

Run:  python continue.py [--chords 32] [--opening 0] [--seed 0]     (writes continuation.mid)

The receipt's continuation takes the most active owners above the threshold, which locks
onto held chords; this draws each pitch with its calibrated probability instead, keeps at
most four, and so moves. It reads net.json and is not part of the receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train import midi_bytes

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chords", type=int, default=32)
    parser.add_argument("--opening", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "continuation.mid")
    args = parser.parse_args()
    net = json.loads((HERE / "net.json").read_text())
    n, rule, music = net["n"], net["rule"], net["music"]
    weights = np.asarray(net["W"]).reshape(n, n)
    bias = np.asarray(net["bias"])
    inputs, outputs = net["sets"]["input"], net["sets"]["output"]
    low, pitches, window = music["low"], music["high"] - music["low"] + 1, music["window"]
    rng = np.random.default_rng(args.seed)

    def act(v: np.ndarray) -> np.ndarray:
        r = 1 / (1 + np.exp(-rule["slope"] * (v - rule["threshold"]))) - rule["rest"]
        return np.where(r > 0, r / (1 - rule["rest"]), rule["leak"] * r / rule["rest"])

    def settle(drive: np.ndarray) -> np.ndarray:
        v = np.zeros(n)
        s = np.zeros(n)
        for _ in range(100):
            v = v + rule["dt"] * (-v + s @ weights + drive + bias)
            fresh = act(v)
            moved = np.abs(fresh - s).max()
            s = fresh
            if moved < 1e-4:
                break
        return s

    chords = [list(c) for c in net["meta"]["openings"][args.opening]]
    for _ in range(args.chords):
        drive = np.zeros(n)
        for b, chord in enumerate(chords[-window:]):
            for p in chord:
                if low <= p < low + pitches:
                    drive[inputs[b * pitches + p - low]] = rule["clamp"]
        s = settle(drive)[outputs]
        stuck = len(chords) >= 2 and chords[-1] == chords[-2]  # a held chord: draw more freely so the music moves
        slope = music["slope"] / (3.0 if stuck else 1.0)
        prob = 1 / (1 + np.exp(-slope * (s - music["threshold"])))
        order = np.argsort(-s)
        chosen = [int(k) for k in order if rng.random() < prob[k]][:4] or [int(order[0])]
        chords.append(sorted(low + k for k in chosen))
    args.output.write_bytes(midi_bytes(chords))
    print(f"wrote {args.output} with {len(chords)} chords; the last eight: {chords[-8:]}")


if __name__ == "__main__":
    main()
