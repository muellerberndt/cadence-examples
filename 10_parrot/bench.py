"""Look at what a parrot sings: for each household sound, the original, the memory's replay to the horizon, and the
imitation, as cochleagrams with their pitch tracks, plus the receipt's numbers and the mirror's sanity checks.

python bench.py                      # net.json -> figures/coco_<sound>.png
python bench.py net_1.json pepper    # the second parrot

This is the instrument that told the memory's blur from the mirror's compression from the body's buzz; it is not
part of the receipt, and it needs matplotlib.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import brain as B  # noqa: E402
import train as T  # noqa: E402
from brain import Brain, cue_of, sharpen  # noqa: E402
from syrinx import CHANNELS, F_HI, F_LO, FRAME, SR, Cochlea, Syrinx, envelope, pitch_of, tone, waves  # noqa: E402

HERE = Path(__file__).resolve().parent


def load(path: Path) -> Brain:
    net = json.loads(path.read_text())
    b = Brain(net["seed"])
    assert net["edges"]["pre"] == b.wiring.pre.tolist(), "the net was made by another wiring"
    b.learner.engine = b.learner.engine.with_parameters(edge_scale=np.array(net["edges"]["scale"]), bias=np.array(net["bias"]))
    b.cue_projection = np.array(net["cue_projection"]).reshape(B.SEQUENCE, -1) / 100.0
    b.clock_projection = np.array(net["clock_projection"]).reshape(B.SEQUENCE, -1) / 100.0
    return b


def pitch_track(g: np.ndarray, loud: float = 0.3) -> np.ndarray:
    return np.where(g.max(axis=1) > loud, g.argmax(axis=1), np.nan)


def main() -> None:
    net_path = HERE / (sys.argv[1] if len(sys.argv) > 1 else "net.json")
    label = sys.argv[2] if len(sys.argv) > 2 else "coco"
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    b = load(net_path)
    grams = {n: Cochlea().frames(x) for n, x in waves().items()}
    rows = []
    for name, fr in grams.items():
        rec, _ = T.recall(b, fr)
        full = b.replay(cue_of(fr), T.REPLAY_FRAMES)
        sound, cmds = T.imitate(b, fr)
        prod = Cochlea().frames(sound) if len(sound) >= FRAME else np.zeros((1, CHANNELS))
        pitch, dist, cov = T.pitch_similarity(fr, prod)
        shape, rhythm, sim = T.shape_similarity(fr, prod), T.rhythm_similarity(fr, prod), T.similarity(fr, prod)
        rows.append((rec, pitch, dist, shape, rhythm, sim))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
        panels = ((f"{name}: original ({len(fr)} frames)", fr), (f"memory replay to the horizon (recall {rec:.2f})", full), (f"imitation (pitch r {pitch:.2f}, {dist:.1f} ch off, rhythm {rhythm:.2f}, {len(prod)} frames)", prod))
        for ax, (title, g) in zip(axes, panels, strict=True):
            ax.imshow(g.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1, extent=[0, len(g) * FRAME / SR, 0, CHANNELS])
            ax.plot(np.arange(len(g)) * FRAME / SR + 0.005, pitch_track(g), color="cyan", lw=1.2)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("s")
        axes[0].set_ylabel("cochlear channel")
        fig.tight_layout()
        fig.savefig(out / f"{label}_{name}.png", dpi=110)
        plt.close(fig)
        tens = [c["tension"] for c in cmds] or [0.0]
        print(f"{name:9s} recall {rec:6.3f} | imitation: similarity {sim:6.3f}, pitch r {pitch:6.3f} ({dist:4.1f} channels off on {cov:4.0%} of the loud frames), shape {shape:6.3f}, rhythm {rhythm:6.3f}, {len(prod):3d} frames for {len(fr):3d}; tension {min(tens):.2f}-{max(tens):.2f}", flush=True)
    m = np.mean(rows, axis=0)
    print(f"MEAN recall {m[0]:.3f}, pitch r {m[1]:.3f}, pitch distance {m[2]:.2f} channels, shape {m[3]:.3f}, rhythm {m[4]:.3f}, similarity {m[5]:.3f}")
    edges = np.geomspace(F_LO, F_HI, CHANNELS + 1)

    def channel_of(f: float) -> int:
        return int(np.clip(np.searchsorted(edges, f) - 1, 0, CHANNELS - 1))

    def peak_frame(c: int) -> np.ndarray:
        f = np.zeros((B.MIRROR_WINDOW, CHANNELS))
        f[:, c] = 0.8
        return f

    print("mirror on a lone peak, channel -> channel of the pitch it commands:", " ".join(f"{c}->{channel_of(pitch_of(b.motor(peak_frame(c)[None])['tension'][0]))}" for c in range(4, 22, 2)))
    print("mirror on pure tones:", " ".join(f"{f} Hz -> {pitch_of(b.motor(Brain.windows_of(Cochlea().frames(tone(np.full(4800, float(f)), 0.45 * envelope(4800))))[-1:])['tension'][0]):.0f} Hz" for f in (400, 600, 900, 1300, 1900, 2800)))
    print("mirror on its own voice, tension -> tension:", " ".join(f"{tn} -> {b.motor(Brain.windows_of(Cochlea().frames(Syrinx().render(np.full(30, tn), np.full(30, 0.8), np.full(30, 0.5))))[-1:])['tension'][0]:.2f}" for tn in (0.1, 0.3, 0.5, 0.7, 0.9)), f"| quiet -> pressure {float(b.motor(np.zeros((1, B.MIRROR_WINDOW, CHANNELS)))['pressure'][0]):.2f}")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
