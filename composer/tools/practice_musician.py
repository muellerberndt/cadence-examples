"""Practice: the musician composes, listens to what it made, keeps what it judged best and
learns from it, while rehearsing real music so the corpus is not forgotten.

Each round:
  1. compose ``pieces`` short pieces in random moods (bounded detuning for variety);
  2. score each with the Listener (own surprise near the corpus target, mood fit, health);
  3. learn from the pieces whose score is above the round's mean, each stream weighted by
     its advantage (a valence: how much better than the round it was);
  4. rehearse ``rehearse`` updates of real streams (general and focus);
  5. measure held-out surprise (general and focus) and the round's mean score.

Self-imitation weighted by its own judge can only sharpen what the judge measures; the
held-out surprise on real music is the guard against drifting away from the corpus.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.listen import Listener
from composer.musician import MOOD_NAMES, MOOD_WIDTHS, WINDOW, Musician, Senses, describe, primed
from composer.perform import compose
from tools.train_musician import Mixed, Streams, evaluate

ROOT = Path(__file__).resolve().parents[1]


def random_brief(rng):
    return np.array([rng.integers(0, w) for w in MOOD_WIDTHS], dtype=int)


class OwnStreams:
    """The selected pieces walked in lockstep as streams, with their briefs and weights."""

    def __init__(self, pieces, briefs, weights, bars):
        self.pieces, self.briefs, self.weights = pieces, briefs, weights
        self.batch = len(pieces)
        self.position = np.full(self.batch, WINDOW)
        self.senses = [primed(p[:WINDOW], bars * 16) for p in pieces]
        self.fresh = np.ones(self.batch, bool)

    def batch_rows(self):
        at = [min(i, len(p) - 1) for p, i in zip(self.pieces, self.position)]  # ended rows hold their last event
        context = np.stack([np.asarray(p[i - WINDOW : i]) for p, i in zip(self.pieces, at)])
        sense = np.stack([s.raw() for s in self.senses])
        labels = np.stack([np.asarray(p[i]) for p, i in zip(self.pieces, at)])
        return context, sense, np.stack(self.briefs), labels, self.fresh.copy()

    def advance(self):
        for k, (p, s) in enumerate(zip(self.pieces, self.senses)):
            if self.position[k] < len(p):
                s.observe(p[self.position[k]])
        self.position += 1
        self.fresh[:] = False

    def alive(self):
        return np.array([i < len(p) for p, i in zip(self.pieces, self.position)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", default="practice")
    p.add_argument("--dataset", default="musician")
    p.add_argument("--focus", default="musician-focus")
    p.add_argument("--focus-share", type=float, default=0.5)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--pieces", type=int, default=8)
    p.add_argument("--bars", type=int, default=16)
    p.add_argument("--futures", type=int, default=6)
    p.add_argument("--rehearse", type=int, default=300, help="real-stream updates per round")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--detune", type=float, default=0.03)
    p.add_argument("--max-seconds", type=int, default=6 * 3600)
    p.add_argument("--backend", default="torch")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--settle", type=int, default=96, help="settling iterations per event when playing")
    p.add_argument("--own-scale", type=float, default=0.25, help="step scale for learning from own pieces (tiny batches)")
    p.add_argument("--guard", type=float, default=1.03, help="roll a round back when real-music surprise exceeds the round's start × this")
    a = p.parse_args()
    musician = Musician.load(a.checkpoint, backend=a.backend, device=a.device)
    musician.settle_steps = a.settle
    receipt_path = Path(a.checkpoint).with_name("receipt.json")
    target = json.loads(receipt_path.read_text()).get("best_validation_nll", 1.0) if receipt_path.exists() else 1.0
    rng = np.random.default_rng(a.seed)
    folder = ROOT / "runs" / a.name
    folder.mkdir(parents=True, exist_ok=True)
    real = Mixed(
        Streams(ROOT / "data" / a.dataset, "train", a.batch, rng),
        Streams(ROOT / "data" / a.focus, "train", a.batch, np.random.default_rng(a.seed + 1)),
        a.focus_share,
        rng,
    )
    real_state = musician.fresh(a.batch)
    receipt = {
        "checkpoint": a.checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        "brain": describe(musician),
        "target_surprise": target,
        "settings": vars(a),
        "sources": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__), ROOT / "composer/musician.py", ROOT / "composer/perform.py", ROOT / "composer/listen.py"]},
        "rounds": [],
        "claim": "Self-practice weighted by the musician's own listener; the listener is supplied and is not a proof of beauty. Held-out surprise on real music guards against drift.",
    }
    from dataclasses import replace

    started = time.monotonic()
    best = float("inf")
    base_eta = musician.learner.config.eta
    own_scale = a.own_scale
    baseline = evaluate(musician, Streams(ROOT / "data" / a.dataset, "validation", 64, np.random.default_rng(7), cap=256))["mean_nll"]
    receipt["baseline_general_nll"] = baseline
    for round_ in range(a.rounds):
        tick = time.monotonic()
        musician.save(folder / "round-start.npz")
        pieces, briefs, scores = [], [], []
        for k in range(a.pieces):
            brief = random_brief(rng)
            result = compose(musician, brief, bars=a.bars, futures=a.futures, horizon=24, edits=1, seed=int(rng.integers(1 << 30)), tempo=100, target_surprise=target)
            pieces.append(result["prime"] + result["events"])
            briefs.append(brief)
            scores.append(result["final_score"]["score"])
            print(json.dumps({"round": round_, "piece": k, "brief": brief.tolist(), "events": len(result["events"]), "score": result["final_score"]["score"], "seconds": time.monotonic() - tick}), flush=True)
        scores = np.asarray(scores)
        keep = scores >= scores.mean()
        weights = np.clip((scores[keep] - scores.mean()) / (scores.std() + 1e-6), 0.2, 2.0)
        own = OwnStreams([pieces[i] for i in np.flatnonzero(keep)], [briefs[i] for i in np.flatnonzero(keep)], weights, a.bars)
        state = musician.fresh(own.batch)
        own_updates = 0
        musician.learner.config = replace(musician.learner.config, eta=base_eta * own_scale, eta_bias=base_eta * own_scale / 10)
        while own.alive().any():
            alive = own.alive()
            context, sense, mood, labels, fresh = own.batch_rows()
            rows = np.flatnonzero(alive)
            sub = state.copy_rows(rows)
            musician.learn(context[rows], sense[rows], mood[rows], labels[rows], sub, weight=own.weights[rows])
            # carry the learned rows' state back (the other rows have ended)
            if state.trace is not None:
                state.trace.trace[rows] = sub.trace.trace
                state.trace.last[rows] = sub.trace.last
                state.trace.cold[rows] = sub.trace.cold
            if state.piece is not None:
                state.piece.trace[rows] = sub.piece.trace
                state.piece.last[rows] = sub.piece.last
                state.piece.cold[rows] = sub.piece.cold
            if state.memory is not None:
                state.memory.strength[rows] = sub.memory.strength
                state.memory.mass[rows] = sub.memory.mass
            state.bar[rows] = sub.bar
            state.warm = None  # rows of different lengths: start the next free phase from rest
            own.advance()
            own_updates += 1
        musician.learner.config = replace(musician.learner.config, eta=base_eta, eta_bias=base_eta / 10)
        for _ in range(a.rehearse):
            context, sense, mood, labels, fresh, conditioning = real.batch_rows()
            fresh = fresh | real.is_focus & real.focus.fresh | ~real.is_focus & real.general.fresh
            real_state.reset_rows(np.flatnonzero(fresh))
            musician.learn(context, sense, mood, labels, real_state, conditioning=conditioning)
            real.advance()
        general = evaluate(musician, Streams(ROOT / "data" / a.dataset, "validation", 64, np.random.default_rng(7), cap=256))
        focus = evaluate(musician, Streams(ROOT / "data" / a.focus, "validation", 28, np.random.default_rng(7), cap=512), updates=256)
        rolled = False
        if a.guard and general["mean_nll"] > baseline * a.guard:
            # the round hurt real music: back to the round's start, and practise more gently
            settle = musician.settle_steps
            musician = Musician.load(folder / "round-start.npz", backend=a.backend, device=a.device)
            musician.settle_steps = settle
            real_state = musician.fresh(a.batch)
            own_scale *= 0.5
            rolled = True
        else:
            baseline = min(baseline, general["mean_nll"])
        row = {
            "round": round_,
            "rolled_back": rolled,
            "own_scale": own_scale,
            "seconds": time.monotonic() - started,
            "round_seconds": time.monotonic() - tick,
            "scores": scores.tolist(),
            "mean_score": float(scores.mean()),
            "kept": int(keep.sum()),
            "own_updates": own_updates,
            "general_nll": general["mean_nll"],
            "focus_nll": focus["mean_nll"],
            "focus_per_attribute": focus["nll"],
        }
        receipt["rounds"].append(row)
        if focus["mean_nll"] < best:
            best = focus["mean_nll"]
            musician.save(folder / "brain.npz")
        musician.save(folder / "latest.npz")
        (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))
        (folder / f"round-{round_}.json").write_text(json.dumps({"pieces": [[list(map(int, t)) for t in q] for q in pieces], "briefs": [b.tolist() for b in briefs], "scores": scores.tolist()}))
        print(json.dumps(row), flush=True)
        if time.monotonic() - started > a.max_seconds:
            break
    receipt["complete"] = True
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
