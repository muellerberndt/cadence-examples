"""Imitation of whole pieces as streams: the musician hears each piece event by event,
carries its working memory and form record along, and learns the next event.

One GPU per run. Every update advances every stream by one event: one free phase from
the previous equilibrium, two nudged phases, one local update. No autograd graph or
backpropagation is constructed. A stream whose piece ended starts the next piece from
a reset working memory, form record and warm state.

Design version 3 (``--version 3``) adds the plan: every row also hears the profile of
the bar it is in (absent for a ``--plan-dropout`` share of the rows) and the profiles of
the eight bars before, the theme record is read at the bar's measured return lag, and the
labels carry the eight profile classes for the plan head. The prepared dataset must have
``bars.npy``, ``bar_offsets.npy`` and ``event_bar.npy`` (``tools/prepare_musician.py``).
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.form import PLAN_WIDTHS, history_of
from composer.musician import SIZES, WINDOW, Conditioning, Design, build, describe

ROOT = Path(__file__).resolve().parents[1]


class Streams:
    """Pieces of one split, walked in parallel: each row owns one piece and one position."""

    def __init__(self, path, split, batch, rng, *, cap=1024, plan_dropout=0.0):
        folder = Path(path) / split
        self.tokens = np.load(folder / "tokens.npy", mmap_mode="r")
        self.sense = np.load(folder / "sense.npy", mmap_mode="r")
        self.mood = np.load(folder / "mood.npy")
        self.offsets = np.load(folder / "offsets.npy")
        self.planned = (folder / "bars.npy").exists()
        if self.planned:
            self.bars = np.load(folder / "bars.npy", mmap_mode="r")
            self.bar_offsets = np.load(folder / "bar_offsets.npy")
            self.event_bar = np.load(folder / "event_bar.npy", mmap_mode="r")
        self.plan_dropout = plan_dropout
        self.pieces = len(self.offsets) - 1
        self.rng = rng
        self.cap = cap
        self.batch = batch
        self.order = rng.permutation(self.pieces)
        self.cursor = 0
        self.piece = np.zeros(batch, np.int64)
        self.position = np.zeros(batch, np.int64)
        self.end = np.zeros(batch, np.int64)
        self.fresh = np.ones(batch, bool)
        self.events_seen = 0
        self.pieces_seen = 0
        for row in range(batch):
            self._start(row)

    def _next_piece(self):
        if self.cursor >= len(self.order):
            self.order = self.rng.permutation(self.pieces)
            self.cursor = 0
        piece = int(self.order[self.cursor])
        self.cursor += 1
        return piece

    def _start(self, row):
        while True:
            piece = self._next_piece()
            length = int(self.offsets[piece + 1] - self.offsets[piece])
            if length > WINDOW + 1:
                break
        self.piece[row] = piece
        self.position[row] = self.offsets[piece] + WINDOW
        self.end[row] = min(self.offsets[piece + 1], self.offsets[piece] + self.cap)
        self.fresh[row] = True
        self.pieces_seen += 1

    def random_rows(self):
        """Independent samples instead of streams: random pieces and positions, every row fresh."""
        piece = self.rng.integers(self.pieces, size=self.batch)
        length = self.offsets[piece + 1] - self.offsets[piece]
        piece = np.where(length > WINDOW + 1, piece, self.piece)
        length = self.offsets[piece + 1] - self.offsets[piece]
        self.piece = piece
        self.position = self.offsets[piece] + WINDOW + (self.rng.random(self.batch) * (length - WINDOW)).astype(np.int64)
        self.fresh[:] = True
        return self.batch_rows()

    def batch_rows(self):
        """The current event of every row: context, sense, mood, label, which rows just
        started a new piece (their stream state must be reset before this event), and the
        version-3 conditioning (``None`` for a dataset without bar profiles)."""
        fresh = self.fresh.copy()
        # a row a mixer has handed to another pool keeps a stale position here; clamp it
        at = np.minimum(self.position, len(self.tokens) - 1)
        context = np.stack([self.tokens[p - WINDOW : p] for p in at]).astype(np.int64)
        sense = self.sense[at]
        mood = self.mood[self.piece]
        labels = self.tokens[at].astype(np.int64)
        conditioning = None
        if self.planned:
            bar = self.event_bar[at].astype(int)
            plans, histories, lags = [], [], []
            for row, (piece, b) in enumerate(zip(self.piece, bar)):
                table = self.bars[self.bar_offsets[piece] : self.bar_offsets[piece + 1]]
                b = min(int(b), len(table) - 1)
                bar[row] = b
                plans.append(table[b])
                histories.append(history_of(table, b))
                lags.append(int(table[b, 7]))
            plans = np.array(plans, dtype=int)
            labels = np.concatenate([labels, plans], axis=1)
            shown = plans.copy()
            if self.plan_dropout:
                shown[self.rng.random(self.batch) < self.plan_dropout] = -1
            conditioning = Conditioning(shown, np.array(histories), bar, np.array(lags))
        return context, np.asarray(sense), mood, labels, fresh, conditioning

    def advance(self):
        self.fresh[:] = False
        self.position += 1
        self.events_seen += self.batch
        for row in np.flatnonzero(self.position >= self.end):
            self._start(row)


def _replace_config(config, **changes):
    from dataclasses import replace

    return replace(config, **changes)


class Mixed:
    """Two pools of streams, general and focus: a row that starts a new piece draws it from
    the focus pool with probability ``share``. Each pool keeps its own row arrays; a focus row
    reads from the focus pool through the mask ``is_focus``."""

    def __init__(self, general, focus, share, rng):
        self.general, self.focus, self.share, self.rng = general, focus, share, rng
        self.batch = general.batch
        self.planned = general.planned and focus.planned
        self.is_focus = np.zeros(self.batch, bool)
        self.events_seen = 0
        self.pieces_seen = 0
        for row in range(self.batch):
            self._restart(row)

    def _restart(self, row):
        self.is_focus[row] = self.rng.random() < self.share
        pool = self.focus if self.is_focus[row] else self.general
        pool._start(row)
        self.pieces_seen += 1

    def batch_rows(self):
        g = self.general.batch_rows()
        f = self.focus.batch_rows()
        m = self.is_focus
        out = []
        for a, b in zip(g[:5], f[:5]):
            c = np.array(a)
            c[m] = np.asarray(b)[m]
            out.append(c)
        conditioning = None
        if g[5] is not None and f[5] is not None:
            rows = np.flatnonzero(m)
            conditioning = g[5]
            for name in ("plan", "bars", "bar", "lag"):
                mine = getattr(conditioning, name)
                mine[rows] = getattr(f[5], name)[rows]
        return (*out, conditioning)

    def advance(self):
        for pool, mine in ((self.general, ~self.is_focus), (self.focus, self.is_focus)):
            pool.fresh[:] = False
            pool.position[mine] += 1  # a row advances only in the pool that owns it
        self.events_seen += self.batch
        for row in range(self.batch):
            pool = self.focus if self.is_focus[row] else self.general
            if pool.position[row] >= pool.end[row]:
                self._restart(row)


def evaluate(musician, streams, *, updates=192):
    """Held-out streams heard from their start without learning: mean surprise per slot
    over every event after the first ``WINDOW`` of each piece. ``mean_nll`` covers the five
    event slots (comparable across designs); ``plan_nll`` the eight plan classes (version
    3, with the plan input absent, as when composing)."""
    state = musician.fresh(streams.batch)
    slots = len(musician.slots)
    losses = np.zeros(slots)
    correct = np.zeros(slots)
    count = 0
    for _ in range(updates):
        context, sense, mood, labels, fresh, conditioning = streams.batch_rows()
        if conditioning is not None:
            conditioning.plan[:] = -1
        state.reset_rows(np.flatnonzero(fresh))
        probabilities, free = musician.predict(context, sense, mood, state, conditioning)
        for k, p in enumerate(probabilities):
            if k >= labels.shape[1]:
                break
            losses[k] -= np.log(p[np.arange(len(labels)), labels[:, k]].clip(1e-12)).sum()
            correct[k] += (p.argmax(1) == labels[:, k]).sum()
        count += len(labels)
        musician.advance(state, free, sense, conditioning)
        streams.advance()
    out = {
        "nll": (losses / count).tolist(),
        "mean_nll": float(losses[: len(SIZES)].sum() / count / len(SIZES)),
        "accuracy": (correct / count).tolist(),
        "events": int(count),
    }
    if slots > len(SIZES):
        out["plan_nll"] = float(losses[len(SIZES) :].sum() / count / len(PLAN_WIDTHS))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="musician")
    p.add_argument("--name", default="musician")
    p.add_argument("--cortex", type=int, default=2048)
    p.add_argument("--phrase", type=int, default=2048)
    p.add_argument("--embedding", type=int, default=64)
    p.add_argument("--no-working-memory", action="store_true")
    p.add_argument("--no-form-memory", action="store_true")
    p.add_argument("--no-belt-feedback", action="store_true")
    p.add_argument("--no-belt", action="store_true", help="the ear projects straight into the cortices")
    p.add_argument("--no-warm", action="store_true", help="start every free phase from rest")
    p.add_argument("--iid", action="store_true", help="independent random events instead of streams (a control)")
    p.add_argument("--eta", type=float, default=0.005)
    p.add_argument("--tonic", type=float, default=1.0)
    p.add_argument("--eta-final", type=float, default=None, help="linear decay of eta to this value at the last update")
    p.add_argument("--free-steps", type=int, default=32)
    p.add_argument("--nudged-steps", type=int, default=16)
    p.add_argument("--settle", type=int, default=None, help="settling budget of the held-out evaluation (default: free steps)")
    p.add_argument("--decay", type=float, default=1e-6, help="per-update shrink of every plastic synapse and bias")
    p.add_argument("--normalize", type=float, default=0.98, help="RMS normalisation of the step; 0 for plain momentum steps")
    p.add_argument("--rollback", type=float, default=1.15, help="restore the best checkpoint and halve the step when held-out surprise exceeds best × this (0: off)")
    p.add_argument("--version", type=int, default=1, help="2: interval sense, slow piece trace, form cortex; 3: the plan and the theme record")
    p.add_argument("--piece-decay", type=float, default=None, help="decay of the slow piece trace (default 0.98; 0.995 for version 3)")
    p.add_argument("--plan-dropout", type=float, default=0.5, help="version 3: share of rows that learn without the plan input")
    p.add_argument("--updates", type=int, default=30000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--evaluate-every", type=int, default=1000)
    p.add_argument("--max-seconds", type=int, default=6 * 3600)
    p.add_argument("--backend", default="torch")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--resume", help="a checkpoint to continue from")
    p.add_argument("--focus", help="a second prepared dataset mixed into training (e.g. musician-focus)")
    p.add_argument("--focus-share", type=float, default=0.3, help="probability that a new piece comes from the focus set")
    a = p.parse_args()
    if a.backend == "torch":
        import torch

        torch.set_num_threads(4)
    piece_decay = a.piece_decay if a.piece_decay is not None else (0.995 if a.version >= 3 else 0.98)
    design = Design(
        a.cortex,
        a.phrase,
        a.embedding,
        a.seed,
        working_memory=not a.no_working_memory,
        form_memory=not a.no_form_memory,
        belt=not a.no_belt,
        belt_feedback=not a.no_belt_feedback,
        eta=a.eta,
        tonic=a.tonic,
        version=a.version,
        free_steps=a.free_steps,
        nudged_steps=a.nudged_steps,
        decay=a.decay,
        normalize=a.normalize,
        piece_decay=piece_decay,
        plan_dropout=a.plan_dropout,
    )
    from composer.musician import Musician

    if a.resume:
        musician = Musician.load(a.resume, backend=a.backend, device=a.device)
        musician.learner.config = _replace_config(musician.learner.config, decay=a.decay, free_steps=a.free_steps, nudged_steps=a.nudged_steps)
    else:
        musician = build(design, backend=a.backend, device=a.device)
    if a.settle:
        musician.settle_steps = a.settle
    dropout = musician.design.plan_dropout if musician.planned else 0.0
    folder = ROOT / "runs" / a.name
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    training = Streams(ROOT / "data" / a.dataset, "train", a.batch, rng, plan_dropout=dropout)
    if musician.planned and not training.planned:
        raise SystemExit("design version 3 needs a dataset prepared with bar profiles (tools/prepare_musician.py)")
    if a.focus:
        training = Mixed(training, Streams(ROOT / "data" / a.focus, "train", a.batch, np.random.default_rng(a.seed + 1), plan_dropout=dropout), a.focus_share, rng)

    def held_out(dataset, batch, cap, seed=7):
        return Streams(ROOT / "data" / dataset, "validation", batch, np.random.default_rng(seed), cap=cap)

    receipt = {
        "design": asdict(musician.design),
        "brain": describe(musician),
        "config": musician.learner.config.to_dict(),
        "dataset": a.dataset,
        "dataset_manifest_sha256": hashlib.sha256(
            (ROOT / "data" / a.dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "batch_streams": a.batch,
        "focus": a.focus,
        "focus_share": a.focus_share if a.focus else 0.0,
        "resume": a.resume,
        "device": a.device,
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), ROOT / "composer/musician.py", ROOT / "composer/form.py", ROOT / "tools/prepare_musician.py"]
        },
        "objective": (
            "Next-event imitation of whole pieces as streams: pitch, duration, time to next "
            "attack, instrument family and velocity"
            + ("; and the eight profile classes of the bar being written (the plan head)" if musician.planned else "")
            + ". Held-out surprise is not a musical-quality score."
        ),
        "history": [],
    }
    state = musician.fresh(a.batch)
    started = time.monotonic()
    best = float("inf")
    window = []
    training_seconds = 0.0
    from dataclasses import replace as _replace

    activity = []
    scale = 1.0  # halved at every rollback
    rollbacks = []
    for update in range(a.updates):
        tick = time.monotonic()
        if a.eta_final is not None:
            eta = (a.eta + (a.eta_final - a.eta) * update / max(1, a.updates - 1)) * scale
            musician.learner.config = _replace(musician.learner.config, eta=eta, eta_bias=eta / 10)
        context, sense, mood, labels, fresh, conditioning = training.random_rows() if a.iid else training.batch_rows()
        if a.focus:
            fresh = np.where(training.is_focus, training.focus.fresh, training.general.fresh)
        state.reset_rows(np.flatnonzero(fresh))
        learned, report = musician.learn(context, sense, mood, labels, state, warm=not a.no_warm, conditioning=conditioning)
        if update % 50 == 0:
            act = np.atleast_2d(learned.free.activation)
            cortex = act[:, list(musician.populations["phrase"])]
            out = act[:, musician.output_index]
            activity.append((float(cortex.mean()), float((cortex > 0.95).mean()), float((cortex <= 0).mean()), float(out.std(1).mean()), float(np.abs(musician.brain.efficacy).mean()) if update % 1000 == 0 else float("nan")))
        if not a.iid:
            training.advance()
        else:
            training.events_seen += a.batch
        training_seconds += time.monotonic() - tick
        window.append(report.get("free_steps", 0))
        if (update + 1) % a.evaluate_every == 0 or update == a.updates - 1:
            measured = evaluate(musician, held_out(a.dataset, 64, 256))
            if a.focus:
                measured["focus"] = evaluate(musician, held_out(a.focus, 28, 512), updates=256)
            row = {
                "update": update + 1,
                "seconds": time.monotonic() - started,
                "training_seconds": training_seconds,
                "seconds_per_update": training_seconds / (update + 1),
                "events_seen": training.events_seen,
                "pieces_seen": training.pieces_seen,
                "mean_free_steps": float(np.mean(window)),
                "eta": musician.learner.config.eta,
                "phrase_activity": {"mean": float(np.mean([x[0] for x in activity])), "saturated": float(np.mean([x[1] for x in activity])), "silent": float(np.mean([x[2] for x in activity]))},
                "intention_spread": float(np.mean([x[3] for x in activity])),
                "mean_abs_efficacy": float(np.nanmean([x[4] for x in activity])),
                **measured,
            }
            activity = []
            window = []
            receipt["history"].append(row)
            selected = row["focus"]["mean_nll"] if a.focus else row["mean_nll"]
            if a.rollback and best < float("inf") and selected > best * a.rollback:
                # the run diverged: back to the best brain, half the step, fresh stream state
                settle = musician.settle_steps
                musician = Musician.load(folder / "brain.npz", backend=a.backend, device=a.device)
                musician.settle_steps = settle
                state = musician.fresh(a.batch)
                scale *= 0.5
                rollbacks.append({"update": update + 1, "surprise": selected, "best": best, "eta_scale": scale})
                row["rolled_back"] = True
                receipt["rollbacks"] = rollbacks
            if selected < best:
                best = selected
                musician.save(folder / "brain.npz")
                if "plan_nll" in row:
                    receipt["best_plan_nll"] = row["plan_nll"]
            musician.save(folder / "latest.npz")
            receipt.update(
                best_validation_nll=best,
                updates=update + 1,
                seconds=time.monotonic() - started,
            )
            (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps(row), flush=True)
            if time.monotonic() - started >= a.max_seconds:
                break
    receipt["complete"] = True
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
