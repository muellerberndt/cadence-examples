"""Audit a learned polyphonic rollout; arrangement is predicted, with explicit rendering guards."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cadence as cd
import mido
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.brain import describe
from composer.encoding import DURATIONS
from composer.ensemble import (
    DELTAS,
    EXTRA_RAW,
    FAMILIES,
    HISTORY,
    PROGRAMS,
    SIZES,
    Dataset,
    drives,
)
from composer.music import parse_prompt
from tools.train_ensemble import evaluate

ROOT = Path(__file__).resolve().parents[1]


def draw_event(brain, state, drive, rng, *, conditional=False):
    """Repair remaining attributes after a choice, without changing any weights.

    This uses the existing Nudge; it is not an exact joint probability sampler.
    Each call draws one uniform per head before any repair. Matching RNG states
    therefore share those draws; later rendering guards may consume extra draws.
    """
    uniforms = rng.random(len(SIZES))
    token, probabilities = [0] * len(SIZES), [None] * len(SIZES)
    offsets = np.cumsum((0,) + SIZES[:-1])
    mask = np.zeros(brain.brain.connectome.n)
    target = np.zeros_like(drive)
    for k in (3, 0, 2, 1, 4):
        neurons = brain.output_index[offsets[k] : offsets[k] + SIZES[k]]
        z = state.activation[0, neurons] / brain.config.temperature
        q = np.exp((z - z.max()) / 0.85)
        q /= q.sum()
        probabilities[k] = q
        token[k] = min(SIZES[k] - 1, int(np.searchsorted(q.cumsum(), uniforms[k])))
        if conditional and k != 4:
            mask[neurons] = 1
            target[0, neurons[token[k]]] = 1
            state = brain.brain.settle_batch(
                drive,
                steps=16,
                state=state,
                tolerance=0,
                nudge=cd.Nudge(
                    mask=mask.copy(),
                    target=target.copy(),
                    beta=brain.config.beta / brain.slot_count,
                    softmax_temperature=brain.config.temperature,
                    groups=brain.output_groups,
                ),
            )
    return token, probabilities, state


def extra(events, previous, brief):
    x = np.zeros(EXTRA_RAW, np.uint8)
    x[:5] = [
        brief.mode,
        brief.style,
        brief.arousal,
        previous % 16,
        (previous // 16) % 16,
    ]
    pcs = np.zeros(12)
    held = np.zeros((8, 12))
    for e in events:
        pcs *= 0.96
        if e["family"] != 7:
            pcs[e["pitch"] % 12] += 1
        if e["step"] + e["duration"] > previous:
            held[e["family"], e["pitch"] % 12] += 1
    x[5:17] = np.round(pcs / max(1, pcs.sum()) * 255)
    x[17:] = np.minimum(1, held).ravel() * 255
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/ensemble-2048-four-gpu/brain.npz")
    p.add_argument("--dataset", default="ensemble")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt", default="heroic orchestral theme")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument(
        "--conditional",
        action="store_true",
        help="Repair remaining attributes after each choice",
    )
    p.add_argument(
        "--cold",
        action="store_true",
        help="Ablate retained state between generated events",
    )
    a = p.parse_args()
    torch.set_num_threads(2)
    brain = cd.Learner.load(ROOT / a.checkpoint, backend="torch", device=a.device)
    suffix = "-conditioned" if a.conditional else ""
    out = (
        ROOT
        / "runs/ensemble-samples"
        / f"{a.seed}-{'cold' if a.cold else 'warm'}{suffix}"
    )
    out.mkdir(parents=True, exist_ok=True)
    evaluation = {}
    if a.evaluate:
        test = Dataset(ROOT / "data" / a.dataset, "test")
        evaluation["test"] = evaluate(brain, test, 8192)
        train = Dataset(ROOT / "data" / a.dataset, "train").sample(
            np.random.default_rng(90), 100000
        )["labels"]
        ids = np.random.default_rng(77).choice(
            test.count, min(8192, test.count), replace=False
        )
        labels = test.rows(ids)["labels"]
        losses = []
        for k, width in enumerate(SIZES):
            counts = np.bincount(train[:, k], minlength=width) + 0.5
            prob = counts / counts.sum()
            losses.append(float(-np.log(prob[labels[:, k]]).mean()))
        evaluation["training_unigram_nll"] = losses
        (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    brief = parse_prompt(a.prompt)
    rng = np.random.default_rng(a.seed)
    family = 1 if brief.instrument == "orchestra" else 0
    pitches = [60, 63 if brief.mode else 64, 67, 72] * 2
    history = [[pitch - 24, 3, 4, family, 5] for pitch in pitches]
    seed_events = [
        {"step": -32 + i * 4, "pitch": pitch, "duration": 4, "family": family}
        for i, pitch in enumerate(pitches)
    ]
    events = []
    end = brief.bars * 16
    position = 0
    guards = {"duplicate_note": 0, "dense_attack": 0}
    started = time.monotonic()
    state = None
    for _ in range(2400):
        x = extra(seed_events + events, position, brief)
        drive = drives(brain, np.asarray([history[-HISTORY:]], np.uint8), x[None, :])
        state = brain.free(drive, warm=None if a.cold else state)
        token, probabilities, state = draw_event(
            brain, state, drive, rng, conditional=a.conditional
        )
        delta = int(DELTAS[token[2]])
        if sum(e["step"] == position for e in events[-20:]) >= 16 and delta == 0:
            delta = 1
            token[2] = 1
            guards["dense_attack"] += 1
        step = position + delta
        if step >= end:
            break
        occupied = {
            e["pitch"] - 24
            for e in events
            if e["family"] == token[3] and e["step"] <= step < e["step"] + e["duration"]
        }
        if token[0] in occupied:
            q = probabilities[0].copy()
            q[list(occupied)] = 0
            if q.sum() <= 1e-12:
                position = step + 1
                continue
            q /= q.sum()
            token[0] = int(rng.choice(73, p=q))
            guards["duplicate_note"] += 1
        duration = min(int(DURATIONS[token[1]]), end - step)
        events.append(
            {
                "step": step,
                "duration": duration,
                "pitch": token[0] + 24,
                "family": token[3],
                "velocity": 8 + 16 * token[4],
                "token": token,
            }
        )
        history.append(token)
        position = step
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack(
        [
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(brief.bpm)),
            mido.MetaMessage("time_signature", numerator=4, denominator=4),
            mido.MetaMessage("end_of_track", time=end * 120),
        ]
    )
    midi.tracks.append(meta)
    for fam, program in enumerate(PROGRAMS):
        channel = 9 if fam == 7 else fam
        track = mido.MidiTrack(
            [
                mido.MetaMessage("track_name", name=FAMILIES[fam]),
                mido.Message("program_change", program=program, channel=channel),
            ]
        )
        messages = []
        for e in events:
            if e["family"] == fam:
                messages.extend(
                    [
                        (e["step"] * 120, True, e),
                        (min(end, e["step"] + e["duration"]) * 120, False, e),
                    ]
                )
        last = 0
        for tick, on, e in sorted(messages, key=lambda x: (x[0], x[1])):
            track.append(
                mido.Message(
                    "note_on" if on else "note_off",
                    note=e["pitch"],
                    velocity=e["velocity"] if on else 0,
                    channel=channel,
                    time=tick - last,
                )
            )
            last = tick
        track.append(mido.MetaMessage("end_of_track", time=end * 120 - last))
        midi.tracks.append(track)
    midi.save(out / "ensemble.mid")
    receipt = {
        "prompt": a.prompt,
        "seed": a.seed,
        "events": events,
        "brain": describe(brain),
        "seconds": time.monotonic() - started,
        "midi_seconds": midi.length,
        "instrument_counts": dict(
            zip(
                FAMILIES,
                np.bincount([e["family"] for e in events], minlength=8).tolist(),
            )
        ),
        "checkpoint_sha256": hashlib.sha256(
            (ROOT / a.checkpoint).read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "guards": guards,
        "retained_state": not a.cold,
        "conditional_attribute_repairs": a.conditional,
        "attribute_repair_steps": 64 if a.conditional else 0,
        "settling": f"{brain.config.free_steps} local steps per generated event; finite budget, no claim of equation convergence",
        "supplied": [
            "eight-note generic tonal context",
            "keyword mode/style/arousal",
            "maximum 16 notes per simultaneous attack",
            "no overlapping duplicate pitch within instrument family",
            "GM rendering and exact one-minute boundary",
        ],
        "limits": "Raw learned rollout, not the established phrase planner. No guarantee of form, theme, aesthetic quality or long-range coherence.",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k != "events"}), flush=True)


if __name__ == "__main__":
    main()
