"""Rehearse candidate phrases in isolated state, commit a score, re-listen and revise."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cadence as cd
import numpy as np
from cadence.plasticity import Valence

from .brain import describe, drives, settle_checked
from .encoding import DURATIONS, OFFSETS, music_features
from .intuition import Intuition
from .motif import MotifBrain
from .music import critique, harmonic_plan, parse_prompt, phrase_role, triad, write_midi
from .render import render
from .telemetry import Recorder, capture
from .topology import weight_changes

ROOT = Path(__file__).resolve().parents[1]


def sample(p, rng, temperature=0.8, top=16):
    p = np.asarray(p, float).clip(1e-15)
    ids = np.argsort(p)[-top:]
    q = p[ids] ** (1 / temperature)
    q /= q.sum()
    return int(rng.choice(ids, p=q))


class Composer:
    def __init__(self, checkpoint=None, taste=None):
        self.path = Path(checkpoint or next((p for p in (ROOT / "checkpoints/phrase-composer/brain.npz", ROOT / "checkpoints/best/brain.npz") if p.exists()), ROOT / "checkpoints/phrase-composer/brain.npz"))
        self.learner = cd.Learner.load(self.path, backend="cpu", precision="float64")
        self.intuition = Intuition(taste)
        self.performer = MotifBrain(self.learner, self.intuition)
        self.reward = Valence(cap=1)
        self.last = None
        self.theme = None
        self.expected_quality = None

    def appraise(self, scores):
        """Prediction error against recent selected quality, not a guaranteed-positive rank."""
        scores = np.asarray(scores, float)
        expected = (
            float(scores.mean())
            if self.expected_quality is None
            else self.expected_quality
        )
        winner = float(scores.max())
        valence = float(self.reward(np.asarray([winner - expected]))[0])
        self.expected_quality = 0.85 * expected + 0.15 * winner
        return {
            "selected_quality": winner,
            "expected_quality": expected,
            "valence": valence,
        }

    def phrase(
        self,
        brief,
        previous,
        start_bar,
        bars,
        rng,
        *,
        variants=6,
        motif=None,
        detune=0.06,
        plan=None,
        progress=None,
        phase="Imagining",
    ):
        learner = self.performer
        brain = learner.brain
        plan = plan if plan is not None else harmonic_plan(brief, self.intuition)
        role = phrase_role(brief, start_bar)
        start = start_bar * 16
        end = (start_bar + bars) * 16
        histories = [list(previous) for _ in range(variants)]
        events = [[] for _ in range(variants)]
        positions = np.full(variants, start)
        last_drives = [None] * variants
        settling = []
        rehearsal_state = None
        hidden = np.concatenate(
            [
                np.asarray(brain.connectome.populations[name])
                for name in ["harmony", "rhythm", "phrase_memory"]
            ]
        )
        for iteration in range(bars * 16):
            active = positions < end
            if not active.any():
                break
            context = np.array([h[-4:] for h in histories])
            extra = np.array(
                [
                    music_features(h, int(pos), brief.mode, brief.style, brief.arousal)
                    for h, pos in zip(histories, positions)
                ]
            )
            if self.theme is not None:
                extra[:, 44:48] = (np.asarray(self.theme) - 30) / 30
            drive = drives(learner, context, extra)
            learner.expect(drive, brief, context, positions)
            drive[:, hidden] += rng.normal(0, detune, size=(variants, len(hidden)))
            for j in range(variants):
                chord = plan[min(brief.bars - 1, int(positions[j]) // 16)]
                # Current harmonic intention is a bounded local drive into this same graph.
                drive[j, learner.output_index[OFFSETS[2] + chord]] += 0.15
            if motif:
                learner.cue(drive, iteration)
            observed_row = int(np.flatnonzero(active)[iteration % int(active.sum())])
            recorder = (
                Recorder(
                    learner,
                    drive,
                    row=observed_row,
                    modulators={
                        "detuning": {
                            "value": detune,
                            "effect": "actual Gaussian perturbation of the rehearsing latent neurons",
                        },
                        "plasticity": {
                            "value": 0,
                            "effect": "imagined candidates cannot write themselves into memory",
                        },
                    },
                )
                if progress is not None and iteration % 4 == 0
                else None
            )
            state, receipt = settle_checked(
                learner,
                drive,
                observer=recorder.observe if recorder else None,
                warm=rehearsal_state,
            )
            if recorder is not None:
                self.last_observation = (
                    brain,
                    state,
                    observed_row,
                    "actual rehearsal",
                )
            rehearsal_state = state
            settling.append(receipt)
            logits = (
                state.activation[:, learner.output_index] / learner.config.temperature
            )
            for j in np.flatnonzero(active):
                last_drives[j] = drive[j : j + 1].copy()
            for j in range(variants):
                if not active[j]:
                    continue
                pos = int(positions[j])
                chord = plan[min(brief.bars - 1, pos // 16)]
                prev = int(histories[j][-1][0]) + 36
                z = logits[j, :61].copy()
                pitch = np.arange(61) + 36
                # Learned interval relationships reward plausible movement, not an exact tune.
                previous_interval = int(histories[j][-1][0]) - int(histories[j][-2][0])
                expected = self.intuition.interval_probability(previous_interval)
                z += 0.3 * np.log(expected[np.clip(pitch - prev, -12, 12) + 12])
                # Supplied range/voice-leading priors are separately ablatable in receipts.
                z -= np.maximum(0, np.abs(pitch - prev) - 3) * 0.17
                z[(pitch < 52) | (pitch > 84)] -= 8
                center = 72 if role in ("development", "climax") else 65
                z -= 0.018 * np.abs(pitch - center)
                allowed = (
                    [0, 2, 3, 5, 7, 8, 10] if brief.mode else [0, 2, 4, 5, 7, 9, 11]
                )
                allowed = sorted(set(allowed + triad(chord)))
                z[~np.isin(pitch % 12, allowed)] -= 2
                if pos % 4 == 0:
                    z[np.isin(pitch % 12, triad(chord))] += 0.8
                if len(histories[j]) > 2 and histories[j][-1][0] == histories[j][-2][0]:
                    z[int(histories[j][-1][0])] -= 2
                melody = sample(np.exp(z - z.max()), rng, 0.85)
                duration_logits = logits[j, OFFSETS[1] : OFFSETS[2]].copy()
                duration_logits[DURATIONS > 16] -= 10
                if brief.arousal == 0:
                    duration_logits[DURATIONS < 3] -= 1.3
                duration_logits[DURATIONS > end - pos] -= 20
                duration = sample(
                    np.exp(duration_logits - duration_logits.max()), rng, 0.85, top=7
                )
                # Harmonic intentions change on bar boundaries. Notes may breathe before the next onset.
                length = min(int(DURATIONS[duration]), end - pos, 16 - pos % 16)
                if pos >= end - 8:
                    length = end - pos
                    if role == "return and resolution":
                        melody = (
                            int(
                                min(
                                    (p for p in range(52, 85) if p % 12 == 0),
                                    key=lambda p: abs(p - prev),
                                )
                            )
                            - 36
                        )
                    else:
                        melody = (
                            int(
                                min(
                                    (
                                        p
                                        for p in range(52, 85)
                                        if p % 12 in triad(chord)
                                    ),
                                    key=lambda p: abs(p - prev),
                                )
                            )
                            - 36
                        )
                duration = int(np.argmin(abs(DURATIONS - length)))
                bass = sample(
                    np.exp(logits[j, OFFSETS[3] :] - logits[j, OFFSETS[3] :].max()),
                    rng,
                    top=8,
                )
                token = [melody, duration, chord, bass]
                histories[j].append(token)
                events[j].append(
                    {
                        "step": pos,
                        "duration": length,
                        "pitch": melody + 36,
                        "chord": chord,
                        "bass": bass + 24,
                        "token": token,
                        "gate": 0.65
                        if pos + length == end and role != "return and resolution"
                        else 0.88,
                        "phrase_role": role,
                    }
                )
                positions[j] += length
            if progress is not None:
                progress(
                    phase,
                    bar=start_bar,
                    candidate=observed_row + 1,
                    candidates=variants,
                    trace=recorder.finish(packed=True) if recorder else None,
                    trials=[
                        {"candidate": j + 1, "notes": list(e)}
                        for j, e in enumerate(events)
                    ],
                    note_options=[
                        {
                            "pitch": int(k) + 36,
                            "activation": float(logits[observed_row, k]),
                        }
                        for k in np.argsort(logits[observed_row, :61])[-8:][::-1]
                    ],
                )
        scores = [
            critique(e, brief, intuition=self.intuition, motif=motif) for e in events
        ]
        winner = int(np.argmax([s["score"] for s in scores]))
        if progress is not None:
            progress(
                "Comparing futures",
                bar=start_bar,
                winner=winner + 1,
                trial_scores=[s["score"] for s in scores],
            )
        return (
            events[winner],
            histories[winner],
            {
                "candidates": [
                    {"notes": e, "critique": s} for e, s in zip(events, scores)
                ],
                "winner": winner,
                "settling": settling,
                "detuning_std": detune,
            },
            last_drives[winner],
        )

    def compose(self, prompt, *, seed=17, variants=6, render_audio=True, progress=None):
        self.performer = MotifBrain(self.learner, self.intuition)
        self.theme = None
        self.expected_quality = None
        brief = parse_prompt(prompt)
        rng = np.random.default_rng(seed)
        plan = harmonic_plan(brief, self.intuition, rng)
        start_time = time.monotonic()
        name = f"{time.time_ns()}-{seed}"
        folder = ROOT / "runs/compositions" / name
        folder.mkdir(parents=True)

        rehearsals = {}

        def notify(stage, **info):
            event = {"stage": stage, "seconds": time.monotonic() - start_time, **info}
            if (info.get("trace") or {}).get("origin") == "actual rehearsal":
                key = (stage, info.get("bar", 0))
                entry = {
                    k: event[k]
                    for k in ("stage", "bar", "candidate", "trace")
                    if k in event
                }
                if key not in rehearsals:
                    rehearsals[key] = [entry]
                elif len(rehearsals[key]) == 1:
                    rehearsals[key].append(entry)
                else:
                    rehearsals[key][-1] = entry
            (folder / "progress.json").write_text(
                json.dumps({k: v for k, v in event.items() if k != "trace"})
            )
            if progress:
                progress(event)

        notify("Preparing", brief=asdict(brief), committed=[])
        previous = [
            [24, 3, 12 if brief.mode else 0, 12],
            [28 if not brief.mode else 27, 3, 12 if brief.mode else 0, 12],
            [31, 3, 7, 19],
            [28 if not brief.mode else 27, 3, 7, 19],
        ]
        events = []
        phrases = []
        motif = None
        traces = []
        contexts = []
        exploration = 0.06
        for bar in range(0, brief.bars, 4):
            notify("Imagining", bar=bar, total_bars=brief.bars, candidates=variants)
            contexts.append([list(e) for e in previous])
            phrase, previous, record, drive = self.phrase(
                brief,
                previous,
                bar,
                min(4, brief.bars - bar),
                rng,
                variants=variants,
                motif=motif
                if phrase_role(brief, bar) in ("answer", "return and resolution")
                else None,
                detune=exploration,
                plan=plan,
                progress=notify if progress is not None else None,
            )
            if motif is None:
                motif = [e["pitch"] for e in phrase[:6]]
                self.performer.remember(motif)
                self.theme = [e["token"][0] for e in phrase[:4]]
                self.theme += [self.theme[-1]] * (4 - len(self.theme))
            events += phrase
            record["bar"] = bar
            phrases.append(record)
            record["appraisal"] = self.appraise(
                [c["critique"]["score"] for c in record["candidates"]]
            )
            valence = record["appraisal"]["valence"]
            exploration = float(np.clip(0.06 * np.exp(-valence), 0.03, 0.12))
            trace = capture(
                self.performer,
                drive,
                steps=512,
                packed=True,
                modulators={
                    "valence": {
                        "value": valence,
                        "effect": "selected quality minus expected quality; a shortfall broadens the next rehearsal",
                    },
                    "detuning": {
                        "value": record["detuning_std"],
                        "effect": "Gaussian hidden-neuron drive during rehearsal",
                    },
                    "plasticity": {
                        "value": self.performer.memory.writes if bar == 0 else 0,
                        "effect": "one-trial motif synapse writes; corpus weights held until explicit feedback",
                    },
                },
            )
            trace["bar"] = bar
            traces.append(trace)
            notify(
                "Phrase committed",
                bar=bar,
                notes=len(phrase),
                valence=valence,
                trace=trace,
                committed=list(events),
            )
        draft = list(events)
        before = critique(draft, brief, intuition=self.intuition)
        write_midi(draft, brief, folder / "draft.mid")
        notify("Listening to the draft", bar=brief.bars)
        audio_before = (
            render(folder / "draft.mid", folder / "draft.wav") if render_audio else None
        )
        # Revise a complete phrase in context, then judge the whole score before committing.
        weakest = int(
            np.argmin(
                [p["candidates"][p["winner"]]["critique"]["score"] for p in phrases]
            )
        )
        bar = phrases[weakest]["bar"]
        notify("Revising", bar=bar)
        alternative, _, revision, _ = self.phrase(
            brief,
            contexts[weakest],
            bar,
            min(4, brief.bars - bar),
            rng,
            variants=variants * 2,
            motif=motif,
            detune=0.1,
            plan=plan,
            progress=notify if progress is not None else None,
            phase="Revising",
        )
        proposed = [
            e for e in draft if not bar * 16 <= e["step"] < (bar + 4) * 16
        ] + alternative
        proposed.sort(key=lambda e: e["step"])
        after = critique(proposed, brief, intuition=self.intuition)
        accept = after["score"] > before["score"] + 1e-8
        write_midi(proposed if accept else draft, brief, folder / "final.mid")
        audio_after = (
            render(folder / "final.mid", folder / "final.wav") if render_audio else None
        )
        if (
            accept
            and audio_after
            and (
                audio_after["clipped_fraction"] > 0.001
                or audio_after["silent_fraction"]
                > max(0.15, audio_before["silent_fraction"] + 0.03)
            )
        ):
            accept = False
            write_midi(draft, brief, folder / "final.mid")
            audio_after = render(folder / "final.mid", folder / "final.wav")
        final = proposed if accept else draft
        notify(
            "Revision accepted" if accept else "Revision rejected",
            bar=bar,
            committed=final,
            revision={"accepted": accept, "bar": bar, "old": draft, "new": proposed},
        )
        report = {
            "id": name,
            "brief": asdict(brief),
            "seed": seed,
            "brain": describe(self.performer),
            "checkpoint_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "seconds": time.monotonic() - start_time,
            "duration_seconds": 60,
            "draft": draft,
            "events": final,
            "before": before,
            "after": critique(final, brief, intuition=self.intuition),
            "harmonic_plan": plan,
            "intuition_sha256": hashlib.sha256(
                self.intuition.path.read_bytes()
            ).hexdigest()
            if self.intuition.tables is not None
            else None,
            "revision": {
                "bar": bar,
                "accepted": accept,
                "proposal": after,
                "search": revision,
            },
            "phrases": phrases,
            "motif": motif,
            "audio_before": audio_before,
            "audio_after": audio_after,
            "traces": traces,
            "rehearsals": [r for records in rehearsals.values() for r in records],
            "files": {
                "draft_midi": f"/output/{name}/draft.mid",
                "midi": f"/output/{name}/final.mid",
                "draft_audio": f"/output/{name}/draft.wav",
                "audio": f"/output/{name}/final.wav",
            },
            "supplied_structure": [
                "bounded keyword prompt parser",
                "four-bar formal destinations, cadence constraints and phrase breathing",
                "range/voice-leading priors",
                "motif readback schedule",
                "accompaniment and orchestration",
                "explicit symbolic critic and PCM health checks",
            ],
            "learned": [
                "636-input event-conditioned Cadence predictions of melody, onset duration, chord, bass",
                "aggregate corpus relationships for sustained harmony, onset intervals and interval-to-interval expectations",
            ],
            "claim": "Research prototype. No established human-level musicality, learned orchestration, unrestricted language understanding, or advantage over transformers.",
        }
        (folder / "composition.json").write_text(
            json.dumps(report, separators=(",", ":")) + "\n"
        )
        self.last = report
        notify("Ready", id=name, revision_accepted=accept)
        return report

    def hear(self, report, seconds, version="final"):
        """Read the playing MIDI score into an isolated retained state, without learning.

        This is symbolic readback. The waveform is not an input to this encoder.
        """
        if (
            not np.isfinite(seconds)
            or not 0 <= seconds <= 60
            or version not in ("draft", "final")
        ):
            raise ValueError("Invalid playback position or version")
        brief = parse_prompt(report["brief"]["prompt"])
        position = min(brief.bars * 16 - 1, int(seconds * brief.bpm / 60 * 4))
        events = report["draft"] if version == "draft" else report["events"]
        heard = [e["token"] for e in events if e["step"] <= position]
        if not heard:
            heard = [[24, 3, 12 if brief.mode else 0, 12]]
        context = np.asarray([[heard[0]] * max(0, 4 - len(heard)) + heard[-4:]])
        features = np.asarray(
            [music_features(heard, position, brief.mode, brief.style, brief.arousal)]
        )
        drive = drives(self.performer, context, features)
        self.performer.expect(drive, brief, context, np.asarray([position]))
        identity = (report["id"], version, id(self.performer.brain))
        warm = getattr(self, "hearing_state", None)
        if getattr(self, "hearing_id", None) != identity or seconds < getattr(
            self, "hearing_seconds", 0
        ):
            warm = None
        recorder = Recorder(
            self.performer,
            drive,
            origin="MIDI score readback · not waveform perception",
        )
        state, _ = settle_checked(
            self.performer, drive, warm=warm, observer=recorder.observe
        )
        self.last_observation = (self.performer.brain, state, 0, "MIDI score readback")
        self.hearing_state, self.hearing_id, self.hearing_seconds = (
            state,
            identity,
            seconds,
        )
        return {
            "trace": recorder.finish(packed=True),
            "seconds": seconds,
            "step": position,
        }

    def release(self):
        """Release the last measured state on its captured connectome in an isolated copy."""
        if not hasattr(self, "last_observation"):
            raise ValueError("Compose or play a score first")
        brain, state, row, origin = self.last_observation
        isolated = cd.BrainState(
            v=state.v[row : row + 1].copy(),
            activation=state.activation[row : row + 1].copy(),
            adaptation=state.adaptation[row : row + 1].copy(),
            steps=state.steps,
        )
        result = capture(
            SimpleNamespace(brain=brain),
            np.zeros_like(isolated.activation),
            state=isolated,
            include_release=False,
            packed=True,
        )
        result.update(released=True, origin="Input released from " + origin)
        return result

    def feedback(self, report, rating):
        """Teach relative musical relationships, without rehearsing exact note labels."""
        if rating not in (-1, 1):
            raise ValueError("Feedback must be +1 or -1.")
        baseline = dict(np.load(ROOT / "checkpoints/intuition/validation.npz"))
        before = self.intuition.retention(baseline)
        old_tables = {k: v.copy() for k, v in self.intuition.tables.items()}
        self.performer = MotifBrain(self.learner, self.intuition)
        self.performer.remember(report["motif"])
        old_weights = self.performer.brain.weights.copy()
        events = report["events"]
        brief = parse_prompt(report["brief"]["prompt"])
        tokens = [e["token"] for e in events]
        indices = np.arange(4, len(tokens))
        contexts = np.array([tokens[i - 4 : i] for i in indices])
        extra = np.array(
            [
                music_features(
                    tokens[:i],
                    events[i]["step"],
                    brief.mode,
                    brief.style,
                    brief.arousal,
                )
                for i in indices
            ]
        )
        delta = float(self.reward(np.array([rating]))[0])
        self.intuition.teach(events, brief, rating)
        after = self.intuition.retention(baseline)
        accepted = after["mean_nll"] <= before["mean_nll"] * 1.01
        if accepted:
            self.intuition.save(ROOT / "checkpoints/personal/music.npz")
        else:
            self.intuition.tables = old_tables
        self.performer = MotifBrain(self.learner, self.intuition)
        self.performer.remember(report["motif"])
        change_weights = self.performer.brain.weights - old_weights
        drive = drives(self.performer, contexts[:1], extra[:1])
        self.performer.expect(drive, brief, contexts[:1], [events[4]["step"]])
        trace = capture(
            self.performer,
            drive,
            steps=256,
            include_release=False,
            packed=True,
            modulators={
                "valence": {
                    "value": delta,
                    "effect": "signed user preference changes abstract interval, harmony and rhythm associations",
                },
                "plasticity": {
                    "value": float(np.linalg.norm(change_weights)),
                    "effect": "retained effective synapse change norm; zero after rollback",
                },
            },
        )
        trace["edge_changes_url"] = weight_changes(change_weights)
        result = {
            "trace": trace,
            "rating": rating,
            "valence": delta,
            "retained": accepted,
            "before": before,
            "after": after,
            "change": {
                "max_effective_synapse_change": float(np.abs(change_weights).max()),
                "changed_synapses": int(np.count_nonzero(change_weights)),
            },
            "guard": "Reject if held-out relationship NLL increases by more than 1%; this is not a music-quality guarantee.",
            "learning": "Local signed association updates to key-relative chord, interval and rhythm patterns. No exact-note rehearsal; pretrained event weights remain held.",
        }
        (ROOT / "runs/compositions" / report["id"] / "feedback.json").write_text(
            json.dumps(result, indent=2)
        )
        return result
