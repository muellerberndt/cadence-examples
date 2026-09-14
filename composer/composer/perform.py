"""Performing: compose a piece from a mood, record the brain while it does so, and record
it again while it listens to the finished piece.

Two recordings, both exact replays of the settling that decided the music:

* ``live/{j}.bin``: every iteration of every committed event while composing, in the
  order the events were committed (phrases, then accepted edits).
* ``listen/{k}.bin``: every iteration of every final event while the brain hears the
  finished piece from the start in one stream; this is what playback is synchronised to.

Each file is float32 little-endian, shape ``(iterations, 3, neurons)``: activation, the
change of activation since the previous iteration (the repair), and every neuron's
equation mismatch at that iteration.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from .listen import Listener
from .musician import (
    BAR,
    DELTAS,
    DURATIONS,
    MOOD_NAMES,
    WINDOW,
    Senses,
    describe,
    events_from_tokens,
    primed,
    write_score,
)

ROOT = Path(__file__).resolve().parents[1]

WORDS = {
    "mode": [
        (r"minor|sad|dark|melanch|tragic|mourn|myster|tense|dramatic|grief", 1),
        (r"major|bright|joy|happy|hero|triumph|serene|calm|warm", 0),
    ],
    "tempo": [
        (r"slow|adagio|largo|calm|gentle|lento|still", 0),
        (r"fast|allegro|presto|energetic|vivace|urgent|dance|driving", 2),
    ],
    "energy": [
        (r"sparse|calm|still|quiet|gentle|slow|spacious", 0),
        (r"busy|energetic|driving|virtuos|fast|dense|restless", 2),
    ],
    "dynamics": [
        (r"soft|quiet|pianissimo|gentle|whisper|intimate", 1),
        (r"loud|forte|powerful|epic|heroic|triumph|thunder", 3),
        (r"expressive|dynamic|dramatic|swell", 2),
    ],
    "register": [(r"low|deep|dark|bass|cello", 0), (r"high|bright|light|sparkl|flute|bell", 2)],
    "texture": [
        (r"solo|piano|guitar|single|harpsichord", 0),
        (r"chamber|duet|trio|quartet|strings only", 1),
        (r"orchestr|symphon|epic|cinematic|film|ensemble|band|full|brass", 2),
    ],
    "tension": [
        (r"simple|folk|hymn|pure|consonant|children", 0),
        (r"chromatic|tense|dissonan|myster|jazz|modern|anxious|uneasy", 2),
    ],
}
DEFAULT = {"mode": 0, "tempo": 1, "energy": 1, "dynamics": 0, "register": 1, "texture": 0, "tension": 0}


def parse_mood(text):
    """A bounded keyword map from a brief onto the seven measured mood classes."""
    text = (text or "").lower()
    classes = dict(DEFAULT)
    for name, rules in WORDS.items():
        for pattern, value in rules:
            if re.search(pattern, text):
                classes[name] = value
                break
    return np.array([classes[k] for k in MOOD_NAMES], dtype=int), classes


def prime_tokens(brief):
    """Sixteen quiet events that set the key, texture and register before the musician
    begins: a supplied opening, heard with the clock rewound and dropped from the piece."""
    mode, texture, register = int(brief[0]), int(brief[5]), int(brief[4])
    families = [0] if texture == 0 else [0, 1, 4] if texture == 1 else [1, 2, 3, 4]
    base = 36 if register == 0 else 48 if register == 1 else 60
    triad = [0, 3 if mode else 4, 7]
    velocity = 4 if int(brief[3]) in (0, 2) else 2 if int(brief[3]) == 1 else 6
    out = []
    for k in range(WINDOW):
        fam = families[k % len(families)]
        pitch = base + triad[k % 3] + 12 * ((k // 3) % 2) - 24
        out.append([int(np.clip(pitch, 0, 72)), 5, 0 if k % 4 else 6, fam, velocity])
    return out


class Recording:
    """Writes one frame file per replayed event and keeps an index of what each file is."""

    def __init__(self, folder, neurons, iterations):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.neurons, self.iterations = neurons, iterations
        self.index = []
        self.current = None
        self.regions = None

    def observer(self, k, iteration, activation, repair, mismatch, drive):
        if iteration == 1:
            self.current = np.zeros((self.iterations, 3, self.neurons), "<f4")
        self.current[iteration - 1, 0] = activation
        self.current[iteration - 1, 1] = repair
        self.current[iteration - 1, 2] = mismatch
        if iteration == self.iterations:
            j = len(self.index)
            (self.folder / f"{j}.bin").write_bytes(self.current.tobytes())
            self.index.append({"file": j, "max_mismatch": float(np.abs(mismatch).max()), "event": None})
            self.current = None

    def label(self, start, count, **meta):
        for j in range(start, start + count):
            self.index[j].update(meta, event=j - start + meta.get("first_event", 0))


def steps_of(tokens):
    out, step = [], 0
    for t in tokens:
        step += int(DELTAS[int(t[2])])
        out.append(step)
    return out


def compose(
    musician,
    brief,
    *,
    bars=32,
    futures=8,
    horizon=24,
    edits=2,
    seed=17,
    tempo=100,
    target_surprise=1.0,
    progress=None,
    folder=None,
    temperature=0.9,
    top=0,
    detune=0.0,
):
    """Compose ``bars`` bars in the mood ``brief``; record the brain when ``folder`` is given."""
    rng = np.random.default_rng(seed)
    listener = Listener(target_surprise, brief, tempo)
    total_steps = bars * BAR
    prime = prime_tokens(brief)
    tokens = [list(t) for t in prime]
    senses = primed(prime, total_steps)
    state = musician.fresh(1)
    iterations = musician.settle_steps
    live = Recording(Path(folder) / "live", musician.n, iterations) if folder else None
    phrases = []
    started = time.monotonic()

    def notify(stage, **info):
        if progress:
            progress({"stage": stage, "seconds": time.monotonic() - started, **info})

    notify("priming", prime=prime, brief={k: int(v) for k, v in zip(MOOD_NAMES, brief)}, bars=bars, tempo=tempo)
    while senses.step < total_steps:
        stop = min(total_steps, (senses.step // BAR + 2) * BAR)
        before = (list(tokens), senses, state)
        out = musician.imagine(tokens, senses, brief, futures=futures, horizon=horizon, rng=rng, state=state, stop_at=stop, temperature=temperature, top=top, detune=detune)
        scores, winner = listener.rank(out)
        chosen = [list(map(int, t)) for t in out["tokens"][winner]]
        if not chosen:
            break
        advantage = scores[winner]["score"] - float(np.mean([s["score"] for s in scores]))
        phrase = {
            "index": len(phrases),
            "from_step": int(senses.step),
            "first_event": len(tokens) - WINDOW,
            "winner": winner,
            "valence": float(advantage),
            "candidates": [
                {"tokens": [list(map(int, t)) for t in toks], "surprise": [float(x) for x in sur], **score}
                for toks, sur, score in zip(out["tokens"], out["surprise"], scores)
            ],
        }
        notify("imagined", phrase=phrase, committed=len(tokens) - WINDOW)
        if live is not None:
            first = len(live.index)
            _, _, surprise = musician.replay(before[0], before[1], before[2], chosen, brief, live.observer, total_steps=total_steps)
            live.label(first, len(chosen), phase="phrase", phrase=len(phrases), first_event=phrase["first_event"], step=int(senses.step))
            phrase["live"] = [first, first + len(chosen)]
            phrase["surprise"] = surprise
        tokens = tokens + chosen
        senses = out["senses"][winner]
        state = out["state"].copy_rows([winner])
        phrases.append(phrase)
        notify("committed", phrase=phrase["index"], events=chosen, starts=steps_of(tokens)[WINDOW:][-len(chosen):], step=int(senses.step), total=total_steps, live=phrase.get("live"))
    draft = [list(t) for t in tokens]

    def whole(all_tokens):
        review = musician.review(all_tokens, brief, total_steps=total_steps)
        return review, listener.score(all_tokens[WINDOW:], review["surprise"])

    notify("listening", events=len(draft) - WINDOW)
    review, before_score = whole(draft)
    current, edit_log = draft, []
    for round_ in range(edits):
        bars_of = review["bars"]
        candidates = []
        for start in range(0, bars, 4):
            mask = (bars_of >= start) & (bars_of < start + 4)
            if mask.sum() < 4:
                continue
            passage = [t for t, b in zip(current[WINDOW:], bars_of) if start <= b < start + 4]
            candidates.append((listener.score(passage, review["surprise"][mask])["score"], start))
        if not candidates:
            break
        _, start = min(candidates)
        snapshot = review["snapshots"].get(start)
        if snapshot is None:
            break
        branch_state, branch_senses, length = snapshot
        left = current[:length]
        right = [list(t) for t, b in zip(current[WINDOW:], bars_of) if b >= start + 4]
        stop = min(total_steps, (start + 4) * BAR)
        notify("editing", bar=int(start), round=round_)
        out = musician.imagine(left, branch_senses, brief, futures=futures * 2, horizon=horizon * 3, rng=rng, state=branch_state, stop_at=stop, temperature=temperature, top=top, detune=detune)
        scores, winner = listener.rank(out)
        replacement = [list(map(int, t)) for t in out["tokens"][winner]]
        proposed = left + replacement
        if right:
            gap = stop - out["senses"][winner].step
            right[0][2] = int(np.argmin(np.abs(DELTAS - (DELTAS[right[0][2]] + gap))))
            proposed = proposed + right
        new_review, after_score = whole(proposed)
        accepted = after_score["score"] > before_score["score"] + 1e-9
        entry = {
            "round": round_,
            "bar": int(start),
            "before": before_score,
            "after": after_score,
            "accepted": bool(accepted),
            "candidates": [s["score"] for s in scores],
            "winner": winner,
            "replacement": replacement,
            "first_event": length - WINDOW,
            "removed": len(current) - length - len(right),
        }
        if accepted and live is not None:
            first = len(live.index)
            _, _, surprise = musician.replay(left, branch_senses, branch_state, replacement, brief, live.observer, total_steps=total_steps)
            live.label(first, len(replacement), phase="edit", edit=round_, first_event=length - WINDOW, step=int(start * BAR))
            entry["live"] = [first, first + len(replacement)]
            entry["surprise"] = surprise
        edit_log.append(entry)
        notify("edited", **{k: v for k, v in entry.items() if k not in ("before", "after")}, before=before_score["score"], after=after_score["score"], events=proposed[WINDOW:] if accepted else None, starts=steps_of(proposed)[WINDOW:] if accepted else None)
        if accepted:
            current, review, before_score = proposed, new_review, after_score
    final = current
    listen_surprise = review["surprise"].tolist()
    if live is not None:
        listen = Recording(Path(folder) / "listen", musician.n, iterations)
        notify("listening back", events=len(final) - WINDOW)
        _, _, listen_surprise = musician.replay(final[:WINDOW], primed(final[:WINDOW], total_steps), musician.fresh(1), final[WINDOW:], brief, listen.observer, total_steps=total_steps)
        listen.label(0, len(final) - WINDOW, phase="listen", first_event=0)
        (Path(folder) / "live_index.json").write_text(json.dumps(live.index))
        (Path(folder) / "listen_index.json").write_text(json.dumps(listen.index))
    return {
        "brief": {k: int(v) for k, v in zip(MOOD_NAMES, brief)},
        "seed": seed,
        "bars": bars,
        "tempo_bpm": tempo,
        "prime": prime,
        "draft": [list(map(int, t)) for t in draft[WINDOW:]],
        "draft_starts": steps_of(draft)[WINDOW:],
        "events": [list(map(int, t)) for t in final[WINDOW:]],
        "starts": steps_of(final)[WINDOW:],
        "durations": [int(DURATIONS[int(t[1])]) for t in final[WINDOW:]],
        "phrases": phrases,
        "edits": edit_log,
        "draft_score": listener.score(draft[WINDOW:], musician.review(draft, brief, total_steps=total_steps)["surprise"]),
        "final_score": before_score,
        "listen_surprise": [float(x) for x in listen_surprise],
        "per_bar_surprise": {int(b): float(np.mean(review["surprise"][review["bars"] == b])) for b in np.unique(review["bars"])},
        "frames": {"iterations": iterations, "neurons": musician.n, "live": len(live.index) if live else 0, "listen": len(final) - WINDOW if live else 0},
        "seconds": time.monotonic() - started,
        "supplied": [
            "keyword brief onto seven measured mood classes",
            "sixteen-event opening prime, heard with the clock rewound, dropped from the piece",
            "listener: surprise target, mood-class fit, anti-collapse health",
            "two-bar imagination horizon, four-bar edit passages",
            "GM program per instrument family when rendering",
        ],
        "learned": ["every pitch, duration, timing, instrument and velocity intention: the trained musician"],
    }


def perform(musician, mood_text, *, folder, key=0, render=False, record=True, checkpoint=None, target_surprise=1.0, progress=None, **options):
    """Compose from a brief into ``folder``: composition.json, draft/final MIDI, recordings, audio."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    brief, classes = parse_mood(mood_text)
    result = compose(musician, brief, folder=folder if record else None, target_surprise=target_surprise, progress=progress, **options)
    tempo = result["tempo_bpm"]
    write_score(events_from_tokens(result["draft"], key=key), folder / "draft.mid", bpm=tempo, name=mood_text)
    write_score(events_from_tokens(result["events"], key=key), folder / "final.mid", bpm=tempo, name=mood_text)
    result.update(
        id=folder.name,
        mood_text=mood_text,
        mood_classes=classes,
        key=key,
        checkpoint=str(checkpoint) if checkpoint else None,
        brain=describe(musician),
        target_surprise=target_surprise,
        files={"midi": f"/output/{folder.name}/final.mid", "draft_midi": f"/output/{folder.name}/draft.mid"},
    )
    if render:
        from .render import render as synthesize

        try:
            result["audio"] = {v: synthesize(folder / f"{v}.mid", folder / f"{v}.wav") for v in ("draft", "final")}
            result["files"].update(audio=f"/output/{folder.name}/final.wav", draft_audio=f"/output/{folder.name}/draft.wav")
        except RuntimeError as error:
            result["audio"] = {"error": str(error)}
    (folder / "composition.json").write_text(json.dumps(result, separators=(",", ":")))
    return result
