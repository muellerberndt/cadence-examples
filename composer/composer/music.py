"""Musical planning and criticism: explicit application wiring, not a claim of learned taste."""

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Brief:
    prompt: str
    mode: int
    style: int
    arousal: int
    bars: int
    bpm: int
    instrument: str
    mood: str


def parse_prompt(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Describe a piece first.")
    if len(prompt) > 500:
        raise ValueError("Please use at most 500 characters.")
    text = prompt.lower()
    orchestra = bool(
        re.search(r"orchestr|cinema|film|williams|epic|symphon|theme", text)
    )
    minor = bool(re.search(r"sad|minor|melanch|dark|tragic|mourn|myster", text))
    calm = bool(re.search(r"sad|calm|gentle|quiet|soft|slow|peace", text))
    bright = bool(
        re.search(r"fast|energetic|bright|joy|hero|epic|triumph|groove", text)
    )
    style = 1 if orchestra else (2 if "bach" in text or "baroque" in text else 0)
    bars = 24 if bright else 16 if calm else 20
    return Brief(
        prompt,
        int(minor),
        style,
        0 if calm else 2 if bright else 1,
        bars,
        bars * 4,
        "orchestra" if orchestra else "piano",
        "reflective" if minor else "radiant",
    )


def phrase_role(brief, bar):
    phrase = int(bar) // 4
    count = brief.bars // 4
    if phrase == 0:
        return "statement"
    if phrase == 1:
        return "answer"
    if phrase == count - 1:
        return "return and resolution"
    if phrase == count - 2 and count > 4:
        return "climax"
    return "development"


def harmonic_plan(brief, intuition=None, rng=None):
    """Supplied formal destinations, learned transition preferences, varied routes."""
    from .intuition import Intuition

    intuition = intuition or Intuition()
    rng = rng or np.random.default_rng(17)
    tonic = 12 if brief.mode else 0
    allowed = np.array([12, 3, 17, 7, 8, 10] if brief.mode else [0, 14, 16, 5, 7, 21])
    result = []
    previous = tonic
    for bar in range(0, brief.bars, 4):
        role = phrase_role(brief, bar)
        destination = tonic if role in ("answer", "return and resolution") else 7
        route = []
        for position in range(4):
            if position == 3:
                choice = destination
            elif position == 0 and role in ("statement", "return and resolution"):
                choice = tonic
            elif role == "return and resolution" and position == 2:
                choice = 7
            else:
                p = intuition.chord_probability(brief.mode, previous)[allowed] ** 0.65
                p[allowed == previous] *= 0.3
                # Rehearse how this choice leads toward the phrase destination.
                if position == 2:
                    p *= (
                        np.array(
                            [
                                intuition.chord_probability(brief.mode, int(c))[
                                    destination
                                ]
                                for c in allowed
                            ]
                        )
                        ** 0.4
                    )
                if len(result) >= 4:
                    p[allowed == result[-4 + position]] *= 0.5
                p /= p.sum()
                choice = int(rng.choice(allowed, p=p))
            route.append(choice)
            previous = choice
        result.extend(route)
    return result[: brief.bars]


def triad(chord):
    root = chord % 12
    return [root, (root + (3 if chord >= 12 else 4)) % 12, (root + 7) % 12]


def critique(events, brief, *, intuition=None, motif=None):
    if len(events) < 2:
        return {"score": -100.0, "issues": ["too few notes"]}
    notes = np.array([e["pitch"] for e in events])
    duration = np.array([e["duration"] for e in events])
    onset = np.array([e["step"] for e in events])
    chords = np.array([e["chord"] for e in events])
    scale = [0, 2, 3, 5, 7, 8, 10] if brief.mode else [0, 2, 4, 5, 7, 9, 11]
    diatonic = float(
        np.mean(
            [n % 12 in scale or n % 12 in triad(int(c)) for n, c in zip(notes, chords)]
        )
    )
    leaps = float((np.abs(np.diff(notes)) > 9).mean())
    repeated = float((np.diff(notes) == 0).mean())
    aligned = np.array([n % 12 in triad(c) for n, c in zip(notes, chords)])
    strong = onset % 4 == 0
    harmony = float(aligned[strong].mean()) if strong.any() else float(aligned.mean())
    unique = len(set(notes.tolist()))
    # A phrase moved later in the score must receive exactly the same density score.
    density = len(notes) / max(1, float((onset[-1] + duration[-1] - onset[0]) / 16))
    long_holds = float((duration > 16).mean())
    intervals = np.diff(notes)
    supplied_motif = motif is not None
    motif = (
        np.diff(motif)[:4] if supplied_motif else intervals[: min(4, len(intervals))]
    )
    # Compare actual phrase openings. Searching every possible window produces
    # accidental matches and overstates thematic development in long pieces.
    starts = (
        [0]
        if supplied_motif
        else np.flatnonzero(np.diff((onset - onset[0]) // 64) > 0) + 1
    )
    openings = [
        intervals[i : i + len(motif)]
        for i in starts
        if i + len(motif) <= len(intervals)
    ]
    recurrence = float(sum(np.array_equal(part, motif) for part in openings))
    development = (
        float(np.mean([np.mean(np.sign(part) == np.sign(motif)) for part in openings]))
        if openings
        else 0.0
    )
    expectancy = intuition.evaluate(events) if intuition is not None else {}
    extreme = expectancy.get("extreme_surprise_rate", 0.0)
    cadence = float(notes[-1] % 12 in triad(int(chords[-1])))
    # Reward coherent movement and a few returns, with explicit anti-collapse penalties.
    score = (
        2 * diatonic
        + 1.5 * harmony
        - 2 * leaps
        - 3 * max(0, repeated - 0.35)
        - long_holds
        + 0.1 * min(unique, 8)
        + 0.25 * development
        + 0.3 * cadence
        - 0.4 * extreme
        - 0.15 * abs(density - (5 if brief.arousal == 0 else 7))
    )
    issues = []
    if repeated > 0.45:
        issues.append("repeated-note collapse")
    if leaps > 0.18:
        issues.append("too many wide melodic leaps")
    if harmony < 0.5:
        issues.append("weak strong-beat harmony")
    if unique < 5:
        issues.append("little pitch variety")
    return {
        "score": float(score),
        "scale_fit": diatonic,
        "strong_beat_harmony": harmony,
        "wide_leap_rate": leaps,
        "repeated_note_rate": repeated,
        "unique_pitches": unique,
        "notes_per_bar": density,
        "motif_returns": recurrence,
        "motif_contour_similarity": development,
        "phrase_ending_harmony": cadence,
        **expectancy,
        "issues": issues,
    }


def write_midi(events, brief, path):
    import mido

    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    midi.tracks.append(meta)
    meta.append(mido.MetaMessage("track_name", name="Cadence: " + brief.prompt[:80]))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(brief.bpm)))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    tracks = [[], [], []]
    for e in events:
        onset = int(e["step"] * 120)
        duration = max(30, int(e["duration"] * 120 * e.get("gate", 0.88)))
        bar = e["step"] // 16
        role = phrase_role(brief, bar)
        arc = (
            -10
            if role == "statement"
            else 4
            if role == "answer"
            else 15
            if role == "climax"
            else 8
            if role == "development"
            else -3
        )
        if brief.arousal == 2 and role == "return and resolution":
            arc = 10
        if bar == brief.bars - 1:
            arc -= 8
        velocity = int(
            np.clip(
                63
                + brief.arousal * 9
                + arc
                + 8 * np.sin(np.pi * (e["step"] % (16 * 4)) / (16 * 4))
                + (5 if e["step"] % 4 == 0 else 0),
                35,
                110,
            )
        )
        tracks[0].extend(
            [
                (onset, True, e["pitch"], velocity),
                (onset + duration, False, e["pitch"], 0),
            ]
        )
    # The score owns the chosen harmony. Rendering must never silently generate a different plan.
    fallback = harmonic_plan(brief)
    plan = [
        next((int(e["chord"]) for e in events if e["step"] // 16 == bar), fallback[bar])
        for bar in range(brief.bars)
    ]
    previous_voicing = [48, 55, 60]
    for bar, chord in enumerate(plan):
        # Accompaniment/orchestration is supplied. No claim of learned full orchestral instrumentation.
        pcs = triad(chord)
        # Smooth common-tone voice leading, instead of octave-folded root-position jumps.
        import itertools

        voicings = [
            v
            for v in itertools.combinations(range(45, 69), 3)
            if sorted(p % 12 for p in v) == sorted(pcs) and v[-1] - v[0] <= 16
        ]
        voicing = min(
            voicings, key=lambda v: sum(abs(a - b) for a, b in zip(v, previous_voicing))
        )
        previous_voicing = voicing
        role = phrase_role(brief, bar)
        expression = -6 if role == "statement" else 7 if role == "climax" else 0
        if bar == brief.bars - 1:
            expression = -10
        for beat in range(4):
            onset = (bar * 16 + beat * 4) * 120
            if brief.instrument == "piano":
                pitches = [voicing[[0, 1, 2, 1][beat]]]
            else:
                # An energetic brief gets a clear string pulse beneath the long melodic line.
                pitches = list(voicing) if (beat == 0 or brief.arousal == 2) else []
            for pitch in pitches:
                dur = (
                    (330 if brief.arousal == 2 else 1680)
                    if brief.instrument == "orchestra"
                    else 420
                )
                tracks[1].extend(
                    [
                        (onset, True, pitch, 48 + brief.arousal * 7 + expression),
                        (onset + dur, False, pitch, 0),
                    ]
                )
        bar_events = [e for e in events if e["step"] // 16 == bar]
        predicted_bass = (
            bar_events[0].get("bass", 36 + chord % 12)
            if bar_events
            else 36 + chord % 12
        )
        bass_candidates = [pitch for pitch in range(28, 53) if pitch % 12 in pcs]
        bass = min(bass_candidates, key=lambda pitch: abs(pitch - predicted_bass))
        tracks[2].extend(
            [(bar * 1920, True, bass, 54), (bar * 1920 + 1800, False, bass, 0)]
        )
    programs = [0, 0, 0] if brief.instrument == "piano" else [73, 48, 42]
    end = brief.bars * 1920
    for channel, (messages, program) in enumerate(zip(tracks, programs)):
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(
            mido.Message("program_change", channel=channel, program=program, time=0)
        )
        if channel == 0 and brief.instrument == "orchestra":
            colors = (
                {
                    "statement": 60,
                    "answer": 73,
                    "development": 48,
                    "climax": 56,
                    "return and resolution": 60,
                }
                if brief.arousal == 2
                else {
                    "statement": 73,
                    "answer": 68,
                    "development": 48,
                    "climax": 60,
                    "return and resolution": 73,
                }
            )
            # Instrument roles are supplied arrangement choices, recorded as MIDI program changes.
            messages += [
                (bar * 1920, None, colors[phrase_role(brief, bar)], 0)
                for bar in range(0, brief.bars, 4)
            ]
        previous = 0
        for tick, on, note, velocity in sorted(
            messages,
            key=lambda x: (x[0], 0 if x[1] is False else 1 if x[1] is None else 2),
        ):
            tick = min(tick, end)
            if on is None:
                track.append(
                    mido.Message(
                        "program_change",
                        channel=channel,
                        program=note,
                        time=tick - previous,
                    )
                )
                previous = tick
                continue
            track.append(
                mido.Message(
                    "note_on" if on else "note_off",
                    channel=channel,
                    note=int(note),
                    velocity=velocity,
                    time=tick - previous,
                )
            )
            previous = tick
        track.append(mido.MetaMessage("end_of_track", time=end - previous))
    meta.append(mido.MetaMessage("end_of_track", time=end))
    midi.save(path)
