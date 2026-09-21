# Amen: a jungle composer in one patch

One Cadence record patch learned sixteen jungle tracks as events per half-beat: a slice of a
drum break, a sub-bass note, a change flag and a texture. It hears four bars of a track it
never trained on, or nothing at all, and then plays sixteen bars of its own in a closed
loop, hearing each half-beat it plays. The page plays the result and replays the brain in
time with the sound.

## What is here

- `web/index.html`: the page. A waveform, a notes panel (every slice, bass note, change
  point and texture band of the clip), and the brain: the heard event, the 128 context
  channels with their gates, the strongest synaptic drives, the 8,192 record cells with
  the 48 that fire and their push into the output, the output scores, and the loop
  through the world. Static, no build step, no external service.
- `web/audio/`, `web/traces/`, `web/weights/`: the twelve clips, their per-half-beat brain
  traces and the readout weights the page draws. `web/traces/index.json` orders them.
- `runs/<name>/receipt.json`: one receipt per composer run whose clips are shown. Every
  number on the page and in this file comes from one of them.
- `verify.py`: rechecks the receipts against the files.

The development project (training sources, transcription, fixtures, listening rounds) is
kept elsewhere and is not part of this repository.

## Run it

    python -m http.server --directory amen/web 8000     # then open http://127.0.0.1:8000/
    python amen/verify.py

## What each receipt says

Each receipt names the composer run's own sealed receipt and its independent verification
(`results.run_receipt_sha256`, `results.independent_verification_sha256`), the Cadence
commit it trained with (`environment.library_commit`), the size of the brain, and
next-event accuracy on three held-out tracks with the records writing online, against the
baselines measured on the same rows. `sources` hashes every file of that run shown on the
page; `results.clips` gives, per clip, the bars it generated and how many of them occur
nowhere verbatim in the training corpus.

| Run | What it is | Next slice | Next bass note | Texture error |
|---|---|---|---|---|
| `record-composer-v10` | drums, bass and texture; record reading fixed (cadence `3d655c8`) | 0.84 to 0.86 (most frequent slice 0.03 to 0.05) | 0.54 to 0.64 (repeat previous 0.14 to 0.48) | 0.085 to 0.090 (repeat previous 0.099 to 0.117) |
| `record-composer-v7` | drums and bass; departures at predicted change points | 0.84 to 0.85 | 0.50 to 0.62 | not measured |
| `record-composer-v6` | drums and bass; an earlier transcription whose loop prior was stronger | 0.92 to 0.96 | 0.37 to 0.54 | not measured |

The slice figures of `record-composer-v6` are higher because its transcription assigned
the break's own loop order more freely, not because the brain was better; the later
transcription marks the half-beats that depart from the bar before and scores those too.

## Supplied and learned

Supplied: the transcription of recordings into events (the break's position per half-beat,
the sub-bass semitone, a change point where a half-beat departs from the same position one
bar earlier, the sustain above 250 Hz with the played slice's own tail removed), the
instrument that renders events, the eight-position clock, the count-in (at the wake the
brain hears the break's last half-beat), and the two moves a departure may make (a roll or
a retrigger). Learned from random parameters and empty records: every event the brain
plays, the gains, the bass line, when a change point is due, and the texture.

## Limits

One DJ mix, sixteen training tracks and three held-out tracks from the same mix; one seed.
No transformer or recurrent baseline on the same stream yet. Change points are hard to
predict (recall 0.07 to 0.15). The generated texture is smoother than the real one,
because a squared-error readout predicts the mean. Slice identity beyond the kind of sound
is not measurable in a mixed recording, so outside change points the transcription
supplies the loop position by declaration.

The clips are rendered from slices of a sampled drum break and from sampled sub-bass
notes. The raw samples and the reference recordings are not included here.
