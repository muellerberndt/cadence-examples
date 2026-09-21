# Amen: a jungle composer in one patch, running in the browser

One Cadence record patch learned sixteen jungle tracks as events per half-beat: a slice of a
drum break, a sub-bass note, a change flag and a texture. The page ships that trained brain.
Nothing on it is recorded: press the button and the brain starts from silence, hears a
count-in and computes a track in the browser, one half-beat at a time, hearing each half-beat
it plays, while the page shows its activity. When the track is computed it is rendered
through the instrument and played, with every note and the brain in time with the sound.

## What is here

- `web/index.html`: the page. A button, a Departures slider (it scales the probability of
  leaving the loop at a predicted change point) and a bar count; then the sound, a notes
  panel (every slice, bass note, change point and texture band), and the brain: the heard
  event, the 128 context channels with their gates, the strongest synaptic drives, the
  8,192 record cells with the 48 that fire and their push into the output, the output
  scores, and the loop through the world. Static, no build step, no external service.
- `web/engine.js`: the brain's forward pass and its playing rule, the same arithmetic as the
  library: a gated linear context, a k-winner record code over a fixed random projection
  regenerated from the seed, a record read added to a linear readout.
- `web/models/<name>/`: one trained brain each: the slow parameters (float64), the record
  tables (float32), the running mean of the record reading, and `model.json` with every
  constant. `web/models/index.json` lists them with what each was trained on and its
  held-out figures; the page offers them as a selector, so checkpoints can be compared by ear.
- `web/kit/`: the instrument: 32 half-beat slices of a drum break and twelve sub-bass notes
  as 16-bit wav, and `kit.json` with gains and the texture bands.
- `runs/record-composer-v10/receipt.json`: the receipt every number here comes from, and
  `parity.json`, the archived run's own sixteen bars from silence.
- `parity.mjs`: the browser engine against that archived run. `verify.py`: rechecks the
  receipt's sources, the model files, and runs the parity test when node is installed.

The development project (training sources, transcription, fixtures, listening rounds) is
kept elsewhere and is not part of this repository.

## Run it

    python -m http.server --directory amen/web 8000     # then open http://127.0.0.1:8000/
    python amen/verify.py
    node amen/parity.mjs

## What the receipt says

It names the composer run's own sealed receipt and its independent verification, the Cadence
commit it trained with (`3d655c8`), the size of the brain (80 input ports, 128 context
channels, 71 output ports, 29,895 slow parameters, 8,192 record cells of which 48 fire),
the training cost (under seven CPU minutes, no GPU), and next-event accuracy on three
held-out tracks with the records writing online, against baselines on the same rows:

| Measure | Brain | Baseline |
|---|---|---|
| Next drum slice | 0.84 to 0.86 | most frequent slice 0.03 to 0.05 |
| Next bass note | 0.54 to 0.64 | repeat the previous note 0.14 to 0.48 |
| Texture bands, mean error | 0.085 to 0.090 | repeat the previous half-beat 0.099 to 0.117 |
| Change points recalled | 0.07 to 0.15 | predicted probability about twice as high at true change points |

Parity: on the archived generation from silence in the highest-score mode the browser
engine reproduces all 128 slices, bass notes and change points, with output scores equal
to 2e-8 (the record tables ship as float32). The browser instrument renders the same events
at the same level as the Python instrument (rms within one percent).

## Supplied and learned

Supplied: the transcription of recordings into events (the break's position per half-beat,
the sub-bass semitone, a change point where a half-beat departs from the same position one
bar earlier, the sustain above 250 Hz with the played slice's own tail removed), the
instrument, the eight-position clock, the count-in (at the wake the brain hears the break's
last half-beat), and the two moves a departure may make (a roll repeats the slice just
heard, a retrigger restarts the break at a bar start). Learned from random parameters and
empty records: every event the brain plays, the gains, the bass line, when a change point
is due, and the texture.

## Limits

One DJ mix, sixteen training tracks and three held-out tracks from the same mix; one seed.
No transformer or recurrent baseline on the same stream yet. Change points are hard to
predict. The texture is smoother than a real one, because a squared-error readout predicts
the mean. Slice identity beyond the kind of sound is not measurable in a mixed recording,
so outside change points the transcription supplies the loop position by declaration.
From silence the brain has no track in its records to lean on, so its tracks share one
beat and differ in their departures, bass and texture.

The instrument's drum slices come from a sampled drum break and its bass notes from a
sample pack. Whether they may be redistributed publicly has not been verified; the kit is a
separate folder with its own description so it can be replaced without touching the brain.
