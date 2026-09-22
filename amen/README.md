# Amen: a jungle composer in one patch, running in the browser

One Cadence record patch learned jungle tracks as events per half-beat: a slice of a drum
break, a sub-bass note, a change flag and a texture. The page ships that brain as a row of
checkpoints, each of which heard more material than the one before and can be trained
further. Nothing on the page is recorded: press the button and the brain starts from
silence, hears a count-in and computes a track in the browser, one half-beat at a time,
hearing each half-beat it plays, while the page shows its activity. When the track is
computed it is rendered through the instrument and played, with every note and the brain
in time with the sound. The page says on its face that the brain is still training.

## What this example shows

- **Creation.** The brain starts from silence and computes a track of its own, one half-beat at a time. Nothing on the page is recorded.
- **A loop through the world.** It hears each half-beat it plays, so what it does next depends on what it did.
- **Slow and fast memory in one patch.** The slow parameters carry what the corpus sounds like and 8,192 record cells carry particular readings. The held-out figures are measured with the records writing online; the page keeps those writes off.
- **Prediction on tracks it never heard**, against baselines on the same rows.

## What is here

- `web/index.html`: the page, in two columns: the studio and the sound on the left, the
  brain pinned on the right so it stays in view. A button, a Departures slider (it scales
  the probability of leaving the loop at a predicted change point), a Variation slider (how
  much each dub differs, see below), a bar count and a brain selector; then the sound, a
  notes panel (every slice, bass note, change point and texture band; a returning pattern
  slice in gold, a fresh departure in red), and the brain: the heard event, the 128 context
  channels with their gates, the strongest synaptic drives, the 8,192 record cells with the
  48 that fire and their push into the output, the output scores, and the loop through the
  world. Static, no build step, no external service.
- `web/engine.js`: the brain's forward pass and its playing rule, the same arithmetic as the
  library: a gated linear context, a k-winner record code over a fixed random projection
  regenerated from the seed, a record read added to a linear readout.
- `web/models/<name>/`: one trained brain each: the slow parameters (float64), the record
  tables (float32), the running mean of the record reading, and `model.json` with every
  constant. `web/models/index.json` lists them with what each was trained on and its
  held-out figures; the page offers them as a selector, so checkpoints can be compared by ear.
- `web/card.png`: the 1200 by 630 social card the page's Open Graph and Twitter tags point
  to; they carry the absolute URL of the page's current home and must follow it if it moves.
- `web/kit/`: the instrument: 32 half-beat slices of a drum break and twelve sub-bass notes
  as 16-bit wav, and `kit.json` with gains and the texture bands.
- `runs/<run>/receipt.json`: one receipt per brain, which every number here comes from, and
  `parity.json`, the archived run's own sixteen bars from silence and its self-primed
  continuation.
- `parity.mjs`: the browser engine against every archived run. `verify.py`: rechecks the
  receipt's sources, the model files, and runs the parity test when node is installed.

The development project (training sources, transcription, fixtures, listening rounds) is
kept elsewhere and is not part of this repository.

## Run it

    python -m http.server --directory amen/web 8000     # then open http://127.0.0.1:8000/
    python amen/verify.py
    node amen/parity.mjs

## What the receipt says

Each brain has a receipt under `runs/record-composer-v10/`, `runs/record-composer-v11/`, `runs/record-composer-v12/`, `runs/record-composer-v13/`, `runs/record-composer-v14/`. A receipt names the composer run's own sealed receipt
and its independent verification, the Cadence commit it trained with, the size of the brain
(80 input ports, 128 context channels, 71 output ports, 29,895 slow parameters, 8,192 record
cells of which 48 fire), the training cost (CPU minutes on a laptop, no GPU), and next-event
accuracy on held-out tracks with the records writing online, against baselines on the same
rows (in brackets):

| Brain | Heard | Held-out tracks | Next drum slice (most frequent slice) | Next bass note (repeat the previous) | Texture error (repeat the previous) | Change points recalled |
|---|---|---|---|---|---|---|
| One mix | 16 tracks of one DJ mix, 70 minutes | 3 | 0.84 to 0.86 (0.03 to 0.05) | 0.54 to 0.64 (0.14 to 0.48) | 0.085 to 0.090 (0.099 to 0.117) | 0.07 to 0.15 |
| Two mixes | 26 tracks and windows of two DJ mixes, 133 minutes | 5 | 0.84 to 0.86 (0.03 to 0.05) | 0.44 to 0.65 (0.13 to 0.48) | 0.083 to 0.117 (0.099 to 0.134) | 0.11 to 0.19 |
| Three corpora | 71 tracks and windows: two DJ mixes and 45 full tracks, 6.2 hours | 8 | 0.85 to 0.88 (0.03 to 0.05) | 0.41 to 0.64 (0.13 to 0.52) | 0.080 to 0.134 (0.099 to 0.173) | 0.08 to 0.22 |
| Three corpora, longer | the same 71 tracks and windows, six epochs in all | 8 | 0.84 to 0.87 (0.03 to 0.05) | 0.42 to 0.65 (0.13 to 0.52) | 0.082 to 0.137 (0.099 to 0.173) | 0.06 to 0.29 |
| Three corpora, on 0.12.0 | the same 71 tracks and windows, six epochs, trained on Cadence 0.12.0 | 8 | 0.84 to 0.87 (0.03 to 0.05) | 0.44 to 0.64 (0.13 to 0.52) | 0.080 to 0.137 (0.099 to 0.173) | 0.09 to 0.27 |

The page opens with Three corpora, on 0.12.0 (`default` in `web/models/index.json`): the same recipe as Three corpora, longer, trained from scratch on the current library release. Three times the
material did not move the held-out scores, and neither did three more epochs on it (the
fourth brain): the instrument and the transcription set the ceiling. What changes is the
playing from silence. The three-corpora brains leave the break's order by themselves and
range over more bass notes; the longer-trained one predicts change points more often from
silence, so its dubs are busier.

Parity: on each archived generation from silence in the highest-score mode the browser
engine reproduces all 128 slices, bass notes and change points, with output scores equal
to 2e-8 (the record tables ship as float32). It also reproduces the library's record writes:
after observing its own first four bars, the continuation matches to 4e-8. The page keeps
those writes off; measured on six seeds they did not make a dub's opening return.

## Supplied and learned

Supplied: the transcription of recordings into events (the break's position per half-beat,
the sub-bass semitone, a change point where a half-beat departs from the same position one
bar earlier, the sustain above 250 Hz with the played slice's own tail removed), the
instrument, the eight-position clock, the count-in, and the three moves a departure may make
(a roll repeats the slice just heard, a retrigger restarts the break at a bar start, a bar
jump plays the same position in another bar of the break). The moves are declared: among
the corpus's change points only the roll stands out from chance (0.12 against 0.03); a
retrigger occurs at 0.12, its chance level, and a bar jump at 0.05. Learned from random
parameters and empty records: every event the brain plays, the gains, the bass line, when a
change point is due, and the texture.

## Why two dubs differ

From silence, the highest score at every port gives one and the same track, and the bass
settles on the same note every time. Each dub therefore draws the following from its seed,
scaled by the Variation slider; at zero none of it is drawn (departures still are: with both
sliders at zero every dub is the same track).

- The bass of the first two bars, and of every departure, is drawn from the brain's own
  scores over notes.
- A drum pattern of the dub's own. The break enters on one of its four bars. Over the first
  one, two or four bars departures are drawn at a raised probability, with the dub's own
  shares of the three moves and its preferred targets. That opening is the dub's loop: the
  slice at the cycle start and at each of the opening's departures returns at the same place
  in every cycle, unless the brain draws a fresh departure there. Between those places the
  brain plays on from what it hears, so a fill stays local and the groove comes back. This
  is a rule of the instrument: the transcription reads every repeating bar as the break in
  order, so a track's own chop is not in the training data.
- Instrument settings: a tempo between 165 and 182 bpm, the break's pitch (up to four
  semitones either way, set apart from the tempo as on a sampler), a
  key (bass and pad transposed by two semitones down to five up), the pad's waveform, chord, register
  and detune, how it is played (held, a skank on beats two and four, a dotted stab or a
  swell per bar), and a filter sweep over 4, 8 or 16 bars. The pad sounds at the bass note each
  bar plays most; the brain's texture bands shape it over time, and the instrument sets the
  layer 12 dB under the drums and bass.

The seed is printed under the button. The same seed and settings give the same dub.

## Limits

The break is one break, so the drum vocabulary is its 32 slices. One training seed. No
transformer or recurrent baseline on the same stream yet. Change points are hard to predict,
and the brain has no phrase clock, so it places departures by what it just heard and not by
where a sixteen-bar phrase ends. The texture is smoother than a recorded one, because a
squared-error readout predicts the mean. Slice identity beyond the kind of sound is not
measurable in a mixed recording, so outside change points the transcription supplies the
loop position by declaration.

The instrument's drum slices come from a sampled drum break and its bass notes from a
sample pack. Whether they may be redistributed publicly has not been verified; the kit is a
separate folder with its own description so it can be replaced without touching the brain.

## Build on it

Fork it and train a composer on other material: the engine, the page and the receipt format do not depend on the break.
