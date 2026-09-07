# 07 · Music

A patch net learns to continue Bach chorales, chord by chord, and plays them for you in
the browser. This rung changes two things at once: several output owners can be right at
the same time (a chord is a set of pitches), so the nudge is the quadratic one on a
multi-hot target; and the learned model composes by settling once per beat. Read
[How a patch net learns](../HOW_IT_LEARNS.md) first.

```bash
pip install "cadence-net[accel]>=0.2"
python train.py                    # about an hour; downloads the chorale corpus once
python build_page.py               # embeds net.json into index.html; open it and press play
python train.py --verify receipt.json
```

## 1. The data

The JSB chorales at quarter-note resolution (Boulanger-Lewandowski, Bengio, and Vincent
2012; the JSON distributed by czhuang, pinned by digest in the receipt): 382 four-part
chorales as sequences of chords, each chord the MIDI pitches sounding on that beat, 229 for
training, 76 for validation, 77 held out. Pitches run from 43 to 96, so 54 pitch owners
cover them. Training rows are transposed by up to three semitones either way, which
multiplies the training material by seven and which a chorale tolerates; validation and
test chorales are not transposed.

## 2. From chords to a clamp

The last eight chords become eight blocks of 54 input owners, one block per beat, with
the owners of the sounding pitches clamped at 1: 432 input owners. A hidden layer of 128
or 256 owners (the validation chorales choose); 54 output owners, one per pitch. After a
settlement, the output owners above a threshold are the next chord, at most four of them.

## 3. How it learns

The nudge is the quadratic one: `β (target − s)` on all 54 output owners at once, with the
target 1 on the pitches of the next chord and 0 elsewhere, so a seam that helps two
correct pitches is pulled twice. Otherwise the rule is section 3 of
[How a patch net learns](../HOW_IT_LEARNS.md) unchanged. Ten epochs, batches of 256.

Two things are chosen on the validation chorales and nowhere else: the threshold at which
an output owner counts as a sounding pitch (the one with the best validation F1), and a
logistic calibration that turns activations into per-pitch probabilities for the
bits-per-chord figure. The baselines get the same calibration on the same validation set.

## 4. The numbers

From `receipt.json`: validation hidden 128: F1 0.470, hidden 256: F1 0.276; the selected net, 64,253 parameters, trained
10 epochs on 83,825 transposed rows; test rows 4,109 from 77 held-out chorales,
read once. Threshold 0.3 and calibration slope 12.0 chosen on validation; the baselines
calibrated the same way on the same validation chorales.

| model | parameters | epochs | training | pitch-set F1 | precision | recall | bits per chord | exact chords |
|---|---|---|---|---|---|---|---|---|
| patch net 432-128-54, quadratic nudge | 64,253 | 10 | 1052 s | 0.483 | 0.455 | 0.514 | 16.6 | 0.045 |
| MLP 432-128-54, sigmoid outputs, Adam | 62,390 | 10 | 2 s | 0.470 | 0.371 | 0.639 | 24.7 | 0.008 |
| repeat the last chord | 0 | | | 0.369 | 0.369 | 0.368 | 40.3 | 0.114 |
| per-pitch unigram | 54 | | | 0.000 | 0.000 | 0.000 | 23.5 | 0.004 |

Read it plainly. This is the first rung where the patch net is ahead of the same-shape
backprop network on its own terms: a slightly better pitch-set F1 and eight fewer bits per
chord at the same parameter count and epochs, with the quadratic nudge on a multi-hot
target doing what a sigmoid-and-cross-entropy output layer does for the MLP. Both are far
from exact: a chord is right in its entirety one time in twenty, and "repeat the last
chord" is right one time in nine, because chorales hold notes. Wall-clock is the usual
factor, here several hundred, and the larger hidden layer (256) fell apart on validation
(F1 0.276) at this learning rate, which is why the receipt binds 128.

The continuation in the receipt takes the most active owners above the threshold and
locks onto a held chord within a few beats; that is what the most likely next chord of a
chorale is. `continue.py` and the page draw each pitch with its calibrated probability
instead, keep at most four, and move.

## 5. The page

Press "Write 16 chords": the page clamps the last eight chords of a held-out opening onto
the input owners, settles the net in JavaScript, reads the pitch owners above the
threshold, appends the chord, and repeats. Press "Play" to hear the opening and the
continuation on plain oscillators, one beat per chord. "Another opening" starts from a
different held-out chorale.
