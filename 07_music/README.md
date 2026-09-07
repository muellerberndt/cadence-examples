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

{{NUMBERS}}

## 5. The page

Press "Write 16 chords": the page clamps the last eight chords of a held-out opening onto
the input owners, settles the net in JavaScript, reads the pitch owners above the
threshold, appends the chord, and repeats. Press "Play" to hear the opening and the
continuation on plain oscillators, one beat per chord. "Another opening" starts from a
different held-out chorale.
