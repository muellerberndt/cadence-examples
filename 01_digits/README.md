# 01 · Digits

A patch net learns to read the 8×8 scikit-learn digits with the owner-local free/nudged
rule. No backward pass: every seam moves on what its own two endpoints did in two
settlements. This rung is the tutorial for `cadence.Learner`; read
[How a patch net learns](../HOW_IT_LEARNS.md) first if the words *owner*, *seam*, *clamp*,
or *nudge* are new.

```bash
pip install "cadence-net>=0.7" scikit-learn
python train.py                    # about 20 s on a laptop
python train.py --verify receipt.json
python build_page.py               # embeds net.json into index.html; open it and draw a digit
```

## 1. The data

1,797 pictures of handwritten digits, 8×8 pixels each with 17 grey levels, ten classes.
`train.py` divides pixel values by 16 so each is in [0, 1], and splits the pictures 80/20,
stratified by class, with a fixed seed: 1,437 for training, 360 held out. The held-out 360
are read once, at the very end, after every decision has been made.

## 2. The net

64 input owners, one per pixel. A hidden layer of 16, 32, or 64 owners. 10 output owners,
one per class. `cadence.layered(64, hidden, 10, density=1.0)` builds:

- an input→hidden overlap from every pixel owner to every hidden owner (64 × hidden), with
  random signs and fan-scaled magnitudes;
- a hidden→output overlap from every hidden owner to every output owner (hidden × 10);
- for each hidden→output overlap, its reverse output→hidden overlap with the same
  strength: together they are one **seam**, and the learner keeps them equal;
- ten output↔output lateral overlaps starting at 0.

There are no overlaps into the input owners, so their potentials sit at their clamps.
The numbers to learn are one per seam plus one bias per owner: 1,319, 2,519, or 4,919
for the three sizes. The rule is `learning_rule(dt=1.0)`: activation `tanh(v/2)` above
rest and a tenth of that below, so rest publishes exactly zero and no owner is ever dead.

## 3. From a picture to a prediction

A picture becomes a clamp: pixel `k`'s level is the drive on input owner `k`, and every
other owner's drive is zero. The net settles from all-zero: input owners rise to their
clamps; hidden owners integrate the weighted pixel activations; output owners integrate
the hidden ones; the output activations feed back into the hidden owners; and every owner
keeps moving its potential toward its total drive until no activation changes by more
than 10⁻⁴ in a step, about 30 steps. The most active of the ten output owners is the
prediction. Nothing about the label was used.

## 4. How it learns

For every batch of 32 pictures:

1. **Free phase.** Settle under the pictures' clamps to a tolerance of 3·10⁻³.
2. **Nudged phases.** From the free rest state, settle again with the extra drive
   `β (target − softmax(s_out / T))` on the ten output owners, `β = 0.1`, `T = 0.1`, target
   1 on the label's owner and 0 elsewhere; then again with `−β`. The nudge moves the
   output owners; through the feedback seams it moves the hidden owners a little.
3. **Update.** Each seam changes by `η · (s⁺_i s⁺_j − s⁻_i s⁻_j) / (2β)`, averaged over the
   batch; each bias by `η_b · (s⁺_i − s⁻_i) / (2β)`. `η` starts at 3 and decays by 0.8 per
   epoch; `η_b` is `η/100`.

Twenty epochs, so about 900 updates. Section 3 of [How a patch net learns](../HOW_IT_LEARNS.md)
walks one such update through a six-owner net with every number.

## 5. Selection, then one look at the test set

The training set is split again, 80/20, into a fit set and a validation set. Each hidden
size in the declared grid (16, 32, 64) is trained twice on the fit set and read on the
validation set; the best mean validation accuracy wins, smaller net on ties. The winner is
retrained on the whole training set with three seeds, and each seed's net reads the test
set once. Baselines, logistic regression and small MLPs trained by Adam, are fit on the
same training set and read on the same test set by the same script.

## 6. The numbers

From `receipt.json`: seed 0 split, validation selected 64 hidden owners, three seeds of
that net, one laptop core shared with other runs.

| model | parameters | epochs | training time | held-out accuracy |
|---|---|---|---|---|
| patch net 64-64-10, free/nudged rule | 4,919 | 20 | 5.6 s | 0.962 ± 0.003 |
| MLP 64-32-10, Adam, batch 32 | 2,410 | 50 | 0.84 s | 0.967 |
| MLP 64-32-10, Adam, batch 32 | 2,410 | 12 | 0.20 s | 0.925 |
| MLP 64-16-10, Adam, batch 32 | 1,210 | 50 | 0.70 s | 0.961 |
| logistic regression, lbfgs | 650 | 73 | 0.07 s | 0.967 |

Read it plainly. The rule reaches the accuracy of the small MLPs, and it gets there in far
fewer passes over the data: at 12 epochs the MLPs are at 0.91 to 0.93 and the patch net is
past 0.96. It does not get there in less wall-clock or with fewer parameters. Every update
is three settlements of tens of steps each rather than one forward and one backward pass,
and on a CPU at this size that is a factor of seven in time; and the validation split, with
fewer images to fit on, prefers the largest net in the grid. Trained on the full training
set, the 32-hidden net reaches 0.969 ± 0.002 over three seeds (2,519 parameters, 2.3 s),
but that number was read after looking at the test set and is not what the receipt binds.
The shuffled-label control, a net trained on permuted labels, scores 0.06: the accuracy
comes from the labels.

## 7. What is different from an MLP

The MLP in the table computes `softmax(W₂ relu(W₁x + b₁) + b₂)` in one pass and learns by
backprop: Adam receives the gradient of the cross-entropy for every weight from a
backward pass that multiplies the output error by `W₂ᵀ`. The patch net has no pass in
either direction. Its prediction is a rest state reached by tens of owner-local repairs,
its hidden owners feel the output owners, and its weights learn from a second settlement
under a nudge. The two nets have the same shape and the same number of numbers, and they
end at the same accuracy. Section 6 of [How a patch net learns](../HOW_IT_LEARNS.md) has
the full comparison.

## 8. What to look at in the receipt

- `validation`: every configuration tried, its parameters, its validation accuracies.
- `runs`: per seed, the untrained accuracy (chance), train and test accuracy, seconds, and
  the mean number of settlement steps per phase.
- `shuffled_label_control_test_accuracy`: chance, as it must be.
- `conformance`: the trained engine against the owner-by-owner reference with its message
  ledger; the deviation is rounding.
- `comparison`: the baselines, measured here.
- `boundary`: what the example claims and what it does not.

`python train.py --verify receipt.json` recomputes the digest, checks that `train.py` still
hashes to what the receipt says, and re-derives the accuracy from the confusion matrix and
the selection from the validation table.

## 9. The page

`index.html` is self-contained: the trained net's dense overlap matrix and biases are
embedded (`net.json`, exported by `train.py` from the first seed's net), and the settlement
runs in JavaScript, owner by owner, with the same rule. Draw on the 8×8 grid, or load one of
the ten held-out pictures the page ships, one per class; the bars replay the ten output
owners step by step and the strip shows the hidden owners at rest. On those ten pictures the
page's activations agree with the Python engine to 5·10⁻⁵ (the matrix is rounded to five
decimals). A hand-drawn stroke is off the training distribution, which was scanned and
blurred, so a thin "1" or a "7" without a bar is read wrong more often than the receipt's
accuracy suggests; that is what a 1,437-picture training set buys.

## 10. Things to try

- `--seeds 5` for tighter error bars; `--seed 1` for a different split.
- Change `GRID` to include `{"hidden": 32, "eta": 2.0}` and watch the validation table.
- Set `centered=False` in `CONFIG` to use a one-sided nudge: one settlement fewer per
  batch, a noisier update.
- Set `tolerance=None` and `free_steps=20` to see what a phase that stops mid-transient
  does to learning (nothing good; that was the first bug in this example).
