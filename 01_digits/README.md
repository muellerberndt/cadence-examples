# 01 · Digits

A patch net learns to read the 8×8 scikit-learn digits with the owner-local free/nudged
rule. No backward pass: every overlap moves on what its own two endpoints did in two
settlements. This example is the tutorial for `cadence.Learner`.

```bash
pip install "cadence-net>=0.2" scikit-learn
python train.py                    # about 20 s on a laptop
python train.py --verify receipt.json
```

## What the net is

Every pixel is an owner, so 64 input owners; 16, 32, or 64 hidden owners; 10 output
owners, one per class. `cadence.layered(64, hidden, 10, density=1.0)` wires input → hidden
and hidden → output, and ties every hidden ↔ output pair into one *seam*: a forward overlap
and a feedback overlap that share one scale. The feedback is what lets a nudge at the
outputs reach the hidden owners. The numbers to learn are one per seam plus one bias per
owner: 1,319, 2,519, or 4,919. The validation split picks the size.

An image is a clamp. A pixel's level in [0, 1] is the drive on its owner, and nothing else
is clamped. The net settles; the most active output owner is the answer.

## How it learns

For every batch of 32 images:

1. **Free phase.** Settle from rest under the image clamp until no owner moves by more
   than 3e-3 in a step (about 30 steps). This state is the answer; no label has entered.
2. **Nudged phases.** From the free state, settle again while the ten output owners feel
   an extra drive `beta * (target − softmax(s / T))`, a cross-entropy nudge at
   `beta = 0.1`, `T = 0.1`, for at most 12 steps. Then the same with `−beta`. The nudge
   spreads back over the feedback seams and moves the hidden owners a little.
3. **Update.** Each seam moves its scale by `eta / (2 beta)` times the difference of its
   endpoint products in the two nudged phases; each owner moves its bias likewise. That is
   all: two numbers per seam, one per owner.

`eta` starts at 3 and decays by 0.8 per epoch over 20 epochs. The rule is `learning_rule(dt=1.0)`:
unit slope, threshold 0, a leak of 0.1 so an owner below rest still responds, and rest an
exact zero.

## What the script does with the data

The 1,797 images are split 80/20, stratified. The training 80% is split again 80/20 into a
fit set and a validation set; every configuration in the declared grid (hidden 16, 32, 64)
is trained on the fit set and read on the validation set; the best is retrained on the
whole training set, three seeds, and the test set is read once per seed. Baselines are
trained on the same training set and read on the same test set. All of it is in the receipt.

## The numbers

From `receipt.json`: seed 0 split, validation selected 64 hidden owners, three seeds of
that net, one laptop core.

| model | parameters | epochs | training time | held-out accuracy |
|---|---|---|---|---|
| patch net 64-64-10, free/nudged rule | 4,919 | 20 | 2.5 s | 0.962 ± 0.003 |
| MLP 64-32-10, Adam, batch 32 | 2,410 | 50 | 0.19 s | 0.967 |
| MLP 64-32-10, Adam, batch 32 | 2,410 | 12 | 0.05 s | 0.925 |
| MLP 64-16-10, Adam, batch 32 | 1,210 | 50 | 0.18 s | 0.961 |
| logistic regression, lbfgs | 650 | 73 | 0.02 s | 0.967 |

Read it plainly. The rule reaches the accuracy of the small MLPs, and it gets there in far
fewer passes over the data: at 12 epochs the MLPs are at 0.91 to 0.93 and the patch net is
past 0.96. It does not get there in less wall-clock or with fewer parameters. Every update
is three settlements of tens of steps each rather than one forward and one backward pass,
and on a CPU at this size that is a factor of ten in time; and the validation split, with
fewer images to fit on, prefers the largest net in the grid. Trained on the full training
set, the 32-hidden net reaches 0.969 ± 0.002 over three seeds (2,519 parameters, 2.3 s),
but that number was read after looking at the test set and is not what the receipt binds.
The shuffled-label control, a net trained on permuted labels, scores 0.06: the accuracy
comes from the labels.

## What to look at in the receipt

- `validation`: every configuration tried, its parameters, its validation accuracy.
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
