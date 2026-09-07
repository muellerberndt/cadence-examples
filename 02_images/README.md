# 02 · Images

The same rule as the digits example on MNIST: 784 pixels a picture, 60,000 training
images, 10,000 held out. Nothing changes but the scale, and the scale is what this rung
is for: training runs on the accelerated backend, the readout on the float64 CPU backend,
and the receipt records how the two agree.

```bash
pip install "cadence-net[accel]>=0.2" scikit-learn pandas
python train.py                    # about 15 minutes on an M-series Mac; downloads MNIST once (OpenML)
python train.py --verify receipt.json
```

## What the net is

784 input owners, a hidden layer of 128 or 256 owners (the validation split chooses), 10
output owners; forward and feedback overlaps tied into seams. A pixel's level in [0, 1]
is the clamp on its owner.

## How it learns

Batches of 256: one free settlement to a tolerance of 3e-3, one settlement nudged toward
the label with the cross-entropy nudge and one away, and every seam moving on the
difference of its own two endpoints. Ten epochs, learning rate 3 decaying by 0.8 per
epoch. Selection on 10,000 validation images carved from the training set; the test set
is read once, on the CPU backend in float64. Logistic regression and an MLP of the
selected size, trained by Adam on the same images, are measured in the same script.

## The numbers

From `receipt.json`: validation hidden 128: 0.9611, hidden 256: 0.9669; the selected net retrained on all 60,000 images;
test set read once on the CPU backend in float64. Training on mps float32; the
CPU readout and the accelerated readout agree on every one of the first 2,000 test
predictions (`backend_prediction_agreement_first_2k` = 1.0000), and the
trained engine matches the owner-by-owner reference to 4.4e-16.

| model | parameters | epochs | training | held-out accuracy |
|---|---|---|---|---|
| patch net 784-256-10, free/nudged rule, torch on Apple silicon | 204,359 | 10 | 1650 s | 0.9744 |
| logistic regression (lbfgs, 100 iterations) | 7,850 | 100 | 7 s | 0.9258 |
| MLP 784-256-10 (adam, batch 256) | 203,530 | 10 | 15 s | 0.9779 |
| MLP 784-256-10 (adam, batch 256) | 203,530 | 30 | 48 s | 0.9821 |

Read it plainly. The rule scales: 97.44% on MNIST with the same settings as the 8×8 digits,
untouched, in ten epochs, and the float32 accelerator and the float64 reference agree on
every prediction. It does not reach the same-size MLP's 97.79% at ten epochs, and it costs a
hundred times the wall-clock, because each of the 37-step free settlements and the two
11-step nudged settlements per batch is a full pass over a 1,050-owner net. The
untrained net scores 0.065. Nothing here was tuned on MNIST.
