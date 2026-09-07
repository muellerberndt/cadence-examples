# 02 · Images

The same rule as the digits example on MNIST: 784 pixels a picture, 60,000 training
images, 10,000 held out. Nothing changes but the scale, and the scale is what this rung is
for: training runs on the accelerated backend, the readout on the float64 CPU backend,
and the receipt records how the two agree. If the mechanism is new to you, read
[How a patch net learns](../HOW_IT_LEARNS.md) first; this page assumes it.

```bash
pip install "cadence-net[accel]>=0.2" scikit-learn pandas
python train.py                    # about 45 minutes on an M-series Mac; downloads MNIST once (OpenML)
python train.py --verify receipt.json
```

## 1. The data

MNIST from OpenML (`mnist_784`, version 1), cached under `data/` as a compressed array
the first time. Pixel values are divided by 255 so each is in [0, 1]. The first 60,000
images are the training set and the last 10,000 the test set, the standard split. The
receipt records the SHA-256 of the arrays actually used.

## 2. The net

784 input owners, one per pixel; a hidden layer of 128 or 256 owners (the validation
split chooses); 10 output owners. `cadence.layered(784, hidden, 10, density=1.0)`, so every
pixel owner reaches every hidden owner, every hidden owner reaches every output owner,
and every hidden↔output pair is one seam. 102,599 or 204,359 numbers to learn, one per
seam plus one bias per owner. Same rule as the digits: `learning_rule(dt=1.0)`.

## 3. From a picture to a prediction, at this size

A picture's 784 levels are the clamp on its 784 owners. A settlement is the same
owner-local rule as always; what changes is the arithmetic. With 1,050 owners, one step
for a batch of 256 pictures is one 256 × 1,050 by 1,050 × 1,050 matrix product plus the
elementwise repair, about 2.6 ms on the Apple GPU in float32. The free phase takes about
37 steps to a tolerance of 3·10⁻³; the readout uses 10⁻⁴.

The library's torch backend does exactly what the NumPy backend does (one scatter of every
overlap's message, or below `dense_limit` owners the equivalent matrix product) in the
device's precision. Receipts are made in float64, so the script trains on the GPU and then
reads the test set with the CPU backend, and it checks that the two backends make the same
prediction on the first 2,000 test images.

## 4. How it learns

Batches of 256: a free settlement, a settlement nudged toward the label with the
cross-entropy nudge (`β = 0.1`, `T = 0.1`) and one nudged away, and the local update of
every seam and bias, exactly as in section 3 of [How a patch net learns](../HOW_IT_LEARNS.md).
Ten epochs, 235 updates each, `η` from 3 decaying by 0.8 per epoch.

## 5. Selection, then one look at the test set

10,000 of the 60,000 training images are set aside as a validation set. Each hidden size
(128, 256) is trained on the remaining 50,000 and read on the validation set; the best is
retrained on all 60,000 and reads the test set once, on the CPU backend. Logistic
regression and an MLP of the selected size, trained by Adam for 10 and 30 epochs, are fit
on the same 60,000 and read on the same test set.

## 6. The numbers

From `receipt.json`: validation hidden 128: 0.9611, hidden 256: 0.9669; the selected net
retrained on all 60,000 images; test set read once on the CPU backend in float64. Training
on mps float32; the CPU readout and the accelerated readout agree on every one of the first
2,000 test predictions, and the trained engine matches the owner-by-owner reference to
4.4·10⁻¹⁶.

| model | parameters | epochs | training | held-out accuracy |
|---|---|---|---|---|
| patch net 784-256-10, free/nudged rule, torch on Apple silicon | 204,359 | 10 | 1650 s | 0.9744 |
| logistic regression (lbfgs, 100 iterations) | 7,850 | 100 | 7 s | 0.9258 |
| MLP 784-256-10 (adam, batch 256) | 203,530 | 10 | 15 s | 0.9779 |
| MLP 784-256-10 (adam, batch 256) | 203,530 | 30 | 48 s | 0.9821 |

Read it plainly. The rule scales: 97.44% on MNIST with the same settings as the 8×8
digits, untouched, in ten epochs, and the float32 accelerator and the float64 reference
agree on every prediction. It does not reach the same-size MLP's 97.79% at ten epochs, and
it costs a hundred times the wall-clock, because each of the 37-step free settlements and
the two 11-step nudged settlements per batch is a full pass over a 1,050-owner net. The
untrained net scores 0.065. Nothing here was tuned on MNIST. (The 1,650 s were measured
with two other training runs sharing the machine; alone it is about half that.)

## 7. What is different from the MLP

Same shape, same count of numbers, same data, same ten epochs. The MLP makes each
prediction in one pass and learns from a backward pass; the patch net makes each
prediction by settling to rest and learns from two more settlements under a nudge. The
gap of 0.35 points at ten epochs is what the rule's small-nudge gradient estimate and its
plain (non-adaptive) step cost against Adam on this data; the gap in time is what tens of
steps per settlement cost against one pass. Section 6 of
[How a patch net learns](../HOW_IT_LEARNS.md) has the full comparison.

## 8. What to look at in the receipt

- `training_backend` and `test_accuracy_on_training_backend` next to `test_accuracy`: the
  same net read on the GPU and on the CPU.
- `backend_prediction_agreement_first_2k` and `conformance`.
- `validation`, `selected`, `comparison`, `boundary`, as in the digits example.

## 9. Things to try

- `--backend cpu` to run everything in float64 (slow: hours).
- A `GRID` with `{"hidden": 512}`, or `SCHEDULE["epochs"] = 20`.
- Fashion-MNIST: `fetch_openml("Fashion-MNIST")` has the same shape; nothing else changes.
