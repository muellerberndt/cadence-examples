# 05 · Text

The same rule as the classifiers on language: predict the next character of Shakespeare
from the last sixteen. This rung asks a narrow question, whether a patch net learns the
statistics of text at all, and answers it against a bigram model, a same-window MLP, and
a one-layer transformer trained on the same text. It is a windowed model, not a recurrent
one, and the tutorial says what that costs. Read [How a patch net learns](../HOW_IT_LEARNS.md)
first.

```bash
pip install "cadence-net[accel]>=0.2"
python train.py                    # a few hours on an M-series Mac; downloads the corpus once
python train.py --verify receipt.json
```

## 1. The data

Tiny Shakespeare (the 1.1 MB text from Karpathy's char-rnn, pinned by digest in the
receipt): 65 distinct characters. The first 300,000 characters train, the next 50,000
validate, the last 100,000 are the held-out test. Nothing from the tail is seen before the
end.

## 2. From text to a clamp

The last sixteen characters become sixteen blocks of 65 input owners, one block per
position, with the owner of that position's character clamped at 1: 1,040 input owners.
A hidden layer of 256 or 512 owners; 65 output owners, one per character. The output
owner that is most active at rest is the prediction; the softmax of the output
activations at a temperature chosen on the validation text is the model's distribution,
and its cross-entropy on the test tail is the bits per character.

## 3. How it learns

Classification with 65 classes, exactly section 3 of [How a patch net learns](../HOW_IT_LEARNS.md):
a free settlement under the window, a settlement nudged toward the next character with
the cross-entropy nudge and one nudged away, and every seam moving on its own two
endpoints. Batches of 512 windows; five epochs over the training text on the torch
backend.

Writing is settling in a loop: clamp the last sixteen characters, settle, draw a character
from the output distribution, slide the window, repeat. The receipt carries 400
characters written from a held-out opening.

## 4. The numbers

{{NUMBERS}}

## 5. What a window cannot do

A transformer or a recurrent net carries context across the whole sequence; this net sees
sixteen characters and nothing before them. That is the model class, not the rule: the
MLP in the table has the same window and the same blindness. Carrying state across time
with a settlement, so that the rest state of one beat becomes part of the clamp of the
next, is the open problem for this rule, because credit over time is not local in time.
The [library docs](https://github.com/muellerberndt/cadence/blob/main/docs/games.md) say
where that stands.
