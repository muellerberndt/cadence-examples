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

From `receipt.json`: validation hidden 256: 4.01 bits, hidden 512: 3.94 bits; the selected net trained five epochs on the
first 300,000 characters; the last 100,000 read once; softmax temperature 0.1 chosen on
validation; the training backend was mps float32 with three other runs sharing the machine.

| model | parameters | epochs | training | bits per character | next-character accuracy |
|---|---|---|---|---|---|
| patch net 1040-512-65, free/nudged rule | 569,457 | 5 | 10903 s | 3.33 | 37.2% |
| bigram (Laplace) | 4,225 | 1 | 0 s | 3.71 | 26.5% |
| MLP, same window, Adam | 566,337 | 5 | 12 s | 3.11 | 41.3% |
| one-layer transformer, Adam | 59,521 | 5 | 31 s | 3.08 | 42.1% |

Read it plainly. At this window and budget every model is weak, and the patch net is the
weakest by about a quarter of a bit: a bigram model is a fair summary of what a sixteen-
character window has learned in five epochs, and the one-layer transformer with a tenth
of the parameters is the best of the lot by a hair. The ranking is the interesting part.
On every supervised rung with a dense, low-dimensional input the rule matched a same-shape
backprop net; here, with 1,040 mostly-silent one-hot input owners and 65 classes, it
trails, and it costs three hours where the MLP costs twelve seconds. Two things are worth
trying before reading more into it: a learned character embedding (a first layer of seams
from 65 owners to a few dozen, shared across positions) instead of 1,040 raw owners, and a
learning-rate schedule that does not decay to a tenth by the fifth epoch.

What the net wrote from a held-out opening, drawing characters at the validation temperature:

    t I tringto
    s pating and hyrour'd nothone of the stes gecon me ofAat Whonot owringRongoublyoI wollgnkent now that sover heve doHZ not githt now lo tho gareit te live fore yourfettis bot it of newi&qy, and ou muce salp too a to cominge ofeby wnour? My guregrarting head 
    RoGeryI youronse that limf with Mornesy beplide connt theik wee coun:
    'sene difvoI seart,
    The a may lighte heabKs to murthitW the 


## 5. What a window cannot do

A transformer or a recurrent net carries context across the whole sequence; this net sees
sixteen characters and nothing before them. That is the model class, not the rule: the
MLP in the table has the same window and the same blindness. Carrying state across time
with a settlement, so that the rest state of one beat becomes part of the clamp of the
next, is the open problem for this rule, because credit over time is not local in time.
The [library docs](https://github.com/muellerberndt/cadence/blob/main/docs/games.md) say
where that stands.
