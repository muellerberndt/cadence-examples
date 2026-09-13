# 02 · Recall

A memory that is the seams. A context of key-value pairs, then a query key: what was its
value? A patch net answers with no training at all, at any context length, and a
transformer trained on the task in the same script does not. This is the rung where the
claim "attention is one settlement step of a Hebbian memory" is a number. Read
[How a patch net learns](../HOW_IT_LEARNS.md) first for owners, seams and settlement; no
learning rule is needed here.

```bash
pip install "cadence-net>=0.8" torch
python train.py                    # about seven minutes; the transformer baseline is most of it
python train.py --verify receipt.json
```

## 1. The task

A vocabulary of 128 symbols. An episode draws `n` distinct keys, a value for each (values
may repeat), and asks for the value of one of the keys. `n` runs over 4, 8, 16, 32, 48, 64,
96 and 128 pairs; a hundred episodes per length, three seeds. At 128 pairs every key in the
vocabulary is in the context.

## 2. The net

Two ranges of owners, 128 key owners and 128 value owners, every key joined to every value
by a seam that starts at zero: 32,768 seams, no other structure. Writing a pair is one local
Hebbian outer product: the seam between the key's owner and the value's owner gains one
unit, in both directions, because the two were active together. Nothing else in the net
changes. Reading is a settlement with the key's owner clamped: the value owners come to
rest, and the most active one is the answer.

One settlement step after the clamp is exactly an attention read, the softmax over stored
patterns of a modern Hopfield network (Ramsauer et al., 2021), with the stored patterns as
seams instead of a window of tokens. More steps clean the answer. The receipt records both:
`patch_one_read` (two steps: the key lights, one transport) and `patch_settled` (to a
tolerance of 1e-4, at most thirty steps).

## 3. The comparison

A two-layer transformer (model width 64, four heads, 108,224 parameters), the smallest
architecture that can express recall with an induction head, trained for 5,000 Adam steps
of 32 episodes on contexts of 2 to 32 pairs, then tested at every length. The literature's
result is that attention learns recall inside the lengths it was trained on and does not
carry it beyond them; in this budget the baseline did not learn the task at all (final
training loss 3.99 against a chance level of 4.85 nats), and the receipt says so rather than
tuning until it does. Four times the budget (20,000 steps, loss 3.45) gave 0.27 at 4 pairs,
so the row is a statement about this architecture at this size and budget, not about
attention; the comparison the rung rests on is the one in the receipt's first two rows,
exact recall at every length with nothing trained.

## 4. The numbers

From `receipt.json`, three seeds, a hundred episodes per length:

| pairs | 4 | 8 | 16 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|
| patch net, one read | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| patch net, settled | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| transformer, trained | 0.30 | 0.10 | 0.05 | 0.03 | 0.02 | 0.03 | 0.01 | 0.01 |

The patch net has no trained parameter and no window: the context length is whatever was
written, and a hundred episodes at every length take a few seconds on a laptop core,
against about two minutes of training for the baseline per seed. The reference
engine settles the same memory owner by owner and agrees with the fast engine to 0.0.

## 5. The page

`index.html` is the same memory in JavaScript: 128 key owners, 128 value owners, 32,768
seams at zero, the rule of `train.py` (unit slope, leak 0.1, clamp 3.0, at most thirty steps
to a tolerance of 10⁻⁴). Write a random context of 4 to 128 pairs, or add a pair of your own,
then click any key: the bars replay the eight most active value owners coming to rest, the
strip shows all 128, and the caption gives both readings, the settled one and the one-step
attention read. Under the receipt's protocol (a hundred random contexts per length, distinct
keys, random values) the page scores 1.00 at every length, settled and in one read. Nothing
in it was trained; the memory is whatever you wrote.

## 6. What it shows, and what it does not

It shows the thing the paper argues: the retrieval a transformer does with attention over
a window, a patch net does with seams it wrote itself, and one settlement step is the
same operation. It does not show that a patch net is a language model, and the values
recalled here are symbols, not text. What a context of pairs cannot test is interference:
with a vocabulary of 128 and distinct keys there is none, and a memory of this kind
degrades when keys resemble each other. That is the next question, and it is not on this
rung.
