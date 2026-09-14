# 02 · Recall and correction

A memory learns on each write: read the current answer, compare it with the new
observation, and write the residual. This example uses the current
`cadence.FastSeams(rule="delta")`; the browser implements the same update.

```bash
python serve.py recall
python 02_recall/train.py
python 02_recall/train.py --verify 02_recall/receipt.json
```

Write a key/value pair, repeat it, then change the value. The next query should
return the correction. No offline imitation stage is needed for an explicitly
addressed memory: the observed value already supplies its teaching signal.

```python
import numpy as np
import cadence as cd

memory = cd.FastSeams(np.arange(2), np.arange(2, 4), rule="delta")
key = np.array([[1., 0.]])
memory.observe(key, np.array([[1., 0.]]))
memory.observe(key, np.array([[0., 1.]]))
print(memory.recall(key))  # [[0., 1.]]
```

The benchmark tests 4–128 distinct keys and corrections after twenty additional
writes of an old value, over three seeds and 100 trials per size. Residual writes
recover every tested distinct and revised association. Additive writes retain the
old value in every revision test. An exact dictionary also succeeds.

There are 16,384 mutable memory entries and zero slow trained parameters. Orthogonal
one-hot keys supply the addressing structure; this result is not a learned parser
or a general advantage over transformers. [Example 05](../05_memory/) tests correlated
keys and includes a transformer with the same explicit key/value inputs.

The receipt binds the producer and installed Cadence source. Historical additive
memory and its older transformer comparison remain in Git history.
