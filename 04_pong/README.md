# 04 · Pong

The paddle sees **one 12×16 pixel frame per decision**. It carries a fading trace
of input activity through Cadence's `Afterglow`; previous frames are not supplied
as a second observation. The frame and 192 trace owners drive 128 hidden owners
and three action owners. Each action follows a recurrent settlement.

```bash
# From the repository root
python -m pip install -r requirements.txt
python serve.py pong
python serve.py pong --learn  # save lessons and resume learning locally
```

You control the left paddle with the mouse or arrow keys. The right paddle is
the trained patch net. The **teacher-only paddle** toggle loads the checkpoint
before practice and disables local learning for that paddle.

## Imitate, experiment, revisit

A teacher demonstrates diagonal returns: aim off-centre to make the other paddle
miss. The teacher uses simulator velocity; the learner must infer motion from its
current frame and activity trace. Training collects 51,200 teacher-labelled states
with occasional random actions, then runs 12 imitation epochs.

Practice follows for 30 iterations of 64 environments × 64 steps. Outcomes shape
advantage-weighted local nudges; teacher corrections and initial demonstrations
are rehearsed alongside experience. The trace is advanced once per real step and
reset at point boundaries. Replay uses the recorded causal trace and never
advances live memory with shuffled samples.

Every ten iterations, validation games decide whether to retain the candidate
checkpoint. The shipped run retained the imitation checkpoint: practice did not
improve validation win rate. Both checkpoints remain available for inspection.
A subsequent game supplies a saved learning episode; it need not improve win rate.

The default trace decay is 0.5, with no surprise weighting (`focus=0`). This is
activity memory, not a learned world model or a claim to reproduce a whole brain.
The layout and training budget were chosen in small validation experiments;
receipts record the final run and its separate test seeds.

## Evaluate the shipped browser weights

```bash
python 04_pong/evaluate.py --net 04_pong/net.json --seed 81234 --points 1000
python 04_pong/evaluate.py --net 04_pong/net.json --opponent-skill 1 --points 1000
```

Report **points won**, losses and draws, as well as balls returned. A long flat
rally can return every ball without winning. A point is drawn after 400 steps.
Opponent skill is the probability that the scripted paddle follows the ball each
step. Skill 1 still denotes a particular tracker, not optimal play or a human.
The receipt contains testing at skills 0.7 and 1.0 across several seeds.

<!-- game-results -->
| tracker skill | wins | losses | draws | points won |
|---|---:|---:|---:|---:|
| 0.7, seeds 101–103 | 2,831 | 173 | 0 | 94.2% |
| 1, seeds 101–103 | 17 | 126 | 2,863 | 0.6% |
<!-- /game-results -->

The always-moving tracker almost always draws. High return rate is useful, but it
is not evidence of winning that matchup.

The reward-only PyTorch MLP control does not receive teacher demonstrations.
It also uses a different reward/discount configuration. Its result is an additional
control, not a matched claim that local learning
outperforms backpropagation. No human win rate has been measured.

## Train on CPU or GPU

```bash
python -m pip install torch
python 04_pong/train.py --output runs/my-pong/receipt.json
python 04_pong/train.py --device cuda --output runs/my-pong-gpu/receipt.json
```

CUDA needs a compatible PyTorch installation and GPU. `--device mps` is available
on supported Macs. CPU remains the default; the shipped run was measured on CPU.
An output directory keeps trial nets separate from the public model. To promote a
run, copy its receipt, both JSON nets, learner and rehearsal checkpoints into
`04_pong/`, then run `python 04_pong/build_page.py`.

With `--learn`, completed points are taught locally in Python, saved, and mixed
with earlier lessons. The browser performs inference; no cloud service receives
its games. [Training a player](../TRAINING.md) explains persistence and retention.
Static HTML remains playable with frozen weights.
