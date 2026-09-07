# 04 · Pong

A paddle learns Pong from pixels and reward, with the same rule as the classifiers and
one change: the target of the nudge is the action that was taken, and the strength of the
nudge is that action's advantage. Actions that paid are pulled toward; actions that cost
are pushed away. That is the policy gradient, and it enters through the nudge alone.

```bash
pip install "cadence-net[accel]>=0.2"      # torch is used for the backprop baseline only
python train.py                             # a few minutes: the patch net, then the baseline, then the receipt
python build_page.py                        # embeds net.json into index.html; open it and play
python train.py --verify receipt.json
```

## The game

A 12×16 pixel field. The agent's paddle is on the right, three pixels tall, and moves one
row per step; a scripted opponent on the left tracks the ball with a 70% chance each
step. The ball moves one column per step and −1, 0, or +1 rows, bounces off the top and
bottom, and leaves a paddle with a vertical velocity set by where it struck. Reaching a
paddle's column uncovered is a miss and ends the point; so does a rally of 400 steps.

The net sees two frames, the current one and the one before, so the ball's direction is
visible: 2 × 192 pixels, ball at 1.0, paddles at 0.6, clamped onto 384 input owners. Three
output owners: up, stay, down. (With one frame both learners returned fewer than 40% of
diagonal balls, because a single frame does not say which way the ball is going.)

## How it learns

Sixty-four games run in parallel for 64 steps; every step is a free settlement and a draw
from the softmax of the three output activations at temperature 0.1. The reward is +1 for
returning the ball, −1 for missing it, and −0.05 times the paddle-to-ball row distance
each step so that credit does not have to travel a whole rally. Discounted returns are
normalised across the rollout to give advantages; then the 4,096 transitions are learned
in batches of 256: one free settlement, one settlement nudged toward the action taken with
the advantage as the nudge's weight, one nudged away, and the local update.

The baseline is the same shape as a plain MLP, 384-32-3, trained by REINFORCE with Adam
on the same rollout budget, same reward, same advantages.

## The numbers

From `receipt.json`: 300 iterations of 64 games × 64 steps for each learner, greedy play
on fresh seeds until 1,000 points had ended, one laptop core.

| policy | parameters | training time | returns per point | balls returned |
|---|---|---|---|---|
| patch net 384-32-3, reward-nudged | 12,806 | 721 s | 2.04 | 79% |
| MLP 384-32-3, REINFORCE with Adam | 12,419 | 10 s | 0.98 | 59% |
| untrained patch net | 12,806 | — | 0.15 | 16% |
| scripted tracker (for scale) | — | — | — | 99.7% |

Both learners cross a 0.7 hit rate in their rollouts within the first 50 iterations
(patch net at 43, baseline at 33) and peak near 0.88 to 0.89; the difference is what the
greedy policy does afterwards, and there the reward-nudged net keeps twice the rally
going. Wall-clock is the other way round by a factor of seventy: a settlement is tens of
steps, a forward pass is one. Neither learner is near the scripted tracker, so this is a
statement about 300 iterations of policy-gradient learning from pixels, not about Pong.
The receipt has the full per-iteration history of both.

## The page

You are the coral paddle on the left, with the mouse or the arrow keys; the net is the
gold paddle on the right. Every step the page clamps the two frames onto the 384 input
owners, settles the net from rest in JavaScript, and moves the paddle on the most active
output owner. The bars show
the three output owners; the strip shows the hidden owners. The physics in the page is the
physics of `pong.py`, step for step.
