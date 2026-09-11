# 04 · Pong

A paddle learns Pong from pixels and reward, with the same rule as the classifiers and
one change: the target of the nudge is the action that was taken, and the strength of the
nudge is that action's advantage. Actions that paid are pulled toward; actions that cost
are pushed away. That is the policy gradient, and it enters through the nudge alone. Read
[How a patch net learns](../HOW_IT_LEARNS.md) first; this page is about how a reward
becomes a nudge, how the paddle learns to be in the right place, and what went wrong the
first time.

```bash
pip install "cadence-net>=0.7" torch        # torch is used for the backprop baseline only
python train.py                             # about ten minutes: the patch net, the baseline, the imitation net, the receipt
python build_page.py                        # embeds net.json into index.html; open it and play
python train.py --verify receipt.json
```

## 1. The game

A 12 × 16 pixel field. The agent's paddle is on the right, three pixels tall, and moves one
row per step; a scripted opponent on the left tracks the ball with a 70% chance each step.
The ball moves one column per step and −1, 0, or +1 rows, bounces off the top and bottom,
and leaves a paddle with a vertical velocity set by where it struck (top third: up;
middle: flat; bottom third: down). Reaching a paddle's column uncovered is a miss and
ends the point; so does a rally of 400 steps. `pong.py` runs 64 games at once as arrays.

## 2. From pixels to a clamp

The net sees two frames, the current one and the one before: 2 × 192 pixels, ball at 1.0,
paddles at 0.6, clamped onto 384 input owners. Two frames because one frame does not say
which way the ball is going; with a single frame both learners in this example returned
fewer than 40% of diagonal balls. Three output owners: up, stay, down. Hidden layer: 32
owners. `cadence.layered(384, 32, 3, density=1.0)`, 12,806 numbers to learn.

## 3. Acting

To act, the net settles under the two frames and the three output activations become a
policy: `p = softmax(s_out / T)` with `T = 0.2`. During training the action is *drawn* from
`p`, so the paddle explores; when the net is evaluated, and on the page, the most active
output is taken. A settlement is about 45 steps at the training tolerance; on the page it
runs every step of the game.

## 4. From reward to a nudge

Sixty-four games run for 64 steps: 4,096 transitions of (frames, action, reward). Each
transition's **return** is the discounted sum of the rewards that followed it within its
point (`γ = 0.5`, so mostly the next two or three steps), and its **advantage** is the
return minus the mean return of the rollout, divided by the standard deviation.

Then the 4,096 transitions are learned in batches of 256 with the rule of section 3 of
[How a patch net learns](../HOW_IT_LEARNS.md), where the target of the nudge is the action
that was taken and the nudge is multiplied by the advantage:

    nudge[k] = β · A · (onehot(action)[k] − p[k])       on the three output owners

A transition whose action paid pulls that action's owner up in that state; one whose
action cost pushes it down. Every seam then reads the difference of its own two endpoint
products between the `+β` and `−β` settlements, as always. Summed over the batch this is
`Σ A · ∇ log p(action | frames)`, the REINFORCE gradient.

One thing is added to the step, and it is still local. Each seam keeps a running average
of its own contrasts (forgetting factor 0.9) and a running mean of their squares (0.999),
both corrected for their short history, and steps by `η = 5·10⁻⁴` times the average over
the root of the mean square. That is the per-parameter step of an adaptive optimiser,
read from a seam's own history and nothing else; the baseline gets the same thing from
Adam. 300 iterations, no anneal. With the plain step (`η` from 2 decaying by 0.99) the
paddle stalled at 78% of balls returned whatever else was varied; the adaptive step is
what moved it, and the numbers below say how far.

## 5. How the paddle learns to be in the right place

The reward is the interesting part. The obvious reward is `+1` when the paddle returns
the ball and `−1` when it misses. But that arrives at the end of a rally, many steps after
the moves that decided it, and the first version of this example, with that reward plus
a small penalty proportional to the distance between paddle and ball, learned something
that looked like Pong and was not. Here is what its policy did, as a table of the action
it chose against the ball's row minus the paddle's centre row, over many positions:

    ball − paddle:  −5   −3   −1    0   +1   +3   +5
    up             5/6  5/8  5/10 5/10 5/10 3/8  1/6
    stay            0    0    0    0    0    0    0
    down           1/6  3/8  5/10 5/10 5/10 5/8  5/6

Never "stay", and a coin flip between up and down whenever the ball is within a row of
the paddle. The net had learned an *absolute* rule, "ball high, go up; ball low, go down",
using the ball's row alone and ignoring where its own paddle was, because that rule
collects some of the delayed reward and the learner found it first. It returned 79% of
balls against the lenient scripted opponent and looked bad against a person.

The fix is not in the rule, it is in the credit: each step's reward now includes how much
closer the paddle's centre came to the ball's row during that step. This is potential-based
shaping (Ng, Harada, and Russell 1999): the extra terms telescope along any trajectory, so
the best policy is unchanged, but each move is told at once whether it helped. With the
short credit horizon, the advantage of "up" in a state where the ball is above the paddle
is positive and of "down" negative, so the seams from the pixels that mean "ball above my
paddle" to the up owner strengthen, and the pixels that mean that include the paddle's
own pixels. The net learns the *relative* rule, the one that tracks. The table for the
trained net is in section 6.

## 6. The numbers

From `receipt.json`: 300 iterations of 64 games × 64 steps for the two reward learners;
greedy play on fresh seeds until 1,000 points had ended; one laptop core, shared with
other runs. The reward is +1 for a return, −1 for a miss, and the potential-based shaping
of section 5 with a credit horizon of γ = 0.5.

| policy | learned from | parameters | training | balls returned | returns per point |
|---|---|---|---|---|---|
| patch net 384-32-3, reward-nudged, adaptive local step | reward | 12,806 | 163 s | 88% | 3.09 |
| MLP 384-32-3, REINFORCE with Adam | reward, same rollouts | 12,419 | 13 s | 93% | 4.00 |
| patch net 384-32-3, free/nudged rule | a scripted tracker's moves, 25,600 rows | 12,806 | 22 s | 96% | 5.64 |
| untrained patch net | | 12,806 | — | 16% | 0.14 |
| scripted tracker (for scale) | | | | 99.7% | |

The policy tables, action chosen against the ball's row minus the paddle's centre, over
every ball row and paddle position with the ball one column away and coming level:

    reward-trained            ball − paddle:  −5    −3    −1     0    +1    +3    +5
    up                                        6/6   8/8   8/10  5/10  6/10  2/8   0/6
    stay                                      0     0     0     0     0     0     0
    down                                      0/6   0/8   2/10  5/10  4/10  6/8   6/6

    imitation-trained         ball − paddle:  −5    −3    −1     0    +1    +3    +5
    up                                        6/6   8/8   9/10  2/10  0     0     0
    stay                                      0     0     1/10  4/10  1/10  0     0
    down                                      0     0     0     4/10  9/10  8/8   6/6

Read it plainly. The same net, the same rule, the same seams: taught by a tracker's moves
it becomes a tracker (the second table) and returns 96% of balls; taught by reward it
learns the relative rule at a distance, up when the ball is three rows above, down when
it is three below, and stays a coin flip within a row or two of the ball (the first
table), returning 88%, while backprop with Adam on the same rollouts reaches 93%. The
adaptive local step is what carried the reward-trained paddle from 78% to 88%: with the
plain step, every variant of the setup (nudge strength, settle tolerance, batch size,
momentum, per-seam normalisation without the history correction, immediate and
potential-based credit, one and two frames) stopped at 79%. With it, three seeds land at
88%, 88% and 90%; longer training, an anneal, a larger nudge, a tighter settle, a larger
hidden layer and a longer credit horizon were each tried and none passed 90%. The five
points left to the baseline are what an exact gradient does with the same noisy
advantages that a small-nudge estimate does not; the receipt has the full per-iteration
history for both. The page ships both paddles; the imitation-trained one is the default
opponent because it plays best, and the toggle is there so you can feel the difference.

## 7. What is different from the baseline

The baseline is a 384-32-3 network of the same shape trained by REINFORCE with Adam: the
same rollouts, the same reward and advantages, the same 300 iterations, and a backward
pass to turn `A · ∇ log p` into weight updates, with Adam's per-parameter step. The patch
net does that with two nudged settlements per batch, a local rule, and the same
per-parameter step read from each seam's own history. Section 6 of
[How a patch net learns](../HOW_IT_LEARNS.md) has the full comparison.

## 8. The page

You are the coral paddle on the left, with the mouse or the arrow keys; the net is the
gold paddle on the right. Every step the page clamps the two frames onto the 384 input
owners, settles the net from rest in JavaScript, and moves the paddle on the most active
output owner. The bars show the three output owners; the strip shows the hidden owners.
The physics in the page is the physics of `pong.py`, step for step, and the net in the page
is `net.json`, the trained net's dense overlap matrix.

## 9. Things to try

- `SHAPING = 0.0` in `pong.py` to reproduce the absolute-rule policy of section 5.
- `HIDDEN = 0` with `skip=True` in the wiring: a direct pixels-to-actions net.
- `opponent_skill=1.0` in `Pong` for a perfect opponent, and see how rallies change.
