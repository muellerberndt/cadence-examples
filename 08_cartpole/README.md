# 08 · Cart-pole

The classic control task from reward: a cart on a track, a pole hinged on top, push left
or right each step, keep the pole up. This rung is the Pong recipe on a benchmark
everyone knows, with a state instead of pixels. Read [How a patch net learns](../HOW_IT_LEARNS.md)
section 4 for how a reward becomes a nudge.

```bash
pip install "cadence-net[accel]>=0.2"      # torch is used for the backprop baseline only
python train.py                             # a few minutes
python train.py --verify receipt.json
```

## 1. The task

The standard cart-pole physics (Barto, Sutton, and Anderson 1983): mass, pole length,
force, and time step as in the common implementations; an episode ends when the pole
leans past 12 degrees or the cart leaves the track, or after 500 steps. Reward: +0.02 for
every upright step, −1 on falling. Sixty-four episodes run in parallel.

## 2. What the net sees

Four numbers (cart position and velocity, pole angle and angular velocity), each clamped
onto twelve input owners as a bump over bins: a place code, so the net reads positions on
a map rather than raw floats, and every state is a pattern of a few active owners. Thirty-
two hidden owners; two output owners, one per push.

## 3. How it learns

Exactly as Pong: act by drawing from the softmax of the output activations, collect 100
steps of 64 episodes, compute discounted returns (γ 0.97) and normalise them into
advantages, then learn the 6,400 transitions in batches of 256 with the nudge's target
the action taken and its weight the advantage. 120 iterations. The baseline is the same
net trained by REINFORCE with Adam on the same rollouts.

## 4. The numbers

From `receipt.json`: 120 iterations each; greedy evaluation on 200 fresh episodes.

| policy | parameters | training | mean episode length | episodes balanced to 500 | best training rollout |
|---|---|---|---|---|---|
| patch net 48-32-2, reward-nudged | 1,683 | 180 s | 154 | 0% | 404 |
| MLP 48-32-2, REINFORCE with Adam | 1,634 | 4 s | 392 | 18% | 500 |
| untrained | 1,683 | — | 17 | 0% | |

Read it plainly. Both learners find balancing policies within ten iterations (both pass a
mean length of 200 by iteration 8) and both are unstable afterwards, as REINFORCE without
a value baseline is on this task; the backprop net ends higher. On every reward rung so
far the nudged rule learns, and learns less than backprop with Adam from the same
rollouts. On the supervised rungs it is at parity. That is the pattern this ladder has
found, and the Pong tutorial has the diagnosis: the reward's credit is noisy, and the
rule's small-nudge gradient estimate suffers from that noise more than an exact gradient
with per-parameter step sizes does.

## 5. Things to try

- More iterations, or a learning-rate decay of 0.98: the late instability is the first
  thing to fix.
- A value baseline: subtract a learned estimate of the return before normalising. That is
  a second patch net trained with the quadratic nudge on the return.
- Pixels: render the cart and pole into a 16×32 frame and clamp that instead.
