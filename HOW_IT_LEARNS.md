# How a patch net learns

This is the tutorial every rung of the ladder refers back to. It explains, with nothing
left implicit, what a patch net is, what happens when it settles, how it learns from a
label, from a teacher's move, and from a reward, and how all of that differs from a
feed-forward network trained by backpropagation. The library page
[learning](https://github.com/muellerberndt/cadence/blob/main/docs/learning.md) has the same
material in reference form with a worked six-owner example; this one is the walk-through.

## 1. Owners, overlaps, clamps

A patch net is a set of **owners**. Each owner holds two numbers of its own: a potential
`v` and an activation `s`. Owners are joined by **overlaps**: directed connections, each
with a signed strength `W`. An overlap from owner `i` to owner `j` delivers `W · s[i]`
into `j`'s inbox. That is the only way one owner affects another.

Some owners are **input owners**. They receive a **clamp**: a fixed drive, one number per
owner, that is the net's input. For the digits, an 8×8 picture is 64 input owners and the
clamp on each is its pixel's brightness in [0, 1]. For Pong, the screen is 192 pixels, supplied once per decision. Another 192 owners
carry a fading trace of earlier input activity through `Afterglow`.
For Connect Four, the board is two planes of 42 owners, the mover's discs and the
opponent's. The temporal context, when present, is also a clamp.

Some owners are **output owners**: one per class, one per column, one per action. Their
activations at rest are the net's answer. The rest are **hidden owners**.

There is no forward direction. A hidden owner has overlaps from the input owners *and*
from the output owners, and the output owners have overlaps back to it. The recurrent hidden/output block uses tied reciprocal **seams**, so `W[i→j] =
W[j→i]` there. Inputs have fixed drive and no incoming overlaps. This symmetry in
the free variables supports the energy argument in section 5.

## 2. A settlement, step by step

Every owner runs the same rule, at the same time, over and over:

    inbox[i] = sum of  W[e] · s[pre of e]   over every overlap e that ends at i
    total[i] = inbox[i] + clamp[i] + bias[i]
    v[i]    <- v[i] + dt · (total[i] − v[i])          the owner moves its potential toward its total drive
    s[i]     = act(v[i])                               and publishes an activation

with

    act(v) = tanh(v / 2)          for v ≥ 0        (0 at rest, 0.76 at v = 2, 0.96 at v = 4)
    act(v) = 0.1 · tanh(v / 2)    for v < 0         (a small negative number: a "leak")

At the start everything is 0. The input owners receive their clamp and, having no
overlaps into them, settle at `v = clamp`, publishing `act(clamp)`. Their activation
reaches the hidden owners through the input→hidden seams; the hidden owners start
publishing; that reaches the output owners through the hidden→output seams; the output
owners start publishing; and that comes *back* to the hidden owners through the same
seams. Every owner keeps repairing its own potential toward its own total drive. After a
few dozen steps no activation changes by more than a tolerance (3·10⁻³ while learning,
10⁻⁴ when reading out) and the net is **at rest**. That rest state is the settlement's
result. The most active output owner is the prediction.

Two things to notice. First, the answer is a joint state of the whole net: the hidden
owners at rest are shaped by the output owners as much as by the inputs. Second, the
same rule with the same code produced it; there is no separate "inference mode".

## 3. Learning from a label

Take one picture with its label. Learning is three settlements and one local update.

**The free phase.** Settle under the picture's clamp until rest. Call the activations
`s⁰`. The label has not been used; this is exactly the settlement that makes a
prediction.

**The nudged phases.** Starting from `s⁰` (not from zero), settle again with one extra
term in the total drive of the output owners, and only theirs:

    nudge[k] = β · (target[k] − p[k]),   p = softmax(s[outputs] / T)

The target is 1 for the label's owner and 0 for the others; `p` is a softmax over the
output owners with temperature `T`. With `β = 0.1` and `T = 0.1` this is a gentle pull on
the right owner and a gentle push on the wrong ones in proportion to how much they were
claiming. Settle until rest: `s⁺`. Then do the same with `−β`, a pull the other way: `s⁻`.

Why does anything but the output owners change? Because the output owners' activations
changed, and they have seams back to the hidden owners, so the hidden owners' inboxes
changed, so their rest potentials changed, by a small amount and in a direction that
depends on how each one is wired to the outputs that moved. That small change *is* the
credit the label sends to each hidden owner, and it arrived by the same overlaps the
prediction used.

**The update.** Every seam looks at its own two endpoints in the two nudged rest states:

    Δ W[i↔j] = η · ( s⁺[i] · s⁺[j] − s⁻[i] · s⁻[j] ) / (2β)

averaged over the batch. If the product of the two endpoints was larger when the net
was pulled toward the target than when it was pushed away, the seam strengthens; if
smaller, it weakens. Every owner does the same with itself:

    Δ bias[i] = η_b · ( s⁺[i] − s⁻[i] ) / (2β)

That is the entire rule. A seam reads two activations; an owner reads one. The free state is kept as the common initial state for both nudged phases. There is
no reverse-mode tape of the settlement trajectory. The output nudge is an error signal
and the softmax couples the output owners; seam updates use their endpoints.

**In numbers.** In the six-owner example in the library docs (two inputs, two hidden,
two outputs, input `(1.0, 0.2)`, label 1), the free phase rests with the two outputs at
0.036 and 0.052, a near coin flip. The `+β` nudge lifts output 5 to 0.078 and lowers
output 4 to 0.012; `−β` does the opposite (0.021 and 0.064). Hidden owner 3, which was
active at 0.14, ends at 0.143 under `+β` and 0.136 under `−β`. So the seam between
hidden 3 and output 5 sees products 0.143 · 0.078 versus 0.136 · 0.021 and strengthens by
+0.042; the seam between hidden 3 and output 4 sees 0.143 · 0.012 versus 0.136 · 0.064 and
weakens by −0.035; the input seam 0→3 strengthens by +0.016 because owner 3 itself ended
higher under `+β`. After this one update the outputs read 0.028 and 0.067. Run
`python examples/worked_update.py` in the cadence repo to see all fourteen seams.

**Over a dataset.** Batches of 32 pictures go through the three phases together (the
settlements are batched; the update averages over the batch). One pass over the training
set is an epoch; `η` starts at 3 and decays by 0.8 per epoch. That is all the digits
example does, twenty times over.

## 4. Learning from a teacher's move, and from a reward

**A teacher.** A game has no labels, but a search, or a player, can say which move it
prefers in any position. Label every position with the teacher's choice for the side to
move, and the problem is section 3 with one class per move. At play time the net settles
under the board, illegal moves are masked, and the most active output owner moves. The
raw net imitates the teacher; the Connect Four page can also examine four plies with
the supplied game rules. Its raw-policy, planning, and search-only results are separate.
Pong first imitates a teacher that demonstrates diagonal returns.

**A reward.** After imitation, Pong experiments through play. Its outcomes include
returning the ball, missing it, and making the opponent miss. The rule stays the same with one change. The net acts by settling and drawing
an action from `softmax(s[outputs] / T)`. After a rollout, each transition has a
**return** (the discounted sum of the rewards that followed) and an **advantage** (the
return minus the batch mean, divided by the batch standard deviation). Then, for each
transition, the target of the nudge is *the action that was taken*, and the nudge is
multiplied by the advantage:

    nudge[k] = β · A · (onehot(action)[k] − p[k])

An action that paid (`A > 0`) is pulled up in the state it was taken in; an action that
cost (`A < 0`) is pushed down. The seam update is unchanged. In the converged small-nudge limit, and up to the temperature scale, this estimates
the policy gradient (REINFORCE): the seams move to make actions with positive advantage
more likely. The reward entered through the nudge and nowhere else.

**Corrective lessons and retention.** Pong adds a reward when the opponent misses,
and discounts returns by 0.97. Teacher corrections and earlier demonstrations are
mixed with experience. The best checkpoint is retained on validation games; practice
is allowed to fail to improve it. See [Pong](04_pong/) for the measured outcome.
The teacher can inspect simulator velocity, while the paddle receives one frame
and a causal activity trace. These are different information privileges.

**Comparing futures.** A controller can branch a state, simulate candidate actions,
and evaluate their consequences before acting. Cadence documents and tests this
[deliberation pattern](https://github.com/muellerberndt/cadence/blob/main/docs/deliberation.md).
Imagined outcomes remain predictions; real outcomes supply learning evidence.

## 5. Why this is learning and not just a heuristic

For fixed input activations and symmetric recurrent seams, the continuous-time
dynamics of the free owners descend this energy:

    E = Σ_i ∫₀^{v[i]} u · act'(u) du − ½ Σ_{i,j} W[i→j] s[i] s[j] − Σ_i d[i] s[i]
    d[i] = clamp[i] + bias[i] + Σ_{k in input} W[k→i] s[k]

Here `i,j` range over free owners only. The fixed inputs contribute to `d`; their
one-way overlaps are not counted in the symmetric recurrent sum.

Stable equilibria can be local minima. A finite Euler step need not lower energy, and
a step budget need not reach equilibrium. The nudge adds `β · L` to the energy, with
`L = T · cross_entropy(p, target)` for the nudge written above. A theorem
(Scellier and Bengio, *Equilibrium Propagation*, 2017) says that, in the limit `β → 0` along a differentiable stable equilibrium branch, the
change in the product `s[i] · s[j]` between the free and the nudged rest, divided by `β`,
approaches minus the derivative of the loss with respect to `W[i↔j]`. So the seam update
is gradient descent on the loss, and the two-sided version (`+β` against `−β`) removes
the first-order error in `β`. The library's test suite checks this on random nets: the
contrast correlates above 0.9 with finite differences of the loss.

The argument requires an energy with reciprocal free-variable couplings, suitable
smoothness near the chosen equilibria, and phases converged on the same stable branch.
The leaky activation is piecewise smooth with a kink at zero; its slope varies with
potential. Tolerances and budgets approximate these requirements, and convergence
should be checked rather than assumed.

## 6. What is different from a feed-forward network with backprop

A feed-forward network computes its answer in one pass, layer by layer, and never comes
back. To learn, a controller outside the network computes the loss, then runs a second,
different computation backward: the error at the outputs is multiplied by the transposed
weights, layer by layer, using stored activations from the forward pass, to produce a
gradient for every weight. Each weight then receives its gradient from the controller.

A patch net has no forward pass and no backward pass. It has one rule that every owner
runs until the net is at rest. To learn, it runs that same rule again with the outputs
nudged, and every seam updates itself from what its own two endpoints did. Credit reaches
a hidden owner not as a number computed elsewhere but as the change in its own rest
state, arriving through the seams it already has.

| | feed-forward + backprop | patch net + free/nudged rule |
|---|---|---|
| answer | one forward pass | a settlement to rest |
| influence | forward only | both ways, over the same seams |
| learning signal | an error computed by a controller and propagated backward with transposed weights | a nudge on the outputs, felt by the rest of the net through its own seams |
| what a weight reads to update | its gradient, delivered | its own two endpoints, twice |
| stored state | layer activations for the backward pass | free and nudged states; no trajectory tape |
| weight sharing | none needed | a seam is one strength in both directions |
| is it a gradient? | exactly | in the small-nudge limit, and checked numerically |
| cost per update | 2 passes | 3 settlements of tens of steps |
| the network at rest | is not a thing; the network is a function | is a state you can inspect, clamp, ablate, and watch |

The receipts measure each concrete task, including controls with the same task
information. Supplied-rule solvers and exact dictionaries can solve their respective
tasks too. Pong's reward-only MLP control has no teacher phase and therefore does not
establish a matched training-efficiency comparison. Repeated settlement has a cost;
learning without a reverse-mode tape does not by itself establish a speed advantage.

## 7. Glossary

- **owner**: one unit; holds a potential and an activation.
- **overlap**: a directed connection with a strength; a **seam** is an overlap and its reverse, sharing one strength.
- **clamp**: the fixed drive on the input owners; the net's input.
- **settlement**: running the owner rule until no activation moves more than a tolerance; **rest** is its end state.
- **free phase**: a settlement under the clamp alone; its output owners are the prediction.
- **nudge**: extra drive on the output owners toward (or away from) a target; **β** its strength; **T** the softmax temperature.
- **contrast**: the difference of an endpoint product between the two nudged rest states, divided by `2β`; the seam's update is `η` times it.
- **advantage**: how much better a transition's return was than average; the weight of the nudge when learning from reward.
- **receipt**: the canonical record of a run, bound to its code and data by digest, with every derived number recomputable.
