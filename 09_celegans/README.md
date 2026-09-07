# 09 · C. elegans

The published connectome of the worm as a patch net: 300 neurons, 3,638 chemical
connections with synapse counts, 1,093 gap junctions. The net learns its seam strengths
and signs from four textbook facts with the free/nudged rule, and is then scored on
seventeen classical laser-ablation phenotypes it was never shown, against a wiring whose
postsynaptic endpoints were shuffled. This rung is the library's original use, a measured
wiring under a declared protocol, and it is the rung where the local rule hits a wall.
The tutorial says where and why. Read [How a patch net learns](../HOW_IT_LEARNS.md) first.

```bash
pip install "cadence-net>=0.2"
python train.py                    # a few minutes
python train.py --verify receipt.json
```

## 1. The wiring and its custody

`fixture.json` is the connectome as the observer patch net programme pinned it: the
hermaphrodite edge list and cell classes from the Varshney/Chklovskii toolbox and the
c302 NeuroML model, each with a URL and a SHA-256 in the fixture's `sources` block, and
the fixture's own digest. Every neuron is an owner. Every chemical connection is a seam
from the presynaptic owner to the postsynaptic one with the synapse count as its contact
count and a sign from the presynaptic neurotransmitter (GABA inhibitory, the rest
excitatory). Every gap junction is a seam in both directions. That is 5,044 seams.

## 2. Stimuli, readout, facts, phenotypes

Six stimuli clamp sensory neurons: gentle anterior touch (ALM, AVM), posterior touch
(PLM), the aversive chemical channel (ASH), nose touch (ASH, FLP, OLQ), a tap (all touch
receptors), and rest. The readout is the one piece of physiology the model is told rather
than shown: B-type ventral cord motor neurons (DB, VB) drive forward locomotion and
A-type (DA, VA) backward, so *speed* is the B-type mean activation minus the A-type mean.

Four training facts: at rest the worm moves forward; anterior touch reverses it; posterior
touch accelerates it; ASH reverses it. Seventeen held-out rows, each a laser-ablation
study: remove the named neurons, settle under the stimulus, and check the predicate
(no reversal, reversal reduced, reversal retained, reversal enhanced, forward impaired,
no acceleration) against the intact worm's speeds. Two rows remove the stimulated
neurons themselves and are marked trivial.

## 3. How it learns, and where it stalls

Every update contrasts all four training facts at once: a free settlement under each
stimulus, a settlement nudged toward the fact's motor pattern (B-type owners at 1 and
A-type at 0 for forward and acceleration, the reverse for reversal) and one nudged away,
and every seam moving on its own two endpoints. Biases learn too, which is how a worm at
rest can move at all: rest is an exact zero in this rule.

What happens depends on how synapse counts become seam strengths.

**Count convention.** Drive per contact times count, the connectome lanes' convention,
with the gain chosen as the largest at which no training stimulus lights more than a
tenth of the net. The rule learns nothing usable: every stimulus ends with the same
speed, the training facts are not met, and the held-out rows fail for the measured and
the shuffled wiring alike. The mechanism is visible in the settlements. A nudge on the
motor owners reaches upstream only through seams that end on owners the nudge moved;
chemical synapses are directed, so the credit stops at the command interneurons, and
those sit saturated at any gain that reaches the motor neurons at all. The seams that can
learn are the last hop, and the last hop cannot satisfy facts that need the anterior and
posterior pathways to act differently.

**Fan-in convention.** Each owner scales its inbox by the square root of its total contact
count, an owner-local normalisation, and the gain is the one of three that passes the
most training facts. The rule now learns something, and the wiring shows through it:
the measured wiring passes more held-out rows than the shuffled one. But the speeds are
still the same under every stimulus, the two facts that pass are the two reversals, and
the net is saturated. What passes are the rows where removing a command interneuron
changes a common-mode speed, which the real wiring carries and the shuffled one does
not. It is a structural signal, not a behavioural model.

## 4. The numbers

From `receipt.json`: eight seeds of the measured wiring and eight shuffled wirings per
convention, 600 updates each, gains chosen per arm on the training facts alone (fan-in
gains chosen: [0.6]). "Active" is the fraction of neurons above 0.5 under the touch
stimuli after learning; the count convention's sparsity cap was 0.10 at calibration.

| convention | wiring | training facts | held-out rows | non-trivial rows | active |
|---|---|---|---|---|---|
| count | measured | 0.00 / 4 | 0.00 / 17 | 0.00 / 15 | 0.25 |
| count | shuffled | 0.00 / 4 | 0.00 / 17 | 0.00 / 15 | 0.37 |
| fan-in | measured | 2.00 / 4 | 5.25 / 17 | 5.25 / 15 | 0.99 |
| fan-in | shuffled | 0.88 / 4 | 1.50 / 17 | 0.88 / 15 | 0.96 |

The rows that separate the wirings most under the fan-in convention, as passes out of eight seeds:

| row | what it asks | measured | shuffled |
|---|---|---|---|
| R06 | remove AVM under anterior touch: reversal retained | 8 / 8 | 0 / 8 |
| R12 | remove nothing under nose touch: reversal | 8 / 8 | 0 / 8 |
| R15 | remove AVAL, AVAR under ash: no reversal | 6 / 8 | 0 / 8 |
| R02 | remove AVAL, AVAR under anterior touch: no reversal | 6 / 8 | 1 / 8 |
| R11 | remove nothing under tap: reversal | 8 / 8 | 3 / 8 |
| R14 | remove AVAL, AVAR, AVDL, AVDR under anterior touch: no reversal | 6 / 8 | 1 / 8 |

Read it plainly. Under the count convention the rule learns nothing on either wiring.
Under the fan-in convention it fits the two reversal facts and none of the forward ones,
the net is saturated (active 0.99), and the measured wiring still passes
5.2 held-out rows to the shuffle's 1.5: removing AVA or AVD or PVC changes the
common-mode speed in the real wiring and not in the shuffled one. That is the wiring's
structure showing through a model that has not learned the behaviour.

For scale: the original lane of the observer patch net programme, which trained the same
wiring with a global gradient through the settlement, passed 9.1 of 17 held-out rows
against 4.6 for the shuffle. The local rule does not get there, and the reason is the one
the theory predicts: equilibrium propagation needs the nudge to reach every seam it is
supposed to teach, and a directed connectome does not carry it back.

## 5. What this rung is for

It marks the edge. On layered nets with tied feedback seams the rule is a gradient and
the ladder shows parity with backprop. On a measured, directed wiring the same rule can
only teach the seams the nudge reaches, and that is not enough for this protocol. Two
routes forward, both honest: symmetrise the wiring as a declared modelling assumption
(every synapse also carries its reverse, tied), which this rung tried and which did not
rescue the training facts at any gain, or accept that measured wirings are for the
protocol layer, settle-and-score with a declared gain and a shuffled control, and leave
learning to nets built to be learnable.
