# Cadence examples

Worked examples for [Cadence](https://github.com/muellerberndt/cadence), the patch-net
settlement library (`pip install cadence-net`, `import cadence`).

Each example is a tutorial with its own README, a runnable script, a receipt that binds its
numbers to the code and data that produced them, and, where it makes sense, a page you can
play with in a browser. Start at the
[hub page](https://claude.ai/code/artifact/ee7a8b53-be8c-4c34-9f91-43d6eaf77be8). An example
stays on this ladder only while its receipt shows the patch net ahead of the strongest
backprop model in the same script on a primary measure; the digits rung stays as the
tutorial, where the advantage is fewer passes over the data at the same accuracy.

<!-- ladder -->
| # | example | what it shows |
|---|---|---|
| 01 | [digits](01_digits/) | 8×8 scikit-learn digits; 64 input owners, a hidden layer, 10 output owners; receipt: 0.962 ± 0.003 held-out in 20 epochs; a same-size MLP 0.967 in 50 |
| 03 | [recall](03_recall/) | associative recall with no trained parameters: each key-value pair is one hebbian outer product, each query a settlement with the key clamped, at any context length; receipt: the value of any key in a context of up to 128 pairs: 1.00 settled, 1.00 in one read, with no trained parameters; a two-layer transformer trained on the task 0.30 at 4 pairs, 0.01 at 128 |
| 04 | [Pong](04_pong/) | a paddle that learned from pixels and reward: the nudge's target is the action taken, its strength the action's advantage; receipt: reward-trained paddle returns 78.0% of balls against 92.9% for backprop REINFORCE on the same rollouts; the same net taught the tracker's moves 96.6%; [play it](https://claude.ai/code/artifact/112fbedd-4191-42dd-b890-064f629befc3) |
| 07 | [chorales](07_music/) | continues bach chorales chord by chord from an eight-chord window with a multi-hot quadratic nudge; ahead of a same-size mlp on both measures; listen in the page; receipt: next-chord pitch-set F1 0.483, 16.6 bits/chord; repeat-last 0.369, MLP 0.470; [play it](https://claude.ai/code/artifact/06f247a3-5acb-4663-91c6-9474f087a51e) |
<!-- /ladder -->
[How a patch net learns](HOW_IT_LEARNS.md) is the tutorial the rungs build on: owners,
seams, settlement, and the one local rule the rungs use, whether the target is a label, a
teacher's move, or a reward.

## What left the ladder, and why

Nine rungs were measured (they remain at tag `v0.5.0`). Five showed parity with the
backprop model in the same script and nothing more, at ten to a hundred times the
wall-clock of the engine they were measured on: MNIST (0.974 against an MLP's 0.978),
Connect Four (0.527 agreement with the search against 0.533), the sign-writing arm
(0.988 against 0.988), the character model of Shakespeare (3.33 bits against a transformer's
3.08) and cart-pole from reward with the plain nudge rule (154 steps against 392). Parity
is not an advantage, so they are gone; the character model continues in another place,
and cart-pole's successor, the three-factor rule that reaches the threshold in half of
PPO's steps with a fifth of the parameters, is measured in the paper and will return here
when it is an example rather than an experiment.

The *C. elegans* rung asked a different question: does the library tell us anything about
the worm's nervous system? The receipt's answer is narrow. Under the connectome's own
convention (drive per contact times synapse count) the owner-local rule learned nothing on
the measured wiring or on a shuffled one; under a fan-in normalisation it fitted two of the
four textbook facts, saturated the net, and passed 5.2 of 17 held-out ablation phenotypes
where the shuffled wiring passed 1.5. What separates the two wirings is structural: removing
a command interneuron (AVA, AVD, PVC) changes the common-mode speed in the real wiring and
not in the shuffle, which is the hub structure anatomy already knows. The behavioural facts
themselves were not learned, and the reason is the one the rule predicts: a nudge on the
motor neurons reaches upstream only along seams that end on owners it moved, and a directed
connectome does not carry it back past the command interneurons. So the rung showed where
the rule stops, not something new about the worm, and a settle-and-score protocol with a
shuffled control on that wiring is a test of graph structure rather than a model of the
animal. It left the ladder for that reason. The measured-wiring machinery (fixtures with
custody, protocols with predicates, the shuffled control) stays in the library, where the
[protocols](https://github.com/muellerberndt/cadence/blob/main/docs/protocols.md) doc
describes it; a claim about the worm would need learning that reaches the whole wiring,
which is what the observer patch net programme's global-gradient lane did (9.1 of 17 against
4.6) and the local rule does not.

## Run an example

```bash
git clone https://github.com/muellerberndt/cadence-examples
cd cadence-examples
python -m pip install cadence-net scikit-learn pandas scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # the baselines each receipt measures
cd 03_recall && python train.py && python train.py --verify receipt.json
```

Every `train.py` selects on a validation split where there is one, reads its test set
once, measures the obvious baselines in the same script, and writes a receipt; `--verify`
checks a receipt against the code in the checkout and its own arithmetic. The datasets
under `data/` are not in the repo: the scripts fetch or generate them, and every receipt
records their digests. Each tutorial says what a full run costs, from seconds (digits) to
a quarter of an hour (Pong).

## Play the pages

The trained nets are in the repo and already embedded in each page's `index.html`, so
nothing needs training or downloading:

```bash
python serve.py                    # serves the repo and opens http://localhost:8765/
```

The hub at that address links to the two pages (Pong, the chorales) and to every tutorial;
opening a page's `index.html` straight from the file system works too. To retrain a page's net and rebuild it: `python train.py && python
build_page.py` in its directory. Each page runs the same settlement in JavaScript that the
receipt scored in Python.

## What holds it together

- `tools/smoke.py` runs every rung end to end at a tiny budget and verifies the receipt it
  writes; the committed receipts stay untouched. It is the test suite, and CI runs it.
- `tools/ladder.py` regenerates the ladder table above and the hub pages from the receipts,
  so the numbers in three places are one set of numbers.
- CI verifies every committed receipt against the checkout, rebuilds every page from its
  committed net and checks that nothing changed, regenerates the ladder and checks the
  same, then runs the smoke suite.

Each example measures the obvious legacy baselines on the same split and puts them in its
receipt next to the patch net's accuracy, parameter count, epochs, and wall-clock. The
ladder is only worth climbing if those comparisons stay in view.
