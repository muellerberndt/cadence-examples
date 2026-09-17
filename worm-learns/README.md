# Which smell means food

A C. elegans nervous system that learns. The worm starts with the published
hermaphrodite wiring (Cook et al. 2019) and no preference between two odours,
diacetyl and butanone. Food lies under one of them. Every step, Cadence's local
learning rule adjusts synapses inside that wiring, rewarded by food. The page
lets you watch it learn, swap which smell means food, and remove neurons.

## Run the page

The page is plain HTML and ES modules with no build step. Serve this folder
over HTTP (modules do not load from `file://`):

```bash
python -m http.server 8000
```

Then open <http://localhost:8000/>. The page opens with a naive worm learning at
the Fast speed. **Load trained brain** starts from the receipt checkpoint.

## What learns, and what is supplied

| Part | Status |
|---|---|
| Neurons and synapses | Published: 300 neurons, 3,638 chemical synapse pairs and 1,093 gap junctions, merged into 5,044 directed overlaps. Each neuron's inputs are divided by its total contacts. GABAergic neurons inhibit; every other chemical synapse excites; gap junctions couple positively both ways. |
| Plasticity | Learned: the scale of 941 published synapses into the chemotaxis circuit (AIY, AIZ, AIA, AIB, RIA, RIB, RIM, AVA, AVB, SMDD, SMDV, RMDD, RMDV), each bounded to at most three times its published strength. No synapse is added. |
| Learning rule | Cadence `ActorCritic`: a free settlement, two nudged settlements toward and away from the chosen action, an eligibility trace per synapse, and a reward-prediction error from a linear critic reading the interneurons. Neuron biases do not learn. |
| Reward | 1 on reaching food, plus 0.3 per full step of approach to the food spot. |
| Senses | Supplied: AWA reads diacetyl and AWC reads butanone, each pair as a rectified left/right comparison of odour concentration at the head. FLP and OLQ feel the dish rim. |
| Body | Supplied: the output softmax picks SMDD (turn left), SMDV (turn right) or AVB (forward); the body turns 0.45 rad or steps forward, with random wobble. |
| Dish | Supplied: two Gaussian odour clouds 0.6 from the start; a search ends at either spot or after 150 steps. |

This is a demonstration of learning inside a fixed biological wiring. It is not
a validated model of worm olfactory learning, whose real mechanisms include
neuromodulators and sensory adaptation that are not modelled here.

## Results

Receipts in `runs/receipts_v2`, three seeds each, 16 worms per update, 8,000
decisions, scored on 1,000 dishes the worm never trained on (drawn actions).
A random walk finds food on 23% of dishes and the empty spot on 23%.

| Brain | Food, held out | Empty, held out | First window ≥ 90% food | Parameters | Wall-clock |
|---|---|---|---|---|---|
| Cook 2019 connectome | 99.9–100% | 0–0.1% | 2,000 decisions (3/3 seeds) | 1,241 | ~390 s |
| Same contacts, shuffled | 53%, 88%, 59% | 22%, 4%, 22% | not reached (0/3) | ~1,260 | ~210 s |
| MLP (6-32-4), backprop A2C | 99.4–100% | 0–0.6% | 500 decisions (3/3) | 356 | ~65 s |

Wall-clock times were measured with 15 runs sharing one machine. The MLP learns
in fewer decisions and far less time; the connectome is the example because
its learning happens inside a biological wiring whose parts can be tested.

**Lesions** (held-out food after removing neurons from a trained connectome worm, three seeds):

| Removed | Food | Reading |
|---|---|---|
| none | 99.9–100% | |
| AWA (diacetyl sensors) | 25–36% | the learned behaviour needs the food odour's sensors |
| AWC (butanone sensors) | 99.7–99.8% | the other odour's sensors are not used |
| RIA | 19–22% | steering is routed through RIA |
| AIB | 84–98% | partial; in two seeds the greedy policy fails |
| AIY, AIZ, AIA, AVA+AVB | no loss | |

**Swapping the food smell.** After 8,000 decisions, food moves under butanone.
All three connectome seeds fall to 11–46% food and relearn to at least 90% within
3,500–7,000 decisions (held out afterwards: 96–99.5%). After relearning, removing
AWC drops food to 24–29% and removing AWA no longer matters (94–99%); removing
RIA still drops it to 16–26%. The shuffled worm did not relearn (57%).

**One worm at a time** (the page's setting, step 0.625): above 90% food by 40,000
decisions, 100% held out.

## Reproduce

```bash
python -m pip install "cadence-net>=0.8" torch pytest
python train.py --kind connectome --seed 0 --checkpoint runs/receipts/connectome_s0.npz --out runs/receipts/connectome_s0.json
python train.py --kind shuffled   --seed 0 --out runs/receipts/shuffled_s0.json
python train.py --kind mlp        --seed 0 --out runs/receipts/mlp_s0.json
python train.py --kind connectome --seed 0 --decisions 16000 --swap-at 8000 --out runs/receipts/connectome_swap_s0.json
python train.py --kind connectome --seed 0 --batch 1 --eta 0.625 --decisions 48000 --out runs/receipts/connectome_batch1_s0.json
python export_page.py --trained runs/receipts/connectome_s0.npz --results runs/receipts/*_s?.json --out net.js
python -m pytest -q tests
python record_parity.py && node tests/parity.mjs
```

The defaults of `train.py` are the receipt configuration. Each result records the
SHA-256 of `train.py`, `worm.py` and the connectome fixture, and the Cadence version.

`tests/parity.mjs` replays recorded Python learning runs through `learner.js`:
every action choice matches and the learned synapse scales agree to about 1e-14.

## Two ways this fails, and what fixed them

**Output biases run away.** When neuron biases learn, the cross-entropy nudge
gives the six output neurons a large bias contrast. Their biases grow until one
turn wins regardless of smell, and the worm circles until every search times
out. Biases are therefore fixed.

**The motor loop latches.** With the synapses into the circuit free to grow to
Cadence's limit of eight times their published strength, the inputs of one
turning neuron strengthen until it stays active under any odour. The softmax
then predicts that turn with probability near 1, the two nudged settlements
barely differ, and the reward signal has little left to correct. A smaller step
only delays this: at three step sizes, one to three of three seeds collapsed
between 2,000 and 3,000 decisions.

Two structural changes stopped the collapse. Freezing the synapses that leave
the head motor and AVB neurons kept all seeds stable, but after the food smell
was swapped the worm slid into turning one way and did not relearn. Bounding
every plastic synapse at three times its published strength also kept all seeds
stable, and it relearned after the swap. The receipts use the bound.

## Files

| File | Role |
|---|---|
| `worm.py` | Connectome wiring and shuffled control, the dish, the brain |
| `train.py` | Training, held-out evaluation with lesions, MLP baseline, receipts |
| `export_page.py` | Writes `net.js` for the page |
| `index.html`, `page.js` | The page |
| `learner.js`, `dish.js` | The learning rule and the dish in the browser |
| `record_parity.py`, `tests/` | Python tests, browser parity and speed checks |
| `data/` | The pinned connectome fixture and the OpenWorm licence |

Connectome data: Cook SJ et al. (2019) Whole-animal connectomes of both
*Caenorhabditis elegans* sexes. *Nature* 571:63–71, as redistributed by
[OpenWorm ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox)
(MIT, see `data/OPENWORM_LICENSE.txt`); neuron classes, positions and transmitters
from [c302](https://github.com/openworm/c302).
