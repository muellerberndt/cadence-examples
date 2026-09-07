# Cadence examples

Worked examples for [Cadence](https://github.com/muellerberndt/cadence), the patch-net
settlement library (`pip install cadence-net`, `import cadence`).

Each example is a tutorial with its own README, a runnable script, a receipt, and, where
it makes sense, a page you can play with in a browser. Start at the
[hub page](https://claude.ai/code/artifact/ee7a8b53-be8c-4c34-9f91-43d6eaf77be8), which lists the
ladder with each receipt's numbers and links to the games. The ladder, from simple to complex:

<!-- ladder -->
| # | example | what it shows |
|---|---|---|
| 01 | [digits](01_digits/) | 8×8 scikit-learn digits; 64 input owners, a hidden layer, 10 output owners; receipt: 0.962 ± 0.003 held-out in 20 epochs; a same-size MLP 0.967 in 50 |
| 02 | [images](02_images/) | mnist, 784 pixels a picture, trained on apple silicon in float32 and read out on the float64 reference backend; receipt: 97.4% held-out in 10 epochs against 97.8% for a same-size MLP; the two backends agree on every prediction |
| 03 | [Connect Four](03_connect_four/) | self-play positions labelled by a depth-4 search; the net imitates the search and plays with no lookahead; receipt: agrees with a depth-4 search on 0.527 of positions (MLP 0.533); 91-0-9 vs random, 2-0-98 vs depth 2; [play it](https://claude.ai/code/artifact/7eaebd77-b8f7-4415-8a08-6aefc9aff570) |
| 04 | [Pong](04_pong/) | a paddle that learned from pixels and reward: the nudge's target is the action taken, its strength the action's advantage; receipt: reward-trained paddle returns 78.0% of balls against 92.9% for backprop REINFORCE on the same rollouts; the same net taught the tracker's moves 96.6%; [play it](https://claude.ai/code/artifact/112fbedd-4191-42dd-b890-064f629befc3) |
| 05 | [text](05_text/) | next-character prediction on shakespeare from a sixteen-character window: 1,040 input owners, 65 output owners, and a sample the net wrote; pending |
| 06 | [sign writer](06_sign/) | show it a sign; it writes what it sees with a two-joint arm, looking through a 7×7 window around its pen; receipt: writes held-out signs at 0.988 overlap with the sign, the teacher's own score; the MLP the same; [play it](https://claude.ai/code/artifact/94e42c3b-134e-47c0-94bb-06106d3f6321) |
| 07 | [chorales](07_music/) | continues bach chorales chord by chord from an eight-chord window with a multi-hot quadratic nudge; listen in the page; receipt: next-chord pitch-set F1 0.483, 16.6 bits/chord; repeat-last 0.369, MLP 0.470; [play it](https://claude.ai/code/artifact/06f247a3-5acb-4663-91c6-9474f087a51e) |
| 08 | [cart-pole](08_cartpole/) | the classic control task from reward, with the state as a place code; receipt: balances for 154 steps of 500 against 392 for backprop REINFORCE |
| 09 | [C. elegans](09_celegans/) | the published connectome learns four textbook facts and is scored on seventeen held-out ablation phenotypes against a shuffled wiring; receipt: count convention: nothing learned; fan-in convention: measured 5.2/17 held-out ablations vs shuffled 1.5/17, a structural signal, not a behavioural model |
<!-- /ladder -->

## Play the games locally

The trained nets are in the repo (`03_connect_four/net.json`, 245 kB; `04_pong/net.json`,
743 kB) and already embedded in each game's `index.html`, so nothing needs training or
downloading:

```bash
git clone https://github.com/muellerberndt/cadence-examples
cd cadence-examples
python serve.py                    # serves the repo and opens http://localhost:8765/
```

The hub at that address links to both games and to every tutorial. Opening
`03_connect_four/index.html` or `04_pong/index.html` straight from the file system works
too. To retrain a game and rebuild its page: `python train.py && python build_page.py` in
its directory (see each tutorial for what that costs). The datasets under `data/` are not
in the repo; the scripts fetch or generate them, and every receipt records their digests.

Status: 01 to 04 are complete with verified receipts and, for the games, pages you can
play; 05 is next.

Each example measures the obvious legacy baselines on the same split and puts them in its
receipt next to the patch net's accuracy, parameter count, epochs, and wall-clock. The
ladder is only worth climbing if those comparisons stay in view.
