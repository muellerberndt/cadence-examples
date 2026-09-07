# Cadence examples

Worked examples for [Cadence](https://github.com/muellerberndt/cadence), the patch-net
settlement library (`pip install cadence-net`, `import cadence`).

Each example is a tutorial with its own README, a runnable script, a receipt, and, where
it makes sense, a page you can play with in a browser. Start at the
[hub page](https://claude.ai/code/artifact/ee7a8b53-be8c-4c34-9f91-43d6eaf77be8), which lists the
ladder with each receipt's numbers and links to the games. The ladder, from simple to complex:

| # | example | what it shows |
|---|---|---|
| 01 | [digits](01_digits/) | a patch net learns to classify with the owner-local free/nudged rule; receipt: 0.962 ± 0.003 held-out, 20 epochs |
| 02 | [images](02_images/) | the same rule on MNIST, trained on the accelerator and read out in float64; receipt: 97.44% held-out in ten epochs against 97.79% for a same-size MLP |
| 03 | [connect four](03_connect_four/) | self-play positions labelled by a depth-4 search, imitated by the rule; receipt: 0.527 agreement with the search, the same as an MLP of its size, and both beat random and lose to a two-ply search; [play it](https://claude.ai/code/artifact/7eaebd77-b8f7-4415-8a08-6aefc9aff570) |
| 04 | [pong](04_pong/) | a paddle that learns from pixels and reward by advantage-weighted nudges; receipt: 79% of balls returned vs 59% for backprop REINFORCE at the same budget; [play it](https://claude.ai/code/artifact/112fbedd-4191-42dd-b890-064f629befc3) |
| 05 | embodiment | a nervous system in a physical body |

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
