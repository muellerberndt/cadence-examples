# Cadence examples

Worked examples for [Cadence](https://github.com/muellerberndt/cadence), the patch-net
settlement library (`pip install cadence-net`, `import cadence`).

Each example is a tutorial with its own README, a runnable script, a receipt, and, where
it makes sense, a page you can play with in a browser. The ladder, from simple to complex:

| # | example | what it shows |
|---|---|---|
| 01 | [digits](01_digits/) | a patch net learns to classify with the owner-local free/nudged rule; receipt: 0.962 ± 0.003 held-out, 20 epochs |
| 02 | image recognition | the same rule on a real image set |
| 03 | [connect four](03_connect_four/) | self-play positions labelled by a depth-4 search, imitated by the rule; receipt: 0.527 agreement with the search, the same as an MLP of its size, and both beat random and lose to a two-ply search; [play it](https://claude.ai/code/artifact/7eaebd77-b8f7-4415-8a08-6aefc9aff570) |
| 04 | [pong](04_pong/) | a paddle that learns from pixels and reward by advantage-weighted nudges; receipt: 79% of balls returned vs 59% for backprop REINFORCE at the same budget; [play it](https://claude.ai/code/artifact/112fbedd-4191-42dd-b890-064f629befc3) |
| 05 | embodiment | a nervous system in a physical body |

Status: 01, 03, and 04 are complete with verified receipts and, for the games, pages you
can play; 02 is running; 05 is next.

Each example measures the obvious legacy baselines on the same split and puts them in its
receipt next to the patch net's accuracy, parameter count, epochs, and wall-clock. The
ladder is only worth climbing if those comparisons stay in view.
