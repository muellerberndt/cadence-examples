# The 1942 player, as an unlinked preview

Reachable only at `https://floatingpragma.io/cadence-examples/preview/player/`. It is not in the
gallery, not in the sitemap, not in the README, and the page carries `<meta name="robots"
content="noindex">` and says on its first line that it previews a stage nothing has accepted.

## What it shows

Seven recorded episodes of NES 1942 beside the whole brain that flew them: the emulated machine
at 240 by 224 on the left, every neuron and synapse settling once per decision on the right, the
records cortex under the scan, and the 64 by 64 grayscale view and one-frame motion field the
agent actually receives as two insets on the screen. Everything that explains them is folded into
one collapsed model card.

Five of the episodes are the gain-16 seed-0 checkpoint of run `s08-development-1789561201`
replayed with learning off, on the first five starts of the stage's held-out suite. Two are
controls on the first of those starts: the same architecture never written to, and uniform random
actions over the codec.

| episode | decisions | score | deaths |
|---|---:|---:|---:|
| the learned player, start 2000 | 956 | 4,250 | 3 |
| the learned player, start 2001 | 781 | 5,650 | 3 |
| the learned player, start 2002 | 780 | 5,750 | 3 |
| the learned player, start 2003 | 808 | 2,600 | 3 |
| the learned player, start 2004 | 912 | 2,800 | 3 |
| born frozen, start 2000 | 841 | 4,000 | 3 |
| random actions, start 2000 | 642 | 300 | 3 |

The random control has no brain behind it, so it carries no settling frames and the page leaves
its brain panel still and says so.

## What the numbers are

The run's own 30-episode medians: the learned player 5,400, born frozen 2,825, the corrupted
world 4,350, a two-line scripted sweep 6,625, random actions 2,025, constant fire 2,100. The
learned player is above its frozen twin and under a script that reads nothing. **No policy in
this stage has cleared a stage of 1942**, so the packet's headline gate is untouched. The gate
table on the page is the run's own receipt, which travels here as `receipt.json`: eleven
predicates, of which four are not met.

## The live path

The page also runs the pinned cartridge in the browser. The ROM is never shipped here and is in
no repository: the visitor picks a file, the page takes its SHA-256 in the browser and refuses
anything but the pinned cartridge. The browser emulator is **jsnes 2.1.0, Apache-2.0, copyright
2020 Ben Firshman**, included unmodified; its notice sits above the inlined copy in `index.html`
and its licence is `LICENSE-jsnes` beside this file. The Python side runs libretro's `fceumm`
through stable-retro; on a fixed 844-frame input script the two cores' screens are equal on 731
frames through a measured colour map, and the machine's RAM is equal on 615 of 843 frames outside
the stack page. The two addresses the episode rule reads never differ. The page prints all of
that in its model card, and there is no brain in the browser yet, so the live actions come from
the boundary's own baselines or from the keyboard.

## The heavy files

`index.html` is 0.3 MB and carries the page, the standard brain-scan renderer, the records view
and the emulator. The 296 film strips, steps files, frames files and the atlas come to 45.0 MB
and stay out of the repository: they are the assets of the release `preview-player-2026-09-17`,
each named `player-preview-<file>`, and `checkpoints.json` holds the size and the SHA-256 of
every one. The Pages workflow downloads them, strips the prefix into `assets/` and runs
`tools/checkpoint_assets.py verify preview/player` over the result.

## Rebuilding it

The recorder, the converter and the page's build all live in the research tree, at
`cadence-paper/experiments/experience/s08_1942/`. The episodes were recorded on the stage box
with the stage's own `record.py` and `tools/record_baseline.py`, converted by
`tools/bundle_from_records.py` into a bundle in the format `s08-public-bundle/1`, and the page
was built from that bundle by `web/build_page.py` and checked at 1280 by 800 and 390 by 844 by
`web/check_page.py`. The bundle contract and the check are written down in that directory's
`web/README.md`.
