# The composer preview, unlinked

This directory serves one page at `https://floatingpragma.io/cadence-examples/preview/composer/`.
Nothing links to it: it is not in the gallery's `index.html`, its `README.md` or its `sitemap.xml`,
and the page carries `<meta name="robots" content="noindex">`. It is here to be tested, not to be
found.

**What it is.** The S06 three-voice SID composer, running in the browser: the emulated chip as
WebAssembly (`sid.wasm`, `sid.mjs`), the ear, the records cortex and the candidate search as
JavaScript (`js/`, `life.js`, `page.js`), the whole brain beside the music (`brain_scan.js`,
`records_view.js`). A visitor picks one of the seven public-domain phrases or plays one on the
keys; the composer hears it voice by voice and answers a block at a time.

**The stage is not accepted and the brain is old.** The page runs the round-five acceptance
checkpoint (`S06-acceptance-20260916T021058Z`, seed 10, the theme checkpoint). The split head and
the waveform reader of the current round are not in this port. The model card at the bottom of the
page carries the run's whole gate table, the failing gates and the open measures marked, straight
from the acceptance receipt.

**The heavy file.** `brain.json` (31,154,912 bytes) is not in this repository. `checkpoints.json`
names the release `preview-composer-2026-09-17` and the asset `composer-preview-brain.json` with
its sha256; the Pages workflow downloads it into `_site/preview/composer/` and verifies it with
`tools/checkpoint_assets.py verify preview/composer`. To run the page locally, copy that file in
beside `index.html`.

**The licence.** What makes the sound is reSIDfp from libsidplayfp through pyresidfp 0.17.0, which
is GPL-2.0-or-later, so this page is a derivative work of GPL sources and is served under the same
licence. `LICENSE` is the GNU General Public License version 2. `source/` carries the
corresponding source of what is served: the pinned upstream archive `pyresidfp-0.17.0.tar.gz`
(sha256 `4a805c7a08b157dcc759012d21d5de1f9eb5b5427aa2fee47e72bc2c1d9a707f`),
`reproducible-state.patch` (the dither ring removed and the resampler ring zeroed, both part of
the pinned build), `single_thread.py` (the filter tables built in turn, because a browser build
has no pthreads), `shim.cpp` (the handle API the page calls) and `build.sh` (the emscripten
build).

**Where it comes from.** The page is a copy of
`cadence-paper/experiments/experience/s06_composer/web/live/` with five changes, which any rebuild
must repeat: the module paths `../js/` become `./js/`; `data-brain`, `data-chip` and `data-pieces`
point beside the page; `<meta name="robots" content="noindex">` is added; the preview notice is
added above the masthead; the model card's licence sentence points at `LICENSE` and `source/`. The
stage's own headless check (`s06_composer/tools/check_live_page.py`) ran against this directory at
1280x800 and 390x844.
