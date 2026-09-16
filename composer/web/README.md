# The S06 C64 composer in the browser

The composer hears a phrase and answers it. One life learned the SID chip by playing it, and
this page replays what it did: the audio the emulated chip made, the level of every block of it,
the settled activation of all 2,355 neurons while it was making that sound, and the code, the
imagined reads and the writes of its records cortex over the same blocks. The page computes
nothing from a brain running in the browser: the instrument is a cycle-exact emulator with no
port in a browser yet. When one runs there, the composer plays live here, the way the arm, the
world, the artist and Connect Four already do. The page states that in its lede and in its notes.

The page shows imitation and continuation. The gates for composing under an instruction, for a
theme returning and for revision are open, and the gate table shows them as open with the value
they stand at. The run does not pass its gates; the gate table carries every value and every
threshold the receipt states.

## Files

| file | what |
| --- | --- |
| `../../web/brain_scan.js` | The standard renderer, a verbatim copy of `cadence/src/cadence/brain_scan.js` (never edited here). |
| `../../web/records_view.js` | `RecordsView`: the records cortex on a Canvas2D, the granule raster with the active cells lit, imagined reads flickering and writes flashing. Shared by every page. |
| `index.html`, `style.css`, `page.js` | The page: the C64 boot header, the piece selector, the level lanes and the transport beside the whole brain, the per-piece numbers, the gate table and the notes. |
| `build_page.py` | `--bundle <dir> --receipt <file> --out composer/index.html`: writes the page with the renderer, the records view, the style and the script inlined, copies the bundle's files to `composer/assets/`, and writes `composer/checkpoints.json` and `composer/receipt.json`. |
| `check_page.py` | Headless Chromium (SwiftShader) check: the layout at 1280 by 800 and at 390 by 844, every piece selected, every track played, the audio sources against the index, the replay advancing, the gate table, screenshots, and no page or console error. |

The style: the VIC-II palette as Pepto measured it, the blue screen inside the light-blue
border, a boot header and chunky keys. No font is fetched; the face falls back through the
monospace stack every system has, so the page renders the same offline and in CI.

## Build and check

```sh
cd /Users/muellerberndt/Projects/oph-meta/cadence-examples
PY=/Users/muellerberndt/Projects/oph-meta/cadence/.venv/bin/python
S=../cadence-paper/experiments/experience/s06_composer

# the page, from the bundle the stage delivers (one command, about half a minute)
$PY composer/web/build_page.py --bundle $S/public \
    --receipt $S/runs/acceptance/receipt.json --out composer/index.html

# the same page against the files already beside it, when the assets must not move
$PY composer/web/build_page.py --bundle $S/public \
    --receipt $S/runs/acceptance/receipt.json --out composer/index.html --keep-assets

# the headless check; screenshots land in runs/composer/web/
$PY composer/web/check_page.py composer/index.html

# the assets as release assets, and what the Pages workflow verifies after downloading them
python tools/checkpoint_assets.py prepare composer <release-tag>
python tools/checkpoint_assets.py verify composer _site/composer/assets
```

A regenerated bundle of the same layout rebuilds the page with the first command and nothing
else: the page holds no hand-written number, no piece list and no frame range.

The built page is about 0.15 MB. Everything heavy sits beside it under `composer/assets/` (about
57 MB for seven pieces) and is fetched when a piece is chosen. `--max-bytes` (3 MB) is the limit
the build refuses to pass. **The page has to be served**: a page opened from `file://` cannot
fetch its neighbours. For development, `--page-index runs/composer/dev/index.json` writes the
same index as a file, and `composer/web/index.html` loads it by the path in
`<body data-bundle>` when the repository root is served.

## What the build reads from the bundle

The bundle is the stage's `public/` directory, format `s06-public-bundle/1`. Every file and
every field the page consumes is listed here. A field the bundle leaves out leaves the page
showing less: a track with no audio has its button disabled and its lane marked `not in this
bundle`, a track with no frames leaves the brain still, and a bundle with no records hides the
records panel.

```
public/
  manifest.json                     the index
  atlas.json                        cadence.atlas/v1, as web/brain_scan.js reads it
  <piece>/demonstration.wav         16 bit PCM mono at manifest.sample_rate_hz
  <piece>/imitation.wav
  <piece>/continuation.wav
  <piece>/imitation.frames.json     s06-public-frames/1
  <piece>/continuation.frames.json
```

### `manifest.json`

| field | what the build does with it |
| --- | --- |
| `format` | checked against `s06-public-bundle/1`; the build stops on anything else |
| `stage` | printed in the page index |
| `brain.seed`, `brain.checkpoint` | named in the page's notes, beside the run id |
| `listening` | named in the page's notes, as how the demonstration is heard |
| `sample_rate_hz` | the header line, and the block clock every WAV is read on |
| `phrase_blocks` | the page index's clock block |
| `atlas.file` | copied beside the page; fetched by the page and given to `BrainScan` |
| `atlas.neurons` | the header line and the boot line |
| `atlas.reading` | the reading the records view draws: the page takes the frame's activation at exactly these neuron indices, and the region of each of them groups the strip under the granule raster |
| `pieces[].name` | the piece's id, its folder, and its title through a table in `build_page.py` |
| `pieces[].piece`, `.family`, `.split`, `.blocks` | shown in the piece's ledger |
| `pieces[].files.{demonstration,imitation,continuation}.path` | copied beside the page and played by the page's audio elements |
| `pieces[].files.{imitation,continuation}_frames.path` | copied beside the page and fetched when that piece is chosen |
| `pieces[].numbers.*` | printed under the lanes; `pitch_within_a_semitone`, `onset_f1`, `imitation_reward`, `continuation_surprise_nats`, `continuation_pitch_within_a_semitone`, `continuation_onset_f1` and `median_pitch_error_cents` have their own labels, any other key is printed with its own name, and a `null` is dropped |

### `<piece>/<episode>.frames.json`

| field | what the build does with it |
| --- | --- |
| `format` | checked against `s06-public-frames/1` |
| `frames.steps`, `frames.n`, `frames.activation` | the frames themselves, in the layout `decodeFrames` expects: one unsigned byte per neuron per frame, `value / 255 * 2 - 1` |
| `blocks[i].phase` | which frames belong to which track (below), and the phase the page names while it replays |
| `blocks[i].records.index`, `.values` | the executed reading's plain code: the cells the records view lights, and the cells its write flashes |
| `blocks[i].records.valuedIndex`, `.valuedValues` | the valued code, drawn in the value colour |
| `blocks[i].records.imagined` | counted in the records ledger |
| `blocks[i].records.writes` | counted, and flashed on that block's own code cells, which is where the head writes them |
| `blocks[i].block` | carried through for the ledger |

`blocks[i].records.reading_mean` and `.reading_top` are read by nothing here: the page takes the
reading itself from the frame at `manifest.atlas.reading`.

### Which frames belong to which track

An episode runs `listen`, `gap`, `attempt`, one frame per block, and the build cuts it by those
phases:

- **imitation** and **continuation**: the `attempt` frames of their own episode, the last
  *N* of them, where *N* is the length of that track's WAV in 20 ms blocks.
- **demonstration**: the last *N* `listen` frames of the imitation episode, where *N* is the
  demonstration WAV's length in blocks. The listen phase plays the phrase once per voice and
  then whole, so those frames are the whole-phrase pass, which is the audio the demonstration
  plays. A listen phase whose length is not a whole number of passes leaves the demonstration
  without frames; nothing is guessed. On this bundle the imitation episode is 968 frames (768
  listening, 8 gap, 192 attempt) and the continuation episode is 488 (384, 8, 96).

### The lanes, and what the bundle does not carry

The bundle carries no note list, and the executed gesture cannot be read off the frames: the
efference and motor neurons are flat in every frame of every episode. The lanes are therefore
the **level of the audio**, computed by the build from the WAV the page plays: the RMS of each
20 ms block over -60 dBFS to 0, as one byte per block. Two fields in a future bundle turn these
lanes into a three-voice piano roll, and the page is written to take them:

- `pieces[].tracks.{demonstration,imitation,continuation}.voices`: three lists of measured
  notes, each `[start, blocks, pitch]`, the start counted inside that track and the pitch as the
  codec numbers it (pitch `p` is MIDI `p + 11`). `evaluate.py` measures these already.
- the phrase window each episode used, so the demonstration's own written notes can be drawn
  against what the composer played.

### The receipt

`--receipt` (default `receipt.json` inside the bundle) is the run receipt. The build reads
`run_id`, `receipt_sha256`, `started_utc`, `seeds.completed`, `sources.emulator.digest`,
`numerics.brain.agent.records` (the records cortex's cells, active count and rates), and the
gate table:

- `acceptance.predicates[]` become the gated rows: `name`, `value`, `threshold`, `passed`,
  `aggregation`.
- `open.measures[]` become the open rows: `name`, `value`, `threshold`, `met`, `aggregation`.

A name in either list is grouped and labelled by the `GATES` table in `build_page.py`
(instrument, imitation, continuation, controls, latency, instruction, theme recall, revision); a
name outside that table is shown under "controls" with its own name, so every name the run
states reaches the page. Nothing is computed here: the value, the threshold and the verdict are
the receipt's.

## What the page shows

- **The lanes**: two panes on one canvas, what the chip was given above and what the composer
  played below (the imitation, or the continuation while that plays). Time runs left to right
  over the track, the level runs bottom to top over -60 dBFS to 0, a line marks every sixteenth
  and a brighter one every bar. Blocks ahead of the playhead are drawn faint and fill as the
  audio reaches them.
- **The whole brain**: the standard scan, replaying the frames of the track that is playing, in
  time with the audio. A frame is one block's settled state; the page steps the scan at most
  eight times an animation frame, and when it has to skip it sets the state, which measures no
  change that did not happen.
- **The records cortex**: the standard view, fed the code, the imagined count and the writes of
  the same blocks, with the reading taken from the frame at the reading's neuron indices.
- **The numbers**: the piece's own measurements under the lanes, and the run's identity in the
  notes.
- **The gates**: the gated set and the open set, each row with its value, what is wanted and
  whether it is met.

## The assets and the release

`build_page.py` writes `composer/checkpoints.json` in the shape `tools/checkpoint_assets.py`
reads: one entry per file under `composer/assets/` with its label, size and sha256.
`prepare composer <tag>` adds the asset names and prints the upload command; `verify composer
<dir>` checks a download against it. `--keep-assets` rebuilds the page against the files already
there and writes neither the assets nor that manifest, for a page rebuilt while its assets are
being uploaded.

Three files outside this directory have to catch up before the page ships, and none of them is
changed here:

- `.gitignore` needs `composer/assets/`. The other stages keep their heavy files out of the
  repository the same way (`arm/checkpoints/`, `world/checkpoints/`, …), and a build leaves
  about 57 MB under `composer/assets/`, which must stay out of a commit.
- `.github/workflows/pages.yml` needs a composer line in the assembly step
  (`cp composer/index.html composer/receipt.json _site/composer/`) and in the asset step. That
  step downloads a stage's files with the pattern `<stage>-*.json`, which covers the other
  stages; this stage ships WAVs as well, so the composer needs `--pattern "composer-*"` with the
  downloaded names stripped of the `composer-` prefix into `_site/composer/assets/`.
- `index.html`, the gallery, needs the composer's card.

`tools/verify_gallery.py` requires every predicate of a listed example to pass, and this stage's
do not, so a `composer` entry there needs a rule for a stage published with gates open. That is
a decision about the gallery, and this page makes no claim on it.

## What is open

- The live composer. It needs a SID emulator in the browser: a WebAssembly build of the pinned
  `pyresidfp` core, or a port of the same cycle-exact model. The transport here drives the
  lanes, the brain and the records off one clock, which is the clock a live agent would drive.
- The notes of each track, and the phrase window of each episode (above). With them the lanes
  become a three-voice roll.
- The three listening passes before the last one (one voice at a time) sit in every imitation
  episode's frames and reach no audio, so the page leaves them out of the replay. A bundle
  carrying the three solo renderings lets a visitor hear what the brain hears voice by voice.
