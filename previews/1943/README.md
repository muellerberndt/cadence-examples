# Flight Lab preview

A recorded 1943 pilot with its entire circuit beside the game. This page is an
unlisted research preview, excluded from the official example gallery and navigation.
The HTML requests `noindex`; the page and its recording are publicly accessible.

The pilot learned through imitation and subsequent trial and error on a GPU.
This recording contains 527 decisions across 35 seconds, with online learning
enabled. It reaches 6,400 points and ends in a loss, without clearing a stage.
The start was selected for a demonstration; this score is not a held-out evaluation.

The map draws all 38,577 core neurons and 1,056,874 core synapses, plus 6,308
addressed route stores and the value/dopamine attachment. Every recorded solver
iteration is available in the repair clock: 10,870 iterations, 527 feedback
reports, 59 nonzero dopamine signals and 60 parameter versions. Population colors
identify function. Repair brightness is normalized within a step, while numerical
inspection preserves the recorded values. The traces are simulated potentials.

Retinal input, visual assembly, association, retained context and motor populations
constrain a shared recurrent state. Local readback and feedback steer repairs toward
an equilibrium. Route memory and the linear value readout are separately labeled
attachments. Positive and negative training nudges measure eligibility; they do
not simulate future games. The browser replays an actual run, rather than training
or running an emulator.

Use **Play** and **Step** to inspect every repair, or choose **Gameplay** for
decision-aligned playback. Seek to any decision. The game frame remains fixed
during its recorded settling iterations; slow downloads slow both together.
The separate gameplay-only video provides a small, quick watch. Zoom the circuit
for individual links and hover neurons to inspect their values.

## Build and serve

From the repository root, with Python 3.11+:

```bash
python tools/build_pages.py
python -m http.server 8000 --directory runs/pages
```

Open `http://localhost:8000/previews/1943/`. The first command downloads the
recording pinned in `recording.json` and verifies its archive and file hashes.
The data is a prerelease asset, so source checkouts stay small. Rebuilding uses a
fresh output directory, selectable with `--output`. No ROM, GPU or Python package
installation is needed to serve the page.

Memory and parameter snapshots XOR their bits against a fixed base. A seek
requires at most one base per packet. Decoding preserves every original array
byte, including float64 parameters, signed zero and tiny repairs. The export's
integrity receipt is included in `recording/integrity.json`. GitHub Pages serves
gzip files directly; the browser detects and decompresses them before decoding.

Game imagery belongs to Capcom. This independent research visualization includes
recorded gameplay, with no ROM, emulator or affiliation with the game publisher.
