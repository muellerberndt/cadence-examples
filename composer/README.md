# Cadence Composer

A music studio built around a trained cadence brain, with a pretrained model. Describe a
mood and the musician imagines continuations through its own predictions, listens to the
whole draft, edits its weakest passage, and shows every neuron and every settling iteration
in sync with playback. This folder is the example's full source: the brain, the trainers,
the studio, the tests and the model data sheet.

## Run the studio

```sh
cd composer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tools/fetch_model.py                     # maestro-1, 4.46 GB, verified by SHA-256
python serve_musician.py --checkpoint checkpoints/maestro-1/brain.npz
```

Open **http://127.0.0.1:8079**. On Apple silicon add `--backend torch --device mps`; on a
CUDA machine `--backend torch --device cuda`. Loading the checkpoint and laying out the
brain takes a few minutes. The model, the MIDI corpus, the SoundFont and generated
recordings stay under ignored local directories (`checkpoints/`, `data/`, `runs/`).
What the model is, what it was trained on and how it measures: [MODEL_CARD.md](MODEL_CARD.md).

The older phrase-planning studio (`serve.py`, port **8078**) runs on
the small `phrase-composer` model (`python tools/fetch_model.py --model phrase-composer`,
then `python serve.py`). Its design and evidence: [BRAIN.md](BRAIN.md), [EVIDENCE.md](EVIDENCE.md).

The browser's language input is a bounded keyword parser: piano/orchestral/baroque,
sad/minor/dark, calm/gentle/slow, bright/heroic/energetic. “Cinematic orchestral”
selects broad instrumentation and mood. It is not an unrestricted language model
or a model of a particular composer's style.

## The musician: maestro-1

The musician is one brain built from scratch as wired cortices (ear, interval sense,
melody, harmony, rhythm, timbre, form, phrase, working memory, form record, piece trace,
intention) that learns by imitation of whole pieces as streams, takes a mood input,
imagines continuations through its own predictions, listens back to the whole draft and
edits its weakest passages. Design and measurements: [MUSICIAN.md](MUSICIAN.md); the
pretrained model's data sheet: [MODEL_CARD.md](MODEL_CARD.md). Reproduce its training on
a GPU machine (the corpus is prepared from PDMX by `tools/acquire.py`):

```sh
python tools/prepare_musician.py --workers 44 --name musician
python tools/train_musician.py --name large-v2-nobelt --version 2 --no-belt --batch 256 --updates 50000 --evaluate-every 2000 --tonic 1.0 --eta 0.001 --eta-final 0.0002 --decay 0.00001 --rollback 1.15 --device cuda:0
python tools/baseline_musician.py --name gru-control --hidden 1024 --batch 256 --updates 60000 --device cuda:1
python tools/practice_musician.py --checkpoint runs/large-v2-nobelt/brain.npz --name large-practice --rounds 40 --pieces 6 --bars 12 --own-scale 0.25 --guard 1.03 --device cuda:2
python tools/compose_musician.py --checkpoint checkpoints/maestro-1/brain.npz --mood "a bright, heroic, loud orchestral theme" --bars 32 --record --render --settle 48
```

## Use the studio

1. Choose a preset or write a brief, then **Compose one minute**. The seed makes
   a given checkpoint and configuration reproducible.
2. Inspect all ten functional populations: **2,614 neurons and 1,702,938 directed
   synapses** in the current performing brain. Wheel/pinch to zoom, drag to pan or
   expand the map. Every neuron is measured and mapped. Live traces sample actual
   rehearsal states; amber/purple score overlays show competing continuations.
3. Listen to **draft** and **final**, and inspect the competing phrase scores.
   Playback feeds the playing MIDI score back into an isolated retained brain state.
   This is symbolic score readback, not acoustic waveform perception.
   A rejected revision preserves the draft. Download MIDI or a 60-second WAV.
4. Teach a preference. A signed valence moves the chord, rhythm and interval
   relationships in the piece toward or away from what it contained; exact notes
   never enter this memory and the trained event synapses stay unchanged. A held-out
   check rejects updates that raise relationship loss by more than 1%. Accepted
   tables are saved to `checkpoints/personal/music.npz`. Launch with `--taste` to
   reuse them.

Each phrase is a future simulation. Six candidate continuations roll forward
through the brain's own predictions in isolated batch rows, the critic compares
where they lead, and the winner is committed. Learned corpus expectations for the
next chord and duration are synapses in the same brain as note intention.

The circuit's plotted waves are measured software states, not clinical EEG.
Valence is an explicit numerical objective, not a claim of feelings or dopamine
chemistry. The memory cue, trained musical populations and note intention exchange
messages in one joint brain. Finite repair budgets are reported; no global
optimality or biological equivalence is assumed.

## Reproduce training

```sh
python tools/acquire.py
python tools/prepare.py --limit 256 --workers 4 --name pilot
python tools/prepare.py --limit 30000 --workers 4 --name classical
python tools/train.py --dataset pilot --name small-pilot --updates 200
python tools/train.py --dataset classical --name gpu-256 --backend torch --device cuda --size 256 --updates 6000 --batch 256
python tools/baselines.py --updates 6000 --size 384 --device cuda
python tools/assess.py --suite
```

The dataset is [PDMX v9](https://zenodo.org/records/15571083), described in the
[upstream project](https://github.com/pnlong/PDMX). Publisher MD5 checksums are
verified before extraction. We require explicit public-domain metadata, no marked
license conflict, deduplication and a classical-music filter. This yields 25,248
usable pieces: 3,795,043 training events, 219,743 validation events and 210,610 test
events. Composition families are assigned before training. Metadata and source
paths remain in the manifests. Metadata filtering is not an independent legal
audit of every uploaded score.

The note-onset vocabulary avoids inflating accuracy with silent “hold” tokens.
It retains a top melody, onset duration, onset chord category and bass pitch;
it does not retain every orchestral voice or expressive articulation.

Install [FluidSynth](https://www.fluidsynth.org/) (`brew install fluid-synth` on
macOS) and obtain [GeneralUser GS](https://schristiancollins.com/generaluser.php).
Place `GeneralUser-GS.sf2` and its license under `data/soundfont/`. This checkout
uses GeneralUser GS 2.0.3 with its license preserved. Synthesis is rendered audio,
then measured for clipping, silence, onset flux and pitch-class energy. Those
checks do not establish sophisticated acoustic musical judgment.

## Scope

This is an executable research prototype, not the finished crown jewel. Current
MLP controls learn next-event prediction faster and more accurately than the
initial Cadence models. The experiment does not establish an efficiency advantage,
human-composer quality, autonomous musical taste, learned long-form orchestration or general
language understanding. The useful result is an inspectable, trainable composition
loop with real memory, coupled repair, counterfactual candidates and reversible
revision; its next improvements must survive the same measured comparisons.

## Polyphonic research run

The separate ensemble encoder predicts pitch, sounding duration, time to the next
attack, instrument family and velocity. It preserves simultaneous voices and held
notes. It uses the broader deduplicated PDMX publicdomain/CC0 metadata pool; rights
and artist provenance remain in the private manifest.

```sh
python tools/prepare_ensemble.py --name ensemble --workers 24
torchrun --standalone --nproc_per_node=4 tools/train_ensemble.py --dataset ensemble --name ensemble-2048-four-gpu --size 2048 --updates 30000 --batch 128 --max-seconds 3600
python tools/sample_ensemble.py --evaluate --prompt "heroic orchestral theme"
python tools/baseline_ensemble.py --size 1024 --updates 30000 --batch 512
python tools/sample_ensemble.py --seed 41 --conditional --prompt "heroic orchestral theme"
```

The completed four-A10G run used 5,270 neurons and 6,653,982 directed synapses, with
5,562,501 trainable parameters, drawing from 65,188 training pieces. It generates
multitrack MIDI; the same-input MLP control still predicts held-out events better.
See [the measured comparison and audio review](EVIDENCE.md#polyphonic-scaling-experiment).

Run the distributed preflight on a small prepared dataset with
`--check-distributed` before scaling. The current studio uses the established
phrase-planning checkpoint. Raw ensemble rollouts are separate research artifacts
until their musical behavior passes listening and held-out checks.

The copying screen compares declared training pieces with a generated composition:

```sh
python tools/similarity_audit.py runs/compositions/<id>/composition.json --workers 4
```

It checks exact 8/16-note interval windows, with and without onset timing. Common
patterns can match by chance; missing matches do not prove originality. The
full receipt identifies its corpus, projection and limitations.
