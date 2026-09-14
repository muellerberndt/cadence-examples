# Model data sheet: maestro-1

`maestro-1` is the pretrained musician shipped with this example. It is one cadence brain
of wired cortices that hears music as a stream of note events, holds the phrase in working
memory, remembers the form of the piece, takes a mood, and intends the next event. The
studio (`serve_musician.py`) composes with it by imagining continuations through its own
predictions, listening to the whole draft and editing the weakest passage.

Fetch it with `python tools/fetch_model.py` (the release assets are listed in
`tools/models.json` with their SHA-256). Design: [MUSICIAN.md](MUSICIAN.md).

## Architecture

| | |
|---|---|
| Library | cadence 0.9 (`cadence-net`), graded neurons, one joint equilibrium, local free/nudged contrast learning, no backpropagation graph |
| Neurons | 17,855 in 14 regions |
| Directed synapses | 68,570,458 |
| Trainable parameters | 52,849,100 (reciprocal pairs share one efficacy; every free neuron has a learned bias) |
| Design | `Design(version=2, cortex=2048, phrase=2048, belt=False, tonic=1.0, seed=41)` |
| Regions | ear 1,824 · sense 144 · mood 21 · interval 392 · melody 2,048 · harmony 2,048 · rhythm 1,024 · timbre 1,024 · form 1,024 · phrase 2,048 · prefrontal 2,048 · recall 2,048 · piece 2,048 · intention 114 |
| Clamped inputs | ear (the last 16 events, one-hot), sense (chroma, sounding notes by family, beat, bar, progress), mood (7 measured classes), interval (relative pitch of the last 8 steps), prefrontal (a trace of the phrase cortex, decay 0.85), piece (a slow trace, decay 0.98), recall (a delta-rule record keyed by the bar of a 16-bar cycle) |
| Output | intention: five softmax choices per event: pitch (73, key-relative), duration (12), time to next attack (13), instrument family (8), velocity (8) |
| Neuron model | `learning_neuron_model(dt=1, leak=0.1)`, resting bias 1.0 on every free neuron |
| Checkpoint | `brain.npz` (cadence checkpoint format 2, uncompressed, 4.46 GB) with `brain.design.json` and the training `receipt.json` |

## Training data

| | |
|---|---|
| Source | [PDMX v9](https://zenodo.org/records/15571083) (Long et al.), MIDI renderings of MuseScore uploads |
| Selection | uploader license `publicdomain` or `cc-zero`, no marked license conflict, the deduplicated subset; every genre, solo and multi-track; pieces with at least 64 notes |
| Pool | 77,354 files selected; 72,589 parsed into streams |
| Split | by composition family before training: train 65,188 pieces (46,808,931 events), validation 3,965 (3,007,293), test 3,436 (2,454,280) |
| Stream | events in fixed order (pitch, duration, time to next attack, family, velocity), the causal sense row before each event, seven mood classes measured on the whole piece; pitches key-relative to a measured tonal center |
| Manifest | `data/musician/manifest.json`, SHA-256 `66cccc0ee00e…` in the receipt; per-piece metadata, uploader license labels and MIDI hashes retained |
| Rights | uploader license metadata is retained verbatim and is not an independent rights audit of every uploaded score. No file outside the public-domain/CC0 pool was used for this model |

## Training recipe

| | |
|---|---|
| Objective | imitation: predict the next event of 256 pieces walked in parallel as streams; one free phase from the previous equilibrium with working memory and form record written in, two nudged phases (cross-entropy nudge, beta 0.3), one local update of every synapse |
| Steps | 38,000 updates (best), 9,728,000 events, 21,817 pieces; 0.445 s per update on one NVIDIA A10G (cuda, float32); 5.2 h |
| Learner | eta 0.001 decaying linearly toward 0.0002, bias eta one tenth, momentum 0.8, RMS-normalised steps (0.98, floor 0.01), weight decay 1e-5 per update, 32 free and 16 nudged settling steps, temperature 0.2 |
| Stability | rollback rule: when held-out surprise exceeded the best by 15 percent the best checkpoint was restored and the step halved; it fired at updates 26k, 30k, 32k, 34k and 36k. The best checkpoint is the one saved at 38k |
| Command | `python tools/train_musician.py --name large-v2-nobelt --no-belt --dataset musician --batch 256 --updates 50000 --evaluate-every 2000 --tonic 1.0 --version 2 --eta 0.001 --eta-final 0.0002 --decay 0.00001 --rollback 1.15 --device cuda:0` |

## Evaluation

Held-out surprise is the mean negative log-likelihood per attribute on 12,288 validation
events heard as streams from the start of each piece (64 streams, 256 events each), without
learning. Lower is better. It is a prediction score, not a musical-quality score.

| Model | Parameters | Held-out surprise | pitch | duration | attack | family | velocity |
|---|---:|---:|---:|---:|---:|---:|---:|
| maestro-1 (this model) | 52.8M | **1.023** | 2.725 | 1.014 | 0.679 | 0.312 | 0.386 |
| GRU control, same streams, backpropagation through time | 7.6M | 0.708 | 1.986 | 0.792 | 0.289 | 0.169 | 0.304 |

Held-out accuracy of the most expected choice (maestro-1): pitch 22.3 %, duration 66.0 %,
attack 78.3 %, family 87.7 %, velocity 86.7 %.

The GRU control (`tools/baseline_musician.py`, hidden 1024, span 32, 60,000 updates,
269 s) predicts held-out events better than this model. The gap is the open research
question of this example; width and settling depth did not close it (runs of 2026-09-14
in the private receipts).

## Intended use and limits

* A research prototype for composing short pieces in the studio from a mood brief and for
  inspecting every neuron and settling iteration while it does so. Composing sixteen bars
  with the recording on takes tens of minutes on a laptop.
* The mood input is seven measured classes chosen by a bounded keyword parser; it is not
  language understanding or a model of a named composer's style.
* Long-form structure is carried only by the working-memory traces and the 16-bar form
  record; the model does not plan a whole piece before writing it.
* Generated pieces should be checked against the training corpus for copied material
  (`tools/similarity_audit.py`) before any use beyond listening.
* Not established: human-level composition, an advantage over conventional sequence
  models, or any claim about biological equivalence.

## Provenance

| | |
|---|---|
| Trained | 2026-09-14, run `large-v2-nobelt`, receipt in `checkpoints/maestro-1/receipt.json` |
| Sources hashed in the receipt | `tools/train_musician.py`, `composer/musician.py`, `tools/prepare_musician.py` at training time |
| Private variant | a fine-tune of this model on a small film-score subset with uploader license labels only (`large-nobelt-focus`) is not published |

## The older phrase-planning composer

`serve.py` (port 8078) runs an earlier, smaller composer (`checkpoints/best`, 2,319 neurons,
1,699,498 directed synapses, 1,387,262 parameters) trained on the 25,248-piece classical
subset with a phrase-memory design; see [BRAIN.md](BRAIN.md) and [EVIDENCE.md](EVIDENCE.md).
It is kept for comparison and is also published as a release asset (`phrase-composer`).
