# Composer evidence and listening review

The current result is a trainable, inspectable composition prototype. Its regions
are bounded observer-like patches with local state, readback and repair; candidate
futures share learned synapses but retain separate transient states. The measurements
below do not establish human-composer quality or an advantage over transformers.

## What the recordings reveal

Four complete 60-second recordings were reviewed by the external audio model
[Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), revision
`ae9e1690543ffd5c0221dc27f79834d0294cba00`. The evaluator received audio and a
neutral review prompt, without the composing architecture or a desired verdict.
Verbatim responses, input audio hashes, model revision and prompts are preserved
under `runs/audio-review-before/` and `runs/audio-review-after/`.
This is one model's fallible judgment, not a human listening study. Some fine
claims about rubato and exact chord extensions contradicted the quantized MIDI
and were discarded.

| Recording | Useful observation | Architectural or setup implication |
| --- | --- | --- |
| Initial sad piano, seed 17 | Melancholic but predictable; little dynamic or phrasing contrast | A local note critic and repeated four-chord scaffold cannot supply long-form development |
| Initial heroic orchestra, seed 23 | Heard as serene/reflective, dominated by strings | Instrument selection and expression did not realize the requested energetic brief |
| First revised piano, seed 17 | Still a simple, repetitive melancholic pattern | Better corpus likelihood and an improved explicit critic do not establish better music |
| First revised orchestra, seed 23 | Flute over strings; gentle, with little contrast or surprise | Changing a lead instrument alone does not create an orchestral dramatic arc |

These findings motivated sustained-note harmonic statistics, real inter-onset
rhythm statistics, varied harmonic routes, a bounded motif cue, phrase breathing,
voicing and expression changes, and a separate polyphonic learning experiment.
The current arrangement includes phrase-dependent dynamics, register and lead
instrument changes. Those arrangement rules are supplied; they are not learned
orchestration. A further full-audio review (`runs/audio-review-expressive/`) recognized the
brass/flute/string changes but still described the result as subdued and serene,
with limited dynamic contrast. This does **not** establish success on the heroic
brief. The evaluator also grouped flutes with brass and inferred counterpoint not
verified by the score; those details were not treated as reliable evidence.

The latest established-studio recording, `1789368733610028000-23/final.wav`
(SHA256 `af1e1fd9ba21a6b7af4640e1824c0f07ea39b251e022425c2cd52be8f3dbbe0d`),
was judged ceremonial, with prominent brass and transitions to woodwinds/strings.
The reviewer still requested greater dynamic contrast, melodic development and
instrumental balance. Its praise also conflicted with some of its own criticism.
This is encouraging qualitative feedback, not a controlled preference result or
proof that any one architectural change caused the improvement. The complete
response is `runs/ensemble-audio-review/audio-input-006.json`.

## Weaknesses fixed in the composition loop

- Phrase density had depended on absolute position in the score. Moving the same
  phrase later now leaves its density score unchanged.
- Rendering could reconstruct a different harmonic plan from the committed score.
  The MIDI now uses the committed chords and recorded note gates.
- Motif recall could keep repeating beyond its opening cue. It now releases after
  six notes and is scheduled for an answer or return.
- Each new imagined note had discarded the previous neural state. Each candidate
  now carries its own state into the next note, without writing imagined notes
  into learned memory or sharing state with other candidates.
- The global motif features continued to describe an artificial starting seed.
  After the first phrase they describe the actual committed opening.
- The motif critic searched every short window and reported chance matches as
  thematic development. It now compares phrase openings.
- Winner-minus-batch-average reward was always nonnegative, even when every
  candidate was poor. Appraisal now compares selected quality with its running
  expectation; shortfalls broaden exploration and positive errors narrow it.
- Preference feedback now teaches relative chord, rhythm and interval relationships,
  without rehearsing the exact score as supervised note labels. A validation-loss
  guard can reject the update. This supports a specific abstract preference memory;
  it does not prove that the complete event learner cannot memorize.

Regression tests cover causal expectation-to-intention links, actual state and
weight changes, isolated candidates, carried-state custody, key invariance,
phrase translation, MIDI duration and full neuron/synapse export.

## Recorded learning results

The first event model uses four prior melody/duration/chord/bass events plus
conditioning. Training data are 3,795,043 events from the strict classical PDMX
split. Validation uses the same fixed 2,048 held-out examples for the following
runs; lower mean negative log likelihood is better. Sampled batches may repeat
training events.

| Model | Updates × batch | Best validation NLL | Recorded training seconds |
| --- | ---: | ---: | ---: |
| Cadence H=96 | 2,000 × 256 | 1.83047 | 132 |
| Cadence H=256 | 6,000 × 256 | 1.79248 | 398 |
| Cadence H=768 | 12,000 × 256 | 1.75164 | 854 |
| Two-hidden-layer MLP, width 384 | 6,000 × 256 | **1.66546** | **14.66** |

The MLP currently predicts these events faster and more accurately. It can learn
statistical relationships too; this comparison does not identify it as merely a
memorizer. The larger Cadence run uses more parameters and updates, so the table
does not isolate a causal effect of width. Receipts live in `runs/gpu-*/` and
`runs/baselines/`; original training sources are under `runs/training_source/`.

The H=768 model's mean NLL on 8,192 test examples was **1.76233**. Lesioning its
harmonic, rhythmic or phrase region changes real output predictions. For example,
harmonic-region ablation changes melody argmax on 22.66% and chord argmax on 14.06%
of 128 probe examples. These are causal effects, not proof that the labels capture
exclusive musical specialization. A separate 128-example settling probe reached
its 512-step cap at maximum equation error **1.13e-4**, above the 1e-5 target.
A finite cap is not convergence. See `runs/assessment.json` and its preserved
source snapshot in `runs/assessment-source/`.

The 3,505-entry relationship memory was learned from 22,673 training pieces.
Held-out test likelihood improved over corpus marginals:

| Relationship | Conditional NLL | Marginal NLL | Test observations |
| --- | ---: | ---: | ---: |
| Chord transition | 1.8642 | 2.2931 | 50,717 |
| Onset interval | 1.2273 | 1.7872 | 147,493 |
| Melodic interval transition | 2.1587 | 2.4341 | 147,493 |

Use `checkpoints/intuition/receipt.json` for exact counts and source hashes.
These tables contain aggregate relationships, not complete pieces. The musical
critic remains supplied and incomplete; conditional likelihood is not beauty.

Parameter totals here exclude frozen biases. Older receipts used a counter that
included them: subtract 636 frozen input biases for the established model and
1,060 for the ensemble model. The public counter is corrected with mask regressions;
original training receipts and learned weights remain unchanged.

## Full circuit display

The established H=768 performer has **2,614 neurons, 1,702,938 directed synapses and
1,389,872 trainable parameters**, including motif and expectation synapses.
The browser receives every neuron and every synapse. A static full-graph raster is
composited with actual activation/repair/mismatch overlays. Density coloring,
zoom and replay speed affect presentation, not the simulated values.

Live snapshots sample every fourth note iteration and early/later solver states
for one explicitly identified candidate. All candidate note sequences remain
available. Diagnostic replays and isolated input-release probes are labeled.
Simulated population traces are not EEG; valence and detuning are numerical
modulators, not measured neurotransmitter concentrations. Quiet equilibria remain
quiet rather than receiving invented waves.

On the same orchestral configuration, full checkpoint recording took about 285s;
reduced temporal sampling took 67s. These were interactive runs, not a controlled
performance benchmark. Every displayed snapshot still contains every neuron.
End-to-end browser checks cover actual audio synthesis, MIDI/WAV export, full maps,
mobile layout, score readback, input release and preference retention. A forced
no-WebGL check also renders all 2,614 neurons and 1,702,938 synapses with a cached
Canvas2D path map; neuron activity remains live, while traveling edge packets use
WebGL2. On this machine, a 180-frame Chrome/Metal probe measured median 16.7ms
and p95 16.8ms frame intervals. These local rendering measurements are not an
ML-efficiency result; receipts and screenshots are under `runs/browser/`.

## Polyphonic scaling experiment

`tools/prepare_ensemble.py` retains simultaneous attacks, sounding duration,
inter-onset time, eight instrument families, velocity, held-note pitch classes and
recent tonal context. Input features are formed from previously heard events;
mode/style/arousal are supplied piece-level conditioning. The pipeline filters
PDMX publicdomain/CC0 uploader metadata, deduplicates and splits families before
training. This selects 77,354 candidate scores, of which 72,589 pass parsing and event-length
requirements: 65,188 training, 3,965 validation and 3,436 test pieces. The resulting
splits contain 21,138,078, 1,320,335 and 1,107,275 sampled events respectively.

A 1,500-score pilot produced 416,298 training events and 23,939 validation events.
After 1,500 updates on one A10G, mean validation NLL was 1.04889 across five new
heads. This number is **not comparable** with the four-head table above. Rare
instrument families were weak or absent in the validation sample.

The larger experiment uses four A10Gs. Each GPU measures local free/nudged synapse
contrasts; corresponding contrasts are averaged before the adaptive update. The
preflight compares that result with a complete batch on one device. Full data,
training and distributed-check receipts live under `data/ensemble/`,
`runs/distributed-check/` and `runs/ensemble-2048-four-gpu/`.
The run completed 30,000 updates at global batch 512: 15,360,000 sampled events,
with replacement, rather than a complete pass over every corpus event. Its
maximum distributed-contrast discrepancy was 3.79e-8 against tolerance 3e-5.

| Polyphonic predictor | Parameters | GPUs | Training seconds | Validation NLL | Test NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cadence H=2048 | 5,562,501 | 4 A10G | 3,965.85 | 0.85524 | 0.82808 |
| MLP, two hidden layers of 1,024 | 2,252,914 | 1 A10G | 146.17 | 0.77528 | **0.73841** |

Both use 30,000 × 512 sampled training events, identical encoded information and
five targets, and the same fixed 2,048 validation / 8,192 test rows. Best checkpoints
are selected on validation only. Seeds, parameter counts and hardware parallelism
differ; there is no hyperparameter sweep or matched hardware efficiency claim.
The MLP control remains stronger on this prediction task. Cadence's test head
accuracies are 32.71% pitch, 70.34% duration, 86.77% onset interval, 91.96%
instrument family and 89.25% velocity. The test subset is 58.2% keys; bass-family
accuracy is only 33.8% across 77 examples. Aggregate accuracy hides this weakness.

`tools/sample_ensemble.py` produces separate raw multitrack rollouts, without the
studio's supplied phrase critic or arrangement. Each uses the same final checkpoint
`8f740fa9b5647696f6cc7fa6f84e8aacfb249bc45a54d796f96ed294bcc6406d`.

| One-minute rollout | Events | Generation seconds | Duplicate-note guard corrections |
| --- | ---: | ---: | ---: |
| Heroic orchestra, seed 41, retained state | 496 | 14.68 | 83 |
| Same brief/seed, state reset each event | 445 | 13.27 | 100 |
| Same brief/seed, retained state plus attribute repair | 259 | 27.50 | 39 |
| Sad piano, seed 43, retained state | 55 | 1.77 | 0 |

These are single-seed observations, not aesthetic rankings. The conditional
readout uses 64 extra local repair steps per event; its lower event count is not
a compute-efficiency gain. All rollouts retain explicit density/duplicate guards,
a generic eight-note prime, coarse keyword controls and a 60-second boundary.
Warm orchestral output uses all eight families; the piano brief stays entirely
in the keys family. This demonstrates learned local instrument choice, not a
learned symphonic arrangement or long-form dramatic structure.

The audio evaluator described the warm orchestral rollout as several disparate
musical excerpts and the piano as melancholic but repetitive. Its cold-rollout
review invented tempo changes and a 3/4 meter, and its conditional-rollout review
degenerated into a repeated list of instruments not present in the rendered GM
programs. Those details and the conditional verdict are rejected as unreliable;
there is no established perceptual winner between decoders. Complete WAVs,
MIDI, note events, render measurements and verbatim reviews are preserved in
`runs/ensemble-samples/`, `runs/ensemble-audio-review/` and
`runs/ensemble-conditioned-audio-review/`. These rollouts do not replace the studio.

The post-training source snapshot is under `runs/four-gpu-source/`. The training
receipt pins an earlier rendering-program tuple; the exact earlier source was
recovered by reversing that tuple change and verifying the complete file SHA256
against the original receipt. See `runs/four-gpu-training-source/custody.json`.
That constant is not consumed by the encoding, graph or training update. Original
receipts remain unchanged; the custody verifier checks both source versions.

All seven local training receipts pass source, manifest and checkpoint checks.
The final run's 94 result/source files and all 702 dataset shards (3,723,468,560
bytes) were copied back and verified against remote SHA256 inventories. Loading
the final checkpoint with the current public core on CPU reproduces a fixed
16-row GPU check with equal accuracies and maximum head-NLL difference 2.61e-7.
This checks portability of the saved model, not musical generalization; receipts
are `runs/final-load-comparison.json` and `runs/ensemble-download-verification.json`.

Metadata identifies 37 eligible John Williams-tagged records before parsing,
after excluding Thomas John Williams. Of the 34 usable records, 27 are training
pieces (12,780 sampled events), two are validation and five are test. Uploader license labels are not independent
verification of underlying composition rights or attribution. These records,
trained weights and music remain private. They do not establish a dedicated
John Williams style model or a public release clearance.

An additional search inspected the official [Lakh dataset](https://colinraffel.com/projects/lmd/)
clean archive (17,259 files) and full filename index. The clean archive contained
no explicitly John Williams-labeled entries; the full index yielded five
unverified artist-name candidates, whose MIDI files were not fetched or trained.
A separate fan archive's database listing was broken. These searches did not
increase the training count. URLs, hashes and candidate metadata are preserved
in `data/lakh/acquisition.json`; the completed run should not be advertised as
trained on a large John Williams collection.

A 16-example validation damping probe found that the existing `dt=1` was fastest:
239 steps / 1.43s versus 279–756 steps / 1.73–4.77s for smaller allowed steps.
All final equation errors were below 1e-8. The attempted tuning was rejected;
`runs/settling-probe.json` retains the comparison and fixed-point differences.

## Bounded copying screen

`tools/similarity_audit.py` scanned all 65,188 pieces in the expanded training
pool with no parse errors. This pool contains every one of the established
model's 22,673 training pieces, verified by source identity. Three generated
melodies had no exact 16-note interval matches, with or without onset timing.
The latest 167-note orchestral sketch had 9/160 eight-note windows matching
both intervals and timing in 44 corpus pieces, and 117/160 matching intervals
alone. This indicates many familiar short contours; it does not demonstrate
memorization of those source pieces.

The scan projects source MIDI to the highest newly attacked pitched note at each
sixteenth-note onset. It can miss inner voices, transformed passages and copying
outside the corpus; common scale patterns can match by chance. No-match results
are not an originality guarantee. The exact query files, matching positions,
examples, hashes and scope are in `runs/similarity-audit.json`; coverage is in
`runs/similarity-coverage.json`.

## Next quality gate

More capacity can improve a prediction task; it cannot recover information an
encoder discarded or make an inadequate critic judge form. The new model has
eight-event context, aggregated held-note/tonal state and supplied global controls;
its training examples do not teach a recurrent minute-long composition trajectory.
The latent region names describe intended roles; their shared input projections
do not establish exclusive harmonic, rhythmic or instrumental specialization.
The coarse instrument-family vocabulary also merges distinct instruments, and
the three supplied style labels are not a learned emotional or artist-style model.

The independent five-head decoder can discard cross-attribute relationships. An
optional masked-Nudge decoder now selects an instrument and lets the same graph
repair the remaining note attributes before reading them. A cut-link regression
confirms a real instrument-to-pitch effect with no weight writes. Its paired
audio comparison did not yield a reliable quality verdict; this is not exact
joint sampling. Keep it optional until a broader listening comparison supports it.

A future phrase-level learner should predict and compare tonal destinations,
motif transformations, instrumentation, tension and release over whole sections,
then be tested on unseen composition families. Merely extending the current
next-event training run does not establish that ability. Promotion requires
held-out multitrack evaluation, independent listening with theme/coherence and
surprise ratings, comparison against strong conventional controls, and a corpus
similarity audit. The current code does not establish original human-level
composition, unlimited musical memory, learned long-range form or consciousness.

## The musician (2026-09-14)

The musician is a second brain (design in [MUSICIAN.md](MUSICIAN.md)) trained by
imitation of whole pieces as streams. Its evidence has its own conventions: the held-out
measure is the mean surprise over five event attributes on 64 validation streams heard from
their start (12,288 events after each piece's first sixteen), the same streams for every
row and every model. It is not comparable with the tables above, which score independent
windows of a different vocabulary.

**Control.** A GRU with the identical per-event information (the sixteen-event window
through a shared embedding, the sense row, the mood), five softmax heads, trained by
backpropagation through time over spans of 32 events on the same streams:
mean held-out surprise **0.708** (pitch 1.99, duration 0.79, time to attack 0.29, family
0.17, velocity 0.30) after 60,000 updates of 256 streams in 269 s on one A10G
(`runs/gru-control/receipt.json`).

**What did not learn, and why.** The first four musician runs (large, large without
memories, medium; 0.26 to 0.39 s per update of 256 streams on one A10G each) sat at the
output marginals, about 2.33, and did not move between 2,000 and 4,000 updates. Four
isolating runs (independent samples instead of streams, no warm start, no memories, no
belt) were equally frozen, so the stream machinery was not the cause. Measured locally:
half of every free stage was silent (the learning neuron model is exactly zero at rest and
the projections are symmetric), and the raw two-phase contrast fell from 4.7e-5 to 3e-6
within five updates while the signs of successive contrasts agreed at chance. The learning
rate of the earlier polyphonic design, 0.16 against the normaliser's floor, moved every
synapse by about one percent of its magnitude on the first update. With a tonic bias of
1.0 on every free neuron and a learning rate of 0.005 the held-out surprise fell steadily
in a local check (1.91 to 1.64 over 400 updates of 32 streams). Corrected runs began at
13:57 UTC; their rows are appended below as they arrive.

| Run | Design | Updates × streams | Held-out surprise | Per attribute | Seconds per update |
|---|---|---:|---:|---|---:|
| medium-v2 | version 2, cortex 1,024, eta 0.005 | 2,000 × 256 | 1.517 | 3.31 / 1.67 / 1.13 / 0.76 / 0.72 | 0.31 |
| medium-v2-eta002 | version 2, cortex 1,024, eta 0.002 | 2,000 × 256 | 1.383 | 3.30 / 1.48 / 1.03 / 0.44 / 0.67 | 0.31 |
| medium-v2-eta002 | same | 4,000 × 256 | 1.215 | 3.12 / 1.20 / 0.85 / 0.36 / 0.54 | 0.31 |
| medium-v2-eta002 | same, diverged | 6,000 × 256 | 2.207 | 3.99 / 2.63 / 1.82 / 1.57 / 1.02 | 0.31 |
| medium-v2 | eta 0.005, oscillating | 2,000 / 4,000 / 6,000 × 256 | 1.517 / 2.114 / 1.544 | | 0.31 |
| large-v2 | version 2, cortex 2,048, eta 0.005 | 2,000 / 4,000 × 256 | 2.197 / 2.111 | 3.44 / 1.81 / 1.66 / 1.96 / 1.68 at 4,000 | 0.50 |
| large-v2-eta002 | version 2, cortex 2,048, eta 0.002 | 2,000 × 256 | 1.751 | 3.38 / 1.58 / 1.57 / 0.90 / 1.33 | 0.49 |
| medium-v2-eta001d | eta 0.001 decaying to 0.0002 | 2,000 × 256 | 1.378 | 3.32 / 1.41 / 1.01 / 0.45 / 0.70 | 0.31 |
| medium-v2-eta0005d | eta 0.0005 decaying to 0.0001 | 2,000 × 256 | 1.389 | 3.39 / 1.37 / 1.05 / 0.49 / 0.64 | 0.31 |
| medium-v2-eta001d | same | 4,000 / 6,000 × 256 | 1.224 / 1.195 | 3.11 / 1.17 / 0.86 / 0.37 / 0.46 at 6,000 | 0.31 |
| medium-v2-eta0005d | same | 4,000 / 6,000 × 256 | 1.260 / 1.220 | 3.21 / 1.18 / 0.87 / 0.38 / 0.46 at 6,000 | 0.31 |
| large-v2-eta001d | version 2, cortex 2,048, eta 0.001 decaying to 0.0002 | 2,000 × 256 | 1.436 | 3.36 / 1.42 / 1.02 / 0.68 / 0.70 | 0.50 |
| large-v2-eta002d | resumed from large-v2-eta002 at 2,000; eta 0.002 decaying to 0.0002 | +2,000 / +6,000 × 256 | 1.448 / 1.858 (oscillating, retired) | | 0.49 |
| large-v2-eta001d | same | 4,000 / 6,000 / 8,000 / 10,000 × 256 | 1.364 / **1.267** / 1.377 / 2.764 (diverged) | 3.14 / 1.23 / 0.90 / 0.58 / 0.48 at 6,000 | 0.50 |
| medium-v2-eta001d | same | 8,000 / 12,000 × 256 | 1.20 / 1.198 (plateau) | 3.02 / 1.20 / 0.92 / 0.38 / 0.46 at 12,000 | 0.31 |
| medium-v2-nobelt | **no belt**: the ear projects straight into the cortices; eta 0.001 decaying | 2,000 / 4,000 × 256 | 1.230 / 1.125 | 3.04 / 1.09 / 0.72 / 0.34 / 0.44 at 4,000 | 0.28 |
| medium-v2-nobelt-r | resumed from the above at 4,000 | 6,000 / 8,000 × 256 | 1.079 / **1.062** | 2.87 / 1.05 / 0.66 / 0.33 / 0.40 at 8,000 | 0.28 |
| large-v2-guarded | resumed from large-v2-eta001d at 6,000; eta 0.0005 decaying, decay 1e-5, rollback guard | +1,000 / +2,000 × 256 | 1.225 / 1.222 | 3.10 / 1.16 / 0.81 / 0.57 / 0.48 | 0.50 |
| medium-v2-focus | resumed from medium-v2-eta001d at 6,000; 40% focus pieces; eta 0.0005 decaying | +1,000 … +5,000 × 256 | general 1.155 → 1.135; focus 1.167 → 1.140 | | 0.31 |
| medium-v2-focus-r | resumed from the above; eta 0.0004 decaying | +1,000 … +3,000 × 256 | general 1.132 → 1.145; focus 1.132 → 1.143 | 3.02 / 0.99 / 0.79 / 0.33 / 0.52 at +1,000 | 0.31 |
| medium-v2-steps96 | 96 free / 48 nudged steps, evaluated at 96 | 1,000 × 256 | 1.482 | 3.49 / 1.46 / 1.17 / 0.46 / 0.83 | 0.9 (stopped: three times the cost per update) |
| medium-v2-nobelt-r2 | resumed from nobelt-r at 8,000 | 10,000 / 12,000 × 256 | 1.045 / 1.053 (plateau) | 2.82 / 1.03 / 0.66 / 0.31 / 0.40 at 10,000 | 0.28 |
| large-v2-nobelt | large, no belt, eta 0.001 decaying, decay 1e-5, rollback | 2,000 … 12,000 × 256 | 1.360 / 1.182 / 1.102 / 1.056 / **1.044** / 1.048 | 2.81 / 1.04 / 0.65 / 0.32 / 0.39 at 10,000 | 0.45 |
| large-v2-nobelt-s48 | as above with 48 free / 24 nudged steps, evaluated at 48 | 2,000 / 4,000 × 256 | 1.361 / 1.183 (identical to the 32-step curve; stopped) | 3.06 / 1.11 / 0.75 / 0.56 / 0.44 at 4,000 | 0.59 |
| large-v2-nobelt | continued | 14,000 / 16,000 × 256 | 1.041 / 1.041 (plateau) | 2.76 / 1.06 / 0.69 / 0.31 / 0.39 | 0.45 |
| large-nobelt-focus | continued to completion | +5,000 … +10,000 × 256 | general 1.045 → 1.043; focus 1.039 → **1.031** (best, at +8,000) | 2.76 / 0.92 / 0.73 / 0.30 / 0.45 | 0.45 |
| large-v2-nobelt | continued | 18,000 / 20,000 / 22,000 × 256 | **1.033** / 1.050 / 1.047 | | 0.45 |
| medium-nobelt-nonorm | no belt, no RMS normalisation (plain momentum), eta 0.3 decaying to 0.05 | 1,000 … 7,000 × 256 | 1.509 / 1.275 / 1.195 / 1.160 / 1.140 / 1.144 / 1.118 | 2.82 / 1.04 / 0.67 / 0.53 / 0.53 at 7,000 | 0.28 |
| medium-nobelt-nonorm | eta 2.0 | 1,000 / 2,000 × 256 | 2.527 / 2.520 (saturated: 59% of the phrase cortex above 0.95; stopped) | | 0.28 |
| large-nobelt-focus | fine-tune of large-v2-nobelt at 10,000; 40% focus pieces; eta 0.0004 decaying | +1,000 × 256 | general 1.039; focus 1.046 | 2.82 / 0.93 / 0.70 / 0.31 / 0.47 | 0.45 |
| medium-nobelt-focus | fine-tune of the no-belt medium (from 1.045); 40% focus pieces; eta 0.0004 decaying | +1,000 … +8,000 × 256 | general 1.043 → 1.043; **focus 1.054 → 1.034** (from 1.19 before the fine-tune) | 2.85 / 0.93 / 0.71 / 0.31 / 0.47 at +1,000 | 0.31 |
| medium-v2-focus-r2 | belt design, continued | +4,000 / +5,000 / +6,000 / +7,000 × 256 | 1.174 / 1.501 (rolled back, step halved) / 1.130 / 1.128 | | 0.31 |
| medium-v2-focus-r2 | resumed focus fine-tune (belt design) | +1,000 … +3,000 × 256 | general 1.136 → 1.138; focus 1.132 → 1.133 (plateau) | | 0.31 |

The best held-out surprise of a musician so far is **1.045** (no belt, 10,000 updates of
256 streams, 2.7 h on one A10G) against **0.708** for the GRU control (60,000 updates,
269 s). The gap is largest on pitch (2.82 against 1.99). Cortex width did not decide it
(medium and large plateau alike at 1.04-1.05), and neither did the settling budget: a large
run with 48 free and 24 nudged steps traced the 32-step curve to the decimal. What remains
is the update itself: RMS-normalised steps move every synapse by about the step whatever
its contrast, and the free/nudged contrast at a 32-step transient is a coarse gradient.
A run without normalisation (`--normalize 0`, plain momentum steps, eta 2.0 decaying)
started at 19:05 UTC under the rollback guard.

A constant step of 0.002 or 0.005 learned for 4,000 updates and then diverged: with
RMS-normalised steps a consistent contrast moves a synapse by about the step every
update, so over thousands of updates the drift dwarfs the initial magnitudes. The runs
with a linearly decaying step (`--eta-final`) replaced them at 14:41 UTC; their rows
report the phrase cortex's mean activity and saturated/silent fractions as a guard.

**Listening.** The same external reviewer as above (Qwen2.5-Omni-7B, revision
`ae9e1690…`, transformers 4.57.6, run on the box with `tools/audio_review.py`, verbatim
responses under `runs/review-output/`) heard the first musician composition, made by the
medium brain after 2,000 updates for the brief "a sad, slow, gentle piano piece"
(`runs/musician-compositions/1789395325305939000-17`, 87 events, 16 bars). Draft: "slow,
gentle piano piece that evokes a sense of sadness… a few basic chords, which are repeated
throughout… not particularly complex". Final, after one accepted edit: "slow tempo and a
steady pulse, with a gradual build-up in the middle section… mostly in the key of C minor,
with some shifts to C major… ends with a slow, gentle cadence." One model's opinion on one
piece; it agrees with the brief and with the brain's stage of training.

**The settling budget.** On the medium brain after 6,000 updates, settling from the warm
state for 32 steps (the learning budget) leaves the intention neurons up to 0.87
activation units away from the 256-step equilibrium, with a maximum equation residual of
0.13; 128 steps still leave 0.35 and 4.7e-2. Held-out surprise on the same events is 1.002
at 32 steps and 0.951 at equilibrium. The two-phase contrast used for learning is
therefore a transient, and every held-out number above understates the brain. From
16:05 UTC the trainer accepts `--free-steps`, `--nudged-steps` and `--settle`; the run
`medium-v2-steps96` (96 free, 48 nudged, evaluated at 96) tests whether a nearer
equilibrium learns further than the 1.20 plateau; composition and practice settle 96
steps per event by default.

The medium brain after 6,000 updates (`checkpoints/musician-medium-6000`, held-out 1.195)
composed "a bright, heroic, loud orchestral theme" (`runs/musician-compositions/1789400830082160000-23`,
455 events, 16 bars, mood fit 1.0, surprise 1.15 against target 1.20). The reviewer: "a
strong, uplifting melody… supported by a full orchestral ensemble… the brass section,
particularly the trumpets and horns, plays a prominent role… the harmonic progression is
dynamic, with a clear and bold use of major chords… the rhythm is steady and driving… the
melody is well-developed, with clear phrasing and a memorable, soaring quality… the ending
is powerful and conclusive." Weaknesses named: repetitive harmony, a melody that could be
varied more, few dynamic shifts, brass/string balance. Verbatim response and audio hash in
`runs/review-output/1789400830082160000-23/`. The instrument families, their entries and
the chords are the brain's choices; the GM programs and the synthesis are supplied.

The no-belt medium brain after 8,000 updates (`checkpoints/musician-nobelt-8000`, held-out
1.062) composed the same heroic brief (`runs/musician-compositions/1789405823321561000-23`,
282 events, mood fit 0.79: it chose a chamber texture of flute and piano rather than the
full orchestra). The reviewer heard "a bright and heroic orchestral theme, primarily driven
by the flute and piano… well-structured and well-balanced, with a strong sense of movement
and progression", and described four sections in which the original melody returns with a
new twist each time. That returning melody is what the form record and the slow piece
trace are for; one piece is not a measurement of it.

**Practice, first round** (`runs/medium-practice-round0/receipt.json`, from the no-belt
checkpoint): six pieces in random moods, mean listener score 1.54, three kept, 513
self-imitation updates weighted by advantage, then 300 rehearsal updates. Held-out
surprise on real music rose from 1.062 to 1.159: with RMS-normalised steps each of the
513 tiny-batch updates moves every synapse by about the full step, and the listener's
preference is not the corpus. The loop now scales the own-piece step by 0.25 and rolls a
round back (halving that scale) when real-music surprise exceeds the round's start by
3%; rounds with the guard are appended below as they arrive.

**Practice, guarded** (`runs/medium-practice/receipt.json`, from the no-belt checkpoint at
1.053 held-out): round 0 with the own-piece step scaled by 0.25: six pieces, mean
listener score 1.46, three kept, 610 self-imitation updates, then 300 rehearsal updates;
held-out general 1.065, focus 1.083 (round kept by the 3% guard). Later rounds append.
Round 1: mean score 1.55, three kept, held-out general 1.101, above the guard: rolled
back to the round's start and the own-piece step halved to 0.125. Self-imitation
weighted by the listener sharpens what the listener measures (its scores rose from 1.46
to 1.55) and, so far, does not improve likelihood on real music; the guard keeps it from
damaging the brain that plays.

The large no-belt brain after 10,000 updates (`checkpoints/musician-large-10k`, held-out
1.044) composed the heroic brief once more (`runs/musician-compositions/1789410637134665000-23`,
282 events, mood fit 0.93, surprise 1.05 against target 1.04, both edits rejected as no
improvement). The reviewer: "begins with a flute melody… accompanied by a brass section,
which adds a sense of grandeur and heroism… supported by a string section… simple yet
memorable, with a clear pulse… phrasing is well-defined, with distinct sections that build
upon each other… surprises… from the unexpected changes in dynamics and harmony… the
ending of the piece is satisfying, with a final flourish… well-crafted, with a clear sense
of direction and purpose." No weaknesses were named this time; the reviewer also praised
"the performance" and "the sound quality", which are GeneralUser GS synthesis, a reminder
of its limits as a judge.
Rounds 2 to 7 (own-piece step 0.125, then 0.0625 after a second rollback at round 5): mean
listener scores 1.64, 1.68, 1.29, 1.48, 1.37, 1.61; held-out general 1.067, 1.068, 1.060,
(1.073 rolled back), 1.064, 1.058. The guard holds real-music surprise within 1% of the
start while the brain practises; whether the practised pieces are better music is a
listening question the listener cannot answer about itself. A second practice loop
started at 20:00 UTC from the focus-tuned large brain (`runs/large-practice/`).

The focus-tuned large brain (`checkpoints/musician-large-focus`, general 1.043, focus 1.031)
composed "an epic, heroic, loud orchestral film theme with brass"
(`runs/musician-compositions/1789416291195318000-41`, 478 events, 16 bars, mood fit 1.0,
surprise 1.06 against target 1.03, 36 minutes on the Mac with the recording). The
reviewer: "a bold and dramatic introduction, likely played by brass instruments… a more
melodic section, possibly featuring woodwinds… builds with a rhythmic pulse… the harmonic
direction is rich and dynamic… well-structured, with clear contrasts between the different
sections… the ending is satisfying… a strong example of an epic, heroic, and loud
orchestral film theme." Weaknesses named: instrument variety, harmonic and melodic
intricacy, dynamic range.

**Practice on the focus-tuned large brain** (`runs/large-practice/receipt.json`, own-piece
step 0.125, rehearsal half from the focus set): round 0 kept, six pieces, mean listener
score 1.49, three kept, held-out general 1.039 (from 1.043) and focus 1.024 (from 1.031),
the first round in which practice plus rehearsal improved both held-out numbers.

The large no-belt run continued with the guard: 26,000 → 1.715 (rolled back to its best,
step halved), 28,000 → **1.030**, its best row. The unnormalised medium plateaued at 1.094
(14,000 to 15,000), above the normalised 1.045.
