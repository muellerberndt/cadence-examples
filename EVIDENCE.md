# Composer evidence and listening review

The current result is a trainable, inspectable composition prototype. Its regions
are bounded observer-like patches with local state, readback and repair; candidate
futures share learned seams but retain separate transient states. The measurements
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
phrase translation, MIDI duration and full owner/seam export.

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
exclusive musical specialization. A separate 128-example settlement probe reached
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

## Full circuit display

The established H=768 performer has **2,614 owners, 1,702,938 directed seams and
1,390,508 trainable parameters**, including motif and expectation wiring.
The browser receives every owner and every seam. A static full-graph raster is
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
performance benchmark. Every displayed snapshot still contains every owner.
End-to-end browser checks cover actual audio synthesis, MIDI/WAV export, full maps,
mobile layout, score readback, input release and preference retention. A forced
no-WebGL check also renders all 2,614 owners and 1,702,938 seams with a cached
Canvas2D path map; owner activity remains live, while traveling edge packets use
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

The larger experiment uses four A10Gs. Each GPU measures local free/nudged seam
contrasts; corresponding contrasts are averaged before the adaptive update. The
preflight compares that result with a complete batch on one device. Full data,
training and distributed-check receipts live under `data/ensemble/` and
`runs/ensemble-2048-four-gpu/` when the run completes. `tools/sample_ensemble.py`
produces separate raw multitrack rollouts; they are not silently substituted for
the established phrase-planning studio.

Metadata identifies 37 eligible John Williams-tagged records before parsing,
after excluding Thomas John Williams. Of the 34 usable records, 27 are training
pieces (12,780 sampled events), two are validation and five are test. Uploader license labels are not independent
verification of underlying composition rights or attribution. These records,
trained weights and music remain private. They do not establish a dedicated
John Williams style model or a public release clearance.

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
A future phrase-level learner should predict and compare tonal destinations,
motif transformations, instrumentation, tension and release over whole sections,
then be tested on unseen composition families. Merely extending the current
next-event training run does not establish that ability. Promotion requires
held-out multitrack evaluation, independent listening with theme/coherence and
surprise ratings, comparison against strong conventional controls, and a corpus
similarity audit. The current code does not establish original human-level
composition, unlimited musical memory, learned long-range form or consciousness.
