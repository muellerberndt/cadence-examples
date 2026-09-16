# Composer

The composer hears a two-bar phrase on a three-voice SID instrument and plays it back, then
plays on from a phrase it heard half of. One life learns the MOS 8580 chip by playing it: a
gesture is 28 registers over 541 motor neurons, the world model is a records cortex holding
what a gesture does to the sound, and the answer is a search over imagined hearings. The
example is published with its composing gates open, and the page states which predicates the
run fails.

- Page: https://floatingpragma.io/cadence-examples/composer/ (seven public-domain pieces: the
  demonstration, the brain's imitation and its continuation as the emulated chip played them,
  with the whole brain's recorded settling and the records cortex's reads and writes replayed
  in step with the audio).
- Receipt: [receipt.json](receipt.json), five acceptance seeds (10 to 14), run
  `S06-acceptance-20260916T021058Z`, measured on the held-out split of the private corpus.
  Gated: pitch within a semitone 0.505 (gate 0.85); onset F1 0.443 (gate 0.85);
  continuation surprise 1.635 nats per event (gate 1.608); imitation margin over the
  procedural composer 0.469 and over shuffled pairing 0.487 (gate 0.20 each); imitation
  margin over the same brain frozen from birth -0.033 (gate 0.20); renderer faults 1
  (gate 0); decision latency p95 19.4 ms (gate 20); illegal fields, unexpected silence,
  reward mismatches and non-finite blocks 0. Seven of the thirteen predicates pass.
- Open, measured every run and stated in the receipt under `open.measures` without a gate:
  composing under an instruction 0.274 (wanted 0.80), theme recall 0.013 (wanted 0.85),
  revision 0.125 (wanted 0.80). The judge of these is a reference judge; the stage packet asks
  for a listener panel of at least eight raters before they gate.
- The born-frozen margin: in this run a brain frozen from birth imitates as well as the trained
  one, so the search and the note reading carry the imitation. The sixth round scores
  candidates by the imagined hearing, as the instrument stage does; the page is rebuilt from
  that run when it lands.
- Supplied: the instrument (pyresidfp 0.17.0, MOS 8580, pinned file by file under
  `sources.emulator`), the hearing (64 log bands, level, onset and silence on the mix and on
  each voice's monitor), the corpus and its splits, the objectives, the search over imagined
  hearings, the settling schedule.
- Learned: the records of the world head (the next change of every hearing field under a
  gesture) and the settled regions' synapses, from empty records.
- Controls: born frozen, the procedural composer, shuffled pairing, a scrambled request.
- Stage sources: the brain, the environment, the corpus tools, the runner and the verifier
  are in the research tree the receipt hashes under `sources.stage_files`; they move into
  this directory when the stage passes its gates. `tools/verify_gallery.py` checks this
  receipt like the others except that its failed predicates are printed instead of failing
  the gallery and that absent stage files are skipped.
- The bundle: `assets/` (55 MB, ignored by git) holds the audio, the settling frames and the
  atlas; [checkpoints.json](checkpoints.json) names every file with its sha256 and the release
  that carries them; the Pages workflow downloads and verifies them before deploying. The
  private corpus (HVSC tunes and renderings of Lakh arrangements) is in no bundle.
- The page: [web/README.md](web/README.md) states the bundle format, the build
  (`web/build_page.py --bundle <dir> --out composer/index.html`) and the headless check
  (`web/check_page.py`). Nothing on the page is computed from a brain in the browser: the
  chip has no browser port yet, so the page replays a recorded life.
