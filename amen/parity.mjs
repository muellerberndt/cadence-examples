// node amen/parity.mjs : the browser engine against the archived Python run, sixteen bars from silence, highest-score mode.
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname, join} from 'node:path';
import {Brain, compose} from './web/engine.js';

const here = dirname(fileURLToPath(import.meta.url));
const model = JSON.parse(readFileSync(join(here, 'web/model/model.json')));
const buffer = name => { const b = readFileSync(join(here, 'web/model', name)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const started = Date.now();
const brain = new Brain(model, buffer(model.files.params), buffer(model.files.records_y), buffer(model.files.records_mean));
const built = Date.now() - started;
const runs = join(here, 'runs'); let failures = 0;
for (const run of ['record-composer-v10']) {
  const parity = JSON.parse(readFileSync(join(runs, run, 'parity.json'))); const L = model.layout;
  let worst = 0, t = 0, crops = 0, notes = 0, changes = 0; const began = Date.now();
  for (const step of compose(brain, {bars: parity.steps / 8, mode: 'argmax'})) {
    for (let k = 0; k < step.out.length; k++) worst = Math.max(worst, Math.abs(step.out[k] - parity.outputs[t][k]));
    const crop = step.played[L.drum_on] > 0.5 ? step.played.slice(0, L.crops).indexOf(1) : -1;
    const note = step.played[L.bass_on] > 0.5 ? step.played.slice(L.note_start, L.note_start + L.notes).indexOf(1) : -1;
    if (crop !== parity.played_crop[t]) crops++; if (note !== parity.played_note[t]) notes++; if ((step.played[L.change] > 0.5 ? 1 : 0) !== parity.played_change[t]) changes++;
    t++;
  }
  const ok = worst < 1e-5 && crops === 0 && notes === 0 && changes === 0; if (!ok) failures++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${run}: ${t} half-beats, largest output difference ${worst.toExponential(2)}, slices differing ${crops}, notes ${notes}, change points ${changes}; brain built in ${built} ms, composed in ${Date.now() - began} ms`);
}
process.exit(failures ? 1 : 0);
