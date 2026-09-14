// Retention is a response magnitude after removing transient weights, not a label score.
import { SynapticMemory, keys } from '../shared/engine.js';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
export function run() {
  const [cue, other] = keys();
  const value = [1, 0, 0, 0], correction = [0, 1, 0, 0];
  const once = new SynapticMemory(), repeated = new SynapticMemory(),
    salient = new SynapticMemory(), disabled = new SynapticMemory({consolidation:0});
  once.observe(cue, value);
  for (let i = 0; i < 40; i++) {
    repeated.observe(cue, value);
    disabled.observe(cue, value);
  }
  salient.observe(cue, value, 1, {salience:19});
  for (const m of [once, repeated, salient, disabled]) m.reset();
  const values = {
    ordinary:once.predict(cue)[0], repeated:repeated.predict(cue)[0],
    salient:salient.predict(cue)[0], consolidation_disabled:disabled.predict(cue)[0],
  };
  for (let i = 0; i < 200; i++) salient.observe(other, correction);
  values.after_distraction = salient.predict(cue)[0];
  for (let i = 0; i < 60; i++) salient.observe(cue, correction);
  salient.reset();
  values.revised = salient.predict(cue)[1];
  return Object.fromEntries(Object.entries(values).map(([k,v])=>[k,+v.toFixed(10)]));
}
export function report() {
  return { schema:'cadence.consolidation/v1',
    description:'Unit-cue response after clearing all transient residual weights. Single stream; orthogonal distractors; salience is supplied, not a neurotransmitter concentration.',
    sources:Object.fromEntries(['shared/engine.js','memory/consolidation_benchmark.mjs'].map(name=>
      [name,createHash('sha256').update(readFileSync(new URL('../'+name,import.meta.url))).digest('hex')])),
    rates:{decay:.9,consolidation:.05}, exposures:{ordinary:1,repeated:40,salient:1,salience:19,distractors:200,corrections:60},
    responses:run() };
}
if (process.argv[1] && pathToFileURL(process.argv[1]).href===import.meta.url) {
  const result = report();
  writeFileSync(new URL('consolidation_evidence.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
  console.log(result.responses);
}
