// Frozen, one-candidate dose-response comparison. No training or live adoption.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, appendFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { SettlingBrain } from '../web/brain.js';
import { Life, HOVER_HEIGHT, ODOUR_SIGMA, ODOUR_CORE, ODOUR_CORE_WEIGHT, ODOUR_BASELINE, ODOUR_COMPARISON } from '../web/life.js';
import { Flight } from '../web/body.js';
import { attendedSenses, candidateScore, targetProbabilities, POLICY_OUTPUTS } from '../web/neural-policy.js';
import { odorReceptorResponse, ODOR_TRANSDUCTION_GENES } from '../web/odor-transduction.js';
const base = new URL('../', import.meta.url), args = process.argv.slice(2);
assert.ok(args.length === 2 && args[0] === '--receipt', 'usage: node tools/check_appetite_encoding.mjs --receipt receipts/NEW.json');
const destination = new URL(args[1], base), journal = new URL(args[1] + '.journal.jsonl', base);
assert.ok(!existsSync(destination) && !existsSync(journal), 'refusing to replace prior evidence');
const hash = x => createHash('sha256').update(x).digest('hex');
const paths = ['web/brain.js', 'web/life.js', 'web/body.js', 'web/senses.js', 'web/neural-policy.js', 'web/motor.js', 'web/assisted-life.js', 'web/settlement-trace.js', 'web/neural-replay.js', 'web/odor-transduction.js', 'web/data/brain_full.json'];
const sourceHashes = () => Object.fromEntries(paths.map(p => [p, hash(readFileSync(new URL(p, base)))]));
const protocol = {
  encoders: ['linear_control', 'hill_candidate'], genes: ODOR_TRANSDUCTION_GENES,
  geneStatus: 'Hand-set candidate; control is current Life.odour. No parameter selection after outcomes. Half-response values span the measured room concentration range, not fitted physiology or rewards.',
  hungers: [0, 1], contexts: ['spawn', 'banana_hover', 'bread_hover'], attentions: [null, 'banana', 'bread'],
  liveCap: 256, diagnosticCap: 1024, tolerance: 1e-6, temperature: .3, heading: .3,
  room: { table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: .75 }, fruits: { banana: { pos: [2.88, 2.33, .795] }, bread: { pos: [3.13, 2.28, .8068] } } },
  input: 'Actual Life.senses at fixed body poses. Only four ORN levels replaced by candidate. Same bilateral plume, no rendered retina, no sugar/contact/reward inputs. Attention follows current shared helper.',
  solve: 'Every branch resets potentials/activities to zero; existing weights/bias unchanged. After a failed 256-step checkpoint continue exactly that trajectory for at most 768 more steps. Record both checkpoints; independent full equation residual.',
  scope: 'Frozen-input encoding diagnostic, not embodied food seeking, endogenous hunger physiology, action admission or learning. Action probabilities use only qualified free readouts; nudged phases are NOT tested. Fed controller would not request a food decision; its conditional neural response is measured here intentionally.',
};
const receipt = { schema: 'cadence.appetite-encoding-candidate/1', started: new Date().toISOString(), protocol, sources: sourceHashes(), producer_sha256: hash(readFileSync(fileURLToPath(import.meta.url))), inputs: {}, probes: [], comparisons: [], completed: false };
const emit = (stage, data) => { appendFileSync(journal, JSON.stringify({ stage, ...data }) + '\n'); console.log(JSON.stringify({ stage, ...data })); };
emit('protocol', { protocol, sources: receipt.sources, producer_sha256: receipt.producer_sha256 });
const start = performance.now();
try {
  const b = new SettlingBrain(JSON.parse(readFileSync(new URL('web/data/brain_full.json', base), 'utf8')));
  const frozenWeights = hash(Buffer.from(b.w.buffer)), frozenBias = hash(Buffer.from(b.bias.buffer));
  receipt.payload = { neurons: b.n, edges: b.edges, initial_weights_sha256: frozenWeights, initial_bias_sha256: frozenBias };
  const positions = { spawn: [1.2, 1, 1.2], banana_hover: protocol.room.fruits.banana.pos.map((v, i) => v + (i === 2 ? HOVER_HEIGHT : 0)), bread_hover: protocol.room.fruits.bread.pos.map((v, i) => v + (i === 2 ? HOVER_HEIGHT : 0)) };
  function bilateral(flight, source) {
    const p = flight.p, R = flight.rotation(), left = [R[1], R[4], R[7]];
    const at = q => { const d2 = q.reduce((sum, v, i) => sum + (v - source[i]) ** 2, 0); return ODOUR_CORE_WEIGHT * Math.exp(-d2 / (2 * ODOUR_CORE ** 2)) + (1 - ODOUR_CORE_WEIGHT) * Math.exp(-d2 / (2 * ODOUR_SIGMA ** 2)); };
    const c = at(p), cl = at(p.map((v, i) => v + ODOUR_BASELINE * left[i])), cr = at(p.map((v, i) => v - ODOUR_BASELINE * left[i]));
    const ratio = Math.log(cl + 1e-9) - Math.log(cr + 1e-9);
    return { c, left: c * (1 + ODOUR_COMPARISON * Math.max(0, ratio)), right: c * (1 + ODOUR_COMPARISON * Math.max(0, -ratio)) };
  }
  function residual() {
    const rest = 1 / (1 + Math.exp(b.slope * b.threshold));
    const s = Float64Array.from(b.v, v => { const raw = 1 / (1 + Math.exp(-b.slope * (v - b.threshold))) - rest; return raw > 0 ? raw / (1 - rest) : raw * b.leak / rest; });
    let max = 0;
    for (let i = 0; i < b.n; i++) { let x = 0; for (let e = b.rowPtr[i]; e < b.rowPtr[i + 1]; e++) x += b.w[e] * s[b.pre[e]]; max = Math.max(max, Math.abs(x + b.drive[i] + b.bias[i] - b.v[i])); }
    return max;
  }
  function checkpoint(solve, totalIterations) {
    const independentResidual = residual();
    assert.ok(Number.isFinite(independentResidual) && Math.abs(solve.residual - independentResidual) <= 1e-10);
    const readouts = Object.fromEntries(POLICY_OUTPUTS.map(n => [n, b.mean(n)]));
    const score = solve.converged ? candidateScore(readouts) : null;
    return { solve, totalIterations, independentResidual, readouts, score, approachP: score === null ? null : 1 / (1 + Math.exp(-score / protocol.temperature)), kcMean: b.mean('kc'), activity_sha256: hash(Buffer.from(b.s.buffer)) };
  }
  for (const hunger of protocol.hungers) for (const context of protocol.contexts) {
    const life = new Life(new Flight(positions[context], protocol.heading), structuredClone(protocol.room), 1);
    life.hunger = hunger;
    const control = life.senses(null, 0), candidate = { ...control }, raw = {};
    for (const [fruit, odor] of [['banana', 'decaying_fruit'], ['bread', 'yeasty']]) {
      raw[fruit] = bilateral(life.flight, life.fruits[fruit].pos);
      for (const side of ['left', 'right']) {
        const name = `orn:${odor}:${side}`, u = raw[fruit][side];
        assert.ok(Math.abs(control[name] - Math.min(1, (.5 + hunger) * u)) <= 1e-14, 'control plume reconstruction');
        candidate[name] = odorReceptorResponse(u, hunger);
      }
    }
    for (const name of Object.keys(control)) if (!name.startsWith('orn:')) assert.equal(candidate[name], control[name]);
    assert.equal(control.leg_touch, 0); assert.equal(control['grn:sugar:labellum'], 0); assert.equal(control['dan:pam'], 0);
    receipt.inputs[`${hunger}:${context}`] = { hunger, context, position: positions[context], raw, linear_control: control, hill_candidate: candidate };
    for (const encoder of protocol.encoders) {
      const group = [];
      for (const attention of protocol.attentions) {
        b.clearStimuli(); b.reset();
        for (const [name, value] of Object.entries(attendedSenses(encoder === 'linear_control' ? control : candidate, attention))) b.stimulate(name, value);
        const t = performance.now(), first = b.settleControl(protocol.liveCap, protocol.tolerance);
        const probe = { hunger, context, encoder, attention, live: checkpoint(first, first.iterations) };
        if (first.converged) probe.final = probe.live;
        else { const extra = b.settleControl(protocol.diagnosticCap - protocol.liveCap, protocol.tolerance); probe.final = checkpoint(extra, first.iterations + extra.iterations); }
        probe.seconds = (performance.now() - t) / 1000;
        receipt.probes.push(probe); group.push(probe); emit('probe', probe);
      }
      const probabilities = stage => group.slice(1).every(p => p[stage].solve.converged) ? targetProbabilities(group.slice(1).map(p => ({ fruit: p.attention, score: p[stage].score })), protocol.temperature) : null;
      const comparison = { hunger, context, encoder, targetPAtLiveCap: probabilities('live'), targetPAtDiagnosticCap: probabilities('final'), unattendedQualifiedAtLiveCap: group[0].live.solve.converged, attendedBothQualifiedAtLiveCap: group.slice(1).every(p => p.live.solve.converged) };
      receipt.comparisons.push(comparison); emit('comparison', comparison);
    }
  }
  receipt.parametersUnchanged = frozenWeights === hash(Buffer.from(b.w.buffer)) && frozenBias === hash(Buffer.from(b.bias.buffer));
  assert.ok(receipt.parametersUnchanged);
  receipt.completed = true;
} catch (error) { receipt.error = String(error); receipt.stack = error.stack; }
receipt.seconds = (performance.now() - start) / 1000; receipt.finished = new Date().toISOString();
receipt.sourcesUnchanged = JSON.stringify(receipt.sources) === JSON.stringify(sourceHashes()); receipt.completed &&= receipt.sourcesUnchanged;
receipt.journal_sha256 = hash(readFileSync(journal)); receipt.body_sha256 = hash(JSON.stringify(receipt));
writeFileSync(destination, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({ completed: receipt.completed, seconds: receipt.seconds, error: receipt.error, receipt: fileURLToPath(destination) }));
if (!receipt.completed) process.exitCode = 1;
