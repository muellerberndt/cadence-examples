/* Parity of sim/patch.js with cadence.RecordPatchNet: node sim/parity.js [fixture.json] */
'use strict';
const fs = require('fs'), path = require('path');
const { RecordPatch } = require('./patch.js');
const F = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, '..', 'ref', 'fixture.json'), 'utf8'));
const cfg = F.config, I = cfg.inputs, H = cfg.hidden, O = cfg.outputs, D = cfg.cells, R = I + H;
let failures = 0;
function close(name, a, b, tol) {
  a = Array.from(a); b = Array.from(b); if (a.length !== b.length) { failures++; console.log('FAIL', name, 'length', a.length, b.length); return; }
  let worst = 0; for (let i = 0; i < a.length; i++) { const e = Math.abs(a[i] - b[i]) / Math.max(1, Math.abs(b[i])); if (e > worst) worst = e; }
  if (!(worst <= tol)) { failures++; console.log('FAIL', name, 'max rel err', worst); } else console.log('ok  ', name, worst.toExponential(2));
}
function same(name, a, b) { if (a !== b) { failures++; console.log('FAIL', name, a, b); } else console.log('ok  ', name, a); }
const flat = rows => Float64Array.from(rows.flat());
const make = () => new RecordPatch({ inputs: I, hidden: H, outputs: O, seed: cfg.seed, cells: D, active: cfg.active, slowest: cfg.slowest, window: 4 });
const setP = (net, p) => { const q = {}; for (const k in p) q[k] = Float64Array.from(p[k]); net.setParameters(q); };

// the fixed cells: the projection and the offsets come from the same generator
{
  const net = make();
  const proj = new Float64Array(D * R); for (let i = 0; i < R; i++) for (let c = 0; c < D; c++) proj[c * R + i] = F.projection[i * D + c];
  close('projection', net.proj, proj, 1e-12); close('offset', net.offset, F.offset, 1e-12); close('scale', net.scale, F.scale, 1e-12);
}
// observe twice, then imagine
{
  const net = make(); setP(net, F.initial);
  const o1 = net.observe(flat(F.streams.U1), 3, flat(F.streams.Y1), { rate: 0.3 });
  close('observe1 output', o1.output, flat(F.observe1.output), 1e-9); close('observe1 loss', [o1.loss, o1.slowLoss], [F.observe1.loss, F.observe1.slow_loss], 1e-9);
  for (const k in F.observe1.delta) close('observe1 delta ' + k, net.grad[k], F.observe1.delta[k], 1e-9);
  for (const k in F.observe1.params) close('observe1 params ' + k, net[k], F.observe1.params[k], 1e-9);
  close('observe1 table', net.table, F.observe1.table, 1e-9); close('observe1 mean', net.mean, F.observe1.mean, 1e-9); same('observe1 seen', net.seen, F.observe1.seen); same('observe1 writes', o1.writes, F.observe1.writes);
  close('observe1 state', net.h, F.observe1.state, 1e-9);
  const o2 = net.observe(flat(F.streams.U2), 3, flat(F.streams.Y2), { rate: 0.3 });
  close('observe2 output', o2.output, flat(F.observe2.output), 1e-9);
  for (const k in F.observe2.params) close('observe2 params ' + k, net[k], F.observe2.params[k], 1e-9);
  close('observe2 table', net.table, F.observe2.table, 1e-9); close('observe2 mean', net.mean, F.observe2.mean, 1e-9);
  const im = net.imagine(flat(F.streams.U3), 4);
  close('imagine output', im.output, flat(F.imagine.output), 1e-9); close('imagine read', im.read, flat(F.imagine.read), 1e-9); close('imagine hidden', im.hidden.subarray(3 * H), F.imagine.hidden_final, 1e-9);
}
// the same first observation through the continuing interface: step, teach, flush with a window of three
{
  const net = new RecordPatch({ inputs: I, hidden: H, outputs: O, seed: cfg.seed, cells: D, active: cfg.active, slowest: cfg.slowest, window: 3 }); setP(net, F.initial);
  const U = F.streams.U1, Y = F.streams.Y1, out = new Float64Array(O), outs = [];
  for (let t = 0; t < 3; t++) { net.step(Float64Array.from(U[t]), out); outs.push(...out); const full = net.teach(Float64Array.from(Y[t])); if (t < 2 && full) { failures++; console.log('FAIL window full early'); } }
  net.flush(0.3);
  close('online output', outs, flat(F.observe1.output), 1e-9);
  for (const k in F.observe1.params) close('online params ' + k, net[k], F.observe1.params[k], 1e-9);
  close('online table', net.table, F.observe1.table, 1e-9); close('online state', net.h, F.observe1.state, 1e-9);
}
// backtracking admission
{
  const net = make(); setP(net, F.initial);
  const o = net.observe(flat(F.streams.U1), 3, flat(F.streams.Y1), { rate: 400.0, backtrack: true });
  same('backtrack updated', o.updated, F.backtrack.updated); same('backtrack reason', o.reason, F.backtrack.reason); same('backtrack replays', o.replays, F.backtrack.replays);
  close('backtrack rate', [o.acceptedRate], [F.backtrack.accepted_rate], 1e-12); close('backtrack losses', o.replayLosses, F.backtrack.replay_losses, 1e-9);
  for (const k in F.backtrack.params) close('backtrack params ' + k, net[k], F.backtrack.params[k], 1e-9);
}
// learning without writing
{
  const net = make(); setP(net, F.initial);
  const o = net.observe(flat(F.streams.U1), 3, flat(F.streams.Y1), { rate: 0.3, write: false });
  let s = 0; for (const v of net.table) s += Math.abs(v); close('nowrite table', [s], [F.nowrite.table_abs_sum], 1e-12); same('nowrite writes', o.writes, F.nowrite.writes);
}
// planning with and without feedback
{
  const net = make(); setP(net, F.initial);
  net.observe(flat(F.streams.U1), 3, flat(F.streams.Y1), { rate: 0.3 }); net.observe(flat(F.streams.U2), 3, flat(F.streams.Y2), { rate: 0.3 });
  close('plan state', net.h, F.plan.state, 1e-9); close('plan table', net.table, F.plan.table, 1e-9);
  const goal = new Float64Array(16).fill(0.5), weights = Float64Array.from([1, 1, 0, 1]);
  const p = net.plan(flat(F.streams.U3), 4, goal, weights, Int32Array.from([1, 4]), -1, 1, { feedback: [[0, 3]], rate: 0.5, maxSteps: 6, maxBacktracks: 8 });
  close('plan inputs', p.inputs, flat(F.plan.inputs), 1e-9); close('plan losses', p.losses, F.plan.losses, 1e-9); close('plan steps', p.steps, F.plan.steps, 1e-12);
  same('plan reason', p.reason, F.plan.reason); same('plan replays', p.replays, F.plan.replays); close('plan output', p.prediction.output, flat(F.plan.output), 1e-9);
  const q = net.plan(flat(F.streams.U3), 4, goal, weights, Int32Array.from([1, 4]), -1, 1, { rate: 0.5, maxSteps: 6, maxBacktracks: 8 });
  close('plan nofeedback inputs', q.inputs, flat(F.plan_nofeedback.inputs), 1e-9); close('plan nofeedback losses', q.losses, F.plan_nofeedback.losses, 1e-9); same('plan nofeedback reason', q.reason, F.plan_nofeedback.reason);
  const a = net.plan(flat(F.streams.U3), 4, new Float64Array(16).fill(-0.3), weights, Int32Array.from([1, 4]), -1, 1, { feedback: [[0, 3], [2, 5]], feedbackAdd: true, rate: 2.0, maxSteps: 6, maxBacktracks: 8 });
  close('plan additive inputs', a.inputs, flat(F.plan_add.inputs), 1e-9); close('plan additive losses', a.losses, F.plan_add.losses, 1e-9); same('plan additive reason', a.reason, F.plan_add.reason); same('plan additive replays', a.replays, F.plan_add.replays);
  const pb = net.plan(flat(F.streams.U3), 4, goal, weights, Int32Array.from([1, 4]), Float64Array.from(F.plan_bounds.lo), Float64Array.from(F.plan_bounds.hi), { rate: 2.0, maxSteps: 6, maxBacktracks: 8 });
  close('plan bounds inputs', pb.inputs, flat(F.plan_bounds.inputs), 1e-9); close('plan bounds losses', pb.losses, F.plan_bounds.losses, 1e-9); same('plan bounds reason', pb.reason, F.plan_bounds.reason); same('plan bounds replays', pb.replays, F.plan_bounds.replays);
  const pf = net.plan(flat(F.streams.U3), 4, goal, weights, Int32Array.from([1, 4]), -1, 1, { feedback: [[0, 3]], feedbackAdd: true, fixedRead: true, rate: 2.0, maxSteps: 6, maxBacktracks: 8 });
  close('plan fixed-read inputs', pf.inputs, flat(F.plan_fixed.inputs), 1e-9); close('plan fixed-read losses', pf.losses, F.plan_fixed.losses, 1e-9); close('plan fixed-read output', pf.prediction.output, flat(F.plan_fixed.output), 1e-9); same('plan fixed-read reason', pf.reason, F.plan_fixed.reason); same('plan fixed-read replays', pf.replays, F.plan_fixed.replays);
  const after = net.parameters(); for (const k in F.plan.params) close('plan leaves params ' + k, after[k], F.plan.params[k], 1e-15);
}
console.log(failures ? `${failures} FAILURES` : 'all parity checks passed');
process.exit(failures ? 1 : 0);
