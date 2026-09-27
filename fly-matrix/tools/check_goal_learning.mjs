// One predeclared full-graph synthetic lesson, followed by weight-only restoration.
// Run from fly-matrix: node tools/check_goal_learning.mjs --receipt receipts/goal_learning_new.json
// Existing receipts are never overwritten. Runtime is bounded by six 1024-step solves.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { SettlingBrain } from '../web/brain.js';
import { ActorCriticLearner } from '../web/learner.js';
const base = new URL('../', import.meta.url);
const args=process.argv.slice(2);
assert.ok(args.length===2 && args[0]==='--receipt', 'usage: node tools/check_goal_learning.mjs --receipt receipts/NEW.json');
const destination=new URL(args[1], base);
assert.equal(destination.protocol,'file:');
assert.ok(!existsSync(destination), 'never overwrite a previous run');
const hash = data => createHash('sha256').update(data).digest('hex');
const paths = ['web/brain.js','web/learner.js','web/settlement-trace.js','web/neural-replay.js','web/data/brain_full.json','web/data/lessons_full.json'];
const sources = () => Object.fromEntries(paths.map(p => [p, hash(readFileSync(new URL(p, base)))]));
const config = {outputs:['mbon:MBON11:right','mbon:MBON05:left'],actions:[0,1],plastic:{pre:['kc'],post:['mbon']},critic:'kc',beta:.1,temperature:.3,nudgedSteps:10,tolerance:1e-3,gamma:.95,lam:.9,eta:1,etaBias:0,etaCritic:.05,cap:3,dopamineCap:1,tonic:{}};
const protocol = {maxSteps:1024,tolerance:1e-6,draw:0,reward:1,done:true,config,
  input:'payload lessons_setup.decision_senses plus fruit .9 bilateral and yeast .16 bilateral; no retinal drive',
  scope:'One synthetic terminal lesson on the full retained graph. Same frozen input, zero initial state for every readout comparison; restore/reinstate only plastic efficacy and effective weights. No browser outcome replay, discrimination, flight, multi-trial improvement or exact-gradient claim.'};
const result = {schema:'cadence.full-graph-causal-learning-probe/1',started:new Date().toISOString(),protocol,sources:sources(),producer_sha256:hash(readFileSync(fileURLToPath(import.meta.url))),passed:false};
const started=performance.now();
try {
  const payload=JSON.parse(readFileSync(new URL('web/data/brain_full.json',base),'utf8'));
  const b=new SettlingBrain(payload), l=new ActorCriticLearner(b,config);
  result.payload={neurons:b.n,edges:b.edges,plastic:l.edges.length};
  const senses={...payload.lessons_setup.decision_senses,'orn:decaying_fruit:left':.9,'orn:decaying_fruit:right':.9,'orn:yeasty:left':.16,'orn:yeasty:right':.16};
  result.senses=senses;for(const [name,value] of Object.entries(senses)) b.stimulate(name,value);
  const initial=Float64Array.from(l.efficacy), initialWeights=hash(Buffer.from(b.w.buffer)), initialBias=hash(Buffer.from(b.bias.buffer));
  function independentResidual(v,beta=0,choice=0) {
    const rest=1/(1+Math.exp(b.slope*b.threshold));
    const a=Float64Array.from(v,x=>{const raw=1/(1+Math.exp(-b.slope*(x-b.threshold)))-rest;return raw>0?raw/(1-rest):raw*(b.leak/rest);});
    const logits=l.outputs.map(i=>a[i]/config.temperature),peak=Math.max(...logits),exps=logits.map(x=>Math.exp(x-peak)),sum=exps.reduce((x,y)=>x+y,0),probs=exps.map(x=>x/sum);
    let residual=0;
    for(let i=0;i<b.n;i++) {let drive=0;for(let e=b.rowPtr[i];e<b.rowPtr[i+1];e++)drive+=b.w[e]*a[b.pre[e]];
      const output=l.outputs.indexOf(i);if(output>=0)drive+=beta*((output===choice?1:0)-probs[output]);
      residual=Math.max(residual,Math.abs(drive+b.drive[i]+b.bias[i]-v[i]));}
    return residual;
  }
  const d=l.actSettled({u:protocol.draw,maxSteps:protocol.maxSteps,tolerance:protocol.tolerance});result.decision=d;assert.equal(d.accepted,true);
  const baseline={outputs:l.outputs.map(i=>b.s[i]),p:[...l.probabilities()],state_sha256:hash(Buffer.from(b.s.buffer)),potential_sha256:hash(Buffer.from(b.v.buffer)),solve:d.solves.free};
  result.independent_phase_residuals={free:independentResidual(b.v),plus:independentResidual(l.pending.plusV,.1,d.choice),minus:independentResidual(l.pending.minusV,-.1,d.choice)};
  for(const r of Object.values(result.independent_phase_residuals))assert.ok(Number.isFinite(r)&&r<=1e-6);
  const expected=Float64Array.from(initial,(value,k)=>{
    const pre=b.pre[l.edges[k]],post=l.post[k],{plus,minus}=l.pending;
    const contrast=(plus[pre]*plus[post]-minus[pre]*minus[post])/(2*config.beta);
    const dopamine=Math.max(-config.dopamineCap,Math.min(config.dopamineCap,protocol.reward-d.value));
    return Math.max(-config.cap,Math.min(config.cap,value+config.eta*dopamine*contrast));
  });
  const update=l.learnSettled(protocol.reward,true);result.update=update;assert.equal(update.accepted,true);
  let maxError=0,maxChange=0,changedOver1e9=0;for(let k=0;k<initial.length;k++){maxError=Math.max(maxError,Math.abs(l.efficacy[k]-expected[k]));maxChange=Math.max(maxChange,Math.abs(l.efficacy[k]-initial[k]));if(Math.abs(l.efficacy[k]-initial[k])>1e-9)changedOver1e9++;}
  result.independent_update_max_error=maxError;result.maximum_efficacy_change=maxChange;result.changed_over_1e_minus_9=changedOver1e9;assert.ok(maxError<=1e-10);
  const learned=Float64Array.from(l.efficacy), learnedWeights=hash(Buffer.from(b.w.buffer));assert.notEqual(initialWeights,learnedWeights);
  function probe(efficacy){l.efficacy.set(efficacy);l.applyWeights();b.reset();const solve=b.settleControl(protocol.maxSteps,protocol.tolerance),residual=independentResidual(b.v);assert.equal(solve.converged,true);assert.ok(residual<=1e-6);return{solve,independent_residual:residual,weights_sha256:hash(Buffer.from(b.w.buffer)),outputs:l.outputs.map(i=>b.s[i]),p:[...l.probabilities()],state_sha256:hash(Buffer.from(b.s.buffer)),potential_sha256:hash(Buffer.from(b.v.buffer))};}
  result.probes={before:baseline,after:probe(learned),restored:probe(initial),reinstated:probe(learned)};
  result.weight_hashes={initial:initialWeights,learned:learnedWeights};
  assert.equal(result.probes.restored.weights_sha256,initialWeights);
  assert.equal(result.probes.reinstated.weights_sha256,learnedWeights);
  result.weights_restored_and_reinstated_exact=true;
  result.probability_change=result.probes.after.p[0]-baseline.p[0];
  assert.equal(result.probes.restored.state_sha256,baseline.state_sha256);assert.equal(result.probes.restored.potential_sha256,baseline.potential_sha256);
  assert.equal(result.probes.reinstated.state_sha256,result.probes.after.state_sha256);assert.equal(result.probes.reinstated.potential_sha256,result.probes.after.potential_sha256);
  assert.equal(hash(Buffer.from(b.bias.buffer)),initialBias);assert.ok(Math.abs(result.probability_change)>1e-8);
  result.passed=true;
} catch(e){result.error=String(e);result.stack=e.stack;}
result.elapsed_seconds=(performance.now()-started)/1000;result.sources_unchanged=JSON.stringify(sources())===JSON.stringify(result.sources);result.passed&&=result.sources_unchanged;
result.completed=new Date().toISOString();result.body_sha256=hash(JSON.stringify(result));writeFileSync(destination,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({passed:result.passed,error:result.error,payload:result.payload,probability_change:result.probability_change,max_update_error:result.independent_update_max_error,elapsed_seconds:result.elapsed_seconds,receipt:fileURLToPath(destination)},null,2));
if(!result.passed)process.exitCode=1;
