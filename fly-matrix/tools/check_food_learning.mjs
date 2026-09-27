// Paired food responses and four one-lesson intervention arms on the full graph.
// No body/worker code is altered. Synthetic reward is explicit; no reward-location
// information enters input. Existing receipt and progress files are never replaced.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, appendFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { SettlingBrain } from '../web/brain.js';
import { ActorCriticLearner } from '../web/learner.js';
import { Life, HOVER_HEIGHT } from '../web/life.js';
import { Flight } from '../web/body.js';
import { attendedSenses, candidateScore, targetProbabilities } from '../web/neural-policy.js';
const base=new URL('../',import.meta.url),args=process.argv.slice(2);
assert.ok(args.length===2&&args[0]==='--receipt','usage: node tools/check_food_learning.mjs --receipt receipts/NEW.json');
const destination=new URL(args[1],base),journal=new URL(args[1]+'.journal.jsonl',base);
assert.ok(!existsSync(destination)&&!existsSync(journal),'refusing to replace prior evidence');
const hash=x=>createHash('sha256').update(x).digest('hex');
const paths=['web/brain.js','web/learner.js','web/life.js','web/body.js','web/senses.js','web/neural-policy.js','web/motor.js','web/assisted-life.js','web/settlement-trace.js','web/neural-replay.js','web/data/brain_full.json','web/room.js'];
const sources=()=>Object.fromEntries(paths.map(p=>[p,hash(readFileSync(new URL(p,base)))]));
const config={outputs:['mbon:MBON11:right','mbon:MBON05:left'],actions:[0,1],plastic:{pre:['kc'],post:['mbon']},critic:'kc',beta:.1,temperature:.3,gamma:.95,lam:.9,eta:1,etaBias:0,etaCritic:.05,cap:3,dopamineCap:1,tonic:{}};
const protocol={steps:1024,tolerance:1e-6,config,hunger:.6,heading:.3,
  room:{table:{x0:2.45,x1:3.55,y0:1.95,y1:2.65,z:.75},fruits:{banana:{pos:[2.88,2.33,.795]},bread:{pos:[3.13,2.28,.8068]}}},
  contexts:['spawn','banana_hover','bread_hover'],trainingArms:['reward_banana','reward_bread','frozen_banana','frozen_bread'],
  training:{lessons:1,actionDraw:0,reward:1,terminal:true,selectedOdor:'named food at its own +HOVER_HEIGHT; approach deliberately sampled with draw0'},
  postProbes:['spawn:banana','spawn:bread','banana_hover:banana','bread_hover:bread'],
  scope:'Actual Life.senses transduction at fixed room/body poses, no rendered retina or dynamics. Explicit synthetic one-lesson intervention, same initial parameters, opposite-food and frozen-actor controls. Near-food response selectivity compares two separate real sensor contexts and is not a current in-place target-choice probability. No innate preference, natural learning, multitrial improvement, generalization or exact-gradient claim.'};
const r={schema:'cadence.food-learning-paired/1',started:new Date().toISOString(),protocol,sources:sources(),producer_sha256:hash(readFileSync(fileURLToPath(import.meta.url))),baselines:{},arms:{},completed:false};
const emit=(stage,data)=>{appendFileSync(journal,JSON.stringify({stage,...data})+'\n');console.log(JSON.stringify({stage,...data}));};
emit('protocol',{protocol,sources:r.sources,producer_sha256:r.producer_sha256});
const start=performance.now();
try{
 const payload=JSON.parse(readFileSync(new URL('web/data/brain_full.json',base),'utf8')),b=new SettlingBrain(payload);
 const first=new ActorCriticLearner(b,config),initial=first.efficacy.slice(),initialW=b.w.slice(),initialBias=b.bias.slice();
 r.payload={neurons:b.n,edges:b.edges,plastic:initial.length};
 const positions={spawn:[1.2,1,1.2],banana_hover:protocol.room.fruits.banana.pos.map((x,i)=>x+(i===2?HOVER_HEIGHT:0)),bread_hover:protocol.room.fruits.bread.pos.map((x,i)=>x+(i===2?HOVER_HEIGHT:0))};
 const senses={};for(const [key,p]of Object.entries(positions)){const life=new Life(new Flight(p,protocol.heading),structuredClone(protocol.room),1);life.hunger=protocol.hunger;senses[key]=life.senses(null,0);}
 r.contexts=Object.fromEntries(Object.keys(senses).map(k=>[k,{position:positions[k],senses:senses[k]}]));
 const saved=new Map();
 const softmax=values=>{const e=values.map(v=>Math.exp((v-Math.max(...values))/config.temperature)),sum=e.reduce((x,y)=>x+y,0);return e.map(x=>x/sum);};
 function residual(v,beta=0,choice=0){const rest=1/(1+Math.exp(b.slope*b.threshold));const s=Float64Array.from(v,x=>{const raw=1/(1+Math.exp(-b.slope*(x-b.threshold)))-rest;return raw>0?raw/(1-rest):raw*b.leak/rest;});const p=softmax(first.outputs.map(i=>s[i]));let max=0;for(let i=0;i<b.n;i++){let input=0;for(let e=b.rowPtr[i];e<b.rowPtr[i+1];e++)input+=b.w[e]*s[b.pre[e]];const j=first.outputs.indexOf(i);if(j>=0)input+=beta*((j===choice?1:0)-p[j]);max=Math.max(max,Math.abs(input+b.drive[i]+b.bias[i]-v[i]));}return max;}
 function input(context,fruit){b.clearStimuli();for(const[n,v]of Object.entries(attendedSenses(senses[context],fruit)))b.stimulate(n,v);}
 function probe(context,fruit,save=false){input(context,fruit);b.reset();const solve=b.settleControl(protocol.steps,protocol.tolerance),independentResidual=residual(b.v);assert.ok(Math.abs(solve.residual-independentResidual)<=1e-10);const readouts=Object.fromEntries(config.outputs.map(n=>[n,b.mean(n)]));const result={context,fruit,solve,independentResidual,readouts,score:solve.converged?candidateScore(readouts):null,approachP:solve.converged?softmax(first.outputs.map(i=>b.s[i]))[0]:null,state_sha256:hash(Buffer.from(b.s.buffer)),potential_sha256:hash(Buffer.from(b.v.buffer)),kcMean:b.mean('kc')};if(save&&solve.converged)saved.set(context+':'+fruit,{v:b.v.slice(),s:b.s.slice(),steps:b.steps});return result;}
 for(const context of protocol.contexts){const pair=[];for(const fruit of ['banana','bread']){const p=probe(context,fruit,true);r.baselines[context+':'+fruit]=p;pair.push(p);emit('baseline',{probe:p});}if(pair.every(p=>p.solve.converged))emit('baseline_target',{context,p:targetProbabilities(pair,config.temperature)});}
 function responseMetrics(probes){const a=probes['banana_hover:banana'],z=probes['bread_hover:bread'],x=probes['spawn:banana'],y=probes['spawn:bread'];return{nearApproachGap:a?.approachP!==null&&z?.approachP!==null?a.approachP-z.approachP:null,spawnTargetP:x?.solve.converged&&y?.solve.converged?targetProbabilities([x,y],config.temperature):null};}
 r.before=responseMetrics(r.baselines);
 for(const arm of protocol.trainingArms){b.w.set(initialW);b.bias.set(initialBias);const fruit=arm.endsWith('banana')?'banana':'bread',context=fruit+'_hover',frozen=arm.startsWith('frozen');const l=new ActorCriticLearner(b,{...config,...(frozen?{eta:0,etaBias:0,etaCritic:0}:{})});input(context,fruit);b.reset();const free=saved.get(context+':'+fruit);if(free){b.v.set(free.v);b.s.set(free.s);b.steps=free.steps;}
   const entry=r.arms[arm]={fruit,frozen,decision:l.actSettled({u:protocol.training.actionDraw,maxSteps:protocol.steps,tolerance:protocol.tolerance}),post:{}};
   if(entry.decision.accepted){entry.independentPhaseResiduals={free:residual(b.v),plus:residual(l.pending.plusV,config.beta,entry.decision.choice),minus:residual(l.pending.minusV,-config.beta,entry.decision.choice)};assert.ok(Object.values(entry.independentPhaseResiduals).every(x=>x<=1e-6));
     const expected=Float64Array.from(initial,(v,k)=>{const pre=b.pre[l.edges[k]],post=l.post[k],p=l.pending;const contrast=(p.plus[pre]*p.plus[post]-p.minus[pre]*p.minus[post])/(2*config.beta);return Math.max(-config.cap,Math.min(config.cap,v+(frozen?0:config.eta)*contrast));});entry.update=l.learnSettled(1,true);assert.ok(entry.update.accepted);entry.independentUpdateMaxError=Math.max(...l.efficacy.map((v,k)=>Math.abs(v-expected[k])));assert.ok(entry.independentUpdateMaxError<=1e-10);
   }
   entry.weight_sha256=hash(Buffer.from(b.w.buffer));entry.initial_weight_sha256=hash(Buffer.from(initialW.buffer));
   for(const key of protocol.postProbes){const [c,f]=key.split(':');entry.post[key]=probe(c,f);}
   entry.metrics=responseMetrics(entry.post);entry.sameInputChanges=Object.fromEntries(protocol.postProbes.map(k=>[k,entry.post[k].approachP===null||r.baselines[k].approachP===null?null:entry.post[k].approachP-r.baselines[k].approachP]));
   if(frozen){assert.equal(entry.weight_sha256,entry.initial_weight_sha256);for(const k of protocol.postProbes)assert.equal(entry.post[k].state_sha256,r.baselines[k].state_sha256);}
   emit('arm',{arm,result:entry});
 }
 r.findings={before:r.before,allTrainingQualified:Object.values(r.arms).every(a=>a.decision.accepted),allPostProbesQualified:Object.values(r.arms).every(a=>Object.values(a.post).every(p=>p.solve.converged)),frozenControlsExact:true,
  bananaRewardSelectivityChange:r.arms.reward_banana.metrics.nearApproachGap===null||r.before.nearApproachGap===null?null:r.arms.reward_banana.metrics.nearApproachGap-r.before.nearApproachGap,
  breadRewardSelectivityChange:r.arms.reward_bread.metrics.nearApproachGap===null||r.before.nearApproachGap===null?null:r.arms.reward_bread.metrics.nearApproachGap-r.before.nearApproachGap};r.completed=true;
}catch(e){r.error=String(e);r.stack=e.stack;}
r.seconds=(performance.now()-start)/1000;r.sourcesUnchanged=JSON.stringify(sources())===JSON.stringify(r.sources);r.completed&&=r.sourcesUnchanged;r.finished=new Date().toISOString();r.journal_sha256=hash(readFileSync(journal));r.body_sha256=hash(JSON.stringify(r));writeFileSync(destination,JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify({completed:r.completed,error:r.error,seconds:r.seconds,findings:r.findings,receipt:fileURLToPath(destination)},null,2));if(!r.completed)process.exitCode=1;
