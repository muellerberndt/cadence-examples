"""Public demo contracts: numerical parity, causal observation, and evidence failures."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark import hashes
from build_composites import sources
from embodied import motor_settling
from model import OnlineMLP, fast_memory, key_bank, reference, samples, worm_brain
from verify import verify


def node(code):
    return json.loads(
        subprocess.check_output(
            ["node", "--input-type=module", "-e", code], cwd=ROOT, text=True
        )
    )


def test_worm_cadence_matches_independent_dynamics_under_lesions():
    d, m = samples(911, 8, lesions=64)
    actual = (
        worm_brain().settle_batch(d, mask=m, steps=200, tolerance=1e-10).activation
    )
    np.testing.assert_allclose(actual, reference(d, m), atol=1e-9, rtol=0)


def test_browser_kernels_match_current_cadence_and_online_mlp():
    result = node("""
import {FastMemory,MLP,keys,Worm} from './shared/engine.js';
import {motorSettling} from './shared/embodied.js';
import fs from 'node:fs';
const ev=JSON.parse(fs.readFileSync('evidence/evidence.json')),data=JSON.parse(fs.readFileSync('worm/worm.json'));
const f=new FastMemory(),net=new MLP(ev.browser.online_mlp),bank=keys(.5);
for(let k=0;k<20;k++){const key=bank[k%8],v=Array(4).fill(0);v[(k+2)%4]=1;f.observe(key,v);net.observe(key,v,10);}
const w=new Worm(data),d=Array(w.n).fill(0),m=Array(w.n).fill(1);d[23]=1;d[71]=.7;m[16]=0;
console.log(JSON.stringify({fast:f.w,mlp:bank.map(k=>net.predict(k)),worm:w.settle(d,m).state,drive:d,mask:m,motor:motorSettling([-1.8,1.1],[.45,.35]).state}));
""")
    f = fast_memory()
    net = OnlineMLP()
    bank = key_bank(0.5)
    for k in range(20):
        v = np.eye(4)[(k + 2) % 4]
        f.observe(bank[k % 8 : k % 8 + 1], v[None])
        net.observe(bank[k % 8], v, 10)
    np.testing.assert_allclose(result["fast"], f.strength[0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        result["mlp"], [net.predict(k) for k in bank], atol=1e-12, rtol=0
    )
    actual = (
        worm_brain()
        .settle(result["drive"], mask=result["mask"], steps=200, tolerance=1e-10)
        .activation
    )
    np.testing.assert_allclose(result["worm"], actual, atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        result["motor"], motor_settling([-1.8, 1.1], [0.45, 0.35]), atol=1e-12, rtol=0
    )


def test_forager_cannot_read_nectar_before_contact():
    result = node("""
import {Forager,FastMemory,flowers} from './shared/engine.js';
const a=new Forager(new FastMemory()),b=new Forager(new FastMemory()),f=flowers(),g=flowers();g.forEach(x=>x.value=3-x.value);
for(let k=0;k<10;k++){a.step(f);b.step(g);}
console.log(JSON.stringify({a:[a.x,a.y,a.target,a.encounters],b:[b.x,b.y,b.target,b.encounters]}));
""")
    assert result["a"] == result["b"] and result["a"][-1] == 0


def test_new_task_revises_without_erasing_old_tasks_and_restores():
    result = node("""
import {TaskLessons} from './shared/embodied.js';
const m=new TaskLessons(),before=m.recall(3);m.teach(3,3);const after=m.recall(3);m.teach(3,2);const restored=new TaskLessons(m.records);
console.log(JSON.stringify({before,after,answers:[0,1,2,3].map(i=>restored.recall(i))}));
""")
    assert result == {"before": None, "after": 3, "answers": [0, 1, 2, 2]}


def test_mouse_stops_when_goal_is_disconnected():
    result = node("""
import {Mouse,neighbors} from './shared/embodied.js';
const m=new Mouse();neighbors(m.world,m.world.goal).forEach(i=>m.world.grid[i]=1);m.repair();for(let k=0;k<500;k++)m.step();
console.log(JSON.stringify({moves:m.moves,next:m.next()}));
""")
    assert result["next"] == -1


def test_receipts_are_complete_and_source_bound():
    verify(json.loads((ROOT / "evidence/evidence.json").read_text()), hashes())
    verify(
        json.loads((ROOT / "evidence/composite_evidence.json").read_text()),
        sources(),
        True,
    )


@pytest.mark.parametrize("mutation", ["source", "accuracy", "control"])
def test_receipt_gate_rejects_forged_results_even_with_recomputed_digest(mutation):
    body = json.loads((ROOT / "evidence/evidence.json").read_text())
    if mutation == "source":
        body["sources"].pop(next(iter(body["sources"])))
    if mutation == "accuracy":
        body["memory"][0]["arms"]["cadence"]["accuracy"] = 0.5
    if mutation == "control":
        body["worm"][0]["rows"][0]["arms"].pop("unrolled32")
    body.pop("digest")
    body["digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError):
        verify(body, hashes())


def test_habitat_circuit_matches_cadence():
    result = node("""
import {WormArena} from './worm/worm_arena.js';
import {readFileSync} from 'node:fs';
const w=new WormArena(JSON.parse(readFileSync('./worm/worm.json')));
w.step();
console.log(JSON.stringify({state:w.state,drive:w.drive,mask:w.mask,moves:w.moves}));
""")
    actual = (
        worm_brain()
        .settle(result["drive"], mask=result["mask"], steps=200, tolerance=1e-10)
        .activation
    )
    np.testing.assert_allclose(result["state"], actual, atol=1e-9, rtol=0)
    assert result["moves"] == 1


def test_diagnostic_replays_match_executed_settling_and_release_decays():
    result = node("""
import {readFileSync} from 'node:fs';
import {Worm,zeros} from './shared/engine.js';
import {Mouse,motorSettling} from './shared/embodied.js';
import {settlingTrace} from './shared/telemetry.js';
const data=JSON.parse(readFileSync('./worm/worm.json')),w=new Worm(data),drive=zeros(w.n),mask=Array(w.n).fill(1);
data.stimuli.odor.forEach(i=>drive[i]=1);
const r=w.settle(drive,mask),m=new Mouse(),a=motorSettling([-1.8,1.1],[.45,.35]);
const sources=[{...r,drive,mask,edges:data.edges},{...m.circuit,edges:m.circuit.brain.data.edges},{...a,steps:80,dt:.25}];
const errors=sources.map(s=>Math.max(...settlingTrace(s).frames.at(-1).map((v,i)=>Math.abs(v-s.state[i]))));
const release=settlingTrace(sources[0],true);
console.log(JSON.stringify({errors,initial:Math.max(...release.frames[0]),released:Math.max(...release.frames.at(-1))}));
""")
    assert max(result["errors"]) < 1e-12
    assert result["initial"] > 0
    assert result["released"] < result["initial"] * 1e-6


def test_motor_controller_matches_cadence_from_retained_state_and_lesions():
    import cadence as cd

    samples = node("""
import {DrawingArm} from './eye-arm/brain.js';
import {imageFixture} from './eye-arm/fixtures.js';
const a=new DrawingArm(imageFixture('square')),out=[];
for(let i=0;i<8;i++)a.step();out.push(a.brain.snapshot());
a.brain.motor.mask[11]=0;a.step();out.push(a.brain.snapshot());
console.log(JSON.stringify(out));
""")
    for s in samples:
        pre, post, weights = zip(*s["edges"])
        brain = cd.Brain(
            cd.Connectome.from_synapses(len(s["state"]), pre=pre, post=post, sign=weights),
            cd.NeuronModel(
                gain=1, slope=2, threshold=0, leak=1, dt=s["dt"], stimulus_amplitude=1
            ),
        )
        state = cd.BrainState(
            v=np.asarray(s["initialPotential"]),
            activation=np.asarray(s["initialState"]),
            adaptation=np.zeros(len(s["state"])),
            steps=0,
        )
        actual = brain.settle(
            s["drive"],
            state=state,
            mask=np.asarray(s["mask"]),
            steps=s["steps"],
            tolerance=0,
        ).activation
        np.testing.assert_allclose(actual, s["state"], atol=1e-12, rtol=0)


def test_composite_replay_respects_retained_state_and_motor_decay():
    result = node("""
import {DrawingArm} from './eye-arm/brain.js';
import {imageFixture} from './eye-arm/fixtures.js';
import {settlingTrace} from './shared/telemetry.js';
const a=new DrawingArm(imageFixture('square'));for(let i=0;i<50;i++)a.step();
const s=a.brain.snapshot(),trace=settlingTrace(s),release=settlingTrace(s,true);
console.log(JSON.stringify({initial:s.initialState.some(v=>v!==0),error:Math.max(...trace.frames.at(-1).map((v,i)=>Math.abs(v-s.state[i]))),start:Math.max(...release.frames[0].map(Math.abs)),end:Math.max(...release.frames.at(-1).map(Math.abs))}));
""")
    assert result["initial"] and result["error"] < 1e-12
    assert result["end"] < result["start"] * 1e-3


def test_current_nervous_system_receipts_bind_all_controller_sources():
    for folder in ["eye-arm", "mouse", "worm", "fly"]:
        body = json.loads((ROOT / folder / "evidence.json").read_text())
        for filename, digest in body["sources"].items():
            assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest
        assert len(body["arm"]) == 18 and len(body["mouse"]) == 36
        assert len(body["worm"]) == 5 and len(body["fly"]) == 2


def test_game_value_and_self_monitor_match_python_cadence():
    import cadence as cd
    from cadence.circuits import ActivityMonitor

    result = node("""
import {drop,valueCircuit,Monitor} from './connect-four/brain.js';
let b=Array(42).fill(0);[3,2,3,4,2].forEach((c,i)=>b=drop(b,c,i%2?-1:1));
const monitor=new Monitor(),reads=[];
for(const [activity,scores,pressure] of [[[0,0],[.1,.1],0],[[.8,-.4],[-1,1],.2],[[0,0],[.1,.1],1]]) {
  reads.push({...monitor.read(activity,scores,pressure),state:monitor.state.slice(),activity,scores});
}
console.log(JSON.stringify({value:valueCircuit(b,-1),reads}));
""")
    value = result["value"]
    pre, post, weights = zip(*value["edges"])
    brain = cd.Brain(
        cd.Connectome.from_synapses(6, pre=pre, post=post, sign=weights),
        cd.NeuronModel(gain=1, slope=2, threshold=0, leak=1, dt=1, stimulus_amplitude=1),
    )
    actual = brain.settle(value["drive"], steps=2, tolerance=0)
    np.testing.assert_allclose(actual.activation, value["state"], atol=1e-12, rtol=0)
    monitor = ActivityMonitor()
    for row in result["reads"]:
        read = monitor.read(row["activity"], row["scores"], pressure=row["pressure"])
        np.testing.assert_allclose(
            read.state.activation, row["state"], atol=1e-12, rtol=0
        )
        assert read.request_more == row["request_more"]


def test_game_search_preserves_board_blocks_threats_and_respects_limits():
    result = node("""
import assert from 'node:assert/strict';
import {drop,reason,legal,winner} from './connect-four/brain.js';
let b=Array(42).fill(0);[0,6,1,6,2].forEach((c,i)=>b=drop(b,c,i%2?-1:1));
const before=b.slice(),block=reason(b,-1,{depth:3});assert.deepEqual(b,before);
assert.equal(block.column,3);
const victory=drop(b,3,1);assert.equal(winner(victory),1);assert.equal(reason(victory,-1).column,null);
const low=reason(b,-1,{maxNodes:1});assert.equal(low.depth,0);assert.ok(low.budgetExhausted);assert.equal(low.nodes,1);assert.ok(legal(b).includes(low.column));
const completed=reason(Array(42).fill(0),1,{maxNodes:20});assert.equal(completed.depth,1);assert.ok(completed.budgetExhausted);assert.equal(completed.nodes,20);
assert.throws(()=>drop(b,7,1));assert.throws(()=>reason(b,-1,{depth:0}));
console.log(JSON.stringify({block:block.column,depth:completed.depth}));
""")
    assert result == {"block": 3, "depth": 1}


def test_strategy_and_runtime_receipts_bind_their_producers():
    for folder in ["connect-four", "memory"]:
        body = json.loads((ROOT / folder / "evidence.json").read_text())
        for filename, digest in body["sources"].items():
            assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest


def test_history_required_comparison_replays_and_has_a_tight_stateless_bound():
    from collections import Counter, defaultdict

    receipt = json.loads((ROOT / "memory/history_evidence.json").read_text())
    actual = node("""
import {runHistoryBenchmark} from './memory/history_benchmark.mjs';
console.log(JSON.stringify(runHistoryBenchmark()));
""")
    assert actual == receipt
    assert len(receipt["rows"]) == receipt["queries"] == 128
    by_input = defaultdict(Counter)
    predictions = defaultdict(set)
    reference = fast_memory()
    for round in range(16):
        rows = receipt["rows"][round * 8 : (round + 1) * 8]
        for _, key, target, *_ in rows:
            assert target == (round + key) % 4
            reference.observe(
                np.array([receipt["cues"][key]]), np.eye(4)[target : target + 1]
            )
        for r, key, target, predicted, frozen, lookup in rows:
            assert r == round
            cue = tuple(receipt["cues"][key])
            by_input[cue][target] += 1
            predictions[cue].add(frozen)
            assert predicted == lookup == target
            assert np.argmax(reference.recall(np.array([cue]))[0]) == predicted
    # Hindsight's best fixed answer for each identical input is an upper bound
    # on every frozen deterministic query-only network, whatever its size/training.
    bound = sum(max(counts.values()) for counts in by_input.values()) / 128
    assert bound == receipt["scores"]["best_fixed_query_only"] == 0.25
    assert all(len(answers) == 1 for answers in predictions.values())
    for filename, digest in receipt["sources"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest


def test_motor_gate_holds_real_action_and_cancels_stale_world():
    result = node("""
import {DrawingArm} from './eye-arm/brain.js';
import {imageFixture} from './eye-arm/fixtures.js';
import {MotorGate} from './shared/motor_gate.js';
const arm=new DrawingArm(imageFixture('square')),gate=new MotorGate();
gate.enabled=true;const before=arm.q.slice();
gate.prepare(()=>arm.step(true),()=>24);
const held=JSON.stringify(before)===JSON.stringify(arm.q);
gate.advance(23);const still=JSON.stringify(before)===JSON.stringify(arm.q);
gate.advance(1);const moved=JSON.stringify(before)!==JSON.stringify(arm.q);
const next=arm.q.slice();gate.prepare(()=>arm.step(true),()=>24);gate.cancel();gate.advance(100);
console.log(JSON.stringify({held,still,moved,cancelled:JSON.stringify(next)===JSON.stringify(arm.q)}));
""")
    assert all(result.values())


def test_population_traces_measure_signed_oscillation_and_equation_error():
    result = node("""
import {settlingTrace} from './shared/telemetry.js';
const source={state:[0,0],initialState:[.3,0],initialPotential:[Math.atanh(.3),0],drive:[0,0],edges:[[0,1,.8],[1,0,-.8]],steps:40,groups:['a','b']};
const trace=settlingTrace(source);
const independently=trace.frames.map((s,t)=>[ -.8*s[1]-trace.potentials[t][0], .8*s[0]-trace.potentials[t][1]]);
console.log(JSON.stringify({error:Math.max(...trace.mismatches.flatMap((r,t)=>r.map((v,i)=>Math.abs(v-independently[t][i])))),positive:trace.populations.a.some(v=>v.mean>0),negative:trace.populations.a.some(v=>v.mean<0),decay:trace.populations.a.at(-1).rms<trace.populations.a[0].rms}));
""")
    assert result['error'] < 1e-12
    assert result['positive'] and result['negative'] and result['decay']
