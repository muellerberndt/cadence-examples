"""Live memories must match the audited core, including what survives reset."""
import json
import subprocess
from pathlib import Path

import cadence as cd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def node(code):
    return json.loads(subprocess.check_output(
        ['node', '--input-type=module', '-e', code], cwd=ROOT, text=True))


def test_browser_consolidation_matches_core_under_interference_and_partial_feedback():
    rows = node('''
import {SynapticMemory, keys} from './shared/engine.js';
const m = new SynapticMemory(), bank = keys(.5), rows = [];
for (let i = 0; i < 24; i++) {
  const key = bank[i % 8], value = Array.from({length:4}, (_, j) => +(j === i % 4));
  const valueMask = Array.from({length:4}, (_, j) => i % 3 !== 0 || j === i % 4);
  const salience = i % 5;
  m.observe(key, value, 1, {salience, valueMask});
  if (i === 12) m.reset();
  rows.push({key, value, salience, valueMask, w:m.w, c:m.consolidated});
}
console.log(JSON.stringify(rows));
''')
    m = cd.SynapticMemory(np.arange(8), np.arange(8, 12))
    for i, row in enumerate(rows):
        m.observe(np.array([row['key']]), np.array([row['value']]),
                  salience=np.array([row['salience']]), value_mask=np.array([row['valueMask']]))
        if i == 12:
            m.reset(1)
        np.testing.assert_allclose(row['w'], m.strength[0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(row['c'], m.consolidated, atol=1e-12, rtol=0)


def test_mouse_checkpoint_preserves_strengths_without_replaying_or_forgetting_repetition():
    result = node('''
import {TaskLessons} from './shared/embodied.js';
const a = new TaskLessons();
for (let i = 0; i < 40; i++) a.teach(3, 3);
a.teach(3, 2);
const b = new TaskLessons(JSON.parse(JSON.stringify(a)));
console.log(JSON.stringify({same:JSON.stringify(a)===JSON.stringify(b),
  recalled:b.recall(3), strength:b.memory.consolidated[3][3], revised:b.memory.w[3][2]}));
''')
    assert result['same'] and result['recalled'] == 2
    assert result['strength'] > .8 and result['revised'] == 1


def test_repeat_salience_and_distraction_have_measured_persistent_effects():
    result = node('''
import {run} from './memory/consolidation_benchmark.mjs';
console.log(JSON.stringify(run()));
''')
    assert result['ordinary'] < .1
    assert result['repeated'] > .85
    assert result['salient'] == result['after_distraction'] == 1
    assert result['revised'] > .95
    assert result['consolidation_disabled'] == 0


def test_retention_receipt_replays_exactly_and_binds_sources():
    receipt = json.loads((ROOT / 'memory/consolidation_evidence.json').read_text())
    actual = node("import {report} from './memory/consolidation_benchmark.mjs'; console.log(JSON.stringify(report()));")
    assert actual == receipt
