"""Render the runtime comparison from its recorded observations."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "memory/evidence.json").read_text())
base = r["rows"][0]["median_ms"]
labels = {"cadence":"Cadence fast reference", "consolidating":"Cadence consolidating memory",
          "1":"MLP, 1 update", "10":"MLP, 10 updates", "100":"MLP, 100 updates"}
table = "| Learner | Correct queries | Median stream time | Time / fast reference |\n|---|---:|---:|---:|\n"
for row in r["rows"]:
    table += (f"| {labels[row['kind']]} | {row['correct']}/{row['queries']} ({100*row['accuracy']:.1f}%) "
              f"| {row['median_ms']:.3f} ms | {row['median_ms']/base:.1f}× |\n")
p = ROOT / "ADVANTAGES.md"
s = p.read_text()
a, b = s.index("| Learner | Correct queries"), s.index("\nSub-millisecond")
p.write_text(s[:a] + table + s[b:])
