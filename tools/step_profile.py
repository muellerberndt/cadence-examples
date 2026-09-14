"""Fixed-budget GPU work: measure synchronization overhead on identical batches."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cadence as cd
import cadence.settle
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.brain import drives

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--tag", default="profile")
p.add_argument("--checkpoint", default="runs/gpu-256/brain.npz")
a = p.parse_args()
torch.set_num_threads(2)
data = dict(np.load(ROOT / "data/pilot/train.npz"))
ids = np.random.default_rng(42).integers(len(data["labels"]), size=(100, 256))
rows = []
for tolerance in [0, None, 1e-4]:
    b = cd.Learner.load(
        ROOT / a.checkpoint, backend="torch", device="cuda", precision="float32"
    )
    b.config = replace(b.config, tolerance=tolerance)
    batches = [
        (drives(b, data["context"][ix], data["extra"][ix]), data["labels"][ix])
        for ix in ids
    ]
    for x, y in batches[:10]:
        b.step(x, y)
    torch.cuda.synchronize()
    start = time.perf_counter()
    free = []
    nudged = []
    for x, y in batches[10:]:
        _, r = b.step(x, y)
        free.append(r["free_steps"])
        nudged.append(r["nudged_steps"])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    rows.append(
        {
            "tolerance": tolerance,
            "updates": 90,
            "seconds": elapsed,
            "median_free_steps": float(np.median(free)),
            "median_nudged_steps": float(np.median(nudged)),
        }
    )
report = {
    "rows": rows,
    "device": torch.cuda.get_device_name(),
    "source_sha256": hashlib.sha256(
        Path(cadence.settle.__file__).read_bytes()
    ).hexdigest(),
}
(ROOT / "runs" / f"{a.tag}.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
