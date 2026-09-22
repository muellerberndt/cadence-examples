"""Sweep 3: does sleep pay? python3 receipts/sweep3-2026-09-22/summarize_sleep.py [folder]

Per world: the share of sleepers among the living at four times, the pooled median lifetime of
sleepers against non-sleepers that learn by day, the model's late-life error by sleep, and how
many dreams and ticks asleep a sleeper had at death, over the last 4,000 ticks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(file: Path) -> list[dict]:
    return [json.loads(line) for line in file.read_text().splitlines() if line.strip().startswith("{")]


def fmt(v, digits=0):
    if v is None:
        return "-"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def main(folder: str) -> None:
    files = sorted(Path(folder).glob("*.jsonl"))
    print(f"{'world':16} {'n':>4} {'sleep% @2k':>10} {'@5k':>5} {'@10k':>5} {'@end':>5} {'life sleep':>10} {'life awake':>10} {'err sleep':>9} {'err awake':>9} {'dreams':>7} {'slept':>6} {'sec':>5}")
    for f in files:
        ticks = [r for r in load(f) if r["kind"] == "tick"]
        if not ticks:
            print(f"{f.stem:16} (no ticks)")
            continue
        last = ticks[-1]
        at = {}
        for t in (2000, 5000, 10000, last["t"]):
            row = min(ticks, key=lambda r: abs(r["t"] - t))
            at[t] = 100 * row["alive"].get("sleepers", 0) / max(1, row["n"])
        window = [r for r in ticks if r["t"] > last["t"] - 4000]

        def pooled(key):
            pairs = [(r["lifetime"][key]["median"], r["lifetime"][key]["n"]) for r in window if r["lifetime"].get(key) and r["lifetime"][key]["median"] is not None]
            return sum(m * n for m, n in pairs) / max(1, sum(n for _, n in pairs)) if pairs else None

        def pooled_err(key):
            vals = [r["model"].get(key) for r in window if r["model"].get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        print(
            f"{f.stem:16} {last['n']:4d} {fmt(at[2000]):>10} {fmt(at[5000]):>5} {fmt(at[10000]):>5} {fmt(at[last['t']]):>5} "
            f"{fmt(pooled('sleep')):>10} {fmt(pooled('noSleep')):>10} {fmt(pooled_err('errLateSleep'), 4):>9} {fmt(pooled_err('errLateNoSleep'), 4):>9} "
            f"{fmt(pooled_err('dreamsMean')):>7} {fmt(pooled_err('sleptMean')):>6} {fmt(last['seconds']):>5}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "receipts/sweep3-2026-09-22")
