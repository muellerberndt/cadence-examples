"""Tabulate a folder of probe logs: python3 sim/summarize.py receipts/<folder> [> SUMMARY.txt]

For every *.jsonl in the folder: the population at the end, mean nodes and speed, the share of the population with
records, planning and slow learning, lifetime at death by trait over the last 2,000 ticks, and the model's error
late in life with records against without."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(file: Path) -> list[dict]:
    return [json.loads(line) for line in file.read_text().splitlines() if line.strip()]


def fmt(v, digits=1):
    if v is None:
        return "-"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def main(folder: str) -> None:
    files = sorted(Path(folder).glob("*.jsonl"))
    print(f"{'world':34} {'n':>5} {'nodes':>5} {'speed':>6} {'rec%':>5} {'plan%':>5} {'slow%':>5} {'life':>6} {'rec':>6} {'norec':>6} {'plan':>6} {'noplan':>6} {'errRec':>7} {'errNo':>7} {'plans':>7} {'conf%':>5} {'s':>6}")
    for f in files:
        rows = load(f)
        ticks = [r for r in rows if r["kind"] == "tick"]
        if not ticks:
            print(f"{f.stem:34} (no ticks)")
            continue
        last = ticks[-1]
        window = [r for r in ticks if r["t"] > last["t"] - 2000]

        def pooled(key: str, field: str = "median"):
            ns = [r["lifetime"][key]["n"] for r in window]
            vals = [r["lifetime"][key][field] for r in window if r["lifetime"][key][field] is not None]
            if not vals:
                return None
            return sum(v * n for v, n in zip(vals, [r["lifetime"][key]["n"] for r in window if r["lifetime"][key][field] is not None])) / max(1, sum(ns))

        def pooled_err(key: str):
            vals = [r["model"][key] for r in window if r["model"][key] is not None]
            return sum(vals) / len(vals) if vals else None

        n = last["n"]
        a = last["alive"]
        plans = sum(r["plans"]["total"] for r in window)
        acc = sum(r["plans"].get("confirmed", 0) for r in window)
        print(
            f"{f.stem:34} {n:5d} {fmt(a['nodesMean'])!s:>5} {fmt(a['speedMean'], 3)!s:>6} "
            f"{fmt(100 * a['records'] / max(1, n), 0)!s:>5} {fmt(100 * a['plan'] / max(1, n), 0)!s:>5} {fmt(100 * a['slow'] / max(1, n), 0)!s:>5} "
            f"{fmt(pooled('all'), 0)!s:>6} {fmt(pooled('records'), 0)!s:>6} {fmt(pooled('noRecords'), 0)!s:>6} {fmt(pooled('plan'), 0)!s:>6} {fmt(pooled('noPlan'), 0)!s:>6} "
            f"{fmt(pooled_err('errLateRecords'), 4)!s:>7} {fmt(pooled_err('errLateNoRecords'), 4)!s:>7} {plans:7d} {fmt(100 * acc / max(1, plans), 0)!s:>5} {fmt(last['seconds'], 0)!s:>6}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "receipts")
