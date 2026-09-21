"""Why did one lineage win? Read a chronicle and say what it did differently while it was winning.

    python3 sim/why.py <chronicle.json> [--lineage L] [--rivals 3]

A chronicle comes from the page's Export button or from `sim/probe.js ... '{"chronicle":"file.json"}'`.
It holds, every hundred ticks, per lineage: population, births, deaths (starved or killed), kills,
bites given and taken, meals, mean speed, age, energy, genome nodes, muscles, channels, cortices,
the shares with records and planning, plans and confirmed repairs, mean age at death, hue; plus the
events (births, deaths with cause, bites, injuries), genome samples of the leading lineages every
thousand ticks, and the full genomes of the largest living lineages at export.

The report: the winner and when it rose (first 5, 10, 25, 50 per cent of the population); its
per-capita rates against everyone else over the rise; kills between the winner and the rest; how
its genome differs from the founder and from its rivals at the time of the rise; and where it lived.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def load(path):
    with open(path) as f:
        return json.load(f)


def rows(chron):
    keys = chron["meta"]["keys"]
    for fr in chron["frames"]:
        for lid, arr in fr["lineages"].items():
            yield fr["t"], lid, dict(zip(keys, arr))


def series(chron):
    """{lineage: [(t, row)]} and the total population per frame."""
    per = defaultdict(list)
    total = {}
    for t, lid, r in rows(chron):
        per[lid].append((t, r))
        total[t] = total.get(t, 0) + r["n"]
    return per, total


def share_crossings(per, total, lid):
    out = {}
    for t, r in per[lid]:
        if total[t] == 0:
            continue
        s = r["n"] / total[t]
        for level in (0.05, 0.10, 0.25, 0.50, 0.90):
            if level not in out and s >= level:
                out[level] = t
    return out


def window_rates(per, lid, t0, t1, exclude=()):
    """Per-capita rates over [t0, t1] for one lineage, or for everyone else when lid is None."""
    acc = defaultdict(float)
    for L, ser in per.items():
        if lid is not None and L != lid:
            continue
        if lid is None and L in exclude:
            continue
        for t, r in ser:
            if t < t0 or t > t1:
                continue
            n = r["n"]
            acc["being_frames"] += n
            for k in ("births", "deaths", "starved", "killed", "kills", "bitesGiven", "bitesTaken", "meals", "plans", "confirmed"):
                acc[k] += r[k]
            for k in ("speed", "energy", "age", "nodes", "muscles", "channels", "cortices", "records", "planning"):
                acc["w_" + k] += r[k] * n
            if r["ageAtDeath"] is not None:
                acc["ageAtDeath_sum"] += r["ageAtDeath"] * r["deaths"]
                acc["ageAtDeath_n"] += r["deaths"]
    bf = max(acc["being_frames"], 1e-9)
    out = {k: acc[k] / bf for k in ("births", "deaths", "starved", "killed", "kills", "bitesGiven", "bitesTaken", "meals")}
    for k in ("speed", "energy", "age", "nodes", "muscles", "channels", "cortices", "records", "planning"):
        out[k] = acc["w_" + k] / bf
    out["confirmed_share"] = acc["confirmed"] / acc["plans"] if acc["plans"] else None
    out["ageAtDeath"] = acc["ageAtDeath_sum"] / acc["ageAtDeath_n"] if acc["ageAtDeath_n"] else None
    out["being_frames"] = acc["being_frames"]
    return out


def fmt(v, d=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def ratio(a, b):
    if a is None or b is None or b == 0:
        return "-"
    return f"{a / b:.2f}x"


def genome_diff(a, b):
    """Differences between two genome summaries: genes, body, brain."""
    lines = []
    for k in a["genes"]:
        if b and a["genes"][k] != b["genes"].get(k):
            lines.append(f"{k} {b['genes'][k]} -> {a['genes'][k]}")
    if b:
        if a["nodes"] != b["nodes"] or a["muscles"] != b["muscles"]:
            lines.append(f"body {b['nodes']} nodes, {b['muscles']} muscles -> {a['nodes']} nodes, {a['muscles']} muscles")
        for k in ("along", "back", "side"):
            if abs(a["grips"][k] - b["grips"][k]) > 0.05:
                lines.append(f"grip {k} {b['grips'][k]:.2f} -> {a['grips'][k]:.2f}")

        def brain(s):
            return " | ".join(f"{p}: " + " + ".join(f"{c['channels']}ch" for c in s["brain"][p]) for p in ("policy", "model"))

        if brain(a) != brain(b):
            lines.append(f"brain {brain(b)} -> {brain(a)}")
        for p in ("policy", "model"):
            na, nb = a["thetaNorm"][p], b["thetaNorm"][p]
            if nb and abs(na - nb) / nb > 0.15:
                lines.append(f"{p} weights norm {nb:.2f} -> {na:.2f}")
    return lines


def where(chron, lid, t0, t1):
    xs, ys = [], []
    for e in chron["events"]:
        if e["lineage"] == lid and t0 <= e["t"] <= t1 and e["kind"] == "birth":
            xs.append(e["x"]); ys.append(e["y"])
    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys), len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chronicle")
    ap.add_argument("--lineage", type=int, default=None)
    ap.add_argument("--rivals", type=int, default=3)
    args = ap.parse_args()
    chron = load(args.chronicle)
    per, total = series(chron)
    last_t = max(total)
    final = {L: ser[-1][1]["n"] for L, ser in per.items() if ser[-1][0] == last_t}
    winner = str(args.lineage) if args.lineage is not None else max(final, key=final.get)
    cross = share_crossings(per, total, winner)
    peak = max(per[winner], key=lambda x: x[1]["n"])
    print(f"Chronicle to tick {last_t:,}; {len(per)} lineages seen; {total[last_t]} alive at the end.")
    print(f"Lineage {winner} holds {final.get(winner, 0)} of {total[last_t]} ({100 * final.get(winner, 0) / max(1, total[last_t]):.0f}%) at the end; its peak was {peak[1]['n']} at tick {peak[0]:,}.")
    print("It reached " + ", ".join(f"{int(100 * k)}% at tick {v:,}" for k, v in sorted(cross.items())) + ".")
    t0 = cross.get(0.05, per[winner][0][0]); t1 = cross.get(0.50, last_t)
    if t1 <= t0:
        t1 = min(last_t, t0 + 2000)
    print(f"\nThe rise: ticks {t0:,} to {t1:,}. Per being and per hundred ticks, the winner against everyone else alive then:")
    w = window_rates(per, winner, t0, t1); o = window_rates(per, None, t0, t1, exclude=(winner,))
    print(f"{'measure':22} {'winner':>9} {'others':>9} {'ratio':>7}")
    for k, label in [("births", "births"), ("deaths", "deaths"), ("starved", "  starved"), ("killed", "  killed"), ("kills", "kills made"), ("bitesGiven", "bites given"), ("bitesTaken", "bites taken"), ("meals", "meals")]:
        print(f"{label:22} {fmt(w[k], 3):>9} {fmt(o[k], 3):>9} {ratio(w[k], o[k]):>7}")
    print(f"{'--- means':22}")
    for k, label in [("speed", "speed, cells/tick"), ("energy", "energy held"), ("ageAtDeath", "age at death"), ("nodes", "genome nodes"), ("muscles", "muscles"), ("channels", "brain channels"), ("cortices", "cortices"), ("records", "share with records"), ("planning", "share planning"), ("confirmed_share", "repairs confirmed")]:
        print(f"{label:22} {fmt(w[k], 3):>9} {fmt(o[k], 3):>9} {ratio(w[k], o[k]) if k not in ('records', 'planning', 'confirmed_share') else '':>7}")
    # verdict in words
    reasons = []
    if o["births"] and w["births"] / o["births"] > 1.15:
        reasons.append(f"it bred {w['births'] / o['births']:.2f} times as often per being")
    if o["meals"] and w["meals"] / o["meals"] > 1.1:
        reasons.append(f"it ate {w['meals'] / o['meals']:.2f} times as much per being")
    if o["starved"] and w["starved"] / o["starved"] < 0.85:
        reasons.append(f"it starved at {w['starved'] / o['starved']:.2f} of the others' rate")
    if o["speed"] and w["speed"] / o["speed"] > 1.15:
        reasons.append(f"it crawled {w['speed'] / o['speed']:.2f} times as fast")
    if w["kills"] > 0 and (o["kills"] == 0 or w["kills"] / o["kills"] > 1.5):
        reasons.append("it killed more than the others")
    if w["killed"] and o["killed"] and w["killed"] / o["killed"] < 0.7:
        reasons.append("it was killed less")
    print("\nIn words: " + ("; ".join(reasons) if reasons else "no single rate stands out beyond 15 per cent; the sweep looks like drift at equal fitness, or the advantage lies outside these rates") + ".")
    # kills between the winner and the rest
    kills_by = defaultdict(int); killed_by_winner = 0; winner_killed = defaultdict(int)
    for e in chron["events"]:
        if e["kind"] == "die" and e.get("cause") == "killed" and "by" in e:
            if str(e["by"]) == winner and str(e["lineage"]) != winner:
                killed_by_winner += 1
            if str(e["lineage"]) == winner and str(e["by"]) != winner:
                winner_killed[str(e["by"])] += 1
    if killed_by_winner or winner_killed:
        print(f"\nKills in the event log: the winner killed {killed_by_winner} of other lineages; it was killed {sum(winner_killed.values())} times by others" + (", most by lineage " + max(winner_killed, key=winner_killed.get) if winner_killed else "") + ".")
    # genome: the winner's sample nearest the rise against the founder and the rivals
    samples = chron.get("samples", {})
    ws = samples.get(winner) or []
    if ws:
        near = min(ws, key=lambda s: abs(s["t"] - t1))
        print(f"\nThe winner's genome at tick {near['t']:,} (eldest #{near['id']}, age {near['age']}), against the founder:")
        diffs = genome_diff(near, chron.get("founder"))
        print("  " + ("\n  ".join(diffs) if diffs else "no difference in genes, body or brain layout; the weights differ (drift and learning)"))
        # rivals: the largest other lineages during the rise
        rival_size = defaultdict(int)
        for L, ser in per.items():
            if L == winner:
                continue
            for t, r in ser:
                if t0 <= t <= t1:
                    rival_size[L] = max(rival_size[L], r["n"])
        rivals = sorted(rival_size, key=rival_size.get, reverse=True)[: args.rivals]
        for L in rivals:
            rs = samples.get(L) or []
            if not rs:
                continue
            rn = min(rs, key=lambda s: abs(s["t"] - t1))
            rr = window_rates(per, L, t0, t1)
            d = genome_diff(near, rn)
            print(f"\nRival lineage {L} (peak {rival_size[L]} beings in the rise; births {fmt(rr['births'])}, meals {fmt(rr['meals'])}, speed {fmt(rr['speed'])} per being-frame): the winner differs by")
            print("  " + ("\n  ".join(d) if d else "nothing in genes, body or brain layout"))
    loc = where(chron, winner, t0, t1)
    if loc:
        print(f"\nWhere: the winner's {loc[2]} births in the rise centre on x {loc[0]:.0f}, y {loc[1]:.0f} of the 96 by 96 torus.")
    genomes = chron.get("genomes", {})
    if winner in genomes:
        g = genomes[winner]
        print(f"\nThe full genome of the winner's eldest at export (#{g['id']}, age {g['age']}) is in the chronicle under genomes[{winner}].")


if __name__ == "__main__":
    main()
