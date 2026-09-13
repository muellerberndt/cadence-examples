"""Regenerate the ladder table in README.md and the rung cards in the hub pages from the receipts.

Run:  python tools/ladder.py            (from the repo root; rewrites README.md, index.html, hub_published.html)

Each rung declares how to summarise its receipt; a rung without a receipt is listed as pending.
The markers <!-- ladder --> ... <!-- /ladder --> in README.md and <!-- rungs --> ... <!-- /rungs -->
in the hub pages bound the generated regions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = {  # the published copies of each rung's index.html
    "01_digits": "https://claude.ai/code/artifact/0f6136f7-79ca-4b0e-9cef-65fd70fc6618",
    "02_recall": "https://claude.ai/code/artifact/ff0e3f63-b674-494e-8c0f-a99d845d678b",
    "03_connect_four": "https://claude.ai/code/artifact/a75ef805-c396-4c9d-b64d-6ca5fe60badc",
    "04_pong": "https://claude.ai/code/artifact/4b3fe725-e687-4acb-a29c-5f2eb9a69e04",
}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def summarise(rung: str, b: dict) -> tuple[str, list[tuple[str, str]]]:
    """(one-line result for the README, [(label, value), ...] facts for the hub card)."""
    if rung == "01_digits":
        m = [c for c in b["comparison"] if "64-32-10" in c["model"] and c["epochs"] >= 50][0]
        return (f"{b['test_accuracy_mean']:.3f} ± {b['test_accuracy_std']:.3f} held-out in {b['schedule']['epochs']} epochs; a smaller MLP ({m['parameters']:,} parameters) {m['test_accuracy']:.3f} in {m['epochs']}",
                [("held-out", f"{b['test_accuracy_mean']:.3f} ± {b['test_accuracy_std']:.3f}"), ("parameters", f"{b['learner']['parameters']:,}"), ("epochs", str(b["schedule"]["epochs"])), ("smaller MLP", f"{m['test_accuracy']:.3f}")])
    if rung == "02_recall":
        summary = b["summary"]
        length = str(max(b["task"]["lengths"]))
        return (f"historical distinct-key recall {summary['patch_settled'][length]:.2f}; dictionary also solves this lookup task; original transformer comparison has parser and position disadvantages",
                [("historical distinct-key recall", f"{summary['patch_settled'][length]:.2f}"), ("capacity", "128 keys"), ("trained parameters", "0"), ("comparison", "see tutorial limits")])
    if rung == "05_memory":
        rows = [row for run in b["runs"] for row in run["rows"] if row["length"] == 128 and row["key_correlation"] == 0.0]
        means = {arm: sum(row["arms"][arm]["accuracy"] for row in rows) / len(rows) for arm in ("delta", "hebb", "dictionary", "transformer")}
        return (f"128 writes, orthogonal keys: residual {means['delta']:.3f}, additive {means['hebb']:.3f}, dictionary {means['dictionary']:.3f}, trained transformer {means['transformer']:.3f}; correlated-key results in the receipt",
                [("residual memory", f"{means['delta']:.3f}"), ("additive memory", f"{means['hebb']:.3f}"), ("exact dictionary", f"{means['dictionary']:.3f}"), ("trained transformer", f"{means['transformer']:.3f}")])
    if rung == "06_interventions":
        rows = [row for run in b["runs"] for row in run["rows"]]
        residual = max(row["arms"]["cadence"]["max_residual"] for row in rows)
        mlp_mse = sum(row["arms"]["mlp"]["mse"] for row in rows) / len(rows)
        return (f"declared feedback circuits answer new interventions without training; maximum residual {residual:.1e}, trained MLP mean MSE {mlp_mse:.4f}; direct and tied-recurrence controls also solve the task",
                [("owners", "16"), ("Cadence training examples", "0"), ("maximum residual", f"{residual:.1e}"), ("trained MLP mean MSE", f"{mlp_mse:.4f}")])
    if rung == "03_connect_four":
        s = {r["opponent"]: r for r in b["strength"]}
        m = b["comparison"][0]
        rec = lambda r: f"{r['wins']}-{r['draws']}-{r['losses']}"  # noqa: E731
        return (f"agrees with a depth-4 search on {b['test_agreement']:.3f} of positions (MLP {m['agreement']:.3f}); {rec(s['random'])} vs random, {rec(s['search-2'])} vs depth 2",
                [("agreement with the search", f"{b['test_agreement']:.3f}"), ("MLP same size", f"{m['agreement']:.3f}"), ("vs random", rec(s["random"])), ("vs depth 2", rec(s["search-2"]))])
    if rung == "04_pong":
        base = b["comparison"][0]["final"]
        im = b.get("imitation", {}).get("final")
        extra = f"; the same net taught the tracker's moves {pct(im['return_rate'])}" if im else ""
        return (f"reward-trained paddle returns {pct(b['final']['return_rate'])} of balls against {pct(base['return_rate'])} for backprop REINFORCE on the same rollout budget{extra}",
                [("balls returned, from reward", pct(b["final"]["return_rate"])), ("backprop REINFORCE, same budget", pct(base["return_rate"])), ("from a tracker's moves", pct(im["return_rate"]) if im else "—"), ("parameters", f"{b['learner']['parameters']:,}")])
    raise KeyError(rung)


RUNGS = [
    ("01_digits", "digits", "the tutorial, draw one", "8×8 scikit-learn digits; 64 input owners, a hidden layer, 10 output owners. Historical classification measurement: selection on validation data and three patch-net seeds, with fixed MLP baselines. Draw a digit in the page and watch the ten output owners settle."),
    ("02_recall", "recall", "memory as seams, play it", "Distinct one-hot keys written into 128-key Hebbian memory. Linear transport retrieves an association; a dictionary solves the same task. The historical transformer comparison has important limitations. Write a context in the page and ask it."),
    ("03_connect_four", "Connect Four", "imitation, play it", "Self-play positions labelled by a depth-4 search; the net imitates the search and plays with no lookahead. The historical agreement measurement predates a reflection-split fix. You play coral, the net plays gold."),
    ("04_pong", "Pong", "reward, play it", "A paddle that learned from pixels and reward: the nudge's target is the action taken, its strength the action's advantage, each seam's step read from its own history. Behind backprop REINFORCE with Adam on the same rollout budget; the same net taught a tracker's moves returns 96%. You play coral, the net plays gold."),
    ("05_memory", "updating memory", "residual writes", "A fixed-size memory corrects its own readback when an association changes. Additive memory, exact lookup, and a trained transformer receive identical explicit key/value episodes."),
    ("06_interventions", "circuit interventions", "known local rules", "Owners read incoming messages and repair their local state as drives, wiring and ablations change. A supplied feedback rule reaches checked equilibrium without training. A trained MLP, direct solver and tied recurrence receive the same complete circuit."),
]


def intervention_tables(body: dict) -> str:
    arms = ("cadence", "cadence_warm", "mlp", "one_propagation", "unrolled8", "unrolled32", "newton")
    labels = ("Cadence", "warm", "MLP", "one propagation", "8 steps", "32 steps", "Newton")
    all_rows = [row for run in body["runs"] for row in run["rows"]]
    trials = body["task"]["trials"]
    lines = ["Mean squared error, averaged over the three seeds:", "",
             "| slice | " + " | ".join(labels) + " |", "|" + "---|" * 8]
    for condition in body["task"]["conditions"]:
        rows = [r for r in all_rows if r["condition"] == condition]
        means = [sum(r["arms"][a]["mse"] for r in rows) / len(rows) for a in arms]
        lines.append(f"| {condition} | " + " | ".join(f"{value:.2e}" for value in means) + " |")
    largest = max(r["arms"][a]["max_residual"] for r in all_rows for a in ("cadence", "cadence_warm"))
    bound = max(r["arms"][a]["max_error_bound"] for r in all_rows for a in ("cadence", "cadence_warm"))
    lines += ["", f"Across both Cadence paths, maximum independent residual **{largest:.2e}**, "
              f"giving a maximum state-error bound of **{bound:.2e}**.", "",
              "Mean settling steps:", "", "| slice | cold | warm |", "|---|---|---|"]
    for condition in body["task"]["conditions"]:
        rows = [r for r in all_rows if r["condition"] == condition]
        values = [sum(r["arms"][a]["mean_steps"] for r in rows) / len(rows) for a in ("cadence", "cadence_warm")]
        lines.append(f"| {condition} | {values[0]:.1f} | {values[1]:.1f} |")
    lines += ["", "Mean CPU microseconds per query across all five slices:", "",
              "| arm | sequential | within a batch of 128 |", "|---|---|---|"]
    for arm, label in zip(arms, labels):
        serial = sum(r["arms"][arm]["sequential_seconds"] for r in all_rows) / len(all_rows) / trials * 1e6
        batch = (sum(r["arms"][arm]["batch_seconds"] for r in all_rows) / len(all_rows) / trials * 1e6
                 if "batch_seconds" in all_rows[0]["arms"][arm] else None)
        lines.append(f"| {label} | {serial:.1f} | " + (f"{batch:.1f}" if batch is not None else "not measured") + " |")
    training = sum(run["mlp_training"]["seconds"] for run in body["runs"])
    preparation = sum(run["mlp_training"]["preparation_seconds"] for run in body["runs"])
    binding = sum(r["arms"][a]["binding_seconds"] for r in all_rows for a in ("cadence", "cadence_warm"))
    initialization = sum(r["arms"][a]["initialization_seconds"] for r in all_rows for a in ("cadence", "cadence_warm"))
    lines += ["", f"Across the complete run, both MLP candidates and their checkpoint selection took "
              f"{training:.2f} s, plus {preparation:.2f} s to prepare training/validation inputs and labels. "
              f"All Cadence cold/warm graph bindings took {binding:.3f} s, plus {initialization:.3f} s "
              "to initialize kernels. These costs are additional to the query timings."]
    return "\n".join(lines)


def main() -> None:
    readme_rows, cards = [], []
    for rung, name, tag, blurb in RUNGS:
        k = int(rung[:2])
        receipt = ROOT / rung / "receipt.json"
        page = PAGES.get(rung)
        if receipt.exists():
            body = json.loads(receipt.read_text())["body"]
            line, facts = summarise(rung, body)
            status = "done"
            if rung == "06_interventions":
                path = ROOT / rung / "README.md"
                text = re.sub(r"<!-- intervention-results -->.*?<!-- /intervention-results -->",
                              "<!-- intervention-results -->\n" + intervention_tables(body)
                              + "\n<!-- /intervention-results -->", path.read_text(), flags=re.S)
                path.write_text(text)
            if rung == "05_memory":
                rows = ["| writes | key cosine | residual | additive | transformer | exact lookup |",
                        "|---|---|---|---|---|---|"]
                for correlation in body["task"]["correlations"]:
                    for length in body["task"]["lengths"]:
                        readings = [row for run in body["runs"] for row in run["rows"]
                                    if row["mode"] == "random_values" and row["length"] == length and row["key_correlation"] == correlation]
                        arms = ("delta", "hebb", "transformer", "dictionary")
                        means = [sum(row["arms"][arm]["accuracy"] for row in readings) / len(readings) for arm in arms]
                        rows.append(f"| {length} | {correlation:g} | " + " | ".join(f"{m:.3f}" for m in means) + " |")
                for correlation in (0.5, 0.9):
                    balanced = [row for run in body["runs"] for row in run["rows"]
                                if row["mode"] == "balanced_first_pass" and row["key_correlation"] == correlation]
                    means = [sum(row["arms"][arm]["accuracy"] for row in balanced) / len(balanced) for arm in arms]
                    rows.append(f"| 8, balanced values | {correlation:g} | " + " | ".join(f"{m:.3f}" for m in means) + " |")
                path = ROOT / rung / "README.md"
                prose = re.sub(r"<!-- memory-results -->.*?<!-- /memory-results -->",
                               "<!-- memory-results -->\n" + "\n".join(rows) + "\n<!-- /memory-results -->",
                               path.read_text(), flags=re.S)
                path.write_text(prose)
        else:
            line, facts, status = "pending", [("status", "running")], "pending"
        link = f"[{name}]({rung}/)"
        play = f"; [play it]({page})" if page else ""
        lead = blurb.split(".")[0]
        lead = lead[0].lower() + lead[1:]
        readme_rows.append(f"| {k:02d} | {link} | {lead}; receipt: {line}{play} |" if status == "done" else f"| {k:02d} | {link} | {lead}; {line} |")
        facts_html = "".join(f"<span>{label} <b>{value}</b></span>" for label, value in facts)
        actions = (f'<a class="btn play" href="{{PLAY_{rung}}}">Play</a>' if page else "") + f'<a class="btn" href="{{README_{rung}}}">Read the receipt</a>'
        cards.append(f'    <article class="rung{"" if status == "done" else " next"}">\n      <div class="num">{k:02d}</div>\n      <div>\n        <h2>{name[0].upper() + name[1:]} <span class="tag">{tag}</span></h2>\n        <p>{blurb}</p>\n        <div class="facts">{facts_html}</div>\n      </div>\n      <div class="actions">{actions}</div>\n    </article>')
    table = "| # | example | what it shows |\n|---|---|---|\n" + "\n".join(readme_rows)
    readme = ROOT / "README.md"
    text = readme.read_text()
    text = re.sub(r"<!-- ladder -->.*?<!-- /ladder -->", "<!-- ladder -->\n" + table + "\n<!-- /ladder -->", text, flags=re.S)
    readme.write_text(text)
    for page_file, local in (("index.html", True), ("hub_published.html", False)):
        html = "\n".join(cards)
        for rung, _, _, _ in RUNGS:
            html = html.replace(f"{{README_{rung}}}", f"{rung}/README.md" if local else f"https://github.com/muellerberndt/cadence-examples/tree/main/{rung}")
            html = html.replace(f"{{PLAY_{rung}}}", f"{rung}/index.html" if local else (PAGES.get(rung) or f"{rung}/index.html"))
        path = ROOT / page_file
        t = path.read_text()
        t = re.sub(r"<!-- rungs -->.*?<!-- /rungs -->", "<!-- rungs -->\n" + html + "\n<!-- /rungs -->", t, flags=re.S)
        path.write_text(t)
    print("ladder regenerated for", sum(1 for r, *_ in RUNGS if (ROOT / r / "receipt.json").exists()), "receipts")


if __name__ == "__main__":
    main()
