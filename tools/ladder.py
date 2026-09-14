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
PAGES = {name: f"{name}/index.html" for name in
         ("01_digits", "02_recall", "03_connect_four", "04_pong")}



def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def summarise(rung: str, b: dict) -> tuple[str, list[tuple[str, str]]]:
    """(one-line result for the README, [(label, value), ...] facts for the hub card)."""
    if rung == "01_digits":
        m = [c for c in b["comparison"] if "64-32-10" in c["model"] and c["epochs"] >= 50][0]
        return (f"{b['test_accuracy_mean']:.3f} ± {b['test_accuracy_std']:.3f} held-out in {b['schedule']['epochs']} epochs; a smaller MLP ({m['parameters']:,} parameters) {m['test_accuracy']:.3f} in {m['epochs']}",
                [("held-out", f"{b['test_accuracy_mean']:.3f} ± {b['test_accuracy_std']:.3f}"), ("parameters", f"{b['learner']['parameters']:,}"), ("epochs", str(b["schedule"]["epochs"])), ("smaller MLP", f"{m['test_accuracy']:.3f}")])
    if rung == "02_recall":
        rows = b["rows"]
        revised = sum(r["correct"]["revised"] for r in rows) / sum(r["queries"] for r in rows)
        return (f"residual writes: {pct(revised)} correct after replacing repeatedly written associations; exact dictionary also succeeds",
                [("corrected associations", pct(revised)), ("capacity", "128 keys"), ("offline training", "none"), ("memory", "residual fast seams")])
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
        raw = {r["opponent"]: r for r in b["strength"]}
        deployed = {r["opponent"]: r for r in b["deployed_strength"]}
        control = {r["opponent"]: r for r in b["search_only_strength"]}
        win = lambda r: pct(r["wins"] / r["games"])
        return (f"with depth-4 lookahead: {win(deployed['random'])} wins vs random, {win(deployed['search-2'])} vs depth 2; search alone {win(control['search-2'])}; raw policy {win(raw['search-2'])}",
                [("lookahead vs depth 2", win(deployed["search-2"])), ("search alone vs depth 2", win(control["search-2"])), ("raw policy vs depth 2", win(raw["search-2"])), ("held-out teacher agreement", f"{b['test_agreement']:.3f}")])
    if rung == "04_pong":
        f = b["final"]
        return (f"one frame plus a fading neural trace; imitation then practice: {pct(f['win_rate'])} points won vs skill-0.7 tracker, {pct(f['return_rate'])} balls returned; stronger opponent results in the receipt",
                [("points won, skill 0.7", pct(f["win_rate"])), ("balls returned", pct(f["return_rate"])), ("incoming frames", "1"), ("parameters", f"{b['learner']['parameters']:,}")])
    raise KeyError(rung)


RUNGS = [
    ("01_digits", "digits", "the tutorial, draw one", "8×8 scikit-learn digits; 64 input owners, a hidden layer, 10 output owners. Classification: selection on validation data and three patch-net seeds, with fixed MLP baselines. Draw a digit in the page and watch the ten output owners settle."),
    ("02_recall", "recall", "memory as seams, play it", "Residual fast seams learn on each write and correct old associations. A dictionary solves the same explicitly addressed task. Write a context, then replace a value."),
    ("03_connect_four", "Connect Four", "imitation, play it", "Imitate a teacher, play, and revisit mistakes. Optional four-ply deliberation uses the game rules; raw-policy and search-only controls distinguish learned skill from planning. Local learning mode preserves lessons."),
    ("04_pong", "Pong", "reward, play it", "A paddle sees one frame and keeps a fading trace of its activity. It imitates diagonal returns, practises, and revisits teacher examples. Local learning mode saves every completed game as a reusable lesson."),
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
            if rung in ("03_connect_four", "04_pong"):
                if rung == "03_connect_four":
                    rows = ["Wins / draws / losses, 100 games per opponent:", "",
                            "| player | random | depth 2 | depth 4 |", "|---|---|---|---|"]
                    for label, key in (("raw policy", "strength"), ("policy + depth-4 search", "deployed_strength"), ("depth-4 search alone", "search_only_strength")):
                        games = {r["opponent"]: r for r in body[key]}
                        values = [f"{games[k]['wins']} / {games[k]['draws']} / {games[k]['losses']}" for k in ("random", "search-2", "search-4")]
                        rows.append("| " + label + " | " + " | ".join(values) + " |")
                else:
                    rows = ["| tracker skill | wins | losses | draws | points won |", "|---|---:|---:|---:|---:|"]
                    for skill in (.7, 1.):
                        games = [r for r in body["held_out_play"] if r["opponent_skill"] == skill]
                        counts = {k: sum(r[k] for r in games) for k in ("wins", "losses", "draws", "points")}
                        rows.append(f"| {skill:g}, seeds 101–103 | {counts['wins']:,} | {counts['losses']:,} | {counts['draws']:,} | {pct(counts['wins']/counts['points'])} |")
                path = ROOT / rung / "README.md"
                path.write_text(re.sub(r"<!-- game-results -->.*?<!-- /game-results -->",
                                       "<!-- game-results -->\n" + "\n".join(rows) + "\n<!-- /game-results -->",
                                       path.read_text(), flags=re.S))
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
        play = f"; [browser page]({page})" if page else ""
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
    for page_file, local in (("tutorials.html", True),):
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
