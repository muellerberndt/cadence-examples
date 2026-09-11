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
    "01_digits": "https://claude.ai/code/artifact/20255a93-c6f1-4a65-8ae0-62362fd6636c",
    "02_recall": "https://claude.ai/code/artifact/3e74cec0-ac36-4e81-8bbb-be9997f495bf",
    "03_connect_four": "https://claude.ai/code/artifact/7eaebd77-b8f7-4415-8a08-6aefc9aff570",
    "04_pong": "https://claude.ai/code/artifact/112fbedd-4191-42dd-b890-064f629befc3",
}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def summarise(rung: str, b: dict) -> tuple[str, list[tuple[str, str]]]:
    """(one-line result for the README, [(label, value), ...] facts for the hub card)."""
    if rung == "01_digits":
        m = [c for c in b["comparison"] if "64-32-10" in c["model"] and c["epochs"] >= 50][0]
        return (f"{b['test_accuracy_mean']:.3f} ± {b['test_accuracy_std']:.3f} held-out in {b['schedule']['epochs']} epochs; a same-size MLP {m['test_accuracy']:.3f} in {m['epochs']}",
                [("held-out", f"{b['test_accuracy_mean']:.3f} ± {b['test_accuracy_std']:.3f}"), ("parameters", f"{b['learner']['parameters']:,}"), ("epochs", str(b["schedule"]["epochs"])), ("MLP same size", f"{m['test_accuracy']:.3f}")])
    if rung == "02_recall":
        s = b["summary"]
        L = str(max(b["task"]["lengths"]))
        tr = b["runs"][0]["transformer"]
        return (f"the value of any key in a context of up to {L} pairs: {s['patch_settled'][L]:.2f} settled, {s['patch_one_read'][L]:.2f} in one read, with no trained parameters; a two-layer transformer given {int(b['runs'][0]['transformer']['model'].split('trained ')[1].split(' steps')[0]):,} Adam steps on the task did not learn it ({s['transformer']['4']:.2f} at 4 pairs, {s['transformer'][L]:.2f} at {L})",
                [("recall at 128 pairs, settled", f"{s['patch_settled'][L]:.2f}"), ("one read (two steps)", f"{s['patch_one_read'][L]:.2f}"), ("trained parameters", "0"), ("transformer, 5,000 Adam steps", f"{s['transformer']['4']:.2f} at 4, {s['transformer'][L]:.2f} at {L}")])
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
        return (f"reward-trained paddle returns {pct(b['final']['return_rate'])} of balls against {pct(base['return_rate'])} for backprop REINFORCE on the same rollouts{extra}",
                [("balls returned, from reward", pct(b["final"]["return_rate"])), ("backprop REINFORCE, same budget", pct(base["return_rate"])), ("from a tracker's moves", pct(im["return_rate"]) if im else "—"), ("parameters", f"{b['learner']['parameters']:,}")])
    raise KeyError(rung)


RUNGS = [
    ("01_digits", "digits", "the tutorial, draw one", "8×8 scikit-learn digits; 64 input owners, a hidden layer, 10 output owners. The rule in the open: selection on a validation split, the test set read once, three seeds; the accuracy of a same-size MLP in fewer epochs. Draw a digit in the page and watch the ten output owners settle."),
    ("02_recall", "recall", "memory as seams, play it", "Associative recall with no trained parameters: each key-value pair is one Hebbian outer product, each query a settlement with the key clamped, at any context length. A transformer trained on the task is the comparison. Write a context in the page and ask it."),
    ("03_connect_four", "Connect Four", "imitation, play it", "Self-play positions labelled by a depth-4 search; the net imitates the search and plays with no lookahead, at the MLP's agreement. You play coral, the net plays gold."),
    ("04_pong", "Pong", "reward, play it", "A paddle that learned from pixels and reward: the nudge's target is the action taken, its strength the action's advantage, each seam's step read from its own history. Behind backprop REINFORCE with Adam on the same rollouts; the same net taught a tracker's moves returns 96%. You play coral, the net plays gold."),
]


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
