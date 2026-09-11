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
PAGES = {
    "04_pong": "https://claude.ai/code/artifact/4b3fe725-e687-4acb-a29c-5f2eb9a69e04",
    "07_music": "https://claude.ai/code/artifact/ed371f22-b7ea-4545-85f7-b6b573448528",
}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def summarise(rung: str, b: dict) -> tuple[str, list[tuple[str, str]]]:
    """(one-line result for the README, [(label, value), ...] facts for the hub card)."""
    if rung == "01_digits":
        m = [c for c in b["comparison"] if "64-32-10" in c["model"] and c["epochs"] >= 50][0]
        return (f"{b['test_accuracy_mean']:.3f} ± {b['test_accuracy_std']:.3f} held-out in {b['schedule']['epochs']} epochs; a same-size MLP {m['test_accuracy']:.3f} in {m['epochs']}",
                [("held-out", f"{b['test_accuracy_mean']:.3f} ± {b['test_accuracy_std']:.3f}"), ("parameters", f"{b['learner']['parameters']:,}"), ("epochs", str(b["schedule"]["epochs"])), ("MLP same size", f"{m['test_accuracy']:.3f}")])
    if rung == "03_recall":
        s = b["summary"]
        L = str(max(b["task"]["lengths"]))
        tr = b["runs"][0]["transformer"]
        return (f"the value of any key in a context of up to {L} pairs: {s['patch_settled'][L]:.2f} settled, {s['patch_one_read'][L]:.2f} in one read, with no trained parameters; a two-layer transformer trained on the task {s['transformer']['4']:.2f} at 4 pairs, {s['transformer'][L]:.2f} at {L}",
                [("recall at 128 pairs, settled", f"{s['patch_settled'][L]:.2f}"), ("one read (two steps)", f"{s['patch_one_read'][L]:.2f}"), ("trained parameters", "0"), ("transformer, trained", f"{s['transformer']['4']:.2f} at 4, {s['transformer'][L]:.2f} at {L}")])
    if rung == "04_pong":
        base = b["comparison"][0]["final"]
        im = b.get("imitation", {}).get("final")
        extra = f"; the same net taught the tracker's moves {pct(im['return_rate'])}" if im else ""
        return (f"reward-trained paddle returns {pct(b['final']['return_rate'])} of balls against {pct(base['return_rate'])} for backprop REINFORCE on the same rollouts{extra}",
                [("balls returned, from reward", pct(b["final"]["return_rate"])), ("backprop REINFORCE, same budget", pct(base["return_rate"])), ("from a tracker's moves", pct(im["return_rate"]) if im else "—"), ("parameters", f"{b['learner']['parameters']:,}")])
    if rung == "07_music":
        rows = {c["model"]: c for c in b["comparison"]}
        mlp = [v for k, v in rows.items() if "MLP" in k][0]
        return (f"next-chord pitch-set F1 {b['test']['f1']:.3f}, {b['test']['bits_per_chord']:.1f} bits/chord; repeat-last {rows['repeat the last chord']['test']['f1']:.3f}, MLP {mlp['test']['f1']:.3f}",
                [("pitch-set F1", f"{b['test']['f1']:.3f}"), ("bits per chord", f"{b['test']['bits_per_chord']:.1f}"), ("repeat the last chord", f"{rows['repeat the last chord']['test']['f1']:.3f}"), ("MLP same size", f"{mlp['test']['f1']:.3f}")])
    raise KeyError(rung)


RUNGS = [
    ("01_digits", "digits", "the tutorial", "8×8 scikit-learn digits; 64 input owners, a hidden layer, 10 output owners. The rule in the open: selection on a validation split, the test set read once, three seeds; parity with a same-size MLP in fewer epochs."),
    ("03_recall", "recall", "memory as seams", "Associative recall with no trained parameters: each key-value pair is one Hebbian outer product, each query a settlement with the key clamped, at any context length. A transformer trained on the task is the comparison."),
    ("04_pong", "Pong", "reward, play it", "A paddle that learned from pixels and reward: the nudge's target is the action taken, its strength the action's advantage. More balls from the same rollouts than backprop REINFORCE."),
    ("07_music", "chorales", "music, play it", "Continues Bach chorales chord by chord from an eight-chord window with a multi-hot quadratic nudge; ahead of a same-size MLP on both measures; listen in the page."),
]


def main() -> None:
    readme_rows, cards = [], []
    for rung, name, tag, blurb in RUNGS:
        k = int(rung[:2])  # the rung keeps its number; the ladder has gaps where rungs left it
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
        readme_rows.append(f"| {k:02d} | {link} | {blurb.split('.')[0].lower()}; receipt: {line}{play} |" if status == "done" else f"| {k:02d} | {link} | {blurb.split('.')[0].lower()}; {line} |")
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
