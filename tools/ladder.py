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
    "03_connect_four": "https://claude.ai/code/artifact/7eaebd77-b8f7-4415-8a08-6aefc9aff570",
    "04_pong": "https://claude.ai/code/artifact/112fbedd-4191-42dd-b890-064f629befc3",
    "06_sign": "https://claude.ai/code/artifact/94e42c3b-134e-47c0-94bb-06106d3f6321",
    "07_music": "https://claude.ai/code/artifact/06f247a3-5acb-4663-91c6-9474f087a51e",
}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def summarise(rung: str, b: dict) -> tuple[str, list[tuple[str, str]]]:
    """(one-line result for the README, [(label, value), ...] facts for the hub card)."""
    if rung == "01_digits":
        m = [c for c in b["comparison"] if "64-32-10" in c["model"] and c["epochs"] >= 50][0]
        return (f"{b['test_accuracy_mean']:.3f} ± {b['test_accuracy_std']:.3f} held-out in {b['schedule']['epochs']} epochs; a same-size MLP {m['test_accuracy']:.3f} in {m['epochs']}",
                [("held-out", f"{b['test_accuracy_mean']:.3f} ± {b['test_accuracy_std']:.3f}"), ("parameters", f"{b['learner']['parameters']:,}"), ("epochs", str(b["schedule"]["epochs"])), ("MLP same size", f"{m['test_accuracy']:.3f}")])
    if rung == "02_images":
        m = [c for c in b["comparison"] if c["epochs"] == b["schedule"]["epochs"] and "MLP" in c["model"]][0]
        return (f"{pct(b['test_accuracy'])} held-out in {b['schedule']['epochs']} epochs against {pct(m['test_accuracy'])} for a same-size MLP; the two backends agree on every prediction",
                [("held-out", pct(b["test_accuracy"])), ("parameters", f"{b['learner']['parameters']:,}"), ("epochs", str(b["schedule"]["epochs"])), ("MLP same size", pct(m["test_accuracy"]))])
    if rung == "03_connect_four":
        s = {r["opponent"]: r for r in b["strength"]}
        m = b["comparison"][0]
        rec = lambda r: f"{r['wins']}-{r['draws']}-{r['losses']}"
        return (f"agrees with a depth-4 search on {b['test_agreement']:.3f} of positions (MLP {m['agreement']:.3f}); {rec(s['random'])} vs random, {rec(s['search-2'])} vs depth 2",
                [("agreement with the search", f"{b['test_agreement']:.3f}"), ("MLP same size", f"{m['agreement']:.3f}"), ("vs random", rec(s["random"])), ("vs depth 2", rec(s["search-2"]))])
    if rung == "04_pong":
        base = b["comparison"][0]["final"]
        im = b.get("imitation", {}).get("final")
        extra = f"; the same net taught the tracker's moves {pct(im['return_rate'])}" if im else ""
        return (f"reward-trained paddle returns {pct(b['final']['return_rate'])} of balls against {pct(base['return_rate'])} for backprop REINFORCE on the same rollouts{extra}",
                [("balls returned, from reward", pct(b["final"]["return_rate"])), ("backprop REINFORCE, same budget", pct(base["return_rate"])), ("from a tracker's moves", pct(im["return_rate"]) if im else "—"), ("parameters", f"{b['learner']['parameters']:,}")])
    if rung == "05_text":
        rows = {c["model"]: c for c in b["comparison"]}
        tr = [v for k, v in rows.items() if "transformer" in k][0]
        return (f"{b['test']['bits_per_char']:.2f} bits/char, next-character accuracy {pct(b['test']['accuracy'])}; bigram {rows['bigram (Laplace)']['bits_per_char']:.2f}, one-layer transformer {tr['bits_per_char']:.2f}",
                [("bits per character", f"{b['test']['bits_per_char']:.2f}"), ("accuracy", pct(b["test"]["accuracy"])), ("bigram", f"{rows['bigram (Laplace)']['bits_per_char']:.2f}"), ("transformer", f"{tr['bits_per_char']:.2f}")])
    if rung == "06_sign":
        return (f"writes held-out signs at {b['held_out']['iou_sign']:.3f} overlap with the sign, the teacher's own score; the MLP the same",
                [("overlap with the sign", f"{b['held_out']['iou_sign']:.3f}"), ("teacher", f"{b['held_out']['teacher_iou_sign']:.3f}"), ("parameters", f"{b['learner']['parameters']:,}"), ("MLP same size", f"{b['comparison'][0]['held_out']['iou_sign']:.3f}")])
    if rung == "07_music":
        rows = {c["model"]: c for c in b["comparison"]}
        mlp = [v for k, v in rows.items() if "MLP" in k][0]
        return (f"next-chord pitch-set F1 {b['test']['f1']:.3f}, {b['test']['bits_per_chord']:.1f} bits/chord; repeat-last {rows['repeat the last chord']['test']['f1']:.3f}, MLP {mlp['test']['f1']:.3f}",
                [("pitch-set F1", f"{b['test']['f1']:.3f}"), ("bits per chord", f"{b['test']['bits_per_chord']:.1f}"), ("repeat the last chord", f"{rows['repeat the last chord']['test']['f1']:.3f}"), ("MLP same size", f"{mlp['test']['f1']:.3f}")])
    if rung == "08_cartpole":
        base = b["comparison"][0]["final"]
        return (f"balances for {b['final']['mean_length']:.0f} steps of 500 against {base['mean_length']:.0f} for backprop REINFORCE",
                [("mean episode length", f"{b['final']['mean_length']:.0f} / 500"), ("backprop REINFORCE", f"{base['mean_length']:.0f} / 500"), ("parameters", f"{b['learner']['parameters']:,}"), ("iterations", str(b["environment"]["iterations"]))])
    if rung == "09_celegans":
        s = b["summary"]
        f = s.get("fan_in", {})
        return (f"count convention: nothing learned; fan-in convention: measured {f.get('measured_mean', 0):.1f}/17 held-out ablations vs shuffled {f.get('shuffled_mean', 0):.1f}/17, a structural signal, not a behavioural model",
                [("fan-in, measured", f"{f.get('measured_mean', 0):.1f} / 17"), ("fan-in, shuffled", f"{f.get('shuffled_mean', 0):.1f} / 17"), ("count, either wiring", f"{s['count']['measured_mean']:.1f} / 17"), ("training facts", f"{f.get('measured_training_mean', 0):.1f} / 4")])
    raise KeyError(rung)


RUNGS = [
    ("01_digits", "digits", "classification", "8×8 scikit-learn digits; 64 input owners, a hidden layer, 10 output owners. Selection on a validation split, the test set read once, three seeds."),
    ("02_images", "images", "image recognition", "MNIST, 784 pixels a picture, trained on Apple silicon in float32 and read out on the float64 reference backend."),
    ("03_connect_four", "Connect Four", "game, play it", "Self-play positions labelled by a depth-4 search; the net imitates the search and plays with no lookahead. You play coral, the net plays gold."),
    ("04_pong", "Pong", "reward, play it", "A paddle that learned from pixels and reward: the nudge's target is the action taken, its strength the action's advantage. A second paddle learned from a tracker's moves."),
    ("05_text", "text", "language", "Next-character prediction on Shakespeare from a sixteen-character window: 1,040 input owners, 65 output owners, and a sample the net wrote."),
    ("06_sign", "sign writer", "see and act, play it", "Show it a sign; it writes what it sees with a two-joint arm, looking through a 7×7 window around its pen."),
    ("07_music", "chorales", "music, play it", "Continues Bach chorales chord by chord from an eight-chord window with a multi-hot quadratic nudge; listen in the page."),
    ("08_cartpole", "cart-pole", "reward, a body", "The classic control task from reward, with the state as a place code."),
    ("09_celegans", "C. elegans", "measured wiring", "The published connectome learns four textbook facts and is scored on seventeen held-out ablation phenotypes against a shuffled wiring."),
]


def main() -> None:
    readme_rows, cards = [], []
    for k, (rung, name, tag, blurb) in enumerate(RUNGS, start=1):
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
    cards.append('    <article class="rung next">\n      <div class="num">10</div>\n      <div>\n        <h2>Embodiment <span class="tag">next</span></h2>\n        <p>A nervous system in a physical body: the FlyWire brain and MANC nerve cord driving a biomechanical fly in MuJoCo, ported onto the library.</p>\n        <div class="facts"><span>status <b>not started</b></span></div>\n      </div>\n      <div class="actions"></div>\n    </article>')
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
