"""Embed the two trained parrots into the app and the receipt's numbers into the README.

page_template.html + net.json + net_1.json + receipt.json (+ receipt_1.json) -> index.html;
the README's results block is regenerated from receipt.json. The cochlea's filter
coefficients are computed here from syrinx.py so the page hears exactly what the
training runs heard.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from syrinx import Cochlea

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    nets = [json.loads((HERE / name).read_text()) for name in ("net.json", "net_1.json") if (HERE / name).exists()]
    receipts = [json.loads((HERE / name).read_text())["body"] for name in ("receipt.json", "receipt_1.json") if (HERE / name).exists()]
    first = nets[0]
    recorded = [json.loads((HERE / name).read_text()) for name in ("imitations.json", "imitations_1.json") if (HERE / name).exists()]
    payload = {
        "parrots": nets, "channels": first["channels"], "window": first["window"], "mirror_window": first["mirror_window"], "levels": first["levels"], "phases": first["phases"], "max_phase": first["max_phase"],
        "cochlea": {"sos": [sec.round(9).tolist() for sec in Cochlea().sos]},
        "recorded": [{name: {"commands": [[round(c["tension"], 4), round(c["pressure"], 4), round(c["tract"], 4)] for c in r[name]["commands"]], "similarity": r[name]["similarity"]} for name in r} for r in recorded],
        "facts": {},
    }
    for k, receipt in enumerate(receipts):
        s = receipt["summary"]
        name = ["Coco", "Pepper"][k] if k < 2 else f"parrot {k + 1}"
        payload["facts"][f"{name}: recall of a sound heard often / rarely / untrained"] = f"{s['recall_heard_often']:.3f} / {s['recall_heard_rarely']:.3f} / {s['recall_untrained']:.3f}"
        payload["facts"][f"{name}: imitation, heard often / rarely / untrained"] = f"{s['imitation_heard_often']:.3f} / {s['imitation_heard_rarely']:.3f} / {s['imitation_untrained']:.3f}"
        payload["facts"][f"{name}: pitch track of the imitation, heard often / rarely / untrained"] = f"{s['pitch_heard_often']:.3f} / {s['pitch_heard_rarely']:.3f} / {s['pitch_untrained']:.3f}"
        payload["facts"][f"{name}: pitch distance in channels, heard often / rarely / untrained"] = f"{s['pitch_distance_heard_often']:.2f} / {s['pitch_distance_heard_rarely']:.2f} / {s['pitch_distance_untrained']:.2f}"
        payload["facts"][f"{name}: rhythm of the imitation, heard often / rarely / untrained"] = f"{s['rhythm_heard_often']:.3f} / {s['rhythm_heard_rarely']:.3f} / {s['rhythm_untrained']:.3f}"
        payload["facts"][f"{name}: spontaneous bouts on day A"] = f"{s['bouts']} ({s['babble_bouts']} babbles, {s['imitation_bouts']} imitations)"
    r0 = receipts[0]
    payload["facts"]["a day"] = f"{r0['day']['minutes']:.0f} minutes, {r0['day']['events']} household sounds; two days each, the second with the frequent and rare roles swapped"
    payload["facts"]["brain"] = f"{r0['brain']['owners']} owners, {r0['brain']['parameters']:,} parameters; memory rate {r0['brain']['eta']}, mirror rate {r0['brain']['mirror_eta']}, seam decay {r0['brain']['decay']}"
    html = (HERE / "page_template.html").read_text().replace("/*NETS*/null", json.dumps(payload, separators=(",", ":")))
    (HERE / "index.html").write_text(html)
    print(f"index.html: {len(html) / 1e3:.0f} kB, {len(nets)} parrots of {first['n']} owners")
    s = r0["summary"]
    rows = ["| sound | heard often / rarely | recall, heard often | recall, heard rarely | recall, untrained | imitation, heard often | imitation, heard rarely | imitation, untrained | pitch track, heard often | pitch distance (channels), heard often | rhythm, heard often |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in r0["per_sound"].items():
        rows.append(f"| {name} | {r['heard_often']} / {r['heard_rarely']} | {r['recall_heard_often']:.3f} | {r['recall_heard_rarely']:.3f} | {r['recall_untrained']:.3f} | {r['imitation_heard_often']:.3f} | {r['imitation_heard_rarely']:.3f} | {r['imitation_untrained']:.3f} | {r['pitch_heard_often']:.3f} | {r['pitch_distance_heard_often']:.2f} | {r['rhythm_heard_often']:.3f} |")
    second = ""
    if len(receipts) > 1:
        s1 = receipts[1]["summary"]
        second = f" The second parrot (seed 1, its own receipt) recalls {s1['recall_heard_often']:.3f} / {s1['recall_heard_rarely']:.3f} and imitates at {s1['imitation_heard_often']:.3f} / {s1['imitation_heard_rarely']:.3f} against {s1['imitation_untrained']:.3f} untrained, with a pitch distance of {s1['pitch_distance_heard_often']:.2f} channels."
    block = "\n".join([
        "<!-- results -->",
        f"Two {r0['day']['minutes']:.0f}-minute days, the second with the roles swapped so every sound is scored once heard often and once heard rarely (day A: {r0['day']['events']} household sounds, {s['bouts']} spontaneous bouts, {s['babble_bouts']} babbles and {s['imitation_bouts']} imitations; {r0['seconds']:.0f} s of wall-clock a day). The first parrot, seed 0:",
        "",
        *rows,
        "",
        f"Means: recall {s['recall_heard_often']:.3f} when a sound was heard often against {s['recall_heard_rarely']:.3f} when it was rare and {s['recall_untrained']:.3f} untrained; {s['sounds_recalled_better_when_heard_often']} of {len(r0['per_sound'])} sounds are recalled better after the day that repeated them. Imitation (whole cochleagram) {s['imitation_heard_often']:.3f} / {s['imitation_heard_rarely']:.3f} / {s['imitation_untrained']:.3f}; spectral shape {s['shape_heard_often']:.3f} / {s['shape_heard_rarely']:.3f} / {s['shape_untrained']:.3f}; pitch track {s['pitch_heard_often']:.3f} / {s['pitch_heard_rarely']:.3f} / {s['pitch_untrained']:.3f}; pitch distance {s['pitch_distance_heard_often']:.2f} / {s['pitch_distance_heard_rarely']:.2f} / {s['pitch_distance_untrained']:.2f} channels; rhythm {s['rhythm_heard_often']:.3f} / {s['rhythm_heard_rarely']:.3f} / {s['rhythm_untrained']:.3f}. Brain: {r0['brain']['owners']} owners, {r0['brain']['parameters']:,} parameters.{second} The receipts bind these numbers to the code and the seeds.",
        "<!-- /results -->",
    ])
    readme = HERE / "README.md"
    if readme.exists():
        text = readme.read_text()
        readme.write_text(re.sub(r"<!-- results -->.*?<!-- /results -->", lambda m: block, text, flags=re.S))
        print("README results block regenerated")
