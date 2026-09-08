"""Embed the trained brain into the page and the receipt's numbers into the README.

page_template.html + net.json + receipt.json -> index.html; the README's results block is
regenerated from receipt.json. The cochlea's filter coefficients are computed here from
syrinx.py so the page hears exactly what the training run heard.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from syrinx import Cochlea

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    net = json.loads((HERE / "net.json").read_text())
    receipt = json.loads((HERE / "receipt.json").read_text())["body"]
    s = receipt["summary"]
    net["cochlea"] = {"sos": [sec.round(9).tolist() for sec in Cochlea().sos]}
    net["facts"] = {
        "day": f"{receipt['day']['minutes']:.0f} minutes, {receipt['day']['events']} household sounds",
        "recall, a sound heard often / rarely / untrained": f"{s['recall_heard_often']:.3f} / {s['recall_heard_rarely']:.3f} / {s['recall_untrained']:.3f}",
        "imitation, a sound heard often / rarely / untrained": f"{s['imitation_heard_often']:.3f} / {s['imitation_heard_rarely']:.3f} / {s['imitation_untrained']:.3f}",
        "bouts": f"{s['bouts']} ({s['imitation_bouts']} imitations, {s['babble_bouts']} babbles)",
        "owners / parameters": f"{receipt['brain']['owners']} / {receipt['brain']['parameters']:,}",
    }
    html = (HERE / "page_template.html").read_text().replace("/*NET*/null", json.dumps(net, separators=(",", ":")))
    (HERE / "index.html").write_text(html)
    print(f"index.html: {len(html) / 1e3:.0f} kB, {net['n']} owners")
    rows = ["| sound | heard often / rarely | recall, heard often | recall, heard rarely | recall, untrained | imitation, heard often | imitation, heard rarely | imitation, untrained |", "|---|---|---|---|---|---|---|---|"]
    for name, r in receipt["per_sound"].items():
        rows.append(f"| {name} | {r['heard_often']} / {r['heard_rarely']} | {r['recall_heard_often']:.3f} | {r['recall_heard_rarely']:.3f} | {r['recall_untrained']:.3f} | {r['imitation_heard_often']:.3f} | {r['imitation_heard_rarely']:.3f} | {r['imitation_untrained']:.3f} |")
    block = "\n".join([
        "<!-- results -->",
        f"Two {receipt['day']['minutes']:.0f}-minute days, the second with the roles swapped so every sound is scored once heard often and once heard rarely (day A: {receipt['day']['events']} household sounds, {s['bouts']} spontaneous bouts, {s['babble_bouts']} babbles and {s['imitation_bouts']} imitations; {receipt['seconds']:.0f} s of wall-clock a day):",
        "",
        *rows,
        "",
        f"Means: recall {s['recall_heard_often']:.3f} when a sound was heard often against {s['recall_heard_rarely']:.3f} when it was rare and {s['recall_untrained']:.3f} untrained; {s['sounds_recalled_better_when_heard_often']} of {len(receipt['per_sound'])} sounds are recalled better after the day that repeated them. Imitation {s['imitation_heard_often']:.3f} / {s['imitation_heard_rarely']:.3f} / {s['imitation_untrained']:.3f}. Brain: {receipt['brain']['owners']} owners, {receipt['brain']['parameters']:,} parameters. The receipt binds these numbers to the code and the seed.",
        "<!-- /results -->",
    ])
    readme = HERE / "README.md"
    text = readme.read_text()
    readme.write_text(re.sub(r"<!-- results -->.*?<!-- /results -->", lambda m: block, text, flags=re.S))
    print("README results block regenerated")
