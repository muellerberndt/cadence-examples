"""Embed the trained net into the page: page_template.html + net.json -> index.html."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    net = json.loads((HERE / "net.json").read_text())
    receipt = HERE / "receipt.json"
    if receipt.exists():  # the page draws pitches with the calibration the receipt selected on validation
        net["music"]["slope"] = json.loads(receipt.read_text())["body"]["slope"]
    html = (HERE / "page_template.html").read_text().replace("/*NET*/null", json.dumps(net, separators=(",", ":")))
    (HERE / "index.html").write_text(html)
    print(f"index.html: {len(html) / 1e3:.0f} kB, {net['n']} owners")
