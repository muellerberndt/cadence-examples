"""Embed the trained net into the page: page_template.html + net.json -> index.html."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE / "index.html")
    args = parser.parse_args()
    net = json.loads((args.model_dir / "net.json").read_text())
    html = (HERE / "page_template.html").read_text().replace("/*NET*/null", json.dumps(net, separators=(",", ":")))
    imitation = args.model_dir / "net_imitation.json"
    if imitation.exists():
        html = html.replace("/*NET_IMITATION*/null", imitation.read_text())
    else:
        html = html.replace("/*NET_IMITATION*/null", "null")
    hub = os.path.relpath(HERE.parent / "index.html", args.output.resolve().parent)
    html = html.replace('href="../index.html"', f'href="{Path(hub).as_posix()}"')
    html = html.replace("/*LOCAL_LESSONS*/", (HERE.parent / "learn.js").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"{args.output}: {len(html) / 1e3:.0f} kB, {net['n']} owners")
