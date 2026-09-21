"""Build web/index.html from web/page.html with sim/patch.js, sim/core.js and sim/founder.json inlined.

python3 web/build.py            writes web/index.html
python3 web/build.py --check    exits 1 if web/index.html is stale
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build() -> str:
    page = (ROOT / "web" / "page.html").read_text()
    patch = (ROOT / "sim" / "patch.js").read_text()
    core = (ROOT / "sim" / "core.js").read_text()
    founder_file = ROOT / "sim" / "founder.json"
    founder = json.loads(founder_file.read_text())["founder"] if founder_file.exists() else None
    marker = "<!-- CORE -->"
    assert marker in page, "page.html needs the CORE marker"
    scripts = "<script>\n" + patch + "\n</script>\n<script>\n" + core + "\n</script>\n<script>const CB_FOUNDER = " + json.dumps(founder) + ";</script>"
    return page.replace(marker, scripts)


if __name__ == "__main__":
    out = ROOT / "web" / "index.html"
    html = build()
    if "--check" in sys.argv:
        sys.exit(0 if out.exists() and out.read_text() == html else 1)
    out.write_text(html)
    print(f"wrote {out} ({len(html)} bytes)")
