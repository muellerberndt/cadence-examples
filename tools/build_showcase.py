"""Build six independent demo pages and a small gallery from authored shells."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "mouse": "mouse",
    "eye-arm": "arm",
    "fly": "fly",
    "worm": "worm",
    "memory": "memory",
    "connect-four": "game",
}
shell = (ROOT / "shared/shell.html").read_text()
for folder, mode in PAGES.items():
    page = shell.replace("<head>", '<head>\n    <base href="../" />').replace(
        "<body>", f'<body data-demo="{mode}">'
    )

    def link(match, current_mode=mode):
        key, label = match.groups()
        slug = {"arm": "eye-arm", "game": "connect-four"}.get(key, key)
        current = ' aria-current="page"' if key == current_mode else ""
        return f'<a href="{slug}/" data-tab="{key}"{current}>{label}</a>'

    page = re.sub(
        r'<button data-tab="([^"]+)" aria-pressed="(?:true|false)">(.*?)</button\s*>',
        link,
        page,
        flags=re.DOTALL,
    )
    page = page.replace('src="shared/app.js"', f'src="{folder}/page.js"')
    (ROOT / folder / "index.html").write_text(page)
    (ROOT / folder / "page.js").write_text(
        '// This page owns its demo; rendering and numerical utilities are shared.\nimport "../shared/app.js";\n'
    )
for name in ("index.html", "hub_published.html"):
    (ROOT / name).write_bytes((ROOT / "shared/gallery.html").read_bytes())
print("Six demo pages and gallery rebuilt.")
