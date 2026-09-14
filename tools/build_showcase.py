"""Build the default and publication hub from one authored shell."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in ("index.html", "hub_published.html"):
    (ROOT / name).write_bytes((ROOT / "showcase/shell.html").read_bytes())
print("Showcase hubs rebuilt.")
