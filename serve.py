"""Serve the examples locally and open the hub in a browser.

Run:  python serve.py            (then http://localhost:8765/)

The pages are plain HTML with the trained nets embedded; nothing is computed on the
server. If a game page is missing, run that example's build_page.py first.
"""

from __future__ import annotations

import http.server
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 8765


def main() -> int:
    missing = [p for p in ("03_connect_four/index.html", "04_pong/index.html") if not (HERE / p).exists()]
    for p in missing:
        print(f"note: {p} is not built yet; run build_page.py in that directory")
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(HERE), **k)  # noqa: E731
    with socketserver.TCPServer(("", PORT), handler) as server:
        url = f"http://localhost:{PORT}/"
        print(f"serving {HERE} at {url}  (Ctrl-C to stop)")
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
