"""Serve the examples locally and open the hub, or one page, in a browser.

Run:  python serve.py            (the hub at http://localhost:8765/)
      python serve.py pong       (straight into a page: digits, recall, connect-four, pong)

The pages are plain HTML with the trained nets embedded; nothing is computed on the
server. If a page is missing, run that example's build_page.py first (the recall page has no net to embed).
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
PAGES = {"digits": "01_digits", "recall": "02_recall", "connect-four": "03_connect_four", "connect4": "03_connect_four", "c4": "03_connect_four", "pong": "04_pong"}


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files as they are, declared UTF-8; the favicon request answered empty rather than 404."""

    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".html": "text/html; charset=utf-8", ".md": "text/markdown; charset=utf-8", ".json": "application/json; charset=utf-8"}

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, directory=str(HERE), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def main() -> int:
    page = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if page and page not in PAGES:
        print(f"unknown page {page!r}; one of: {', '.join(sorted(set(PAGES)))}")
        return 2
    missing = [p for p in ("01_digits/index.html", "03_connect_four/index.html", "04_pong/index.html") if not (HERE / p).exists()]
    for p in missing:
        print(f"note: {p} is not built yet; run build_page.py in that directory")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as server:
        url = f"http://localhost:{PORT}/" + (f"{PAGES[page]}/index.html" if page else "")
        print(f"serving {HERE} at {url}  (Ctrl-C to stop)")
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
