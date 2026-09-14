"""Serve the examples locally and open the hub, or one page, in a browser.

Run:  python serve.py            (the hub at http://localhost:8765/)
      python serve.py pong       (straight into a page: digits, recall, connect-four, pong)
      python serve.py --port 0 --no-browser  (choose an available port; print its URL)

The pages are plain HTML with trained nets embedded. With --learn, the local
Python server also learns from completed games and saves checkpoints. If a page is missing, run that example's build_page.py first (the recall page has no net to embed).
"""

from __future__ import annotations

import argparse
import http.server
import json
import sys
import threading
import webbrowser
from pathlib import Path
from typing import ClassVar

HERE = Path(__file__).resolve().parent
PORT = 8765
PAGES = {
    "worm": "#worm",
    "fly": "#fly",
    "memory": "#memory",
    "mouse": "#mouse",
    "eye-arm": "#arm",
    "digits": "01_digits",
    "recall": "02_recall",
    "connect-four": "03_connect_four",
    "connect4": "03_connect_four",
    "c4": "03_connect_four",
    "pong": "04_pong",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files as they are, declared UTF-8; the favicon request answered empty rather than 404."""

    extensions_map: ClassVar[dict[str, str]] = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".json": "application/json; charset=utf-8",
    }

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, directory=str(HERE), **kwargs)

    def send_json(self, data, status=200):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        lessons = getattr(self.server, "lessons", {})
        game = self.path.removeprefix("/learning/")
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            self.send_json({"error": "local same-origin lessons only"}, 403)
            return
        if (
            game not in lessons
            or self.headers.get("Content-Type") != "application/json"
        ):
            self.send_json({"error": "learning is not enabled for this game"}, 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 5_000_000:
                raise ValueError("invalid lesson size")
            result = lessons[game].teach(json.loads(self.rfile.read(size)))
            self.send_json(result)
        except (ValueError, TypeError, KeyError) as error:
            self.send_json({"error": str(error)}, 400)

    def do_GET(self) -> None:
        if self.path.startswith("/learning/"):
            game = self.path.removeprefix("/learning/")
            lesson = getattr(self.server, "lessons", {}).get(game)
            self.send_json(lesson.model() if lesson else {"enabled": False})
            return
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, format_string: str, *args: object) -> None:
        pass


def port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "port must be an integer from 0 to 65535"
        ) from error
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be an integer from 0 to 65535")
    return port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "page",
        nargs="?",
        default="",
        type=str.lower,
        help=f"page to open: {', '.join(PAGES)}; default: hub",
    )
    parser.add_argument(
        "--port",
        type=port_number,
        default=PORT,
        help="local port (default: 8765); 0 selects an available port",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="print the URL without opening a browser",
    )
    parser.add_argument(
        "--learn",
        action="store_true",
        help="learn locally from every game and save progress (requires training dependencies)",
    )
    args = parser.parse_args(argv)
    if args.page and args.page not in PAGES:
        parser.error(f"unknown page {args.page!r}; choose from {', '.join(PAGES)}")
    missing = [
        p
        for p in (
            "01_digits/index.html",
            "03_connect_four/index.html",
            "04_pong/index.html",
        )
        if not (HERE / p).exists()
    ]
    for p in missing:
        print(f"note: {p} is not built yet; run build_page.py in that directory")
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        parser.exit(1, f"Cannot start the local server: {error}. Try --port 0.\n")
    with server:
        server.lessons = {}
        if args.learn:
            from learning_server import GAMES, Lessons

            selected = {"c4": "connect-four", "connect4": "connect-four"}.get(
                args.page, args.page
            )
            for game in GAMES:
                if not selected or selected == game:
                    server.lessons[game] = Lessons(game)
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/" + (
            (PAGES[args.page] if PAGES[args.page].startswith("#") else f"{PAGES[args.page]}/index.html") if args.page else ""
        )
        print(f"serving {HERE} at {url}  (Ctrl-C to stop)", flush=True)
        opener = None
        if not args.no_browser:
            opener = threading.Timer(0.5, lambda: webbrowser.open(url))
            opener.daemon = True
            opener.start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if opener is not None:
                opener.cancel()
    return 0


if __name__ == "__main__":
    sys.exit(main())
