"""Open a Cadence demo: python serve.py [mouse|eye-arm|fly|worm|memory|connect-four].

Only Python's standard library is needed. Every demo ships its browser assets.
The server binds to this computer only. Ctrl-C stops it.
"""

from __future__ import annotations

import argparse
import http.server
import sys
import threading
import webbrowser
from pathlib import Path
from typing import ClassVar

HERE = Path(__file__).resolve().parent
PAGES = {
    "mouse": "mouse",
    "eye-arm": "eye-arm",
    "arm": "eye-arm",
    "fly": "fly",
    "worm": "worm",
    "memory": "memory",
    "connect-four": "connect-four",
    "game": "connect-four",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map: ClassVar[dict[str, str]] = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, format_string, *args):
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
        "page", nargs="?", default="mouse", type=str.lower, choices=PAGES
    )
    parser.add_argument(
        "--port",
        type=port_number,
        default=0,
        help="local port; default: select an available port",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="print the URL without opening a browser",
    )
    args = parser.parse_args(argv)
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        parser.exit(1, f"Cannot start the local server: {error}. Try --port 0.\n")
    with server:
        url = f"http://127.0.0.1:{server.server_address[1]}/{PAGES[args.page]}/"
        print(f"Cadence · {args.page}\n{url}\nCtrl-C to stop.", flush=True)
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
