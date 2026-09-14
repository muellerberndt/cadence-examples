"""Open a Cadence example: python serve.py [eye-arm|worm|composer|fly|connect-four].

The four browser demos need only Python's standard library and ship their assets.
`composer` starts the composer studio, which needs its requirements and the
pretrained maestro-1 model (see composer/README.md). Servers bind to this
computer only. Ctrl-C stops them.
"""

from __future__ import annotations

import argparse
import http.server
import importlib.util
import os
import sys
import threading
import webbrowser
from pathlib import Path
from typing import ClassVar

HERE = Path(__file__).resolve().parent
PAGES = {
    "eye-arm": "eye-arm",
    "arm": "eye-arm",
    "fly": "fly",
    "worm": "worm",
    "connect-four": "connect-four",
    "game": "connect-four",
}

COMPOSER = HERE / "composer"
COMPOSER_CHECKPOINT = COMPOSER / "checkpoints" / "maestro-1" / "brain.npz"
COMPOSER_SETUP = """The composer studio needs its requirements and the maestro-1 model:

  cd composer
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python tools/fetch_model.py
  cd .. && python serve.py composer

Details: composer/README.md
"""


def serve_composer(port: int) -> int:
    """Replace this process with the composer studio, or print its setup steps."""
    if importlib.util.find_spec("cadence") is None or not COMPOSER_CHECKPOINT.is_file():
        sys.stderr.write(COMPOSER_SETUP)
        return 1
    port = port or 8079
    print(
        f"Cadence · composer\nhttp://127.0.0.1:{port}/\n"
        "Open the address once the studio reports that maestro-1 is loaded (a few minutes).",
        flush=True,
    )
    os.chdir(COMPOSER)
    args = [sys.executable, "serve_musician.py", "--checkpoint", str(COMPOSER_CHECKPOINT), "--port", str(port)]
    os.execv(sys.executable, args)
    return 0


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
        "page", nargs="?", default="eye-arm", type=str.lower, choices=[*PAGES, "composer"]
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
    if args.page == "composer":
        return serve_composer(args.port)
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
