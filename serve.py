"""Private local composer studio. Launch: ../cadence/.venv/bin/python serve.py"""

import argparse
import gzip
import json
import mimetypes
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from composer.brain import describe
from composer.compose import ROOT, Composer
from composer.telemetry import capture


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8078)
    p.add_argument("--checkpoint")
    p.add_argument("--taste", help="Optional saved relationship preference memory")
    a = p.parse_args()
    composer = Composer(a.checkpoint, taste=a.taste)
    lock = threading.Lock()
    model_lock = threading.Lock()
    worker = ThreadPoolExecutor(max_workers=1)
    events = deque(maxlen=160)
    event_id = 0
    state = {
        "busy": False,
        "progress": {"stage": "Ready"},
        "result": None,
        "error": None,
        "trace": None,
        "trace_revision": 0,
    }
    state["trace"] = capture(
        composer.performer,
        np.zeros((1, composer.performer.engine.wiring.n)),
        packed=True,
    )
    state["trace"]["origin"] = "Rest with learned biases; no sensory input"
    state["trace_revision"] = 1
    existing = sorted((ROOT / "runs/compositions").glob("*/composition.json"))
    if existing:
        state["result"] = json.loads(existing[-1].read_text())

    def update(event):
        nonlocal event_id
        with lock:
            event_id += 1
            events.append(
                {
                    "sequence": event_id,
                    **{k: v for k, v in event.items() if k != "trace"},
                }
            )
            state["progress"] = {k: v for k, v in event.items() if k != "trace"}
            if event.get("trace") is not None:
                state["trace"] = event["trace"]
                state["trace_revision"] += 1

    def job(prompt, seed):
        try:
            with model_lock:
                result = composer.compose(prompt, seed=seed, progress=update)
            with lock:
                state.update(result=result, error=None)
        except Exception as exc:  # noqa: BLE001 — report a worker failure to its browser session
            with lock:
                state["error"] = str(exc)
        finally:
            with lock:
                state["busy"] = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, code=200, kind="application/json"):
            payload = json.dumps(data).encode() if kind == "application/json" else data
            self.send_response(code)
            self.send_header("Content-Type", kind)
            if kind == "application/json" and "gzip" in self.headers.get(
                "Accept-Encoding", ""
            ):
                payload = gzip.compress(payload, compresslevel=1)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlparse(self.path).path
            query = parse_qs(urlparse(self.path).query)
            if path == "/api/events":
                try:
                    after = int(query.get("after", ["0"])[0])
                except ValueError:
                    return self.send({"error": "Invalid event cursor"}, 400)
                with lock:
                    latest = [e for e in events if e["sequence"] > after]
                return self.send({"events": latest, "latest": event_id})
            if path == "/api/state":
                with lock:
                    response = dict(state)
                if (
                    response.get("result", {})
                    and query.get("result", [None])[0] == response["result"]["id"]
                ):
                    response["result"] = None
                if query.get("trace", [None])[0] == str(response["trace_revision"]):
                    response["trace"] = None
                response["brain"] = describe(composer.performer)
                receipt = composer.path.parent / "receipt.json"
                if receipt.exists():
                    response["training"] = json.loads(receipt.read_text())
                return self.send(response)
            if re.fullmatch(r"/graph/[a-f0-9]{64}\.bin", path):
                target = ROOT / "runs/topology" / path.split("/")[-1]
            elif path.startswith("/output/"):
                if not re.fullmatch(
                    r"/output/[0-9]+-[0-9]+/(draft|final)\.(mid|wav)", path
                ):
                    return self.send({"error": "Not found"}, 404)
                target = ROOT / "runs/compositions" / path.removeprefix("/output/")
            elif path in ("/", "/app.js", "/style.css", "/circuit_map.js"):
                target = ROOT / "web" / ("index.html" if path == "/" else path[1:])
            else:
                return self.send({"error": "Not found"}, 404)
            if not target.is_file():
                return self.send({"error": "Not found"}, 404)
            return self.send(
                target.read_bytes(),
                kind=mimetypes.guess_type(target)[0] or "application/octet-stream",
            )

        def do_POST(self):
            # Local API: JSON-only plus same-origin validation blocks cross-site form submissions.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                return self.send({"error": "Origin rejected"}, 403)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send({"error": "Use JSON"}, 415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if self.path == "/api/compose":
                    from composer.music import parse_prompt

                    prompt = request.get("prompt", "")
                    parse_prompt(prompt)
                    seed = int(request.get("seed", 17))
                    if not 0 <= seed < 2**32:
                        raise ValueError("Invalid seed")
                    with lock:
                        if state["busy"]:
                            return self.send(
                                {"error": "A composition is already in progress"}, 409
                            )
                        state.update(
                            busy=True, error=None, progress={"stage": "Preparing"}
                        )
                    worker.submit(job, prompt, seed)
                    return self.send({"started": True}, 202)
                if self.path == "/api/release":
                    if not model_lock.acquire(blocking=False):
                        return self.send({"error": "Brain is busy"}, 409)
                    try:
                        return self.send({"trace": composer.release()})
                    finally:
                        model_lock.release()
                if self.path == "/api/hear":
                    with lock:
                        result = state["result"]
                        busy = state["busy"]
                    if (
                        busy
                        or result is None
                        or request.get("id") != result["id"]
                        or not model_lock.acquire(blocking=False)
                    ):
                        return self.send({"error": "Brain is busy"}, 409)
                    try:
                        return self.send(
                            composer.hear(
                                result,
                                float(request["seconds"]),
                                request.get("version", "final"),
                            )
                        )
                    finally:
                        model_lock.release()
                if self.path == "/api/feedback":
                    with lock:
                        if state["busy"] or not state["result"]:
                            return self.send(
                                {"error": "Finish a composition first"}, 409
                            )
                        state["busy"] = True
                        result = state["result"]
                    try:
                        with model_lock:
                            return self.send(
                                composer.feedback(result, int(request["rating"]))
                            )
                    finally:
                        with lock:
                            state["busy"] = False
                return self.send({"error": "Not found"}, 404)
            except (ValueError, KeyError, TypeError) as exc:
                return self.send({"error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(
        f"Cadence Composer: http://127.0.0.1:{a.port} · {describe(composer.performer)['owners']:,} owners",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.shutdown(wait=True)


if __name__ == "__main__":
    main()
