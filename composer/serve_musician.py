"""The musician's studio: compose live in the browser and watch the whole brain.

    ../cadence/.venv/bin/python serve_musician.py --checkpoint runs/large/brain.npz [--backend torch --device mps]

Open http://127.0.0.1:8079. Every neuron and synapse is mapped; every settling
iteration of every committed event is recorded and replayed in sync with the music.
"""

import argparse
import gzip
import json
import mimetypes
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from composer.musician import Musician, describe
from composer.perform import ROOT, perform
from composer.topology import topology

COMPOSITIONS = ROOT / "runs/musician-compositions"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8079)
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/musician/brain.npz"))
    p.add_argument("--backend", default="cpu")
    p.add_argument("--device", default=None)
    p.add_argument("--settle", type=int, default=48, help="settling iterations per event when playing (96 is nearer the equilibrium, twice as slow)")
    p.add_argument("--edge-limit", type=int, default=30000000, help="draw at most this many synapses (the strongest); every neuron is always drawn")
    a = p.parse_args()
    musician = Musician.load(a.checkpoint, backend=a.backend, device=a.device)
    musician.settle_steps = a.settle
    receipt_path = Path(a.checkpoint).with_name("receipt.json")
    training = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    target = training.get("best_validation_nll", 1.0)
    graph = topology(musician.brain, a.edge_limit)
    brain_info = describe(musician)  # counting parameters over tens of millions of synapses takes seconds
    lock = threading.Lock()
    worker = ThreadPoolExecutor(max_workers=1)
    events = deque(maxlen=400)
    counter = {"id": 0}
    state = {"busy": False, "progress": {"stage": "Ready"}, "result": None, "error": None}
    existing = sorted(COMPOSITIONS.glob("*/composition.json"), key=lambda f: f.stat().st_mtime)
    if existing:
        state["result"] = json.loads(existing[-1].read_text())

    def emit(event):
        with lock:
            counter["id"] += 1
            events.append({"sequence": counter["id"], **event})
            state["progress"] = {k: v for k, v in event.items() if k in ("stage", "step", "total", "bar", "seconds")}

    def job(mood, bars, seed, futures, edits, tempo, key, render, temperature=0.9, top=0):
        try:
            folder = COMPOSITIONS / f"{time.time_ns()}-{seed}"
            result = perform(
                musician, mood, folder=folder, key=key, render=render, checkpoint=a.checkpoint,
                target_surprise=target, progress=emit, bars=bars, futures=futures, edits=edits, seed=seed, tempo=tempo,
                temperature=temperature, top=top,
            )
            with lock:
                state.update(result=result, error=None)
            emit({"stage": "ready", "id": result["id"]})
        except Exception as exc:  # noqa: BLE001 — report a worker failure to the browser
            with lock:
                state["error"] = f"{type(exc).__name__}: {exc}"
            emit({"stage": "failed", "error": str(exc)})
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
            if kind == "application/json" and "gzip" in self.headers.get("Accept-Encoding", ""):
                payload = gzip.compress(payload, compresslevel=1)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            if path == "/api/events":
                after = int(query.get("after", ["0"])[0])
                with lock:
                    return self.send({"events": [e for e in events if e["sequence"] > after], "latest": counter["id"]})
            if path == "/api/state":
                with lock:
                    response = dict(state)
                if response.get("result") and query.get("result", [None])[0] == response["result"]["id"]:
                    response["result"] = None
                response.update(brain=brain_info, topology=graph, training={k: training.get(k) for k in ("best_validation_nll", "updates", "seconds", "design")}, checkpoint=a.checkpoint, backend=a.backend)
                return self.send(response)
            if re.fullmatch(r"/graph/[a-f0-9]{64}\.bin", path):
                target_file = ROOT / "runs/topology" / path.split("/")[-1]
            elif re.fullmatch(r"/frames/[0-9]+-[0-9]+/(live|listen)/[0-9]+\.bin", path):
                target_file = COMPOSITIONS / path.removeprefix("/frames/")
            elif re.fullmatch(r"/output/[0-9]+-[0-9]+/(draft|final)\.(mid|wav)", path):
                target_file = COMPOSITIONS / path.removeprefix("/output/")
            elif path in ("/", "/musician.js", "/brain_scan.js", "/style.css", "/musician.css"):
                target_file = ROOT / "web" / ("musician.html" if path == "/" else path[1:])
            else:
                return self.send({"error": "Not found"}, 404)
            if not target_file.is_file():
                return self.send({"error": "Not found"}, 404)
            return self.send(target_file.read_bytes(), kind=mimetypes.guess_type(target_file)[0] or "application/octet-stream")

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                return self.send({"error": "Origin rejected"}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if self.path == "/api/compose":
                    mood = str(request.get("mood", ""))[:300]
                    bars = int(request.get("bars", 16))
                    seed = int(request.get("seed", 17))
                    futures = int(request.get("futures", 8))
                    edits = int(request.get("edits", 2))
                    tempo = int(request.get("tempo", 100))
                    key = int(request.get("key", 0))
                    render = bool(request.get("render", True))
                    temperature = float(request.get("temperature", 0.9))
                    top = int(request.get("top", 0))
                    if not (4 <= bars <= 128 and 0 <= seed < 2**32 and 2 <= futures <= 32 and 0 <= edits <= 8 and 40 <= tempo <= 200 and -11 <= key <= 11 and 0.3 <= temperature <= 1.5 and 0 <= top <= 73):
                        raise ValueError("Out of range")
                    with lock:
                        if state["busy"]:
                            return self.send({"error": "A composition is already in progress"}, 409)
                        state.update(busy=True, error=None, progress={"stage": "starting"})
                    worker.submit(job, mood, bars, seed, futures, edits, tempo, key, render, temperature, top)
                    return self.send({"started": True}, 202)
                return self.send({"error": "Not found"}, 404)
            except (ValueError, KeyError, TypeError) as exc:
                return self.send({"error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Cadence musician studio: http://127.0.0.1:{a.port} · {graph['neurons']:,} neurons · {graph['synapses']:,} of {graph['synapses_total']:,} synapses drawn", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.shutdown(wait=True)


if __name__ == "__main__":
    main()
