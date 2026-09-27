"""Bind browser assays to local source files AND the bytes actually served.

The context listener includes dedicated-worker traffic. Small bodies are hashed
after requestfinished. The full connectome exceeds Chromium's inspector cache:
its actual intercepted request is fetched, checked and fulfilled with those same
bytes. External CDN/font bytes are outside this binding. No bodies are retained
in the receipt, and no independent verification fetch stands in for delivery.
"""

import hashlib
import re
import time
from urllib.parse import urljoin, urlsplit, urlunsplit


WEB_SOURCES = (
    "web/index.html", "web/page.js", "web/worker.js", "web/brain-console.js",
    "web/assisted-life.js", "web/demo-life.js", "web/goal-life.js", "web/target-evidence.js", "web/neural-policy.js",
    "web/retina.js", "web/life.js", "web/neural-state.js", "web/physics-clock.js", "web/worker-scheduling.js",
    "web/path-history.js", "web/path-view.js",
    "web/settlement-trace.js", "web/settlement-view.js", "web/neural-replay.js", "web/settlement-progress.js",
    "web/senses.js", "web/room.js",
    "web/body.js", "web/motor.js", "web/brain.js", "web/learner.js",
    "web/fly.js", "web/eye.js", "web/brain_scan.js",
    "web/data/brain_full.json", "web/data/lessons_full.json", "web/data/atlas.json",
)


def source_hashes(root, runner):
    """Freeze assets, this verifier and the calling assay before launching."""
    names = (*WEB_SOURCES, "tools/browser_source_binding.py", runner)
    hashes = {}
    for name in names:
        with (root / name).open("rb") as source:
            hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
    return hashes


def _url_key(url):
    """Query strings may select a view/cache version, never different bytes."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", "", ""))


class ServedSourceBinding:
    def __init__(self, context, page_url, frozen_sources):
        self.expected = {name: frozen_sources[name] for name in WEB_SOURCES}
        self.by_url = {
            _url_key(urljoin(page_url, name.removeprefix("web/"))): name
            for name in WEB_SOURCES
        }
        # A directory document and its explicit index.html are the same asset.
        self.by_url[_url_key(page_url)] = "web/index.html"
        self.origin = urlsplit(page_url)[:2]
        self.assets = {}
        self.errors = []
        self.routed = {}
        context.on("requestfinished", self._finished)
        context.on("requestfailed", self._failed)
        payload_url = _url_key(urljoin(page_url, "data/brain_full.json"))
        context.route(re.compile("^" + re.escape(payload_url) + r"(?:\?.*)?$"), self._route_payload)

    def _record(self, name, digest, size, url, method):
        item = self.assets.setdefault(name, {
            "sha256": digest, "bytes": size, "responses": 0, "urls": [], "binding": method,
        })
        item["responses"] += 1
        if url not in item["urls"]:
            item["urls"].append(url)

    def _digest(self, name, body, url):
        digest = hashlib.sha256(body).hexdigest()
        if digest != self.expected[name]:
            raise AssertionError(
                f"{name}: served SHA256 {digest} differs from frozen "
                f"{self.expected[name]} at {url}"
            )
        return digest

    def _record_routed(self, entry):
        # requestfinished can fire while the synchronous fulfill call yields.
        # Require both successful delivery and the actual request's completion.
        if entry["fulfilled"] and entry["completed"] and not entry["recorded"]:
            self._record("web/data/brain_full.json", entry["sha256"], entry["bytes"],
                         entry["url"], "intercepted request: fetch, hash, fulfill identical body")
            entry["recorded"] = True

    def _route_payload(self, route):
        response = None
        try:
            response = route.fetch(timeout=30000, max_redirects=0)
            if not response.ok:
                raise AssertionError(f"full payload HTTP status {response.status}")
            body = response.body()
            digest = self._digest("web/data/brain_full.json", body, response.url)
            entry = {"sha256": digest, "bytes": len(body), "url": response.url,
                     "fulfilled": False, "completed": False, "recorded": False}
            self.routed[route.request] = entry
            # APIResponse.body is decoded. Preserve the response metadata but
            # remove transfer/content encodings and set the decoded byte length.
            headers = {k: v for k, v in response.headers.items()
                       if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")}
            headers["content-length"] = str(len(body))
            route.fulfill(status=response.status, headers=headers, body=body)
            entry["fulfilled"] = True
            self._record_routed(entry)
        except Exception as error:
            self.errors.append(f"cannot deliver bound full payload: {error}")
            try:
                route.abort()
            except Exception:
                pass  # The recorded error is fatal even if the request already closed.
        finally:
            if response is not None:
                response.dispose()

    def _finished(self, request):
        name = self.by_url.get(_url_key(request.url))
        parts = urlsplit(request.url)
        try:
            response = request.response()
            if name is None:
                # A newly introduced local executable/data dependency must be
                # pinned explicitly. An absent optional checkpoint (404) is fine.
                if (parts[:2] == self.origin and
                        parts.path.endswith((".js", ".json", ".html")) and
                        response is not None and response.ok):
                    self.errors.append(f"unbound local response: {request.url}")
                return
            if response is None or not response.ok:
                status = None if response is None else response.status
                self.errors.append(f"{name}: HTTP status {status}")
                return
            if name == "web/data/brain_full.json":
                entry = self.routed.get(request)
                if entry is None:
                    raise AssertionError("full payload bypassed the verified delivery route")
                entry["completed"] = True
                self._record_routed(entry)
                return
            body = response.body()
            digest = self._digest(name, body, response.url)
            self._record(name, digest, len(body), response.url, "completed browser response body")
        except Exception as error:
            # Event callbacks must not hide retrieval failures or let a later
            # successful request erase evidence of stale/wrong served bytes.
            self.errors.append(f"cannot bind {name or request.url}: {error}")

    def _failed(self, request):
        name = self.by_url.get(_url_key(request.url))
        if name is not None:
            self.errors.append(f"{name}: request failed: {request.failure}")

    def _check_errors(self):
        if self.errors:
            raise AssertionError("served source binding failed: " + "; ".join(self.errors))

    def wait_for_complete(self, page, timeout_ms=30000):
        """Pump Playwright events for a bounded interval, failing closed."""
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            self._check_errors()
            missing = sorted(self.expected.keys() - self.assets.keys())
            if not missing:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError("unobserved browser response bodies: " + ", ".join(missing))
            page.wait_for_timeout(min(100, remaining * 1000))

    def receipt(self):
        self._check_errors()
        missing = sorted(self.expected.keys() - self.assets.keys())
        if missing:
            raise AssertionError("unobserved browser response bodies: " + ", ".join(missing))
        if any(not entry["recorded"] for entry in self.routed.values()):
            raise AssertionError("a routed full-payload request has not completed verified delivery")
        return {
            "scope": "Local page and dedicated-worker response bodies; external CDN/font bytes are not pinned.",
            "instrumentation": "Only brain_full.json is intercepted: route.fetch (30s, no redirects), SHA256 verification, then fulfill with the same decoded body; transfer/content encoding headers are removed and decoded content-length is set. Other local responses are read after download. Routing disables the context HTTP cache.",
            "assets": self.assets,
        }
