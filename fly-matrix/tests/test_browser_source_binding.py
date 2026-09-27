"""Fail-closed served-byte binding checks, with tiny fake network events only."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from browser_source_binding import ServedSourceBinding, WEB_SOURCES, source_hashes


BASE_URL = "http://localhost:8817/demo/"
PAYLOAD = "web/data/brain_full.json"


class Response:
    def __init__(self, url, body, status=200, *, cache_evicted=False):
        self.url, self.data, self.status = url, body, status
        self.ok = 200 <= status < 300
        self.cache_evicted = cache_evicted
        self.disposed = False
        self.headers = {"content-type": "application/json", "etag": '"fixture"',
                        "content-encoding": "gzip", "transfer-encoding": "chunked",
                        "content-length": "999"}

    def body(self):
        if self.cache_evicted:
            raise RuntimeError("Request content was evicted from inspector cache")
        return self.data

    def dispose(self):
        self.disposed = True


class Request:
    def __init__(self, response):
        self.url = response.url
        self.received = response
        self.failure = "net::ERR_ABORTED"

    def response(self):
        return self.received


class Context:
    def __init__(self):
        self.handlers = {}

    def on(self, event, handler):
        self.handlers[event] = handler

    def route(self, pattern, handler):
        self.pattern, self.route_handler = pattern, handler

    def finish(self, request):
        self.handlers["requestfinished"](request)


class Route:
    def __init__(self, context, body, *, status=200, finish=True, fail_fulfill=False):
        self.context = context
        url = BASE_URL + "data/brain_full.json?version=fixture"
        # API fetch and browser response are different objects. The browser
        # cache cannot provide this body, just as in the actual 42.9 MB case.
        self.fetched = Response(url, body, status)
        self.request = Request(Response(url, body, status, cache_evicted=True))
        self.finish_during_fulfill, self.fail_fulfill = finish, fail_fulfill
        self.aborted = False
        self.delivered = None
        self.fetch_options = None

    def fetch(self, **options):
        self.fetch_options = options
        return self.fetched

    def fulfill(self, **response):
        if self.fail_fulfill:
            raise RuntimeError("fulfill failed before delivery")
        self.delivered = response
        if self.finish_during_fulfill:
            # Exercise the race where requestfinished arrives while the
            # synchronous Playwright fulfill operation yields to the browser.
            self.context.finish(self.request)

    def abort(self):
        self.aborted = True


@pytest.fixture
def fixture():
    bodies = {name: name.encode() for name in WEB_SOURCES}
    bodies[PAYLOAD] = b'{"n":150802,"edges":1877099,"weight":1}'
    pins = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
    context = Context()
    binding = ServedSourceBinding(context, BASE_URL, pins)
    return context, binding, bodies


def serve_assets(context, bodies, *, omit=()):
    for name, body in bodies.items():
        if name == PAYLOAD or name in omit:
            continue
        url = BASE_URL + name.removeprefix("web/")
        if name == "web/index.html":
            url = BASE_URL + "?noscan=1"
        context.finish(Request(Response(url, body)))


def serve_payload(context, bodies, **kwargs):
    route = Route(context, bodies[PAYLOAD], **kwargs)
    assert context.pattern.fullmatch(route.request.url)
    context.route_handler(route)
    return route


@pytest.mark.parametrize("finish_before_return", [False, True])
def test_exact_delivery_requires_completed_request_in_either_event_order(fixture, finish_before_return):
    context, binding, bodies = fixture
    serve_assets(context, bodies)
    route = serve_payload(context, bodies, finish=finish_before_return)
    if not finish_before_return:
        with pytest.raises(AssertionError, match="unobserved browser response"):
            binding.receipt()
        context.finish(route.request)
    receipt = binding.receipt()
    assert set(receipt["assets"]) == set(WEB_SOURCES)
    assert route.delivered["body"] is route.fetched.data
    assert route.fetch_options == {"timeout": 30000, "max_redirects": 0}
    assert route.fetched.disposed
    headers = route.delivered["headers"]
    assert headers["content-type"] == "application/json" and headers["etag"] == '"fixture"'
    assert "content-encoding" not in headers and "transfer-encoding" not in headers
    assert int(headers["content-length"]) == len(bodies[PAYLOAD])
    assert receipt["assets"][PAYLOAD]["responses"] == 1
    assert "intercepted" in receipt["instrumentation"]


def test_stale_payload_with_same_dimensions_and_byte_count_is_rejected_permanently(fixture):
    context, binding, bodies = fixture
    serve_assets(context, bodies)
    stale = bodies[PAYLOAD].replace(b'"weight":1', b'"weight":2')
    assert len(stale) == len(bodies[PAYLOAD])
    assert {k: json.loads(stale)[k] for k in ("n", "edges")} == {
        k: json.loads(bodies[PAYLOAD])[k] for k in ("n", "edges")}
    route = Route(context, stale)
    context.route_handler(route)
    assert route.aborted and route.delivered is None and route.fetched.disposed
    serve_payload(context, bodies)  # A later correct response must not erase the mismatch.
    with pytest.raises(AssertionError, match="served SHA256.*differs from frozen"):
        binding.receipt()


def test_failed_fulfill_does_not_authorize_actually_undelivered_bytes(fixture):
    context, binding, bodies = fixture
    serve_assets(context, bodies)
    route = serve_payload(context, bodies, fail_fulfill=True)
    assert route.aborted and route.delivered is None
    with pytest.raises(AssertionError, match="fulfill failed before delivery"):
        binding.receipt()


def test_missing_completion_on_a_repeat_is_not_hidden_by_prior_valid_delivery(fixture):
    context, binding, bodies = fixture
    serve_assets(context, bodies)
    serve_payload(context, bodies)
    assert binding.receipt()
    serve_payload(context, bodies, finish=False)
    with pytest.raises(AssertionError, match="has not completed verified delivery"):
        binding.receipt()


def test_browser_response_cannot_bypass_the_verified_payload_route(fixture):
    context, binding, bodies = fixture
    serve_assets(context, bodies)
    context.finish(Request(Response(BASE_URL + "data/brain_full.json", bodies[PAYLOAD])))
    with pytest.raises(AssertionError, match="bypassed the verified delivery route"):
        binding.receipt()


@pytest.mark.parametrize("name", ["web/senses.js", "web/neural-state.js", "web/index.html",
                                  "web/retina.js", "web/neural-policy.js", "web/goal-life.js", "web/settlement-progress.js"])
def test_wrong_other_asset_fails_even_after_an_initial_match(fixture, name):
    context, binding, bodies = fixture
    serve_assets(context, bodies)
    serve_payload(context, bodies)
    context.finish(Request(Response(BASE_URL + name.removeprefix("web/") + "?old=1", b"stale")))
    with pytest.raises(AssertionError, match="served SHA256.*differs from frozen"):
        binding.receipt()


def test_missing_asset_reaches_bounded_failure_without_a_browser(fixture):
    context, binding, bodies = fixture
    serve_assets(context, bodies, omit=("web/neural-state.js",))
    serve_payload(context, bodies)
    # No browser wait is necessary when the declared deadline has already expired.
    with pytest.raises(AssertionError, match="unobserved browser response bodies: web/neural-state.js"):
        binding.wait_for_complete(None, timeout_ms=0)


@pytest.mark.parametrize("status", [302, 404, 500])
def test_payload_redirect_and_error_status_fail_before_delivery(fixture, status):
    context, binding, bodies = fixture
    serve_assets(context, bodies)
    route = serve_payload(context, bodies, status=status)
    assert route.aborted and route.delivered is None
    with pytest.raises(AssertionError, match=f"HTTP status {status}"):
        binding.receipt()


def test_unknown_successful_local_dependency_requires_a_pin(fixture):
    context, binding, bodies = fixture
    serve_assets(context, bodies)
    serve_payload(context, bodies)
    context.finish(Request(Response(BASE_URL + "new-controller.js", b"code")))
    with pytest.raises(AssertionError, match="unbound local response"):
        binding.receipt()


def test_missing_optional_checkpoint_and_external_dependencies_have_declared_scope(fixture):
    context, binding, bodies = fixture
    serve_assets(context, bodies)
    serve_payload(context, bodies)
    context.finish(Request(Response(BASE_URL + "data/learned.json", b"missing", 404)))
    context.finish(Request(Response("https://cdn.example/renderer.js", b"external")))
    assert "external CDN/font bytes are not pinned" in binding.receipt()["scope"]


def test_required_request_failure_is_fatal_after_prior_valid_responses(fixture):
    context, binding, bodies = fixture
    serve_assets(context, bodies)
    serve_payload(context, bodies)
    context.handlers["requestfailed"](Request(Response(BASE_URL + "worker.js", b"")))
    with pytest.raises(AssertionError, match="request failed"):
        binding.receipt()


def test_source_snapshot_covers_assets_helper_and_runner_and_detects_mutation(tmp_path):
    runner = "tools/check_assisted_page.py"
    names = (*WEB_SOURCES, "tools/browser_source_binding.py", runner)
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    frozen = source_hashes(tmp_path, runner)
    assert set(frozen) == set(names)
    (tmp_path / "web/senses.js").write_bytes(b"changed transduction")
    current = source_hashes(tmp_path, runner)
    assert current != frozen
    assert [name for name in names if current[name] != frozen[name]] == ["web/senses.js"]
