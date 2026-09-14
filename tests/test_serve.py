"""The local launcher must be usable without a browser or its default port."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "serve.py"
spec = importlib.util.spec_from_file_location("serve_examples", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("page,fragment", list(module.PAGES.items()))
def test_headless_server_uses_selected_port_and_loopback(
    page, fragment, monkeypatch, capsys
):
    class Server:
        server_address = ("127.0.0.1", 41234)

        def __init__(self, address, handler):
            assert address == ("127.0.0.1", 0)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def serve_forever(self):
            raise KeyboardInterrupt

    def unexpected_browser(*args):
        pytest.fail("headless serving opened a browser")

    monkeypatch.setattr(module.http.server, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(module.webbrowser, "open", unexpected_browser)
    assert module.main([page, "--port", "0", "--no-browser"]) == 0
    assert f"http://127.0.0.1:41234/{fragment}/" in capsys.readouterr().out


def test_server_bind_error_has_actionable_message(monkeypatch, capsys):
    def occupied(*args):
        raise OSError("Address already in use")

    monkeypatch.setattr(module.http.server, "ThreadingHTTPServer", occupied)
    with pytest.raises(SystemExit) as error:
        module.main(["--no-browser"])
    assert error.value.code == 1
    assert "Try --port 0" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args,code",
    [
        (["--help"], 0),
        (["--port", "-1"], 2),
        (["--port", "65536"], 2),
        (["missing"], 2),
        (["memory"], 2),
    ],
)
def test_invalid_or_help_arguments_do_not_start_a_server(args, code, monkeypatch):
    def unexpected_server(*args):
        pytest.fail("invalid arguments started a server")

    monkeypatch.setattr(module.http.server, "ThreadingHTTPServer", unexpected_server)
    with pytest.raises(SystemExit) as error:
        module.main(args)
    assert error.value.code == code
