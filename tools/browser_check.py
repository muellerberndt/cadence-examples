"""Exercise the running private studio end to end, including real synthesis and export."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--existing",
    action="store_true",
    help="Inspect a completed run without composing again",
)
args = parser.parse_args()
with sync_playwright() as p:
    installed = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    print("Launching browser", flush=True)
    b = p.chromium.launch(
        executable_path=str(installed) if installed.exists() else None,
        args=["--use-angle=metal"] if installed.exists() else [],
        timeout=30000,
    )
    print("Browser ready", flush=True)
    page = b.new_page(viewport={"width": 1366, "height": 768})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8078")
    print("Studio loaded", flush=True)
    expect(page.locator("#owners")).to_contain_text("owners", timeout=20000)
    if not args.existing:
        page.locator("#prompt").fill("A bright heroic orchestral theme")
        page.locator("#seed").fill("23")
        page.locator("#compose").click()
    expect(page.locator("#compose")).to_be_enabled(timeout=300000)
    expect(page.locator("#review")).to_contain_text("Explicit score")
    print("Completed composition loaded", flush=True)
    page.locator("#final").click()
    page.wait_for_function(
        'document.querySelector("audio").currentTime>0', timeout=10000
    )
    assert abs(page.locator("audio").evaluate("el=>el.duration") - 60) < 0.1
    page.wait_for_function(
        "window.__composerDebug().origin?.startsWith('MIDI score readback')",
        timeout=15000,
    )
    graph = page.evaluate("window.__composerDebug().graph")
    assert (
        graph["owners"] == 2614
        and graph["seams"] == 1702938
        and graph["allEdgesSubmitted"]
    )
    assert page.evaluate("window.__composerDebug().glError") == 0
    print("Audio readback and complete graph verified", flush=True)
    page.locator("audio").evaluate("el=>el.pause()")
    for link in ["midi", "wav"]:
        assert page.request.get(
            "http://127.0.0.1:8078" + page.locator("#" + link).get_attribute("href")
        ).ok
    for key in ["brain", "waves", "score"]:
        bounds = page.locator("#" + key).bounding_box()
        assert bounds["y"] + bounds["height"] <= 768, (key, bounds)
    page.locator("#signal").select_option("mismatch")
    page.locator("#replay").click()
    page.wait_for_timeout(700)
    out = ROOT / "runs/browser"
    out.mkdir(exist_ok=True)
    page.screenshot(path=str(out / "desktop.png"))
    page.locator("#release").click()
    page.wait_for_timeout(200)
    expect(page.locator("#thought")).to_contain_text("Input released", timeout=15000)
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(200)
    assert not page.evaluate("document.documentElement.scrollWidth>innerWidth")
    page.screenshot(path=str(out / "mobile.png"), full_page=True)
    page.set_viewport_size({"width": 1366, "height": 768})
    page.locator("#like").click()
    expect(page.locator("#feedback-status")).to_contain_text(
        "validation loss", timeout=30000
    )
    expect(page.locator("#signal")).to_have_value("plasticity")
    assert not errors, errors
    print(
        json.dumps(
            {
                "desktop": True,
                "mobile": True,
                "real_composition_and_audio": True,
                "used_existing_composition": args.existing,
                "midi_export": True,
                "errors": errors,
            }
        )
    )
    b.close()
