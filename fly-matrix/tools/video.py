#!/usr/bin/env python3
"""A one-minute tour of the page for social media: the whole page, the nervous system expanded,
the close-up as the main view, the compound eye full screen, the room, a closing title with the
link. Recorded with Playwright's screencast at 1920 by 1080 and encoded for X with ffmpeg.

    python tools/video.py --url http://127.0.0.1:8813/ --out runs/video/fly-matrix.mp4
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

TITLE = "A FLY IN THE MATRIX"
LINE2 = "150,802 neurons wired as measured, flying a body with physics, learning which smell means sugar"
LINK = "floatingpragma.io/cadence-examples/fly-matrix"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8813/"); ap.add_argument("--out", default="runs/video/fly-matrix.mp4"); ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    vdir = out.parent / "raw"; shutil.rmtree(vdir, ignore_errors=True); vdir.mkdir()
    with sync_playwright() as p:
        b = p.chromium.launch(headless=False, args=["--use-angle=metal", f"--window-size={args.w},{args.h}", "--window-position=0,0"])
        ctx = b.new_context(viewport={"width": args.w, "height": args.h}, device_scale_factor=1, record_video_dir=str(vdir), record_video_size={"width": args.w, "height": args.h})
        pg = ctx.new_page()
        pg.goto(f"{args.url}?seed={args.seed}&dpr=1")
        for _ in range(120):
            if pg.evaluate("window.__app && window.__app.S && window.__app.S.ready"):
                break
            time.sleep(0.5)
        t0 = time.time()
        def until(t: float) -> None:
            while time.time() - t0 < t:
                time.sleep(0.2)
        pg.evaluate("window.__app.S.speed = 0.5")
        until(15)                                             # the whole page
        pg.click("#brain-big"); until(26)                     # the nervous system, every region labelled
        pg.click("#brain-big"); time.sleep(0.6)
        pg.click("#inset"); until(40)                         # the close-up as the main view
        pg.click("#inset"); time.sleep(0.4)
        pg.evaluate("window.__app.setEyeMain(true)"); until(49)   # the compound eye as the whole screen
        pg.evaluate("window.__app.setEyeMain(false); window.__app.setCam('room')"); until(55)
        pg.evaluate("window.__app.setCam('follow')")
        pg.evaluate(f"""() => {{ const d = document.createElement('div'); d.id = 'endcard';
          d.style.cssText = 'position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;background:rgba(0,0,0,.72);z-index:50;opacity:0;transition:opacity 1.2s';
          d.innerHTML = `<div style="font:400 64px/1 \\'Share Tech Mono\\',monospace;letter-spacing:.18em;color:#39ff6a;text-shadow:0 0 24px rgba(57,255,106,.7)">{TITLE}</div>
            <div style="font:16px/1.5 \\'IBM Plex Mono\\',monospace;color:#c8ffd8;max-width:900px;text-align:center">{LINE2}</div>
            <div style="font:600 26px/1 \\'IBM Plex Mono\\',monospace;color:#e8fff0;margin-top:10px;text-shadow:0 0 16px rgba(57,255,106,.6)">{LINK}</div>
            <div style="font:13px/1 \\'IBM Plex Mono\\',monospace;color:#7fe39a;letter-spacing:.14em;text-transform:uppercase;margin-top:6px">powered by Cadence</div>`;
          document.body.appendChild(d); requestAnimationFrame(() => {{ d.style.opacity = '1'; }}); }}""")
        until(66)
        pg.close(); ctx.close(); time.sleep(1.0); b.close()   # the page first, so the screencast is flushed to its file
    webm = sorted(glob.glob(str(vdir / "*.webm")), key=os.path.getmtime)[-1]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", webm, "-ss", "2.0", "-vf", f"fps=30,scale={args.w}:{args.h},fade=t=in:st=0:d=0.8,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-movflags", "+faststart", "-an", str(out)], check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,r_frame_rate", "-of", "default=nw=1", str(out)], capture_output=True, text=True).stdout.replace("\n", " ")
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB {probe}")


if __name__ == "__main__":
    main()
