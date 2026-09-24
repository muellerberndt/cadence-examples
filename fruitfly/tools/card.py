#!/usr/bin/env python3
"""The social card: the fly alone in close-up, 1200 by 630, from the page's card mode.

    python tools/card.py --url http://127.0.0.1:8813/ --out web/card.jpg --wait 14
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8813/"); ap.add_argument("--out", default="web/card.jpg"); ap.add_argument("--wait", type=float, default=14); ap.add_argument("--seed", type=int, default=5); ap.add_argument("--d", type=float, default=0.0065); ap.add_argument("--quality", type=int, default=82); ap.add_argument("--mode", default="sit", help="sit: standing on the banana (card=sit); fly: in flight (card=1)"); ap.add_argument("--yaw", type=float, default=0.6); ap.add_argument("--pose", default="grooming")
    args = ap.parse_args()
    png = Path(args.out).with_suffix(".png")
    with sync_playwright() as p:
        b = p.chromium.launch(headless=False, args=["--use-angle=metal", "--window-size=1200,630"])
        pg = b.new_page(viewport={"width": 1200, "height": 630}, device_scale_factor=1)
        pg.goto(f"{args.url}?card={'sit' if args.mode == 'sit' else 1}&seed={args.seed}&d={args.d}&yaw={args.yaw}&pose={args.pose}")
        t0 = time.time()
        while time.time() - t0 < args.wait:
            time.sleep(0.5)
        pg.screenshot(path=str(png)); b.close()
    subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", str(args.quality), str(png), "--out", args.out], check=True, capture_output=True)
    png.unlink()
    print(f"{args.out}: {Path(args.out).stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
