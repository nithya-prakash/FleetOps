"""Record the Grafana dashboard while the demo runs, as numbered PNG frames (then assembled into a GIF with ffmpeg).

Needs a throw-away environment, not a project dependency:
    python -m venv /tmp/pw && /tmp/pw/bin/pip install playwright
    /tmp/pw/bin/python scripts/demo_media/record_dashboard.py --out /tmp/frames      # uses the installed Google Chrome
Run against a freshly started stack (`make reset && make up`); it logs in with the generated Grafana admin password, opens
the FleetOps dashboard in kiosk mode, runs `scripts/demo.py all`, and screenshots the page every --interval seconds until
the demo has finished plus --tail seconds.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--tail", type=float, default=20.0)
    ap.add_argument("--port", default="13000")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    password = (ROOT / "deploy" / "secrets" / "grafana_admin_password").read_text().strip()
    base = f"http://127.0.0.1:{args.port}"
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(f"{base}/login")
        page.fill("input[name=user]", "admin")
        page.fill("input[name=password]", password)
        page.keyboard.press("Enter")
        page.wait_for_load_state("networkidle")
        page.goto(f"{base}/d/fleetops-overview/fleetops-overview?orgId=1&refresh=5s&from=now-5m&to=now&kiosk")
        page.wait_for_timeout(5000)
        demo = subprocess.Popen(
            [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "demo.py"), "all"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
        )
        start, n, done_at = time.monotonic(), 0, None
        while True:
            page.screenshot(path=str(args.out / f"frame_{n:04d}.png"))
            n += 1
            if demo.poll() is not None and done_at is None:
                done_at = time.monotonic()
            if done_at is not None and time.monotonic() - done_at > args.tail:
                break
            time.sleep(max(0.0, args.interval - 0.4))
        browser.close()
    print(f"{n} frames in {time.monotonic() - start:.0f}s, demo exit code {demo.returncode}")


if __name__ == "__main__":
    main()
