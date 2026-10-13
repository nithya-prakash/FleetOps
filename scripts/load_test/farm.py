"""Device farm: N real EdgeAgent instances (real MQTT connection each, real artifact downloads, real inference).

    python -m scripts.load_test.farm --devices 100 --procs 4 --hb 2 --sensor-hz 1

Run `python scripts/provision.py --load-devices N` first (broker credentials) and restart Mosquitto. Devices are
registered through the admin API in group ``load``. Stop with Ctrl-C / SIGTERM.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import signal
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from fleetops.config.settings import EdgeSettings
from fleetops.edge.agent import EdgeAgent
from fleetops.messaging.bus import MqttBus, MqttConfig
from fleetops.security.device_identity import derive_device_secret
from scripts.load_test.common import CP_URL, ENV, MQTT_PORT, ROOT, TLS, VERIFY, secret_file


def device_ids(count: int, prefix: str = "load-") -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(1, count + 1)]


def register(ids: list[str], group: str = "load") -> None:
    key = ENV["FLEETOPS_ADMIN_API_KEY"]
    with httpx.Client(base_url=CP_URL, verify=VERIFY, timeout=30, headers={"Authorization": f"Bearer {key}"}) as client:

        def one(device_id: str) -> int:
            return client.post("/api/v1/devices/register", json={"device_id": device_id, "group": group}).status_code

        with ThreadPoolExecutor(max_workers=8) as pool:
            codes = list(pool.map(one, ids))
    bad = [c for c in codes if c not in (201, 409)]
    if bad:
        raise SystemExit(f"device registration failed: {set(bad)}")


class Farm:
    def __init__(self, ids: list[str], hb: float = 2.0, sensor_hz: float = 1.0, stagger: float = 0.01) -> None:
        self.ids, self.hb, self.sensor_hz, self.stagger = ids, hb, sensor_hz, stagger
        self.agents: list[EdgeAgent] = []
        self._root = Path(tempfile.mkdtemp(prefix="fleetops-farm-"))
        self._master = secret_file("fleetops_device_key_master")

    def start(self) -> None:
        ca = ROOT / "deploy" / "secrets" / "tls" / "ca.crt"
        for index, device_id in enumerate(self.ids):
            secret = derive_device_secret(self._master, device_id, 1)
            settings = EdgeSettings(
                _env_file=None,
                device_id=device_id,
                device_secret=secret,
                data_dir=self._root / device_id,
                control_plane_url=CP_URL,
                control_plane_ca=ca if TLS else None,
                mqtt_tls_ca=ca if TLS else None,
                heartbeat_interval_seconds=self.hb,
                telemetry_interval_seconds=10,
                prediction_flush_seconds=5,
                sensor_hz=self.sensor_hz,
                sensor_seed=index + 1,
                download_max_attempts=3,
            )
            holder: dict[str, EdgeAgent] = {}

            def announce(h: dict[str, EdgeAgent] = holder) -> None:
                h["agent"].announce_online()

            bus = MqttBus(
                MqttConfig(
                    host="127.0.0.1",
                    port=MQTT_PORT,
                    client_id=f"fleetops-{device_id}",
                    username=device_id,
                    password=secret,
                    tls_ca=ca if TLS else None,
                    keepalive=15,
                    reconnect_delay_max=5,
                ),
                will=EdgeAgent.will_for(device_id),
                on_connect=announce,
            )
            agent = EdgeAgent(settings, bus)
            holder["agent"] = agent
            agent.start()
            self.agents.append(agent)
            time.sleep(self.stagger)  # avoid a connection stampede at the broker

    def stop(self) -> None:
        for agent in self.agents:
            agent.stop()
        shutil.rmtree(self._root, ignore_errors=True)


def _worker(ids: list[str], hb: float, sensor_hz: float, stop: mp.synchronize.Event) -> None:
    farm = Farm(ids, hb, sensor_hz)
    farm.start()
    stop.wait()
    farm.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=10)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--hb", type=float, default=2.0, help="heartbeat interval seconds")
    ap.add_argument("--sensor-hz", type=float, default=1.0)
    ap.add_argument("--prefix", default="load-")
    args = ap.parse_args()
    ids = device_ids(args.devices, args.prefix)
    register(ids)
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    shards = [ids[i :: args.procs] for i in range(args.procs)]
    procs = [
        ctx.Process(target=_worker, args=(shard, args.hb, args.sensor_hz, stop), daemon=True)
        for shard in shards
        if shard
    ]
    for p in procs:
        p.start()
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    signal.signal(signal.SIGINT, lambda *_: done.set())
    print(f"farm running: {len(ids)} devices in {len(procs)} processes", flush=True)
    done.wait()
    stop.set()
    for p in procs:
        p.join(timeout=20)


if __name__ == "__main__":
    main()
