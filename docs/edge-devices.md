# Edge devices: architectures, requirements, provisioning

## What was verified, and what was not
| Claim | Status |
|---|---|
| Edge image builds for `linux/arm64` | **Verified** (native, Docker Desktop on Apple M4) |
| Edge image builds for `linux/amd64` | **Verified** by building under QEMU emulation on the arm64 host (`docker buildx build --platform linux/amd64 --target edge`); hash-verified install from `uv.lock` succeeded |
| Model loads and infers inside both images with `--network none` | **Verified**: scores differ from the host by <= 1.8e-7 and all decisions are identical (`x86_64` and `aarch64`) |
| Performance on x86_64 | **Not measured** (emulation timings would be meaningless) |
| Raspberry Pi / mini-PC / industrial gateway | **Not verified on physical hardware.** The guide below is derived from the verified container behaviour and standard Linux practice |
| Control-plane and MLflow images for amd64 | Not built (only the edge image is needed on devices) |

## Architecture notes
* The edge runtime is **pure Python + NumPy + safetensors** (no PyTorch, no ONNX, no native code of our own), so it runs wherever the NumPy wheel exists: `manylinux`/`aarch64` and `x86_64` wheels. 32-bit ARM (armv7) is **not tested** and not covered by the lock (the build targets 64-bit only).
* Base image `python:3.12-slim` pinned by digest; digests refer to the multi-arch index, so the same Dockerfile builds both architectures.
* Build both: `docker buildx build --platform linux/amd64,linux/arm64 --target edge -t <registry>/fleetops-edge:<tag> --push .` (needs a registry; a multi-platform image cannot be `--load`ed locally). Not done here.
* The measured benchmark of the inference runtimes (NumPy vs ONNX) is in `docs/edge-inference-benchmarks.md`.

## Resource requirements (measured in Docker on Apple M4; treat as a lower bound for a Pi-class CPU)
| Resource | Measured |
|---|---|
| Image size | ~71 MB (amd64 and arm64) |
| Memory (agent, idle to 1.2k req/s) | 64-66 MB RSS |
| CPU | one core saturates at ~1.2k `/predict` req/s; ~53% of a core at 550 req/s; heartbeat/telemetry overhead is small |
| Disk | model bundle 2.4 KB here; the agent keeps active + previous + staging and refuses a download unless free space >= `size * 2 + 128 MB` (`FLEETOPS_EDGE_MIN_FREE_DISK_MB`, `..._STORAGE_HEADROOM_FACTOR`) |
| Network | MQTT (1883 plaintext dev / 8883 TLS) to the broker and HTTPS to the control plane for artifact download; inference itself needs no network |

Suggested starting point for a device: 1 core, 256 MB RAM, 1 GB free disk. These are engineering guesses from the numbers above, not tested minimums.

## Provisioning a device
1. Register it: `POST /api/v1/devices/register {"device_id": "...", "group": "..."}` (admin). The response contains the **one-time device credential**; store it as a file readable only by the agent user (`FLEETOPS_EDGE_DEVICE_SECRET` via secrets file, never in an image or environment dump).
2. For TLS, give the device the **CA certificate only** (`FLEETOPS_EDGE_CONTROL_PLANE_CA`, `FLEETOPS_EDGE_MQTT_TLS_CA`) and set `FLEETOPS_EDGE_REQUIRE_TLS=true`; hostnames in the URLs must match the certificate SANs. With mTLS also supply `..._MQTT_TLS_CERT/KEY`.
3. Run the container with a persistent volume at `/data` (model store), `--read-only` root filesystem, `--cap-drop ALL`, non-root user (the image uses uid 10001), and the settings: `FLEETOPS_EDGE_DEVICE_ID`, `FLEETOPS_EDGE_MQTT_HOST`, `FLEETOPS_EDGE_CONTROL_PLANE_URL`, `FLEETOPS_EDGE_API_TOKEN` (protects `/predict`), `FLEETOPS_EDGE_METRICS_TOKEN`. **Never** set `FLEETOPS_EDGE_ALLOW_FAULTS` or the simulated sensor on a real device.
4. Confirm: `curl -H "Authorization: Bearer <api token>" http://<device>:9100/healthz`; the device appears ONLINE in `GET /api/v1/devices`; deploy a model.
5. Reference (works on any Docker host, shown for orientation): see the `edge-*` services in `compose.yaml` and `deploy/k3s/fleetops.yaml`.

### Credential and certificate rotation on a device
* Device credential: `POST /devices/{id}/revoke` bumps the key epoch (old credentials stop working immediately), then re-register/re-provision the device with the new secret and restart the agent.
* Certificates: replace the CA bundle file, restart the agent container; see `docs/security.md` (rotation) and `docs/runbooks/tls-certificate-failure.md`.

## Raspberry Pi / mini-PC guide — **UNVERIFIED on physical hardware**
Target: Raspberry Pi 4/5 (64-bit OS, aarch64) or an x86_64 mini-PC running a 64-bit Linux with Docker Engine.
1. Install a 64-bit OS and Docker Engine; check `uname -m` is `aarch64` or `x86_64`. Sync the clock (`timedatectl`/chrony): commands carry expiry and the agent rejects skewed clocks (`DeviceClockSkew` alert).
2. Pull the multi-arch image you pushed to your registry (or `docker load` an exported image for an air-gapped site).
3. Create `/etc/fleetops/` (mode 0700) with the device secret, API/metrics tokens and CA files (mode 0400/0444); run the container with those files mounted read-only at `/run/secrets`, the settings above, `--restart unless-stopped`, `--read-only`, `--tmpfs /tmp`, a named volume at `/data`, and the port bound to the plant-network interface only.
4. SD cards wear out: put `/data` on SSD/industrial storage if possible, and keep logging at INFO. Power loss during a model update is covered by the staging design (partial downloads are removed at start; the active model is never modified in place), which is tested in software but **not** with real power cuts.
5. Expect to tune: CPU frequency scaling and thermal throttling on a Pi can change latency by an unknown factor; measure with `scripts/benchmarks/` on the device before committing to rates.

## Offline behaviour (verified in tests and in containers)
With no network at all the agent keeps serving the last known-good model; telemetry/predictions are buffered (bounded) and flushed on reconnect; new models cannot arrive until the link returns. See `docs/runbooks/device-offline.md`.
