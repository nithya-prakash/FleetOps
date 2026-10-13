#!/usr/bin/env bash
# Print the registry digest of each pinned base image. To update a pin: run this, review the upstream release
# notes, edit the ARG/image line (Dockerfile, compose.yaml), rebuild, then run `make scan && make test`.
set -euo pipefail
for image in python:3.12-slim eclipse-mosquitto:2.0.22 prom/prometheus:v2.55.1 grafana/grafana:11.3.0 \
             ghcr.io/astral-sh/uv:0.12.5 aquasec/trivy:0.75.0; do
  printf '%-34s ' "$image"
  docker buildx imagetools inspect "$image" | awk '/^Digest:/ && !seen {print $2; seen=1}'
done
