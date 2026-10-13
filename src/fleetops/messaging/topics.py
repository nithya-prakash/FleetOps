"""MQTT topic convention: ``fleetops/devices/{device_id}/{channel}``."""

from __future__ import annotations

import re
from enum import StrEnum

from fleetops.domain.models import DEVICE_ID_PATTERN

ROOT = "fleetops/devices"
_DEVICE_RE = re.compile(DEVICE_ID_PATTERN)


class Channel(StrEnum):
    HEARTBEAT = "heartbeat"
    TELEMETRY = "telemetry"
    PREDICTIONS = "predictions"
    COMMANDS = "commands"
    STATUS = "status"


# (qos, retained) policy per channel when *publishing*.
QOS = {
    Channel.HEARTBEAT: 0,
    Channel.TELEMETRY: 0,
    Channel.PREDICTIONS: 1,
    Channel.STATUS: 1,
    Channel.COMMANDS: 1,
}
RETAIN = {Channel.STATUS: True}

DEVICE_UPLINK = (Channel.HEARTBEAT, Channel.TELEMETRY, Channel.PREDICTIONS, Channel.STATUS)


def topic(device_id: str, channel: Channel) -> str:
    if not _DEVICE_RE.match(device_id):
        raise ValueError(f"invalid device id {device_id!r}")
    return f"{ROOT}/{device_id}/{channel.value}"


def uplink_wildcard(channel: Channel) -> str:
    return f"{ROOT}/+/{channel.value}"


def parse_topic(value: str) -> tuple[str, Channel] | None:
    """Return ``(device_id, channel)`` or None if the topic is not a FleetOps device topic."""
    parts = value.split("/")
    if len(parts) != 4 or "/".join(parts[:2]) != ROOT:
        return None
    device_id, raw = parts[2], parts[3]
    if not _DEVICE_RE.match(device_id):
        return None
    try:
        return device_id, Channel(raw)
    except ValueError:
        return None


def topic_matches(pattern: str, value: str) -> bool:
    """MQTT wildcard match (``+`` single level, ``#`` multi-level suffix)."""
    p, v = pattern.split("/"), value.split("/")
    for i, seg in enumerate(p):
        if seg == "#":
            return True
        if i >= len(v):
            return False
        if seg != "+" and seg != v[i]:
            return False
    return len(p) == len(v)
