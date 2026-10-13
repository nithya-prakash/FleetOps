from __future__ import annotations

import json
import logging

import pytest

from fleetops.config.logging import JsonFormatter, configure_logging, log_context


def start(capsys: pytest.CaptureFixture[str]) -> None:
    # configured inside the test body: pytest's logging plugin restores root handlers after fixture setup
    configure_logging("unit-test", "INFO")
    capsys.readouterr()


def lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_json_lines_with_service_level_and_message(capsys: pytest.CaptureFixture[str]) -> None:
    start(capsys)
    logging.getLogger("fleetops.test").warning("hello %s", "world")
    (rec,) = lines(capsys)
    assert rec["service"] == "unit-test" and rec["level"] == "WARNING" and rec["msg"] == "hello world"
    assert rec["logger"] == "fleetops.test" and str(rec["ts"]).endswith("+00:00")


def test_correlation_ids_are_attached_and_scoped(capsys: pytest.CaptureFixture[str]) -> None:
    start(capsys)
    log = logging.getLogger("fleetops.test")
    with log_context(deployment_id="dep-1", device_id="edge-01"):
        log.info("inside")
        with log_context(command_id="cmd-9", device_id=None):  # None must not erase an outer value
            log.info("nested")
    log.info("outside")
    inside, nested, outside = lines(capsys)
    assert (inside["deployment_id"], inside["device_id"]) == ("dep-1", "edge-01") and "command_id" not in inside
    assert nested["command_id"] == "cmd-9" and nested["device_id"] == "edge-01"
    assert "deployment_id" not in outside and "device_id" not in outside


def test_extra_fields_and_exceptions_are_serialised(capsys: pytest.CaptureFixture[str]) -> None:
    start(capsys)
    log = logging.getLogger("fleetops.test")
    log.warning("clock", extra={"reason": "clock_skew", "skew_seconds": 41.5})
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("failed")
    skew, failed = lines(capsys)
    assert skew["reason"] == "clock_skew" and skew["skew_seconds"] == 41.5
    assert failed["level"] == "ERROR" and "ValueError: boom" in str(failed["exc"])


def test_unserialisable_extras_do_not_break_logging() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", (), None)
    record.payload = object()  # type: ignore[attr-defined]
    assert "payload" in json.loads(JsonFormatter("svc").format(record))  # default=str keeps the line valid JSON


def test_plain_text_mode_for_local_debugging(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("unit-test", "DEBUG", json_logs=False)
    capsys.readouterr()
    logging.getLogger("fleetops.test").debug("readable")
    out = capsys.readouterr().out
    assert "readable" in out and not out.lstrip().startswith("{")
