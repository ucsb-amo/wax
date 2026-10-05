"""TelemetryHub logs a failing provider once, not on every poll while the same
error persists (it used to compare against a differently formatted string and
log every poll), and logs recovery once.  No network, no Qt."""
import logging

from waxx.util.device_state.telemetry import Sample, TelemetryHub, TelemetryProvider


class Flaky(TelemetryProvider):
    name = "flaky"
    interval_s = 0.1

    def __init__(self):
        self.fail = None

    def poll(self):
        if self.fail:
            raise self.fail
        return {"x": Sample(1.0, age_s=0.0)}


def _lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "waxx.device_control.telemetry"]


def test_a_repeated_error_is_logged_once_and_recovery_once(caplog):
    caplog.set_level(logging.DEBUG, logger="waxx.device_control.telemetry")
    p = Flaky()
    hub = TelemetryHub([p])
    p.fail = ConnectionError("server unreachable")
    for _ in range(50):
        hub.poll_once("flaky")
    assert len(_lines(caplog)) == 1
    assert hub.status()["flaky"]["error"] == "ConnectionError: server unreachable"

    p.fail = TimeoutError("slow")              # a different error: logged
    hub.poll_once("flaky")
    hub.poll_once("flaky")
    assert len(_lines(caplog)) == 2

    p.fail = None
    for _ in range(5):
        hub.poll_once("flaky")
    lines = _lines(caplog)
    assert len(lines) == 3 and "recovered" in lines[-1]
    assert hub.status()["flaky"]["ok"]
