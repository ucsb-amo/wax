"""A dashboard supervisor calls a server EXTERNAL only on a *fresh* beacon
that is not its own stopped child's, and drops EXTERNAL once the other
instance stops beaconing.

The discovery registry is a fake put in ``sys.modules``: importing the real
``beacon.discovery.client`` would bind UDP 50099 on this machine.  No process
is started.
"""
import os
import sys
import time
import types

import pytest
from PyQt6.QtWidgets import QApplication

from waxx.util.dashboard import server_supervisor as ss
from waxx.util.dashboard.server_supervisor import ServerSupervisor, SupervisorState


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def heard(monkeypatch):
    """``heard[server_id] = time.monotonic() stamp`` of its latest beacon."""
    seen: dict[str, float] = {}
    fake = types.ModuleType("beacon.discovery.client")

    def discover_entries(prefix="", max_age=None, collect_for=0.0):
        now = time.monotonic()
        return {sid: types.SimpleNamespace(last_seen=t) for sid, t in seen.items()
                if sid.startswith(prefix) and (max_age is None or now - t <= max_age)}

    fake.discover_entries = discover_entries
    monkeypatch.setitem(sys.modules, "beacon.discovery.client", fake)
    return seen


@pytest.fixture
def sup(qapp):
    s = ServerSupervisor("thing", ["unused"], requires_data_dir=False, beacon_id="thing")
    yield s
    s.deleteLater()


def test_only_a_fresh_beacon_counts(heard):
    now = time.monotonic()
    heard["thing"] = now - 5.0              # stopped long ago, entry kept forever
    assert not ss._beacon_seen("thing")
    heard["thing"] = now - 0.2
    assert ss._beacon_seen("thing")
    assert not ss._beacon_seen("thing", heard_after=now)   # heard before the stamp
    assert not ss._beacon_seen("other")


def test_own_stopped_childs_beacon_is_not_external(sup, heard):
    """16:02-16:03 on 2026-09-28: the interlock was stopped from the dashboard,
    then Start refused it as EXTERNAL on its own cached beacon."""
    t_exit = time.monotonic()
    heard["thing"] = t_exit - 0.1           # its last beacon, still fresh
    sup._child_exited_at = t_exit
    assert not sup.check_external()
    assert sup.state == SupervisorState.IDLE
    heard["thing"] = t_exit + ss._OWN_EXIT_MARGIN_S + 0.05   # a new instance beacons
    assert sup.check_external()
    assert sup.state == SupervisorState.EXTERNAL


def test_external_goes_idle_when_the_other_instance_stops(sup, heard):
    heard["thing"] = time.monotonic()
    assert sup.check_external()
    assert sup._external_timer.isActive()
    sup._recheck_external()                 # still beaconing
    assert sup.state == SupervisorState.EXTERNAL
    heard["thing"] = time.monotonic() - 10.0
    sup._recheck_external()
    assert sup.state == SupervisorState.IDLE
    assert not sup._external_timer.isActive()
