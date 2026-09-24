"""WavemeterController / WavemeterClient with the TCP link failing: the
latch trips on the first failed query, every query during the cooldown is
skipped without touching the socket, lock_status returns the failure value
0. immediately, and the controller reconnects after the cooldown.

MOGDevice's socket handling is replaced at ``reconnect`` / ``ask`` (what the
controller's overrides call through ``super()``), so nothing touches the
network.
"""

import pytest

from waxx.control.misc import moglabs_wavemeter as mw
from waxx.control.misc.moglabs import MOGDevice
from waxx.util.link_latch import LinkLatch


class FakeDevice:
    def __init__(self):
        self.up = True
        self.asks = []
        self.reconnects = 0
        self.channel = 2

    def reconnect(self, dev, timeout=1, check=True):
        self.reconnects += 1
        if not self.up:
            raise OSError("connect timed out")
        dev.dev = object()          # "a socket"
        if check:
            dev.info = dev.ask("info")

    def ask(self, dev, cmd):
        if not self.up:
            raise TimeoutError("timed out")
        self.asks.append(cmd)
        if cmd == "info":
            return "FZW"
        if cmd == "OPTSW,SET":
            return str(self.channel)
        if cmd.startswith("OPTSW,SET,"):
            self.channel = int(cmd.split(",")[-1])
            return "OK"
        if cmd == "MEAS,FREQ":
            return "391.016170 THz"
        if cmd == "MEAS,SAT":
            return "55"
        if cmd.startswith("OPTSW,EXP"):
            return "5 ms"
        return "OK"


class Clock:
    t = 0.

    def __call__(self):
        return self.t


@pytest.fixture
def fake(monkeypatch):
    f = FakeDevice()
    # Plain functions: a bound method stored on the class is not re-bound.
    monkeypatch.setattr(MOGDevice, "reconnect",
                        lambda dev, *a, **k: f.reconnect(dev, *a, **k))
    monkeypatch.setattr(MOGDevice, "ask",
                        lambda dev, *a, **k: f.ask(dev, *a, **k))
    monkeypatch.setattr(mw.time, "sleep", lambda t: None)
    monkeypatch.setattr(mw.WavemeterController, "_instances", {})
    return f


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ctrl(fake, clock):
    c = mw.WavemeterController("10.0.0.9")
    c.latch = LinkLatch("wavemeter test", retry_after=30., clock=clock)
    return c


@pytest.fixture
def client(ctrl):
    cl = mw.WavemeterClient(ch=2, target_freq=391.016170e12, wavemeter_device=ctrl,
                            locked_tolerance=60.e6)
    cl.key = "ry_test"
    return cl


def test_lock_status_when_up(client, fake):
    f = client.lock_status(frequency_shift=0., robust=True)
    assert f == pytest.approx(391.016170e12)
    assert "MEAS,FREQ" in fake.asks
    assert not client.link_down


def test_first_failure_trips_latch_and_drops_socket(ctrl, fake, capsys):
    fake.up = False
    assert ctrl.get_frequency(2) == 0.0        # existing failure value
    assert ctrl.link_down
    assert ctrl.dev is None
    assert "link down" in capsys.readouterr().out


def test_down_link_costs_no_socket_calls(client, ctrl, fake, capsys):
    fake.up = False
    client.lock_status(robust=True)            # trips
    capsys.readouterr()
    n_asks, n_reconnects = len(fake.asks), fake.reconnects
    assert client.lock_status(robust=True) == 0.
    assert client.get_frequency() == 0.
    assert len(fake.asks) == n_asks
    assert fake.reconnects == n_reconnects
    out = capsys.readouterr().out
    assert "link down" not in out              # printed once, at the trip
    assert "unlocked" not in out               # no verdict without a reading


def test_reconnects_after_cooldown(client, ctrl, fake, clock, capsys):
    fake.up = False
    client.lock_status()
    fake.up = True
    clock.t += 31.
    f = client.lock_status()
    assert f == pytest.approx(391.016170e12)
    assert fake.reconnects >= 2                 # construction + recovery
    assert not ctrl.link_down
    assert "link back" in capsys.readouterr().out


def test_failed_construction_does_not_poison_singleton(fake, monkeypatch):
    fake.up = False
    with pytest.raises(Exception):
        mw.WavemeterController("10.0.0.10")
    fake.up = True
    c = mw.WavemeterController("10.0.0.10")
    assert c._initialized
    assert not c.link_down


def test_dummies_report_link_up():
    assert mw.DummyWavemeterController.link_down is False
    assert mw.DummyWavemeterClient().link_down is False
