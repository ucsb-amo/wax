"""SDG6000X_CH with the LAN link failing: writes never raise, are reported,
and are re-sent once the link is back; reads raise; the latch makes a down
link cost no timeout per call.

The VXI-11 layer is replaced at ``vxi11.Instrument.write/ask`` (what the
driver's guarded overrides call through ``super()``), so nothing touches the
network. ``SDG6000X.__init__`` itself never connects (vxi11 opens lazily).
"""

import pytest
import vxi11

from waxx.control.misc import sdg6000x
from waxx.control.misc.sdg6000x import SDG6000X_CH, dv
from waxx.util.link_latch import LinkLatch, LinkDownError


class FakeLink:
    """Stands in for the instrument on the far end of the LAN."""
    def __init__(self):
        self.up = True
        self.writes = []
        self.asks = []
        self.frequency = 417.0e6
        self.amplitude = 0.64
        self.state = "ON"

    def write(self, instr, message, encoding='utf-8'):
        if not self.up:
            raise OSError("host unreachable")
        self.writes.append(message)
        if "BSWV FRQ," in message:
            self.frequency = float(message.split("FRQ,")[1])
        elif "BSWV AMP," in message:
            self.amplitude = float(message.split("AMP,")[1])
        elif ":OUTP " in message:
            self.state = message.split(":OUTP ")[1]

    def ask(self, instr, message, num=-1, encoding='utf-8'):
        if not self.up:
            raise OSError("host unreachable")
        self.asks.append(message)
        if message.endswith("BSWV?"):
            return f"C1:BSWV WVTP,SINE,FRQ,{self.frequency}HZ,AMP,{self.amplitude}V"
        if message.endswith("OUTP?"):
            return f"C1:OUTP {self.state},LOAD,50"
        raise AssertionError(message)


class Clock:
    t = 0.

    def __call__(self):
        return self.t


@pytest.fixture
def link(monkeypatch):
    fake = FakeLink()
    # Plain functions: a bound method stored on the class is not re-bound,
    # so the instance would arrive in the wrong slot.
    monkeypatch.setattr(vxi11.Instrument, "write",
                        lambda instr, *a, **k: fake.write(instr, *a, **k))
    monkeypatch.setattr(vxi11.Instrument, "ask",
                        lambda instr, *a, **k: fake.ask(instr, *a, **k))
    monkeypatch.setattr(sdg6000x.time, "sleep", lambda t: None)
    return fake


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ch(link, clock):
    c = SDG6000X_CH(ch=1, ip="10.0.0.1", frequency=417.0e6, amplitude_vpp=0.64,
                    max_amplitude_vpp=1., min_frequency=50.e6, max_frequency=499.e6)
    c._instr.latch = LinkLatch("siglent test", retry_after=30., clock=clock)
    return c


def test_write_succeeds_when_link_up(ch, link):
    ch.set_rpc(frequency=420.e6)
    assert link.writes == ["C1:BSWV FRQ,420000000.0"]
    assert ch._stale == set()
    ch.set_rpc(frequency=420.e6)          # unchanged -> no write
    assert len(link.writes) == 1


def test_failed_write_does_not_raise_and_is_reported(ch, link, capsys):
    link.up = False
    ch.set_rpc(frequency=420.e6)          # no exception
    out = capsys.readouterr().out
    assert "FAILED" in out and "NOT updated" in out
    assert "link down" in out
    assert 'frequency' in ch._stale
    assert ch._instr.link_down
    assert link.writes == []


def test_down_link_is_skipped_without_retrying_until_cooldown(ch, link, clock, capsys):
    link.up = False
    ch.set_rpc(frequency=420.e6)
    capsys.readouterr()
    # Latch tripped: the next request must not even reach the (fake) socket.
    calls_before = len(link.writes) + len(link.asks)
    with pytest.raises(LinkDownError):
        ch._instr.write("anything")
    assert len(link.writes) + len(link.asks) == calls_before
    ch.set_rpc(frequency=421.e6)          # reported, still no raise
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "Calls will be skipped" not in out   # the outage was printed once, earlier


def test_stale_setting_is_resent_when_link_returns(ch, link, clock, capsys):
    link.up = False
    ch.set_rpc(frequency=420.e6)
    link.up = True
    clock.t += 31.
    # Same value as the (failed) request: without the stale mark this would
    # be skipped as "unchanged" and the hardware would never get it.
    ch.set_rpc(frequency=420.e6)
    assert link.writes == ["C1:BSWV FRQ,420000000.0"]
    assert ch._stale == set()
    assert not ch._instr.link_down
    assert "link back" in capsys.readouterr().out


def test_read_raises_and_marks_all_stale(ch, link):
    link.up = False
    with pytest.raises(OSError):
        ch.get_frequency()
    assert ch._stale == {'frequency', 'amplitude', 'state'}
    with pytest.raises(LinkDownError):
        ch.fetch_state()                  # latched: immediate


def test_fetch_state_clears_stale(ch, link):
    ch._stale.update(('frequency', 'amplitude', 'state'))
    link.frequency = 430.e6
    assert ch.get_frequency() == 430.e6
    assert ch._stale == set()
    assert ch._p.state == 1


def test_sweep_with_unreadable_state_writes_nothing(ch, link, capsys):
    link.up = False
    ch.sweep_rpc(frequency_end=420.e6)
    assert link.writes == []
    assert "sweep" in capsys.readouterr().out
    ch.sweep_rpc(reset=True)              # latched path, still no raise
    assert link.writes == []


def test_sweep_steps_when_link_up(ch, link):
    link.frequency = 417.0e6
    ch.sweep_rpc(frequency_end=419.5e6, frequency_step=1.e6)
    freqs = [float(w.split("FRQ,")[1]) for w in link.writes]
    assert freqs == [418.0e6, 419.0e6, 419.5e6]


def test_sweep_stops_after_a_failed_step(ch, link, monkeypatch):
    link.frequency = 417.0e6
    steps = []

    def failing_write(instr, message, encoding='utf-8'):
        steps.append(message)
        if len(steps) == 2:
            link.up = False
        FakeLink.write(link, instr, message, encoding)

    monkeypatch.setattr(vxi11.Instrument, "write", failing_write)
    ch.sweep_rpc(frequency_end=422.e6, frequency_step=1.e6)
    assert len(steps) == 2                # second step failed, no further steps
    assert 'frequency' in ch._stale


def test_output_switch_failure_is_retried(ch, link, clock):
    link.up = False
    ch.set_output_rpc(state=0)
    assert 'state' in ch._stale
    assert ch._p.state == 1               # cache untouched on failure
    link.up = True
    clock.t += 31.
    ch.set_output_rpc(state=0)
    assert link.writes == ["C1:OUTP OFF"]
    assert ch._p.state == 0
    assert 'state' not in ch._stale


def test_instrument_timeouts_are_bounded():
    instr = sdg6000x.SDG6000X("10.0.0.2")
    assert instr.timeout == sdg6000x.T_TIMEOUT
    assert instr._connect_timeout == sdg6000x.T_TIMEOUT
    assert sdg6000x.T_TIMEOUT < 10.       # the python-vxi11 default
