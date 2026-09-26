"""The tweezer AWG connection driver (waxx.control.tweezer.awg_agent_driver)
against a fake controller: no card, no spcm calls."""
import pytest

from waxx.control.tweezer.awg_agent_driver import TweezerAwgDriver, write_static_tones


class FakeCore:
    def __init__(self, log, idx):
        self.log, self.idx = log, idx

    def amp(self, a):
        self.log.append(("amp", self.idx, a))


class FakeDDS:
    def __init__(self):
        self.log = []

    def __getitem__(self, idx):
        return FakeCore(self.log, idx)

    def exec_at_trg(self):
        self.log.append(("exec",))

    def write(self):
        self.log.append(("write",))


class FakeController:
    def __init__(self, close_result=True):
        self.card = None
        self.dds = FakeDDS()
        self.static = []
        self.inits = []
        self.closes = []
        self.close_result = close_result
        self.core_list = [hex(2 ** n) for n in range(20)]

    def awg_init(self, two_d=False, t_wait_in_use=30.):
        self.inits.append(t_wait_in_use)
        self.card = object()

    def close(self, timeout=None, when=""):
        self.closes.append(timeout)
        self.card = None
        return self.close_result

    def set_static_tweezers(self, freqs, amps, phases=None):
        self.static.append((list(freqs), list(amps), phases))


def _driver(**kw):
    return TweezerAwgDriver("TCPIP::192.168.1.83::inst0::INSTR", controller=FakeController(**kw),
                            t_wait_in_use=7., close_timeout=2.)


def test_removed_tones_are_zeroed():
    tw = FakeController()
    write_static_tones(tw, [[72.e6, 0.2], [73.e6, 0.2], [74.e6, 0.2]])
    assert tw._panel_n_tones == 3
    tw.dds.log.clear()
    assert write_static_tones(tw, [[72.e6, 0.3]]) == 1
    assert tw.static[-1] == ([72.e6], [0.3], None)
    assert ("amp", 1, 0.) in tw.dds.log and ("amp", 2, 0.) in tw.dds.log
    assert tw.dds.log[-2:] == [("exec",), ("write",)]
    assert tw._panel_traps == [[72.e6, 0.3]]


def test_all_zero_amplitudes_pass_explicit_phases():
    tw = FakeController()
    write_static_tones(tw, [[72.e6, 0.0], [73.e6, 0.0]])
    assert tw.static[-1][2] == [0., 0.]           # compute_tweezer_phases would divide by 0


def test_bad_tone_tables_are_refused():
    tw = FakeController()
    with pytest.raises(ValueError, match="sum to"):
        write_static_tones(tw, [[72.e6, 0.6], [73.e6, 0.6]])
    with pytest.raises(ValueError, match="negative"):
        write_static_tones(tw, [[72.e6, -0.1]])
    with pytest.raises(ValueError, match="DDS cores"):
        write_static_tones(tw, [[70.e6 + i * 1e5, 0.01] for i in range(21)])
    assert tw.static == []


def test_open_close_detail_and_commands():
    d = _driver()
    assert not d.is_open()
    with pytest.raises(RuntimeError, match="not open"):
        d.write_traps([[72.e6, 0.1]])
    d.open()
    assert d.tw.inits == [7.] and d.is_open()
    assert d.detail() == "192.168.1.83 · no tones loaded"
    assert d.write_traps([[72.e6, 0.2], [73.e6, 0.2]]) == {"tones": 2}
    assert d.detail() == "192.168.1.83 · 2 tone(s): 72.000, 73.000 MHz"
    assert d.close() is True and d.tw.closes == [2.]
    assert not d.is_open() and d.tw._panel_traps == []
    with pytest.raises(RuntimeError, match="not open"):
        d.force_trigger()
    assert set(d.COMMANDS) == {"write_traps", "force_trigger"}


def test_a_close_that_gave_up_is_reported():
    d = _driver(close_result=False)
    d.open()
    assert d.close() is False
