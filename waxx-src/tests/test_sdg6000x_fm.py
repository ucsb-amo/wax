"""SDG6000X_FM_CH against a fake SDG6022X: limits refuse before anything is
sent, every write is read back, FM writes wait for FM to be on, and the
carrier frequency is only ever written to put it back at its fixed value.

The fake models what the notebook (k-jam/jk/siglent_fm_control.ipynb) saw on
2026-11-05: FM parameter writes are ignored while modulation is off,
modulation comes on with the last-used type, and FM values are reported only
while FM is on. Nothing touches the network (vxi11 is patched at
``Instrument.write/ask``; ``SDG6000X.__init__`` never connects).
"""

import pytest
import vxi11

from waxx.control.misc import sdg6000x
from waxx.control.misc.sdg6000x import SDG6000X_FM_CH, SDG6000X_CH, dv
from waxx.util.link_latch import LinkLatch


class FakeSDG:
    def __init__(self):
        self.up = True
        self.writes = []
        self.carrier = 80.e6
        self.amp = 1.0
        self.output = "ON"
        self.mod_on = False
        self.mod_type = "AM"
        self.dev = 1.e3
        self.rate = 100.
        self.max_dev = 1.e12        # instrument clamp (set lower to test)

    def write(self, instr, message, encoding='utf-8'):
        if not self.up:
            raise OSError("host unreachable")
        self.writes.append(message)
        body = message.split(":", 1)[1]
        if body.startswith("BSWV FRQ,"):
            self.carrier = float(body.split(",")[1])
        elif body.startswith("BSWV AMP,"):
            self.amp = float(body.split(",")[1])
        elif body == "MDWV STATE,ON":
            self.mod_on = True
        elif body == "MDWV STATE,OFF":
            self.mod_on = False
        elif not self.mod_on:
            pass                                # ignored while modulation is off
        elif body == "MDWV FM":
            self.mod_type = "FM"
        elif body.startswith("MDWV FM,DEVI,") and self.mod_type == "FM":
            self.dev = min(float(body.split(",")[2]), self.max_dev)
        elif body.startswith("MDWV FM,FRQ,") and self.mod_type == "FM":
            self.rate = float(body.split(",")[2])

    def ask(self, instr, message, num=-1, encoding='utf-8'):
        if not self.up:
            raise OSError("host unreachable")
        if message.endswith("MDWV?"):
            if not self.mod_on:
                return "C1:MDWV STATE,OFF"
            if self.mod_type == "FM":
                return (f"C1:MDWV STATE,ON,FM,MDSP,SINE,SRC,INT,FRQ,{self.rate}HZ,"
                        f"DEVI,{self.dev}HZ,CARR,WVTP,SINE,FRQ,{self.carrier}HZ,AMP,{self.amp}V")
            return "C1:MDWV STATE,ON,AM,MDSP,SINE,SRC,INT,FRQ,100HZ,DEPTH,100"
        if message.endswith("BSWV?"):
            return (f"C1:BSWV WVTP,SINE,FRQ,{self.carrier}HZ,PERI,1.25e-08S,"
                    f"AMP,{self.amp}V,AMPVRMS,0.35Vrms,OFST,0V,PHSE,0")
        if message.endswith("OUTP?"):
            return f"C1:OUTP {self.output},LOAD,50,PLRT,NOR"
        raise AssertionError(message)


class Clock:
    t = 0.

    def __call__(self):
        return self.t


@pytest.fixture
def sdg(monkeypatch):
    fake = FakeSDG()
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
def fm(sdg, clock):
    c = SDG6000X_FM_CH(ch=1, ip="10.0.0.3", frequency_carrier=80.e6,
                       max_amplitude_vpp=2.0, fm_deviation=2.e6, fm_rate=100.e3,
                       max_fm_deviation=10.e6, max_fm_rate=1.e6)
    c._instr.latch = LinkLatch("siglent fm test", retry_after=30., clock=clock)
    return c


def test_constructor_refuses_defaults_out_of_limits(sdg):
    with pytest.raises(ValueError):
        SDG6000X_FM_CH(ch=1, ip="10.0.0.3", frequency_carrier=80.e6,
                       max_amplitude_vpp=2.0, fm_deviation=20.e6, max_fm_deviation=10.e6)


def test_init_turns_fm_on_from_am_and_keeps_amplitude(fm, sdg):
    fm.init_rpc()
    assert sdg.mod_on and sdg.mod_type == "FM"
    assert (sdg.dev, sdg.rate) == (2.e6, 100.e3)
    assert sdg.amp == 1.0                       # amplitude kept, not written
    assert not any("BSWV" in w for w in sdg.writes)
    assert not any("OUTP" in w for w in sdg.writes)   # output never switched
    assert fm.fm_state == 1 and fm._stale == set()


@pytest.mark.parametrize("amp", [2.0001, 5., float('nan')])
def test_amplitude_above_limit_sends_nothing(fm, sdg, amp):
    fm.init_rpc()
    n = len(sdg.writes)
    with pytest.raises(ValueError):
        fm.set_rpc(amplitude_vpp=amp, fm_deviation=1.e6)
    assert len(sdg.writes) == n                 # not even the valid deviation
    assert sdg.amp == 1.0


def test_fm_guards_send_nothing(fm, sdg):
    fm.init_rpc()
    n = len(sdg.writes)
    for kw in (dict(fm_deviation=10.1e6), dict(fm_rate=1.1e6)):
        with pytest.raises(ValueError):
            fm.set_rpc(**kw)
    assert len(sdg.writes) == n


def test_amplitude_at_limit_is_written_and_verified(fm, sdg):
    fm.init_rpc()
    fm.set_rpc(amplitude_vpp=2.0)
    assert sdg.amp == 2.0 and 'amplitude_vpp' not in fm._stale
    n = len(sdg.writes)
    fm.set_rpc(amplitude_vpp=2.0)               # unchanged -> no write
    assert len(sdg.writes) == n


def test_set_deviation_while_fm_on(fm, sdg):
    fm.init_rpc()
    fm.set_rpc(fm_deviation=500.e3, amplitude_vpp=0.5)
    assert sdg.dev == 500.e3 and sdg.amp == 0.5
    assert sdg.carrier == 80.e6
    assert fm._stale == set()


def test_fm_values_set_while_off_are_written_by_fm_on(fm, sdg):
    fm.init_rpc(fm_on=False)
    assert not sdg.mod_on
    n = len(sdg.writes)
    fm.set_rpc(fm_deviation=3.e6)
    assert len(sdg.writes) == n                 # ignored by the instrument: not sent
    fm.fm_on_rpc()
    assert sdg.mod_type == "FM" and sdg.dev == 3.e6


def test_init_restores_carrier_and_lowers_amplitude_over_limit(fm, sdg, capsys):
    sdg.carrier, sdg.amp = 79.e6, 3.0
    fm.init_rpc()
    assert sdg.carrier == 80.e6
    assert sdg.amp == 2.0
    assert "outside 0-2.0 Vpp" in capsys.readouterr().out


def test_clamped_deviation_is_reported_and_fm_switched_off(fm, sdg, capsys):
    sdg.max_dev = 1.e6
    fm.init_rpc()
    out = capsys.readouterr().out
    assert "does NOT hold" in out and "FM did not come up" in out
    assert not sdg.mod_on
    assert 'fm_deviation' in fm._stale


def test_unknown_fm_state_is_read_before_fm_write(fm, sdg):
    # FM already on by hand (e.g. from the notebook); no init() called.
    sdg.mod_on, sdg.mod_type = True, "FM"
    fm.set_rpc(fm_deviation=1.5e6)
    assert sdg.dev == 1.5e6 and fm.fm_state == 1


def test_link_down_never_raises_and_resends(fm, sdg, clock, capsys):
    fm.init_rpc()
    sdg.up = False
    fm.set_rpc(amplitude_vpp=0.7)
    assert "NOT updated" in capsys.readouterr().out
    assert 'amplitude_vpp' in fm._stale
    sdg.up = True
    clock.t += 31.
    fm.set_rpc(amplitude_vpp=0.7)               # same value: re-sent because stale
    assert sdg.amp == 0.7 and fm._stale == set()


def test_fm_off(fm, sdg):
    fm.init_rpc()
    fm.fm_off_rpc()
    assert not sdg.mod_on and fm.fm_state == 0


def test_base_channel_amplitude_limit_checks_the_request(sdg):
    # Regression: SDG6000X_CH used to check the previously cached amplitude,
    # so a request above the maximum was written.
    c = SDG6000X_CH(ch=1, ip="10.0.0.4", frequency=80.e6, amplitude_vpp=0.5,
                    max_amplitude_vpp=1.0, min_frequency=70.e6, max_frequency=90.e6)
    with pytest.raises(ValueError):
        c.set_rpc(frequency=81.e6, amplitude=1.5)
    assert sdg.writes == []                     # neither frequency nor amplitude sent
    c.set_rpc(amplitude=0.9)
    assert sdg.writes == ["C1:BSWV AMP,0.9"]
