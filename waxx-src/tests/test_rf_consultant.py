"""waxx RF consultant control (TPI-1005-A under ARTIQ): limits, the
link-failure policy, sweeps, the frame's promotion -- against a fake device,
no network -- and an offline compile of every kernel method (no core-device
connection; skipped without %db%)."""
from __future__ import annotations

import os

import pytest

import waxx.control.misc.rf_consultant as rfc
from beacon.tpi.rf_consultant_base import RfConsultantId
from waxx.control.misc.rf_consultant import RFConsultant, dv, dv_dbm


class FakeTpiDevice:
    """What RFConsultant uses of beacon's TpiDevice (Hz / dBm)."""

    def __init__(self, frequency=100.e6):
        self.frequency = frequency
        self.calls = []
        self.fail = None

    def _do(self, *call):
        self.calls.append(call)
        if self.fail:
            return {"ok": False, "error": self.fail}
        return {"ok": True}

    def set_frequency(self, frequency):
        r = self._do("freq", frequency)
        if r["ok"]:
            self.frequency = frequency
        return r

    def set_amplitude(self, dbm):
        return self._do("amplitude", dbm)

    def set_rf(self, on):
        return self._do("rf", on)


def _rf(**assign):
    ident = RfConsultantId("299", **assign)
    ident.key = "raman_rf"
    rf = RFConsultant(ident)
    rf._device = FakeTpiDevice()
    return rf


def test_writes_in_si_and_skips_unchanged_values():
    rf = _rf()
    rf.set_frequency_rpc(80.e6)
    rf.set_frequency_rpc(80.e6)
    rf.set_amplitude_rpc(-10.)
    rf.set_output_rpc(1)
    assert rf._device.calls == [("freq", 80.e6), ("amplitude", -10), ("rf", True)]
    rf.set_frequency_rpc(dv)
    rf.set_amplitude_rpc(dv_dbm)
    assert len(rf._device.calls) == 3


def test_an_amplitude_above_the_beacon_limit_is_refused_before_sending():
    rf = _rf(max_amplitude_dbm=-5)
    with pytest.raises(ValueError, match="above the configured maximum"):
        rf.set_amplitude_rpc(0.)
    assert rf._device.calls == []


def test_frequency_limits_come_from_the_beacon_assignment():
    rf = _rf(min_frequency=50.e6, max_frequency=600.e6)
    assert rf.min_frequency == 50.e6 and rf.max_frequency == 600.e6
    with pytest.raises(ValueError, match="outside"):
        rf.set_frequency_rpc(601.e6)
    assert rf._device.calls == []


def test_limit_only_tightens():
    rf = _rf(max_amplitude_dbm=-5)
    rf.limit(max_amplitude_dbm=3.)          # looser than beacon's: ignored
    assert rf.max_amplitude_dbm == -5.
    rf.limit(max_amplitude_dbm=-12., max_frequency=200.e6)
    assert rf.max_amplitude_dbm == -12. and rf.max_frequency == 200.e6
    with pytest.raises(ValueError):
        rf.set_amplitude_rpc(-10.)
    with pytest.raises(ValueError, match="outside"):
        rf.set_frequency_rpc(300.e6)


def test_a_tightened_limit_below_the_default_amplitude_is_an_error():
    rf = _rf(amplitude_dbm=-10)
    with pytest.raises(ValueError, match="default amplitude"):
        rf.limit(max_amplitude_dbm=-20.)


def test_a_failed_write_is_reported_and_retried_next_time(capsys):
    rf = _rf()
    rf._device.fail = "COM port COM7 was closed by pat@kong"
    rf.set_frequency_rpc(80.e6)
    out = capsys.readouterr().out
    assert "FAILED -- hardware NOT updated" in out and "pat@kong" in out
    rf._device.fail = None
    rf.set_frequency_rpc(80.e6)              # same value, but stale: written
    assert rf._device.calls[-1] == ("freq", 80.e6) and len(rf._device.calls) == 2


def test_device_not_found_does_not_raise(monkeypatch, capsys):
    import beacon.tpi.client as client
    monkeypatch.setattr(client, "find_device",
                        lambda name, collect_for=2.0: (_ for _ in ()).throw(LookupError("none")))
    rf = RFConsultant(RfConsultantId("299"))
    rf.set_output_rpc(1)
    assert "output ON FAILED" in capsys.readouterr().out and "state" in rf._stale


def test_sweep_steps_from_the_hardware_frequency(monkeypatch):
    monkeypatch.setattr(rfc.time, "sleep", lambda s: None)
    rf = _rf()
    rf._device.frequency = 100.e6
    rf.sweep_to_rpc(97.5e6, 1.e6, t_step=0.)
    assert [c[1] for c in rf._device.calls] == [99.e6, 98.e6, 97.5e6]


def test_sweep_without_an_end_goes_to_the_default_and_without_one_refuses(monkeypatch):
    monkeypatch.setattr(rfc.time, "sleep", lambda s: None)
    rf = _rf(frequency=102.e6)
    rf.sweep_to_rpc(dv, 1.e6, t_step=0.)
    assert rf._device.calls[-1] == ("freq", 102.e6)
    with pytest.raises(ValueError, match="no end frequency"):
        _rf().sweep_to_rpc(dv, 1.e6, t_step=0.)


def test_sweep_to_defaults_match_the_siglent_and_reset_goes_home(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rfc.time, "sleep", sleeps.append)
    rf = _rf(frequency=101.e6)
    rf._device.frequency = 100.e6
    rf.sweep_to_rpc(90.e6, reset=True)          # reset wins: to the default
    assert [c[1] for c in rf._device.calls] == [100.5e6, 101.e6]
    assert sleeps == [0.1]


def test_sweep_from_outside_the_limits_is_refused_with_the_reason(monkeypatch):
    monkeypatch.setattr(rfc.time, "sleep", lambda s: None)
    rf = _rf(max_frequency=700.e6)
    rf._device.frequency = 1000.e6
    with pytest.raises(ValueError, match="device is at 1000.0000 MHz, outside its limits"):
        rf.sweep_to_rpc(500.e6)
    assert rf._device.calls == []


def test_sweep_stops_at_the_first_failed_step(monkeypatch, capsys):
    monkeypatch.setattr(rfc.time, "sleep", lambda s: None)
    rf = _rf()
    rf._device.fail = "server away"
    rf.sweep_to_rpc(90.e6, 1.e6, t_step=0.)
    assert len(rf._device.calls) == 1


def test_init_walks_to_the_defaults_and_turns_the_output_on(monkeypatch):
    """As the Siglent's init: a walk, never a jump, then output on."""
    monkeypatch.setattr(rfc.time, "sleep", lambda s: None)
    rf = _rf(frequency=101.e6, amplitude_dbm=-12)
    rf._device.frequency = 100.e6
    rf.init_rpc()
    assert rf._device.calls == [("freq", 100.5e6), ("freq", 101.e6), ("amplitude", -12),
                                ("rf", True)]


def test_the_siglent_interface_refuses_a_vpp_amplitude():
    rf = _rf()
    with pytest.raises(ValueError, match="takes dBm"):
        rf.set_rpc(frequency=80.e6, amplitude=0.19)
    assert rf._device.calls == []
    rf.set_rpc(frequency=80.e6)
    assert rf._device.calls == [("freq", 80.e6)]
    assert rf.get_frequency() == 80.e6 and rf._frequency_default == dv


def test_frame_promotes_every_assignment(monkeypatch):
    import beacon.tpi.rf_consultant_id as beacon_id

    def init(self):
        self.setup()
        self.raman_rf = self.assign("299", name="Raman RF", max_amplitude_dbm=-5)
        self.antenna = self.assign("301")
        self.cleanup()
    monkeypatch.setattr(beacon_id.rf_consultant_frame, "__init__", init)
    from waxx.config.rf_consultant_id import rf_consultant_frame
    f = rf_consultant_frame(core=None)
    assert isinstance(f.raman_rf, RFConsultant) and f.raman_rf.key == "raman_rf"
    assert f.raman_rf.max_amplitude_dbm == -5. and f.antenna.serial == "301"
    assert {c.key for c in f.consultants()} == {"raman_rf", "antenna"}


@pytest.mark.skipif(not os.environ.get("db"), reason="needs the device db (%db%)")
def test_every_kernel_method_compiles():
    from artiq.experiment import EnvExperiment, kernel
    from artiq.language.environment import ProcessArgumentManager
    from artiq.master.databases import DeviceDB
    from artiq.master.worker_db import DeviceManager

    class Expt(EnvExperiment):
        def build(self):
            self.setattr_device("core")
            ident = RfConsultantId("299", frequency=80.e6)
            ident.key = "raman_rf"
            self.rf = RFConsultant(ident, core=self.core)
            self.f_end = 82.e6

        @kernel
        def run_kernel(self):
            self.rf.set_frequency(80.e6)
            self.rf.set_amplitude(-10.)
            self.rf.sweep_to(85.e6, frequency_step=0.5e6)
            self.rf.sweep_to()
            self.rf.sweep_to(reset=True)
            self.rf.sweep_to(self.f_end)
            self.rf.on()
            self.rf.off()
            self.rf.init()

    class _NoDatasets:
        def __getattr__(self, name):
            raise RuntimeError(name)

    dm = DeviceManager(DeviceDB(os.environ["db"]))
    try:
        exp = Expt((dm, _NoDatasets(), ProcessArgumentManager({}), {}))
        # Compile only -- never run (no core-device connection is made).
        exp.core.compile(Expt.run_kernel, [exp], {}, attribute_writeback=False,
                         print_as_rpc=False)
    finally:
        dm.close_devices()
