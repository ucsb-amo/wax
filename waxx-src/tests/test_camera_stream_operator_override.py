"""Offline tests: a triggered camera stream as a run stream with an operator
override of one of its frames (beacon.camera.core, "Run streams"). The fake
stream substitutes the override into the stream's writes for that frame the
way the server does; no network.
"""
import json
import time
from types import SimpleNamespace

import pytest

from test_camera_stream_client import (FakeExpt, FakeTTL, TrigFakeStream, DEFAULTS, M,
                                       wait_settled)


class OverrideFakeStream(TrigFakeStream):
    """TrigFakeStream plus RUN_STREAM / operator state / per-frame writes."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.declared = None
        self.override = None            # {frame: {key: value}}
        self.op_rev = 0
        self.tagged = []                # (frame, values asked) per write naming a frame

    def declare_run_stream(self, frames, max_exposure_time=None):
        self.declared = (list(frames), dict(max_exposure_time or {}))
        self.op_rev += 1
        return self.operator_state()

    def operator_state(self):
        return {"rev": self.op_rev,
                "run_stream": None if self.declared is None else
                {"label": "test", "host": "here", "frames": self.declared[0],
                 "max_exposure_time": self.declared[1]},
                "override": None if not self.override else
                {"values": self.override, "by": "viewer", "host": "pc2", "at": "12:00:00"},
                "requested": {}}

    def operator(self, values):
        """The operator sets (or with None clears) the override."""
        self.override = values
        self.op_rev += 1

    def status(self, refresh=True):
        st = super().status(refresh)
        return SimpleNamespace(state=st.state, instance=st.instance,
                               raw={**st.raw, "operator_rev": self.op_rev})

    def set_values(self, values, frame=None):
        self.tagged.append((frame, dict(values)))
        values, subs = dict(values), []
        if self.override and frame is not None:
            for k, v in (self.override.get(frame) or {}).items():
                if k in values and values[k] != v:
                    values[k] = v
                    subs.append(k)
        out = self.set(**values)
        out["operator_substituted"] = subs
        return out


def make(keys, stream=None, **kw):
    from waxx.control.cameras.camera_stream_client import TriggeredCameraStreamClient
    fe = FakeExpt()
    stream = stream or OverrideFakeStream()
    kw.setdefault("t_match_window", 0.6)
    kw.setdefault("t_stray_grace", 0.1)
    cs = TriggeredCameraStreamClient(fe, "12345", keys, ttl=FakeTTL(), trigger_source="Line2",
                                     _stream=stream, _server_defaults=DEFAULTS, **kw)
    return fe, stream, cs


def shot(cs, stream, j, value, counter=None, ahead=0.3):
    if counter is not None:
        cs._expt.scan_xvars[0].counter = counter
    cs.announce_trigger_mu(int(ahead * 1e9), j)
    time.sleep(ahead / 2)
    stream.edge(value=value)
    wait_settled(cs)


def test_the_stream_declares_its_frames_and_each_frames_max():
    fe, stream, cs = make(["img_a", "img_b"], frame_max_exposure={"img_b": 5e-3})
    try:
        assert stream.declared == (["img_a", "img_b"], {"img_b": 5e-3})
        assert cs._run_stream and cs._base_settings == {"exposure_time": 19e-6, "gain": 0.0}
    finally:
        cs.finish()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_a"])
    assert rec["run_stream"] is True and rec["frame_max_exposure"] == {"img_b": 5e-3}
    assert rec["operator_overrides"] == [] and rec["operator_override_frames"] == {}


def test_own_exposures_above_a_frames_max_are_refused_in_prepare():
    with pytest.raises(ValueError, match="img_b exposure_time 0.002 s above"):
        make(["img_a", "img_b"], frame_max_exposure={"img_b": 1e-3},
             frame_settings={"img_b": {"exposure_time": 2e-3}})
    with pytest.raises(ValueError, match="img_a exposure_time 0.003 s above"):
        make(["img_a"], frame_max_exposure={"img_a": 1e-3}, exposure_time=3e-3)
    with pytest.raises(ValueError, match="no such frame"):
        make(["img_a"], frame_max_exposure={"img_zz": 1e-3})


def test_an_override_of_one_frame_reaches_its_next_frames_only(capsys):
    fe, stream, cs = make(["img_a", "img_b"], frame_max_exposure={"img_b": 5e-3})
    fe.data.init()
    try:
        shot(cs, stream, 0, 1, counter=0)
        shot(cs, stream, 1, 2)
        assert M(fe, "img_b", "gain")[0] == 0.0
        stream.operator({"img_b": {"exposure_time": 2e-3, "gain": 9.0}})
        time.sleep(0.4)                     # the worker reads the status block
        for counter in (1, 2):
            shot(cs, stream, 0, 1, counter=counter)
            shot(cs, stream, 1, 2)
        assert list(M(fe, "img_a", "gain")[:3]) == [0.0, 0.0, 0.0]
        assert list(M(fe, "img_a", "exposure")[:3]) == [19e-6] * 3
        assert list(M(fe, "img_b", "gain")[:3]) == [0.0, 9.0, 9.0]
        assert M(fe, "img_b", "exposure")[1] == 2e-3
        # the stream sent img_b's own values, naming it; the server put the override in
        assert ("img_b", {"exposure_time": 19e-6, "gain": 0.0}) in stream.tagged
        assert not cs.counts.get("frame_settings_late") and cs._n_missing() == 0
    finally:
        cs.finish()
    out = capsys.readouterr().out
    assert "!! operator override" in out and "OPERATOR OVERRIDE during this run" in out
    rec = json.loads(fe._extra_file_texts["camera_stream_img_b"])
    (change,) = rec["operator_overrides"]
    assert change["values"] == {"img_b": {"exposure_time": 2e-3, "gain": 9.0}}
    assert change["by"] == "viewer" and change["requests_before"] == 2
    assert rec["operator_override_frames"] == {"img_b": 2}


def test_an_override_of_another_streams_frame_is_ignored():
    fe, stream, cs = make(["img_a"])
    fe.data.init()
    try:
        stream.operator({"img_other": {"gain": 9.0}})
        time.sleep(0.4)
        shot(cs, stream, 0, 1, counter=0)
        assert M(fe, "img_a", "gain")[0] == 0.0
        assert cs.operator_overrides == []
    finally:
        cs.finish()


def test_a_cleared_override_puts_the_runs_own_settings_back():
    fe, stream, cs = make(["img_a"])
    fe.data.init()
    try:
        stream.operator({"img_a": {"gain": 6.0}})
        time.sleep(0.4)
        shot(cs, stream, 0, 1, counter=0)
        assert M(fe, "img_a", "gain")[0] == 6.0
        stream.operator(None)
        time.sleep(0.4)
        shot(cs, stream, 0, 2, counter=1)
        assert M(fe, "img_a", "gain")[1] == 0.0
    finally:
        cs.finish()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_a"])
    assert [c["values"] for c in rec["operator_overrides"]] == [{"img_a": {"gain": 6.0}}, {}]


def test_atomdata_banner_names_the_override_once_per_camera():
    from waxa.atomdata_base import operator_override_banner
    fe, stream, cs = make(["img_a", "img_b"])
    fe.data.init()
    try:
        stream.operator({"img_b": {"gain": 9.0}})
        time.sleep(0.4)
        shot(cs, stream, 0, 1, counter=0)
        shot(cs, stream, 1, 2)
    finally:
        cs.finish()
    attrs = {k: v for k, v in fe._extra_file_texts.items() if k.startswith("camera_stream_")}
    assert set(attrs) == {"camera_stream_img_a", "camera_stream_img_b"}
    banner = operator_override_banner(123, attrs)
    assert banner.startswith("!! run 123: an operator overrode")
    assert banner.count("basler_usb:12345") == 1 and "'img_b': 1" in banner
    assert operator_override_banner(123, {"camera_stream_x": json.dumps({"key": "x"})}) == ""


def test_a_server_without_run_streams_changes_nothing():
    class OldServerStream(TrigFakeStream):
        def declare_run_stream(self, frames, max_exposure_time=None):
            from beacon.camera.backend import CameraError
            raise CameraError("RUN_STREAM on x failed (unknown_command): Unknown command")
    fe, stream, cs = make(["img_a"], stream=OldServerStream())
    fe.data.init()
    try:
        assert cs._run_stream is False and cs.counts.get("run_stream_unsupported") == 1
        shot(cs, stream, 0, 1, counter=0)
        assert M(fe, "img_a", "exposure")[0] == 19e-6
    finally:
        cs.finish()


def test_a_diagnostic_images_max_reaches_the_stream_and_the_record():
    from waxx.control.cameras.camera_param_classes import BaslerParams
    from waxx.control.cameras.diagnostic_frame import diagnostic_frame
    cam = BaslerParams(serial_number="12345")
    cam.key = "cam_x"

    class frame(diagnostic_frame):
        def __init__(self):
            self.setup()
            self.img_a = self.image(cam, exposure_time=1e-4, gain=0.)
            self.img_b = self.image(cam, exposure_time=1e-3, gain=6., max_exposure_time=5e-3)
            self.cleanup()

    f = frame()
    (c,) = f.entries()
    assert c.frame_max_exposure == {"img_b": 5e-3}
    assert "max_exposure_time=0.005" in f.img_b.assignment()
    assert "max_exposure_time" not in f.img_a.assignment()
    with pytest.raises(ValueError, match="above its max_exposure_time"):
        diagnostic_frame().image(cam, exposure_time=1e-2, max_exposure_time=5e-3)
