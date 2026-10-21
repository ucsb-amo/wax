"""The remote viewer shows what the acquisition window shows (status strip,
Abort by run id, a late joiner's catch-up), and both show the person hold
(HoldBadge).  No network: the badge gets a fake monitor link, the viewer a
fake ``_request``; discovery is never started."""
import gc
import sys
import threading

import pytest

import liveod_qt_helpers as qt


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


# -- the hold badge ---------------------------------------------------------------------

class FakeLink:
    def __init__(self, hold=None):
        self.hold = hold
        self.requests = []
        self.answer = True

    def status(self):
        if not self.answer:
            return None
        return {"state_name": "READY", "person_hold": self.hold}

    def request(self, obj):
        self.requests.append(dict(obj))
        if obj.get("action") == "release":
            self.hold = {"active": False}
            return {"status": "ok", "person_hold": self.hold}
        return {"status": "error", "msg": "?"}


HOLD = {"active": True, "since": 1_760_000_000, "by": "jp@kong", "reason": "aligning",
        "owner": "person", "source": "liveod"}


def _badge(link, log=None):
    from waxx.util.live_od.gui.hold_badge import HoldBadge
    badge = HoldBadge(link=link, synchronous=True, log=log, by="liveOD on test")
    badge.start(listen=False)
    return badge


def test_the_badge_shows_only_while_a_hold_is_on(app):
    from waxx.util.dashboard import theme
    lines = []
    link = FakeLink(HOLD)
    badge = _badge(link, log=lambda text, level=20: lines.append((level, text)))
    assert badge.held() and not badge.isHidden()
    assert theme.HELD in badge.styleSheet()
    assert "jp@kong" in badge.toolTip() and "aligning" in badge.toolTip()
    assert any("Person hold ON" in t for _, t in lines)
    # no answer: the last known state stays
    link.answer = False
    badge.poll()
    assert badge.held()
    link.answer = True
    link.hold = {"active": False}
    badge.poll()
    assert not badge.held() and badge.isHidden()
    assert lines[-1][1] == "Person hold released."
    badge.shutdown()


def test_one_click_sends_the_queue_release(app):
    link = FakeLink(HOLD)
    badge = _badge(link)
    badge.click()
    assert link.requests == [{"type": "run_queue", "action": "release", "owner": "person",
                              "by": "liveOD on test"}]
    assert not badge.held() and badge.isHidden()
    badge.shutdown()


def test_a_broadcast_only_makes_it_ask(app):
    link = FakeLink({"active": False})
    badge = _badge(link)
    link.hold = HOLD
    badge.on_broadcast({"type": "person_hold", "person_hold": {"active": False}})
    assert badge.held()         # the poll's answer, not the broadcast's payload
    badge.shutdown()


# -- the remote viewer -----------------------------------------------------------------

@pytest.fixture
def viewer(app, monkeypatch):
    from waxx.util.live_od.gui import remote_viewer_window as rv
    from waxx.util.live_od.gui.plotter import LiveODPlotter
    slot_errors = qt.SlotErrors()
    monkeypatch.setattr(sys, "excepthook", slot_errors)
    gc.collect()
    app.processEvents()
    views_before = qt.pyqtgraph_views()
    threads_before = set(threading.enumerate())
    monkeypatch.setattr(rv.RemoteViewerWindow, "_start_discovery", lambda self: None)
    monkeypatch.setattr(LiveODPlotter, "start", lambda self, *a: None)
    win = rv.RemoteViewerWindow()
    sent = []
    answers = {}

    def request(payload, label, quiet=False):
        sent.append(dict(payload))
        return answers.get(payload.get("tag"))
    win._request = request
    # worker threads in place, and their GUI hop too
    import types
    monkeypatch.setattr(rv, "threading", types.SimpleNamespace(
        Thread=lambda target, args=(), daemon=None: _Now(target, args)))
    win._on_gui_signal.disconnect()
    win._on_gui_signal.connect(lambda fn: fn(), type=_direct())
    yield win, sent, answers
    win.close()
    problems = [qt.join_or_keep(t) for t in qt.new_threads(threads_before)]
    qt.forget_views_since(views_before)
    qt.delete_widgets(app, [win.live_scalar_plot_window, win.fk_tof_window,
                            win._adjust_panel, win])
    assert not [p for p in problems if p], problems
    assert not slot_errors.seen, slot_errors.seen


class _Now:
    def __init__(self, target, args):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def _direct():
    from PyQt6.QtCore import Qt
    return Qt.ConnectionType.DirectConnection


INFO = {"expt_name": "hf_bec.py", "camera_key": "andor", "capture_images": True,
        "save_data": True, "n_shots": 40, "xvar_ranges": {"t_tof": [1e-3, 2e-3]},
        "pixel_size_m": 1.1e-6, "params": {"t_tof": 1e-3}}


def test_run_started_fills_the_strip_and_a_final_state_stays(viewer):
    win, _sent, _ = viewer
    win._on_run_started({"tag": "RUN_STARTED", "run_id": 86001, "xvarnames": ["t_tof"],
                         "info": INFO})
    strip = win.status_strip
    assert strip.run_label.text() == "86001 · hf_bec.py"
    assert strip.progress.maximum() == 40
    assert win.viewer_window._camera_key == "andor"
    assert win.viewer_window._px_size_m == pytest.approx(1.1e-6)
    win._on_run_state("running", "")
    assert win._run_in_progress and "e53935" in win.abort_button.styleSheet()
    win._on_shot_progress(4, 40, {"t_tof": 1e-3})
    assert strip.progress.value() == 5
    strip.set_timing(8.2, "14:32")
    assert "14:32" in strip.timing_label.text()
    win._on_run_state("aborted", "")
    assert strip.state() == "aborted" and strip.pill.text() == "Aborted"
    assert not win._run_in_progress and win.abort_button.styleSheet() == ""
    assert win.windowTitle().startswith("LiveOD Viewer — Run 86001 — Aborted")


def test_an_unsaved_run_says_so(viewer):
    win, _sent, _ = viewer
    win._on_run_started({"run_id": 0, "xvarnames": [], "info": dict(INFO, save_data=False)})
    assert win.status_strip.run_label.text().startswith("NOT SAVING")


def test_an_older_server_without_info_still_works(viewer):
    win, sent, answers = viewer
    answers["GET_ADJUST_VALUES"] = {"ok": True, "specs": [], "values": {}}
    win._on_run_started({"run_id": 86002, "xvarnames": []})
    assert win.status_strip.run_label.text() == "86002"
    assert {"tag": "GET_ADJUST_VALUES"} in sent


def test_abort_sends_a_reset_naming_the_run_and_nothing_without_one(viewer):
    win, sent, answers = viewer
    answers["POLL"] = {"ok": True, "run_in_progress": False, "run_id": 86000}
    win._on_reset_clicked()
    assert [p["tag"] for p in sent] == ["POLL"]      # no RESET: it would put the hold on
    assert "no run to abort" in win.viewer_window.output_window.toPlainText()
    answers["POLL"] = {"ok": True, "run_in_progress": True, "run_id": 86003}
    win._on_reset_clicked()
    assert sent[-1] == {"tag": "RESET", "source": "person", "run_id": 86003}
    sent.clear()
    answers["POLL"] = {"ok": True, "run_in_progress": True, "run_id": 0}    # unsaved
    win._on_reset_clicked()
    assert sent[-1] == {"tag": "RESET", "source": "person"}
    sent.clear()
    answers.pop("POLL")
    win._on_reset_clicked()
    assert [p["tag"] for p in sent] == ["POLL"]
    assert "did not answer" in win.viewer_window.output_window.toPlainText()


def test_a_late_joiner_catches_up_from_poll_and_get_log(viewer):
    win, sent, answers = viewer
    answers["POLL"] = {"ok": True, "run_in_progress": True, "run_id": 86004,
                       "run_state": "running", "expt_name": "hf_bec.py",
                       "run_camera_key": "andor", "save_data": True,
                       "n_shots_expected": 40, "n_shots": 12}
    answers["GET_LOG"] = {"ok": True, "run": {"run_id": 86004},
                          "records": [{"t": 1.0, "level": 30, "msg": "shot 3 slow"}]}
    win._catch_up()
    strip = win.status_strip
    assert strip.run_label.text() == "86004 · hf_bec.py"
    assert strip.state() == "running" and strip.progress.value() == 12
    assert win._run_in_progress
    text = win.viewer_window.output_window.toPlainText()
    assert "shot 3 slow" in text and "before this viewer joined" in text
    # a run the viewer saw start is not restarted by a later catch-up
    answers["POLL"] = dict(answers["POLL"], n_shots=13)
    win._catch_up()
    assert strip.progress.value() == 12


def test_adjust_specs_broadcast_fills_the_panel(viewer):
    win, _sent, _ = viewer
    spec = {"key": "amp", "min_val": 0.0, "max_val": 1.0, "step": 0.01, "current_val": 0.5}
    win._on_adjust_state_fetched([spec], {})
    assert win._adjust_button.isEnabled() and win._adjust_button.text() == "Adjust (1)"
    win._on_adjust_state_fetched([], {})
    assert not win._adjust_button.isEnabled()


def test_the_broadcaster_sends_the_new_messages():
    from waxx.util.live_od.live_od_broadcaster import LiveODBroadcaster
    b = LiveODBroadcaster.__new__(LiveODBroadcaster)
    import queue
    b._queue = queue.Queue()
    b.broadcast_run_started(5, ["t"], info={"expt_name": "x"})
    b.broadcast_run_started(6, ["t"])
    b.broadcast_shot_timing(1.5, "12:00")
    b.broadcast_next_run_id(7)
    b.broadcast_adjust_specs([{"key": "a"}])
    msgs = [b._queue.get_nowait() for _ in range(5)]
    assert msgs[0] == {"tag": "RUN_STARTED", "run_id": 5, "xvarnames": ["t"],
                       "info": {"expt_name": "x"}}
    assert "info" not in msgs[1]
    assert msgs[2] == {"tag": "SHOT_TIMING", "delta_t": 1.5, "eta": "12:00"}
    assert msgs[3] == {"tag": "NEXT_RUN_ID", "run_id": 7}
    assert msgs[4] == {"tag": "ADJUST_SPECS", "specs": [{"key": "a"}]}
