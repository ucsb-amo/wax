"""The camera settings dialog (the ⚙): Persist (asked first, exactly the
whitelisted fields, never exposure or a run-owned field, locked during a run),
the footer, refusals put the field back, a field being edited is never
overwritten (a chip instead), EM gain above 100 needs the unlock, and the mouse
wheel never changes an unfocused field."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QWheelEvent  # noqa: E402

from cam_ctrl_fakes import FakeHost, HostRefused  # noqa: E402
from liveod_qt_helpers import delete_widgets, session_app  # noqa: E402

ANDOR_WHITELIST = ["gain", "hs_speed", "preamp", "vs_speed", "vs_amp", "baseline_clamp"]
RUN_OWNED = {"exposure_time", "trigger_mode", "trigger_source", "frame_transfer", "sensor_roi",
             "shutter", "pixel_format", "acquisition_mode"}


@pytest.fixture(scope="module")
def app():
    return session_app()


class Asker:
    """``confirm(title, text)``: records the question, answers ``answer``."""

    def __init__(self, answer=True):
        self.answer = answer
        self.asked = []

    def __call__(self, title, text):
        self.asked.append((title, text))
        return self.answer


def _dialog(app, host, key="andor", answer=True, **kw):
    from waxx.util.live_od.gui.camera_settings_dialog import CameraSettingsDialog
    asker = Asker(answer)
    dlg = CameraSettingsDialog(key, host, confirm=asker, async_calls=kw.pop("async_calls", False),
                               **kw)
    dlg.show()
    app.processEvents()
    return dlg, asker


def test_persist_on_asks_first_listing_exactly_the_whitelist(app):
    from beacon.camera.schema import ANDOR_EMCCD
    host = FakeHost()
    dlg, asker = _dialog(app, host, answer=False)
    try:
        assert dlg.persist_keys == ANDOR_WHITELIST
        assert not (set(dlg.persist_keys) & RUN_OWNED)
        dlg.persist_box.click()                          # cancelled
        assert len(asker.asked) == 1
        title, text = asker.asked[0]
        listed = [ANDOR_EMCCD.setting(k).label for k in ANDOR_WHITELIST]
        for label in listed:
            assert label in text
        # nothing else is listed: every indented line is one of the whitelisted fields
        lines = [ln.strip() for ln in text.splitlines() if ln.startswith("    ")]
        assert len(lines) == len(ANDOR_WHITELIST)
        assert all(any(ln.startswith(lab) for lab in listed) for ln in lines)
        for key in RUN_OWNED:
            if ANDOR_EMCCD.has(key):
                assert not any(ln.startswith(ANDOR_EMCCD.setting(key).label) for ln in lines)
        assert host.calls_of("set_persist") == []
        assert not dlg.persist_box.isChecked()

        asker.answer = True
        dlg.persist_box.click()                          # confirmed
        assert host.calls_of("set_persist") == [("set_persist", "andor", True)]
        assert dlg.persist_box.isChecked()
        assert dlg.persist_status.text().startswith("ON since")
        assert "gain=30" in dlg.persist_status.text()
        assert sorted(dlg.marked_fields()) == sorted(ANDOR_WHITELIST)
        assert "exposure_time" not in dlg.marked_fields()
        assert "Persist is ON" in dlg.footer.text()

        n = len(asker.asked)
        dlg.persist_box.click()                          # off: no question
        assert len(asker.asked) == n
        assert host.calls_of("set_persist")[-1] == ("set_persist", "andor", False)
        assert not dlg.persist_box.isChecked() and dlg.marked_fields() == []
    finally:
        delete_widgets(app, [dlg])


def test_basler_persists_gain_only(app):
    host = FakeHost()
    dlg, asker = _dialog(app, host, key="xy_basler")
    try:
        assert dlg.persist_keys == ["gain"]
        dlg.persist_box.click()
        lines = [ln for ln in asker.asked[0][1].splitlines() if ln.startswith("    ")]
        assert len(lines) == 1 and "Gain" in lines[0]
        assert dlg.marked_fields() == ["gain"]
    finally:
        delete_widgets(app, [dlg])


def test_the_whitelist_never_takes_exposure_even_if_a_host_says_so(app):
    from beacon.camera.schema import ANDOR_EMCCD
    from waxx.util.live_od.gui.camera_settings_dialog import persist_fields
    assert persist_fields(ANDOR_EMCCD, ["exposure_time", "gain", "trigger_mode"]) == ["gain"]
    assert persist_fields(ANDOR_EMCCD) == ANDOR_WHITELIST


def test_footer(app):
    from waxx.util.live_od.gui.camera_settings_dialog import FOOTER_TEXT
    assert FOOTER_TEXT == ("Live settings. A run applies camera_params in full; with Persist "
                           "on (red) the marked fields are applied on top and recorded.")
    dlg, _ = _dialog(app, FakeHost())
    try:
        assert dlg.footer.text() == FOOTER_TEXT
        assert not dlg.isModal()
    finally:
        delete_widgets(app, [dlg])


def test_locked_during_a_run(app):
    host = FakeHost()
    host.set_state("andor", "acquiring", run_id=80713)
    dlg, asker = _dialog(app, host)
    try:
        assert dlg.is_locked()
        assert dlg.tabs.currentIndex() == dlg.run_tab_index          # opens on the Run tab
        assert not dlg.persist_box.isEnabled()
        assert "Run 80713 holds andor" in dlg.persist_reason.text()
        assert not dlg.live_form.editable and not dlg.advanced_form.editable
        assert not dlg.unlock_box.isEnabled()
        # the run ends: editable again, Persist can change
        host.set_state("andor", "idle", run_id=None)
        dlg.set_snapshot(host.snapshot())
        assert not dlg.is_locked() and dlg.persist_box.isEnabled() and dlg.live_form.editable
        # a run starts while it is open
        host.set_state("andor", "run_locked", run_id=80714)
        dlg.set_snapshot(host.snapshot())
        assert not dlg.persist_box.isEnabled() and "Run 80714" in dlg.persist_reason.text()
    finally:
        delete_widgets(app, [dlg])


def test_persist_refused_by_the_host_is_undone(app):
    host = FakeHost()
    dlg, _ = _dialog(app, host)
    try:
        def refuse(key, on, values=None):
            raise HostRefused(f"persist for {key} cannot change while run 80713 holds the camera")
        host.set_persist = refuse
        dlg.persist_box.click()
        assert not dlg.persist_box.isChecked()
        assert "refused" in dlg.message.text() and "80713" in dlg.message.text()
    finally:
        delete_widgets(app, [dlg])


def test_frame_transfer_is_never_shown(app):
    host = FakeHost()
    real = host.describe
    host.describe = lambda key: real(key, include_hidden=True)    # a schema that has it
    dlg, _ = _dialog(app, host)
    try:
        keys = set(dlg.live_form.keys()) | set(dlg.advanced_form.keys())
        assert "frame_transfer" not in keys
        assert "gain" in keys and "vs_speed" in keys
        rows = [dlg.run_table.item(i, 0).text() for i in range(dlg.run_table.rowCount())]
        assert "Frame transfer" not in rows
    finally:
        delete_widgets(app, [dlg])


def test_a_commit_goes_to_set_live_and_a_refusal_reverts_it(app):
    from beacon.camera.backend import ApplyRefused
    host = FakeHost()
    dlg, _ = _dialog(app, host)
    try:
        field = dlg.advanced_form.field("vs_amp")
        field.setValue(1)                                 # an edit that finished
        assert host.calls_of("set_live")[-1] == (
            "set_live", "andor", {"vs_amp": 1, "em_gain_unlocked": False})
        assert host.cams["andor"]["settings"]["vs_amp"] == 1
        assert dlg.advanced_form.values()["vs_amp"] == 1

        host.refuse_next = ApplyRefused("vs_amp", "vs_amp=0 refused (live): VS 0.3 us with "
                                                  "Normal amplitude transfers no charge")
        field.setValue(0)
        assert field.value() == 1                         # put back
        assert dlg.advanced_form.values()["vs_amp"] == 1
        assert "refused" in dlg.message.text() and "transfers no charge" in dlg.message.text()
        assert not dlg.advanced_form.message_label.isHidden()      # under its form too
    finally:
        delete_widgets(app, [dlg])


def test_a_field_being_edited_is_not_overwritten(app, monkeypatch):
    host = FakeHost()
    dlg, _ = _dialog(app, host)
    try:
        gain = dlg.live_form.field("gain")
        monkeypatch.setattr(gain, "hasFocus", lambda: True)          # being edited
        host.cams["andor"]["settings"].update(gain=55, vs_amp=2)     # another program
        host.cams["andor"]["settings_rev"] += 1
        dlg.set_snapshot(host.snapshot())
        assert gain.value() == 30 and dlg.live_form.values()["gain"] == 30
        assert dlg.stale_fields() == {"gain": 55}
        assert dlg.reload_chip.isVisibleTo(dlg) and "reload" in dlg.reload_chip.text()
        assert "gain" in dlg.reload_chip.text()
        # a field nobody is editing just follows
        assert dlg.advanced_form.values()["vs_amp"] == 2
        monkeypatch.setattr(gain, "hasFocus", lambda: False)
        dlg.reload_chip.click()
        assert gain.value() == 55 and dlg.stale_fields() == {}
        assert not dlg.reload_chip.isVisibleTo(dlg)
    finally:
        delete_widgets(app, [dlg])


def test_em_gain_above_the_cap_needs_the_unlock(app):
    host = FakeHost()
    dlg, asker = _dialog(app, host, answer=False)
    try:
        gain = dlg.live_form.field("gain")
        gain.setValue(150)                                # declined
        assert len(asker.asked) == 1 and "Unlock EM gain above 100" in asker.asked[0][0]
        assert host.calls_of("set_live") == []
        assert gain.value() == 30 and not dlg.unlock_box.isChecked()

        asker.answer = True
        gain.setValue(150)                                # unlocked
        assert dlg.unlock_box.isChecked()
        assert host.calls_of("set_live")[-1] == (
            "set_live", "andor", {"gain": 150, "em_gain_unlocked": True})
        assert dlg.live_form.values()["gain"] == 150
        n = len(asker.asked)
        gain.setValue(160)                                # already unlocked: no question
        assert len(asker.asked) == n
        # the unlock cannot be taken off while the gain is above the cap
        dlg.unlock_box.setChecked(False)
        assert dlg.unlock_box.isChecked() and "lower it" in dlg.message.text()
    finally:
        delete_widgets(app, [dlg])


def test_the_wheel_never_changes_an_unfocused_field(app):
    host = FakeHost()
    dlg, _ = _dialog(app, host)
    try:
        gain = dlg.live_form.field("gain")
        assert not gain.hasFocus()
        center = QPointF(gain.width() / 2, gain.height() / 2)
        event = QWheelEvent(center, QPointF(gain.mapToGlobal(QPoint(5, 5))), QPoint(0, 0),
                            QPoint(0, 120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        app.sendEvent(gain, event)
        app.processEvents()
        assert gain.value() == 30
        assert host.calls_of("set_live") == []
    finally:
        delete_widgets(app, [dlg])


def test_run_tab_shows_request_persisted_applied_and_source(app):
    from waxx.util.live_od.gui.camera_settings_dialog import RUN_COLUMNS
    host = FakeHost()
    host.cams["andor"]["run_request"] = {"exposure_time": 1e-5, "gain": 300, "vs_speed": 1}
    dlg, _ = _dialog(app, host)
    try:
        dlg.persist_box.click()
        table = dlg.run_table
        assert [table.horizontalHeaderItem(j).text() for j in range(table.columnCount())] == list(
            RUN_COLUMNS)
        rows = {table.item(i, 0).text(): [table.item(i, j).text() for j in range(1, 5)]
                for i in range(table.rowCount())}
        assert rows["EM gain"][0] == "300" and rows["EM gain"][1] == "30"
        assert rows["EM gain"][2] == "30" and rows["EM gain"][3] == "read from the camera"
        assert rows["Exposure"][1] == "never"               # never persisted
        assert rows["Exposure"][0].startswith("10")          # 1e-5 s shown in µs
        assert rows["Vertical clock amplitude"][3] == "as commanded (no readback)"
        assert rows["Shutter"][0].startswith("fixed")
        for i in range(table.rowCount()):
            assert not (table.item(i, 1).flags() & Qt.ItemFlag.ItemIsEditable)
    finally:
        delete_widgets(app, [dlg])


def test_async_commit_comes_back_on_the_gui_thread(app):
    import time
    host = FakeHost()
    dlg, _ = _dialog(app, host, async_calls=True)
    try:
        dlg.advanced_form.field("vs_amp").setValue(2)
        assert dlg.wait_idle(5.0)
        deadline = time.monotonic() + 5.0
        while dlg._pending and time.monotonic() < deadline:
            app.processEvents()
        assert dlg._pending == {}
        assert host.calls_of("set_live")[-1][2]["vs_amp"] == 2
        assert dlg.advanced_form.values()["vs_amp"] == 2
        # the snapshot of our own write, before or after its answer, is not "changed by X"
        dlg.set_snapshot(host.snapshot())
        assert dlg.stale_fields() == {}
    finally:
        dlg.wait_idle(5.0)
        delete_widgets(app, [dlg])


def test_async_calls_answer_in_edit_order(app):
    """Two quick edits of one field: the host sees them, and the dialog takes their
    answers, in the order they were made, even when the first answer is slow."""
    import threading
    import time
    host = FakeHost()
    real = host.set_live
    gate = threading.Event()
    order = []

    def slow_first(key, values, timeout_s=15.0):
        if values.get("vs_amp") == 1:
            gate.wait(2.0)
        order.append(values.get("vs_amp"))
        return real(key, values)
    host.set_live = slow_first
    dlg, _ = _dialog(app, host, async_calls=True)
    try:
        field = dlg.advanced_form.field("vs_amp")
        field.setValue(1)
        field.setValue(2)
        gate.set()
        assert dlg.wait_idle(5.0)
        deadline = time.monotonic() + 5.0
        while dlg._pending and time.monotonic() < deadline:
            app.processEvents()
        assert order == [1, 2]
        assert dlg.advanced_form.values()["vs_amp"] == 2
        assert host.cams["andor"]["settings"]["vs_amp"] == 2
        assert dlg._worker is None                  # nothing left running
    finally:
        dlg.wait_idle(5.0)
        delete_widgets(app, [dlg])


def test_describe_failure_leaves_a_read_only_dialog(app):
    host = FakeHost()

    def broken(key):
        raise HostRefused("andor: liveOD's camera host is not running")
    host.describe = broken
    dlg, _ = _dialog(app, host)
    try:
        assert "not running" in dlg.message.text()
        assert not dlg.live_form.editable
    finally:
        delete_widgets(app, [dlg])
