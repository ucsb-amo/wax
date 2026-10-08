"""Dashboard layouts: working layout autosave + named layouts from the toolbar dropdown.
Offscreen Qt; QSettings is redirected to an INI file in tmp_path (never the registry)."""
import os
import time

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QInputDialog, QLabel, QMessageBox

from waxx.util.dashboard import dashboard_window as dw
from waxx.util.dashboard.layout_store import DEFAULT_NAME, UNSAVED_LABEL, LayoutSnapshot, LayoutStore
from waxx.util.dashboard.panel_container import ClientPanel


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ini(tmp_path, monkeypatch):
    path = str(tmp_path / "dash.ini")
    real = QSettings
    monkeypatch.setattr(dw, "QSettings", lambda *_a, **_k: real(path, real.Format.IniFormat))
    monkeypatch.setattr(dw.DashboardMainWindow, "_probe_data_dir", lambda self: None)
    return path


def _pump(qapp, cond=lambda: False, rounds=300):
    """Run the event loop (timers included) until *cond* holds, ~10 ms per round."""
    for _ in range(rounds):
        qapp.processEvents()
        if cond():
            return
        time.sleep(0.01)


def _make_window(qapp):
    panels = [
        dw.PanelPlacement(ClientPanel(pid, pid.upper(), lambda pid=pid: QLabel(pid)), area="right")
        for pid in ("alpha", "beta", "gamma")
    ]
    win = dw.DashboardMainWindow("client", "Test dashboard", panels, host_ip="test")
    win.resize(900, 600)
    win.show()
    _pump(qapp, lambda: win._layout_ready)
    assert win._layout_ready
    win._applying_until = 0.0
    return win


def _close(win):
    win._layout_ready = False
    win.hide()
    win.deleteLater()


# --- store --------------------------------------------------------------------

def test_store_add_rename_delete_and_active(ini):
    store = LayoutStore(QSettings(ini, QSettings.Format.IniFormat), "dashboard/client/test")
    snap = LayoutSnapshot(geometry=b"\x01\x02", state=b"\x03", popped=["beta"], popped_geometry={"beta": b"\x04"})
    store.put("Lab", snap)
    store.put("Night", LayoutSnapshot(state=b"\x05"))
    assert store.names() == ["Lab", "Night"]
    got = store.get("lab")                       # names match case-insensitively
    assert got.state == b"\x03" and got.popped == ["beta"] and got.popped_geometry == {"beta": b"\x04"}

    store.put("Lab", LayoutSnapshot(state=b"\x09"))   # overwrite keeps the position
    assert store.names() == ["Lab", "Night"] and store.get("Lab").state == b"\x09"

    store.active = "Lab"
    store.rename("Lab", "Day")
    assert store.names() == ["Day", "Night"] and store.active == "Day"
    store.delete("Day")
    assert store.names() == ["Night"] and store.active == ""


def test_store_rejects_bad_names(ini):
    store = LayoutStore(QSettings(ini, QSettings.Format.IniFormat), "g")
    store.put("Lab", LayoutSnapshot(state=b"\x01"))
    assert store.validate_name("  ") is not None
    assert store.validate_name(DEFAULT_NAME.lower()) is not None
    assert store.validate_name(UNSAVED_LABEL) is not None
    assert store.validate_name("LAB") is not None
    assert store.validate_name("Lab", renaming="Lab") is None
    assert store.validate_name("Other") is None
    assert store.unique_name("Lab") == "Lab (2)"


# --- window -------------------------------------------------------------------

def test_first_launch_shows_default_and_autosaves_a_resize(qapp, ini):
    win = _make_window(qapp)
    try:
        assert win._layouts.active == DEFAULT_NAME
        assert win._layout_picker.dropdown.text() == DEFAULT_NAME
        assert not win._layout_picker.remove_btn.isEnabled()
        win._layout_save_timer.stop()
        win.resize(1000, 650)
        _pump(qapp, rounds=5)
        assert win._layout_save_timer.isActive()        # resize schedules the autosave
        assert win._layouts.modified
        assert win._layout_picker.dropdown.text().endswith("•")
    finally:
        _close(win)


def test_named_layout_add_apply_and_reopen(qapp, ini, monkeypatch):
    win = _make_window(qapp)
    try:
        win.panel("gamma").hide()
        _pump(qapp, rounds=5)
        monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Compact", True)))
        win._add_layout()
        assert win._layouts.names() == ["Compact"] and win._layouts.active == "Compact"
        assert not win._layouts.modified
        assert win._layout_picker.dropdown.text() == "Compact"
        assert win._layout_picker.remove_btn.isEnabled()

        win._panel_action_triggered(win.panel("gamma"), True)    # a user change
        _pump(qapp, rounds=5)
        assert win._layouts.modified and not win.panel("gamma").isHidden()

        win._select_layout("Compact")
        _pump(qapp, rounds=5)
        assert win.panel("gamma").isHidden()
        assert not win._layouts.modified
        win._save_layout()
    finally:
        _close(win)

    # The next launch opens to the working layout and still names the selection.
    win2 = _make_window(qapp)
    try:
        assert win2.panel("gamma").isHidden()
        assert win2._layouts.active == "Compact"
        assert win2._layout_picker.dropdown.text() == "Compact"
    finally:
        _close(win2)


def test_row_buttons_rename_overwrite_delete(qapp, ini, monkeypatch):
    win = _make_window(qapp)
    try:
        monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("One", True)))
        win._add_layout()
        picker = win._layout_picker
        assert picker.row(DEFAULT_NAME).buttons == {}
        assert set(picker.row("One").buttons) == {"rename", "overwrite", "delete"}

        monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Two", True)))
        picker.row("One").buttons["rename"].click()
        _pump(qapp, lambda: win._layouts.names() == ["Two"])
        assert win._layouts.names() == ["Two"] and win._layouts.active == "Two"

        monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
        win.panel("beta").hide()
        _pump(qapp, rounds=5)
        picker.row("Two").buttons["overwrite"].click()
        _pump(qapp, lambda: not win._layouts.modified)
        win.panel("beta").show()
        win._select_layout("Two")
        _pump(qapp, rounds=5)
        assert win.panel("beta").isHidden()             # the overwrite stored the hidden panel

        picker.remove_btn.click()                        # '-' deletes the selected layout
        _pump(qapp, lambda: win._layouts.names() == [])
        assert win._layouts.names() == [] and win._layouts.active == ""
        assert picker.dropdown.text() == UNSAVED_LABEL
    finally:
        _close(win)


def test_discard_keeps_named_layouts(qapp, ini, monkeypatch):
    win = _make_window(qapp)
    try:
        monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Keep", True)))
        win._add_layout()
        win._save_layout()
        win._discard_saved_layout("test")
        assert win._settings.value(win._layout_key("state")) is None
        assert win._layouts.names() == ["Keep"]
        assert win._layouts.active == DEFAULT_NAME
    finally:
        _close(win)
