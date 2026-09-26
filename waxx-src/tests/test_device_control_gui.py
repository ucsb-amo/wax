"""Device-control GUI layout: one card per channel, no row/column headers,
TTL banks with no output channel skipped, and the changes log in a pop-out
window backed by a buffer that outlives the window.

Offscreen Qt only. The network threads (state listener, update sender,
snapshot worker, monitor status poller) are stubbed out and QSettings is
faked so nothing is written to the registry.
"""
import os

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel

from waxx.util.guis import device_control_gui as dc


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class FakeSettings:
    store: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def value(self, key, default=None, type=None):
        return FakeSettings.store.get(key, default)

    def setValue(self, key, value):
        FakeSettings.store[key] = value


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    FakeSettings.store = {}
    monkeypatch.setattr(dc, "QSettings", FakeSettings)
    return FakeSettings


@pytest.fixture
def gui(qapp, monkeypatch):
    for name in ("_setup_update_sender", "_setup_state_listener", "_setup_state_worker",
                 "setup_status_checker", "setup_timer", "request_state"):
        monkeypatch.setattr(dc.DeviceStateGUI, name, lambda self, *a, **k: None)
    g = dc.DeviceStateGUI()
    yield g
    g.close()


def _dds(uru, ch, dac_ch=-1):
    return {"frequency": 110e6, "amplitude": 0.4, "v_pd": 5.0, "sw_state": 0,
            "urukul_idx": uru, "ch": ch, "transition": "None", "dac_ch": dac_ch,
            "force_update_counter": 0}


CONFIG = {
    "dds": {"imaging": _dds(0, 0, dac_ch=3), "d1_3d_c": _dds(2, 3)},
    "dac": {"dac_ch0": {"ch": 0, "voltage": 1.0}, "dac_ch9": {"ch": 9, "voltage": -2.0}},
    "ttl": {
        "a": {"ch": 0, "ttl_state": 0}, "b": {"ch": 5, "ttl_state": 1},
        "c": {"ch": 48, "ttl_state": 0}, "d": {"ch": 87, "ttl_state": 1},
    },
}


@pytest.fixture
def loaded(gui):
    gui.config_data = {k: {n: dict(c) for n, c in v.items()} for k, v in CONFIG.items()}
    gui.update_device_widgets()
    return gui


# --- grids ------------------------------------------------------------------

def test_no_header_labels(loaded):
    for lay in (loaded.dds_layout, loaded.dac_layout, loaded.ttl_layout):
        for i in range(lay.count()):
            w = lay.itemAt(i).widget()
            if isinstance(w, QLabel):
                # only the dimmed "dac N" / "ttl N" placeholders remain
                assert not w.isEnabled(), w.text()
                assert w.text().split()[0] in ("dac", "ttl"), w.text()


def test_dds_grid_positions(loaded):
    assert loaded.dds_layout.itemAtPosition(0, 0).widget() is loaded.device_widgets["dds.imaging"]
    assert loaded.dds_layout.itemAtPosition(3, 2).widget() is loaded.device_widgets["dds.d1_3d_c"]
    assert loaded.dds_layout.rowCount() == 4
    assert loaded.dds_layout.columnCount() == 3


def test_dac_grid_positions(loaded):
    assert loaded.dac_layout.columnCount() == 2
    assert loaded.dac_layout.rowCount() == 8
    assert loaded.dac_layout.itemAtPosition(1, 1).widget() is loaded.device_widgets["dac.dac_ch9"]
    assert loaded.dac_layout.itemAtPosition(0, 1).widget().text() == "dac 8"


def test_ttl_banks_without_outputs_are_skipped(loaded):
    # banks 0, 6 (48-55) and 10 (80-87) hold outputs -> exactly 3 columns;
    # banks 1-5 and 7-9 (e.g. the TTL-input bank) are absent, not empty.
    assert loaded.ttl_layout.columnCount() == 3
    assert loaded.ttl_layout.rowCount() == 8
    assert loaded.ttl_layout.itemAtPosition(0, 1).widget() is loaded.device_widgets["ttl.c"]
    assert loaded.ttl_layout.itemAtPosition(7, 2).widget() is loaded.device_widgets["ttl.d"]
    assert loaded.ttl_layout.itemAtPosition(1, 0).widget().text() == "ttl 1"


def test_cells_are_cards_with_scoped_container_style(loaded):
    for w in loaded.device_widgets.values():
        assert w.objectName() == "device_cell"
        assert w.testAttribute(Qt.WidgetAttribute.WA_StyledBackground)
    for name in ("dds", "dac", "ttl"):
        container = getattr(loaded, f"{name}_container")
        assert container.objectName() == f"{name}_container"
        assert f"QWidget#{name}_container" in container.styleSheet()
    assert loaded.dds_layout.horizontalSpacing() == dc.GRID_SPACING
    assert loaded.dds_layout.verticalSpacing() == dc.GRID_SPACING


def test_cells_are_separated_after_layout(loaded, qapp):
    loaded.resize(900, 600)
    loaded.show()
    loaded.tab_widget.setCurrentIndex(2)
    qapp.processEvents()
    loaded.ttl_layout.activate()
    top, below = loaded.ttl_layout.cellRect(0, 0), loaded.ttl_layout.cellRect(1, 0)
    assert top.height() > 0
    assert below.top() - top.bottom() >= dc.GRID_SPACING - 1
    loaded.tab_widget.setCurrentIndex(0)
    qapp.processEvents()
    loaded.dds_layout.activate()
    left, right = loaded.dds_layout.cellRect(0, 0), loaded.dds_layout.cellRect(0, 1)
    assert right.left() - left.right() >= dc.GRID_SPACING


def test_dds_amp_voltage_switch_kept(loaded):
    w = loaded.device_widgets["dds.imaging"]
    assert w._has_dac()
    assert w._amp_unit == "V"
    assert w.vpd_spinbox.isVisibleTo(w) and not w.amp_spinbox.isVisibleTo(w)
    w.on_amp_unit_changed("Amp")
    assert w.amp_spinbox.isVisibleTo(w) and not w.vpd_spinbox.isVisibleTo(w)
    w2 = loaded.device_widgets["dds.d1_3d_c"]
    assert not w2._has_dac() and w2._amp_unit == "Amp"


# --- unit switchers ------------------------------------------------------------

def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_amp_voltage_combo(loaded, fake_settings):
    w = loaded.device_widgets["dds.imaging"]
    combo = w.amp_unit_combo
    assert combo.isVisibleTo(w)
    assert _items(combo) == ["Amp", "V"]
    assert combo.currentText() == "V"
    assert w.vpd_spinbox.suffix() == "" and w.amp_spinbox.suffix() == ""

    combo.setCurrentText("Amp")
    assert w._amp_unit == "Amp"
    assert w.amp_spinbox.isVisibleTo(w) and not w.vpd_spinbox.isVisibleTo(w)
    assert fake_settings.store["amp_unit/imaging"] == "Amp"
    assert not w.has_unsaved_changes           # a unit switch is not an edit

    w.on_amp_unit_changed("V")                 # right-click path keeps it in sync
    assert combo.currentText() == "V"


def test_every_card_has_both_combos(loaded):
    # A channel that cannot switch still shows the combos (so every card
    # looks the same), with a single entry.
    w = loaded.device_widgets["dds.d1_3d_c"]  # no DAC, no transition
    assert w.amp_unit_combo.isVisibleTo(w) and _items(w.amp_unit_combo) == ["Amp"]
    assert w.freq_unit_combo.isVisibleTo(w) and _items(w.freq_unit_combo) == ["MHz"]
    assert w.amp_spinbox.suffix() == "" and w.freq_spinbox.suffix() == ""
    w.on_amp_unit_changed("V")
    assert w._amp_unit == "Amp" and w.amp_unit_combo.currentText() == "Amp"
    # and a DAC channel without a transition gets MHz alone
    assert _items(loaded.device_widgets["dds.imaging"].freq_unit_combo) == ["MHz"]


def test_combos_line_up_and_keep_card_height(loaded):
    cards = [w for k, w in loaded.device_widgets.items() if k.startswith("dds.")]
    widths = {c.width() for w in cards for c in (w.freq_unit_combo, w.amp_unit_combo)}
    assert len(widths) == 1
    for w in cards:
        assert w.freq_unit_combo.height() <= w.freq_spinbox.sizeHint().height()
        assert w.amp_unit_combo.height() <= w.amp_spinbox.sizeHint().height()


def test_cards_do_not_stretch(loaded, qapp):
    # Spare tab height goes below the grid, not into the cards.
    loaded.resize(1400, 1400)
    loaded.show()
    loaded.tab_widget.setCurrentIndex(0)
    qapp.processEvents()
    for key, w in loaded.device_widgets.items():
        if key.startswith("dds."):
            assert w.height() == w.sizeHint().height(), key


def test_tall_composite_page_scrolls_on_its_own(qapp, monkeypatch):
    # A tab widget's minimum height is its tallest page's; the Composite page
    # scrolls itself so it cannot inflate the window and the DDS cards.
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import QScrollArea, QWidget
    from waxx.util.guis import composite_panel

    class TallPanel(QWidget):
        hazards_changed = pyqtSignal()

        def __init__(self, *a, **k):
            super().__init__()
            self.setMinimumHeight(2000)

        def set_config(self, config):
            pass

        def hazards(self):
            return []

        def ops_allowed(self):
            return False, "stand-in"

        def set_busy(self, seconds):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(composite_panel, "CompositePanel", TallPanel)
    for name in ("_setup_update_sender", "_setup_state_listener", "_setup_state_worker",
                 "setup_status_checker", "setup_timer", "request_state"):
        monkeypatch.setattr(dc.DeviceStateGUI, name, lambda self, *a, **k: None)
    g = dc.DeviceStateGUI(composite_devices=[object()])
    try:
        page = g.tab_widget.widget(g.tab_widget.count() - 1)
        assert isinstance(page, QScrollArea) and page.widget() is g.composite_panel
        assert g.tab_widget.minimumSizeHint().height() < 1000
    finally:
        g.close()


class _FakeDDS:
    """Linear stand-in for DDS.frequency_to_detuning / detuning_to_frequency
    (both in Hz): 0 Γ at 100 MHz, 6 MHz per Γ."""
    transition = "D2"
    frequency, amplitude, v_pd = 100e6, 0.3, 4.0

    def frequency_to_detuning(self, f_hz):
        return (f_hz - 100e6) / 6e6

    def detuning_to_frequency(self, gamma):
        return 100e6 + 6e6 * gamma


class _FakeFrame:
    def __init__(self):
        self.probe = _FakeDDS()
        self.dds_array = [[self.probe]]


def _transition_dds():
    cfg = _dds(0, 0)
    cfg["frequency"] = 112e6
    cfg["transition"] = "D2"
    return cfg


def test_gamma_combo_converts_and_sends_hz(qapp, fake_settings):
    w = dc.DDSWidget("probe", _transition_dds(), dds_frame_obj=_FakeFrame())
    combo = w.freq_unit_combo
    assert _items(combo) == ["MHz", "Γ"]
    assert combo.currentText() == "MHz" and w.freq_spinbox.suffix() == ""
    assert w.freq_spinbox.value() == pytest.approx(112.0)

    combo.setCurrentText("Γ")
    assert w._freq_unit == "Γ"
    assert w.freq_spinbox.value() == pytest.approx(2.0)
    assert w.get_updated_config()["frequency"] == pytest.approx(112e6)
    assert fake_settings.store["freq_unit/probe"] == "Γ"
    assert not w.has_unsaved_changes

    w.update_from_config(dict(w.device_config, frequency=106e6))
    assert w.freq_spinbox.value() == pytest.approx(1.0)

    combo.setCurrentText("MHz")
    assert w.freq_spinbox.value() == pytest.approx(106.0)
    assert w.get_updated_config()["frequency"] == pytest.approx(106e6)


def test_saved_gamma_preference_restores_combo(qapp, fake_settings):
    fake_settings.store["freq_unit/probe"] = "Γ"
    w = dc.DDSWidget("probe", _transition_dds(), dds_frame_obj=_FakeFrame())
    assert w._freq_unit == "Γ" and w.freq_unit_combo.currentText() == "Γ"
    assert w.freq_spinbox.value() == pytest.approx(2.0)


def test_default_button_shows_gamma_in_combo(qapp):
    w = dc.DDSWidget("probe", _transition_dds(), dds_frame_obj=_FakeFrame())
    w.on_default_undo_clicked()                 # stage the frame default
    assert w.freq_unit_combo.currentText() == "Γ"
    assert w.freq_spinbox.value() == pytest.approx(0.0)
    w.on_default_undo_clicked()                 # undo restores unit and value
    assert w.freq_unit_combo.currentText() == "MHz"
    assert w.freq_spinbox.value() == pytest.approx(112.0)


def test_transition_without_frame_offers_no_gamma(qapp, fake_settings):
    # Without a DDS object the MHz <-> Γ conversion is impossible, so Γ is
    # not offered (and a saved Γ preference is ignored) rather than showing
    # an MHz value labelled Γ.
    fake_settings.store["freq_unit/probe"] = "Γ"
    w = dc.DDSWidget("probe", _transition_dds(), dds_frame_obj=None)
    assert _items(w.freq_unit_combo) == ["MHz"]
    assert w._freq_unit == "MHz" and w.freq_unit_combo.currentText() == "MHz"
    w.on_freq_unit_changed("Γ")
    assert w._freq_unit == "MHz"
    assert w.freq_spinbox.value() == pytest.approx(112.0)


def test_failed_conversion_keeps_unit(qapp):
    frame = _FakeFrame()
    w = dc.DDSWidget("probe", _transition_dds(), dds_frame_obj=frame)

    def boom(_):
        raise RuntimeError("no conversion")
    frame.probe.frequency_to_detuning = boom
    w.freq_unit_combo.setCurrentText("Γ")
    assert w._freq_unit == "MHz"
    assert w.freq_unit_combo.currentText() == "MHz"
    assert w.freq_spinbox.value() == pytest.approx(112.0)


# --- changes log -------------------------------------------------------------

def test_no_recent_changes_strip(gui):
    assert not hasattr(gui, "recent_list")
    assert gui.changes_button.text() == "Changes"


def test_changes_popout_window(loaded, qapp):
    loaded._record_change("dac", "dac_ch0", {"voltage": 1.0}, {"voltage": 1.5})
    assert loaded.changes_button.text() == "Changes (1)"
    assert loaded._changes_window is None

    loaded.show_changes_log()
    win = loaded._changes_window
    assert win is not None and win.parent() is None and win.isWindow()
    assert win.list.count() == 1
    assert "1.000 V → 1.500 V" in win.list.item(0).text()

    loaded._record_change("ttl", "a", {"ttl_state": 0}, {"ttl_state": 1})
    assert win.list.count() == 2
    assert loaded.changes_button.text() == "Changes (2)"

    loaded.show_changes_log()
    assert loaded._changes_window is win        # raised, not duplicated

    win._clear()
    assert len(loaded._changes) == 0
    assert loaded.changes_button.text() == "Changes"

    loaded._record_change("dds", "imaging", {"frequency": 110e6}, {"frequency": 111e6})
    win.close()
    qapp.processEvents()
    assert loaded._changes_window is None
    assert len(loaded._changes) == 1            # the buffer outlives the window
    assert "ui/changes_geometry" in FakeSettings.store

    loaded.show_changes_log()
    assert loaded._changes_window.list.count() == 1
    loaded._changes_window.close()
    qapp.processEvents()


def test_changes_ring_buffer_cap(loaded):
    for _ in range(dc.CHANGES_LOG_MAX_ROWS + 20):
        loaded._record_change("ttl", "a", {"ttl_state": 0}, {"ttl_state": 1})
    assert len(loaded._changes) == dc.CHANGES_LOG_MAX_ROWS


def test_main_window_close_closes_popout(loaded, qapp):
    loaded._record_change("ttl", "a", {"ttl_state": 0}, {"ttl_state": 1})
    loaded.show_changes_log()
    closed = []
    loaded._changes_window.closed.connect(lambda: closed.append(True))
    loaded.close()
    qapp.processEvents()          # the pop-out is delete-on-close: gone now
    assert loaded._changes_window is None
    assert closed == [True]
