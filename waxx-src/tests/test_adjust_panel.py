"""The liveOD Adjust panel: units are display-only, values stay SI, and the
panel cannot move a param by accident.

Offscreen Qt only — no window is shown, no socket is opened, no hardware is
touched. QSettings is faked so the real registry is never written.
"""
import os
import types

import pytest

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication

from waxx.util.live_od.gui import adjust_panel as ap


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class FakeSettings:
    """QSettings stand-in backed by a dict shared by every instance."""

    store: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def value(self, key, default=None):
        return FakeSettings.store.get(key, default)

    def setValue(self, key, value):
        FakeSettings.store[key] = value


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    FakeSettings.store = {}
    monkeypatch.setattr(ap, "QSettings", FakeSettings)
    return FakeSettings


def spec(key="t_tof", min_val=20.e-6, max_val=20.e-3, step=1.e-6,
         current_val=20.e-6, dtype='float', unit="µs"):
    out = {'key': key, 'min_val': min_val, 'max_val': max_val, 'step': step,
           'dtype': dtype, 'current_val': current_val}
    if unit is not None:
        out['unit'] = unit
    return out


# ---------------------------------------------------------------- units

def test_the_row_displays_in_its_unit_and_holds_the_si_value(qapp):
    row = ap.AdjustParamRow(spec())
    assert row.unit == "µs"
    assert row._spinbox.value() == pytest.approx(20.0)      # shown as 20 µs
    assert row.value() == pytest.approx(2.e-5)              # held as SI
    assert row._spinbox.minimum() == pytest.approx(20.0)
    assert row._spinbox.maximum() == pytest.approx(20000.0)


def test_editing_publishes_an_exact_si_value(qapp):
    """20 µs -> 2e-05 exactly, not 1.9999999999999998e-05."""
    row = ap.AdjustParamRow(spec(current_val=100.e-6))
    published = []
    row.value_changed.connect(lambda key, value: published.append((key, value)))

    row._spinbox.setValue(20.0)                             # user types 20 (µs)

    assert published == [("t_tof", 2.e-5)]
    assert repr(published[0][1]) == "2e-05"
    assert row.value() == 2.e-5


def test_switching_unit_rescales_the_display_and_publishes_nothing(qapp):
    row = ap.AdjustParamRow(spec(current_val=2.e-3))
    published = []
    row.value_changed.connect(lambda *a: published.append(a))

    row.set_unit("ms")

    assert published == []                                  # display only
    assert row.unit == "ms"
    assert row._spinbox.value() == pytest.approx(2.0)
    assert row.value() == pytest.approx(2.e-3)
    assert row._spinbox.maximum() == pytest.approx(20.0)    # range came along
    # and editing in the new unit still publishes SI
    row._spinbox.setValue(5.0)
    assert published == [("t_tof", 5.e-3)]


def test_a_param_with_no_family_has_no_dropdown(qapp):
    volts = ap.AdjustParamRow(spec(key="v_xshim_current", min_val=0., max_val=9.9,
                                   step=0.1, current_val=4.0, unit="V"))
    assert volts._unit_box is None
    assert volts.value() == pytest.approx(4.0)

    times = ap.AdjustParamRow(spec())
    assert [times._unit_box.itemText(i) for i in range(times._unit_box.count())] \
        == ["ns", "µs", "ms", "s"]


def test_a_spec_from_before_units_is_guessed_from_its_key(qapp):
    """An older server sends specs with no 'unit' key at all."""
    row = ap.AdjustParamRow(spec(unit=None))
    assert row.unit == "µs"
    assert row.value() == pytest.approx(2.e-5)


def test_small_values_survive_the_spinbox(qapp):
    """decimals=6 used to round an 8 ns param straight to zero."""
    row = ap.AdjustParamRow(spec(key="t_rtio", min_val=0., max_val=1.e-6,
                                 step=1.e-9, current_val=8.e-9, unit="ns"))
    assert row.value() == pytest.approx(8.e-9)
    assert row._spinbox.value() == pytest.approx(8.0)

    unitless = ap.AdjustParamRow(spec(key="mystery", min_val=0., max_val=1.e-3,
                                      step=1.e-9, current_val=1.25e-7, unit=""))
    assert unitless.value() == pytest.approx(1.25e-7)


# ---------------------------------------------------------------- safety

def _wheel_at(widget):
    return QWheelEvent(QPointF(5, 5), widget.mapToGlobal(QPointF(5, 5)),
                       QPoint(0, 0), QPoint(0, 120),
                       Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                       Qt.ScrollPhase.NoScrollPhase, False)


def test_the_wheel_does_nothing_unless_the_spinbox_has_focus(qapp):
    """Scrolling the list must not move a live experiment param."""
    row = ap.AdjustParamRow(spec(current_val=100.e-6))
    published = []
    row.value_changed.connect(lambda *a: published.append(a))

    row._spinbox.clearFocus()
    row._spinbox.wheelEvent(_wheel_at(row._spinbox))
    assert published == []
    assert row.value() == pytest.approx(1.e-4)

    row._spinbox.setFocus()
    if row._spinbox.hasFocus():          # offscreen platforms may refuse focus
        row._spinbox.wheelEvent(_wheel_at(row._spinbox))
        assert published, "a focused spinbox should still step on the wheel"


def test_a_shot_update_is_ignored_mid_edit(qapp):
    row = ap.AdjustParamRow(spec(current_val=100.e-6))
    row._on_text_edited("3")                                # user is typing
    row.update_value(50.e-6)
    assert row.value() == pytest.approx(1.e-4)

    row._on_editing_finished()
    row.update_value(50.e-6)
    assert row.value() == pytest.approx(5.e-5)
    assert row._spinbox.value() == pytest.approx(50.0)      # in µs


def test_changed_values_are_marked_and_reset_reverts_them(qapp):
    row = ap.AdjustParamRow(spec(current_val=100.e-6))
    assert not row.is_changed() and not row._reset_btn.isEnabled()

    row._spinbox.setValue(200.0)
    assert row.is_changed() and row._reset_btn.isEnabled()
    assert row._label.styleSheet() == ap._CHANGED_STYLE

    row.reset()
    assert not row.is_changed() and row._label.styleSheet() == ""
    assert row.value() == pytest.approx(1.e-4)


# ---------------------------------------------------------------- panel

def test_the_panel_copies_si_values_written_like_source(qapp):
    panel = ap.AdjustPanel()
    panel.populate([
        spec(),
        spec(key="i_mot", min_val=0., max_val=80., step=1., current_val=60.,
             unit="A"),
        spec(key="frequency_detuned_imaging", min_val=0., max_val=100.e6,
             step=1.e6, current_val=24.e6, unit="MHz"),
    ])

    panel._copy_params_to_clipboard()
    lines = [l.strip() for l in QApplication.clipboard().text().splitlines()]
    assert lines == [
        "self.p.t_tof = 20.e-6",
        "self.p.i_mot = 60.",
        "self.p.frequency_detuned_imaging = 24.e6",
    ]

    panel._rows["t_tof"].set_checked(False)
    panel._expt_params_btn.setChecked(False)
    panel._copy_params_to_clipboard()
    lines = [l.strip() for l in QApplication.clipboard().text().splitlines()]
    assert lines == ["self.i_mot = 60.", "self.frequency_detuned_imaging = 24.e6"]


def test_the_copied_value_follows_the_unit_the_row_shows(qapp):
    panel = ap.AdjustPanel()
    panel.populate([spec(current_val=2.e-3)])
    panel._rows["t_tof"].set_unit("ms")
    panel._copy_params_to_clipboard()
    assert QApplication.clipboard().text() == "self.p.t_tof = 2.e-3"


def test_filtering_hides_rows_without_touching_their_values(qapp):
    panel = ap.AdjustPanel()
    panel.populate([spec(), spec(key="i_mot", min_val=0., max_val=80., step=1.,
                                 current_val=60., unit="A")])
    panel._filter_box.setText("mot")
    # isHidden, not isVisible: the panel itself is never shown in the test
    assert panel._rows["t_tof"].isHidden() and not panel._rows["i_mot"].isHidden()
    panel._filter_box.setText("")
    assert not panel._rows["t_tof"].isHidden()
    panel._copy_params_to_clipboard()
    assert "t_tof" in QApplication.clipboard().text()


def test_a_new_run_keeps_the_units_and_checkboxes_the_panel_was_set_up_with(qapp):
    panel = ap.AdjustPanel()
    panel.populate([spec(), spec(key="i_mot", min_val=0., max_val=80., step=1.,
                                 current_val=60., unit="A")])
    panel._rows["t_tof"].set_unit("ms")
    panel._rows["i_mot"].set_checked(False)

    panel.populate([spec(), spec(key="i_mot", min_val=0., max_val=80., step=1.,
                                 current_val=60., unit="A")])

    assert panel._rows["t_tof"].unit == "ms"
    assert panel._rows["i_mot"].is_checked() is False
    assert FakeSettings.store["adjust/units/t_tof"] == "ms"

    # and a fresh panel in a later session picks the stored unit back up
    later = ap.AdjustPanel()
    later.populate([spec()])
    assert later._rows["t_tof"].unit == "ms"


def test_a_stored_unit_from_another_family_is_ignored(qapp):
    FakeSettings.store["adjust/units/t_tof"] = "MHz"
    panel = ap.AdjustPanel()
    panel.populate([spec()])
    assert panel._rows["t_tof"].unit == "µs"


def test_an_empty_run_clears_the_panel(qapp):
    panel = ap.AdjustPanel()
    panel.populate([spec()])
    assert panel.param_count() == 1 and panel._empty_label.isHidden()
    panel.populate([])
    assert panel.param_count() == 0 and not panel._empty_label.isHidden()
    assert panel._scroll.isHidden() and not panel._copy_btn.isEnabled()


def test_broadcast_values_reach_the_rows_in_si(qapp):
    panel = ap.AdjustPanel()
    panel.populate([spec()])
    published = []
    panel.value_changed_signal.connect(lambda *a: published.append(a))

    panel.update_values({"t_tof": 5.e-3})

    assert panel._rows["t_tof"].value() == pytest.approx(5.e-3)
    assert panel._rows["t_tof"]._spinbox.value() == pytest.approx(5000.0)  # µs
    assert published == []           # a broadcast must not echo back


def test_reset_all_changed_only_touches_changed_rows(qapp):
    panel = ap.AdjustPanel()
    panel.populate([spec(), spec(key="i_mot", min_val=0., max_val=80., step=1.,
                                 current_val=60., unit="A")])
    published = []
    panel.value_changed_signal.connect(lambda key, _v: published.append(key))

    panel._rows["t_tof"].update_value(1.e-3)
    panel.reset_all_changed()

    assert published == ["t_tof"]
    assert panel._rows["t_tof"].value() == pytest.approx(2.e-5)


# ---------------------------------------------------------------- spec dialog

def test_the_spec_dialog_takes_display_units_and_gives_back_si(qapp):
    dialog = ap.AdjustSpecDialog(spec(), unit="µs")
    assert dialog._min_sb.value() == pytest.approx(20.0)
    assert dialog._max_sb.value() == pytest.approx(20000.0)

    dialog._min_sb.setValue(50.0)
    dialog._max_sb.setValue(500.0)
    dialog._step_sb.setValue(5.0)
    assert dialog.get_values() == (5.e-5, 5.e-4, 5.e-6)


def test_applying_a_spec_moves_the_range_not_the_value(qapp):
    row = ap.AdjustParamRow(spec(current_val=100.e-6))
    row.apply_spec(50.e-6, 500.e-6, 5.e-6)
    assert row.value() == pytest.approx(1.e-4)
    assert row._spinbox.minimum() == pytest.approx(50.0)
    assert row._spinbox.maximum() == pytest.approx(500.0)
    assert row._spinbox.singleStep() == pytest.approx(5.0)


# ---------------------------------------------------------------- experiment side

def test_adjust_gives_every_spec_a_unit():
    """Scanner.adjust detects the unit host-side, where ExptParams is."""
    import types as _types
    from waxx.base.scanner import Scanner

    class FakeParams:
        def __init__(self):
            self.t_probe = 1.e-3      # s
            self.t_tof = 20.e-6
            self.i_mot = 60.
            self.n_repeats = 3

    stub = _types.SimpleNamespace(params=FakeParams(), _adjust_specs=[], xvarnames=[])
    stub._detect_adjust_unit = _types.MethodType(Scanner._detect_adjust_unit, stub)
    adjust = _types.MethodType(Scanner.adjust, stub)

    adjust('t_tof', min_val=20.e-6, max_val=20.e-3)
    adjust('t_probe', min_val=0., max_val=10.e-3)
    adjust('i_mot', min_val=0., max_val=80.)
    adjust('n_repeats', min_val=1, max_val=10)
    adjust('t_tof_forced', min_val=0., max_val=1., default_val=0.5, unit='ms')

    units = {s.key: s.unit for s in stub._adjust_specs}
    assert units == {'t_tof': 'µs', 't_probe': 'ms', 'i_mot': 'A',
                     'n_repeats': '', 't_tof_forced': 'ms'}
    # and the unit rides along in the dict that goes over the wire
    assert stub._adjust_specs[0].to_dict()['unit'] == 'µs'
