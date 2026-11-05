"""The run queue panel's columns: moved by dragging a header, and -- given a
``settings`` -- the order, widths, shown set and the filter remembered on
every change and restored by the next panel.

Offscreen Qt.  Settings are an INI file in tmp_path (never the registry); no
requester is called (the panel is never shown or fed)."""
import os

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from waxx.util.guis import run_queue_panel as rqp


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ini(tmp_path):
    return str(tmp_path / "run_queue_panel.ini")


def _panel(ini_path=None):
    settings = QSettings(ini_path, QSettings.Format.IniFormat) if ini_path else None
    return rqp.RunQueuePanel(lambda obj: {"status": "ok"}, synchronous=True, settings=settings)


def _col(key):
    return rqp.COLUMN_KEYS.index(key)


def test_headers_can_be_dragged(qapp):
    p = _panel()
    assert p.table.horizontalHeader().sectionsMovable()
    p.shutdown()


def test_order_widths_shown_and_filter_come_back(qapp, ini):
    p = _panel(ini)
    header = p.table.horizontalHeader()
    header.moveSection(header.visualIndex(_col("run_id")), 0)
    header.resizeSection(_col("label"), 333)
    p.set_column_shown(_col("path"), True)
    p.set_column_shown(_col("est"), False)
    p.set_filter("pending")
    assert p._save_timer.isActive()            # written after the debounce ...
    p.shutdown()                               # ... or at shutdown, whichever is first

    q = _panel(ini)
    header = q.table.horizontalHeader()
    assert header.visualIndex(_col("run_id")) == 0
    assert header.sectionSize(_col("label")) == 333
    assert not q.table.isColumnHidden(_col("path"))
    assert q.table.isColumnHidden(_col("est"))
    assert q.filter == "pending"
    assert q.filter_combo.currentIndex() == rqp.FILTERS.index("pending")
    q.shutdown()


def test_the_last_filters_n_comes_back(qapp, ini):
    p = _panel(ini)
    assert p.filter == "last" and p.last_n == 15                 # the defaults
    p.last_spin.setValue(40)
    p.shutdown()
    q = _panel(ini)
    assert q.filter == "last" and q.last_n == 40 and q.last_spin.value() == 40
    assert q.model.last_n == 40
    q.shutdown()


def test_the_debounced_save_writes_without_a_shutdown(qapp, ini):
    p = _panel(ini)
    header = p.table.horizontalHeader()
    header.moveSection(header.visualIndex(_col("state")), 0)
    p.save_columns()                           # what the timer calls
    q = _panel(ini)
    assert q.table.horizontalHeader().visualIndex(_col("state")) == 0
    p.shutdown()
    q.shutdown()


def test_default_columns_resets_order_widths_and_shown(qapp, ini):
    p = _panel(ini)
    header = p.table.horizontalHeader()
    header.moveSection(header.visualIndex(_col("waiting")), 0)
    header.resizeSection(_col("label"), 300)
    p.set_column_shown(_col("due"), True)
    p.default_columns()
    for i, key in enumerate(rqp.COLUMN_KEYS):
        assert header.visualIndex(i) == i
        assert p.table.isColumnHidden(i) == (key in rqp.DEFAULT_HIDDEN)
    assert header.sectionSize(_col("label")) == rqp.DEFAULT_WIDTHS["label"]
    p.shutdown()
    q = _panel(ini)
    assert q.table.horizontalHeader().visualIndex(_col("waiting")) == _col("waiting")
    q.shutdown()


def test_a_layout_saved_for_other_columns_is_not_applied(qapp, ini):
    p = _panel(ini)
    header = p.table.horizontalHeader()
    header.moveSection(header.visualIndex(_col("run_id")), 0)
    p.shutdown()
    QSettings(ini, QSettings.Format.IniFormat).setValue("columns/keys", "id,state")
    q = _panel(ini)
    assert q.table.horizontalHeader().visualIndex(_col("run_id")) == _col("run_id")
    q.shutdown()


def test_an_unknown_saved_filter_falls_back_to_last(qapp, ini):
    QSettings(ini, QSettings.Format.IniFormat).setValue("filter", "bogus")
    p = _panel(ini)
    assert p.filter == "last"
    p.shutdown()


def test_without_settings_nothing_is_remembered(qapp):
    p = _panel()
    header = p.table.horizontalHeader()
    header.moveSection(header.visualIndex(_col("run_id")), 0)
    p.set_filter("pending")
    assert not p._save_timer.isActive()
    p.shutdown()
    q = _panel()
    assert q.filter == "last"
    assert q.table.horizontalHeader().visualIndex(_col("run_id")) == _col("run_id")
    q.shutdown()


def test_follow_is_on_by_default_and_remembered(qapp, ini):
    p = _panel(ini)
    assert p.follow_new
    p.set_follow_new(False)
    assert not p.follow_button.isChecked()
    p.shutdown()
    q = _panel(ini)
    assert not q.follow_new and not q.follow_button.isChecked()
    q.shutdown()


def _shown(p):
    """The shown columns' keys, left to right."""
    header = p.table.horizontalHeader()
    keys = sorted(rqp.COLUMN_KEYS, key=lambda k: header.visualIndex(_col(k)))
    return [k for k in keys if not p.table.isColumnHidden(_col(k))]


def test_the_built_in_columns(qapp):
    p = _panel()
    assert _shown(p) == ["position", "state", "file", "label", "run_id", "est"]
    assert dict(rqp.COLUMNS)["file"] == "file" and dict(rqp.COLUMNS)["path"] == "filepath"
    assert p.table.isColumnHidden(_col("path"))
    p.shutdown()


def test_a_layout_saved_before_the_file_column_gives_way_to_the_new_defaults(qapp, ini):
    """Every PC's old saved layout names other columns: it starts from these."""
    p = _panel(ini)
    p.set_column_shown(_col("waiting"), True)
    p.shutdown()
    QSettings(ini, QSettings.Format.IniFormat).setValue(
        "columns/keys", "position,id,state,owner,label,expt_class,path,run_id")
    q = _panel(ini)
    assert _shown(q) == ["position", "state", "file", "label", "run_id", "est"]
    q.shutdown()


def test_saved_default_columns(qapp, ini):
    p = _panel(ini)
    p.set_column_shown(_col("owner"), True)
    p.set_column_shown(_col("est"), False)
    p.table.horizontalHeader().resizeSection(_col("label"), 333)
    p.save_default_columns()
    assert "default" in p.message.text()
    mine = _shown(p)
    p.set_column_shown(_col("owner"), False)
    p.set_column_shown(_col("due"), True)
    assert _shown(p) != mine
    p.default_columns()                         # Default columns: the saved one
    assert _shown(p) == mine
    assert p.table.horizontalHeader().sectionSize(_col("label")) == 333
    p.shutdown()

    q = _panel(ini)                             # it outlives the panel
    q.set_column_shown(_col("due"), True)
    q.default_columns()
    assert _shown(q) == mine
    q.forget_default_columns()                  # back to the columns as built
    assert _shown(q) == ["position", "state", "file", "label", "run_id", "est"]
    q.set_column_shown(_col("due"), True)
    q.default_columns()
    assert _shown(q) == ["position", "state", "file", "label", "run_id", "est"]
    q.shutdown()


def test_a_new_panel_with_no_layout_starts_from_the_saved_default(qapp, ini):
    p = _panel(ini)
    p.set_column_shown(_col("owner"), True)
    p.save_default_columns()
    p.shutdown()
    settings = QSettings(ini, QSettings.Format.IniFormat)
    settings.remove("columns/state")
    settings.remove("columns/keys")
    settings.sync()
    q = _panel(ini)
    assert not q.table.isColumnHidden(_col("owner"))
    q.shutdown()


def test_the_header_menu(qapp, ini):
    p = _panel(ini)
    titles = [a.text() for a in p.header_menu().actions() if a.text()]
    for title in ("filepath", "Save as default columns", "Default columns", "Follow new jobs",
                  "Refresh list"):
        assert title in titles
    assert "Back to the built-in columns" not in titles       # nothing saved yet
    p.save_default_columns()
    assert "Back to the built-in columns" in [a.text() for a in p.header_menu().actions()]
    p.shutdown()
    q = _panel()                                              # no settings: nowhere to save
    save = [a for a in q.header_menu().actions() if a.text() == "Save as default columns"]
    assert not save[0].isEnabled()
    q.shutdown()
