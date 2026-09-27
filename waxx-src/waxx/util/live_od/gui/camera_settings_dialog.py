"""One camera's settings in liveOD (the ⚙ of ``CameraControl``), when liveOD's
camera host owns the cameras.

    [x] Persist settings into runs     ON since 14:02:11: gain=30, vs_speed=1
    +-------+----------+-----+
    | Live  | Advanced | Run |          (fields from the camera's schema)
    +-------+----------+-----+
    Live settings. A run applies camera_params in full; with Persist on (red) the
    marked fields are applied on top and recorded.

**Live** and **Advanced** are beacon's ``SettingsForm`` over the camera's schema
(``host.describe(key)["schema"]``, i.e. ``schema.to_wire(category)``: frame
transfer and the other hidden settings are never shown).  A field commits when
its edit is finished, through ``host.set_live`` off the GUI thread; a refusal
puts the field back and says why under the form.  A field being edited, or
waiting for its own answer, is never overwritten by a snapshot: a chip says
another program changed it, and "reload" takes that value.  Andor EM gain above
the live cap (100) asks for "Unlock EM gain above 100" first.

**Run** is read-only: per run setting, the last run's request (camera_params),
the persisted value, what the camera has now (applied), and where that reading
comes from.  While a run holds the camera everything is read-only and the dialog
opens on this tab.

**Persist settings into runs** (PLAN C6, INTERFACES D-a) is outside the tabs.
Turning it on asks first (default Cancel), listing exactly the fields that will
be carried into every run -- Andor gain, hs_speed, vs_speed, vs_amp, preamp,
baseline_clamp; Basler gain; never exposure_time or a field a run owns.  Turning
it off does not ask.  It cannot change while a run holds the camera.  The host
keeps Persist (off after a liveOD restart); this dialog only shows it.

Host methods used: ``describe``, ``snapshot``, ``set_live``, ``set_persist``.
"""

import collections
import math
import threading
import time
from typing import Mapping, Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QFormLayout, QHBoxLayout,
                             QHeaderView, QLabel, QMessageBox, QSizePolicy, QTabWidget,
                             QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from beacon.camera.schema import (ANDOR_LIVE_EM_GAIN_CAP, NEVER_PERSIST, category_from_wire,
                                  get_category)
from beacon.camera.viewer.settings_bar import SettingsForm, format_value

from waxx.util.live_od.gui.camera_control import PERSIST_FILL, RUN_PHASES, camera_entries

FOOTER_TEXT = ("Live settings. A run applies camera_params in full; with Persist on (red) the "
               "marked fields are applied on top and recorded.")
#: never given a field, whatever a schema says (INTERFACES D-g)
NEVER_SHOWN = ("frame_transfer",)
#: keys a live apply takes that are not settings
CONTROL_KEYS = ("em_gain_unlocked",)
PERSIST_MARK = "◆ "
READBACK_WORDS = {"hw": "read from the camera", "driver_cache": "driver's record",
                  "commanded": "as commanded (no readback)"}
RUN_COLUMNS = ("Setting", "camera_params (request)", "Persisted", "Applied", "Readback")


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _plain(value):
    """A readback (``Readback`` or ``{"value": ...}``) as its plain value."""
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return getattr(value, "value", value)


def shown_schema(wire: Mapping) -> dict:
    """``wire`` without the settings no GUI shows (and the constraints on them)."""
    wire = dict(wire)
    settings = [dict(s) for s in wire.get("settings", ()) if s.get("key") not in NEVER_SHOWN
                and s.get("group") != "hidden"]
    keys = {s["key"] for s in settings}
    wire["settings"] = settings
    wire["constraints"] = [c for c in wire.get("constraints", ())
                           if all(cond[0] in keys for cond in c.get("when", ()))]
    return wire


def persist_fields(category, persistable=None) -> list:
    """The fields Persist carries into runs: the category's ``persistable`` ones
    (and those the host says), never exposure_time or a run-owned field (D-a)."""
    keys = [k for k in category.persistable_keys() if k not in NEVER_PERSIST
            and category.setting(k).run == "param"]
    if persistable is not None:
        allowed = set(persistable)
        keys = [k for k in keys if k in allowed]
    return keys


def _ask(parent, title: str, text: str) -> bool:
    box = QMessageBox(QMessageBox.Icon.Warning, title, text,
                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, parent)
    box.setDefaultButton(QMessageBox.StandardButton.Cancel)
    return box.exec() == QMessageBox.StandardButton.Yes


class _Relay(QObject):
    """Brings a host call's result back to the GUI thread."""
    done = pyqtSignal(object, object)       # values sent, (ok, result or exception)


class CameraSettingsDialog(QDialog):
    """See the module docstring.

    ``CameraSettingsDialog(camera_key, host, parent=None, *, snapshot_signal=None,
    confirm=None, async_calls=True)``.  ``snapshot_signal``: a signal carrying the
    host's snapshots on the GUI thread (``HostQtBridge.snapshot_changed``); without
    one the owner calls ``set_snapshot``.  ``confirm(title, text) -> bool`` asks
    before Persist is turned on and before EM gain is unlocked (default: a
    Yes/Cancel box defaulting to Cancel).  ``async_calls=False`` calls
    ``host.set_live`` on the GUI thread (tests).
    """

    persist_changed = pyqtSignal(str, bool)         # camera_key, on (the host accepted)

    def __init__(self, camera_key: str, host, parent=None, *, snapshot_signal=None,
                 confirm=None, async_calls: bool = True):
        super().__init__(parent)
        self.key = str(camera_key)
        self.host = host
        self._confirm = confirm or (lambda title, text: _ask(self, title, text))
        self._async = bool(async_calls)
        self._jobs: collections.deque = collections.deque()     # (values, sent) in edit order
        self._jobs_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._pending: dict = {}            # key -> value sent, answer not back yet
        self._stale: dict = {}              # key -> value another program set meanwhile
        self._entry: dict = {}
        self._last_settings: dict = {}
        self._locked = False
        self._lock_applied = None
        self._persist_on = False
        self._label_look: dict = {}         # key -> the form label's own (text, style, tip)
        self._relay = _Relay(self)
        self._relay.done.connect(self._on_set_live_done)

        self.setWindowTitle(f"{self.key} — camera settings")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)

        self.describe = self._describe()
        wire = shown_schema(self.describe.get("schema") or {"name": "?", "label": "?",
                                                             "settings": []})
        self.category = category_from_wire(wire)
        try:
            full = get_category(self.category.name)
        except KeyError:
            full = self.category
        self.persist_keys = persist_fields(full, self.describe.get("persistable"))
        self._andor = self.category.name == "andor_emccd" and self.category.has("gain")
        dynamic = dict(self.describe.get("dynamic") or {})
        values = dict(self.describe.get("settings") or {}) or dict(
            self.describe.get("live_profile") or {})

        # -- header: Persist, outside the tabs ------------------------------------
        self.persist_box = QCheckBox("Persist settings into runs")
        self.persist_box.setToolTip(
            "Carry this camera's live values of " + (", ".join(self.persist_keys) or "nothing")
            + " into every run on it, on top of the experiment's camera_params (each "
              "difference is recorded in the run file). Never exposure_time or a field a "
              "run owns. Off after a liveOD restart.")
        self.persist_box.toggled.connect(self._on_persist_toggled)
        self.persist_status = QLabel("")
        self.persist_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.persist_status.setWordWrap(True)
        self.persist_reason = QLabel("")
        self.persist_reason.setWordWrap(True)
        self.persist_reason.hide()
        header = QHBoxLayout()
        header.addWidget(self.persist_box)
        header.addWidget(self.persist_status, 1)

        # -- tabs ----------------------------------------------------------------------
        self.live_form = SettingsForm(wire, values, editable=True, on_change=self._commit,
                                      dynamic=dynamic, groups=("basic", "status"), owner=True,
                                      confirm=self._confirm_form, context="live")
        self.advanced_form = SettingsForm(wire, values, editable=True, on_change=self._commit,
                                          dynamic=dynamic, groups=("advanced",), owner=True,
                                          confirm=self._confirm_form, context="live")
        self._forms = (self.live_form, self.advanced_form)
        live_page = QWidget()
        live_layout = QVBoxLayout(live_page)
        live_layout.addWidget(self.live_form)
        self.unlock_box = QCheckBox(f"Unlock EM gain above {ANDOR_LIVE_EM_GAIN_CAP} (live)")
        self.unlock_box.setToolTip(
            f"Live EM gain is capped at {ANDOR_LIVE_EM_GAIN_CAP} unless this is ticked; never "
            f"above 300, never advanced mode. EM gain on bright light ages the gain register.")
        self.unlock_box.toggled.connect(self._on_unlock_toggled)
        self.unlock_box.setVisible(self._andor)
        live_layout.addWidget(self.unlock_box)
        live_layout.addStretch(1)

        self.run_note = QLabel("")
        self.run_note.setWordWrap(True)
        self.run_table = QTableWidget(0, len(RUN_COLUMNS))
        self.run_table.setHorizontalHeaderLabels(list(RUN_COLUMNS))
        self.run_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.run_table.verticalHeader().setVisible(False)
        self.run_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        run_page = QWidget()
        run_layout = QVBoxLayout(run_page)
        run_layout.addWidget(self.run_note)
        run_layout.addWidget(self.run_table, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(live_page, "Live")
        self.tabs.addTab(self.advanced_form, "Advanced")
        self.run_tab_index = self.tabs.addTab(run_page, "Run")

        # -- chip, messages, footer -------------------------------------------------
        self.reload_chip = QToolButton()
        self.reload_chip.setAutoRaise(False)
        self.reload_chip.setStyleSheet("QToolButton { background: #5a4300; color: #fff3c8; "
                                       "border-radius: 8px; padding: 1px 8px; }")
        self.reload_chip.clicked.connect(self.reload_stale)
        self.reload_chip.hide()
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.hide()
        self.footer = QLabel(FOOTER_TEXT)
        self.footer.setWordWrap(True)
        self.footer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        root = QVBoxLayout(self)
        root.addLayout(header)
        root.addWidget(self.persist_reason)
        root.addWidget(self.tabs, 1)
        root.addWidget(self.reload_chip, 0, Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self.message)
        root.addWidget(self.footer)

        if not self.describe.get("ok", False):
            self.show_message(f"{self.key}: {self.describe.get('error', 'no settings from the host')}")
            for form in self._forms:
                form.set_editable(False, "no settings from liveOD's camera host")

        self._last_settings = dict(values)
        try:
            snapshot = host.snapshot()
        except Exception as exc:
            snapshot = {}
            self.show_message(f"{self.key}: no snapshot from the camera host ({exc})")
        self.set_snapshot(snapshot, initial=True)
        if self._locked:
            self.tabs.setCurrentIndex(self.run_tab_index)
        if snapshot_signal is not None:
            snapshot_signal.connect(self.set_snapshot)

    # ------------------------------------------------------------------
    # Host state in
    # ------------------------------------------------------------------

    def _describe(self) -> dict:
        try:
            d = dict(self.host.describe(self.key) or {})
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        d.setdefault("ok", True)
        return d

    def set_snapshot(self, snapshot, initial: bool = False) -> None:
        """Slot for the host's snapshots: lock, Persist, the Run tab, and settings
        another program changed."""
        entry = camera_entries(snapshot).get(self.key)
        if entry is None:
            return
        self._entry = entry
        self._update_lock(entry)
        self._update_persist(entry)
        self._update_run_tab(entry)
        if not initial:
            self._take_settings(dict(entry.get("settings") or {}))

    def _run_words(self, entry) -> str:
        try:
            run_id = int(entry.get("run_id") or 0)
        except (TypeError, ValueError):
            run_id = 0
        return f"run {run_id}" if run_id else "an unsaved run"

    def _update_lock(self, entry) -> None:
        locked = (str(entry.get("host_state", "")) in RUN_PHASES or bool(entry.get("locked")))
        self._locked = locked
        reason = (f"{self._run_words(entry).capitalize()} holds {self.key}: its settings "
                  f"and Persist cannot change until the run ends (Abort frees it)."
                  if locked else "")
        changed = (locked, reason) != self._lock_applied
        self._lock_applied = (locked, reason)
        if not changed:
            return
        self.persist_reason.setText(reason)
        self.persist_reason.setStyleSheet("color: #64b5f6;")
        self.persist_reason.setVisible(locked)
        self.persist_box.setEnabled(not locked)
        self.unlock_box.setEnabled(not locked)
        if locked:
            for form in self._forms:
                form.set_editable(False, reason)
        elif self.describe.get("ok", False):
            for form in self._forms:
                form.set_editable(True)

    def _update_persist(self, entry) -> None:
        on = bool(entry.get("persist"))
        persisted = dict(entry.get("persisted") or {})
        self._persist_on = on
        self.persist_box.blockSignals(True)
        self.persist_box.setChecked(on)
        self.persist_box.blockSignals(False)
        if on:
            fields = ", ".join(f"{k}={self._fmt(k, v)}" for k, v in persisted.items())
            self.persist_status.setText(f"ON since {entry.get('persist_since') or '?'}: "
                                        f"{fields or 'no fields'}")
            self.persist_status.setStyleSheet(f"background: {PERSIST_FILL}; color: white; "
                                              "font-weight: bold; padding: 1px 6px; "
                                              "border-radius: 4px;")
            self.persist_box.setStyleSheet(f"QCheckBox {{ color: {PERSIST_FILL}; "
                                           "font-weight: bold; }")
            self.footer.setText(
                FOOTER_TEXT + f"\nPersist is ON: {', '.join(persisted) or 'no field'} "
                f"{'is' if len(persisted) == 1 else 'are'} applied on top of every run's "
                f"camera_params on {self.key} and recorded in the run file (camera_overrides).")
            self.footer.setStyleSheet(f"color: {PERSIST_FILL};")
        else:
            self.persist_status.setText("off: runs use the experiment's camera_params")
            self.persist_status.setStyleSheet("")
            self.persist_box.setStyleSheet("")
            self.footer.setText(FOOTER_TEXT)
            self.footer.setStyleSheet("")
        self._mark_persisted(set(persisted) if on else set())

    def _mark_persisted(self, keys: set) -> None:
        for form in self._forms:
            for layout in form.findChildren(QFormLayout):
                for key in form.keys():
                    label = layout.labelForField(form.field(key))
                    if label is None:
                        continue
                    text, style, tip = self._label_look.setdefault(
                        key, (label.text(), label.styleSheet(), label.toolTip()))
                    if key in keys:
                        label.setText(PERSIST_MARK + text)
                        label.setStyleSheet(f"color: {PERSIST_FILL}; font-weight: bold;")
                        label.setToolTip("Persisted: this value replaces camera_params in "
                                         "every run" + (f"\n{tip}" if tip else ""))
                    elif label.text() != text:
                        label.setText(text)
                        label.setStyleSheet(style)
                        label.setToolTip(tip)

    def marked_fields(self) -> list:
        """The fields whose label carries the Persist mark."""
        out = []
        for form in self._forms:
            for layout in form.findChildren(QFormLayout):
                for key in form.keys():
                    label = layout.labelForField(form.field(key))
                    if label is not None and label.text().startswith(PERSIST_MARK):
                        out.append(key)
        return out

    def _update_run_tab(self, entry) -> None:
        request = dict(entry.get("run_request") or {})
        persisted = dict(entry.get("persisted") or {}) if entry.get("persist") else {}
        applied = dict(entry.get("settings") or {})
        sources = dict(entry.get("readback_sources") or {})
        rows = [s for s in self.category.settings if s.run in ("param", "fixed")]
        self.run_table.setRowCount(len(rows))
        for i, s in enumerate(rows):
            if s.run == "fixed":
                req = f"fixed: {format_value(s, s.run_value)}"
            else:
                req = self._fmt(s.key, request[s.key]) if s.key in request else "—"
            if s.key in persisted:
                per = self._fmt(s.key, persisted[s.key])
            else:
                per = "—" if s.key in self.persist_keys else "never"
            app = self._fmt(s.key, applied[s.key]) if s.key in applied else "—"
            src = sources.get(s.key) or READBACK_WORDS.get(s.readback, s.readback)
            for j, text in enumerate((s.label, req, per, app, src)):
                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                if j == 2 and s.key in persisted:
                    item.setForeground(Qt.GlobalColor.red)
                self.run_table.setItem(i, j, item)
        run = self._run_words(entry)
        if self._locked:
            head = f"{run.capitalize()} holds {self.key}: read only until it ends."
        else:
            head = f"{self.key} is not in a run."
        note = ("Request: the run's camera_params (camera_params/ in its file). Persisted: "
                "carried into runs while Persist is on. Applied: the camera now -- after a "
                "run it keeps the run's settings until live settings change.")
        if not request:
            note += " (No run request from the camera host yet.)"
        self.run_note.setText(head + "\n" + note)

    def _fmt(self, key, value) -> str:
        if not self.category.has(key):
            return repr(value)
        dyn = dict(self.describe.get("dynamic") or {}).get(key) or {}
        labels = {}
        if dyn.get("choices") is not None and isinstance(dyn.get("labels"), (list, tuple)):
            labels = dict(zip(dyn["choices"], dyn["labels"]))
        return format_value(self.category.setting(key), value, labels)

    # ------------------------------------------------------------------
    # Settings another program changed
    # ------------------------------------------------------------------

    def _form_for(self, key) -> Optional[SettingsForm]:
        for form in self._forms:
            if key in form.keys():
                return form
        return None

    def _editing(self, key) -> bool:
        form = self._form_for(key)
        return form is not None and (form.field(key).hasFocus() or key in self._pending)

    def _take_settings(self, settings: dict) -> None:
        changed = {k: v for k, v in settings.items()
                   if self._form_for(k) is not None
                   and not _same(v, self._last_settings.get(k, object()))}
        self._last_settings.update(settings)
        for key, value in changed.items():
            form = self._form_for(key)
            if key in self._pending and _same(self._pending[key], value):
                continue                    # our own write, before its answer came back
            if self._editing(key):
                self._stale[key] = value
            elif not _same(form.values().get(key), value):
                form.set_values({key: value})
                self._stale.pop(key, None)
        self._show_chip()

    def _show_chip(self) -> None:
        if not self._stale:
            self.reload_chip.hide()
            return
        who = self._entry.get("changed_by") or "another program"
        fields = ", ".join(f"{k} → {self._fmt(k, v)}" for k, v in self._stale.items())
        self.reload_chip.setText(f"changed by {who}: {fields} — reload")
        self.reload_chip.setToolTip("Another program changed these while you were editing "
                                    "them; your edit was kept. Click to show its values.")
        self.reload_chip.show()

    def stale_fields(self) -> dict:
        return dict(self._stale)

    def reload_stale(self) -> None:
        """Show the values another program set (the chip)."""
        for key, value in list(self._stale.items()):
            form = self._form_for(key)
            if form is not None:
                form.field(key).clearFocus()
                form.set_values({key: value})
        self._stale.clear()
        self._show_chip()

    # ------------------------------------------------------------------
    # Live settings out
    # ------------------------------------------------------------------

    def _confirm_form(self, text: str) -> bool:
        """SettingsForm's "confirm" rule (EM gain above the live cap): only with unlock."""
        if self.unlock_box.isChecked():
            return True
        ok = self._confirm(f"Unlock EM gain above {ANDOR_LIVE_EM_GAIN_CAP}?",
                           f"{text}\n\nThis ticks 'Unlock EM gain above "
                           f"{ANDOR_LIVE_EM_GAIN_CAP}' for {self.key}. EM gain on bright light "
                           f"ages the gain register.")
        if ok:
            self.unlock_box.blockSignals(True)
            self.unlock_box.setChecked(True)
            self.unlock_box.blockSignals(False)
        return ok

    def _on_unlock_toggled(self, on: bool) -> None:
        if on:
            return
        gain = self.live_form.values().get("gain")
        try:
            over = gain is not None and float(gain) > ANDOR_LIVE_EM_GAIN_CAP
        except (TypeError, ValueError):
            over = False
        if over:
            self.unlock_box.blockSignals(True)
            self.unlock_box.setChecked(True)
            self.unlock_box.blockSignals(False)
            self.show_message(f"EM gain is {gain}, above {ANDOR_LIVE_EM_GAIN_CAP}: lower it "
                              f"first, then untick the unlock.", "warn")

    def _commit(self, values: dict) -> None:
        """A field's edit finished (after the form's own checks): to the host."""
        if self._locked:
            form_keys = [k for k in values]
            for key in form_keys:
                form = self._form_for(key)
                if form is not None:
                    form.revert([key])
            self.show_message(self.persist_reason.text() or f"{self.key} is held by a run.")
            return
        values = dict(values)
        for key in values:
            self._stale.pop(key, None)
        self._pending.update(values)
        self._show_chip()
        send = dict(values)
        if self._andor:
            send["em_gain_unlocked"] = bool(self.unlock_box.isChecked())
        if not self._async:
            self._relay.done.emit(values, self._call_set_live(send))
            return
        # One call at a time, in the order the edits were made (two quick edits of a
        # field must not answer out of order); the thread ends when there is nothing left.
        with self._jobs_lock:
            self._jobs.append((values, send))
            if self._worker is None:
                self._worker = threading.Thread(target=self._drain, daemon=True,
                                                name=f"liveOD-settings-{self.key}")
                self._worker.start()

    def _drain(self) -> None:
        while True:
            with self._jobs_lock:
                if not self._jobs:
                    self._worker = None
                    return
                values, send = self._jobs.popleft()
            result = self._call_set_live(send)
            try:
                self._relay.done.emit(values, result)
            except RuntimeError:
                pass            # the dialog is gone

    def _call_set_live(self, send: dict):
        try:
            return True, self.host.set_live(self.key, send)
        except Exception as exc:
            return False, exc

    def _on_set_live_done(self, values: dict, outcome) -> None:
        ok, result = outcome
        for key in values:
            self._pending.pop(key, None)
        if not ok:
            for key in values:
                form = self._form_for(key)
                if form is not None:
                    form.revert([key])
            shown = ", ".join(f"{k} = {self._fmt(k, v)}" for k, v in values.items())
            self.show_message(f"{shown} refused: {result}")
            for form in self._forms:
                if any(k in form.keys() for k in values):
                    form.show_message(f"{shown} refused: {result}")
            return
        readback = {k: _plain(v) for k, v in dict(result or {}).items() if k not in CONTROL_KEYS}
        for form in self._forms:
            form.set_values({k: v for k, v in readback.items() if k in form.keys()})
        self._last_settings.update(readback)
        for key, value in readback.items():         # the camera's answer, not someone else's
            if key in self._stale and _same(self._stale[key], value):
                del self._stale[key]
        clamped = [f"{k} {self._fmt(k, values[k])} -> {self._fmt(k, readback[k])}"
                   for k in values if k in readback and not _same(values[k], readback[k])]
        if clamped:
            self.show_message("The camera applied: " + "; ".join(clamped), "warn")
        else:
            self.show_message("")
        self._show_chip()

    def show_message(self, text: str, level: str = "error") -> None:
        colours = {"error": "#ff7070", "warn": "#e0c050", "info": "#a0a6cc"}
        self.message.setText(text or "")
        self.message.setStyleSheet(f"color: {colours.get(level, colours['info'])};")
        self.message.setVisible(bool(text))

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def persist_confirm_text(self) -> str:
        """What turning Persist on asks: exactly the fields that will be carried."""
        fresh = self._describe()                 # the live profile now, as the host will take it
        profile = dict(fresh.get("live_profile") or {}) if fresh.get("ok") else {}
        profile.update({k: v for k, v in self._last_settings.items() if k not in profile})
        width = max([len(self.category.setting(k).label) for k in self.persist_keys
                     if self.category.has(k)] + [4])
        lines = []
        for k in self.persist_keys:
            label = self.category.setting(k).label if self.category.has(k) else k
            value = self._fmt(k, profile[k]) if k in profile else "(the live value)"
            lines.append(f"    {label:<{width}}  {value}")
        return (f"Persist {self.key}'s settings into runs?\n\n"
                f"Every run on {self.key} will use these live values instead of the "
                f"experiment's camera_params, and each run file records every difference "
                f"(camera_overrides):\n\n" + "\n".join(lines) + "\n\n"
                f"Nothing else is carried into runs: not the exposure, and none of the "
                f"settings a run owns. Persist stays on until you turn it off here or liveOD "
                f"restarts.")

    def _on_persist_toggled(self, on: bool) -> None:
        if on:
            if not self.persist_keys:
                self._set_box(False)
                self.show_message(f"{self.key} has no setting that may persist into runs.")
                return
            if not self._confirm("Persist settings into runs?", self.persist_confirm_text()):
                self._set_box(False)
                return
        try:
            state = self.host.set_persist(self.key, bool(on))
        except Exception as exc:
            self._set_box(not on)
            self.show_message(f"Persist {'on' if on else 'off'} refused: {exc}")
            return
        self.show_message("")
        entry = dict(self._entry)
        entry["persist"] = bool(getattr(state, "on", on))
        entry["persisted"] = dict(getattr(state, "values", {}) or {})
        entry["persist_since"] = getattr(state, "since_iso", None) or entry.get("persist_since")
        self._entry = entry
        self._update_persist(entry)
        self._update_run_tab(entry)
        self.persist_changed.emit(self.key, bool(entry["persist"]))

    def _set_box(self, on: bool) -> None:
        self.persist_box.blockSignals(True)
        self.persist_box.setChecked(on)
        self.persist_box.blockSignals(False)

    # ------------------------------------------------------------------

    def is_locked(self) -> bool:
        return self._locked

    def wait_idle(self, timeout_s: float = 5.0) -> bool:
        """Wait for the host calls in flight (tests, shutdown); True if none is left."""
        deadline = time.monotonic() + float(timeout_s)
        while True:
            with self._jobs_lock:
                worker = self._worker
            if worker is None:
                return True
            worker.join(max(0.0, deadline - time.monotonic()))
            if worker.is_alive():
                return False


__all__ = ["CameraSettingsDialog", "FOOTER_TEXT", "NEVER_SHOWN", "persist_fields",
           "shown_schema", "RUN_COLUMNS"]
