"""The camera in the liveOD status row:

    [Running] 80545 · hf_tweezer_bec  [(o) andor  3 |▾][⚙][🎥]  [====> 37/120] ...

``CameraControl`` replaced ``CameraMenuButton`` (kept only for old imports) in
every liveOD window, so a camera button looks the same with and without the
camera host.  Without it (and in the remote viewer) it is made with
``glyphs=False``: no ⚙ and no 🎥, since the settings dialog and the live view
need the host.  One camera is on the main button -- the run's, or before any run
the first connected one -- coloured by its state; the arrow drops down a row
(button, ⚙, 🎥) for each of the others.  The ⚙ opens the camera's settings
dialog, the 🎥 its live view.  Both glyphs are painted (the characters would
render as colour emoji).

What the main button does depends on the camera (``decide``):

    closed                      Connect
    held by another program     Take from beacon…   (asks first)
    Andor, open                 Close SDK…          (asks first)
    Basler, open                Give back to beacon
    error / faulted / absent    Retry
    a run holds it              nothing: "Run N uses <cam> — Abort to free it"

**Persist** (a camera's settings carried into runs, PLAN C6) is bright red
``#ff1744`` with a white diagonal hatch -- a shape, not only a colour -- and a
10 px LED in the state colour.  An error is solid ``#c62828`` with a "!" LED, so
the two are never confused.  The ▾ turns red when a camera that is *not* shown
has Persist on (its tooltip names them, and their rows in the drop-down are
hatched).  A badge counts the camera's subscribers ("99+").  With no snapshot for
``stale_after_s`` (5 s) every camera is shown as unknown: grey with a red border.

Only displays and asks: ``action_requested(key, action)`` (connect / close_sdk /
give_back / take / retry; ``HOST_REQUEST`` maps them onto ``CameraHost.request``),
``settings_requested(key)``, ``live_view_requested(key, on)``, and -- for a
camera known only by its legacy state word (``set_state``/``set_states``) --
``toggle_requested(key)``, as ``CameraMenuButton`` did.  The states come in
through ``set_snapshot`` (a ``CameraHost.snapshot()``, or ``{key: entry}``), or
as legacy words through ``set_state``/``set_states``, with Persist for those
through ``set_persist`` (CAMERA_STATE's additive ``persist``).
No host, camera or config imports.

The total width is fixed, computed from the camera names only, so nothing a run
does to a camera can change the width of the status row (the window must never
change width when a run starts).
"""

import time
from dataclasses import dataclass
from typing import Mapping, Optional

from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPalette, QPen,
                         QPolygonF, QTransform)
from PyQt6.QtWidgets import (QAbstractButton, QHBoxLayout, QMenu, QMessageBox, QSizePolicy,
                             QWidget, QWidgetAction)

from waxx.util.live_od.gui.camera_menu import CONNECTED_STATES, STATES

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

#: host_state -> the legacy word POLL / CAMERA_STATE use (T14). A copy of the
#: camera host's LEGACY_STATE, so that this module needs no host import (the
#: remote viewer imports the gui package without the host); a test keeps them equal.
LEGACY_STATE = {
    "absent": "closed", "closed": "closed", "reserved": "closed", "held_elsewhere": "closed",
    "virtual": "closed",
    "claiming": "loading", "opening": "loading", "returning": "loading",
    "idle": "open", "streaming": "open",
    "run_locked": "grabbing", "arming": "grabbing", "acquiring": "grabbing", "draining": "grabbing",
    "error": "failed", "faulted": "failed", "run_fault": "failed", "hung": "failed",
}
RUN_PHASES = ("run_locked", "arming", "acquiring", "draining")

# host_state -> (colour, what the tooltip calls it, kind). The colours of the
# legacy words are kept (closed grey, loading purple, open green, run blue, failed red).
HOST_STATES = {
    "absent":         ("#616161", "not found (unplugged or off?)", "absent"),
    "virtual":        ("#607d8b", "not a camera (nothing to connect)", "virtual"),
    "held_elsewhere": ("#8d6e63", "held by another program", "held"),
    "reserved":       ("#8d6e63", "lent to another program", "reserved"),
    "closed":         (STATES["closed"][0], "not connected", "closed"),
    "claiming":       (STATES["loading"][0], "taking it from the beacon camera server…", "busy"),
    "opening":        (STATES["loading"][0], "connecting…", "busy"),
    "returning":      (STATES["loading"][0], "giving it back to the beacon camera server…", "busy"),
    "idle":           (STATES["open"][0], "connected, idle", "open"),
    "streaming":      ("#00897b", "connected, streaming live", "open"),
    "run_locked":     (STATES["grabbing"][0], "locked by a run", "run"),
    "arming":         (STATES["grabbing"][0], "arming for a run", "run"),
    "acquiring":      (STATES["grabbing"][0], "acquiring for a run", "run"),
    "draining":       (STATES["grabbing"][0], "finishing a run", "run"),
    "error":          (STATES["failed"][0], "error", "error"),
    "faulted":        (STATES["failed"][0], "faulted (gave up reopening)", "error"),
    "run_fault":      (STATES["failed"][0], "camera fault during a run", "run_fault"),
    "hung":           (STATES["failed"][0], "not responding", "hung"),
}
ERROR_KINDS = ("error", "run_fault", "hung")

PERSIST_FILL = "#ff1744"
ERROR_FILL = STATES["failed"][0]            # "#c62828"
UNKNOWN_FILL = "#757575"
UNKNOWN_BORDER = "#ff1744"
ARROW_FILL = "#546e7a"
LIVE_ON_FILL = "#1e88e5"

#: action -> what ``CameraHost.request`` is asked (open / close); the integration
#: uses it. "take" and "connect" both open: the host borrows a Basler from the beacon
#: server that has it before opening; "give_back" closes, and the host returns a
#: borrowed Basler as part of the close.
HOST_REQUEST = {"connect": "open", "take": "open", "retry": "open",
                "close_sdk": "close", "give_back": "close"}
ACTIONS = tuple(HOST_REQUEST)

STALE_AFTER_S = 5.0

# geometry (px)
HEIGHT = 22
LED = 10
PAD = 5
GAP = 5
ARROW_WIDTH = 16
GLYPH_WIDTH = 22
SPACING = 2
HATCH_STEP = 8
HATCH_WIDTH = 2
BADGE_MAX = "99+"


def camera_entries(snapshot) -> dict:
    """``{key: entry}`` from a ``CameraHost.snapshot()`` (``{"cameras": {...}, ...}``)
    or from a plain ``{key: entry}``."""
    if not isinstance(snapshot, Mapping):
        return {}
    cams = snapshot.get("cameras")
    if isinstance(cams, Mapping):
        return {str(k): dict(v) for k, v in cams.items() if isinstance(v, Mapping)}
    return {str(k): dict(v) for k, v in snapshot.items() if isinstance(v, Mapping)}


def legacy_word(entry: Mapping) -> str:
    """The legacy state word of an entry: its ``state``, else from ``host_state``."""
    word = entry.get("state")
    if word in STATES:
        return word
    return LEGACY_STATE.get(str(entry.get("host_state", "")), "closed")


def badge_text(n_subs) -> str:
    try:
        n = int(n_subs or 0)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    return BADGE_MAX if n > 99 else str(n)


def holder_text(holder) -> str:
    """"spot finder on kong" from a holder dict ("another program" if unknown)."""
    if not isinstance(holder, Mapping) or not holder:
        return "another program"
    label = str(holder.get("label") or holder.get("server_id") or holder.get("holder_id")
                or "another program")
    host = str(holder.get("host") or "")
    return f"{label} on {host}" if host else label


def _run_name(entry: Mapping) -> str:
    run_id = entry.get("run_id")
    try:
        run_id = int(run_id or 0)
    except (TypeError, ValueError):
        run_id = 0
    return f"Run {run_id}" if run_id else "An unsaved run"


def _is_andor(entry: Mapping) -> bool:
    return (str(entry.get("category", "")) == "andor_emccd"
            or str(entry.get("camera_type", "")) == "andor")


@dataclass(frozen=True)
class Decision:
    """What one camera's buttons do and show (``decide``)."""
    kind: str                   # legacy | unknown | a HOST_STATES kind
    action: str                 # "" | one of ACTIONS | "toggle" (legacy)
    main_enabled: bool
    confirm: str                # "" or the question asked before ``action``
    main_tip: str               # what clicking does, or why it can't
    cog_enabled: bool
    cog_run_tab: bool           # the dialog opens on its read-only Run tab
    live_enabled: bool          # opening a live view is allowed


def decide(key: str, entry: Optional[Mapping], *, unknown: bool = False,
           disabled: bool = False) -> Decision:
    """The enable/emit table for one camera.  ``entry`` without ``host_state`` is a
    legacy camera (a state word only): the main button toggles, nothing else works."""
    entry = dict(entry or {})
    if "host_state" not in entry:
        word = legacy_word(entry)
        verb = "disconnect" if word in CONNECTED_STATES else "connect"
        return Decision("legacy", "toggle", not disabled, "", f"Click to {verb}.",
                        False, False, False)
    if unknown:
        return Decision("unknown", "", False, "",
                        "No word from liveOD's camera host: its state is unknown.",
                        True, False, False)
    hs = str(entry.get("host_state"))
    kind = HOST_STATES.get(hs, HOST_STATES["error"])[2]
    run = _run_name(entry)
    if kind == "closed":
        return Decision(kind, "connect", True, "", "Click to connect.", True, False, True)
    if kind == "held":
        who = holder_text(entry.get("holder"))
        question = (f"{key} is held by {who}.\n\nTake it? The beacon camera server closes it "
                    f"and lends it to liveOD; a viewer watching it there loses it until "
                    f"liveOD gives it back.")
        return Decision(kind, "take", True, question, f"Held by {who}. Click to take it "
                        f"from beacon… (asks first)", True, False, False)
    if kind == "reserved":
        return Decision(kind, "", False, "", f"Lent to {holder_text(entry.get('holder'))}: "
                        f"it comes back when they return it.", True, False, False)
    if kind == "busy":
        return Decision(kind, "", False, "", "Busy: wait for it to finish.", True, False, False)
    if kind == "open":
        if _is_andor(entry):
            question = (f"Close the Andor SDK for {key}?\n\nliveOD stops acquisition, closes "
                        f"the shutter and releases the SDK, so another program can open the "
                        f"camera. liveOD takes no frames from {key} until it is connected again.")
            return Decision(kind, "close_sdk", True, question,
                            "Click to close the Andor SDK… (asks first)", True, False, True)
        return Decision(kind, "give_back", True, "",
                        "Click to give it back to the beacon camera server (its viewers "
                        "can open it again).", True, False, True)
    if kind == "run":
        return Decision(kind, "", False, "", f"{run} uses {key} — Abort to free it.",
                        True, True, False)
    if kind == "run_fault":
        return Decision(kind, "", False, "", f"{run} hit a camera fault on {key} — Abort "
                        f"to free it.", True, True, False)
    if kind == "hung":
        return Decision(kind, "", False, "", f"{key}'s camera thread is not responding; "
                        f"restart liveOD to recover it.", True, False, False)
    if kind in ("error", "absent"):
        return Decision(kind, "retry", True, "", "Click to try connecting again.",
                        True, False, False)
    # virtual: an entry with no camera behind it (the APD)
    return Decision(kind, "", False, "", f"{key} is not a camera: there is nothing to "
                    f"connect.", False, False, False)


# ---------------------------------------------------------------------------
# Painting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Look:
    """How one camera button is painted."""
    text: str
    fill: str
    hatch: bool = False
    led: str = ""                # LED colour ("" = none)
    led_mark: str = ""           # "" | "!" | "?"
    border: str = ""             # "" = none
    badge: str = ""


def look_for(key: str, entry: Optional[Mapping], *, unknown: bool = False) -> Look:
    entry = dict(entry or {})
    badge = badge_text(entry.get("n_subs"))
    if "host_state" not in entry:
        word = legacy_word(entry)
        color = STATES.get(word, STATES["closed"])[0]
        mark = "!" if word == "failed" else ""
        if entry.get("persist"):
            return Look(key, PERSIST_FILL, hatch=True, led=color, led_mark=mark, badge=badge)
        return Look(key, color, led=color, led_mark=mark, badge=badge)
    if unknown:
        return Look(key, UNKNOWN_FILL, led=UNKNOWN_FILL, led_mark="?", border=UNKNOWN_BORDER)
    color, _words, kind = HOST_STATES.get(str(entry.get("host_state")), HOST_STATES["error"])
    mark = "!" if kind in ERROR_KINDS else ""
    if entry.get("persist"):
        return Look(key, PERSIST_FILL, hatch=True, led=color, led_mark=mark, badge=badge)
    return Look(key, ERROR_FILL if mark else color, led=color, led_mark=mark, badge=badge)


def _bold(font: QFont) -> QFont:
    f = QFont(font)
    f.setBold(True)
    return f


def _badge_font(font: QFont) -> QFont:
    f = _bold(font)
    f.setPixelSize(9)
    return f


def badge_width(font: QFont) -> int:
    return QFontMetrics(_badge_font(font)).horizontalAdvance(BADGE_MAX) + 8


def main_width(keys, font: QFont) -> int:
    """The main button's width for these camera names (never for a state)."""
    metrics = QFontMetrics(_bold(font))
    longest = max([metrics.horizontalAdvance(k) for k in keys] + [metrics.horizontalAdvance("no camera")])
    return PAD + LED + GAP + longest + GAP + badge_width(font) + PAD


def total_width(keys, font: QFont, glyphs: bool = True) -> int:
    width = main_width(keys, font) + ARROW_WIDTH
    return width + 2 * GLYPH_WIDTH + 3 * SPACING if glyphs else width


def _rounded(rect: QRectF, radius: float, left: bool = True, right: bool = True) -> QPainterPath:
    """A rect with rounded corners on the chosen sides."""
    path = QPainterPath()
    x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
    rl = radius if left else 0.0
    rr = radius if right else 0.0
    path.moveTo(x0 + rl, y0)
    path.lineTo(x1 - rr, y0)
    if rr:
        path.arcTo(QRectF(x1 - 2 * rr, y0, 2 * rr, 2 * rr), 90, -90)
    path.lineTo(x1, y1 - rr)
    if rr:
        path.arcTo(QRectF(x1 - 2 * rr, y1 - 2 * rr, 2 * rr, 2 * rr), 0, -90)
    path.lineTo(x0 + rl, y1)
    if rl:
        path.arcTo(QRectF(x0, y1 - 2 * rl, 2 * rl, 2 * rl), 270, -90)
    path.lineTo(x0, y0 + rl)
    if rl:
        path.arcTo(QRectF(x0, y0, 2 * rl, 2 * rl), 180, -90)
    path.closeSubpath()
    return path


def paint_hatch(painter: QPainter, path: QPainterPath, rect: QRectF) -> None:
    """White diagonal lines over ``path`` (crisp, so they read as a pattern)."""
    painter.save()
    painter.setClipPath(path)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.setPen(QPen(QColor("#ffffff"), HATCH_WIDTH))
    h = rect.height()
    x = rect.left() - h
    while x < rect.right() + h:
        painter.drawLine(QPointF(x, rect.bottom()), QPointF(x + h, rect.top()))
        x += HATCH_STEP
    painter.restore()


def paint_led(painter: QPainter, center: QPointF, color: str, mark: str) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    r = LED / 2.0
    if mark == "!":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(center, r, r)
        painter.setPen(QPen(QColor(ERROR_FILL), 2))
        painter.drawLine(QPointF(center.x(), center.y() - r + 2.5), QPointF(center.x(), center.y() + 0.8))
        painter.drawPoint(QPointF(center.x(), center.y() + r - 2.2))
    else:
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(QColor(color))
        painter.drawEllipse(center, r - 0.75, r - 0.75)
        if mark == "?":
            f = QFont(painter.font())
            f.setBold(True)
            f.setPixelSize(8)
            painter.setFont(f)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(center.x() - r, center.y() - r, 2 * r, 2 * r),
                             int(Qt.AlignmentFlag.AlignCenter), "?")
    painter.restore()


def paint_outlined_text(painter: QPainter, rect: QRectF, text: str, font: QFont) -> None:
    """White bold text with a dark outline: legible on any fill, hatch included."""
    metrics = QFontMetrics(font)
    shown = metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(rect.width()))
    w = metrics.horizontalAdvance(shown)
    x = rect.left() + (rect.width() - w) / 2.0
    y = rect.top() + (rect.height() + metrics.ascent() - metrics.descent()) / 2.0
    path = QPainterPath()
    path.addText(QPointF(x, y), font, shown)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor(0, 0, 0, 150), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                        Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(path)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#ffffff"))
    painter.drawPath(path)
    painter.restore()


def paint_camera_button(painter: QPainter, rect: QRectF, look: Look, font: QFont, *,
                        round_right: bool = False, enabled: bool = True) -> None:
    """The main button (and each drop-down row's): fill (+ hatch), LED, name, badge."""
    path = _rounded(rect.adjusted(0.5, 0.5, -0.5, -0.5), 8.0, left=True, right=round_right)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.fillPath(path, QColor(look.fill))
    if look.hatch:
        paint_hatch(painter, path, rect)
    if look.border:
        painter.setPen(QPen(QColor(look.border), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(_rounded(rect.adjusted(1, 1, -1, -1), 7.0, left=True, right=round_right))
    if look.led:
        paint_led(painter, QPointF(rect.left() + PAD + LED / 2.0, rect.center().y()),
                  look.led, look.led_mark)
    bw = badge_width(font)
    text_rect = QRectF(rect.left() + PAD + LED + GAP, rect.top(),
                       rect.width() - (PAD + LED + GAP) - (GAP + bw + PAD), rect.height())
    paint_outlined_text(painter, text_rect, look.text, _bold(font))
    if look.badge:
        bfont = _badge_font(font)
        bm = QFontMetrics(bfont)
        w = bm.horizontalAdvance(look.badge) + 8
        brect = QRectF(rect.right() - PAD - w, rect.center().y() - 7, w, 14)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 110))
        painter.drawRoundedRect(brect, 7, 7)
        painter.setFont(bfont)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(brect, int(Qt.AlignmentFlag.AlignCenter), look.badge)
    # A disabled button keeps its colours (Persist red stays exactly #ff1744): the
    # state already says why it cannot be clicked, and the tooltip says what frees it.
    painter.restore()


def paint_cog(painter: QPainter, rect: QRectF, color: QColor) -> None:
    c = rect.center()
    path = QPainterPath()
    path.addEllipse(c, 5.2, 5.2)
    for i in range(8):
        tooth = QPainterPath()
        tooth.addRect(QRectF(-1.6, -7.5, 3.2, 3.4))
        t = QTransform()
        t.translate(c.x(), c.y())
        t.rotate(45.0 * i)
        path = path.united(t.map(tooth))
    hole = QPainterPath()
    hole.addEllipse(c, 2.2, 2.2)
    path = path.subtracted(hole)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.fillPath(path, color)
    painter.restore()


def paint_movie_camera(painter: QPainter, rect: QRectF, color: QColor) -> None:
    c = rect.center()
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    body = QRectF(c.x() - 7.5, c.y() - 1.5, 10.0, 7.0)
    painter.drawRoundedRect(body, 1.5, 1.5)
    painter.drawEllipse(QPointF(c.x() - 5.0, c.y() - 4.2), 2.4, 2.4)       # the reels
    painter.drawEllipse(QPointF(c.x() - 0.2, c.y() - 4.2), 2.4, 2.4)
    lens = QPolygonF([QPointF(c.x() + 2.5, c.y() + 2.0), QPointF(c.x() + 7.5, c.y() - 1.0),
                      QPointF(c.x() + 7.5, c.y() + 5.0)])
    painter.drawPolygon(lens)
    painter.restore()


# ---------------------------------------------------------------------------
# Buttons
# ---------------------------------------------------------------------------

class _CameraButton(QAbstractButton):
    """The painted camera button (the main one, and one per drop-down row)."""

    def __init__(self, width: int, round_right: bool = False, parent=None):
        super().__init__(parent)
        self._look = Look("no camera", STATES["closed"][0])
        self._round_right = round_right
        self.setFixedSize(width, HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_look(self, look: Look) -> None:
        if look != self._look:
            self._look = look
            self.update()

    def look(self) -> Look:
        return self._look

    def sizeHint(self) -> QSize:
        return QSize(self.width(), HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.EnabledChange:
            self.setCursor(Qt.CursorShape.PointingHandCursor if self.isEnabled()
                           else Qt.CursorShape.ArrowCursor)
        super().changeEvent(event)

    def paintEvent(self, _event):
        painter = QPainter(self)
        paint_camera_button(painter, QRectF(self.rect()), self._look, self.font(),
                            round_right=self._round_right, enabled=self.isEnabled())
        painter.end()


class _ArrowButton(QAbstractButton):
    """The ▾ segment: neutral, or red when a camera not shown has Persist on."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._red = False
        self.setFixedSize(ARROW_WIDTH, HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_red(self, red: bool) -> None:
        if red != self._red:
            self._red = red
            self.update()

    def is_red(self) -> bool:
        return self._red

    def sizeHint(self) -> QSize:
        return QSize(ARROW_WIDTH, HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect())
        path = _rounded(rect.adjusted(0.5, 0.5, -0.5, -0.5), 8.0, left=False, right=True)
        painter.fillPath(path, QColor(PERSIST_FILL if self._red else ARROW_FILL))
        if self._red:
            paint_hatch(painter, path, rect)
        painter.setPen(QPen(QColor(255, 255, 255, 110), 1))
        painter.drawLine(QPointF(0.5, 3), QPointF(0.5, rect.height() - 3))
        c = rect.center()
        tri = QPolygonF([QPointF(c.x() - 4, c.y() - 2), QPointF(c.x() + 4, c.y() - 2),
                         QPointF(c.x(), c.y() + 3)])
        painter.setPen(QPen(QColor(0, 0, 0, 150), 1.5))
        painter.setBrush(QColor("#ffffff") if self.isEnabled() else QColor(255, 255, 255, 110))
        painter.drawPolygon(tri)
        painter.end()


class _GlyphButton(QAbstractButton):
    """⚙ (``"cog"``) or 🎥 (``"movie"``), painted in the palette's text colour; a
    checked 🎥 (its live view is open) is white on blue."""

    def __init__(self, glyph: str, parent=None):
        super().__init__(parent)
        self.glyph = glyph
        self.setFixedSize(GLYPH_WIDTH, HEIGHT)
        self.setCheckable(glyph == "movie")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        return QSize(GLYPH_WIDTH, HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        pal = self.palette()
        on = self.isChecked()
        bg = QColor(LIVE_ON_FILL) if on else pal.color(QPalette.ColorRole.Button)
        if self.underMouse() and self.isEnabled() and not on:
            bg = bg.lighter(125)
        painter.setPen(QPen(pal.color(QPalette.ColorRole.Mid), 1))
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, 6, 6)
        color = QColor("#ffffff") if on else pal.color(QPalette.ColorRole.ButtonText)
        if not self.isEnabled():
            color.setAlpha(90)
        if self.glyph == "cog":
            paint_cog(painter, rect, color)
        else:
            paint_movie_camera(painter, rect, color)
        painter.end()


class _CameraRow(QWidget):
    """One camera in the drop-down: its button, ⚙ and 🎥 (the button only
    without ``glyphs``)."""

    def __init__(self, key: str, width: int, parent=None, glyphs: bool = True):
        super().__init__(parent)
        self.key = key
        self.button = _CameraButton(width, round_right=True)
        self.cog = _GlyphButton("cog", self)
        self.live = _GlyphButton("movie", self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(SPACING + 1)
        layout.addWidget(self.button)
        if glyphs:
            layout.addWidget(self.cog)
            layout.addWidget(self.live)
        else:
            self.cog.hide()
            self.live.hide()


# ---------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------

def _ask(parent, title: str, text: str) -> bool:
    box = QMessageBox(QMessageBox.Icon.Warning, title, text,
                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, parent)
    box.setDefaultButton(QMessageBox.StandardButton.Cancel)
    return box.exec() == QMessageBox.StandardButton.Yes


class CameraControl(QWidget):
    """See the module docstring.

    ``CameraControl(camera_keys=(), parent=None, *, confirm=None, expect_snapshots=False,
    stale_after_s=5.0, clock=time.monotonic, glyphs=True)``.  ``confirm(title, text) -> bool``
    asks before "Close SDK…" and "Take from beacon…" (default: a Yes/Cancel box
    defaulting to Cancel).  ``expect_snapshots``: the owner will feed
    ``set_snapshot``; the unknown marking then starts ``stale_after_s`` after
    construction even if no snapshot ever comes.  ``glyphs=False``: no ⚙ and no
    🎥, on the main button or in the drop-down (no camera host to serve them).
    """

    action_requested = pyqtSignal(str, str)         # camera_key, action (ACTIONS)
    settings_requested = pyqtSignal(str)            # camera_key: open its ⚙ dialog
    live_view_requested = pyqtSignal(str, bool)     # camera_key, open (True) / close
    toggle_requested = pyqtSignal(str)              # camera_key (legacy cameras only)

    def __init__(self, camera_keys=(), parent=None, *, confirm=None, expect_snapshots=False,
                 stale_after_s: float = STALE_AFTER_S, clock=time.monotonic,
                 glyphs: bool = True):
        super().__init__(parent)
        self._entries: dict = {}            # camera_key -> snapshot entry, in the lab's order
        self._current = None                # the run's camera, once there has been one
        self._disabled = set()              # legacy: waiting for an answer
        self._persist_words = {}            # legacy: camera_key -> Persist on (set_persist)
        self._live_open = set()             # cameras whose live view is open
        self._rows: dict = {}               # camera_key -> (QWidgetAction, _CameraRow)
        self._confirm = confirm or (lambda title, text: _ask(self, title, text))
        self._clock = clock
        self._stale_after_s = float(stale_after_s)
        self._host_mode = bool(expect_snapshots)
        self._glyphs = bool(glyphs)
        self._last_snapshot_t = clock()
        self._unknown = False
        self._width_keys = None

        self.main_button = _CameraButton(10)
        self.arrow_button = _ArrowButton()
        self.cog_button = _GlyphButton("cog", self)
        self.live_button = _GlyphButton("movie", self)
        self.cog_button.setToolTip("Camera settings")
        self.live_button.setToolTip("Live view")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.main_button)
        layout.addWidget(self.arrow_button)
        if self._glyphs:
            layout.addSpacing(SPACING + 1)
            layout.addWidget(self.cog_button)
            layout.addSpacing(SPACING + 1)
            layout.addWidget(self.live_button)
        else:
            self.cog_button.hide()
            self.live_button.hide()
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._menu = QMenu(self)
        self.main_button.clicked.connect(lambda: self._on_main(self.shown_camera()))
        self.arrow_button.clicked.connect(self._show_menu)
        self.cog_button.clicked.connect(lambda: self._on_cog(self.shown_camera()))
        self.live_button.clicked.connect(lambda: self._on_live(self.shown_camera()))

        for key in camera_keys:
            self._add_camera(str(key))
        self._fit_width()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self._render()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def set_snapshot(self, snapshot) -> None:
        """Slot for ``HostQtBridge.snapshot_changed``: every camera's host state."""
        entries = camera_entries(snapshot)
        added = False
        for key, entry in entries.items():
            if key not in self._entries:
                self._add_camera(key)
                added = True
            self._entries[key] = entry
        self._host_mode = True
        self._last_snapshot_t = self._clock()
        self._unknown = False
        if added:
            self._fit_width()
        self._render()

    def set_state(self, camera_key: str, state: str) -> None:
        """A legacy camera's state word (``CameraButton.state_changed``).  Ignored
        for a camera the host's snapshots describe: those are the authority."""
        self.set_states({camera_key: state})

    def set_states(self, states: Mapping) -> None:
        """Several legacy state words at once (a remote viewer's CAMERA_STATE)."""
        added = False
        for key, state in dict(states).items():
            if key not in self._entries:
                self._add_camera(key)
                added = True
            elif "host_state" in self._entries[key]:
                continue
            self._entries[key] = {"key": key, "state": state if state in STATES else "closed",
                                  "persist": self._persist_words.get(key, False)}
        if added:
            self._fit_width()
        self._render()

    def set_persist(self, persist: Mapping) -> None:
        """Which legacy cameras have Persist on (CAMERA_STATE's additive
        ``persist``); cameras not named keep what they had.  Ignored for a camera
        the host's snapshots describe: their own ``persist`` is the authority."""
        for key, on in dict(persist or {}).items():
            key = str(key)
            self._persist_words[key] = bool(on)
            entry = self._entries.get(key)
            if entry is not None and "host_state" not in entry:
                entry["persist"] = bool(on)
        self._render()

    def set_current(self, camera_key: str) -> None:
        """The camera the run uses: the one on the main button."""
        if camera_key and camera_key not in self._entries:
            self._add_camera(camera_key)
            self._fit_width()
        self._current = camera_key or None
        self._render()

    def set_camera_enabled(self, camera_key: str, enabled: bool) -> None:
        """Grey one legacy camera out while a request for it is on its way."""
        if enabled:
            self._disabled.discard(camera_key)
        else:
            self._disabled.add(camera_key)
        self._render()

    def set_live_view_open(self, camera_key: str, open_: bool) -> None:
        """Whether ``camera_key``'s live view is open (its 🎥 shows checked)."""
        if open_:
            self._live_open.add(camera_key)
        else:
            self._live_open.discard(camera_key)
        self._render()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def camera_keys(self) -> list:
        return list(self._entries)

    def entry(self, camera_key: str) -> dict:
        return dict(self._entries.get(camera_key, {}))

    def state(self, camera_key: str) -> str:
        """The legacy word (closed / loading / open / grabbing / failed)."""
        return legacy_word(self._entries.get(camera_key, {}))

    def host_state(self, camera_key: str) -> str:
        return str(self._entries.get(camera_key, {}).get("host_state", ""))

    def is_unknown(self) -> bool:
        return self._unknown

    def persisted(self, camera_key: str) -> bool:
        return bool(self._entries.get(camera_key, {}).get("persist"))

    def decision(self, camera_key: str) -> Decision:
        return decide(camera_key, self._entries.get(camera_key),
                      unknown=self._unknown, disabled=camera_key in self._disabled)

    def hidden_persisted(self) -> list:
        """Cameras with Persist on that are not on the main button."""
        shown = self.shown_camera()
        return [k for k, e in self._entries.items() if k != shown and e.get("persist")]

    def shown_camera(self):
        """The camera on the main button: the run's; before any run, the first
        connected one, else the first there is; None with no cameras."""
        if self._current in self._entries:
            return self._current
        for key, entry in self._entries.items():
            if legacy_word(entry) in CONNECTED_STATES:
                return key
        return next(iter(self._entries), None)

    # ------------------------------------------------------------------
    # Clicks
    # ------------------------------------------------------------------

    def _on_main(self, key) -> None:
        if key is None:
            return
        d = self.decision(key)
        if not d.main_enabled or not d.action:
            return
        self._menu.close()
        if d.action == "toggle":
            self.toggle_requested.emit(key)
            return
        if d.confirm:
            title = {"close_sdk": "Close the Andor SDK?",
                     "take": "Take the camera from beacon?"}.get(d.action, "Camera")
            if not self._confirm(title, d.confirm):
                return
        self.action_requested.emit(key, d.action)

    def _on_cog(self, key) -> None:
        if key is not None and self.decision(key).cog_enabled:
            self._menu.close()
            self.settings_requested.emit(key)

    def _on_live(self, key) -> None:
        if key is None:
            return
        opening = key not in self._live_open
        if opening and not self.decision(key).live_enabled:
            self._render()          # undo the checkable button's own toggle
            return
        self._menu.close()
        self.live_view_requested.emit(key, opening)
        self._render()              # the owner confirms with set_live_view_open

    def _show_menu(self) -> None:
        if len(self._entries) > 1:
            self._menu.popup(self.mapToGlobal(self.main_button.geometry().bottomLeft()))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _add_camera(self, key: str) -> None:
        self._entries[key] = {"key": key, "state": "closed",
                              "persist": self._persist_words.get(key, False)}
        row = _CameraRow(key, 10, glyphs=self._glyphs)
        row.button.clicked.connect(lambda _=False, k=key: self._on_main(k))
        row.cog.clicked.connect(lambda _=False, k=key: self._on_cog(k))
        row.live.clicked.connect(lambda _=False, k=key: self._on_live(k))
        action = QWidgetAction(self._menu)
        action.setDefaultWidget(row)
        self._menu.addAction(action)
        self._rows[key] = (action, row)

    def _fit_width(self) -> None:
        """Fix every width from the camera names, once per set of cameras."""
        keys = tuple(self._entries)
        if keys == self._width_keys:
            return
        self._width_keys = keys
        width = main_width(keys, self.font())
        self.main_button.setFixedSize(width, HEIGHT)
        for _action, row in self._rows.values():
            row.button.setFixedSize(width + ARROW_WIDTH, HEIGHT)
        self.setFixedSize(total_width(keys, self.font(), self._glyphs), HEIGHT)

    def _tick(self) -> None:
        if not self._host_mode:
            return
        stale = self._clock() - self._last_snapshot_t > self._stale_after_s
        if stale != self._unknown:
            self._unknown = stale
            self._render()

    def _tip(self, key: str, d: Decision) -> str:
        entry = self._entries.get(key, {})
        if d.kind == "legacy":
            word = legacy_word(entry)
            tip = f"{key}: {STATES.get(word, STATES['closed'])[1]}. {d.main_tip}"
            if entry.get("persist"):
                tip += (f"\nPERSIST ON: liveOD applies {key}'s persisted settings on top of "
                        f"every run's camera_params (recorded in the run file).")
            return tip
        if d.kind == "unknown":
            age = self._clock() - self._last_snapshot_t
            return f"{key}: state unknown — no snapshot from liveOD's camera host for {age:.0f} s."
        words = HOST_STATES.get(str(entry.get("host_state")), ("", "unknown state", ""))[1]
        lines = [f"{key}: {words}.", d.main_tip]
        if entry.get("error"):
            lines.append(f"Error: {entry['error']}")
        n = badge_text(entry.get("n_subs"))
        if n:
            lines.append(f"{n} subscriber(s) watching it.")
        if entry.get("persist"):
            fields = ", ".join(f"{k}={v!r}" for k, v in dict(entry.get("persisted") or {}).items())
            since = entry.get("persist_since") or "?"
            lines.append(f"PERSIST ON since {since}: {fields or 'no fields'} replace the run's "
                         f"camera_params in every run on {key} (recorded in the run file).")
        return "\n".join(lines)

    def _render(self) -> None:
        shown = self.shown_camera()
        for key, (action, row) in self._rows.items():
            action.setVisible(key != shown)
            d = self.decision(key)
            row.button.set_look(look_for(key, self._entries.get(key), unknown=self._unknown))
            row.button.setEnabled(d.main_enabled)
            row.button.setToolTip(self._tip(key, d))
            row.cog.setEnabled(d.cog_enabled)
            live_on = key in self._live_open
            row.live.setEnabled(d.live_enabled or live_on)
            row.live.setChecked(live_on)
            row.cog.setToolTip(f"{key}: settings")
            row.live.setToolTip(f"{key}: {'close the' if live_on else 'open a'} live view")
        if shown is None:
            self.main_button.set_look(Look("no camera", STATES["closed"][0]))
            self.main_button.setEnabled(False)
            self.main_button.setToolTip("No cameras yet")
            for b in (self.arrow_button, self.cog_button, self.live_button):
                b.setEnabled(False)
            self.arrow_button.set_red(False)
            return
        d = self.decision(shown)
        self.main_button.set_look(look_for(shown, self._entries.get(shown), unknown=self._unknown))
        self.main_button.setEnabled(d.main_enabled)
        self.main_button.setToolTip(self._tip(shown, d))
        self.cog_button.setEnabled(d.cog_enabled)
        self.cog_button.setToolTip(f"{shown}: settings" + (
            " (read only while the run holds it)" if d.cog_run_tab else ""))
        live_on = shown in self._live_open
        self.live_button.setEnabled(d.live_enabled or live_on)
        self.live_button.setChecked(live_on)
        self.live_button.setToolTip(f"{shown}: {'close the' if live_on else 'open a'} live view"
                                    + ("" if d.live_enabled or live_on else " (not now)"))
        others = len(self._entries) > 1
        hidden = self.hidden_persisted()
        self.arrow_button.setEnabled(others)
        self.arrow_button.set_red(bool(hidden))
        tip = "The other cameras." if others else "No other cameras."
        if hidden:
            tip = (f"Persist is ON for a camera not shown: {', '.join(hidden)}.\n"
                   f"Its settings replace camera_params in every run on it.\n" + tip)
        self.arrow_button.setToolTip(tip)


__all__ = ["CameraControl", "Decision", "Look", "decide", "look_for", "camera_entries",
           "legacy_word", "badge_text", "holder_text", "HOST_STATES", "LEGACY_STATE",
           "RUN_PHASES", "HOST_REQUEST", "ACTIONS", "PERSIST_FILL", "ERROR_FILL",
           "UNKNOWN_FILL", "UNKNOWN_BORDER", "STALE_AFTER_S", "total_width", "main_width"]
