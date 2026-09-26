"""The code dock: source tabs (sequence Python, generated QUA, config) with
a line-number gutter whose chips carry the colour of the lane each line
drives, Python syntax highlighting, and highlight-both-ways line linking.
"""

import re

from PyQt6.QtCore import Qt, QRect, QSize, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QFontDatabase, QPainter,
                         QSyntaxHighlighter, QTextCharFormat, QTextCursor,
                         QTextFormat, QTextDocument)
from PyQt6.QtWidgets import (QPlainTextEdit, QTabWidget, QWidget, QVBoxLayout,
                             QTableWidget, QTableWidgetItem, QLabel,
                             QTextEdit, QHeaderView, QAbstractItemView)

_KEYWORDS = ('and as assert async await break class continue def del elif '
             'else except finally for from global if import in is lambda '
             'None nonlocal not or pass raise return True False try while '
             'with yield').split()

CURRENT_COLOR = '#2f65ca'     # current-line tint when no lane colour is given


class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, doc):
        super().__init__(doc)
        def fmt(color, bold=False, italic=False):
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            if italic:
                f.setFontItalic(True)
            return f
        self.rules = [
            (re.compile(r'\b(' + '|'.join(_KEYWORDS) + r')\b'), fmt('#c586c0')),
            (re.compile(r'\bctx\.(\w+)'), fmt('#dcdcaa')),
            (re.compile(r'\bctx\.p\.(\w+)'), fmt('#9cdcfe')),
            (re.compile(r'\b(play|wait|align|measure|save|update_frequency|'
                        r'frame_rotation_2pi|reset_frame|reset_if_phase|'
                        r'ramp_to_zero|wait_for_trigger)\b(?=\()'),
             fmt('#dcdcaa')),
            (re.compile(r'@\w+'), fmt('#4ec9b0')),
            (re.compile(r'\b\d+(\.\d*)?([eE][-+]?\d+)?\b'), fmt('#b5cea8')),
            (re.compile(r"'[^'\n]*'|\"[^\"\n]*\""), fmt('#ce9178')),
            (re.compile(r'#.*$'), fmt('#6a9955', italic=True)),
        ]
        self.tri = fmt('#ce9178')

    def highlightBlock(self, text):
        for rx, f in self.rules:
            for m in rx.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), f)
        # triple-quoted strings across blocks
        self.setCurrentBlockState(0)
        start = 0
        if self.previousBlockState() != 1:
            start = text.find('"""')
            if start < 0:
                start = text.find("'''")
        while start >= 0:
            end = text.find('"""', start + 3)
            if end < 0:
                end = text.find("'''", start + 3)
            if end < 0:
                self.setCurrentBlockState(1)
                self.setFormat(start, len(text) - start, self.tri)
                break
            self.setFormat(start, end + 3 - start, self.tri)
            start = text.find('"""', end + 3)
            if start < 0:
                start = text.find("'''", end + 3)


class _Gutter(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.gutter_width(), 0)

    def paintEvent(self, ev):
        self.editor.paint_gutter(ev)


class SourceView(QPlainTextEdit):
    """Read-only source with a gutter, chips and highlights.

    Lines are 1-based *file* lines: `first_line` is the file line of the
    first text line (a notebook cell's function source starts elsewhere).
    """
    lineClicked = pyqtSignal(int)          # 1-based file line
    lineDoubleClicked = pyqtSignal(int)

    def __init__(self, text, first_line=1, language='python', parent=None):
        super().__init__(parent)
        self.first_line = int(first_line)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(9)
        self.setFont(font)
        self._gutter_bold = QFont(font)
        self._gutter_bold.setBold(True)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(' '))
        self.setPlainText(text)
        if language == 'python':
            self.hl = PythonHighlighter(self.document())
        self.chips = {}          # file line -> [colors]
        self.hot = set()         # file lines that emitted anything
        self.current = set()     # strong highlight
        self.related = set()     # soft highlight
        self.colors = {}         # current line -> accent colour
        self.gutter = _Gutter(self)
        self.blockCountChanged.connect(self._update_gutter_width)
        self.updateRequest.connect(self._update_gutter)
        self._update_gutter_width()
        self.setMouseTracking(True)

    # --- gutter ---
    def gutter_width(self):
        digits = len(str(self.first_line + self.blockCount()))
        return 14 + self.fontMetrics().horizontalAdvance('9') * digits + 10

    def _update_gutter_width(self, *a):
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _update_gutter(self, rect, dy):
        if dy:
            self.gutter.scroll(0, dy)
        else:
            self.gutter.update(0, rect.y(), self.gutter.width(), rect.height())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        cr = self.contentsRect()
        self.gutter.setGeometry(QRect(cr.left(), cr.top(), self.gutter_width(),
                                      cr.height()))

    def paint_gutter(self, ev):
        p = QPainter(self.gutter)
        p.fillRect(ev.rect(), QColor('#202226'))
        block = self.firstVisibleBlock()
        n = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(
            self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        w = self.gutter.width()
        fm = self.fontMetrics()
        while block.isValid() and top <= ev.rect().bottom():
            if block.isVisible() and bottom >= ev.rect().top():
                line = n + self.first_line
                if line in self.current:
                    p.fillRect(0, top, w, bottom - top, QColor('#2c3038'))
                    p.fillRect(w - 4, top, 4, bottom - top,
                               QColor(self.colors.get(line, CURRENT_COLOR)))
                elif line in self.related:
                    p.fillRect(w - 3, top + 1, 3, bottom - top - 2, QColor('#8a8a8a'))
                colors = self.chips.get(line)
                if colors:
                    x = 3
                    for c in colors[:3]:
                        p.fillRect(x, top + 3, 4, bottom - top - 6, QColor(c))
                        x += 5
                strong = line in self.current
                p.setPen(QColor('#ffffff' if strong else
                                ('#c8c8c8' if line in self.hot else '#6a6a6a')))
                p.setFont(self._gutter_bold if strong else self.font())
                p.drawText(0, top, w - 7, fm.height(),
                           Qt.AlignmentFlag.AlignRight, str(line))
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            n += 1

    # --- highlights ---
    def set_chips(self, chips):
        self.chips = {int(k): v for k, v in chips.items()}
        self.hot = set(self.chips)
        self.gutter.update()

    def set_highlight(self, current=(), related=(), colors=None):
        """`current`: the lines the selection comes from, tinted with their
        colour in `colors` (line -> colour; the lane colour of the pulse)
        and marked in the gutter; `related`: the call chain above them,
        a soft grey band."""
        self.current = set(int(x) for x in current)
        self.related = set(int(x) for x in related) - self.current
        self.colors = {int(k): v for k, v in (colors or {}).items()}
        sels = []
        for line in self.related:
            sel = self._sel(line, QColor(255, 255, 255, 30))
            if sel:
                sels.append(sel)
        for line in self.current:
            c = QColor(self.colors.get(line, CURRENT_COLOR))
            c.setAlpha(85)
            sel = self._sel(line, c)
            if sel:
                sels.append(sel)
        self.setExtraSelections(sels)
        self.gutter.update()

    def _sel(self, line, color):
        blk = self.document().findBlockByNumber(line - self.first_line)
        if not blk.isValid():
            return None
        sel = QTextEdit.ExtraSelection()
        sel.format.setBackground(color)
        sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        sel.cursor = QTextCursor(blk)
        sel.cursor.clearSelection()
        return sel

    def line_visible(self, line):
        blk = self.document().findBlockByNumber(int(line) - self.first_line)
        if not blk.isValid():
            return False
        g = self.blockBoundingGeometry(blk).translated(self.contentOffset())
        return g.top() >= 0 and g.bottom() <= self.viewport().height()

    def scroll_to(self, line):
        """Centre `line` -- only when it is off screen, so clicking through
        pulses on one screenful of code does not make the text jump."""
        if self.line_visible(line):
            return
        blk = self.document().findBlockByNumber(int(line) - self.first_line)
        if not blk.isValid():
            return
        cur = QTextCursor(blk)
        self.setTextCursor(cur)
        self.centerCursor()

    def line_at(self, pos):
        cur = self.cursorForPosition(pos)
        return cur.blockNumber() + self.first_line

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        # a drag that selected text (to copy it) is not a line click
        if (ev.button() == Qt.MouseButton.LeftButton
                and not self.textCursor().hasSelection()):
            self.lineClicked.emit(self.line_at(ev.pos()))

    def mouseDoubleClickEvent(self, ev):
        super().mouseDoubleClickEvent(ev)
        self.lineDoubleClicked.emit(self.line_at(ev.pos()))


class ParamsTable(QTableWidget):
    def __init__(self, params, shots, parent=None):
        super().__init__(parent)
        names = list(params.get('names', []))
        disp = params.get('display', {})
        n = max([len(v) for v in disp.values()] + [0])
        self.setColumnCount(len(names) + 1)
        self.setRowCount(n)
        self.setHorizontalHeaderLabels(['shot'] + names)
        for s in range(n):
            self.setItem(s, 0, QTableWidgetItem(str(s)))
            for j, name in enumerate(names):
                vals = disp.get(name, [])
                self.setItem(s, j + 1, QTableWidgetItem(
                    vals[s] if s < len(vals) else ''))
        self.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)


class CodePane(QTabWidget):
    """Tabs: one SourceView per source, the parameter table, an About tab."""
    lineClicked = pyqtSignal(str, int)         # source id, file line
    lineDoubleClicked = pyqtSignal(str, int)
    shotClicked = pyqtSignal(int)

    ORDER = ('seq', 'qua', 'qua_real', 'config')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.views = {}
        self.setDocumentMode(True)

    def load(self, meta):
        cur = self.currentIndex()
        self.clear()
        self.views = {}
        sources = meta.get('sources', {})
        keys = [k for k in self.ORDER if k in sources] + [
            k for k in sources if k not in self.ORDER]
        for key in keys:
            src = sources[key]
            v = SourceView(src.get('text', ''), src.get('first_line', 1),
                           src.get('language', 'python'))
            v.lineClicked.connect(lambda line, k=key: self.lineClicked.emit(k, line))
            v.lineDoubleClicked.connect(
                lambda line, k=key: self.lineDoubleClicked.emit(k, line))
            self.views[key] = v
            self.addTab(v, src.get('title', key))
            note = src.get('note')
            if note:
                self.setTabToolTip(self.count() - 1, note)
        pt = ParamsTable(meta.get('params', {}), meta.get('shots', []))
        pt.cellClicked.connect(lambda r, c: self.shotClicked.emit(int(r)))
        self.addTab(pt, 'params')
        about = QLabel()
        about.setWordWrap(True)
        about.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        about.setAlignment(Qt.AlignmentFlag.AlignTop)
        info = meta.get('info', {})
        lines = [f"<b>{meta.get('title', '')}</b>", meta.get('subtitle', '')]
        for k, v in info.items():
            if k in ('title', 'subtitle'):
                continue
            lines.append(f"<b>{k}</b>: {v}")
        about.setText('<br>'.join(str(x) for x in lines if x))
        about.setContentsMargins(8, 8, 8, 8)
        self.addTab(about, 'about')
        if 0 <= cur < self.count():
            self.setCurrentIndex(cur)

    def set_chips(self, key, chips):
        v = self.views.get(key)
        if v is not None:
            v.set_chips(chips)

    def highlight(self, key, current=(), related=(), scroll=True, raise_tab=False,
                  colors=None, scroll_line=None):
        v = self.views.get(key)
        if v is None:
            return
        v.set_highlight(current, related, colors)
        if scroll and current:
            v.scroll_to(scroll_line if scroll_line is not None else min(current))
        if raise_tab:
            self.setCurrentWidget(v)

    def clear_highlights(self):
        for v in self.views.values():
            v.set_highlight((), ())
