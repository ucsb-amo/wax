"""Qt hygiene for the liveOD window tests, so that they cannot take the pytest
process down or leave work for a later test.

Two ways a Qt test aborts the whole process ("Fatal Python error: Aborted"):

* a QThread whose Python object is destroyed while the thread still runs
  (qFatal "QThread: Destroyed while thread is still running");
* an exception in a slot while ``sys.excepthook`` is Python's default, which
  PyQt turns into qFatal. pyqtgraph raises one when a named ViewBox (ImageView
  registers two) is destroyed after another view's menu: its ``destroyed``
  handler walks every registered view.

Both used to happen whenever the garbage collector got round to a window from an
earlier test. So: every thread a test starts is stopped and joined here with a
bound, and one that will not stop is kept referenced (never destroyed) and
reported; windows are deleted and collected inside their own test, under a
recording excepthook.
"""
import gc
import threading
import weakref

from PyQt6.QtWidgets import QApplication

# threads that did not stop in time: kept for the life of the process, so that
# they are never destroyed while running
UNJOINED = []

_APP = []


def session_app():
    """The QApplication, kept for the rest of the session once made here. An
    application dropped between test modules takes Qt objects that other
    modules still hold (theme's notifier, windows awaiting collection) with it."""
    app = QApplication.instance() or QApplication([])
    if not _APP:
        _APP.append(app)
    return app


def join_or_keep(thread, timeout_s=10.0) -> str:
    """Wait for a QThread or threading.Thread to finish; "" if it did, else what
    is still running (and the thread is kept, never destroyed)."""
    if thread is None:
        return ""
    if hasattr(thread, "isRunning"):                  # QThread
        if thread.isRunning() and not thread.wait(int(timeout_s * 1000)):
            UNJOINED.append(thread)
            return f"{type(thread).__name__} {getattr(thread, 'name', '')} still running after {timeout_s:g} s"
        return ""
    thread.join(timeout_s)
    if thread.is_alive():
        UNJOINED.append(thread)
        return f"thread {thread.name} still running after {timeout_s:g} s"
    return ""


def new_threads(before):
    """Real threading.Threads started since ``before`` (a set from
    threading.enumerate()); QThreads show up only as dummy entries, skipped."""
    return [t for t in threading.enumerate()
            if t not in before and t is not threading.current_thread()
            and not isinstance(t, threading._DummyThread)]


class SlotErrors:
    """A sys.excepthook that records instead of letting PyQt abort."""

    def __init__(self):
        self.seen = []

    def __call__(self, exc_type, exc, tb):
        self.seen.append(f"{exc_type.__name__}: {exc}")


def pyqtgraph_views():
    """A weak snapshot of pyqtgraph's registered views."""
    from pyqtgraph.graphicsItems.ViewBox.ViewBox import ViewBox
    return weakref.WeakSet(ViewBox.AllViews.keys())


def forget_views_since(before):
    """Take the views registered since ``before`` out of pyqtgraph's registry and
    drop their ``destroyed`` handler, so deleting them walks no other view."""
    from pyqtgraph.graphicsItems.ViewBox.ViewBox import ViewBox
    for view in list(ViewBox.AllViews.keys()):
        if view in before:
            continue
        try:
            view.destroyed.disconnect()
        except (TypeError, RuntimeError):
            pass
        ViewBox.AllViews.pop(view, None)
        name = getattr(view, "name", None)
        if name is not None and ViewBox.NamedViews.get(name) is view:
            ViewBox.NamedViews.pop(name, None)


def delete_widgets(app, widgets):
    """Delete the widgets' C++ objects now, then collect their Python side."""
    from PyQt6 import sip
    for w in widgets:
        if w is not None and not sip.isdeleted(w):
            sip.delete(w)
    app.processEvents()
    gc.collect()
    app.processEvents()
