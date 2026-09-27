"""The Camera Viewer dashboard panels live in waxx and stay machine-agnostic:
importing them pulls in beacon and the waxx dashboard helpers, never kexp."""
from __future__ import annotations

import os
import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=180, env=env)


def test_the_panel_module_imports_no_kexp():
    r = _run(
        "import sys\n"
        "import waxx.util.guis.camera_viewer.camera_viewer_panel as p\n"
        "import beacon.camera.viewer.main_window\n"
        "bad = sorted(m for m in sys.modules if m == 'kexp' or m.startswith('kexp.'))\n"
        "print('KEXP', bad)\n"
        "assert p.CameraViewerServerPanel.LAYOUT_KEY == 'dashboard_server'\n"
        "assert p.CameraViewerClientPanel.LAYOUT_KEY == 'dashboard_client'\n"
        "sys.exit(1 if bad else 0)\n")
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_panels_embed_the_viewer_with_their_own_layout_key():
    import inspect
    from waxx.util.guis.camera_viewer import camera_viewer_panel as p
    src = inspect.getsource(p)
    assert "kexp" not in src.replace("machine-agnostic", "")
    assert "layout_key=self.LAYOUT_KEY" in src and "embed_as_window=True" in src
    assert issubclass(p.CameraViewerClientPanel, p.CameraViewerServerPanel)
