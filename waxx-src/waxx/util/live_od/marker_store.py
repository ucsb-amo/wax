"""Markers (pins) on the liveOD image, kept by the liveOD server.

One list of markers per camera, in a JSON file on the machine that runs liveOD, so
the acquisition window and every remote viewer see the same pins and they survive
restarts. A marker is ``{"x": px, "y": px, "shape": one of SHAPES}``, in image
pixel coordinates.

No Qt here; the file is only opened when markers are first read or written.
"""

import json
import os
import threading

DEFAULT_PATH = os.path.join(os.path.expanduser("~"), ".waxx", "live_od_markers.json")

# pyqtgraph symbol names, and what the right-click menu calls them
SHAPES = {
    "crosshair": "Crosshair",
    "+": "Plus",
    "x": "Cross",
    "o": "Circle",
    "s": "Square",
    "d": "Diamond",
    "t": "Triangle",
    "star": "Star",
}
DEFAULT_SHAPE = "crosshair"
DEFAULT_SIZE_PX = 30.0          # a marker's extent, in image pixels: it zooms with the image
DEFAULT_COLOR = "#ff4081"
SIZE_LIMITS_PX = (1.0, 5000.0)
MAX_LABEL_LENGTH = 40
MAX_MARKERS_PER_CAMERA = 50


def _clean_color(value) -> str:
    text = str(value or "").strip().lower()
    if len(text) == 7 and text[0] == "#" and all(c in "0123456789abcdef" for c in text[1:]):
        return text
    return DEFAULT_COLOR


def clean_markers(markers) -> list:
    """Whatever arrived (from a file, from the network) as a list of valid markers:
    ``{"x", "y"}`` in image pixels, ``"shape"`` (a SHAPES key), ``"size"`` in image
    pixels, ``"color"`` as #rrggbb, ``"label"``, ``"hidden"`` (kept, not drawn).
    Markers saved before size, color, label and hidden existed get the defaults."""
    cleaned = []
    for marker in list(markers or [])[:MAX_MARKERS_PER_CAMERA]:
        try:
            x, y = float(marker["x"]), float(marker["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if x != x or y != y:    # NaN
            continue
        shape = marker.get("shape", DEFAULT_SHAPE)
        try:
            size = float(marker.get("size", DEFAULT_SIZE_PX))
            if size != size:
                size = DEFAULT_SIZE_PX
        except (TypeError, ValueError):
            size = DEFAULT_SIZE_PX
        cleaned.append({
            "x": x, "y": y,
            "shape": shape if shape in SHAPES else DEFAULT_SHAPE,
            "size": min(max(size, SIZE_LIMITS_PX[0]), SIZE_LIMITS_PX[1]),
            "color": _clean_color(marker.get("color")),
            "label": str(marker.get("label") or "")[:MAX_LABEL_LENGTH],
            "hidden": bool(marker.get("hidden", False)),
        })
    return cleaned


class MarkerStore:
    def __init__(self, path=None):
        self._path = path or DEFAULT_PATH
        self._lock = threading.Lock()
        self._markers = None    # camera_key -> list of markers, once loaded

    def _load(self):
        if self._markers is not None:
            return
        self._markers = {}
        try:
            with open(self._path, encoding="utf-8") as f:
                for camera_key, markers in dict(json.load(f)).items():
                    self._markers[str(camera_key)] = clean_markers(markers)
        except FileNotFoundError:
            pass
        except Exception:
            self._markers = {}      # unreadable: start again rather than fail a run

    def get(self, camera_key: str) -> list:
        with self._lock:
            self._load()
            return [dict(m) for m in self._markers.get(str(camera_key), [])]

    def set(self, camera_key: str, markers) -> list:
        """Replace one camera's markers and write the file. Returns what was stored."""
        cleaned = clean_markers(markers)
        with self._lock:
            self._load()
            self._markers[str(camera_key)] = cleaned
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._markers, f, indent=1)
            os.replace(tmp, self._path)
        return [dict(m) for m in cleaned]
