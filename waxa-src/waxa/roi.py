import numpy as np
import pandas as pd
import os
import cv2
import sys

from waxa.data.server_talk import server_talk as st
from waxa.image_processing.compute_ODs import compute_OD
from waxa.config.img_types import img_types

import h5py

try:
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout,
        QWidget, QLabel, QComboBox, QSizePolicy, QFrame, QPushButton,
        QMessageBox,
    )
    from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QShortcut, QKeySequence, QFont, QIcon
    from PyQt6.QtCore import Qt, QRect, QPoint, QTimer, QCoreApplication
    _PYQT6_AVAILABLE = True
except ImportError:
    _PYQT6_AVAILABLE = False

_ROI_EXCEL_CACHE = {}

def _load_roi_excel_cached(path):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None

    cached = _ROI_EXCEL_CACHE.get(path)
    if cached is not None and cached['mtime'] == mtime:
        return cached['df']

    df = pd.read_excel(path)
    _ROI_EXCEL_CACHE[path] = {
        'mtime': mtime,
        'df': df,
    }
    return df

class ROI():
    def __init__(self,
                 run_id=0,
                 roi_id=None,
                 key="",
                 use_saved_roi=True,
                 lite=False,
                 printouts=True,
                 server_talk=None,
                 current_file_path=None,
                 current_saved_roi=None):
        
        if server_talk == None:
            self.server_talk = st()
        else:
            self.server_talk = server_talk

        self.roix = [-1,-1]
        self.roiy = [-1,-1]
        self.key = key
        self.run_id = run_id
        self._current_file_path = current_file_path
        self._current_saved_roi = current_saved_roi
        self.load_roi(roi_id,
                      use_saved=use_saved_roi,
                      lite=lite,
                      printouts=printouts)

    def crop(self,OD):
        """Crops the given ndarray according the ROI.

        Args:
            OD (np.ndarray): The ndarray to be cropped.

        Returns:
            ndarray: The cropped ndarray.
        """        
        OD: np.ndarray
        idx_y = range(self.roiy[0],self.roiy[1])
        idx_x = range(self.roix[0],self.roix[1])
        cropOD = OD.take(idx_y,axis=OD.ndim-2).take(idx_x,axis=OD.ndim-1)
        return cropOD

    def load_roi(self,roi_id=None,use_saved=True,lite=False,
                 printouts=True):
        """Loads an ROI according to the provided roi_id.

        Args:
            roi_id (None, int, or str): Specifies which crop to use. If None,
            defaults to the ROI saved in the data if it exists, otherwise
            prompts the user to select an ROI using the GUI. If an int,
            interpreted as an run ID, which will be checked for a saved ROI and
            that ROI will be used. If a string, interprets as a key in the
            roi.xlsx document in the PotassiumData folder.
            printouts (bool): If True, prints out information about the
            ROI loading process.

            use_saved (bool): If False, ignores saved ROI and forces creation of
            a new one.
        """
        
        # Check for ROI saved in the current data file.
        saved_roi_bool = self.read_roi_from_h5(lite=lite,printouts=printouts)
        if roi_id == None:
            if saved_roi_bool and use_saved:
                if printouts: print("Using saved ROI.")
                pass
            elif saved_roi_bool:
                if printouts: print("Saved ROI was found, but is being overridden.")
                if printouts: print("Specify the new ROI.")
                self.select_roi()
            else:
                if printouts: print("Specify the new ROI.")
                self.select_roi()
        else:
            if saved_roi_bool:
                if printouts: print("Saved ROI was found, but is being overridden.")
                pass

        # Checks for ROI saved in the specified run ID.
        if isinstance(roi_id,int):
            if printouts: print("ROI specified by Run ID. Attempting to load ROI...")
            saved_roi_bool = self.read_roi_from_h5(roi_id,printouts=printouts)
            if saved_roi_bool:
                if printouts: print(f"Using ROI loaded from run {roi_id}.")
            else:
                if printouts: print(f"Specify the new ROI.")
                self.select_roi()

        if isinstance(roi_id,str):
            if printouts: print("ROI specified by string. Referencing roi.xslx spreadsheet (PotassiumData)...")
            roi_exists = self.read_roi_from_excel(roi_id,printouts=printouts)
            if not roi_exists:
                if printouts: print(f"Creating ROI for key {roi_id}.")
                self.select_roi()
                self._update_excel()

        if self.check_for_blank_roi():
            if printouts: print("ROI was not specified. Defaulting to whole image.")
            px, py = self.get_image_size()
            self.roix = [0,px]
            self.roiy = [0,py]

    def save_roi_h5(self, lite=False, printouts=False):
        fpath, _ = self.server_talk.get_data_file(self.run_id,lite=lite)
        with h5py.File(fpath,'r+') as f:
            f.attrs['roix'] = self.roix
            f.attrs['roiy'] = self.roiy
        if printouts: print(f"ROI saved to h5 file at {fpath}")

    def save_roi_excel(self,key=""):
        if self.key == "" and key == "":
            raise ValueError("You must specify a key to save the ROI to the spreadsheet.")
        if not isinstance(key,str):
            raise ValueError("The specified key must be a string.")
        
        if not key == "":
            self.key = key
        else:
            # saving will use the key already associated with self.roi.
            pass
        self._update_excel()

    def get_image_size(self):
        """Gets the size in pixels of the images (horizontal, vertical) in this run.

        Returns:
            px, py: The image dimensions (rows, columns).
        """        
        # Reuse the already-resolved file path when available to avoid
        # re-triggering network drive checks.
        if self._current_file_path is not None:
            fpath = self._current_file_path
        else:
            fpath, _ = self.server_talk.get_data_file(self.run_id)
        with h5py.File(fpath) as f:
            py, px = f['data']['images'].shape[-2:]
        return px, py

    def read_roi_from_h5(self, run_id=[], lite=False, printouts=True):
        """Looks in the hdf5 file with the corresponding run ID and attempts to
        read out a saved ROI. Returns True if successful and False otherwise.

        Args:
            run_id (int): The run ID of the run in which to look for a
            saved ROI.

        Returns:
            bool: Returns True if successful and False otherwise.
        """        
        if run_id == []:
            run_id = self.run_id
        try:
            # Fast path: atomdata already loaded this file and optionally read
            # roix/roiy, so avoid another resolver + file-open roundtrip.
            if run_id == self.run_id and self._current_saved_roi is not None:
                if self._current_saved_roi is False:
                    if printouts: print(f"No ROI saved in run {run_id} (cached).")
                    return False
                roix, roiy = self._current_saved_roi
                self.roix = roix
                self.roiy = roiy
                if printouts: print(f"ROI loaded from run {run_id} (cached).")
                return True

            if run_id == self.run_id and self._current_file_path is not None:
                fpath = self._current_file_path
            else:
                fpath, run_id = self.server_talk.get_data_file(run_id,lite=lite)

            with h5py.File(fpath) as f:
                roix = f.attrs['roix']
                roiy = f.attrs['roiy']
            self.roix = roix
            self.roiy = roiy
            if printouts: print(f"ROI loaded from run {run_id}.")
            return True
        except Exception as e:
            if printouts: print(f"No ROI saved in run {run_id}.")
            return False
        
    def read_roi_from_excel(self,key,printouts=True):
        """Reads in the ROI with the corresponding key from the roi spreadsheet
        (roi.xlsx in PotassiumData), if it exists. Returns True if successful
        and False otherwise.

        Args:
            key (str): The key of the ROI to retrieve from the spreadsheet.

        Returns:
            bool: Returns True if successful and False otherwise.
        """        
        self.key = key
        if key == "":
            raise ValueError("ROI key must be a non-empty string.")
        roicsv = _load_roi_excel_cached(self.server_talk.roi_csv_path)
        keymatch = roicsv.loc[ roicsv['key'] == self.key ]
        if np.any(keymatch):
            if printouts: print(f"ROI {key} found.")
            self.key = key
            self.roix = [ keymatch['roix0'].values[0], keymatch['roix1'].values[0] ]
            self.roiy = [ keymatch['roiy0'].values[0], keymatch['roiy1'].values[0] ]
            return True
        else:
            if printouts: print(f"ROI with key {key} does not exist.")
            return False

    def check_for_blank_roi(self):
        """The ROI selection GUI output is all -1s if no ROI is selected.
        Detects this result so that it can be handled.

        Returns:
            bool: whether or not the ROI selection is valid.
        """        
        return np.all(np.array([*self.roix,*self.roiy]) == -1)
    
    def select_roi(self, run_id=[]):
        """Brings up the GUI to select a new ROI rectangle. The user should
        click and drag (LMB) in order to select a rectangle, then hit Enter to
        submit their selection. RMB clears the rectangle, and Escape/the X
        button will close the window without selecting an ROI (in most cases,
        resulting in an ROI that spans the entire image).

        Args:
            run_id (int, optional): The run_id to use for displaying ODs during
            ROI selection.
        """        
        if run_id == []:
            run_id = self.run_id
        file_path = self._current_file_path if run_id == self.run_id else None
        update_bool, roix, roiy = roi_creator(
            run_id,
            self.key,
            self.server_talk,
            file_path=file_path,
        ).get_roi_rectangle()
        if update_bool:
            self.roix, self.roiy = roix, roiy
        else:
            print("ROI not selected, aborting.")

    def _update_excel(self):
        """Saves the ROI to the excel spreadsheet (roi.xlsx) in the
        PotassiumData folder. If the key already exists, updates the existing
        ROI. If not, creates a new line.
        """        
        key = self.key
        if key == "":
            raise ValueError("The key must be nonempty in order to save the ROI.")

        # Read the excel file
        df = pd.read_excel(self.server_talk.roi_csv_path)
        new_values = [*self.roix, *self.roiy]

        # Check if the label exists
        if key in df.iloc[:, 0].values:
            # Find the index of the row with the given label
            index = df[df.iloc[:, 0] == key].index[0]
            # Replace the row values with new values
            df.iloc[index, 1:] = new_values
        else:
            # Append a new row with the given label and new values
            new_row = pd.DataFrame([[key] + new_values], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)

        # Save the updated dataframe back to the excel file
        df.to_excel(self.server_talk.roi_csv_path, index=False)
        # Invalidate cached spreadsheet after write.
        if self.server_talk.roi_csv_path in _ROI_EXCEL_CACHE:
            del _ROI_EXCEL_CACHE[self.server_talk.roi_csv_path]
        print(f"Updated the spreadsheet ROI with key {key}.")

class _RoiImageWidget(QWidget):
    """Image display area for ROI selection. Handles rendering and forwards
    mouse events to the owning _RoiSelectorDialog."""

    def __init__(self, dialog, parent=None):
        super().__init__(parent)
        self._dialog = dialog
        self._pixmap = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(180, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def _image_rect(self):
        """QRect within the widget where the image is letterboxed."""
        if self._pixmap is None or self._pixmap.width() <= 0 or self._pixmap.height() <= 0:
            return QRect(0, 0, self.width(), self.height())
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        iw, ih = int(pw * scale), int(ph * scale)
        return QRect((ww - iw) // 2, (wh - ih) // 2, iw, ih)

    def widget_to_display(self, px, py):
        """Map widget pixel coordinates to display_image pixel coordinates."""
        rect = self._image_rect()
        dw, dh = self._dialog.display_size()
        lx = max(0, min(px - rect.x(), rect.width() - 1))
        ly = max(0, min(py - rect.y(), rect.height() - 1))
        dx = int(lx * dw / max(rect.width(), 1))
        dy = int(ly * dh / max(rect.height(), 1))
        return max(0, min(dx, dw - 1)), max(0, min(dy, dh - 1))

    def display_to_widget(self, dx, dy):
        """Map display_image pixel coordinates to widget coordinates."""
        rect = self._image_rect()
        dw, dh = self._dialog.display_size()
        wx = rect.x() + int(dx * rect.width() / max(dw, 1))
        wy = rect.y() + int(dy * rect.height() / max(dh, 1))
        return wx, wy

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if self._pixmap is not None:
            painter.drawPixmap(self._image_rect(), self._pixmap)

        # ── Active ROI rectangle ───────────────────────────────────────────
        roi_disp = self._dialog.active_roi_in_display_coords()
        if roi_disp is not None:
            dx0, dx1, dy0, dy1 = roi_disp
            wx0, wy0 = self.display_to_widget(dx0, dy0)
            wx1, wy1 = self.display_to_widget(dx1, dy1)
            is_preset = self._dialog.active_roi_source.startswith("preset:")
            pen = QPen(QColor(255, 220, 0) if is_preset else QColor(255, 255, 255), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRect(QPoint(wx0, wy0), QPoint(wx1, wy1)))

        # ── In-progress ROI drag (dashed white) ───────────────────────────
        drag = self._dialog.drag_rect_in_display_coords()
        if drag is not None:
            dx0, dy0, dx1, dy1 = drag
            wx0, wy0 = self.display_to_widget(dx0, dy0)
            wx1, wy1 = self.display_to_widget(dx1, dy1)
            pen = QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRect(QPoint(wx0, wy0), QPoint(wx1, wy1)))

        # ── In-progress zoom drag (dotted cyan) ───────────────────────────
        zoom_drag = self._dialog.zoom_drag_in_display_coords()
        if zoom_drag is not None:
            dx0, dy0, dx1, dy1 = zoom_drag
            wx0, wy0 = self.display_to_widget(dx0, dy0)
            wx1, wy1 = self.display_to_widget(dx1, dy1)
            pen = QPen(QColor(0, 200, 255), 1, Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRect(QPoint(wx0, wy0), QPoint(wx1, wy1)))

        painter.end()

    def mousePressEvent(self, event):
        dx, dy = self.widget_to_display(event.pos().x(), event.pos().y())
        self._dialog.on_mouse_press(event.button(), dx, dy)
        self.update()

    def mouseMoveEvent(self, event):
        dx, dy = self.widget_to_display(event.pos().x(), event.pos().y())
        self._dialog.on_mouse_move(dx, dy)
        self.update()

    def mouseReleaseEvent(self, event):
        dx, dy = self.widget_to_display(event.pos().x(), event.pos().y())
        self._dialog.on_mouse_release(event.button(), dx, dy)
        self.update()

    def wheelEvent(self, event):
        self._dialog.on_wheel(event.angleDelta().y())
        self.update()

    def keyPressEvent(self, event):
        self._dialog.keyPressEvent(event)

    def resizeEvent(self, event):
        self.update()


class _RoiSelectorDialog(QDialog):
    """Full PyQt6 ROI selector window.

    Top bar: instructions, preset dropdown, live ROI status, warning.
    Below: resizable image canvas with all mouse interactions.
    """

    def __init__(self, creator, preset_entries, parent=None):
        # Set window flags before calling super().__init__()
        if sys.platform.startswith("win"):
            # Use Qt.WindowType instead of passing to super
            pass
        super().__init__(parent)
        
        if sys.platform.startswith("win"):
            self.setWindowFlags(
                self.windowFlags()
                | Qt.WindowType.WindowStaysOnTopHint
            )
        
        self.creator = creator
        self.preset_entries = preset_entries
        self.preset_keys = [e[0] for e in preset_entries]

        # ── Image / zoom state ────────────────────────────────────────────
        self.original_image = creator.image.copy()
        self.display_image = self.original_image.copy()
        self.zoom_region = None          # (x0, y0, x1, y1) in original coords
        self.img_index = 0

        # ── ROI state ─────────────────────────────────────────────────────
        self.active_roi_bounds = None    # (x0, x1, y0, y1) in original coords
        self.active_roi_source = "none"
        self.warning_message = ""
        self.preset_index = -1

        # ── Drawing state (in display_image pixel coords) ─────────────────
        self._draw_start = None
        self._draw_end = None
        self._is_drawing = False
        self._zoom_start = None
        self._zoom_end = None
        self._is_zooming = False

        # ── Result ────────────────────────────────────────────────────────
        self.result_update_bool = False
        self.result_roix = np.array([-1, -1])
        self.result_roiy = np.array([-1, -1])

        self._build_ui()
        self._setup_hotkeys()
        self._refresh_image()
        self.setWindowTitle("ROI Selector")
        self._set_emoji_window_icon()
        self.resize(450, 550)

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.setStyleSheet(
            "QDialog { background: #0f1117; }"
            "QFrame#top_bar {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #171c2b, stop:1 #22293d);"
            "  border: 1px solid #3a435e; border-radius: 8px;"
            "}"
            "QLabel#instructions { color: #a8b2c8; font-size: 11px; }"
            "QLabel#group_label { color: #d4d8e8; font-size: 10px; }"
            "QLabel#status_label { color: #ffe18a; font-size: 12px; font-family: Consolas, monospace; }"
            "QLabel#warning_label { color: #ff8a8a; font-size: 11px; }"
            "QComboBox { background: #252c40; color: white; border: 1px solid #576081; "
            "  padding: 2px 6px; font-size: 11px; min-height: 22px; border-radius: 5px; }"
            "QComboBox::drop-down { border: 0px; width: 20px; }"
            "QComboBox::down-arrow { image: none; width: 0px; height: 0px; }"
            "QComboBox QAbstractItemView { background: #252c40; color: white; "
            "  selection-background-color: #3f5b96; border: 1px solid #576081; }"
            "QPushButton {"
            "  background: #2a324a; color: #f3f5fb; border: 1px solid #5f6a90; "
            "  border-radius: 5px; padding: 4px 10px; min-height: 26px; font-size: 11px;"
            "}"
            "QPushButton:hover { background: #364264; }"
            "QPushButton:pressed { background: #20273a; }"
            "QPushButton#accent { background: #3f6fd8; border: 1px solid #82a4f1; }"
            "QPushButton#accent:hover { background: #5582e5; }"
            "QPushButton#mini {"
            "  padding: 0px 2px; min-height: 18px; min-width: 18px;"
            "  font-size: 12px; border-radius: 4px;"
            "}"
            "QPushButton#mini_accent {"
            "  padding: 0px 2px; min-height: 18px; min-width: 18px;"
            "  font-size: 12px; border-radius: 4px;"
            "  background: #3f6fd8; border: 1px solid #82a4f1; color: #f3f5fb;"
            "}"
            "QPushButton#mini_accent:hover { background: #5582e5; }"
        )

        # Top bar
        top_bar = QFrame()
        top_bar.setObjectName("top_bar")
        top_bar.setFrameShape(QFrame.Shape.StyledPanel)
        top_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_layout = QVBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 6, 8, 6)
        top_layout.setSpacing(6)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(6)

        img_widget = QWidget()
        img_layout = QVBoxLayout(img_widget)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(2)
        img_buttons_widget = QWidget()
        img_buttons_layout = QHBoxLayout(img_buttons_widget)
        img_buttons_layout.setContentsMargins(0, 0, 0, 0)
        img_buttons_layout.setSpacing(4)
        self.btn_prev_img = self._make_toolbar_button("⬅️", self._action_prev_image, mini=True)
        self.btn_prev_img.setToolTip("Previous image")
        self.btn_next_img = self._make_toolbar_button("➡️", self._action_next_image, mini=True)
        self.btn_next_img.setToolTip("Next image")
        img_buttons_layout.addWidget(self.btn_prev_img)
        img_buttons_layout.addWidget(self.btn_next_img)
        img_layout.addWidget(img_buttons_widget)

        gain_buttons_widget = QWidget()
        gain_buttons_layout = QHBoxLayout(gain_buttons_widget)
        gain_buttons_layout.setContentsMargins(0, 0, 0, 0)
        gain_buttons_layout.setSpacing(4)

        self.btn_dimmer = self._make_toolbar_button("🌙", self._action_dimmer, mini=True)
        self.btn_dimmer.setToolTip("Dimmer (Down Arrow)")
        self.btn_brighter = self._make_toolbar_button("☀️", self._action_brighter, mini=True)
        self.btn_brighter.setToolTip("Brighter (Up Arrow)")
        gain_buttons_layout.addWidget(self.btn_dimmer)
        gain_buttons_layout.addWidget(self.btn_brighter)
        img_layout.addWidget(gain_buttons_widget)

        bold_font = self.btn_prev_img.font()
        bold_font.setBold(True)
        self.btn_prev_img.setFont(bold_font)
        self.btn_next_img.setFont(bold_font)
        self.btn_dimmer.setFont(bold_font)
        self.btn_brighter.setFont(bold_font)

        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(130)
        self.preset_combo.addItem("select preset")
        placeholder_font = QFont(self.preset_combo.font())
        placeholder_font.setItalic(True)
        self.preset_combo.setItemData(0, placeholder_font, Qt.ItemDataRole.FontRole)
        for key in self.preset_keys:
            self.preset_combo.addItem(key)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_combo_changed)
        self.preset_combo.setToolTip("Choose a saved ROI preset")

        self.btn_zoom_out = self._make_toolbar_button("Zoom Out", self._action_zoom_out, mini=True)
        self.btn_zoom_out.setToolTip("Zoom out (Right Mouse Button or Mouse Wheel)")
        self.btn_clear = self._make_toolbar_button("Clear ROI", self._action_clear, mini=True)
        self.btn_clear.setToolTip("Clear current ROI at full scale")
        self.btn_help = self._make_toolbar_button("Instructions", self._show_instructions, mini=True)
        self.btn_help.setMinimumHeight(22)
        self.btn_help.setMaximumHeight(22)
        self.btn_help.setToolTip("Show mouse and keyboard instructions")
        
        zoom_clear_widget = QWidget()
        zoom_clear_layout = QVBoxLayout(zoom_clear_widget)
        zoom_clear_layout.setContentsMargins(0, 0, 0, 0)
        zoom_clear_layout.setSpacing(2)
        zoom_clear_layout.addWidget(self.btn_zoom_out)
        zoom_clear_layout.addWidget(self.btn_clear)
        zoom_clear_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.btn_accept = self._make_toolbar_button("Accept", self._accept_roi, accent=True, mini=True)
        self.btn_accept.setToolTip("Accept ROI (Enter)")
        self.btn_cancel = self._make_toolbar_button("Cancel", self._cancel_dialog, mini=True)
        self.btn_cancel.setToolTip("Cancel ROI selection (Escape)")
        
        action_widget = QWidget()
        action_layout = QVBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(2)
        action_layout.addWidget(self.btn_cancel)
        action_layout.addWidget(self.btn_accept)
        action_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        centered_help_widget = QWidget()
        centered_help_layout = QHBoxLayout(centered_help_widget)
        centered_help_layout.setContentsMargins(0, 0, 0, 0)
        centered_help_layout.setSpacing(0)
        centered_help_layout.addStretch(1)
        centered_help_layout.addWidget(self.btn_help)
        centered_help_layout.addStretch(1)
        centered_help_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        controls_row.addWidget(img_widget)
        controls_row.addWidget(zoom_clear_widget)
        controls_row.addWidget(self.preset_combo)
        controls_row.addWidget(centered_help_widget, 1)
        controls_row.addWidget(action_widget)
        top_layout.addLayout(controls_row)

        info_row = QHBoxLayout()
        info_row.setSpacing(8)

        self.roi_label = QLabel("ROI: none")
        self.roi_label.setObjectName("status_label")
        self.roi_label.setMinimumWidth(120)
        info_row.addWidget(self.roi_label)

        self.warning_label = QLabel("")
        self.warning_label.setObjectName("warning_label")
        self.warning_label.setMinimumWidth(140)
        info_row.addWidget(self.warning_label, 1)

        top_layout.addLayout(info_row)

        layout.addWidget(top_bar)

        # Image canvas
        self.image_widget = _RoiImageWidget(self)
        layout.addWidget(self.image_widget)

        self.image_widget.setStyleSheet("border: 1px solid #2f3447; border-radius: 8px;")

    def _make_toolbar_button(self, text, callback, accent=False, mini=False):
        btn = QPushButton(text)
        if mini and accent:
            btn.setObjectName("mini_accent")
        elif mini:
            btn.setObjectName("mini")
        elif accent:
            btn.setObjectName("accent")
        btn.clicked.connect(callback)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return btn

    def _set_emoji_window_icon(self):
        icon = QIcon()
        emoji = "🖼️"
        for size in (16, 24, 32, 48, 64):
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            font = QFont("Segoe UI Emoji")
            font.setPixelSize(max(10, int(size * 0.78)))
            painter.setFont(font)
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, emoji)
            painter.end()
            icon.addPixmap(pixmap)

        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def _show_instructions(self):
        message = (
            "<div style='background:#0f1117;color:#e6e9f5;font-size:12px;line-height:1.45;'>"
            "<table style='width:100%;border-collapse:collapse;'>"
            "<tr>"
            "<td style='vertical-align:top;padding-right:18px;'>"
            "<div style='margin:0 0 8px 0;'><b>Mouse Controls</b></div>"
            "<div style='margin:0 0 0.2em 0;'>LMB + drag: Select ROI rectangle</div>"
            "<div style='margin:0 0 0.2em 0;'>MMB + drag: Draw zoom region</div>"
            "<div style='margin:0 0 0.2em 0;'>MMB release: Zoom in to selected region</div>"
            "<div style='margin:0 0 0.2em 0;'>RMB: Zoom out if zoomed, otherwise clear ROI</div>"
            "<div style='margin:0 0 0.2em 0;'>Mouse wheel: Zoom out to full scale</div>"
            "</td>"
            "<td style='vertical-align:top;padding-left:18px;'>"
            "<div style='margin:0 0 8px 0;'><b>Keyboard Controls</b></div>"
            "<div style='margin:0 0 0.2em 0;'>Left Arrow: Previous image</div>"
            "<div style='margin:0 0 0.2em 0;'>Right Arrow: Next image</div>"
            "<div style='margin:0 0 0.2em 0;'>Up Arrow: Brighter</div>"
            "<div style='margin:0 0 0.2em 0;'>Down Arrow: Dimmer</div>"
            "<div style='margin:0 0 0.2em 0;'>Enter: Accept ROI</div>"
            "<div style='margin:0 0 0.2em 0;'>Escape: Cancel</div>"
            "</td>"
            "</tr>"
            "</table>"
            "</div>"
        )
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("ROI Selector Instructions")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Close)
        msg_box.setWindowModality(Qt.WindowModality.NonModal)
        msg_box.setModal(False)
        msg_box.setStyleSheet(
            "QMessageBox { background: #0f1117; color: #e6e9f5; }"
            "QLabel { color: #e6e9f5; }"
            "QPushButton {"
            "  background: #2a324a; color: #f3f5fb; border: 1px solid #5f6a90;"
            "  border-radius: 5px; padding: 4px 10px; min-height: 24px;"
            "}"
            "QPushButton:hover { background: #364264; }"
            "QPushButton:pressed { background: #20273a; }"
        )
        self._instructions_box = msg_box
        msg_box.show()

    def _setup_hotkeys(self):
        # Keep arrow hotkeys active even when another child widget (e.g. combo)
        # temporarily owns focus when the dialog first appears.
        self._shortcuts = []

        def bind(key, callback):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

        bind("Left", self._action_prev_image)
        bind("Right", self._action_next_image)
        bind("Up", self._action_brighter)
        bind("Down", self._action_dimmer)

    def _focus_image_canvas(self):
        # Explicit activation helps foreground modal dialogs on Windows.
        if not self.isVisible():
            return
        handle = self.windowHandle()
        if handle is not None:
            handle.requestActivate()
        self.raise_()
        self.activateWindow()
        self.image_widget.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def showEvent(self, event):
        super().showEvent(event)
        self._focus_image_canvas()
        # Retry focus aggressively on all platforms to ensure it sticks.
        QTimer.singleShot(0, self._focus_image_canvas)
        QTimer.singleShot(25, self._focus_image_canvas)
        QTimer.singleShot(75, self._focus_image_canvas)
        QTimer.singleShot(200, self._focus_image_canvas)

    # ── Image rendering ───────────────────────────────────────────────────

    def display_size(self):
        """Return (width, height) of the current display_image."""
        h, w = self.display_image.shape[:2]
        return w, h

    def _refresh_image(self):
        colorized = self.creator._colorize_image(self.display_image)
        h, w = colorized.shape[:2]
        rgb = colorized[:, :, ::-1].copy()   # BGR → RGB
        qimage = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self.image_widget.set_pixmap(QPixmap.fromImage(qimage))
        self._update_labels()

    def _update_labels(self):
        if self.active_roi_bounds is None:
            self.roi_label.setText("ROI: none")
        else:
            x0, x1, y0, y1 = self.active_roi_bounds
            src = self.active_roi_source
            self.roi_label.setText(f"ROI  x:[{x0}, {x1}]  y:[{y0}, {y1}]  ({src})")
        self.warning_label.setText(self.warning_message)

    # ── Preset helpers ────────────────────────────────────────────────────

    def _on_preset_combo_changed(self, index):
        if index <= 0:
            self.preset_index = -1
            return
        self._apply_preset(index - 1)   # offset for placeholder entry

    def _resolve_preset_bounds(self, roix, roiy, image_shape):
        # Backward compatibility: sentinel ROI (-1,-1,-1,-1) means full image.
        if int(roix[0]) == -1 and int(roix[1]) == -1 and int(roiy[0]) == -1 and int(roiy[1]) == -1:
            h, w = image_shape[:2]
            return (0, w, 0, h), False, True, True

        roi_bounds, was_clamped, valid = self.creator._clamp_roi_to_shape(
            roix, roiy, image_shape
        )
        return roi_bounds, was_clamped, valid, False

    def _apply_preset(self, preset_idx):
        if not (0 <= preset_idx < len(self.preset_entries)):
            return
        key, roix, roiy = self.preset_entries[preset_idx]
        roi_bounds, was_clamped, valid, is_full_image_sentinel = self._resolve_preset_bounds(
            roix, roiy, self.original_image.shape
        )
        self.preset_index = preset_idx
        if not valid:
            self.warning_message = f"Preset '{key}' is invalid for this image size."
            self.active_roi_bounds = None
            self.active_roi_source = f"preset:{key}"
            self._draw_start = self._draw_end = None
            self._is_drawing = False
            self._update_labels()
            self.image_widget.update()
            return
        if is_full_image_sentinel:
            self.warning_message = f"Preset '{key}' uses whole image."
        elif was_clamped:
            self.warning_message = f"Preset '{key}' clamped to image bounds."
        else:
            self.warning_message = ""
        self.active_roi_bounds = roi_bounds
        self.active_roi_source = f"preset:{key}"
        self._draw_start = self._draw_end = None
        self._is_drawing = False
        self._update_labels()
        self.image_widget.update()

    def _apply_preset_and_sync_combo(self, preset_idx):
        self._apply_preset(preset_idx)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(preset_idx + 1)   # +1 for placeholder
        self.preset_combo.blockSignals(False)

    def _reset_combo_to_none(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

    # ── Coordinate query methods (called by _RoiImageWidget.paintEvent) ───

    def active_roi_in_display_coords(self):
        """Return (dx0, dx1, dy0, dy1) in display_image coords, or None."""
        if self.active_roi_bounds is None:
            return None
        return self.creator._map_original_roi_to_display(
            self.active_roi_bounds,
            self.zoom_region,
            self.original_image.shape,
            self.display_image.shape,
        )

    def drag_rect_in_display_coords(self):
        """Return (dx0, dy0, dx1, dy1) for the in-progress ROI drag, or None."""
        if not self._is_drawing or self._draw_start is None or self._draw_end is None:
            return None
        return (self._draw_start[0], self._draw_start[1],
                self._draw_end[0], self._draw_end[1])

    def zoom_drag_in_display_coords(self):
        """Return (dx0, dy0, dx1, dy1) for the in-progress zoom drag, or None."""
        if not self._is_zooming or self._zoom_start is None or self._zoom_end is None:
            return None
        return (self._zoom_start[0], self._zoom_start[1],
                self._zoom_end[0], self._zoom_end[1])

    # ── Mouse event handlers (called from _RoiImageWidget) ────────────────

    def on_mouse_press(self, button, dx, dy):
        if button == Qt.MouseButton.LeftButton:
            self._is_drawing = True
            self._draw_start = (dx, dy)
            self._draw_end = (dx, dy)
            self.active_roi_bounds = None
            self.active_roi_source = "none"
            self.warning_message = ""
            self._reset_combo_to_none()
            self._update_labels()
        elif button == Qt.MouseButton.MiddleButton:
            self._is_zooming = True
            self._zoom_start = (dx, dy)
            self._zoom_end = (dx, dy)
        elif button == Qt.MouseButton.RightButton:
            if self.zoom_region is not None:
                self._action_zoom_out()
            else:
                self._action_clear()

    def on_mouse_move(self, dx, dy):
        if self._is_drawing:
            self._draw_end = (dx, dy)
        elif self._is_zooming:
            self._zoom_end = (dx, dy)

    def on_mouse_release(self, button, dx, dy):
        if button == Qt.MouseButton.LeftButton and self._is_drawing:
            self._is_drawing = False
            self._draw_end = (dx, dy)
            if self._draw_start is not None:
                orig_s = self.creator._map_display_point_to_original(
                    self._draw_start[0], self._draw_start[1],
                    self.zoom_region, self.original_image.shape, self.display_image.shape,
                )
                orig_e = self.creator._map_display_point_to_original(
                    self._draw_end[0], self._draw_end[1],
                    self.zoom_region, self.original_image.shape, self.display_image.shape,
                )
                roi_bounds, _, valid = self.creator._clamp_roi_to_shape(
                    [orig_s[0], orig_e[0]], [orig_s[1], orig_e[1]], self.original_image.shape,
                )
                if valid:
                    self.active_roi_bounds = roi_bounds
                    self.active_roi_source = "manual"
                    self.warning_message = ""
                else:
                    self.active_roi_bounds = None
                    self.active_roi_source = "none"
                self._update_labels()

        elif button == Qt.MouseButton.MiddleButton and self._is_zooming:
            self._is_zooming = False
            if self._zoom_start is not None and self._zoom_end is not None:
                orig_s = self.creator._map_display_point_to_original(
                    self._zoom_start[0], self._zoom_start[1],
                    self.zoom_region, self.original_image.shape, self.display_image.shape,
                )
                orig_e = self.creator._map_display_point_to_original(
                    self._zoom_end[0], self._zoom_end[1],
                    self.zoom_region, self.original_image.shape, self.display_image.shape,
                )
                x0, x1 = sorted([orig_s[0], orig_e[0]])
                y0, y1 = sorted([orig_s[1], orig_e[1]])
                if x1 > x0 and y1 > y0:
                    self.zoom_region = (x0, y0, x1, y1)
                    self.display_image = self.creator._extract_display_image(
                        self.original_image, self.zoom_region
                    )
                    self._refresh_image()
            self._zoom_start = self._zoom_end = None

    def on_wheel(self, delta):
        if self.zoom_region is not None:
            self._action_zoom_out()

    # ── Toolbar button actions ────────────────────────────────────────────

    def _adjust_colormap(self, make_brighter):
        step = 0.001 if self.creator.analysis_type == img_types.DISPERSIVE else 0.2
        max_joos = 0.0005 if self.creator.analysis_type == img_types.DISPERSIVE else 0.05
        if make_brighter:
            self.creator.cmap_juice_factor = max(
                self.creator.cmap_juice_factor - step, max_joos
            )
        else:
            self.creator.cmap_juice_factor = min(
                self.creator.cmap_juice_factor + step, 1.0
            )
        self._refresh_image()

    def _action_prev_image(self):
        self.img_index = (self.img_index - 1) % self.creator.N_img
        self._change_image()

    def _action_next_image(self):
        self.img_index = (self.img_index + 1) % self.creator.N_img
        self._change_image()

    def _action_zoom_out(self):
        if self.zoom_region is not None:
            self.zoom_region = None
            self.display_image = self.original_image.copy()
            self._refresh_image()

    def _action_clear(self):
        self._draw_start = self._draw_end = None
        self._is_drawing = False
        self._is_zooming = False
        self._zoom_start = self._zoom_end = None
        self.active_roi_bounds = None
        self.active_roi_source = "none"
        self.warning_message = ""
        self._reset_combo_to_none()
        self._update_labels()
        self.image_widget.update()

    def _action_brighter(self):
        self._adjust_colormap(make_brighter=True)

    def _action_dimmer(self):
        self._adjust_colormap(make_brighter=False)

    def _cancel_dialog(self):
        self.result_update_bool = False
        self.result_roix = np.array([-1, -1])
        self.result_roiy = np.array([-1, -1])
        self.reject()

    # ── Key events ────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._accept_roi()
            return
        if key == Qt.Key.Key_Escape:
            self._cancel_dialog()
            return

        if key == Qt.Key.Key_Right:
            self._action_next_image()
        elif key == Qt.Key.Key_Left:
            self._action_prev_image()

        elif key == Qt.Key.Key_Up:
            self._action_brighter()
        elif key == Qt.Key.Key_Down:
            self._action_dimmer()

    # ── Image switching ───────────────────────────────────────────────────

    def _change_image(self):
        self.original_image = self.creator.get_od(self.img_index).copy()
        self.display_image = self.creator._extract_display_image(
            self.original_image, self.zoom_region
        )
        if self.active_roi_source.startswith("preset:") and 0 <= self.preset_index < len(self.preset_entries):
            key, roix, roiy = self.preset_entries[self.preset_index]
            roi_bounds, was_clamped, valid, is_full_image_sentinel = self._resolve_preset_bounds(
                roix, roiy, self.original_image.shape
            )
            if valid:
                self.active_roi_bounds = roi_bounds
                if is_full_image_sentinel:
                    self.warning_message = f"Preset '{key}' uses whole image."
                else:
                    self.warning_message = f"Preset '{key}' clamped to image bounds." if was_clamped else ""
            else:
                self.active_roi_bounds = None
                self.warning_message = f"Preset '{key}' is invalid for this image size."
        self._refresh_image()

    # ── Accept / finalise ─────────────────────────────────────────────────

    def _accept_roi(self):
        if self.active_roi_bounds is not None:
            x0, x1, y0, y1 = self.active_roi_bounds
            self.result_update_bool = True
            self.result_roix = np.sort([x0, x1])
            self.result_roiy = np.sort([y0, y1])
        else:
            self.result_update_bool = False
            self.result_roix = np.array([-1, -1])
            self.result_roiy = np.array([-1, -1])
        self.accept()


# ─────────────────────────────────────────────────────────────────────────────
class roi_creator():
    window_name = 'recrop'

    def __init__(self, run_id, key, server_talk, file_path=None):

        self.key = key
        self.run_id = run_id
        self.server_talk = server_talk

        if file_path is None:
            filepath, _ = server_talk.get_data_file(run_id)
        else:
            filepath = file_path

        self.h5_file = h5py.File(filepath, 'r')
        self.images = self.h5_file['data']['images']
        self.N_img = self.images.shape[0] // 3
        try:
            self.analysis_type = self.h5_file['run_info']['imaging_type'][()]
        except Exception:
            self.analysis_type = img_types.ABSORPTION
        self._od_cache = {}

        self.image = self.get_od(0)

        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.end_x, self.end_y = -1, -1

    def _clip_point(self, x, y, shape):
        y_max, x_max = shape[:2]
        clipped_x = max(min(int(x), x_max - 1), 0)
        clipped_y = max(min(int(y), y_max - 1), 0)
        return clipped_x, clipped_y

    def _map_display_point_to_original(self, x, y, zoom_region, original_shape, display_shape):
        x, y = self._clip_point(x, y, display_shape)
        if zoom_region is None:
            return x, y

        display_h, display_w = display_shape[:2]
        x0, y0, x1, y1 = zoom_region
        zoom_w = max(x1 - x0, 1)
        zoom_h = max(y1 - y0, 1)

        mapped_x = x0 + int(x * zoom_w / display_w)
        mapped_y = y0 + int(y * zoom_h / display_h)
        return self._clip_point(mapped_x, mapped_y, original_shape)

    def _extract_display_image(self, original_image, zoom_region):
        if zoom_region is None:
            return original_image.copy()

        x0, y0, x1, y1 = zoom_region
        if x1 <= x0 or y1 <= y0:
            return original_image.copy()

        zoomed_region = original_image[y0:y1, x0:x1]
        if zoomed_region.size == 0:
            return original_image.copy()

        return cv2.resize(
            zoomed_region,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    def _colorize_image(self, image):
        max_pixel_value = float(np.max(image))
        if max_pixel_value <= 0.0:
            normalized_image = np.zeros_like(image, dtype=np.uint8)
        else:
            threshold = max(self.cmap_juice_factor * max_pixel_value, np.finfo(float).eps)
            normalized_image = np.clip(image, 0, threshold)
            normalized_image = cv2.normalize(normalized_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(normalized_image, cv2.COLORMAP_VIRIDIS)

    def get_od(self, idx):
        """Computes the idx'th OD for display in the ROI selection GUI.

        Args:
            idx (int): the index of the OD to display

        Returns:
            np.ndarray: the OD to display.
        """        
        if idx not in self._od_cache:
            pwa = self.images[3 * idx]
            pwoa = self.images[3 * idx + 1]
            dark = self.images[3 * idx + 2]
            self._od_cache[idx] = compute_OD(pwa, pwoa, dark, self.analysis_type)
        return self._od_cache[idx]

    def _clamp_roi_to_shape(self, roix, roiy, image_shape):
        height, width = image_shape[:2]
        raw_x0 = int(min(roix))
        raw_x1 = int(max(roix))
        raw_y0 = int(min(roiy))
        raw_y1 = int(max(roiy))

        x0 = max(0, min(raw_x0, width))
        x1 = max(0, min(raw_x1, width))
        y0 = max(0, min(raw_y0, height))
        y1 = max(0, min(raw_y1, height))

        was_clamped = (x0 != raw_x0) or (x1 != raw_x1) or (y0 != raw_y0) or (y1 != raw_y1)
        valid = (x1 > x0) and (y1 > y0)
        return (x0, x1, y0, y1), was_clamped, valid

    def _map_original_roi_to_display(self, roi_bounds, zoom_region, original_shape, display_shape):
        x0, x1, y0, y1 = roi_bounds
        display_h, display_w = display_shape[:2]

        if zoom_region is None:
            dx0 = max(0, min(int(x0), display_w - 1))
            dx1 = max(0, min(int(x1), display_w - 1))
            dy0 = max(0, min(int(y0), display_h - 1))
            dy1 = max(0, min(int(y1), display_h - 1))
            return dx0, dx1, dy0, dy1

        zx0, zy0, zx1, zy1 = zoom_region
        zoom_w = max(zx1 - zx0, 1)
        zoom_h = max(zy1 - zy0, 1)

        def map_x(x):
            return int((x - zx0) * display_w / zoom_w)

        def map_y(y):
            return int((y - zy0) * display_h / zoom_h)

        dx0 = max(0, min(map_x(x0), display_w - 1))
        dx1 = max(0, min(map_x(x1), display_w - 1))
        dy0 = max(0, min(map_y(y0), display_h - 1))
        dy1 = max(0, min(map_y(y1), display_h - 1))
        return dx0, dx1, dy0, dy1

    def _load_excel_roi_presets(self):
        presets = []
        try:
            roidf = _load_roi_excel_cached(self.server_talk.roi_csv_path)
            expected_cols = {'key', 'roix0', 'roix1', 'roiy0', 'roiy1'}
            if not expected_cols.issubset(set(roidf.columns)):
                return presets

            for _, row in roidf.iterrows():
                key = str(row['key']).strip() if not pd.isna(row['key']) else ""
                if key == "":
                    continue
                try:
                    roix = [int(row['roix0']), int(row['roix1'])]
                    roiy = [int(row['roiy0']), int(row['roiy1'])]
                except Exception:
                    continue
                presets.append((key, roix, roiy))
        except Exception:
            return []
        return presets

    def get_roi_rectangle(self):
        """Brings up the GUI to select an ROI over a display of the ODs from a
        given run.

        Controls:
            LMB + drag: Select an ROI rectangle.
            MMB + drag: Draw a dotted-line rectangle for zooming.
            MMB release: Zoom in to the selected region.
            RMB: Zoom out if zoomed; clear the drawn ROI at full scale.
            Mouse wheel scroll down: Zoom back out to full scale.
            L/R arrow keys: Scroll through ODs from the run while keeping zoom.
            Up / Down arrows: Adjust colormap brightness.
            Enter: Submit your selection.
            Escape / "X" button: Close the GUI without submitting selection.

        Returns:
            bool: Whether an ROI has been selected.
            tuple: roix, given as [roix0, roix1] (left and right bounds of the ROI).
            tuple: roiy, given as [roiy0, roiy1] (top and bottom bounds of the ROI).
        """
        if not _PYQT6_AVAILABLE:
            raise ImportError(
                "PyQt6 is required for the ROI selector. Install it with: pip install PyQt6"
            )
        self.cmap_juice_factor = 1.0
        preset_entries = self._load_excel_roi_presets()

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        parent_window = app.activeWindow()
        dialog = _RoiSelectorDialog(self, preset_entries, parent=parent_window)
        try:
            dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            # Show and activate the dialog before entering the modal loop.
            dialog.show()
            # Process pending events to ensure window is fully rendered.
            QCoreApplication.processEvents()
            dialog._focus_image_canvas()
            # Now enter modal exec with proper focus.
            dialog.exec()
        finally:
            self.h5_file.close()

        return dialog.result_update_bool, dialog.result_roix, dialog.result_roiy
