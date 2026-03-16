"""
viewer_nodes.py — 3D volume visualization nodes.

Provides:
  - SliceViewerNode      Embedded Z-slice browser with slider
  - Volume3DViewerNode   Interactive 3D isosurface renderer (Three.js)
"""
from __future__ import annotations

import json
import base64
import threading
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal

from nodes.base import BaseExecutionNode, PORT_COLORS, NodeBaseWidget
from .data_model import VolumeData, VolumeMaskData, VolumeLabelData

_VC = PORT_COLORS.get('volume', (220, 120, 50))
_MC = PORT_COLORS.get('volume_mask', (180, 90, 30))
_LC = PORT_COLORS.get('volume_label', (240, 180, 60))

_VIEWER_DIR = Path(__file__).parent / 'viewer'
_HTML_PATH = _VIEWER_DIR / 'volume_viewer.html'


# ══════════════════════════════════════════════════════════════════════════════
#  SliceViewerNode — embedded Z-slider + 2D slice display
# ══════════════════════════════════════════════════════════════════════════════

class _SliceViewerWidget(NodeBaseWidget):
    """Custom widget with Z-slider, axis selector, and image display."""

    _display_signal = Signal(object)   # thread-safe volume delivery
    _MAX_W = 520
    _MAX_H = 520

    def __init__(self, parent=None):
        super().__init__(parent, name='slice_viewer', label='')

        self._volume = None   # np.ndarray or None
        self._is_label = False

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(3)

        # Axis selector + Z label
        row = QtWidgets.QHBoxLayout()
        self._axis_combo = QtWidgets.QComboBox()
        self._axis_combo.addItems(['XY (Z-slice)', 'XZ (Y-slice)', 'YZ (X-slice)'])
        self._axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        self._z_label = QtWidgets.QLabel('Z: 0 / 0')
        row.addWidget(self._axis_combo)
        row.addStretch()
        row.addWidget(self._z_label)
        layout.addLayout(row)

        # Image display
        self._image_label = QtWidgets.QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            'background-color: #111; border: 1px solid #444;')
        self._image_label.setFixedSize(self._MAX_W, self._MAX_H)
        layout.addWidget(self._image_label)

        # Slider
        self._slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider)

        self.set_custom_widget(container)

        self._display_signal.connect(
            self._apply_volume, Qt.ConnectionType.QueuedConnection)

    def load_volume(self, volume: np.ndarray, is_label: bool = False):
        """Thread-safe volume load."""
        if threading.current_thread() is threading.main_thread():
            self._apply_volume((volume, is_label))
        else:
            self._display_signal.emit((volume, is_label))

    def _apply_volume(self, data):
        vol, is_label = data
        self._volume = vol
        self._is_label = is_label
        axis = self._axis_combo.currentIndex()
        n = vol.shape[axis]
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(n // 2)
        self._update_display()

    def _on_axis_changed(self, _idx):
        if self._volume is None:
            return
        axis = self._axis_combo.currentIndex()
        n = self._volume.shape[axis]
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(n // 2)
        self._update_display()

    def _on_slider(self, val):
        self._update_display()

    def _update_display(self):
        if self._volume is None:
            return
        axis = self._axis_combo.currentIndex()
        idx = self._slider.value()
        n = self._volume.shape[axis]
        axis_names = ['Z', 'Y', 'X']
        self._z_label.setText(f'{axis_names[axis]}: {idx} / {n - 1}')

        # Extract slice
        if axis == 0:
            slc = self._volume[idx, :, :]
        elif axis == 1:
            slc = self._volume[:, idx, :]
        else:
            slc = self._volume[:, :, idx]

        # Convert to PIL Image
        if self._is_label:
            pil = self._label_to_rgb(slc)
        elif slc.dtype == bool:
            pil = Image.fromarray((slc.astype(np.uint8) * 255), 'L')
        else:
            mn, mx = float(slc.min()), float(slc.max())
            if mx > mn:
                norm = ((slc.astype(np.float64) - mn) / (mx - mn) * 255
                        ).astype(np.uint8)
            else:
                norm = np.zeros(slc.shape, dtype=np.uint8)
            pil = Image.fromarray(norm, 'L')

        # Scale to fit display
        from PySide6.QtGui import QPixmap, QImage
        qimg = pil.convert('RGBA')
        data_bytes = qimg.tobytes('raw', 'RGBA')
        qi = QImage(data_bytes, qimg.width, qimg.height,
                    QImage.Format.Format_RGBA8888)
        pm = QPixmap.fromImage(qi)
        pm = pm.scaled(self._MAX_W, self._MAX_H,
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        self._image_label.setPixmap(pm)

    @staticmethod
    def _label_to_rgb(slc: np.ndarray) -> Image.Image:
        from nodes.vision_nodes import _label_palette
        labels = np.unique(slc)
        labels = labels[labels != 0]
        n = int(labels.max()) if len(labels) else 0
        palette = _label_palette(max(n, 1))
        rgb = np.zeros((*slc.shape, 3), dtype=np.uint8)
        for lbl in labels:
            rgb[slc == lbl] = palette[(int(lbl) - 1) % len(palette)]
        return Image.fromarray(rgb, 'RGB')

    def get_value(self):
        return None

    def set_value(self, _v):
        pass


class SliceViewerNode(BaseExecutionNode):
    """Interactive Z-slice browser for 3D volumes.

    Accepts volume, volume_mask, or volume_label input.  Use the slider
    to scrub through slices and the axis selector to view XY/XZ/YZ planes.

    Keywords: slice, viewer, browse, 3D, z-stack, preview, 切片, 檢視, 瀏覽, 體積
    """
    __identifier__ = 'nodes.Volume.Display'
    NODE_NAME      = '3D Slice Viewer'
    PORT_SPEC      = {'inputs': ['volume', 'volume_mask', 'volume_label'],
                      'outputs': []}

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('volume', color=_VC)
        self.add_input('volume_mask', color=_MC)
        self.add_input('volume_label', color=_LC)
        self._viewer = _SliceViewerWidget(self.view)
        self.add_custom_widget(self._viewer, tab='View')

    def evaluate(self):
        # Try each input in priority order
        for port_name, dtype, is_label in [
            ('volume_label', VolumeLabelData, True),
            ('volume_mask', VolumeMaskData, False),
            ('volume', VolumeData, False),
        ]:
            port = self.inputs().get(port_name)
            if port and port.connected_ports():
                cp = port.connected_ports()[0]
                data = cp.node().output_values.get(cp.name())
                if isinstance(data, dtype):
                    self._viewer.load_volume(
                        np.asarray(data.payload), is_label=is_label)
                    return True, None
        return False, "Connect a volume, volume_mask, or volume_label"


# ══════════════════════════════════════════════════════════════════════════════
#  Volume3DViewerNode — Three.js isosurface viewer
# ══════════════════════════════════════════════════════════════════════════════

class _Volume3DWidget(NodeBaseWidget):
    """QWebEngineView widget for Three.js 3D mesh rendering."""

    _DISPLAY_W = 560
    _DISPLAY_H = 500
    _mesh_signal = Signal(str)   # thread-safe JS delivery

    def __init__(self, parent=None):
        super().__init__(parent, name='viewer_3d_vol', label='')

        self._page_ready = False
        self._pending_js: list[str] = []

        container = QtWidgets.QWidget()
        container.setFixedSize(self._DISPLAY_W, self._DISPLAY_H)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        from PySide6.QtWebEngineWidgets import QWebEngineView
        self._web = QWebEngineView()
        self._web.setFixedSize(self._DISPLAY_W, self._DISPLAY_H)
        layout.addWidget(self._web)

        self.set_custom_widget(container)
        container.destroyed.connect(self._cleanup)
        self._mesh_signal.connect(
            self._run_js_main, Qt.ConnectionType.QueuedConnection)
        self._load_html()

    def _cleanup(self):
        if self._web is not None:
            self._web.setPage(None)
            self._web = None

    def _load_html(self):
        if not _HTML_PATH.exists():
            self._web.setHtml(
                '<html><body style="background:#1a1a2e;color:#aaa">'
                '<p>volume_viewer.html not found</p></body></html>')
            return
        html = _HTML_PATH.read_text(encoding='utf-8')
        base_url = QtCore.QUrl.fromLocalFile(str(_VIEWER_DIR) + '/')
        self._web.setHtml(html, base_url)
        self._web.loadFinished.connect(self._on_load)

    def _on_load(self, ok):
        self._page_ready = ok
        if ok and self._pending_js:
            for js in self._pending_js:
                self._web.page().runJavaScript(js)
            self._pending_js.clear()

    def run_js(self, js_code: str):
        """Execute JS — thread-safe."""
        if threading.current_thread() is threading.main_thread():
            self._run_js_main(js_code)
        else:
            self._mesh_signal.emit(js_code)

    def _run_js_main(self, js_code: str):
        if self._page_ready and self._web is not None:
            self._web.page().runJavaScript(js_code)
        else:
            self._pending_js.append(js_code)

    def get_value(self):
        return None

    def set_value(self, _v):
        pass


class Volume3DViewerNode(BaseExecutionNode):
    """Interactive 3D isosurface viewer for volume masks and label volumes.

    Extracts meshes via marching cubes and renders them with Three.js.
    For label volumes, each label gets a distinct colour.

    Keywords: 3D viewer, isosurface, mesh, render, volume, 3D檢視, 等值面, 體積
    """
    __identifier__ = 'nodes.Volume.Display'
    NODE_NAME      = '3D Volume Viewer'
    PORT_SPEC      = {'inputs': ['volume_mask', 'volume_label'], 'outputs': []}

    def __init__(self):
        super().__init__(use_progress=True)
        self.add_input('volume_mask', color=_MC)
        self.add_input('volume_label', color=_LC)
        self._add_float_spinbox('opacity', 'Opacity', value=0.85,
                                min_val=0.05, max_val=1.0, step=0.05)
        self.add_checkbox('wireframe', '', text='Wireframe', state=False)
        self._viewer = _Volume3DWidget(self.view)
        self.add_custom_widget(self._viewer, tab='View')

    def evaluate(self):
        self.reset_progress()
        from skimage.measure import marching_cubes

        opacity = float(self.get_property('opacity') or 0.85)
        wireframe = bool(self.get_property('wireframe'))

        # Check label input first
        label_port = self.inputs().get('volume_label')
        if label_port and label_port.connected_ports():
            cp = label_port.connected_ports()[0]
            data = cp.node().output_values.get(cp.name())
            if isinstance(data, VolumeLabelData):
                return self._render_labels(data, opacity, wireframe)

        # Fall back to mask input
        mask_port = self.inputs().get('volume_mask')
        if mask_port and mask_port.connected_ports():
            cp = mask_port.connected_ports()[0]
            data = cp.node().output_values.get(cp.name())
            if isinstance(data, VolumeMaskData):
                return self._render_mask(data, opacity, wireframe)

        return False, "Connect a volume_mask or volume_label"

    def _render_mask(self, data: VolumeMaskData, opacity, wireframe):
        from skimage.measure import marching_cubes

        vol = np.asarray(data.payload, dtype=np.float32)
        if not vol.any():
            self._viewer.run_js('clearScene();')
            return True, None

        self.set_progress(30)
        verts, faces, _, _ = marching_cubes(vol, level=0.5,
                                            spacing=data.spacing)
        self.set_progress(70)
        self._send_meshes([{
            'vertices': verts, 'faces': faces,
            'color': [0.3, 0.75, 0.9], 'opacity': opacity,
            'wireframe': wireframe,
        }])
        self.set_progress(100)
        return True, None

    def _render_labels(self, data: VolumeLabelData, opacity, wireframe):
        from skimage.measure import marching_cubes
        from nodes.vision_nodes import _label_palette

        label_arr = np.asarray(data.payload)
        labels = np.unique(label_arr)
        labels = labels[labels != 0]

        if len(labels) == 0:
            self._viewer.run_js('clearScene();')
            return True, None

        palette = _label_palette(len(labels))
        meshes = []
        for i, lbl in enumerate(labels):
            self.set_progress(int(20 + 60 * i / len(labels)))
            binary = (label_arr == lbl).astype(np.float32)
            if not binary.any():
                continue
            try:
                verts, faces, _, _ = marching_cubes(
                    binary, level=0.5, spacing=data.spacing)
            except Exception:
                continue
            c = palette[i % len(palette)]
            meshes.append({
                'vertices': verts, 'faces': faces,
                'color': [c[0] / 255, c[1] / 255, c[2] / 255],
                'opacity': opacity,
                'wireframe': wireframe,
            })

        self._send_meshes(meshes)
        self.set_progress(100)
        return True, None

    def _send_meshes(self, meshes: list[dict]):
        """Encode meshes and send to Three.js viewer."""
        js_meshes = []
        for m in meshes:
            v = np.asarray(m['vertices'], dtype=np.float32)
            f = np.asarray(m['faces'], dtype=np.int32)
            js_meshes.append({
                'vertices': base64.b64encode(v.tobytes()).decode('ascii'),
                'faces': base64.b64encode(f.tobytes()).decode('ascii'),
                'n_verts': len(v),
                'n_faces': len(f),
                'color': m['color'],
                'opacity': m.get('opacity', 0.85),
                'wireframe': m.get('wireframe', False),
            })
        payload = json.dumps(js_meshes)
        self._viewer.run_js(f'loadMeshes({payload});')
