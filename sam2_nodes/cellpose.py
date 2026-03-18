"""
cellpose.py — Cellpose ONNX engine + segmentation nodes.

Combines the ONNX inference engine, single-image Cellpose Segment node,
and Cellpose Batch node for folder processing.

Keywords: Cellpose, segmentation, nuclei, cells, automatic,
          microscopy, fluorescence
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from data_models import ImageData, MaskData, LabelData, TableData
from nodes.base import PORT_COLORS
from nodes.base import BaseImageProcessNode
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

logger = logging.getLogger(__name__)

__all__ = ['CellposeONNX', 'CellposeSegmentNode', 'CellposeBatchNode']


class CellposeONNX:
    """Cellpose segmentation using ONNX Runtime (no PyTorch required)."""

    def __init__(self, model_path: str, gpu: bool = True):
        """
        Parameters
        ----------
        model_path : str
            Path to .onnx model file
        gpu : bool
            Use GPU if available
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime required. Install with: pip install onnxruntime")

        self.model_path = model_path
        self.gpu = gpu

        # Load session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, image: np.ndarray, diameter: int = 30,
                channels: tuple = (0, 0),
                cellprob_threshold: float = 0.0,
                niter: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run segmentation on image.

        Parameters
        ----------
        image : np.ndarray
            2D grayscale or 3D RGB image (H, W) or (H, W, C), uint8
        diameter : int
            Expected cell diameter in pixels
        channels : tuple
            (chan1, chan2) — e.g. (0,0) for grayscale
        cellprob_threshold : float
            Cell probability threshold (default 0.0)
        niter : int
            Number of dynamics iterations (default 200)

        Returns
        -------
        masks : np.ndarray
            Integer label image (H, W)
        flows : np.ndarray
            RGB flow visualization (H, W, 3) or empty
        """
        from PIL import Image as _PILImage

        orig_h, orig_w = image.shape[:2]

        # Build 3-channel RGB float32 image
        if image.ndim == 2:
            rgb = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            rgb = np.concatenate([image] * 3, axis=-1)
        else:
            rgb = image[:, :, :3]

        # Resize to 512x512 (model's expected input size)
        pil_img = _PILImage.fromarray(rgb.astype(np.uint8))
        pil_resized = pil_img.resize((512, 512), _PILImage.Resampling.LANCZOS)
        arr = np.asarray(pil_resized, dtype=np.float32)  # (512, 512, 3)

        # Normalize to [0, 1]
        arr = arr / 255.0

        # Transpose to (1, 3, 512, 512) — NCHW
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 512, 512)

        # img_size must be square — the ONNX model's internal upsampling
        # fails on non-square dimensions. Use max(H,W) then resize output.
        sq = max(orig_h, orig_w)

        # Build all required inputs
        feeds = {
            'img': arr.astype(np.float32),
            'img_size': np.array([sq, sq], dtype=np.int64),
            'channels': np.array(list(channels), dtype=np.int64),
            'diameter': np.array([diameter], dtype=np.int64),
            'cellprob_threshold': np.array([cellprob_threshold], dtype=np.float32),
            'niter': np.array([niter], dtype=np.int64),
        }

        # Run inference
        try:
            outputs = self.session.run(self.output_names, feeds)
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return (np.zeros((orig_h, orig_w), dtype=np.int32),
                    np.zeros((orig_h, orig_w, 3), dtype=np.uint8))

        # Output 0: mask (sq, sq) int64 — resize to original dimensions.
        # The input was stretched to 512×512 so the model output fills the
        # full square; use nearest-neighbor resize to preserve label IDs.
        raw_masks = outputs[0].astype(np.int32)
        if raw_masks.shape != (orig_h, orig_w):
            masks = np.asarray(_PILImage.fromarray(raw_masks).resize(
                (orig_w, orig_h), _PILImage.Resampling.NEAREST), dtype=np.int32)
        else:
            masks = raw_masks

        # Output 2: rgb_of_flows (sq, sq, 3) uint8 — resize similarly
        if len(outputs) > 2:
            raw_flows = outputs[2]
            if raw_flows.shape[:2] != (orig_h, orig_w):
                flows = np.asarray(_PILImage.fromarray(raw_flows).resize(
                    (orig_w, orig_h), _PILImage.Resampling.LANCZOS), dtype=np.uint8)
            else:
                flows = raw_flows
        else:
            flows = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        return masks, flows

    @staticmethod
    def download_model(model_name: str = 'cyto3', cache_dir: str | None = None) -> str:
        """
        Get pre-built ONNX model from bundled vendor or HuggingFace.

        Parameters
        ----------
        model_name : str
            Model name: 'nuclei', 'cyto', 'cyto2', 'cyto3'
        cache_dir : str, optional
            Directory to cache models (default: ~/.cellpose/models)

        Returns
        -------
        str
            Path to model file
        """
        # Check for bundled model in vendor directory
        try:
            vendor_dir = Path(__file__).resolve().parent / 'vendor' / 'cellpose_models'
            bundled_model = vendor_dir / f"{model_name}.onnx"
            if bundled_model.exists():
                logger.info(f"Using bundled model: {bundled_model}")
                return str(bundled_model)
        except Exception as e:
            logger.debug(f"Vendor path check failed: {e}")

        # Also try absolute path from sam2_nodes directory
        try:
            import sys
            sam2_dir = Path(sys.modules[__name__].__file__).parent if __name__ in sys.modules else None
            if sam2_dir:
                vendor_dir = sam2_dir / 'vendor' / 'cellpose_models'
                bundled_model = vendor_dir / f"{model_name}.onnx"
                if bundled_model.exists():
                    logger.info(f"Using bundled model: {bundled_model}")
                    return str(bundled_model)
        except Exception as e:
            logger.debug(f"Module path check failed: {e}")

        if cache_dir is None:
            cache_dir = str(Path.home() / '.cellpose' / 'models')
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Map model names to HuggingFace URLs (cyto3 confirmed working)
        model_urls = {
            'cyto3': 'https://huggingface.co/rectlabel/cellpose/resolve/main/cyto3.onnx251120.zip',
            'nuclei': 'https://huggingface.co/rectlabel/cellpose/resolve/main/nuclei.onnx251120.zip',
            'cyto': 'https://huggingface.co/rectlabel/cellpose/resolve/main/cyto.onnx251120.zip',
            'cyto2': 'https://huggingface.co/rectlabel/cellpose/resolve/main/cyto2.onnx251120.zip',
        }

        if model_name not in model_urls:
            # Fall back to cyto3 if model not found
            logger.warning(f"Model {model_name} not available, using cyto3 instead")
            model_name = 'cyto3'

        model_path = Path(cache_dir) / f"{model_name}.onnx"

        # Check if already cached
        if model_path.exists():
            logger.info(f"Using cached model: {model_path}")
            return str(model_path)

        # Download
        logger.info(f"Downloading {model_name} model...")
        import urllib.request
        import zipfile
        import tempfile

        url = model_urls[model_name]
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / f"{model_name}.zip"
            urllib.request.urlretrieve(url, str(zip_path))

            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmpdir)

            # Find .onnx file
            onnx_files = list(Path(tmpdir).glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"No .onnx file in downloaded {model_name}")

            # Copy to cache
            import shutil
            shutil.copy2(onnx_files[0], model_path)

        logger.info(f"Model cached at: {model_path}")
        return str(model_path)


# ═══════════════════════════════════════════════════════════════════════════
# Single-image Cellpose Segment Node
# ═══════════════════════════════════════════════════════════════════════════

class _CellposeSegmentWidget(NodeBaseWidget):
    """UI for single-image Cellpose segmentation."""

    _done_signal = Signal(object)  # result dict or None
    _progress_signal = Signal(str)  # status message

    def __init__(self, parent=None, name='', label=''):
        super().__init__(parent, name, label)

        self._image_data: ImageData | None = None
        self._result_masks: np.ndarray | None = None
        self._result_label_img: Image.Image | None = None
        self._result_overlay: Image.Image | None = None
        self._analyzing = False

        # ── UI ────────────────────────────────────────────────────────
        container = QtWidgets.QWidget()
        main_lay = QtWidgets.QVBoxLayout(container)
        main_lay.setContentsMargins(4, 2, 4, 2)
        main_lay.setSpacing(3)

        # ── Settings row ───────────────────────────────────────────
        r1 = QtWidgets.QHBoxLayout()
        r1.setSpacing(4)
        r1.setContentsMargins(0, 0, 0, 0)

        lbl_model = QtWidgets.QLabel("Model")
        lbl_model.setFixedWidth(35)
        lbl_model.setStyleSheet("font-size:10px;")
        r1.addWidget(lbl_model)
        self._combo_model = QtWidgets.QComboBox()
        self._combo_model.addItems(['cyto3'])
        self._combo_model.setCurrentText('cyto3')
        self._combo_model.setFixedWidth(70)
        self._combo_model.setStyleSheet("font-size:10px;")
        r1.addWidget(self._combo_model)

        lbl_chan = QtWidgets.QLabel("Chan")
        lbl_chan.setFixedWidth(30)
        lbl_chan.setStyleSheet("font-size:10px;")
        r1.addWidget(lbl_chan)
        self._spin_chan = QtWidgets.QSpinBox()
        self._spin_chan.setRange(0, 3)
        self._spin_chan.setValue(0)
        self._spin_chan.setFixedWidth(50)
        self._spin_chan.setStyleSheet("font-size:10px;")
        r1.addWidget(self._spin_chan)

        lbl_diam = QtWidgets.QLabel("Diam")
        lbl_diam.setFixedWidth(30)
        lbl_diam.setStyleSheet("font-size:10px;")
        r1.addWidget(lbl_diam)
        self._spin_diam = QtWidgets.QSpinBox()
        self._spin_diam.setRange(5, 200)
        self._spin_diam.setValue(30)
        self._spin_diam.setFixedWidth(50)
        self._spin_diam.setStyleSheet("font-size:10px;")
        r1.addWidget(self._spin_diam)

        r1.addStretch()
        main_lay.addLayout(r1)

        # ── Preview Canvas ────────────────────────────────────────
        self._preview_label = QtWidgets.QLabel("No image")
        self._preview_label.setMinimumHeight(100)
        self._preview_label.setMaximumHeight(400)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet(
            "border: 1px solid #333; background: #1a1a1a; color: #aaa; font-size: 9px;")
        main_lay.addWidget(self._preview_label)

        # ── Buttons ────────────────────────────────────────────────
        btn_lay = QtWidgets.QHBoxLayout()
        btn_lay.setSpacing(4)
        btn_lay.setContentsMargins(0, 0, 0, 0)

        self._btn_analyze = QtWidgets.QPushButton("▶  Segment")
        self._btn_analyze.setFixedHeight(24)
        self._btn_analyze.setStyleSheet(
            "QPushButton { background:#1a5c1a; color:white; font-weight:bold; "
            "border:1px solid #2a8a2a; border-radius:3px; padding:2px 8px; } "
            "QPushButton:hover { background:#258025; } "
            "QPushButton:disabled { background:#333; color:#777; }")
        btn_lay.addWidget(self._btn_analyze)

        self._status_label = QtWidgets.QLabel("Ready")
        self._status_label.setStyleSheet("color:#aaa; font-size:9px;")
        btn_lay.addWidget(self._status_label, 1)
        main_lay.addLayout(btn_lay)

        main_lay.addStretch()
        self._container = container
        self.set_custom_widget(self._container)

        # ── Connections ────────────────────────────────────────────
        self._btn_analyze.clicked.connect(self._on_analyze)
        self._done_signal.connect(
            self._on_done, Qt.ConnectionType.QueuedConnection)
        self._progress_signal.connect(
            self._on_progress, Qt.ConnectionType.QueuedConnection)

    def _update_node_height(self):
        """Update node height to fit content and redraw (deferred)."""
        # First pass: immediate adjustSize so Qt computes geometry
        if self.widget():
            self.widget().adjustSize()
        # Deferred pass: redraw node after layout settles
        QtCore.QTimer.singleShot(0, self._do_update_node_height)
        # Second deferred pass with small delay for stubborn cases
        QtCore.QTimer.singleShot(50, self._do_update_node_height)

    def _do_update_node_height(self):
        if not self.node:
            return
        if self.widget():
            self.widget().adjustSize()
        if hasattr(self.node, 'view') and hasattr(self.node.view, 'draw_node'):
            self.node.view.draw_node()

    def set_image(self, image_data: ImageData | None):
        """Set input image and enable/disable analyze button."""
        old = self._image_data
        self._image_data = image_data

        if image_data is not None:
            self._btn_analyze.setEnabled(True)
            # Clear stale results if input changed
            if old is not image_data:
                self._result_masks = None
                self._result_label_img = None
                self._result_overlay = None
                self._status_label.setText("Ready")
                self._show_image_preview(image_data.image)
        else:
            self._btn_analyze.setEnabled(False)
            self._result_masks = None
            self._result_label_img = None
            self._result_overlay = None
            self._status_label.setText("No image")
            self._preview_label.setText("No image")
            self._preview_label.setPixmap(QtGui.QPixmap())

    def _show_image_preview(self, pil_img):
        """Display image preview in canvas with aspect-ratio-aware scaling."""
        if isinstance(pil_img, np.ndarray):
            pil_img = Image.fromarray(pil_img if pil_img.dtype == np.uint8
                                      else (np.clip(pil_img, 0, 1) * 255).astype(np.uint8))
        else:
            pil_img = pil_img.copy()
        max_width, max_height = 560, 560
        pil_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        rgb = np.ascontiguousarray(pil_img.convert('RGB'))
        h, w = rgb.shape[:2]
        bpl = w * 3  # bytes per line
        qimg = QtGui.QImage(rgb.data, w, h, bpl,
                           QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self._preview_label.setPixmap(pixmap)
        self._current_pixmap = pixmap  # prevent GC

        self._preview_label.setFixedHeight(h + 4)
        self._preview_label.setFixedWidth(w + 4)
        self._update_node_height()

    def _on_analyze(self):
        if self._image_data is None:
            self._status_label.setText("No image loaded")
            return
        self._btn_analyze.setEnabled(False)
        self._status_label.setText("Analyzing…")

        # Capture widget values on the main thread to avoid QBasicTimer warnings
        params = {
            'model_name': self._combo_model.currentText() or 'cyto3',
            'channel': self._spin_chan.value(),
            'diameter': self._spin_diam.value(),
            'image': np.array(self._image_data.image),
            'orig_image': self._image_data.image,
        }

        t = threading.Thread(target=self._analyze_worker, args=(params,), daemon=True)
        t.start()

    def _analyze_worker(self, params: dict):
        """Background: run Cellpose ONNX on the single image."""
        try:
            model_name = params['model_name']
            channel = params['channel']
            diameter = params['diameter']

            img = params['image']

            logger.info(f"Loading Cellpose {model_name}…")
            model_path = CellposeONNX.download_model(model_name)
            try:
                model = CellposeONNX(model_path, gpu=True)
            except Exception as e:
                logger.warning(f"GPU unavailable, using CPU: {e}")
                model = CellposeONNX(model_path, gpu=False)

            # Handle multi-channel: select channel
            if img.ndim == 3:
                if channel < img.shape[2]:
                    img = img[:, :, channel]
                else:
                    img = img.mean(axis=2).astype(np.uint8)

            # Robust normalization to uint8
            img = img.astype(np.float32)
            p2, p98 = np.percentile(img, [2, 98])
            if p98 > p2:
                img = np.clip((img - p2) / (p98 - p2) * 255, 0, 255)
            elif img.max() > 0:
                img = (img / img.max() * 255)
            img = img.astype(np.uint8)

            # Run Cellpose ONNX
            logger.info("Running segmentation…")
            masks, _flows = model.predict(img, diameter=diameter)

            # Create label + overlay images
            label_img, overlay_img = self._create_label_and_overlay(
                params['orig_image'], masks)

            result = {
                'masks': masks,
                'label_img': label_img,
                'overlay': overlay_img,
            }
            self._done_signal.emit(result)
        except Exception:
            logger.exception("Cellpose analysis failed")
            self._done_signal.emit(None)

    @staticmethod
    def _label_colors(masks: np.ndarray):
        """Return (unique_labels, color_map) for label visualization."""
        from matplotlib import colormaps
        cmap = colormaps['tab20']

        unique_labels = np.unique(masks)
        unique_labels = unique_labels[unique_labels != 0]

        colors = {}
        for i, lbl in enumerate(unique_labels):
            colors[lbl] = (np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8)
        return unique_labels, colors

    def _create_label_and_overlay(self, orig_image, masks: np.ndarray):
        """Create both label image (colors on black) and overlay (blended on original).

        Returns (label_pil, overlay_arr) where overlay_arr is uint8 numpy RGB.
        """
        if isinstance(orig_image, Image.Image):
            orig_arr = np.array(orig_image.convert('RGB'))
        else:
            orig_arr = orig_image
            if orig_arr.ndim == 2:
                orig_arr = np.stack([orig_arr] * 3, axis=-1)
            if orig_arr.dtype != np.uint8:
                orig_arr = (np.clip(orig_arr, 0, 1) * 255).astype(np.uint8)
        h, w = masks.shape

        if orig_arr.shape[:2] != (h, w):
            orig_arr = np.array(Image.fromarray(orig_arr).resize((w, h)))

        unique_labels, colors = self._label_colors(masks)

        # Label image: solid colors on black
        label_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # Overlay: blended on original
        overlay = orig_arr.astype(np.float32).copy()

        for lbl in unique_labels:
            m = masks == lbl
            c = colors[lbl].astype(np.float32)
            label_rgb[m] = colors[lbl]
            overlay[m] = overlay[m] * 0.5 + c * 0.5

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(label_rgb, mode='RGB'), overlay

    def _on_progress(self, msg: str):
        self._status_label.setText(msg)

    def _on_done(self, result):
        self._btn_analyze.setEnabled(True)
        if result is None:
            self._status_label.setText("Failed (check cellpose installation)")
            self._result_masks = None
            self._result_label_img = None
            self._result_overlay = None
            return

        self._result_masks = result['masks']
        self._result_label_img = result['label_img']
        self._result_overlay = result['overlay']
        n_objs = len(np.unique(self._result_masks)) - 1
        self._status_label.setText(f"Done: {n_objs} objects")

        # Show segmentation result as preview
        self._show_overlay_preview(self._result_overlay)

        # Set outputs directly on node and mark downstream dirty
        # (don't mark_dirty on self — that would re-trigger evaluate
        # which overwrites the overlay preview)
        if self.node:
            masks = self._result_masks
            mask_arr = (masks > 0).astype(np.uint8) * 255
            self.node.output_values['mask'] = MaskData(payload=mask_arr)
            self.node.output_values['label_image'] = LabelData(
                payload=masks.astype(np.int32),
                image=self._result_label_img)
            self.node.output_values['overlay'] = ImageData(
                payload=self._result_overlay)
            # Mark only downstream nodes dirty (not self)
            for out_port in self.node.outputs().values():
                for in_port in out_port.connected_ports():
                    dn = in_port.node()
                    if hasattr(dn, 'mark_dirty'):
                        dn.mark_dirty()

    def _show_overlay_preview(self, pil_img):
        """Display segmentation overlay preview with aspect-ratio-aware scaling."""
        if isinstance(pil_img, np.ndarray):
            pil_img = Image.fromarray(pil_img if pil_img.dtype == np.uint8
                                      else (np.clip(pil_img, 0, 1) * 255).astype(np.uint8))
        else:
            pil_img = pil_img.copy()
        max_width, max_height = 560, 560
        pil_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        rgb = np.ascontiguousarray(pil_img.convert('RGB'))
        h, w = rgb.shape[:2]
        bpl = w * 3
        qimg = QtGui.QImage(rgb.data, w, h, bpl,
                           QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self._preview_label.setPixmap(pixmap)
        self._current_pixmap = pixmap  # prevent GC

        self._preview_label.setFixedHeight(h + 4)
        self._preview_label.setFixedWidth(w + 4)
        self._update_node_height()

    def get_value(self) -> str:
        return json.dumps({
            'model': self._combo_model.currentText(),
            'channel': self._spin_chan.value(),
            'diameter': self._spin_diam.value(),
        })

    def set_value(self, value):
        if not value:
            return
        try:
            d = json.loads(value) if isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError):
            return
        if d.get('model'):
            idx = self._combo_model.findText(d['model'])
            if idx >= 0:
                self._combo_model.setCurrentIndex(idx)
        if 'channel' in d:
            self._spin_chan.setValue(int(d['channel']))
        if 'diameter' in d:
            self._spin_diam.setValue(int(d['diameter']))


class CellposeSegmentNode(BaseImageProcessNode):
    """Automatic nuclei and cell segmentation using Cellpose on a single image.

    Connect an image, configure model/channel/diameter, and click "Segment"
    to automatically detect and measure objects.

    Outputs:
    - mask: Binary mask (foreground=1, background=0)
    - label_image: Integer label mask (each object = unique label ID)
    - overlay: Colored segmentation mask on original image

    Keywords: Cellpose, segmentation, nuclei, cells, automatic,
              microscopy, fluorescence, batch
    """

    __identifier__ = 'plugins.Plugins.Segmentation'
    NODE_NAME = 'Cellpose Segment'
    PORT_SPEC = {'inputs': ['image'], 'outputs': ['mask', 'label_image', 'overlay']}

    def __init__(self):
        super().__init__()

        # Inputs
        self.add_input('image', color=PORT_COLORS['image'])

        # Outputs
        self.add_output('mask', color=PORT_COLORS['mask'])
        self.add_output('label_image', color=PORT_COLORS['label'])
        self.add_output('overlay', color=PORT_COLORS['image'])

        # Main widget
        self._widget = _CellposeSegmentWidget(self.view)
        self._widget._node_ref = self
        self.add_custom_widget(self._widget)

        # Set node size
        self.width = 600

    def evaluate(self):
        self.reset_progress()

        # Get input image
        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            self._widget.set_image(None)
            return True, None

        up_node = in_port.connected_ports()[0].node()
        in_data = up_node.output_values.get(in_port.connected_ports()[0].name())
        if not isinstance(in_data, ImageData):
            return False, "Input must be ImageData"

        w = self._widget

        # Only update widget if input image actually changed
        if w._image_data is not in_data:
            w.set_image(in_data)

        # Populate outputs from cached results (set by _on_done)
        if w._result_masks is not None:
            mask_arr = (w._result_masks > 0).astype(np.uint8) * 255
            self.output_values['mask'] = MaskData(payload=mask_arr)

            label_arr = w._result_masks.astype(np.int32)
            self.output_values['label_image'] = LabelData(
                payload=label_arr,
                image=w._result_label_img)

            self.output_values['overlay'] = ImageData(
                payload=w._result_overlay)

            self.set_progress(100)

        return True, None

    def get_value(self):
        return self._widget.get_value()

    def set_value(self, value):
        self._widget.set_value(value)


# ═══════════════════════════════════════════════════════════════════════════
# Cellpose Batch Node (folder processing)
# ═══════════════════════════════════════════════════════════════════════════

def _nat_key(p: Path):
    """Natural sort key for filenames."""
    import re
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r'(\d+)', p.name)]


class _CellposeWidget(NodeBaseWidget):
    """UI for Cellpose batch segmentation parameters and status."""

    _done_signal = Signal(object)  # result dict or None

    _VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def __init__(self, parent=None, name='', label=''):
        super().__init__(parent, name, label)

        self._files: list[Path] = []
        self._video_path: str | None = None   # set when input is a video
        self._result_df: pd.DataFrame | None = None
        self._label_imgs: dict[int, np.ndarray] = {}
        self._all_frames: list[np.ndarray] = []
        self._current_preview_idx: int = 0
        self._analyzing = False

        # ── UI ────────────────────────────────────────────────────────
        container = QtWidgets.QWidget()
        container.setMinimumWidth(380)
        main_lay = QtWidgets.QVBoxLayout(container)
        main_lay.setContentsMargins(4, 2, 4, 2)
        main_lay.setSpacing(3)

        # ── Source selector (folder or video) ──────────────────────────
        r1 = QtWidgets.QHBoxLayout()
        r1.setSpacing(3); r1.setContentsMargins(0, 0, 0, 0)
        lbl_src = QtWidgets.QLabel("Source")
        lbl_src.setFixedWidth(40); lbl_src.setStyleSheet("font-size:10px;")
        r1.addWidget(lbl_src)
        self._le_folder = QtWidgets.QLineEdit()
        self._le_folder.setStyleSheet("font-size:10px; padding:1px 2px;")
        self._le_folder.setPlaceholderText("Image folder or video file…")
        r1.addWidget(self._le_folder, 1)
        self._btn_folder = QtWidgets.QPushButton("Dir")
        self._btn_folder.setFixedSize(30, 25)
        self._btn_folder.setStyleSheet("font-size:9px;")
        r1.addWidget(self._btn_folder)
        self._btn_video = QtWidgets.QPushButton("Vid")
        self._btn_video.setFixedSize(30, 25)
        self._btn_video.setStyleSheet("font-size:9px;")
        r1.addWidget(self._btn_video)
        main_lay.addLayout(r1)

        # ── Settings row ───────────────────────────────────────────
        r2 = QtWidgets.QHBoxLayout()
        r2.setSpacing(4); r2.setContentsMargins(0, 0, 0, 0)

        lbl_pat = QtWidgets.QLabel("Pat")
        lbl_pat.setFixedWidth(20); lbl_pat.setStyleSheet("font-size:10px;")
        r2.addWidget(lbl_pat)
        self._le_pattern = QtWidgets.QLineEdit("*.tif")
        self._le_pattern.setFixedWidth(50); self._le_pattern.setStyleSheet("font-size:10px;")
        r2.addWidget(self._le_pattern)

        lbl_model = QtWidgets.QLabel("Model")
        lbl_model.setFixedWidth(32); lbl_model.setStyleSheet("font-size:10px;")
        r2.addWidget(lbl_model)
        self._combo_model = QtWidgets.QComboBox()
        self._combo_model.addItems(['cyto3'])
        self._combo_model.setCurrentText('cyto3')
        self._combo_model.setFixedWidth(70); self._combo_model.setStyleSheet("font-size:10px;")
        r2.addWidget(self._combo_model)

        lbl_chan = QtWidgets.QLabel("Chan")
        lbl_chan.setFixedWidth(30); lbl_chan.setStyleSheet("font-size:10px;")
        r2.addWidget(lbl_chan)
        self._spin_chan = QtWidgets.QSpinBox()
        self._spin_chan.setRange(0, 3); self._spin_chan.setValue(0)
        self._spin_chan.setFixedWidth(50); self._spin_chan.setStyleSheet("font-size:10px;")
        r2.addWidget(self._spin_chan)

        lbl_diam = QtWidgets.QLabel("Diam")
        lbl_diam.setFixedWidth(30); lbl_diam.setStyleSheet("font-size:10px;")
        r2.addWidget(lbl_diam)
        self._spin_diam = QtWidgets.QSpinBox()
        self._spin_diam.setRange(5, 200); self._spin_diam.setValue(30)
        self._spin_diam.setFixedWidth(50); self._spin_diam.setStyleSheet("font-size:10px;")
        r2.addWidget(self._spin_diam)

        r2.addStretch()
        main_lay.addLayout(r2)

        # ── Preview Canvas ────────────────────────────────────────
        self._preview_label = QtWidgets.QLabel("No images")
        self._preview_label.setMinimumHeight(100)
        self._preview_label.setMaximumHeight(400)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet(
            "border: 1px solid #333; background: #1a1a1a; color: #aaa; font-size: 9px;")
        main_lay.addWidget(self._preview_label)

        # ── Frame slider (hidden until analysis done) ──────────────
        slider_lay = QtWidgets.QHBoxLayout()
        slider_lay.setSpacing(4); slider_lay.setContentsMargins(0, 0, 0, 0)
        self._frame_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.setFixedHeight(16)
        self._frame_slider.setVisible(False)
        slider_lay.addWidget(self._frame_slider)
        self._frame_label = QtWidgets.QLabel("Frame 1/1")
        self._frame_label.setFixedWidth(60)
        self._frame_label.setStyleSheet("font-size:9px;")
        self._frame_label.setVisible(False)
        slider_lay.addWidget(self._frame_label)
        main_lay.addLayout(slider_lay)

        # ── Buttons ────────────────────────────────────────────────
        btn_lay = QtWidgets.QHBoxLayout()
        btn_lay.setSpacing(4); btn_lay.setContentsMargins(0, 0, 0, 0)

        self._btn_analyze = QtWidgets.QPushButton("▶  Analyze")
        self._btn_analyze.setFixedHeight(24)
        self._btn_analyze.setStyleSheet(
            "QPushButton { background:#1a5c1a; color:white; font-weight:bold; "
            "border:1px solid #2a8a2a; border-radius:3px; padding:2px 8px; }"
            "QPushButton:hover { background:#258025; }"
            "QPushButton:disabled { background:#333; color:#777; }")
        btn_lay.addWidget(self._btn_analyze)

        self._status_label = QtWidgets.QLabel("Ready")
        self._status_label.setStyleSheet("color:#aaa; font-size:9px;")
        btn_lay.addWidget(self._status_label, 1)
        main_lay.addLayout(btn_lay)

        main_lay.addStretch()
        self._container = container
        self.set_custom_widget(self._container)

        # ── Connections ────────────────────────────────────────────
        self._btn_folder.clicked.connect(self._on_browse_folder)
        self._btn_video.clicked.connect(self._on_browse_video)
        self._le_folder.editingFinished.connect(self._on_source_edited)
        self._le_pattern.editingFinished.connect(self._on_source_edited)
        self._btn_analyze.clicked.connect(self._on_analyze)
        self._frame_slider.sliderMoved.connect(self._on_frame_slider_moved)
        self._done_signal.connect(self._on_done,
                                  Qt.ConnectionType.QueuedConnection)

    def _update_node_height(self):
        """Update node height to fit content and redraw (deferred)."""
        if self.widget():
            self.widget().adjustSize()
        QtCore.QTimer.singleShot(0, self._do_update_node_height)
        QtCore.QTimer.singleShot(50, self._do_update_node_height)

    def _do_update_node_height(self):
        if not self.node:
            return
        if self.widget():
            self.widget().adjustSize()
        if hasattr(self.node, 'view') and hasattr(self.node.view, 'draw_node'):
            self.node.view.draw_node()

    def _on_browse_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self._container, "Select Image Folder")
        if d:
            self._le_folder.setText(d)
            self._on_source_edited()

    def _on_browse_video(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._container, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All (*)")
        if f:
            self._le_folder.setText(f)
            self._on_source_edited()

    def _on_source_edited(self):
        source = self._le_folder.text().strip()
        self._files = []
        self._video_path = None

        if not source:
            self._status_label.setText("Ready")
            return

        # Check if source is a video file
        if os.path.isfile(source) and Path(source).suffix.lower() in self._VIDEO_EXTS:
            self._video_path = source
            try:
                from .video_utils import _get_reader
                reader = _get_reader(source)
                n_frames = reader.count_frames()
                reader.close()
                self._status_label.setText(f"Ready: video ({n_frames} frames)")
            except Exception as e:
                self._status_label.setText(f"Video error: {e}")
            return

        # Otherwise treat as folder
        if os.path.isdir(source):
            pattern = self._le_pattern.text().strip() or '*.tif'
            matches = sorted(Path(source).glob(pattern), key=_nat_key)
            self._files = [p for p in matches if p.is_file()]
        n = len(self._files)
        self._status_label.setText(f"Ready: {n} images")

    def _on_analyze(self):
        if not self._files and not self._video_path:
            self._status_label.setText("No images or video loaded")
            return
        self._btn_analyze.setEnabled(False)
        self._status_label.setText("Analyzing…")

        # Capture widget values on main thread to avoid QBasicTimer warnings
        params = {
            'model_name': self._combo_model.currentText() or 'cyto3',
            'channel': self._spin_chan.value(),
            'diameter': self._spin_diam.value(),
            'files': list(self._files),
            'video_path': self._video_path,
        }

        t = threading.Thread(target=self._analyze_worker, args=(params,), daemon=True)
        t.start()

    def _analyze_worker(self, params: dict):
        """Background: run Cellpose ONNX on all images (folder or video)."""
        try:
            model_name = params['model_name']
            channel = params['channel']
            diameter = params['diameter']
            files = params['files']
            video_path = params.get('video_path')

            logger.info(f"Loading Cellpose {model_name}…")
            model_path = CellposeONNX.download_model(model_name)
            try:
                model = CellposeONNX(model_path, gpu=True)
            except Exception as e:
                logger.warning(f"GPU unavailable, using CPU: {e}")
                model = CellposeONNX(model_path, gpu=False)

            from skimage.measure import regionprops

            # Build frame iterator: either folder images or video frames
            if video_path:
                frame_iter = self._iter_video_frames(video_path)
            else:
                frame_iter = self._iter_folder_frames(files)

            all_rows = []
            label_imgs = {}
            all_frames = []

            for fi, (orig_img, frame_name) in enumerate(frame_iter):
                try:
                    all_frames.append(orig_img)
                    img = orig_img.copy()

                    # Handle multi-channel: select channel
                    if img.ndim == 3:
                        if channel < img.shape[2]:
                            img = img[:, :, channel]
                        else:
                            img = img.mean(axis=2).astype(np.uint8)

                    # Robust percentile normalization to uint8
                    img = img.astype(np.float32)
                    p2, p98 = np.percentile(img, [2, 98])
                    if p98 > p2:
                        img = np.clip((img - p2) / (p98 - p2) * 255, 0, 255)
                    elif img.max() > 0:
                        img = (img / img.max() * 255)
                    img = img.astype(np.uint8)

                    # Run Cellpose ONNX
                    masks, _flows = model.predict(img, diameter=diameter)

                    # Measure each object
                    props = regionprops(masks.astype(np.int32), intensity_image=img)
                    for prop in props:
                        all_rows.append({
                            'label': prop.label,
                            'frame': fi + 1,
                            'file': frame_name,
                            'area': prop.area,
                            'perimeter': prop.perimeter,
                            'eccentricity': prop.eccentricity,
                            'solidity': prop.solidity,
                            'intensity_mean': prop.mean_intensity,
                            'centroid_y': prop.centroid[0],
                            'centroid_x': prop.centroid[1],
                        })

                    label_imgs[fi] = masks
                    logger.info(f"Frame {fi+1}: {len(props)} objects ({frame_name})")

                except Exception as e:
                    logger.error(f"Error processing frame {fi+1} ({frame_name}): {e}")
                    all_frames.append(None)

            self._done_signal.emit({
                'df': pd.DataFrame(all_rows) if all_rows else pd.DataFrame(),
                'label_imgs': label_imgs,
                'all_frames': all_frames,
            })
        except Exception:
            logger.exception("Cellpose analysis failed")
            self._done_signal.emit(None)

    @staticmethod
    def _iter_folder_frames(files):
        """Yield (numpy_array, filename) for each image file."""
        for fpath in files:
            yield np.array(Image.open(fpath)), fpath.name

    @staticmethod
    def _iter_video_frames(video_path: str):
        """Yield (numpy_array, frame_label) for each frame in a video."""
        from .video_utils import _get_reader
        reader = _get_reader(video_path)
        video_name = Path(video_path).stem
        for fi, frame in enumerate(reader):
            yield np.asarray(frame, dtype=np.uint8), f"{video_name}_f{fi+1:05d}"
        reader.close()

    def _on_frame_slider_moved(self, value: int):
        """Update preview when slider moved."""
        if not self._all_frames or value >= len(self._all_frames):
            return
        self._current_preview_idx = value
        self._update_preview()
        self._frame_label.setText(f"Frame {value+1}/{len(self._all_frames)}")

    def _update_preview(self):
        """Display current frame with mask overlay."""
        if not self._all_frames or self._current_preview_idx >= len(self._all_frames):
            return

        frame = self._all_frames[self._current_preview_idx]
        if frame is None:
            self._preview_label.setText("No frame")
            return

        # Convert frame to RGB for display
        if frame.ndim == 2:
            rgb = np.stack([frame] * 3, axis=2)
        elif frame.shape[2] >= 3:
            rgb = frame[:, :, :3].copy()
        else:
            rgb = frame.copy()

        # Add mask overlay if available
        masks = self._label_imgs.get(self._current_preview_idx)
        if masks is not None:
            from matplotlib import colormaps
            cmap = colormaps['tab20']

            overlay = rgb.astype(np.float32)
            unique_labels = np.unique(masks)
            unique_labels = unique_labels[unique_labels > 0]
            for i, label in enumerate(unique_labels):
                m = masks == label
                color = np.array(cmap(i % 20)[:3]) * 255
                overlay[m] = overlay[m] * 0.5 + color * 0.5
            rgb = np.clip(overlay, 0, 255).astype(np.uint8)

        # Display with aspect-ratio-aware scaling
        pil_img = Image.fromarray(rgb)
        max_width, max_height = 560, 560
        pil_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        rgb_arr = np.ascontiguousarray(pil_img.convert('RGB'))
        h, w = rgb_arr.shape[:2]
        bpl = w * 3
        qimg = QtGui.QImage(rgb_arr.data, w, h, bpl,
                           QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self._preview_label.setPixmap(pixmap)
        self._current_pixmap = pixmap  # prevent GC

        self._preview_label.setFixedHeight(h + 4)
        self._preview_label.setFixedWidth(w + 4)
        self._update_node_height()

    def _on_done(self, result):
        self._btn_analyze.setEnabled(True)
        if result is None:
            self._status_label.setText("Failed (check cellpose installation)")
            return

        self._result_df = result['df']
        self._label_imgs = result['label_imgs']
        self._all_frames = result.get('all_frames', [])
        n_frames = len(result['label_imgs'])
        n_objs = len(result['df']) if not result['df'].empty else 0
        self._status_label.setText(
            f"Done: {n_objs} objects across {n_frames} images")

        # Show preview slider and first frame
        if n_frames > 0:
            self._frame_slider.setMaximum(n_frames - 1)
            self._frame_slider.setValue(0)
            self._frame_slider.setVisible(True)
            self._frame_label.setVisible(True)
            self._current_preview_idx = 0
            self._update_preview()
            self._frame_label.setText(f"Frame 1/{n_frames}")

        # Set outputs directly, mark only downstream dirty
        if self.node:
            if not result['df'].empty:
                self.node.output_values['table'] = TableData(payload=result['df'])
            for out_port in self.node.outputs().values():
                for in_port in out_port.connected_ports():
                    dn = in_port.node()
                    if hasattr(dn, 'mark_dirty'):
                        dn.mark_dirty()

    def get_value(self) -> str:
        return json.dumps({
            'source': self._le_folder.text(),
            'pattern': self._le_pattern.text(),
            'model': self._combo_model.currentText(),
            'channel': self._spin_chan.value(),
            'diameter': self._spin_diam.value(),
        })

    def set_value(self, value):
        if not value:
            return
        try:
            d = json.loads(value) if isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError):
            return
        if d.get('source'):
            self._le_folder.setText(d['source'])
        elif d.get('folder'):
            self._le_folder.setText(d['folder'])
        if d.get('pattern'):
            self._le_pattern.setText(d['pattern'])
        if d.get('model'):
            idx = self._combo_model.findText(d['model'])
            if idx >= 0:
                self._combo_model.setCurrentIndex(idx)
        if 'channel' in d:
            self._spin_chan.setValue(int(d['channel']))
        if 'diameter' in d:
            self._spin_diam.setValue(int(d['diameter']))


class CellposeBatchNode(BaseImageProcessNode):
    """Automatic batch segmentation of an image folder using Cellpose.

    No manual annotation required — processes all images in a folder,
    runs Cellpose segmentation on each, and outputs regionprops measurements.

    Outputs:
    - table: regionprops measurements (area, eccentricity, solidity, intensity)

    Keywords: Cellpose, segmentation, nuclei, cells, automatic,
              microscopy, fluorescence, batch, folder
    """

    __identifier__ = 'plugins.Plugins.VideoAnalysis'
    NODE_NAME = 'Cellpose Batch'
    PORT_SPEC = {'inputs': [], 'outputs': ['table']}

    def __init__(self):
        super().__init__()

        # Outputs
        self.add_output('table', color=PORT_COLORS['table'])

        # Main widget
        self._widget = _CellposeWidget(self.view)
        self._widget._node_ref = self
        self.add_custom_widget(self._widget)

        # Set node size
        self.width = 600

    def evaluate(self):
        self.reset_progress()

        w = self._widget
        if w._result_df is not None and not w._result_df.empty:
            self.output_values['table'] = TableData(payload=w._result_df)
            self.set_progress(100)

        return True, None

    def get_value(self):
        return self._widget.get_value()

    def set_value(self, value):
        self._widget.set_value(value)
