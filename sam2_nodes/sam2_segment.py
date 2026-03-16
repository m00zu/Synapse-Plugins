"""
node.py — SAM2SegmentNode: interactive click-to-segment using SAM2 ONNX.

Click on objects in an image to generate segmentation masks.
Supports multiple objects, each with a distinct label.
"""
from __future__ import annotations

import hashlib
import logging
import threading

import numpy as np
from PIL import Image

from data_models import ImageData, MaskData, LabelData
from nodes.base import BaseExecutionNode, PORT_COLORS
from nodes.base import BaseImageProcessNode

from .engine import SAM2ImageSession
from .model_manager import SAM2ModelManager
from .sam2_widget import SAM2SegmentWidget
from .viewer import _obj_color

logger = logging.getLogger(__name__)

__all__ = ['SAM2SegmentNode']

_model_manager = SAM2ModelManager()


class SAM2SegmentNode(BaseImageProcessNode):
    """Interactive SAM2 segmentation — click to include/exclude regions.

    Connect an image, then click on objects to segment them.
    Use "+ Object" to add multiple objects, each with a distinct label.
    Toggle between Include (+foreground) and Exclude (−background) modes
    with the toolbar button.

    Outputs: binary mask (union of all objects), integer label image
    (each object = unique label), and a colored overlay.

    Keywords: SAM, SAM2, segment anything, click, interactive, segmentation, 分割, 點擊
    """

    __identifier__ = 'plugins.Plugins.Segmentation'
    NODE_NAME      = 'SAM2 Segment'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['mask', 'label_image', 'overlay']}

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress',
        'sam2_state', 'show_preview', 'live_preview',
        'auto_settings', 'auto_grid', 'auto_min_area', 'auto_max_area',
    })

    def __init__(self):
        super().__init__()

        # ── ports ────────────────────────────────────────────────────
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self.add_output('label_image', color=PORT_COLORS['label'])
        self.add_output('overlay', color=PORT_COLORS['image'])

        self.add_combo_menu(
            'model_variant', 'Model', items=['tiny', 'small', 'base_plus', 'large'])

        # Auto-segment settings (single compact row)
        self._add_row('auto_settings', 'Auto', [
            {'name': 'auto_grid', 'label': 'Grid', 'type': 'int',
             'value': 16, 'min_val': 4, 'max_val': 64, 'step': 4},
            {'name': 'auto_score', 'label': 'Score', 'type': 'float',
             'value': 0.85, 'min_val': 0.01, 'max_val': 1.0, 'step': 0.05, 'decimals': 2},
            {'name': 'auto_min_area', 'label': 'Min%', 'type': 'float',
             'value': 0.1, 'min_val': 0.0, 'max_val': 100.0, 'step': 0.1, 'decimals': 1},
            {'name': 'auto_max_area', 'label': 'Max%', 'type': 'float',
             'value': 10.0, 'min_val': 1.0, 'max_val': 100.0, 'step': 1.0, 'decimals': 1},
        ])

        # SAM2 interactive widget
        self._sam2_widget = SAM2SegmentWidget(self.view, name='sam2_state', label='')
        self.add_custom_widget(self._sam2_widget)
        self._sam2_widget.masks_updated.connect(self._on_masks_updated)

        # Session cache
        self._session: SAM2ImageSession | None = None
        self._current_variant: str | None = None
        self._image_hash: str | None = None
        self._session_lock = threading.Lock()

        self._in_batch = False

        # Pre-warm default model in background so it's ready when
        # the user connects an image (avoids freeze on first evaluate).
        threading.Thread(target=self._prewarm, daemon=True).start()

    def on_batch_start(self):
        """Freeze this node during batch — preserve reference annotations."""
        self._in_batch = True

    def on_batch_end(self):
        self._in_batch = False

    def set_property(self, name, value, push_undo=True):
        super().set_property(name, value, push_undo)
        if name == 'model_variant':
            self._reload_model(value)

    def _reload_model(self, variant: str):
        """Reload model + re-encode current image in background thread."""
        if self._sam2_widget._rgb_arr is None:
            return  # no image loaded yet, nothing to reload
        rgb_arr = self._sam2_widget._rgb_arr
        saved = self.get_property('sam2_state') or ''

        def _bg():
            try:
                self.reset_progress()
                self.set_progress(5)
                session = self._ensure_session(variant)
                self.set_progress(20)
                session.set_image(rgb_arr)
                self._image_hash = hashlib.md5(rgb_arr.tobytes()).hexdigest()
                self.set_progress(70)
                self._sam2_widget.setup_from_worker(rgb_arr, session, saved)
                self.set_progress(100)
                self._update_outputs(rgb_arr, session.masks)
                self.mark_clean()
            except Exception:
                logger.error("Model reload failed", exc_info=True)

        threading.Thread(target=_bg, daemon=True).start()

    def _prewarm(self):
        """Load the default model in background at node creation."""
        try:
            self._ensure_session('tiny')
        except Exception:
            logger.debug("SAM2 pre-warm failed (will retry on evaluate)", exc_info=True)

    def _ensure_session(self, variant: str) -> SAM2ImageSession:
        with self._session_lock:
            if self._session is not None and self._current_variant == variant:
                return self._session
            logger.info("Loading SAM2 model '%s' …", variant)
            # Release old session to free GPU/CPU memory
            old = self._session
            self._session = None
            del old
            enc_path, dec_path = _model_manager.get_model_paths(variant)
            self._session = SAM2ImageSession(str(enc_path), str(dec_path))
            self._current_variant = variant
            self._image_hash = None  # force re-encode with new model
            return self._session

    def evaluate(self):
        # During batch, preserve annotations — don't re-evaluate
        if self._in_batch:
            self.mark_clean()
            return True, None

        self.reset_progress()

        # ── get input image ──────────────────────────────────────────
        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            return False, "No image connected"
        up_node = in_port.connected_ports()[0].node()
        up_data = up_node.output_values.get(in_port.connected_ports()[0].name())
        if not isinstance(up_data, ImageData):
            return False, "Input must be ImageData"

        pil = up_data.payload
        if pil.mode != 'RGB':
            pil = pil.convert('RGB')
        rgb_arr = np.asarray(pil, dtype=np.uint8)

        # ── ensure model is loaded (heavy, runs on worker thread) ────
        variant = self.get_property('model_variant') or 'tiny'
        session = self._ensure_session(variant)

        # ── encode image only if content changed ──────────────────────
        img_hash = hashlib.md5(rgb_arr.tobytes()).hexdigest()
        image_changed = (img_hash != self._image_hash)
        if image_changed:
            self.set_progress(10)
            session.set_image(rgb_arr)
            self._image_hash = img_hash
            self.set_progress(50)

            # ── defer all widget ops to main thread ──────────────────
            saved = self.get_property('sam2_state') or ''
            self._sam2_widget.setup_from_worker(rgb_arr, session, saved)

        self.set_progress(100)
        self._update_outputs(rgb_arr, session.masks)
        self.mark_clean()
        return True, None

    def _on_masks_updated(self, all_masks: dict):
        """Called when the widget runs the decoder (interactive click or auto)."""
        if self._sam2_widget._rgb_arr is not None:
            self._update_outputs(self._sam2_widget._rgb_arr, all_masks)
            self.mark_clean()

    def _update_outputs(self, rgb_arr: np.ndarray,
                        all_masks: dict[int, np.ndarray] | None = None):
        """Build multi-object label_image + union mask + colored overlay."""
        h, w = rgb_arr.shape[:2]

        if not all_masks:
            all_masks = {}

        # ── build label image (each object = unique label) ───────────
        label_arr = np.zeros((h, w), dtype=np.int32)
        for obj_id, mask in sorted(all_masks.items()):
            while mask.ndim > 2:
                mask = mask[0]
            if mask.shape[0] != h or mask.shape[1] != w:
                continue
            label_arr[mask > 0] = obj_id

        # ── union mask (any object) ──────────────────────────────────
        union = (label_arr > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(union, mode='L')
        self.output_values['mask'] = MaskData(payload=mask_pil)

        # ── label visualization (colored regions on black) ────────────
        label_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for obj_id in sorted(all_masks.keys()):
            m = label_arr == obj_id
            if not np.any(m):
                continue
            label_rgb[m] = _obj_color(obj_id)
        label_pil = Image.fromarray(label_rgb, mode='RGB')
        self.output_values['label_image'] = LabelData(payload=label_arr, image=label_pil)

        # ── colored overlay (blended on original image) ───────────────
        vis = rgb_arr.astype(np.float32).copy()
        for obj_id in sorted(all_masks.keys()):
            m = label_arr == obj_id
            if not np.any(m):
                continue
            color = np.array(_obj_color(obj_id), dtype=np.float32)
            vis[m] = vis[m] * 0.5 + color * 0.5
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        self.output_values['overlay'] = ImageData(payload=Image.fromarray(vis, mode='RGB'))
