"""
tracking.py — Object tracking: strategies, frame tracker, and SAM2TrackNode.

Combines the tracking strategy/engine (formerly tracker.py) with the
SAM2 Track node (formerly track_node.py).
"""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from data_models import ImageData, MaskData, LabelData
from nodes.base import PORT_COLORS
from nodes.base import BaseImageProcessNode

from .engine import SAM2ImageSession
from .model_manager import SAM2ModelManager
from .video_session import SAM2VideoSession, TrackingCancelled
from .viewer import _obj_color

logger = logging.getLogger(__name__)

__all__ = [
    'TrackingStrategy', 'CentroidTrackingStrategy', 'SAM2FrameTracker',
    'SAM2TrackNode',
]

_model_manager = SAM2ModelManager()


# ═══════════════════════════════════════════════════════════════════════════
# Appearance helpers
# ═══════════════════════════════════════════════════════════════════════════

def _mean_rgb(rgb_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute mean RGB of pixels under a mask. Returns shape (3,) float32."""
    pixels = rgb_arr[mask > 0]
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.float32)
    return pixels.mean(axis=0).astype(np.float32)


def _color_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised colour similarity in [0, 1]. 1 = identical colours."""
    diff = float(np.linalg.norm(a - b))
    return max(0.0, 1.0 - diff / 441.0)


# ═══════════════════════════════════════════════════════════════════════════
# Strategy ABC
# ═══════════════════════════════════════════════════════════════════════════

class TrackingStrategy(ABC):
    """Abstract base for generating prompts and verifying tracked masks."""

    @abstractmethod
    def generate_prompts(
        self, ref_masks: dict[int, np.ndarray], image_shape: tuple[int, int],
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Reference masks → {obj_id: (point_coords_Nx2, point_labels_N)}."""

    @abstractmethod
    def match_masks(
        self,
        ref_masks: dict[int, np.ndarray],
        new_masks: dict[int, np.ndarray],
        new_scores: dict[int, float],
        rgb_arr: np.ndarray | None = None,
        appearances: dict[int, np.ndarray] | None = None,
    ) -> dict[int, np.ndarray]:
        """Verify/filter predicted masks against references."""


# ═══════════════════════════════════════════════════════════════════════════
# Centroid + Appearance strategy
# ═══════════════════════════════════════════════════════════════════════════

class CentroidTrackingStrategy(TrackingStrategy):
    """Re-prompt with mask centroids, verify by IoU + colour similarity.

    Parameters
    ----------
    score_threshold : float
        Minimum decoder confidence to accept a mask.
    iou_threshold : float
        Minimum IoU with reference mask to accept a match.
    appearance_weight : float
        Blend weight for colour similarity (0 = IoU only, 1 = colour only).
    n_boundary_points : int
        Extra background points sampled near the mask boundary (0 = none).
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.2,
        appearance_weight: float = 0.3,
        n_boundary_points: int = 0,
    ):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.appearance_weight = max(0.0, min(1.0, appearance_weight))
        self.n_boundary_points = n_boundary_points

    def generate_prompts(
        self, ref_masks: dict[int, np.ndarray], image_shape: tuple[int, int],
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        prompts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        h, w = image_shape

        for obj_id, mask in ref_masks.items():
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                continue

            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            coords = [[x1, y1], [x2, y2], [cx, cy]]
            labels = [2, 3, 1]

            if self.n_boundary_points > 0:
                boundary = self._sample_boundary(mask, self.n_boundary_points)
                for bx, by in boundary:
                    dx, dy = bx - cx, by - cy
                    norm = max(1, (dx * dx + dy * dy) ** 0.5)
                    ox = int(bx + dx / norm * 3)
                    oy = int(by + dy / norm * 3)
                    ox = max(0, min(w - 1, ox))
                    oy = max(0, min(h - 1, oy))
                    if mask[oy, ox] == 0:
                        coords.append([ox, oy])
                        labels.append(0)

            prompts[obj_id] = (
                np.array(coords, dtype=np.int32),
                np.array(labels, dtype=np.int32),
            )
        return prompts

    def match_masks(
        self,
        ref_masks: dict[int, np.ndarray],
        new_masks: dict[int, np.ndarray],
        new_scores: dict[int, float],
        rgb_arr: np.ndarray | None = None,
        appearances: dict[int, np.ndarray] | None = None,
    ) -> dict[int, np.ndarray]:
        accepted: dict[int, np.ndarray] = {}
        w = self.appearance_weight

        for obj_id, new_mask in new_masks.items():
            score = new_scores.get(obj_id, 0.0)
            if score < self.score_threshold:
                logger.debug("Object %d rejected: score %.3f < %.3f",
                             obj_id, score, self.score_threshold)
                continue

            ref = ref_masks.get(obj_id)
            if ref is not None:
                iou = self._iou(ref, new_mask)

                if w > 0 and rgb_arr is not None and appearances and obj_id in appearances:
                    new_color = _mean_rgb(rgb_arr, new_mask)
                    ref_color = appearances[obj_id]
                    csim = _color_similarity(ref_color, new_color)
                    combined = (1.0 - w) * iou + w * csim
                else:
                    combined = iou

                if combined < self.iou_threshold:
                    logger.debug("Object %d rejected: combined %.3f "
                                 "(IoU %.3f) < %.3f",
                                 obj_id, combined, iou, self.iou_threshold)
                    continue

            accepted[obj_id] = new_mask
        return accepted

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ma, mb = a > 0, b > 0
        inter = int(np.count_nonzero(ma & mb))
        union = int(np.count_nonzero(ma | mb))
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _sample_boundary(mask: np.ndarray, n: int) -> list[tuple[int, int]]:
        from PIL import Image as _Img
        m = mask > 0
        eroded = np.array(
            _Img.fromarray(m.astype(np.uint8) * 255).resize(
                (mask.shape[1], mask.shape[0])),
            dtype=bool)
        boundary = m & ~eroded
        ys, xs = np.where(boundary)
        if len(ys) == 0:
            ys, xs = np.where(m)
        if len(ys) == 0:
            return []
        step = max(1, len(ys) // n)
        indices = list(range(0, len(ys), step))[:n]
        return [(int(xs[i]), int(ys[i])) for i in indices]


# ═══════════════════════════════════════════════════════════════════════════
# Frame tracker with dormant recovery
# ═══════════════════════════════════════════════════════════════════════════

class SAM2FrameTracker:
    """Stateful coordinator: tracks objects across frames using a strategy.

    Supports dormant tracking — when an object is lost, it enters a dormant
    state for up to ``max_lost_frames`` frames before being permanently removed.
    """

    def __init__(
        self,
        session: SAM2ImageSession,
        strategy: TrackingStrategy,
        max_lost_frames: int = 10,
    ):
        self._session = session
        self._strategy = strategy
        self._max_lost_frames = max(0, max_lost_frames)

        self._reference_masks: dict[int, np.ndarray] = {}
        self._appearances: dict[int, np.ndarray] = {}
        self._lost_objects: set[int] = set()
        self._dormant: dict[int, int] = {}
        self._dormant_masks: dict[int, np.ndarray] = {}

    def set_reference_masks(self, label_arr: np.ndarray,
                            rgb_arr: np.ndarray | None = None) -> None:
        self._reference_masks.clear()
        self._appearances.clear()
        self._lost_objects.clear()
        self._dormant.clear()
        self._dormant_masks.clear()

        for obj_id in np.unique(label_arr):
            if obj_id == 0:
                continue
            oid = int(obj_id)
            mask = (label_arr == obj_id).astype(np.uint8)
            self._reference_masks[oid] = mask
            if rgb_arr is not None:
                self._appearances[oid] = _mean_rgb(rgb_arr, mask)

        logger.info("Reference set: %d objects", len(self._reference_masks))

    def track_frame(
        self, rgb_arr: np.ndarray,
    ) -> tuple[dict[int, np.ndarray], dict[int, float]]:
        h, w = rgb_arr.shape[:2]

        self._session.set_image(rgb_arr)

        # 1. Track active objects
        active_refs = {
            oid: m for oid, m in self._reference_masks.items()
            if oid not in self._lost_objects and oid not in self._dormant
        }
        prompts = self._strategy.generate_prompts(active_refs, (h, w))

        raw_masks: dict[int, np.ndarray] = {}
        raw_scores: dict[int, float] = {}
        for obj_id, (coords, labels) in prompts.items():
            mask, scores = self._session.predict(coords, labels, label_id=obj_id)
            raw_masks[obj_id] = mask
            score_val = (float(np.max(np.asarray(scores)))
                         if np.asarray(scores).size > 0 else 0.0)
            raw_scores[obj_id] = score_val

        accepted = self._strategy.match_masks(
            self._reference_masks, raw_masks, raw_scores,
            rgb_arr=rgb_arr, appearances=self._appearances)

        # 2. Try to recover dormant objects
        if self._dormant and self._max_lost_frames > 0:
            dormant_refs = {
                oid: self._dormant_masks[oid]
                for oid in self._dormant if oid in self._dormant_masks
            }
            dormant_prompts = self._strategy.generate_prompts(dormant_refs, (h, w))
            for obj_id, (coords, labels) in dormant_prompts.items():
                mask, scores = self._session.predict(
                    coords, labels, label_id=obj_id)
                score_val = (float(np.max(np.asarray(scores)))
                             if np.asarray(scores).size > 0 else 0.0)

                candidate = {obj_id: mask}
                candidate_scores = {obj_id: score_val}
                recovered = self._strategy.match_masks(
                    {obj_id: self._dormant_masks[obj_id]},
                    candidate, candidate_scores,
                    rgb_arr=rgb_arr, appearances=self._appearances)
                if obj_id in recovered:
                    accepted[obj_id] = recovered[obj_id]
                    raw_scores[obj_id] = score_val
                    logger.info("Object %d recovered from dormant", obj_id)

        # 3. Update state
        newly_dormant = []
        for obj_id in list(self._reference_masks.keys()):
            if obj_id in accepted:
                self._reference_masks[obj_id] = accepted[obj_id]
                self._appearances[obj_id] = _mean_rgb(rgb_arr, accepted[obj_id])
                if obj_id in self._dormant:
                    del self._dormant[obj_id]
                    if obj_id in self._dormant_masks:
                        del self._dormant_masks[obj_id]
            elif obj_id in self._dormant:
                self._dormant[obj_id] -= 1
                if self._dormant[obj_id] <= 0:
                    self._lost_objects.add(obj_id)
                    del self._dormant[obj_id]
                    if obj_id in self._dormant_masks:
                        del self._dormant_masks[obj_id]
                    logger.info("Object %d permanently lost "
                                "(dormant expired)", obj_id)
            elif obj_id not in self._lost_objects and obj_id in prompts:
                if self._max_lost_frames > 0:
                    newly_dormant.append(obj_id)
                else:
                    self._lost_objects.add(obj_id)
                    logger.info("Object %d lost at this frame", obj_id)

        for obj_id in newly_dormant:
            self._dormant[obj_id] = self._max_lost_frames
            self._dormant_masks[obj_id] = self._reference_masks[obj_id].copy()
            logger.info("Object %d entered dormant (%d frames remaining)",
                        obj_id, self._max_lost_frames)

        accepted_scores = {oid: raw_scores.get(oid, 0.0) for oid in accepted}
        return accepted, accepted_scores

    @property
    def lost_objects(self) -> set[int]:
        return self._lost_objects

    @property
    def dormant_objects(self) -> dict[int, int]:
        return self._dormant

    @property
    def reference_masks(self) -> dict[int, np.ndarray]:
        return self._reference_masks

    def reset(self) -> None:
        self._reference_masks.clear()
        self._appearances.clear()
        self._lost_objects.clear()
        self._dormant.clear()
        self._dormant_masks.clear()


# ═══════════════════════════════════════════════════════════════════════════
# SAM2TrackNode
# ═══════════════════════════════════════════════════════════════════════════

class SAM2TrackNode(BaseImageProcessNode):
    """Track objects across timelapse frames using SAM2.

    Connect reference masks from SAM2 Segment (frame 1) and an image
    stream (e.g. VideoIterator or FolderIterator → ImageReader).

    Keywords: SAM2, track, timelapse, video, object tracking, 追蹤, 影片, 時間序列
    """

    __identifier__ = 'plugins.Plugins.VideoAnalysis'
    NODE_NAME      = 'SAM2 Track'
    PORT_SPEC      = {'inputs': ['image', 'label_image'],
                      'outputs': ['mask', 'label_image', 'overlay']}

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress',
        'show_preview', 'live_preview',
    })

    def __init__(self):
        super().__init__()

        self.add_input('image', color=PORT_COLORS['image'])
        self.add_input('label_image', color=PORT_COLORS['label'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self.add_output('label_image', color=PORT_COLORS['label'])
        self.add_output('overlay', color=PORT_COLORS['image'])

        self.add_combo_menu(
            'model_variant', 'Model',
            items=['tiny', 'small', 'base_plus', 'large'])
        self.add_combo_menu(
            'propagation_mode', 'Mode',
            items=['centroid', 'memory'])

        self._add_float_spinbox('score_threshold', 'Min Score',
                                value=0.5, min_val=0.01, max_val=1.0,
                                step=0.05, decimals=2)
        self._add_float_spinbox('iou_threshold', 'Min IoU',
                                value=0.2, min_val=0.0, max_val=1.0,
                                step=0.05, decimals=2)
        self._add_float_spinbox('max_lost_frames', 'Dormant Frames',
                                value=10.0, min_val=0.0, max_val=999.0,
                                step=1.0, decimals=0)
        self._add_float_spinbox('appearance_weight', 'Appearance Wt',
                                value=0.3, min_val=0.0, max_val=1.0,
                                step=0.05, decimals=2)

        self._session: SAM2ImageSession | None = None
        self._current_variant: str | None = None
        self._tracker: SAM2FrameTracker | None = None
        self._frame_count: int = 0
        self._initial_label_arr: np.ndarray | None = None
        self._session_lock = threading.Lock()

        self._video_session: SAM2VideoSession | None = None

        threading.Thread(target=self._prewarm, daemon=True).start()

    def _prewarm(self):
        try:
            self._ensure_session('tiny')
        except Exception:
            logger.debug("SAM2 Track pre-warm failed", exc_info=True)

    def _ensure_session(self, variant: str) -> SAM2ImageSession:
        with self._session_lock:
            if self._session is not None and self._current_variant == variant:
                return self._session
            logger.info("Loading SAM2 model '%s' for tracking …", variant)
            # Release old session to free GPU/CPU memory
            old = self._session
            self._session = None
            del old
            enc_path, dec_path = _model_manager.get_model_paths(variant)
            self._session = SAM2ImageSession(str(enc_path), str(dec_path))
            self._current_variant = variant
            return self._session

    def on_batch_start(self):
        self._frame_count = 0
        self._tracker = None
        self._initial_label_arr = None

        label_port = self.inputs().get('label_image')
        if label_port and label_port.connected_ports():
            connected = label_port.connected_ports()[0]
            up_node = connected.node()
            up_data = up_node.output_values.get(connected.name())
            if isinstance(up_data, LabelData) and up_data.payload is not None:
                self._initial_label_arr = np.asarray(up_data.payload, dtype=np.int32)
                logger.info("Track: reference masks loaded (%d objects)",
                            len(np.unique(self._initial_label_arr)) - 1)

    def on_batch_end(self):
        self._tracker = None
        self._frame_count = 0
        self._initial_label_arr = None
        if self._video_session is not None:
            self._video_session.reset()
            self._video_session = None

    def evaluate(self):
        self.reset_progress()

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

        self.set_progress(10)

        if self._initial_label_arr is None:
            label_port = self.inputs().get('label_image')
            if label_port and label_port.connected_ports():
                connected = label_port.connected_ports()[0]
                up_node_l = connected.node()
                up_data_l = up_node_l.output_values.get(connected.name())
                if isinstance(up_data_l, LabelData) and up_data_l.payload is not None:
                    self._initial_label_arr = np.asarray(up_data_l.payload, dtype=np.int32)

        if self._initial_label_arr is None:
            return False, "No reference label_image connected"

        self.set_progress(20)

        if self.cancel_requested:
            return False, "Cancelled"

        mode = self.get_property('propagation_mode') or 'centroid'
        variant = self.get_property('model_variant') or 'tiny'

        try:
            if mode == 'memory':
                all_masks = self._track_memory(rgb_arr, variant)
            else:
                all_masks = self._track_centroid(rgb_arr, variant)
        except TrackingCancelled:
            return False, "Cancelled"

        if self.cancel_requested:
            return False, "Cancelled"

        self.set_progress(80)

        self._update_outputs(rgb_arr, all_masks)
        self.set_progress(100)
        self.mark_clean()
        return True, None

    def _track_centroid(self, rgb_arr: np.ndarray,
                        variant: str) -> dict[int, np.ndarray]:
        session = self._ensure_session(variant)

        if self._tracker is None:
            score_thr = self.get_property('score_threshold') or 0.5
            iou_thr = self.get_property('iou_threshold') or 0.2
            max_lost = int(self.get_property('max_lost_frames') or 10)
            app_wt = self.get_property('appearance_weight') or 0.3
            strategy = CentroidTrackingStrategy(
                score_threshold=score_thr,
                iou_threshold=iou_thr,
                appearance_weight=app_wt,
            )
            self._tracker = SAM2FrameTracker(
                session, strategy, max_lost_frames=max_lost)
            self._tracker.set_reference_masks(
                self._initial_label_arr, rgb_arr=rgb_arr)

        self._frame_count += 1
        self.set_progress(30)
        all_masks, _ = self._tracker.track_frame(rgb_arr)
        return all_masks

    def _track_memory(self, rgb_arr: np.ndarray,
                      variant: str) -> dict[int, np.ndarray]:
        if self._video_session is None:
            logger.info("Loading SAM2 video models '%s' …", variant)
            paths = _model_manager.get_video_model_paths(variant)
            self._video_session = SAM2VideoSession(
                {k: str(v) for k, v in paths.items()},
                cancel_check=lambda: self.cancel_requested)

        self._frame_count += 1
        self.set_progress(30)

        if self._frame_count == 1:
            self._video_session.set_image(rgb_arr)
            all_masks: dict[int, np.ndarray] = {}
            label_arr = self._initial_label_arr
            for obj_id in np.unique(label_arr):
                if obj_id == 0:
                    continue
                oid = int(obj_id)
                mask = (label_arr == obj_id).astype(np.uint8)
                ys, xs = np.where(mask > 0)
                if len(ys) == 0:
                    continue
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                coords = np.array([[cx, cy]], dtype=np.int32)
                labels = np.array([1], dtype=np.int32)
                all_masks[oid] = self._video_session.initialize_object(
                    oid, coords, labels)
            return all_masks
        else:
            self._video_session.set_image(rgb_arr)
            masks, _ = self._video_session.propagate()
            return masks

    def _update_outputs(self, rgb_arr: np.ndarray,
                        all_masks: dict[int, np.ndarray] | None = None):
        h, w = rgb_arr.shape[:2]

        if not all_masks:
            all_masks = {}

        label_arr = np.zeros((h, w), dtype=np.int32)
        for obj_id, mask in sorted(all_masks.items()):
            while mask.ndim > 2:
                mask = mask[0]
            if mask.shape[0] != h or mask.shape[1] != w:
                continue
            label_arr[mask > 0] = obj_id

        union = (label_arr > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(union, mode='L')
        self.output_values['mask'] = MaskData(payload=mask_pil)

        label_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for obj_id in sorted(all_masks.keys()):
            m = label_arr == obj_id
            if not np.any(m):
                continue
            label_rgb[m] = _obj_color(obj_id)
        label_pil = Image.fromarray(label_rgb, mode='RGB')
        self.output_values['label_image'] = LabelData(payload=label_arr, image=label_pil)

        vis = rgb_arr.astype(np.float32).copy()
        for obj_id in sorted(all_masks.keys()):
            m = label_arr == obj_id
            if not np.any(m):
                continue
            color = np.array(_obj_color(obj_id), dtype=np.float32)
            vis[m] = vis[m] * 0.5 + color * 0.5
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        self.output_values['overlay'] = ImageData(payload=Image.fromarray(vis, mode='RGB'))
