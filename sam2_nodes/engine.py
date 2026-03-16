"""
engine.py — SAM2 ONNX inference engine (pure numpy/PIL, no cv2).

Adapted from ONNX-SAM2-Segment-Anything/sam2/sam2.py.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_ort = None  # lazy-loaded onnxruntime module


def _onnxruntime():
    """Lazy import of onnxruntime — avoids ~1-2s startup penalty."""
    global _ort
    if _ort is None:
        try:
            import onnxruntime
        except ImportError:
            raise ImportError(
                "onnxruntime is required for the SAM2 plugin.\n"
                "Install it with: pip install onnxruntime\n"
                "Or place an extracted onnxruntime wheel in "
                "plugins/sam2_nodes/vendor/")
        _ort = onnxruntime
    return _ort


def _get_providers() -> list[str]:
    """Return best available ONNX Runtime providers.

    CoreML is intentionally excluded — its model compilation dispatches
    work to the main thread via GCD, causing UI freezes even when the
    ONNX session is created on a background thread.  CPU provider on
    Apple Silicon already uses Accelerate/NEON and is fast enough.
    """
    ort = _onnxruntime()
    available = ort.get_available_providers()
    preferred = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [p for p in preferred if p in available] or available


class SAM2ImageEncoder:
    """Encode an RGB image into SAM2 embeddings."""

    def __init__(self, path: str) -> None:
        ort = _onnxruntime()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # suppress warnings (0=verbose..3=error..4=fatal)
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=_get_providers())
        model_inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in model_inputs]
        self.input_shape = model_inputs[0].shape        # (1, 3, H, W)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.encode(image)

    def encode(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode RGB uint8 array (H, W, 3) → (feat0, feat1, embed)."""
        input_tensor = self._prepare(image)
        t0 = time.perf_counter()
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor})
        dt = (time.perf_counter() - t0) * 1000
        logger.info("SAM2 encoder: %.1f ms", dt)
        return outputs[0], outputs[1], outputs[2]

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        """Resize + normalise → (1, 3, H, W) float32."""
        pil = Image.fromarray(image).resize(
            (self.input_width, self.input_height), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)  # (3, H, W)
        return arr[np.newaxis, ...]    # (1, 3, H, W)


class SAM2ImageDecoder:
    """Decode point prompts + embeddings → binary mask."""

    def __init__(self, path: str,
                 encoder_input_size: tuple[int, int],
                 orig_im_size: tuple[int, int],
                 mask_threshold: float = 0.0) -> None:
        ort = _onnxruntime()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=_get_providers())
        self.encoder_input_size = encoder_input_size
        self.orig_im_size = orig_im_size
        self.mask_threshold = mask_threshold
        self.scale_factor = 4
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, image_embed, high_res_feats_0, high_res_feats_1,
                 point_coords, point_labels):
        return self.predict(image_embed, high_res_feats_0, high_res_feats_1,
                            point_coords, point_labels)

    def predict(self, image_embed, high_res_feats_0, high_res_feats_1,
                point_coords, point_labels):
        """Run decoder → (mask_uint8, scores)."""
        inputs = self._prepare_inputs(
            image_embed, high_res_feats_0, high_res_feats_1,
            point_coords, point_labels)
        t0 = time.perf_counter()
        outputs = self.session.run(
            self.output_names,
            {self.input_names[i]: inputs[i] for i in range(len(self.input_names))})
        dt = (time.perf_counter() - t0) * 1000
        logger.info("SAM2 decoder: %.1f ms", dt)
        return self._process_output(outputs)

    def _prepare_inputs(self, image_embed, high_res_feats_0, high_res_feats_1,
                        point_coords, point_labels):
        coords = point_coords[np.newaxis, ...].copy()
        labels = point_labels[np.newaxis, ...]

        # Normalise coords from original image space → encoder input space
        coords[..., 0] = coords[..., 0] / self.orig_im_size[1] * self.encoder_input_size[1]
        coords[..., 1] = coords[..., 1] / self.orig_im_size[0] * self.encoder_input_size[0]

        num_labels = labels.shape[0]
        eh = self.encoder_input_size[0] // self.scale_factor
        ew = self.encoder_input_size[1] // self.scale_factor
        mask_input = np.zeros((num_labels, 1, eh, ew), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        original_size = np.array(
            [self.orig_im_size[0], self.orig_im_size[1]], dtype=np.int32)

        return (image_embed, high_res_feats_0, high_res_feats_1,
                coords.astype(np.float32), labels.astype(np.float32),
                mask_input, has_mask_input, original_size)

    def _process_output(self, outputs):
        scores = outputs[1].squeeze()
        masks = outputs[0]
        masks = (masks > self.mask_threshold).astype(np.uint8).squeeze()
        # Decoder may return multiple candidate masks (3, H, W);
        # select the one with the highest score.
        if masks.ndim == 3:
            best = int(np.argmax(scores))
            masks = masks[best]
            scores = scores[best]
        # Resize mask to original image dimensions if needed
        oh, ow = self.orig_im_size
        if masks.shape[0] != oh or masks.shape[1] != ow:
            masks = np.array(
                Image.fromarray(masks).resize((ow, oh), Image.NEAREST),
                dtype=np.uint8)
        return masks, scores

    def set_image_size(self, orig_im_size: tuple[int, int]):
        self.orig_im_size = orig_im_size


class SAM2ImageSession:
    """High-level session: set image once, add/remove points, get masks."""

    def __init__(self, encoder_path: str, decoder_path: str) -> None:
        self.encoder = SAM2ImageEncoder(encoder_path)
        self.decoder_path = decoder_path
        self._encoder_input_size = self.encoder.input_shape[2:]
        self._orig_im_size: tuple[int, int] = self._encoder_input_size
        self._embeddings: Optional[tuple] = None
        self._decoders: dict[int, SAM2ImageDecoder] = {}
        self._point_coords: dict[int, np.ndarray] = {}
        self._point_labels: dict[int, np.ndarray] = {}
        self._masks: dict[int, np.ndarray] = {}
        self._scores: dict[int, float] = {}

    def set_image(self, rgb_arr: np.ndarray) -> None:
        """Encode image (H, W, 3) uint8 RGB. Resets all points."""
        self._embeddings = self.encoder(rgb_arr)
        self._orig_im_size = (rgb_arr.shape[0], rgb_arr.shape[1])
        self.reset()

    def reset(self) -> None:
        """Clear all points and masks."""
        self._point_coords.clear()
        self._point_labels.clear()
        self._masks.clear()
        self._scores.clear()
        self._decoders.clear()

    def _get_decoder(self, label_id: int) -> SAM2ImageDecoder:
        if label_id not in self._decoders:
            self._decoders[label_id] = SAM2ImageDecoder(
                self.decoder_path, self._encoder_input_size,
                self._orig_im_size)
        return self._decoders[label_id]

    def predict(self, point_coords: np.ndarray,
                point_labels: np.ndarray,
                label_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Run decoder for given points → (mask_uint8_HW, scores).

        point_coords: (N, 2) int — (x, y) in original image space
        point_labels: (N,) int — 1=foreground, 0=background
        """
        if self._embeddings is None:
            raise RuntimeError("Call set_image() first")

        self._point_coords[label_id] = point_coords
        self._point_labels[label_id] = point_labels

        decoder = self._get_decoder(label_id)
        feat0, feat1, embed = self._embeddings

        if point_coords.size == 0:
            h, w = self._orig_im_size
            mask = np.zeros((h, w), dtype=np.uint8)
            scores = np.array([0.0])
        else:
            mask, scores = decoder(embed, feat0, feat1,
                                   point_coords, point_labels)

        self._masks[label_id] = mask
        self._scores[label_id] = float(np.max(np.asarray(scores))) if np.asarray(scores).size > 0 else 0.0
        return mask, scores

    def auto_segment(self, points_per_side: int = 16,
                     score_threshold: float = 0.85,
                     nms_iou_threshold: float = 0.5,
                     min_area_frac: float = 0.001,
                     max_area_frac: float = 0.5,
                     progress_cb=None) -> dict[int, np.ndarray]:
        """Automatic mask generation via a grid of single-point prompts.

        Returns {label_id: mask_uint8_HW} for surviving masks.
        *min_area_frac / max_area_frac*: fraction of total image pixels.
        *progress_cb(fraction)* is called periodically if provided.
        """
        if self._embeddings is None:
            raise RuntimeError("Call set_image() first")

        h, w = self._orig_im_size
        total_pixels = h * w
        min_px = int(total_pixels * min_area_frac)
        max_px = int(total_pixels * max_area_frac)
        feat0, feat1, embed = self._embeddings

        # Build a uniform grid of points (skip edges to avoid background)
        margin = max(1, min(h, w) // (points_per_side * 2))
        xs = np.linspace(margin, w - 1 - margin, points_per_side, dtype=np.int32)
        ys = np.linspace(margin, h - 1 - margin, points_per_side, dtype=np.int32)
        grid = np.array([(x, y) for y in ys for x in xs], dtype=np.int32)
        total = len(grid)

        # One shared decoder for all grid probes
        decoder = SAM2ImageDecoder(
            self.decoder_path, self._encoder_input_size, self._orig_im_size)

        candidates: list[tuple[float, np.ndarray]] = []  # (score, mask)
        for i, pt in enumerate(grid):
            coords = pt.reshape(1, 2)
            labels = np.array([1], dtype=np.int32)
            mask, scores = decoder(embed, feat0, feat1, coords, labels)
            best = float(np.max(np.asarray(scores)))
            if best < score_threshold:
                continue
            area = int(np.count_nonzero(mask > 0))
            if area < min_px or area > max_px:
                continue
            candidates.append((best, mask))
            if progress_cb and i % max(1, total // 20) == 0:
                progress_cb(i / total)

        if progress_cb:
            progress_cb(0.9)

        # NMS — keep higher-scoring mask when IoU > threshold
        candidates.sort(key=lambda c: c[0], reverse=True)
        keep: list[tuple[float, np.ndarray]] = []
        for score, mask in candidates:
            m = mask > 0
            suppress = False
            for _, kept_mask in keep:
                km = kept_mask > 0
                inter = np.count_nonzero(m & km)
                union = np.count_nonzero(m | km)
                if union > 0 and inter / union > nms_iou_threshold:
                    suppress = True
                    break
            if not suppress:
                keep.append((score, mask))

        # Store as labeled masks
        self.reset()
        result: dict[int, np.ndarray] = {}
        for idx, (score, mask) in enumerate(keep, start=1):
            self._masks[idx] = mask
            self._scores[idx] = score
            result[idx] = mask

        if progress_cb:
            progress_cb(1.0)
        logger.info("Auto-segment: %d objects from %d grid points "
                     "(%d candidates)", len(keep), total, len(candidates))
        return result

    def predict_box(self, box_xyxy: np.ndarray,
                    label_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Run decoder with a bounding box prompt.

        box_xyxy: (4,) array [x1, y1, x2, y2] in original image coords.
        Uses point_labels 2 (top-left) and 3 (bottom-right).
        """
        coords = np.array([[box_xyxy[0], box_xyxy[1]],
                            [box_xyxy[2], box_xyxy[3]]], dtype=np.int32)
        labels = np.array([2, 3], dtype=np.int32)
        return self.predict(coords, labels, label_id=label_id)

    @property
    def masks(self) -> dict[int, np.ndarray]:
        return self._masks

    @property
    def scores(self) -> dict[int, float]:
        return self._scores

    @property
    def orig_im_size(self) -> tuple[int, int]:
        return self._orig_im_size

    @property
    def is_image_set(self) -> bool:
        return self._embeddings is not None
