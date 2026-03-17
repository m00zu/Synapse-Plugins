"""
grounding.py — GroundingDINO engine + nodes (detection and text-prompted segmentation).

Combines the ONNX inference engine (formerly grounding_engine.py) with the
GroundingDINO and SAM2 Text Segment nodes.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw

from data_models import ImageData, MaskData, LabelData
from nodes.base import PORT_COLORS
from nodes.base import BaseImageProcessNode

from .engine import SAM2ImageSession
from .model_manager import SAM2ModelManager
from .viewer import _obj_color

logger = logging.getLogger(__name__)

__all__ = [
    'Detection', 'GroundingDINOSession',
    'GroundingDINONode', 'SAM2TextSegmentNode',
]

_ort = None


def _onnxruntime():
    global _ort
    if _ort is None:
        import onnxruntime
        _ort = onnxruntime
    return _ort


def _get_providers() -> list[str]:
    ort = _onnxruntime()
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or available


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boxes in xyxy format."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# GroundingDINO ONNX Session
# ═══════════════════════════════════════════════════════════════════════════

class Detection(NamedTuple):
    box_xyxy: np.ndarray   # (4,) float32 [x1, y1, x2, y2]
    score: float
    label: str


class GroundingDINOSession:
    """GroundingDINO Tiny ONNX inference session.

    Accepts an RGB image + text prompt, returns detected bounding boxes.
    """

    INPUT_SIZE = 800
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        ort = _onnxruntime()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=_get_providers())
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        logger.info("GroundingDINO session loaded (inputs: %s)",
                     self.input_names)

    def detect(self, rgb_arr: np.ndarray, text_prompt: str,
               score_threshold: float = 0.25,
               nms_iou_threshold: float = 0.5) -> list[Detection]:
        orig_h, orig_w = rgb_arr.shape[:2]

        text_prompt = text_prompt.replace(',', ' .')

        pixel_values, pixel_mask, scale_x, scale_y = self._preprocess_image(
            rgb_arr)
        input_ids, token_type_ids, attention_mask, token_spans = \
            self._tokenize(text_prompt)

        feeds = {}
        for name in self.input_names:
            if name == 'pixel_values':
                feeds[name] = pixel_values
            elif name == 'input_ids':
                feeds[name] = input_ids
            elif name == 'token_type_ids':
                feeds[name] = token_type_ids
            elif name == 'attention_mask':
                feeds[name] = attention_mask
            elif name == 'pixel_mask':
                feeds[name] = pixel_mask

        t0 = time.perf_counter()
        outputs = self.session.run(self.output_names, feeds)
        dt = (time.perf_counter() - t0) * 1000
        logger.info("GroundingDINO inference: %.1f ms", dt)

        logits = outputs[0]
        boxes = outputs[1]

        detections = self._postprocess(
            logits[0], boxes[0], token_spans, text_prompt,
            orig_w, orig_h, score_threshold)

        if len(detections) > 1:
            detections = self._nms(detections, nms_iou_threshold)

        return detections

    def _preprocess_image(self, rgb_arr: np.ndarray):
        orig_h, orig_w = rgb_arr.shape[:2]

        scale = min(self.INPUT_SIZE / orig_h, self.INPUT_SIZE / orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        pil = Image.fromarray(rgb_arr).resize((new_w, new_h), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - self.MEAN) / self.STD

        padded = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3),
                          dtype=np.float32)
        padded[:new_h, :new_w, :] = arr

        pixel_mask = np.zeros((1, self.INPUT_SIZE, self.INPUT_SIZE),
                              dtype=np.int64)
        pixel_mask[0, :new_h, :new_w] = 1

        pixel_values = padded.transpose(2, 0, 1)[np.newaxis, ...]

        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        return pixel_values, pixel_mask, scale_x, scale_y

    def _tokenize(self, text_prompt: str):
        text = text_prompt.strip()
        if not text.endswith('.'):
            text = text + ' .'

        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        seq_len = len(ids)

        input_ids = np.array([ids], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)
        attention_mask = np.ones_like(input_ids)

        dot_id = 1012
        spans = []
        start = 1
        for i in range(1, seq_len):
            if ids[i] in (dot_id, 102):
                if i > start:
                    spans.append((start, i))
                start = i + 1

        return input_ids, token_type_ids, attention_mask, spans

    def _postprocess(self, logits: np.ndarray, boxes: np.ndarray,
                     token_spans: list, text_prompt: str,
                     orig_w: int, orig_h: int,
                     score_threshold: float) -> list[Detection]:
        probs = 1.0 / (1.0 + np.exp(-logits))

        categories = [c.strip()
                      for c in text_prompt.replace(',', '.').replace('.', '\n').split('\n')
                      if c.strip()]

        detections: list[Detection] = []

        for cat_idx, span in enumerate(token_spans):
            if cat_idx >= len(categories):
                break
            start, end = span
            span_probs = probs[:, start:end]
            scores = span_probs.max(axis=1)

            mask = scores > score_threshold
            if not np.any(mask):
                continue

            kept_scores = scores[mask]
            kept_boxes = boxes[mask]

            for i in range(len(kept_scores)):
                cx, cy, w, h = kept_boxes[i]
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))

                box = np.array([x1, y1, x2, y2], dtype=np.float32)
                detections.append(Detection(
                    box_xyxy=box,
                    score=float(kept_scores[i]),
                    label=categories[cat_idx],
                ))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    @staticmethod
    def _nms(detections: list[Detection],
             iou_threshold: float) -> list[Detection]:
        if not detections:
            return detections

        keep: list[Detection] = []
        for det in detections:
            suppress = False
            for kept in keep:
                iou = _box_iou(det.box_xyxy, kept.box_xyxy)
                if iou > iou_threshold:
                    suppress = True
                    break
            if not suppress:
                keep.append(det)
        return keep


# ═══════════════════════════════════════════════════════════════════════════
# Shared model manager instance
# ═══════════════════════════════════════════════════════════════════════════

_model_manager = SAM2ModelManager()


# ═══════════════════════════════════════════════════════════════════════════
# GroundingDINONode
# ═══════════════════════════════════════════════════════════════════════════

class GroundingDINONode(BaseImageProcessNode):
    """Detect objects by text description using GroundingDINO.

    Type a text query (e.g. "nucleus" or "cell, membrane") and get
    bounding-box detections as rectangular masks.

    Keywords: GroundingDINO, text, detect, grounding, open vocabulary, 文字, 偵測, 開放詞彙
    """

    __identifier__ = 'plugins.Plugins.Segmentation'
    NODE_NAME      = 'Grounding DINO'
    PORT_SPEC      = {'inputs': ['image'],
                      'outputs': ['mask', 'label_image', 'overlay']}

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress',
        'show_preview', 'live_preview',
    })

    def __init__(self):
        super().__init__()

        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self.add_output('label_image', color=PORT_COLORS['label'])
        self.add_output('overlay', color=PORT_COLORS['image'])

        self.add_text_input('text_prompt', 'Text', text='')
        self._add_float_spinbox('score_threshold', 'Min Score',
                                value=0.7, min_val=0.01, max_val=1.0,
                                step=0.05, decimals=2)

        self._gdino: GroundingDINOSession | None = None
        self._lock = threading.Lock()

    def _ensure_gdino(self) -> GroundingDINOSession:
        with self._lock:
            if self._gdino is not None:
                return self._gdino
            model_path = _model_manager.get_gdino_model_path()
            tok_path = _model_manager.get_gdino_tokenizer_path()
            self._gdino = GroundingDINOSession(str(model_path), str(tok_path))
            return self._gdino

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

        text = self.get_property('text_prompt') or ''
        if not text.strip():
            return False, "No text prompt specified"

        threshold = self.get_property('score_threshold') or 0.3

        self.set_progress(10)
        if self.cancel_requested:
            return False, "Cancelled"

        gdino = self._ensure_gdino()
        self.set_progress(20)

        detections = gdino.detect(rgb_arr, text, score_threshold=threshold)
        self.set_progress(60)

        if self.cancel_requested:
            return False, "Cancelled"

        h, w = rgb_arr.shape[:2]
        self._build_outputs(rgb_arr, detections, h, w)

        self.set_progress(100)
        self.mark_clean()
        logger.info("GroundingDINO: %d detections for '%s'",
                     len(detections), text)
        return True, None

    def _build_outputs(self, rgb_arr: np.ndarray,
                       detections: list, h: int, w: int):
        categories = []
        for d in detections:
            if d.label not in categories:
                categories.append(d.label)
        detections = sorted(detections,
                            key=lambda d: (categories.index(d.label), -d.score))

        label_arr = np.zeros((h, w), dtype=np.int32)

        for i, det in enumerate(detections):
            obj_id = i + 1
            x1, y1, x2, y2 = det.box_xyxy.astype(int)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            label_arr[y1:y2, x1:x2] = obj_id

        union = (label_arr > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(union, mode='L')
        self.output_values['mask'] = MaskData(payload=mask_pil)

        label_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(len(detections)):
            obj_id = i + 1
            m = label_arr == obj_id
            if np.any(m):
                label_rgb[m] = _obj_color(obj_id)
        label_pil = Image.fromarray(label_rgb, mode='RGB')
        self.output_values['label_image'] = LabelData(
            payload=label_arr, image=label_pil)

        overlay_pil = Image.fromarray(rgb_arr, mode='RGB').copy()
        draw = ImageDraw.Draw(overlay_pil)
        for i, det in enumerate(detections):
            obj_id = i + 1
            color = tuple(_obj_color(obj_id))
            x1, y1, x2, y2 = det.box_xyxy.tolist()
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label_text = f"{det.label} {det.score:.2f}"
            draw.text((x1 + 2, max(0, y1 - 12)), label_text, fill=color)

        self.output_values['overlay'] = ImageData(payload=overlay_pil)


# ═══════════════════════════════════════════════════════════════════════════
# SAM2TextSegmentNode
# ═══════════════════════════════════════════════════════════════════════════

class SAM2TextSegmentNode(BaseImageProcessNode):
    """Text-prompted segmentation using GroundingDINO + SAM2.

    Type a text description to detect and segment objects.
    Chains GroundingDINO (text → boxes) with SAM2 (boxes → precise masks).

    Keywords: SAM2, text, segment, grounding, GroundingDINO, open vocabulary, 文字, 分割, 語意
    """

    __identifier__ = 'plugins.Plugins.Segmentation'
    NODE_NAME      = 'SAM2 Text Segment'
    PORT_SPEC      = {'inputs': ['image'],
                      'outputs': ['mask', 'label_image', 'overlay']}

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress',
        'show_preview', 'live_preview',
    })

    def __init__(self):
        super().__init__()

        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self.add_output('label_image', color=PORT_COLORS['label'])
        self.add_output('overlay', color=PORT_COLORS['image'])

        self.add_text_input('text_prompt', 'Text', text='')
        self._add_float_spinbox('score_threshold', 'Min Score',
                                value=0.7, min_val=0.01, max_val=1.0,
                                step=0.05, decimals=2)
        self.add_combo_menu(
            'model_variant', 'SAM2 Model',
            items=['tiny', 'small', 'base_plus', 'large'])

        self._gdino: GroundingDINOSession | None = None
        self._sam2: SAM2ImageSession | None = None
        self._sam2_variant: str | None = None
        self._lock = threading.Lock()

    def _ensure_gdino(self) -> GroundingDINOSession:
        with self._lock:
            if self._gdino is not None:
                return self._gdino
            model_path = _model_manager.get_gdino_model_path()
            tok_path = _model_manager.get_gdino_tokenizer_path()
            self._gdino = GroundingDINOSession(str(model_path), str(tok_path))
            return self._gdino

    def _ensure_sam2(self, variant: str) -> SAM2ImageSession:
        with self._lock:
            if self._sam2 is not None and self._sam2_variant == variant:
                return self._sam2
            enc_path, dec_path = _model_manager.get_model_paths(variant)
            self._sam2 = SAM2ImageSession(str(enc_path), str(dec_path))
            self._sam2_variant = variant
            return self._sam2

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

        text = self.get_property('text_prompt') or ''
        if not text.strip():
            return False, "No text prompt specified"

        threshold = self.get_property('score_threshold') or 0.7
        variant = self.get_property('model_variant') or 'tiny'

        self.set_progress(10)
        if self.cancel_requested:
            return False, "Cancelled"

        gdino = self._ensure_gdino()
        self.set_progress(20)

        detections = gdino.detect(rgb_arr, text, score_threshold=threshold)
        self.set_progress(40)

        if self.cancel_requested:
            return False, "Cancelled"

        categories = []
        for d in detections:
            if d.label not in categories:
                categories.append(d.label)
        detections = sorted(detections,
                            key=lambda d: (categories.index(d.label), -d.score))

        if not detections:
            h, w = rgb_arr.shape[:2]
            self._empty_outputs(rgb_arr, h, w)
            self.set_progress(100)
            self.mark_clean()
            return True, None

        sam2 = self._ensure_sam2(variant)
        sam2.set_image(rgb_arr)
        self.set_progress(60)

        if self.cancel_requested:
            return False, "Cancelled"

        all_masks: dict[int, np.ndarray] = {}
        for i, det in enumerate(detections):
            if self.cancel_requested:
                return False, "Cancelled"
            obj_id = i + 1
            mask, scores = sam2.predict_box(det.box_xyxy, label_id=obj_id)
            all_masks[obj_id] = mask

        self.set_progress(85)

        self._update_outputs(rgb_arr, all_masks)
        self.set_progress(100)
        self.mark_clean()
        logger.info("SAM2 Text Segment: %d objects for '%s'",
                     len(all_masks), text)
        return True, None

    def _empty_outputs(self, rgb_arr: np.ndarray, h: int, w: int):
        mask_pil = Image.fromarray(
            np.zeros((h, w), dtype=np.uint8), mode='L')
        self.output_values['mask'] = MaskData(payload=mask_pil)

        label_arr = np.zeros((h, w), dtype=np.int32)
        label_pil = Image.fromarray(
            np.zeros((h, w, 3), dtype=np.uint8), mode='RGB')
        self.output_values['label_image'] = LabelData(
            payload=label_arr, image=label_pil)

        self.output_values['overlay'] = ImageData(
            payload=Image.fromarray(rgb_arr, mode='RGB'))

    def _update_outputs(self, rgb_arr: np.ndarray,
                        all_masks: dict[int, np.ndarray]):
        h, w = rgb_arr.shape[:2]

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
            if np.any(m):
                label_rgb[m] = _obj_color(obj_id)
        label_pil = Image.fromarray(label_rgb, mode='RGB')
        self.output_values['label_image'] = LabelData(
            payload=label_arr, image=label_pil)

        vis = rgb_arr.astype(np.float32).copy()
        for obj_id in sorted(all_masks.keys()):
            m = label_arr == obj_id
            if np.any(m):
                color = np.array(_obj_color(obj_id), dtype=np.float32)
                vis[m] = vis[m] * 0.5 + color * 0.5
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        self.output_values['overlay'] = ImageData(
            payload=Image.fromarray(vis, mode='RGB'))
