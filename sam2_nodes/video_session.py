"""
video_session.py — SAM2 video propagation engine with temporal memory (ONNX).

Implements the full SAM2 temporal pipeline using 6 ONNX models:
  image_encoder, prompt_encoder, mask_decoder,
  memory_encoder, memory_attention, mlp

Data flow:
  Frame 0 (prompted):  encode → prompt → decode → memory_encode → store
  Frame N (propagate): encode → memory_attention → decode → memory_encode → store

Based on axinc-ai/ailia-models SAM2 inference code.
Models downloaded from Google Cloud Storage (ailia pre-exports).
"""
from __future__ import annotations

import logging
import time

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = ['SAM2VideoSession', 'TrackingCancelled']


class TrackingCancelled(Exception):
    """Raised when the user cancels tracking mid-frame."""

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


def _trunc_normal(shape, std=0.02, seed=3):
    """Truncated normal init matching the ailia reference implementation."""
    rng = np.random.RandomState(seed)
    return rng.normal(0, std, shape).astype(np.float32)


def _get_1d_sine_pe(pos_inds: np.ndarray, dim: int,
                     temperature: float = 10000.0) -> np.ndarray:
    """1-D sinusoidal positional encoding."""
    pe_dim = dim // 2
    dim_t = np.arange(pe_dim, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = np.expand_dims(pos_inds, axis=-1) / dim_t
    return np.concatenate(
        [np.sin(pos_embed), np.cos(pos_embed)], axis=-1,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class SAM2VideoSession:
    """Multi-object video tracking with SAM2 memory attention.

    Usage::

        session = SAM2VideoSession(model_paths)

        # Frame 0 — initialise with user prompts
        session.set_image(rgb_arr_0)
        for obj_id, (coords, labels) in prompts.items():
            mask = session.initialize_object(obj_id, coords, labels)

        # Frames 1..N — propagate
        for frame in frames[1:]:
            session.set_image(frame)
            masks, scores = session.propagate()
    """

    NUM_MASKMEM = 7       # 1 conditioning + 6 recent non-conditioning
    MAX_OBJ_PTRS = 16
    HIDDEN_DIM = 256
    MEM_DIM = 64
    INPUT_SIZE = 1024
    SIGMOID_SCALE = 20.0
    SIGMOID_BIAS = -10.0

    def __init__(self, model_paths: dict[str, str],
                 cancel_check=None) -> None:
        """
        Parameters
        ----------
        cancel_check : callable, optional
            A zero-arg function returning True if the operation should abort.
            Checked between ONNX model calls so the user can stop mid-frame.
        """
        self._cancel_check = cancel_check or (lambda: False)
        providers = _get_providers()
        ort = _onnxruntime()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        logger.info("Loading SAM2 video ONNX models …")
        self._enc = ort.InferenceSession(
            model_paths['image_encoder'], sess_options=opts, providers=providers)
        self._prompt_enc = ort.InferenceSession(
            model_paths['prompt_encoder'], sess_options=opts, providers=providers)
        self._mask_dec = ort.InferenceSession(
            model_paths['mask_decoder'], sess_options=opts, providers=providers)
        self._mem_enc = ort.InferenceSession(
            model_paths['memory_encoder'], sess_options=opts, providers=providers)
        self._mem_attn = ort.InferenceSession(
            model_paths['memory_attention'], sess_options=opts, providers=providers)
        self._mlp = ort.InferenceSession(
            model_paths['mlp'], sess_options=opts, providers=providers)

        self._has_tpos_proj = 'obj_ptr_tpos_proj' in model_paths
        if self._has_tpos_proj:
            self._tpos_proj = ort.InferenceSession(
                model_paths['obj_ptr_tpos_proj'], sess_options=opts, providers=providers)

        # Detect memory_attention input format (split vs combined)
        attn_names = {inp.name for inp in self._mem_attn.get_inputs()}
        self._split_memory = 'memory_1' in attn_names
        self._has_attn_mask = 'attention_mask_1' in attn_names

        # Learned embeddings (approximated — see ailia reference)
        self._maskmem_tpos_enc = _trunc_normal(
            (self.NUM_MASKMEM, 1, 1, self.MEM_DIM))
        self._no_mem_embed = _trunc_normal((1, 1, self.HIDDEN_DIM))
        self._no_obj_ptr = _trunc_normal((1, self.HIDDEN_DIM))

        # Per-frame cached state (set by set_image)
        self._orig_size: tuple[int, int] = (0, 0)
        self._vision_feats: list[np.ndarray] | None = None
        self._vision_pos: list[np.ndarray] | None = None
        self._feat_sizes: list[tuple[int, int]] | None = None

        # Per-object tracking state: obj_id → {frame_idx → memory dict}
        self._cond_mem: dict[int, dict[int, dict]] = {}
        self._noncond_mem: dict[int, dict[int, dict]] = {}
        self._frame_idx = -1

        # Caches
        self._padding_prompt: tuple | None = None
        self._cached_maskmem_pos_enc: np.ndarray | None = None

        logger.info("SAM2 video models loaded (split_mem=%s, attn_mask=%s)",
                     self._split_memory, self._has_attn_mask)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all tracking state."""
        self._cond_mem.clear()
        self._noncond_mem.clear()
        self._frame_idx = -1
        self._vision_feats = None
        self._padding_prompt = None
        self._cached_maskmem_pos_enc = None

    def _check_cancel(self):
        if self._cancel_check():
            raise TrackingCancelled()

    def set_image(self, rgb_arr: np.ndarray) -> None:
        """Encode image for the current frame (shared across objects)."""
        self._check_cancel()
        self._orig_size = (rgb_arr.shape[0], rgb_arr.shape[1])
        self._frame_idx += 1

        # Preprocess
        pil = Image.fromarray(rgb_arr).resize(
            (self.INPUT_SIZE, self.INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = ((arr - mean) / std).transpose(2, 0, 1)[np.newaxis, ...]

        # Run encoder → 7 outputs
        t0 = time.perf_counter()
        outputs = self._enc.run(
            None, {self._enc.get_inputs()[0].name: arr})
        dt = (time.perf_counter() - t0) * 1000
        logger.info("SAM2 video encoder: %.1f ms", dt)

        # Parse encoder outputs — try named first, then shape-based
        out_names = [o.name for o in self._enc.get_outputs()]
        named = dict(zip(out_names, outputs))

        for name, arr in named.items():
            logger.debug("Encoder output '%s': shape %s", name, arr.shape)

        backbone_fpn, vision_pos_enc = [], []
        for i in range(3):
            fpn = named.get(f'backbone_fpn_{i}')
            pe = named.get(f'vision_pos_enc_{i}')
            if fpn is not None and pe is not None:
                backbone_fpn.append(fpn)
                vision_pos_enc.append(pe)

        if not backbone_fpn:
            backbone_fpn, vision_pos_enc = \
                self._detect_encoder_outputs(outputs)

        # Flatten NxCxHxW → HWxNxC (each tensor uses its OWN shape)
        self._vision_feats, self._vision_pos, self._feat_sizes = [], [], []
        for fpn, pe in zip(backbone_fpn, vision_pos_enc):
            Bf, Cf, Hf, Wf = fpn.shape
            Bp, Cp, Hp, Wp = pe.shape
            self._feat_sizes.append((Hf, Wf))
            self._vision_feats.append(
                fpn.reshape(Bf, Cf, Hf * Wf).transpose(2, 0, 1))
            self._vision_pos.append(
                pe.reshape(Bp, Cp, Hp * Wp).transpose(2, 0, 1))

    def initialize_object(
        self, obj_id: int,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> np.ndarray:
        """Initialise tracking for one object on the current image.

        Parameters
        ----------
        point_coords : (N, 2) int — (x, y) in original image space.
        point_labels : (N,) int — 1 = foreground, 0 = background.

        Returns
        -------
        mask : (H, W) uint8 binary mask.
        """
        assert self._vision_feats is not None, "Call set_image() first"

        curr = self._vision_feats[-1]  # (HW, B, C)
        B = curr.shape[1]
        C = curr.shape[2]
        H, W = self._feat_sizes[-1]

        # No memory on first frame → add no_mem_embed
        pix_feat_with_mem = curr + self._no_mem_embed  # broadcasts (1,1,256)
        pix_feat = pix_feat_with_mem.transpose(1, 2, 0).reshape(B, C, H, W)

        high_res = self._get_high_res_features()

        # Normalise coords to encoder input space
        oh, ow = self._orig_size
        coords = point_coords.astype(np.float32).copy()
        coords[:, 0] = coords[:, 0] / ow * self.INPUT_SIZE
        coords[:, 1] = coords[:, 1] / oh * self.INPUT_SIZE

        sparse, dense, dense_pe = self._run_prompt_encoder(
            coords[np.newaxis], point_labels[np.newaxis].astype(np.int32))
        self._check_cancel()

        low_res_masks, iou_pred, sam_tokens, obj_score = \
            self._run_mask_decoder(pix_feat, sparse, dense, dense_pe, high_res)
        self._check_cancel()

        # Select best mask (multimask: tokens 1-3)
        best_idx = np.argmax(iou_pred[:, 1:], axis=-1)
        best_mask = low_res_masks[
            np.arange(B), best_idx + 1][:, np.newaxis]   # (B,1,256,256)
        best_token = sam_tokens[np.arange(B), best_idx + 1]  # (B, 256)

        obj_ptr = self._compute_obj_ptr(best_token, obj_score)
        mask_full = self._upscale_mask(best_mask[0, 0])

        # Memory encode (use raw backbone features, not attended)
        raw_pix = self._vision_feats[-1].transpose(1, 2, 0).reshape(
            B, C, H, W)
        maskmem_feat, maskmem_pe = self._run_memory_encoder(raw_pix, best_mask)

        if self._cached_maskmem_pos_enc is None:
            self._cached_maskmem_pos_enc = maskmem_pe

        self._cond_mem.setdefault(obj_id, {})[self._frame_idx] = {
            'maskmem_features': maskmem_feat,
            'maskmem_pos_enc': maskmem_pe,
            'obj_ptr': obj_ptr,
        }
        return mask_full

    def propagate(self) -> tuple[dict[int, np.ndarray], dict[int, float]]:
        """Propagate all tracked objects on the current image.

        Returns ``(masks, scores)`` where each maps ``obj_id`` to its result.
        """
        assert self._vision_feats is not None, "Call set_image() first"

        curr = self._vision_feats[-1]
        curr_pos = self._vision_pos[-1]
        B = curr.shape[1]
        C = curr.shape[2]
        H, W = self._feat_sizes[-1]
        high_res = self._get_high_res_features()

        # Padding prompt (cached — same for every propagation call)
        if self._padding_prompt is None:
            self._padding_prompt = self._run_prompt_encoder(
                np.zeros((1, 1, 2), dtype=np.float32),
                np.array([[-1]], dtype=np.int32))
        sparse, dense, dense_pe = self._padding_prompt

        masks: dict[int, np.ndarray] = {}
        scores: dict[int, float] = {}
        all_obj_ids = set(self._cond_mem) | set(self._noncond_mem)

        for obj_id in sorted(all_obj_ids):
            self._check_cancel()
            mem_t, mem_p, ptr_t, ptr_p = self._assemble_memory(obj_id)

            if mem_t:
                pix_feat_with_mem = self._run_memory_attention(
                    curr, curr_pos, mem_t, mem_p, ptr_t, ptr_p)
            else:
                pix_feat_with_mem = curr + self._no_mem_embed

            self._check_cancel()
            pix_feat = pix_feat_with_mem.transpose(1, 2, 0).reshape(
                B, C, H, W)

            low_res_masks, iou_pred, sam_tokens, obj_score = \
                self._run_mask_decoder(
                    pix_feat, sparse, dense, dense_pe, high_res)

            best_idx = np.argmax(iou_pred[:, 1:], axis=-1)
            best_mask = low_res_masks[
                np.arange(B), best_idx + 1][:, np.newaxis]
            best_token = sam_tokens[np.arange(B), best_idx + 1]
            confidence = float(iou_pred[0, best_idx[0] + 1])

            obj_ptr = self._compute_obj_ptr(best_token, obj_score)
            mask_full = self._upscale_mask(best_mask[0, 0])

            # Memory encode
            raw_pix = self._vision_feats[-1].transpose(1, 2, 0).reshape(
                B, C, H, W)
            maskmem_feat, maskmem_pe = self._run_memory_encoder(
                raw_pix, best_mask)
            if self._cached_maskmem_pos_enc is None:
                self._cached_maskmem_pos_enc = maskmem_pe

            self._noncond_mem.setdefault(obj_id, {})[self._frame_idx] = {
                'maskmem_features': maskmem_feat,
                'maskmem_pos_enc': maskmem_pe,
                'obj_ptr': obj_ptr,
            }

            masks[obj_id] = mask_full
            scores[obj_id] = confidence

        return masks, scores

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tracked_objects(self) -> set[int]:
        return set(self._cond_mem) | set(self._noncond_mem)

    @property
    def frame_index(self) -> int:
        return self._frame_idx

    # ------------------------------------------------------------------
    # ONNX model wrappers
    # ------------------------------------------------------------------

    def _run_prompt_encoder(self, coords, labels):
        # Prompt encoder expects masks as 3-D: (B, H, W)
        mask_input = np.zeros((1, 256, 256), dtype=np.float32)
        masks_enable = np.array([0], dtype=np.int32)
        out = self._prompt_enc.run(None, {
            'coords': coords.astype(np.float32),
            'labels': labels.astype(np.int32),
            'masks': mask_input,
            'masks_enable': masks_enable,
        })
        return out[0], out[1], out[2]  # sparse, dense, dense_pe

    def _run_mask_decoder(self, pix_feat, sparse, dense, dense_pe, high_res):
        out = self._mask_dec.run(None, {
            'image_embeddings': pix_feat.astype(np.float32),
            'image_pe': dense_pe.astype(np.float32),
            'sparse_prompt_embeddings': sparse.astype(np.float32),
            'dense_prompt_embeddings': dense.astype(np.float32),
            'high_res_features1': high_res[0].astype(np.float32),
            'high_res_features2': high_res[1].astype(np.float32),
        })
        masks = out[0]                                       # (B, 4, 256, 256)
        iou = out[1]                                         # (B, 4)
        tokens = out[2]                                      # (B, 4, 256)
        obj_score = out[3] if len(out) > 3 else np.ones(     # (B, 1)
            (masks.shape[0], 1), dtype=np.float32)
        return masks, iou, tokens, obj_score

    def _run_memory_attention(self, curr, curr_pos,
                               mem_t, mem_p, ptr_t, ptr_p):
        """Returns pix_feat_with_mem (HW, B, C)."""
        memory = np.concatenate(mem_t, axis=0).astype(np.float32)
        memory_pos = np.concatenate(mem_p, axis=0).astype(np.float32)

        B = curr.shape[1]
        if ptr_t:
            obj_ptrs = np.concatenate(ptr_t, axis=0).astype(np.float32)
            obj_ptr_pos = np.concatenate(ptr_p, axis=0).astype(np.float32)
            n_ptr = obj_ptrs.shape[0]
        else:
            obj_ptrs = np.zeros((1, B, self.MEM_DIM), dtype=np.float32)
            obj_ptr_pos = np.zeros((1, B, self.MEM_DIM), dtype=np.float32)
            n_ptr = 0

        t0 = time.perf_counter()

        if self._split_memory:
            feed: dict = {
                'curr': curr.astype(np.float32),
                'memory_1': memory,
                'memory_2': obj_ptrs,
                'curr_pos': curr_pos.astype(np.float32),
                'memory_pos_1': memory_pos,
                'memory_pos_2': obj_ptr_pos,
            }
            if self._has_attn_mask:
                feed['attention_mask_1'] = np.ones(
                    (memory.shape[0], B), dtype=bool)
                feed['attention_mask_2'] = np.ones(
                    (obj_ptrs.shape[0], B), dtype=bool)
        else:
            if n_ptr > 0:
                all_mem = np.concatenate([memory, obj_ptrs], axis=0)
                all_pos = np.concatenate([memory_pos, obj_ptr_pos], axis=0)
            else:
                all_mem = memory
                all_pos = memory_pos
            feed = {
                'curr': curr.astype(np.float32),
                'memory': all_mem,
                'curr_pos': curr_pos.astype(np.float32),
                'memory_pos': all_pos,
                'num_obj_ptr_tokens': np.array(n_ptr, dtype=np.int64),
            }

        out = self._mem_attn.run(None, feed)
        dt = (time.perf_counter() - t0) * 1000
        logger.info("SAM2 memory attention: %.1f ms", dt)
        return out[0]

    def _run_memory_encoder(self, pix_feat, pred_mask):
        """Returns (maskmem_features, maskmem_pos_enc)."""
        mask_bin = (pred_mask > 0).astype(np.float32)
        mask_pil = Image.fromarray(mask_bin[0, 0])
        mask_resized = np.array(
            mask_pil.resize((self.INPUT_SIZE, self.INPUT_SIZE), Image.NEAREST),
            dtype=np.float32)
        mask_scaled = (mask_resized[np.newaxis, np.newaxis]
                       * self.SIGMOID_SCALE + self.SIGMOID_BIAS)
        out = self._mem_enc.run(None, {
            'pix_feat': pix_feat.astype(np.float32),
            'masks': mask_scaled.astype(np.float32),
        })
        return out[0], out[1]

    def _compute_obj_ptr(self, sam_token, obj_score):
        obj_ptr = self._mlp.run(
            None, {'x': sam_token.astype(np.float32)})[0]
        is_visible = (obj_score > 0).astype(np.float32)
        return is_visible * obj_ptr + (1.0 - is_visible) * self._no_obj_ptr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_high_res_features(self) -> list[np.ndarray]:
        """Backbone FPN levels 0 & 1 in BCHW format."""
        result = []
        for vf, fs in zip(self._vision_feats[:-1], self._feat_sizes[:-1]):
            B = vf.shape[1]
            C = vf.shape[2]  # actual channel count (may differ per level)
            result.append(
                vf.transpose(1, 2, 0).reshape(B, C, *fs))
        return result

    @staticmethod
    def _detect_encoder_outputs(outputs):
        """Group encoder outputs by spatial resolution into (fpn, pe) pairs.

        The encoder emits 6-7 4-D tensors.  At each spatial resolution
        (256x256, 128x128, 64x64) there are exactly two tensors — one for
        backbone features and one for positional encoding.  An additional
        ``vision_features`` tensor may exist at one resolution.

        Heuristic: at each resolution, the tensor whose channel dimension
        equals 256 is the positional encoding; the other is the FPN feature.
        If both (or neither) are 256 we just take them in output order.
        """
        by_spatial: dict[tuple[int, int], list[np.ndarray]] = {}
        for arr in outputs:
            if arr.ndim == 4:
                key = (arr.shape[2], arr.shape[3])
                by_spatial.setdefault(key, []).append(arr)

        backbone_fpn, vision_pos_enc = [], []
        # Largest spatial first → level 0 (256x256), then 1, then 2
        for key in sorted(by_spatial, key=lambda k: -(k[0] * k[1])):
            tensors = by_spatial[key]
            if len(tensors) < 2:
                continue  # vision_features or single tensor — skip
            if len(tensors) == 2:
                a, b = tensors
            else:
                # 3+ tensors at this resolution: take the two with largest C
                tensors.sort(key=lambda t: t.shape[1], reverse=True)
                a, b = tensors[0], tensors[1]

            # Identify PE (usually 256-channel) vs FPN feature
            if a.shape[1] == b.shape[1]:
                # Same channels — use output order (first=FPN, second=PE)
                backbone_fpn.append(a)
                vision_pos_enc.append(b)
            elif a.shape[1] > b.shape[1]:
                vision_pos_enc.append(a)
                backbone_fpn.append(b)
            else:
                vision_pos_enc.append(b)
                backbone_fpn.append(a)

        logger.info("Detected encoder layout: %d levels — FPN channels %s, "
                     "PE channels %s, spatial %s",
                     len(backbone_fpn),
                     [f.shape[1] for f in backbone_fpn],
                     [p.shape[1] for p in vision_pos_enc],
                     [f"{f.shape[2]}x{f.shape[3]}" for f in backbone_fpn])
        return backbone_fpn[:3], vision_pos_enc[:3]

    def _upscale_mask(self, mask_256: np.ndarray) -> np.ndarray:
        """Resize (256, 256) logits → (H, W) binary uint8."""
        oh, ow = self._orig_size
        binary = (mask_256 > 0).astype(np.uint8) * 255
        resized = np.array(
            Image.fromarray(binary).resize((ow, oh), Image.NEAREST),
            dtype=np.uint8)
        return (resized > 127).astype(np.uint8)

    def _assemble_memory(self, obj_id: int):
        """Collect memory tensors for one object at the current frame."""
        B = 1
        mem_t: list[np.ndarray] = []
        mem_p: list[np.ndarray] = []
        ptr_t: list[np.ndarray] = []
        ptr_p: list[np.ndarray] = []

        cond = self._cond_mem.get(obj_id, {})
        noncond = self._noncond_mem.get(obj_id, {})

        # 1. Conditioning frames (t_pos = 0)
        for fid in sorted(cond):
            if fid >= self._frame_idx:
                continue
            feats = cond[fid]['maskmem_features']          # (B, 64, 64, 64)
            flat = feats.reshape(B, self.MEM_DIM, -1).transpose(2, 0, 1)
            mem_t.append(flat)

            pos = (self._cached_maskmem_pos_enc
                   if self._cached_maskmem_pos_enc is not None
                   else cond[fid]['maskmem_pos_enc'])
            flat_pos = pos.reshape(B, self.MEM_DIM, -1).transpose(2, 0, 1)
            flat_pos = flat_pos + self._maskmem_tpos_enc[self.NUM_MASKMEM - 1]
            mem_p.append(flat_pos)

        # 2. Recent non-conditioning frames (t_pos = 1 … NUM_MASKMEM-1)
        for t_pos in range(1, self.NUM_MASKMEM):
            t_rel = self.NUM_MASKMEM - t_pos
            prev_idx = (self._frame_idx - 1) if t_rel == 1 else (
                self._frame_idx - t_rel)
            out = noncond.get(prev_idx)
            if out is None:
                continue

            flat = out['maskmem_features'].reshape(
                B, self.MEM_DIM, -1).transpose(2, 0, 1)
            mem_t.append(flat)

            pos = (self._cached_maskmem_pos_enc
                   if self._cached_maskmem_pos_enc is not None
                   else out['maskmem_pos_enc'])
            flat_pos = pos.reshape(B, self.MEM_DIM, -1).transpose(2, 0, 1)
            flat_pos = (flat_pos
                        + self._maskmem_tpos_enc[self.NUM_MASKMEM - t_pos - 1])
            mem_p.append(flat_pos)

        # 3. Object pointers from all past frames
        all_past = []
        for fid in sorted(set(cond) | set(noncond)):
            if fid >= self._frame_idx:
                continue
            entry = cond.get(fid) or noncond.get(fid)
            all_past.append((fid, entry))
        all_past = all_past[-self.MAX_OBJ_PTRS:]

        if all_past:
            ptrs = np.stack(
                [e['obj_ptr'] for _, e in all_past], axis=0)  # (N, B, 256)
            N = ptrs.shape[0]

            # Split each 256-dim pointer into 4 × 64-dim tokens
            ptrs_split = (ptrs.reshape(N, B, 4, self.MEM_DIM)
                          .transpose(0, 2, 1, 3)
                          .reshape(N * 4, B, self.MEM_DIM))
            ptr_t.append(ptrs_split)

            # Temporal position encoding
            t_diffs = np.array(
                [self._frame_idx - fid for fid, _ in all_past],
                dtype=np.float32)
            t_max = max(1.0, float(self.MAX_OBJ_PTRS - 1))

            if self._has_tpos_proj:
                sine_pe = _get_1d_sine_pe(
                    t_diffs / t_max, dim=self.HIDDEN_DIM)
                tpos = np.zeros((N, self.MEM_DIM), dtype=np.float32)
                for i in range(N):
                    tpos[i:i + 1] = self._tpos_proj.run(
                        None, {'x': sine_pe[i:i + 1]})[0]
            else:
                tpos = _get_1d_sine_pe(t_diffs / t_max, dim=self.MEM_DIM)

            tpos_exp = np.tile(tpos[:, np.newaxis, :], (1, B, 1))
            tpos_exp = np.repeat(tpos_exp, 4, axis=0)  # (N*4, B, 64)
            ptr_p.append(tpos_exp)

        return mem_t, mem_p, ptr_t, ptr_p
