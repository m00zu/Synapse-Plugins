"""3D cell segmentation from .ims files — Rust-accelerated version.

Uses image_process_3d_rs for: gaussian_filter, distance_transform_edt,
label, watershed, binary_fill_holes, remove_small_objects (all 3D).
Uses ims_reader_rs for fast IMS file reading.

Pipeline:
  1. Read .ims (Rust), detect active Z-slices (yellow/red channel)
  2. 3D Gaussian blur (Rust, parallel)
  3. Percentile threshold + cleanup (Rust)
  4. 3D watershed to split touching cells (Rust)
  5. Expand labels by N µm (Rust EDT + watershed)
  6. Measure green intensity per cell (raw + percentiles)
  7. Save CSV + composite MIP overlay
"""
import os, time, glob, gc
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter as gaussian_filter_2d, uniform_filter1d
from scipy.stats import skew as scipy_skew
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
import image_process_rs as ip2d
import pandas as pd

# Rust-accelerated 3D operations
import image_process_3d_rs as ip3d
try:
    import ims_reader_rs
    HAS_IMS_RS = True
except ImportError:
    import h5py
    HAS_IMS_RS = False
    print("Warning: ims_reader_rs not available, falling back to h5py")


SKEW_TOP_PCTS = [10, 5]  # compute skewness of top N% brightest voxels


# C2 = Control
# E2 = TGF-b
# B4 = LOXi
# E3 = Nintedanib
# G2 = Rep-Sox


def _skewness_metrics(pixels_u16, prefix, step_um):
    """Compute top-N% skewness for a set of green pixels (uint16 scale)."""
    row = {}
    n = len(pixels_u16)
    if n < 10:
        for pct in SKEW_TOP_PCTS:
            row[f'skew_top{pct}pct_at_{step_um}um'] = 0
        return row
    sorted_px = np.sort(pixels_u16)
    for pct in SKEW_TOP_PCTS:
        n_top = max(10, int(n * pct / 100))
        top = sorted_px[-n_top:].astype(np.float64)
        row[f'skew_top{pct}pct_at_{step_um}um'] = round(float(scipy_skew(top)), 4)
    return row


# ── Settings ──────────────────────────────────────────────────────────────

# 3D segmentation
SIGMA_UM = 1.0          # Gaussian blur in µm (applied anisotropically)
MIN_SIZE_VOXELS = 3000  # remove objects smaller than this
MAX_SIZE_VOXELS = 50000 # remove objects larger than this (background blobs, not cells)
MAX_Z_SLICES = 30        # cells spanning more Z-slices than this are likely background
MIN_SIGNAL_UINT16 = 10  # yellow p99.9 must exceed this to attempt segmentation
CLAHE_CLIP = 0.03       # CLAHE contrast limit
CLAHE_KERNEL = 128      # CLAHE tile size in pixels (XY)
MIN_DISTANCE_UM = 20  # min seed separation for watershed in µm
MAX_BINARY_PCT = 80     # if binary voxels exceed this %, skip as BG (no real cells)
TOP_PERCENTILE = 99.5   # threshold at this percentile (top 0.5% brightest voxels)
MIN_THRESH_UINT16 = 50 # absolute minimum threshold — no-cell images won't go below this

# Local adaptive threshold (applied on MIP, broadcast to 3D)
USE_LOCAL_THRESH = True            # enable local adaptive threshold (replaces global percentile)
LOCAL_BG_RADIUS_PX = 75           # uniform_filter half-size in pixels for local BG estimation
LOCAL_THRESH_OFFSET = 3 / 65535.0  # signal must exceed local BG by this amount (in [0,1] float)
WELL_MIN_UINT16 = 250   # green MIP below this (after blur) = outside well (excluded from BG)

# Expansion
# EXPAND_UM = 70.         # expand cell labels by this many µm (max)
# EXPAND_UM = 30.        # must be >= max(EXPAND_STEPS)
EXPAND_UM = 30.        # must be >= max(EXPAND_STEPS)
# EXPAND_STEPS = [0, 5] + list(range(10, 75, 5))  # 0 (cell only), 5, 10, 15, ..., 50 µm
# EXPAND_STEPS = [0, 5] + list(range(10, 35, 5))  # 0 (cell only), 5, 10, 15, ..., 30 µm
EXPAND_STEPS = [0]    # Uncorrected & Corrected
# EXPAND_STEPS = [70.0]    # Uncorrected & Corrected

# Green thresholds — test multiple cutoffs for pct_above calculation
# GREEN_THRESHOLDS = [600, 800, 1000, 1200, 1400, 1600]  # uint16
# GREEN_THRESHOLDS = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300]  # uint16
# GREEN_THRESHOLDS = list(range(500, 2350, 50))
# GREEN_THRESHOLDS = [800]  # Uncorrected
# GREEN_THRESHOLDS = [1400]  # Corrected
# GREEN_THRESHOLDS = [650]
GREEN_THRESHOLDS = [1600]

# Artifact detection (2D on green MIP, broadcast to 3D)
ARTIFACT_BLUR_PX = 50           # Gaussian blur in pixels (on MIP)
ARTIFACT_MIN_SIZE_2D = 25000     # min blob size in pixels (2D) — lowered to catch smaller artifacts
ARTIFACT_PERCENTILE = 99.0
ARTIFACT_ABS_UINT16 = 4000       # absolute green MIP threshold for artifacts (catches blobs on bright images)
ARTIFACT_MAX_ECCENTRICITY = 0.7 # max eccentricity (0=circle, 1=line) — artifacts are round
ARTIFACT_CELL_ONLY_SKIP = 0.9    # skip blob if >90% is INSIDE cells (blob IS the cell, not an artifact)
ARTIFACT_EDGE_MARGIN_PX = 20     # blob touching this many pixels from border = edge artifact (skip eccentricity)

# Total-image brightness check — flag entire green MIP as suspect
BRIGHT_IMAGE_MEDIAN_UINT16 = 1500  # if green MIP median (excluding outside-well) exceeds this, flag image
BRIGHT_IMAGE_P90_UINT16 = 2000    # or if p90 exceeds this

# Artifact overlap
MAX_ARTIFACT_OVERLAP_PCT = 70

# Well-border exclusion: cells within this many pixels of the well edge are artifacts
# (catches bright autofluorescence lines at well boundary)
WELL_BORDER_MARGIN_PX = 50

# Nucleus detection (channel 2) — new in v2
NUCLEUS_CH_IDX = 2
NUCLEUS_SIGMA_UM = 0.5                       # 2D Gaussian blur (in µm) before local threshold
NUCLEUS_LOCAL_BG_RADIUS_PX = 60              # uniform_filter half-size for local BG estimation
NUCLEUS_LOCAL_THRESH_OFFSET_U16 = 30         # signal must exceed local BG by this many u16
NUCLEUS_ABS_FLOOR_U16 = 100                  # absolute floor — below this is noise regardless of local BG
NUCLEUS_MIN_PX = 80                          # 2D min area in pixels (~5.5 µm disk at 0.6µm/px ≈ 64 px)
NUCLEUS_MAX_PX = 2000                        # 2D max area in pixels — reject oversized blobs (artifacts)
MIN_NUCLEUS_OVERLAP_PCT = 30           # a cell is valid if ≥ this % of any single nucleus (2D) lies inside its XY footprint
DISPLAY_NUCLEUS_LO = 200 / 65535       # display range for nucleus channel in composite (blue)
DISPLAY_NUCLEUS_HI = 600 / 65535
DASH_LEN = 4                           # dash length in pixels for no-nucleus excluded cell contours


# ── IMS Reader (Rust-accelerated) ─────────────────────────────────────────
def _ims_attr_str(attr):
    if isinstance(attr, np.ndarray):
        return ''.join(b.decode() if isinstance(b, bytes) else str(b) for b in attr)
    if isinstance(attr, bytes):
        return attr.decode()
    return str(attr)


def read_ims(path, resolution_level=0, timepoint=0):
    """Read IMS file. Uses Rust reader if available, falls back to h5py."""
    if HAS_IMS_RS:
        vol, meta = ims_reader_rs.read_ims(str(path))
        # vol is (C, Z, Y, X) uint16 — convert to float32 [0,1] per channel
        channels = []
        for c in range(vol.shape[0]):
            channels.append(vol[c].astype(np.float32) / 65535.0)
        del vol; gc.collect()
        x, y, z = meta['x'], meta['y'], meta['z']
        return {
            'channels': channels, 'n_channels': meta['n_channels'],
            'shape': (z, y, x),
            'scale_um': (meta['pixel_size_x'], meta['pixel_size_y'], meta['pixel_size_z']),
            'ext_min': meta['ext_min'],  # [x_min, y_min, z_min] in µm
            'ext_max': meta['ext_max'],  # [x_max, y_max, z_max] in µm
            'recording_date': meta.get('recording_date', ''),
            'channel_names': [f'Ch{i+1}' for i in range(meta['n_channels'])],
            'bit_depth': 16,
        }
    else:
        # h5py fallback
        import h5py
        with h5py.File(path, 'r') as f:
            img_info = f['DataSetInfo']['Image']
            x_size = int(_ims_attr_str(img_info.attrs['X']))
            y_size = int(_ims_attr_str(img_info.attrs['Y']))
            z_size = int(_ims_attr_str(img_info.attrs['Z']))
            ext_min_x = float(_ims_attr_str(img_info.attrs['ExtMin0']))
            ext_max_x = float(_ims_attr_str(img_info.attrs['ExtMax0']))
            ext_min_y = float(_ims_attr_str(img_info.attrs['ExtMin1']))
            ext_max_y = float(_ims_attr_str(img_info.attrs['ExtMax1']))
            ext_min_z = float(_ims_attr_str(img_info.attrs['ExtMin2']))
            ext_max_z = float(_ims_attr_str(img_info.attrs['ExtMax2']))
            x_um = (ext_max_x - ext_min_x) / x_size if x_size > 0 else 1.0
            y_um = (ext_max_y - ext_min_y) / y_size if y_size > 0 else 1.0
            z_um = (ext_max_z - ext_min_z) / z_size if z_size > 1 else 1.0
            tp = f['DataSet']['ResolutionLevel 0']['TimePoint 0']
            n_channels = sum(1 for k in tp.keys() if k.startswith('Channel'))
            channels = []
            for ch_idx in range(n_channels):
                raw = tp[f'Channel {ch_idx}']['Data'][:z_size, :y_size, :x_size]
                channels.append(raw.astype(np.float32) / 65535.0)
            rec_date = _ims_attr_str(img_info.attrs.get('RecordingDate', b''))
        return {
            'channels': channels, 'n_channels': n_channels,
            'shape': (z_size, y_size, x_size),
            'scale_um': (abs(x_um), abs(y_um), abs(z_um)),
            'ext_min': [ext_min_x, ext_min_y, ext_min_z],
            'ext_max': [ext_max_x, ext_max_y, ext_max_z],
            'recording_date': rec_date,
            'channel_names': [f'Ch{i+1}' for i in range(n_channels)],
            'bit_depth': 16,
        }


def _read_single_channel(path, ch_idx=0):
    """Read a single channel from .ims as float32 [0,1]."""
    if HAS_IMS_RS:
        vol, _ = ims_reader_rs.read_ims(str(path), channel=ch_idx)
        return vol.astype(np.float32) / 65535.0
    else:
        import h5py
        with h5py.File(path, 'r') as f:
            img_info = f['DataSetInfo']['Image']
            x = int(_ims_attr_str(img_info.attrs['X']))
            y = int(_ims_attr_str(img_info.attrs['Y']))
            z = int(_ims_attr_str(img_info.attrs['Z']))
            tp = f['DataSet']['ResolutionLevel 0']['TimePoint 0']
            raw = tp[f'Channel {ch_idx}']['Data'][:z, :y, :x]
            return raw.astype(np.float32) / 65535.0


# ── Active Z detection ────────────────────────────────────────────────────
def detect_active_z(yellow, n_samples=50000, threshold=0.15, z_pad=2):
    rng = np.random.default_rng(42)
    nz = yellow.shape[0]
    profile = np.empty(nz)
    for z in range(nz):
        sl = yellow[z].ravel()
        idx = rng.integers(0, len(sl), size=min(n_samples, len(sl)))
        profile[z] = np.percentile(sl[idx], 99)

    y_smooth = uniform_filter1d(profile, size=3)
    p_range = y_smooth.max() - y_smooth.min()
    p_median = np.median(y_smooth)

    if p_range < 0.1 * p_median:
        return 0, nz - 1

    y_normed = (y_smooth - y_smooth.min()) / (p_range + 1e-10)
    active = y_normed > threshold
    active_slices = np.where(active)[0]
    if len(active_slices) > 0:
        z_start = max(0, active_slices[0] - z_pad)
        z_end = min(nz - 1, active_slices[-1] + z_pad)
        return z_start, z_end
    return 0, nz - 1


# ── Detect outside-well region ────────────────────────────────────────────
VOID_MIN_SIZE_PX = 3000     # minimum dark region size to exclude (pixels, 2D)
VOID_THRESH_UINT16 = 80     # pixels below this (after blur) = dark void (absolute)
VOID_RELATIVE_THRESH = 0.88 # per-layer yellow: below this fraction of layer median = void
GREEN_VOID_FRACTION = 0.5   # green: below this fraction of peak-layer median = no collagen
GREEN_VOID_MIN_UINT16 = 300 # absolute floor: green below this is always noise, not collagen


def _detect_outside_well(green_mip, min_uint16=100):
    """Detect the dark region outside the well on a 2D green MIP.

    Only labels dark regions that touch the image border as "outside well".
    Small dark spots inside the well are NOT excluded.

    Returns a 2D bool mask (True = outside well).
    """
    blurred = gaussian_filter_2d(green_mip, sigma=30)
    dark = blurred < (min_uint16 / 65535.0)
    if not dark.any():
        return np.zeros(green_mip.shape, dtype=bool)

    dark_labels, _ = ip2d.label_2d(dark.astype(np.uint8))

    h, w = green_mip.shape
    border_labels = set()
    border_labels |= set(dark_labels[0, :].ravel())
    border_labels |= set(dark_labels[-1, :].ravel())
    border_labels |= set(dark_labels[:, 0].ravel())
    border_labels |= set(dark_labels[:, -1].ravel())
    border_labels.discard(0)

    outside = np.zeros(green_mip.shape, dtype=bool)
    for lbl in border_labels:
        outside[dark_labels == lbl] = True

    return outside


def _detect_dark_voids(yellow_mip, green_mip=None,
                       thresh_uint16=VOID_THRESH_UINT16,
                       min_size=VOID_MIN_SIZE_PX):
    """Detect all large dark void regions (outside-well + bubbles + gaps).

    Uses both yellow and green MIPs if available. A void is any large
    connected region where BOTH channels are very dark (below thresh).
    This catches: outside-well, bubbles, air gaps, empty spaces.

    Returns a 2D bool mask (True = void, exclude from analysis).
    """
    # Use yellow as primary (uniform BG, most reliable)
    blurred_y = gaussian_filter_2d(yellow_mip, sigma=20)
    dark_y = blurred_y < (thresh_uint16 / 65535.0)

    # If green available, also check — void must be dark in both
    if green_mip is not None:
        blurred_g = gaussian_filter_2d(green_mip, sigma=20)
        # Green has higher BG (~500-700 uint16), so use a relative threshold
        green_p10 = np.percentile(blurred_g[blurred_g > 0], 10) if (blurred_g > 0).any() else 0
        dark_g = blurred_g < max(green_p10 * 0.5, thresh_uint16 / 65535.0)
        dark = dark_y | dark_g  # either channel being very dark = void
    else:
        dark = dark_y

    if not dark.any():
        return np.zeros(yellow_mip.shape, dtype=bool)

    # Label connected dark regions, keep only large ones
    dark_labels, _ = ip2d.label_2d(dark.astype(np.uint8))

    void_mask = np.zeros(yellow_mip.shape, dtype=bool)
    n_voids = 0
    for lbl in range(1, dark_labels.max() + 1):
        region = dark_labels == lbl
        area = region.sum()
        if area >= min_size:
            void_mask[region] = True
            n_voids += 1

    if void_mask.any():
        pct = void_mask.sum() / void_mask.size * 100
        print(f"  Dark voids: {n_voids} regions, {void_mask.sum()} px ({pct:.1f}%) excluded")

    return void_mask


def _detect_dark_voids_3d(yellow_vol, green_vol=None,
                          thresh_uint16=VOID_THRESH_UINT16,
                          relative_thresh=VOID_RELATIVE_THRESH,
                          green_void_frac=GREEN_VOID_FRACTION,
                          min_size=VOID_MIN_SIZE_PX, blur_sigma=20):
    """Detect dark void regions per Z-layer (bubbles, gaps, outside-well, no-collagen).

    Yellow: per-layer relative threshold (fraction of layer median).
    Green: fixed threshold = peak-layer median × green_void_frac.
      This correctly handles partial-coverage layers (uneven gel) without
      needing a per-layer dead/alive decision.

    Returns void_mask_3d: (Z, Y, X) bool array (True = void).
    """
    nz, ny, nx = yellow_vol.shape
    void_mask_3d = np.zeros((nz, ny, nx), dtype=bool)
    abs_thresh = thresh_uint16 / 65535.0

    # Green: compute fixed threshold from peak signal across all layers
    green_thresh = 0
    if green_vol is not None:
        layer_medians = np.array([np.median(green_vol[z]) for z in range(nz)])
        green_peak = np.max(layer_medians)
        green_min = GREEN_VOID_MIN_UINT16 / 65535.0
        green_thresh = max(abs_thresh, green_peak * green_void_frac, green_min)
        print(f"  Green void threshold: {green_thresh*65535:.0f} u16 "
              f"(peak median={green_peak*65535:.0f}, ×{green_void_frac}, floor={GREEN_VOID_MIN_UINT16})")

    # Blur the whole volume once (XY only, Rust parallel) — much faster than per-Z.
    if green_vol is not None and green_thresh > 0:
        blurred_vol = ip3d.gaussian_filter_3d(green_vol, 0.01, blur_sigma, blur_sigma)
        dark_vol = blurred_vol < green_thresh
    else:
        blurred_vol = ip3d.gaussian_filter_3d(yellow_vol, 0.01, blur_sigma, blur_sigma)
        # Per-layer relative threshold (yellow fallback)
        dark_vol = np.zeros_like(blurred_vol, dtype=bool)
        for z in range(nz):
            layer_median = np.median(blurred_vol[z])
            y_thresh = max(abs_thresh, layer_median * relative_thresh)
            dark_vol[z] = blurred_vol[z] < y_thresh
    del blurred_vol

    for z in range(nz):
        dark = dark_vol[z]
        if not dark.any():
            continue

        dark_labels, _ = ip2d.label_2d(dark.astype(np.uint8))
        max_lbl = int(dark_labels.max())
        if max_lbl == 0:
            continue

        # Vectorized: bincount gives area per label id, then LUT-index to get mask
        areas = np.bincount(dark_labels.ravel(), minlength=max_lbl + 1)
        kept = areas >= min_size
        kept[0] = False  # never keep background
        void_mask_3d[z] = kept[dark_labels]

    n_void_layers = int((void_mask_3d.any(axis=(1, 2))).sum())
    total_void = void_mask_3d.sum()
    if total_void > 0:
        pct = total_void / void_mask_3d.size * 100
        print(f"  Dark voids 3D: {n_void_layers}/{nz} layers have voids, "
              f"{total_void} voxels ({pct:.1f}%) excluded")
    else:
        print(f"  Dark voids 3D: no voids detected in any layer")

    return void_mask_3d


def _save_void_layers_image(yellow_vol, void_mask_3d, stem, out_dir, max_cols=8):
    """Save per-layer void detection diagnostic montage.

    Each tile shows one Z layer (yellow grayscale) with void overlay in red.
    """
    nz, ny, nx = yellow_vol.shape

    # Downscale thumbnails if images are large
    max_thumb = 256
    scale = min(1.0, max_thumb / max(ny, nx))
    thumb_w = int(nx * scale)
    thumb_h = int(ny * scale)

    n_cols = min(max_cols, nz)
    n_rows = (nz + n_cols - 1) // n_cols

    # Add space for Z-label text (8px top margin per row)
    label_h = 12
    cell_h = thumb_h + label_h
    canvas_w = n_cols * thumb_w
    canvas_h = n_rows * cell_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for z in range(nz):
        row = z // n_cols
        col = z % n_cols
        y0 = row * cell_h + label_h
        x0 = col * thumb_w

        layer = yellow_vol[z]
        p1, p99 = np.percentile(layer, [1, 99])
        if p99 > p1:
            norm = np.clip((layer - p1) / (p99 - p1), 0, 1)
        else:
            norm = np.zeros_like(layer)
        gray = (norm * 255).astype(np.uint8)

        # Resize
        gray_pil = Image.fromarray(gray).resize((thumb_w, thumb_h), Image.BILINEAR)
        gray_rs = np.array(gray_pil)

        void_layer = void_mask_3d[z]
        if scale < 1.0:
            void_pil = Image.fromarray((void_layer * 255).astype(np.uint8)).resize(
                (thumb_w, thumb_h), Image.NEAREST)
            void_rs = np.array(void_pil) > 128
        else:
            void_rs = void_layer

        # RGB tile
        tile = np.stack([gray_rs, gray_rs, gray_rs], axis=-1)

        # Red overlay on void regions
        if void_rs.any():
            tile[void_rs, 0] = np.minimum(tile[void_rs, 0].astype(int) + 120, 255).astype(np.uint8)
            tile[void_rs, 1] = (tile[void_rs, 1] * 0.3).astype(np.uint8)
            tile[void_rs, 2] = (tile[void_rs, 2] * 0.3).astype(np.uint8)
            # Red boundary
            bd = find_boundaries(void_rs.astype(np.int32), mode='thick')
            tile[bd] = [255, 0, 0]

        canvas[y0:y0+thumb_h, x0:x0+thumb_w] = tile

        # Z-label: write z index in top-left of cell
        # Simple: mark void layers with bright label area
        lbl_y0 = row * cell_h
        has_void = void_mask_3d[z].any()
        void_pct = void_mask_3d[z].sum() / (ny * nx) * 100 if has_void else 0
        if has_void:
            # Red tint on label area for void layers
            canvas[lbl_y0:lbl_y0+label_h, x0:x0+thumb_w, 0] = 180
            canvas[lbl_y0:lbl_y0+label_h, x0:x0+thumb_w, 1] = 40
            canvas[lbl_y0:lbl_y0+label_h, x0:x0+thumb_w, 2] = 40

    # Use PIL to add text labels
    from PIL import ImageDraw, ImageFont
    canvas_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 10)
    except Exception:
        font = ImageFont.load_default()

    for z in range(nz):
        row = z // n_cols
        col = z % n_cols
        lbl_y0 = row * cell_h + 1
        x0 = col * thumb_w + 2
        has_void = void_mask_3d[z].any()
        void_pct = void_mask_3d[z].sum() / (ny * nx) * 100 if has_void else 0
        txt = f"z{z}"
        if has_void:
            txt += f" ({void_pct:.0f}%)"
        color = (255, 80, 80) if has_void else (200, 200, 200)
        draw.text((x0, lbl_y0), txt, fill=color, font=font)

    img_path = os.path.join(out_dir, f'{stem}_void_layers.png')
    canvas_img.save(img_path)
    print(f"  Saved: {img_path}")


# ── Nucleus detection (channel 2, new in v2) ────────────────────────────
def _detect_nuclei_2d(nucleus_mip_raw,
                     spacing_um=None,
                     sigma_um=NUCLEUS_SIGMA_UM,
                     bg_radius_px=NUCLEUS_LOCAL_BG_RADIUS_PX,
                     offset_u16=NUCLEUS_LOCAL_THRESH_OFFSET_U16,
                     floor_u16=NUCLEUS_ABS_FLOOR_U16,
                     min_px=NUCLEUS_MIN_PX,
                     max_px=NUCLEUS_MAX_PX):
    """Detect nucleus blobs from the 2D MIP using local adaptive threshold.

    Pipeline:
      1. 2D Gaussian blur in pixel units (converted from µm via spacing).
      2. Compute local BG via uniform filter (radius = bg_radius_px).
      3. Binary mask: smooth > local_bg + offset AND smooth > absolute floor.
      4. Label 2D connected components.
      5. Keep components with min_px ≤ area ≤ max_px.

    Args:
        nucleus_mip_raw: (Y, X) float32 [0, 1] — pre-computed max projection.
        spacing_um: (z_um, y_um, x_um). Used for µm→px conversion of sigma.

    Returns:
        labels_2d: (Y, X) int32 array — kept nucleus labels, contiguous from 1.
        sizes: dict[label_id → pixel_count].
    """
    empty = (np.zeros(nucleus_mip_raw.shape, dtype=np.int32), {})

    if nucleus_mip_raw.max() <= nucleus_mip_raw.min() + 1e-6:
        print(f"  Nucleus detection: constant intensity, skipping")
        return empty

    # Step 1: 2D Gaussian blur (smooths noise before thresholding)
    if sigma_um and sigma_um > 0 and spacing_um is not None:
        _, y_um, x_um = spacing_um
        sigma_px = (sigma_um / y_um, sigma_um / x_um)
        smooth = gaussian_filter_2d(nucleus_mip_raw, sigma=sigma_px)
    else:
        smooth = nucleus_mip_raw

    # Step 2: Local adaptive threshold
    from scipy.ndimage import uniform_filter
    local_bg = uniform_filter(smooth, size=bg_radius_px * 2 + 1)
    offset = offset_u16 / 65535.0
    floor = floor_u16 / 65535.0
    binary = (smooth > local_bg + offset) & (smooth > floor)

    if not binary.any():
        print(f"  Nucleus detection: no pixels above local_bg + {offset_u16} u16")
        return empty

    raw_labels, _ = ip2d.label_2d(binary.astype(np.uint8))

    # Filter components outside [min_px, max_px] and relabel contiguously from 1
    clean_labels = np.zeros_like(raw_labels, dtype=np.int32)
    sizes = {}
    next_id = 1
    all_sizes = []
    n_too_small = 0
    n_too_large = 0
    for r in regionprops(raw_labels):
        if r.area < min_px:
            n_too_small += 1
            continue
        if r.area > max_px:
            n_too_large += 1
            continue
        clean_labels[raw_labels == r.label] = next_id
        sizes[next_id] = int(r.area)
        all_sizes.append(r.area)
        next_id += 1

    if n_too_large:
        print(f"  Nucleus detection: dropped {n_too_large} oversized blobs (>{max_px} px)")

    if not sizes:
        print(f"  Nucleus detection: no blobs in [{min_px}, {max_px}] px")
        return empty

    total = sum(sizes.values())
    print(f"  Nuclei (2D): {len(sizes)} blobs kept, {total} px total, "
          f"median area = {int(np.median(all_sizes))} px")
    return clean_labels, sizes


def _save_nucleus_diagnostic(yellow_mip, nucleus_mip, nucleus_labels_2d, stem, out_dir):
    """Save {stem}_nucleus_mip.png — yellow MIP + nucleus signal (blue) + nucleus mask boundary."""
    # Auto-contrast yellow MIP for display
    p1, p99 = np.percentile(yellow_mip, [1, 99.5])
    if p99 <= p1:
        p99 = p1 + 1e-6
    y_norm = np.clip((yellow_mip - p1) / (p99 - p1), 0, 1)

    # Nucleus MIP — normalize via its own percentiles
    p1n, p99n = np.percentile(nucleus_mip, [1, 99.5])
    if p99n <= p1n:
        p99n = p1n + 1e-6
    n_norm = np.clip((nucleus_mip - p1n) / (p99n - p1n), 0, 1)

    # RGB: yellow shown as R+G tint, nucleus signal in B
    h, w = yellow_mip.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = y_norm * 0.8
    rgb[..., 1] = y_norm * 0.8
    rgb[..., 2] = n_norm

    # Green contour around each detected nucleus (2D label mask)
    mask_2d = nucleus_labels_2d > 0
    if mask_2d.any():
        bd = find_boundaries(mask_2d.astype(np.int32), mode='thick')
        rgb[bd] = [0, 1, 0]

    img_path = os.path.join(out_dir, f'{stem}_nucleus_mip.png')
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(img_path)
    print(f"  Saved: {img_path}")


# ── 3D Segmentation ──────────────────────────────────────────────────────
def segment_3d(volume, spacing_um, sigma_um=1.0, min_size=50, max_size=500000,
               clahe_clip=0.03, clahe_kernel=128, min_distance_um=10.0,
               top_percentile=99.5, debug_dir=None, debug_stem=None,
               nucleus_seeds_2d=None):
    """Segment cells: percentile threshold on raw → 3D watershed.

    Uses a percentile-based threshold on the raw volume (top X% brightest
    voxels are candidates). This avoids CLAHE/Otsu issues with dim cells
    on varying backgrounds.

    Args:
        volume: (Z, Y, X) float32 [0,1]
        spacing_um: (z_um, y_um, x_um) physical voxel sizes
        sigma_um: Gaussian blur in µm
        min_size: minimum object size in voxels
        max_size: maximum object size in voxels
        min_distance_um: min seed separation in µm for watershed
        top_percentile: threshold at this percentile (e.g. 99.5 = top 0.5%)

    Returns:
        labels_3d: (Z, Y, X) int32 label array
    """
    from skimage.feature import peak_local_max
    z_um, y_um, x_um = spacing_um
    nz, ny, nx = volume.shape

    # Step 1: Anisotropy-aware Gaussian blur on raw volume (Rust, parallel)
    sigma_voxels = (sigma_um / z_um, sigma_um / y_um, sigma_um / x_um)
    print(f"  Gaussian blur 3D (Rust): σ={sigma_um}µm → voxels ({sigma_voxels[0]:.2f}, {sigma_voxels[1]:.2f}, {sigma_voxels[2]:.2f})")
    smooth = ip3d.gaussian_filter_3d(volume, sigma_voxels[0], sigma_voxels[1], sigma_voxels[2])

    # Step 2: Threshold — local adaptive (MIP-based, broadcast to 3D) or global
    mip = smooth.max(axis=0)
    abs_thresh = MIN_THRESH_UINT16 / 65535.0

    if USE_LOCAL_THRESH:
        # Local adaptive on MIP for XY detection, then per-Z local adaptive
        # with the SAME 2D local_bg map but a relaxed threshold for Z-connectivity.
        from scipy.ndimage import uniform_filter
        local_bg = uniform_filter(mip, size=LOCAL_BG_RADIUS_PX * 2 + 1)
        diff_mip = mip - local_bg
        # 2D MIP mask: where cells exist in XY
        mip_mask = (diff_mip > LOCAL_THRESH_OFFSET) & (mip > abs_thresh)
        print(f"  Local adaptive threshold on MIP: bg_radius={LOCAL_BG_RADIUS_PX}px, "
              f"offset={int(LOCAL_THRESH_OFFSET*65535)} u16, floor={MIN_THRESH_UINT16} u16")
        print(f"  MIP mask pixels: {mip_mask.sum()} ({mip_mask.sum()/mip_mask.size*100:.2f}%)")

        # Save threshold debug image
        if debug_dir and debug_stem:
            raw_mip = volume.max(axis=0)
            p1, p995 = np.percentile(raw_mip, [1, 99.5])
            norm = np.clip((raw_mip - p1) / (p995 - p1 + 1e-10), 0, 1)
            gray = (norm * 255).astype(np.uint8)
            h, w = gray.shape
            p1_img = np.stack([gray, gray, gray], axis=-1)
            diff_u16 = diff_mip * 65535
            d_max = max(abs(np.nanmin(diff_u16)), abs(np.nanmax(diff_u16)), 20)
            diff_norm = np.clip(diff_u16 / d_max, -1, 1)
            p2_img = np.zeros((h, w, 3), dtype=np.uint8)
            p2_img[:,:,0] = np.clip(diff_norm * 255, 0, 255).astype(np.uint8)
            p2_img[:,:,2] = np.clip(-diff_norm * 255, 0, 255).astype(np.uint8)
            p3_img = np.stack([gray, gray, gray], axis=-1)
            p3_img[mip_mask, 1] = 255
            p3_img[mip_mask, 0] = (p3_img[mip_mask, 0] * 0.3).astype(np.uint8)
            p3_img[mip_mask, 2] = 50
            canvas = np.concatenate([p1_img, p2_img, p3_img], axis=1)
            dbg_path = os.path.join(debug_dir, f'{debug_stem}_thresh_debug.png')
            Image.fromarray(canvas).save(dbg_path)
            print(f"  Saved: {dbg_path}")
            del raw_mip

        # Broadcast to 3D: for each voxel, require both:
        #   (a) its XY position passes the MIP mask (cell is present at this XY)
        #   (b) the voxel itself exceeds local_bg + offset (not just a dim Z-slice)
        binary = np.zeros(smooth.shape, dtype=bool)
        for z in range(nz):
            slice_diff = smooth[z] - local_bg
            binary[z] = mip_mask & (slice_diff > LOCAL_THRESH_OFFSET) & (smooth[z] > abs_thresh)

        del local_bg, diff_mip, mip_mask
    else:
        # Global: percentile threshold (original method)
        rng = np.random.default_rng(42)
        sample = smooth.ravel()[rng.integers(0, smooth.size, size=500000)]
        pct_thresh = np.percentile(sample, top_percentile)
        thresh = max(pct_thresh, abs_thresh)
        used = 'percentile' if pct_thresh >= abs_thresh else 'absolute floor'
        print(f"  Threshold: {thresh:.6f} ({int(thresh*65535)} uint16) [{used}] "
              f"(p{top_percentile}={int(pct_thresh*65535)}, floor={MIN_THRESH_UINT16})")
        binary = smooth > thresh

    del mip

    # Cleanup — fill holes and remove tiny noise (Rust, 3D)
    binary = ip3d.binary_fill_holes_3d(binary.astype(np.uint8)).astype(bool)
    binary = ip3d.remove_small_objects_3d(binary.astype(np.uint8), 100).astype(bool)
    binary_pct = binary.sum() / binary.size * 100
    print(f"  Binary voxels: {binary.sum()} ({binary_pct:.2f}%)")

    if binary.sum() == 0 or binary_pct > MAX_BINARY_PCT:
        if binary_pct > MAX_BINARY_PCT:
            print(f"  Binary too high ({binary_pct:.0f}%) — skipping")
        return np.zeros(volume.shape, dtype=np.int32)

    # Step 4: 3D watershed to split touching cells (Rust EDT + watershed)
    dist = ip3d.distance_transform_edt_3d(binary.astype(np.uint8), spacing_um[0], spacing_um[1], spacing_um[2])
    min_dist_vox = max(1, int(round(min_distance_um / min(spacing_um))))

    # Prefer nucleus-based seeds: broadcast 2D nucleus labels to 3D, mask by binary.
    # Cells without any nucleus overlap fall through to peak_local_max / connected
    # components (they'll be filtered out later by the nucleus-overlap check).
    seeds_3d = None
    if nucleus_seeds_2d is not None and nucleus_seeds_2d.any():
        # Broadcast 2D labels to 3D, keep labels only where cell binary is true
        seeds_3d = np.zeros(binary.shape, dtype=np.int32)
        for z in range(nz):
            seeds_3d[z] = np.where(binary[z], nucleus_seeds_2d, 0)
        n_seeded = len(np.unique(seeds_3d)) - 1  # minus background 0
        if n_seeded > 0:
            print(f"  Watershed seeds (from nuclei): {n_seeded} distinct nuclei")
            labels_3d = ip3d.watershed_3d(-dist, seeds_3d, binary.astype(np.uint8))

            # Cells with no nucleus inside them get no label after watershed —
            # relabel those remaining binary regions with fresh IDs so they survive
            # to be filtered later by the per-cell nucleus-overlap check.
            unlabeled = binary & (labels_3d == 0)
            if unlabeled.any():
                next_id = int(labels_3d.max()) + 1
                extra_labels, n_extra = ip3d.label_3d(unlabeled.astype(np.uint8))
                extra_labels = extra_labels.astype(np.int32)
                extra_labels[extra_labels > 0] += (next_id - 1)
                labels_3d = np.where(extra_labels > 0, extra_labels, labels_3d)
                print(f"  {n_extra} binary regions without any nucleus kept as separate labels")
        else:
            seeds_3d = None  # force fallback below

    if seeds_3d is None:
        coords = peak_local_max(dist, min_distance=min_dist_vox, labels=binary)
        print(f"  Watershed seeds: {len(coords)} (min_distance={min_distance_um}µm → {min_dist_vox}vox)")
        if len(coords) > 0:
            seed_mask = np.zeros(dist.shape, dtype=bool)
            seed_mask[tuple(coords.T)] = True
            seeds, _ = ip3d.label_3d(seed_mask.astype(np.uint8))
            labels_3d = ip3d.watershed_3d(-dist, seeds, binary.astype(np.uint8))
        else:
            labels_3d, _ = ip3d.label_3d(binary.astype(np.uint8))

    # Step 5: Trim Z-extent per cell — keep only slices with real signal
    for lbl in range(1, labels_3d.max() + 1):
        lbl_mask = labels_3d == lbl
        if not lbl_mask.any():
            continue
        # Mean intensity per Z-slice within this cell's XY footprint
        z_means = np.array([volume[z][lbl_mask[z]].mean() if lbl_mask[z].any() else 0
                            for z in range(nz)])
        active_z = np.where(lbl_mask.any(axis=(1, 2)))[0]
        if len(active_z) == 0:
            continue
        # Keep only Z-slices where mean > cell's median Z-mean
        cell_z_means = z_means[active_z]
        z_cutoff = np.percentile(cell_z_means, 25)  # keep top 75% of Z-slices
        for z in active_z:
            if z_means[z] < z_cutoff:
                labels_3d[z][lbl_mask[z]] = 0

    # Compact label IDs after Z-trimming without merging watershed-separated regions.
    # Plain connected-components (ip3d.label_3d on the binary mask) would merge
    # cells split by watershed whenever they physically touch, defeating the split.
    existing = np.unique(labels_3d)
    existing = existing[existing > 0]
    if len(existing) > 0 and (existing != np.arange(1, len(existing) + 1)).any():
        relabel = np.zeros(int(existing.max()) + 1, dtype=np.int32)
        for new_id, old_id in enumerate(existing, start=1):
            relabel[old_id] = new_id
        labels_3d = relabel[labels_3d]

    # Step 6: Remove too small / too large
    removed = 0
    for r in regionprops(labels_3d):
        if r.area < min_size or r.area > max_size:
            labels_3d[labels_3d == r.label] = 0
            removed += 1
    if removed:
        print(f"    Removed {removed} objects by size filter (min={min_size}, max={max_size})")

    # Compact label IDs after size filter, again preserving watershed splits
    existing = np.unique(labels_3d)
    existing = existing[existing > 0]
    if len(existing) > 0 and (existing != np.arange(1, len(existing) + 1)).any():
        relabel = np.zeros(int(existing.max()) + 1, dtype=np.int32)
        for new_id, old_id in enumerate(existing, start=1):
            relabel[old_id] = new_id
        labels_3d = relabel[labels_3d]
    n_final = int(labels_3d.max())

    # Report final cells
    for r in regionprops(labels_3d):
        z_present = np.where((labels_3d == r.label).any(axis=(1, 2)))[0]
        print(f"    Cell {r.label}: {r.area} voxels, Z={z_present.min()}–{z_present.max()} ({len(z_present)} slices)")

    return labels_3d


# ── Detect substrate direction ────────────────────────────────────────────
def detect_substrate_direction(green, labels):
    """Determine which Z-direction is "substrate" (has green signal) vs "air".

    Compares total green intensity above vs below the cells' Z-center.
    Returns 'down' (higher Z = substrate) or 'up' (lower Z = substrate).
    """
    cell_mask = labels > 0
    if not cell_mask.any():
        return 'down'  # default

    # Find Z-center of all cells
    z_coords = np.where(cell_mask.any(axis=(1, 2)))[0]
    z_center = int(np.mean(z_coords))

    # Compare green signal above vs below cell center
    green_above = green[:z_center].mean() if z_center > 0 else 0
    green_below = green[z_center + 1:].mean() if z_center < green.shape[0] - 1 else 0

    direction = 'down' if green_below >= green_above else 'up'
    print(f"  Substrate detection: green above z{z_center}={green_above:.6f}, "
          f"below={green_below:.6f} → substrate is '{direction}'")
    return direction


# ── 3D Label expansion ───────────────────────────────────────────────────
def expand_labels_3d(labels, expand_um, spacing_um, substrate_dir='down'):
    """Expand each label by expand_um, only toward the substrate side.

    Args:
        labels: (Z, Y, X) int32
        expand_um: expansion radius in µm
        spacing_um: (z_um, y_um, x_um)
        substrate_dir: 'down' (higher Z = substrate) or 'up' (lower Z)
    """
    if labels.max() == 0:
        return labels.copy()

    # Full expansion first (Rust EDT + watershed — single EDT call instead of 2)
    bg_mask = (labels == 0).astype(np.uint8)
    dist_from_labels = ip3d.distance_transform_edt_3d(bg_mask, spacing_um[0], spacing_um[1], spacing_um[2])
    expanded_mask = (labels > 0) | (dist_from_labels <= expand_um)
    labels_expanded = ip3d.watershed_3d(dist_from_labels, labels, expanded_mask.astype(np.uint8))

    # Remove expansion on the air side for each cell
    for lbl in range(1, labels.max() + 1):
        orig_mask = labels == lbl
        if not orig_mask.any():
            continue

        # Find Z-range of original cell
        z_present = np.where(orig_mask.any(axis=(1, 2)))[0]
        z_min, z_max = z_present.min(), z_present.max()

        # Remove expanded voxels on the air side
        exp_mask = labels_expanded == lbl
        if substrate_dir == 'down':
            # Air is above (lower Z), keep only expansion at z >= z_min
            air_zone = exp_mask.copy()
            air_zone[z_min:, :, :] = False  # keep everything from z_min down
            labels_expanded[air_zone] = 0
        else:
            # Air is below (higher Z), keep only expansion at z <= z_max
            air_zone = exp_mask.copy()
            air_zone[:z_max + 1, :, :] = False  # keep everything up to z_max
            labels_expanded[air_zone] = 0

    return labels_expanded, dist_from_labels


# ── 3D Artifact detection ─────────────────────────────────────────────────
def detect_artifacts_3d(green, spacing_um, blur_sigma_um=30.0, min_size_voxels=50000,
                        percentile=97.0):
    """Detect large bright blobs in 3D green volume.

    Uses anisotropy-aware Gaussian blur, then thresholds and keeps only
    large connected components. Returns a 3D bool mask.
    """
    z_um, y_um, x_um = spacing_um
    sigma_voxels = (blur_sigma_um / z_um, blur_sigma_um / y_um, blur_sigma_um / x_um)
    blurred = ip3d.gaussian_filter_3d(green, sigma_voxels[0], sigma_voxels[1], sigma_voxels[2])
    thresh = np.percentile(blurred, percentile)
    raw_mask = blurred > thresh
    art_labels, _ = ip3d.label_3d(raw_mask.astype(np.uint8))
    mask = np.zeros_like(raw_mask)
    for r in regionprops(art_labels):
        if r.area >= min_size_voxels:
            mask[art_labels == r.label] = True
            print(f"    Artifact: {r.area} voxels, Z={r.bbox[0]}–{r.bbox[3]}, "
                  f"centroid=({r.centroid[0]:.0f}, {r.centroid[1]:.0f}, {r.centroid[2]:.0f})")
    return mask


# ── Per-Z in-sample detection ─────────────────────────────────────────────
def _in_sample_mask_3d(green, well_min=WELL_MIN_UINT16, blur_sigma=30):
    """Batched 3D version of _in_sample_mask_z — blurs the whole volume in one call.

    Returns bool array of shape green.shape.
    """
    # Use Rust-parallel 3D gaussian; sigma_z=0.01 ≈ no Z blur, much faster than scipy
    blurred = ip3d.gaussian_filter_3d(green, 0.01, blur_sigma, blur_sigma)
    dark_all = blurred < (well_min / 65535.0)
    del blurred

    nz, ny, nx = green.shape
    result = np.empty(green.shape, dtype=bool)
    for z in range(nz):
        dark = dark_all[z]
        if not dark.any():
            result[z] = True
            continue
        dark_labels, _ = ip2d.label_2d(dark.astype(np.uint8))
        border_labels = set()
        border_labels |= set(dark_labels[0, :].ravel())
        border_labels |= set(dark_labels[-1, :].ravel())
        border_labels |= set(dark_labels[:, 0].ravel())
        border_labels |= set(dark_labels[:, -1].ravel())
        border_labels.discard(0)
        # Vectorized: LUT to mark border-touching labels
        if border_labels:
            max_lbl = int(dark_labels.max())
            outside_lut = np.zeros(max_lbl + 1, dtype=bool)
            for lbl in border_labels:
                if lbl <= max_lbl:
                    outside_lut[lbl] = True
            result[z] = ~outside_lut[dark_labels]
        else:
            result[z] = np.ones((ny, nx), dtype=bool)
    return result


def _in_sample_mask_z(green_z, well_min=WELL_MIN_UINT16, blur_sigma=30):
    """Detect in-sample region for a single Z-slice (exclude dark air at edges).

    Returns bool mask (True = in sample).
    """
    blurred = gaussian_filter_2d(green_z, sigma=blur_sigma)
    dark = blurred < (well_min / 65535.0)
    if not dark.any():
        return np.ones(green_z.shape, dtype=bool)
    dark_labels, _ = ip2d.label_2d(dark.astype(np.uint8))
    h, w = green_z.shape
    border_labels = set()
    border_labels |= set(dark_labels[0, :].ravel())
    border_labels |= set(dark_labels[-1, :].ravel())
    border_labels |= set(dark_labels[:, 0].ravel())
    border_labels |= set(dark_labels[:, -1].ravel())
    border_labels.discard(0)
    outside = np.zeros(green_z.shape, dtype=bool)
    for lbl in border_labels:
        outside[dark_labels == lbl] = True
    return ~outside


# ── Z-profile + bleach correction ─────────────────────────────────────────
def _compute_z_bg_means(green, cell_mask_3d, artifact_mask_2d, void_mask_3d=None,
                        in_sample_3d=None):
    """Compute per-Z BG means using per-Z in-sample masking.

    If `in_sample_3d` is provided, it is used instead of recomputing per-Z
    (big speedup when called multiple times on the same green volume).

    Returns (bg_means, n_bg_pixels) arrays of length nz. Values in uint16 scale.
    """
    nz = green.shape[0]
    bg_means = np.full(nz, np.nan)
    n_bg_pixels = np.zeros(nz, dtype=int)

    for z in range(nz):
        in_sample = in_sample_3d[z] if in_sample_3d is not None else _in_sample_mask_z(green[z])
        bg_z = in_sample & ~cell_mask_3d[z] & ~artifact_mask_2d
        if void_mask_3d is not None:
            bg_z = bg_z & ~void_mask_3d[z]
        n_bg = bg_z.sum()
        n_bg_pixels[z] = n_bg
        if n_bg > 0:
            bg_means[z] = green[z][bg_z].mean() * 65535
    return bg_means, n_bg_pixels


def _compute_bleach_correction(bg_means):
    """Compute per-Z correction factors from BG means.

    Uses the first real layer (first above air cutoff) as reference,
    skipping initial bubble/air layers that have no signal.

    Returns correction_factors array of length nz.
    """
    nz = len(bg_means)
    correction_factors = np.ones(nz)

    # Find valid range (above 50% of peak — exclude air/bubble slices)
    peak_bg = np.nanmax(bg_means)
    if np.isnan(peak_bg) or peak_bg <= 0:
        return correction_factors

    air_cutoff = max(250, peak_bg * 0.5)
    valid_z = np.where(bg_means > air_cutoff)[0]
    if len(valid_z) == 0:
        return correction_factors

    z_first = valid_z[0]   # first real layer (skip bubbles)
    z_last = valid_z[-1] + 1

    # Smooth BG profile (only valid range)
    smooth_window = max(3, (z_last - z_first) // 8)
    bg_smooth = bg_means[z_first:z_last].copy()
    valid = ~np.isnan(bg_smooth)
    if valid.sum() < 3:
        return correction_factors
    bg_smooth[valid] = uniform_filter1d(bg_smooth[valid], size=smooth_window)

    # Reference = first real layer
    ref = bg_smooth[0]
    if ref <= 0 or np.isnan(ref):
        return correction_factors

    if z_first > 0:
        print(f"    Bleach correction: skipping {z_first} bubble/air layers, "
              f"ref z={z_first} (bg={ref:.0f})")

    # Bubble layers before z_first: set correction to 1.0 (no correction,
    # these layers have no real signal anyway)
    for z in range(z_first, z_last):
        z_local = z - z_first
        if bg_smooth[z_local] > 0 and not np.isnan(bg_smooth[z_local]):
            correction_factors[z] = ref / bg_smooth[z_local]

    # Air slices beyond z_last: use last valid correction
    if z_last < nz:
        correction_factors[z_last:] = correction_factors[z_last - 1]

    correction_factors = np.clip(correction_factors, 0.5, 5.0)
    return correction_factors


def _save_z_profile(green, cell_mask_3d, artifact_mask_2d, stem, out_dir, void_mask_3d=None,
                    in_sample_3d=None):
    """Save per-Z-layer BG green mean (with per-Z in-sample masking) to CSV."""
    bg_means, n_bg_pixels = _compute_z_bg_means(green, cell_mask_3d, artifact_mask_2d, void_mask_3d,
                                                in_sample_3d=in_sample_3d)
    correction = _compute_bleach_correction(bg_means)

    nz = green.shape[0]
    rows = []
    for z in range(nz):
        rows.append({
            'file': stem,
            'z_index': z,
            'bg_mean_green': round(bg_means[z], 1) if not np.isnan(bg_means[z]) else 0,
            'n_bg_pixels': int(n_bg_pixels[z]),
            'correction_factor': round(correction[z], 4),
        })
    zp_path = os.path.join(out_dir, f'{stem}_z_profile.csv')
    pd.DataFrame(rows).to_csv(zp_path, index=False)
    print(f"  Saved: {zp_path}")
    print(f"  Bleach correction: {correction.min():.3f}x – {correction.max():.3f}x")
    return correction


# ── Edge artifact + brightness helpers ────────────────────────────────────
def _blob_touches_edge(region, img_shape, margin=ARTIFACT_EDGE_MARGIN_PX):
    """Check if a regionprops blob touches the image border (within margin px)."""
    h, w = img_shape
    min_r, min_c, max_r, max_c = region.bbox
    return (min_r <= margin or min_c <= margin or
            max_r >= h - margin or max_c >= w - margin)


def _check_bright_image(green_mip, outside_well=None):
    """Check if the green MIP is abnormally bright overall.

    Returns (is_bright, median_u16, p90_u16) where is_bright flags suspect images.
    """
    if outside_well is not None and outside_well.any():
        pixels = green_mip[~outside_well]
    else:
        pixels = green_mip.ravel()

    if len(pixels) == 0:
        return False, 0, 0

    med_u16 = np.median(pixels) * 65535
    p90_u16 = np.percentile(pixels, 90) * 65535
    is_bright = med_u16 > BRIGHT_IMAGE_MEDIAN_UINT16 or p90_u16 > BRIGHT_IMAGE_P90_UINT16
    return is_bright, med_u16, p90_u16


def _detect_artifacts_2d(green_mip, labels_mip_orig=None):
    """Shared 2D artifact detection used by both cell and no-cell paths.

    Always uses blob-based detection. Edge blobs bypass eccentricity filter.
    Brightness check is separate (flag-only, does not mask).

    Returns artifact_mask_2d (bool).
    """
    h, w = green_mip.shape

    # ── Standard blob-based artifact detection ────────────────────────
    # Use whichever threshold is LOWER (catches more): percentile or absolute
    # On normal images, percentile works. On bright images where percentile
    # is too high, the absolute threshold catches obvious artifacts.
    blurred = gaussian_filter_2d(green_mip, sigma=ARTIFACT_BLUR_PX)
    pct_thresh = np.percentile(blurred, ARTIFACT_PERCENTILE)
    abs_thresh = ARTIFACT_ABS_UINT16 / 65535.0
    art_thresh = min(pct_thresh, abs_thresh)
    used = 'percentile' if pct_thresh <= abs_thresh else 'absolute'
    print(f"    Artifact threshold: {int(art_thresh*65535)} u16 [{used}] "
          f"(p{ARTIFACT_PERCENTILE}={int(pct_thresh*65535)}, abs={ARTIFACT_ABS_UINT16})")
    art_raw = blurred > art_thresh
    art_labels, _ = ip2d.label_2d(art_raw.astype(np.uint8))

    artifact_mask_2d = np.zeros(green_mip.shape, dtype=bool)
    for r in regionprops(art_labels):
        if r.area < ARTIFACT_MIN_SIZE_2D:
            continue

        # Cell-overlap check (only if labels provided)
        if labels_mip_orig is not None:
            blob_mask = art_labels == r.label
            cell_overlap = (blob_mask & (labels_mip_orig > 0)).sum()
            if r.area > 0 and cell_overlap / r.area > ARTIFACT_CELL_ONLY_SKIP:
                print(f"    Blob skipped (cell signal): {r.area} px, "
                      f"{cell_overlap/r.area*100:.0f}% cell overlap, ecc={r.eccentricity:.2f}")
                continue

        # Edge artifacts: skip eccentricity filter (edge blobs can be any shape)
        touches_edge = _blob_touches_edge(r, (h, w))
        if not touches_edge and r.eccentricity > ARTIFACT_MAX_ECCENTRICITY:
            print(f"    Blob skipped (not circular, interior): {r.area} px, ecc={r.eccentricity:.2f}")
            continue

        # Accepted as artifact — adaptive halo expansion
        cy, cx = r.centroid
        equiv_radius = np.sqrt(r.area / np.pi)

        bg_mean = np.median(green_mip)
        yy, xx = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        expanded_radius = equiv_radius * 1.2
        for mult in np.arange(1.5, 4.1, 0.5):
            r_inner = equiv_radius * (mult - 0.5)
            r_outer = equiv_radius * mult
            ring = (dist_from_center >= r_inner) & (dist_from_center < r_outer)
            if ring.sum() == 0:
                break
            ring_mean = green_mip[ring].mean()
            if ring_mean <= bg_mean * 1.2:
                break
            expanded_radius = equiv_radius * mult

        circle = dist_from_center <= expanded_radius
        artifact_mask_2d |= circle
        edge_tag = " [EDGE]" if touches_edge else ""
        print(f"    Artifact{edge_tag}: {r.area} px, ecc={r.eccentricity:.2f}, "
              f"r={equiv_radius:.0f}→{expanded_radius:.0f}px, "
              f"centroid=({cy:.0f}, {cx:.0f})")

    return artifact_mask_2d


# ── Process one file ─────────────────────────────────────────────────────
def process_one(ims_path, out_dir, status_cb=None):
    """Process a single IMS file. status_cb(msg) updates live progress if provided."""
    def _status(msg):
        if status_cb:
            status_cb(msg)
        else:
            print(msg)

    stem = Path(ims_path).stem
    csv_path = os.path.join(out_dir, f'{stem}.csv')
    corr_csv_path = os.path.join(out_dir, f'{stem}_corrected.csv')
    if os.path.exists(corr_csv_path):
        _status("SKIP (corrected csv exists)")
        return

    # Load metadata + both channels (green kept for void detection + measurements)
    _status("Reading IMS...")
    try:
        result = read_ims(ims_path)
    except Exception as e:
        _status(f"ERROR: {e}")
        return
    yellow = result['channels'][1]
    green = result['channels'][0]
    # Nucleus channel (3rd channel, optional — some datasets have only 2 channels)
    if len(result['channels']) > NUCLEUS_CH_IDX:
        nucleus = result['channels'][NUCLEUS_CH_IDX]
        has_nucleus_ch = True
    else:
        nucleus = None
        has_nucleus_ch = False
        print(f"  No nucleus channel (file has {len(result['channels'])} channels)")
    scale = result['scale_um']  # (x_um, y_um, z_um)
    spacing = (scale[2], scale[1], scale[0])  # (z_um, y_um, x_um)
    shape = result['shape']
    ext_min = result.get('ext_min', [0, 0, 0])  # [x_min, y_min, z_min] in µm
    ext_max = result.get('ext_max', [0, 0, 0])
    rec_date = result.get('recording_date', '')
    print(f"  Shape: {shape}, Spacing(Z,Y,X)µm: ({spacing[0]:.2f}, {spacing[1]:.4f}, {spacing[2]:.4f})")
    print(f"  Tile extent: X=[{ext_min[0]:.1f}, {ext_max[0]:.1f}], Y=[{ext_min[1]:.1f}, {ext_max[1]:.1f}]")

    # Check if yellow channel has real signal
    rng = np.random.default_rng(42)
    sample = yellow.ravel()[rng.integers(0, yellow.size, size=100000)]
    yellow_p999 = np.percentile(sample, 99.9) * 65535
    print(f"  Yellow p99.9: {yellow_p999:.0f} uint16 (min signal: {MIN_SIGNAL_UINT16})")

    # Detect nuclei first (2D MIP) so they can seed cell watershed to split linked cells
    if has_nucleus_ch:
        _status("Detecting nuclei (2D MIP)...")
        nucleus_mip = nucleus.max(axis=0)
        nucleus_labels_2d, nucleus_sizes = _detect_nuclei_2d(nucleus_mip, spacing_um=spacing)
    else:
        nucleus_labels_2d = None
        nucleus_sizes = None
        nucleus_mip = None

    if yellow_p999 < MIN_SIGNAL_UINT16:
        print(f"  No yellow signal — BG only")
        n_cells = 0
        labels_3d = np.zeros(yellow.shape, dtype=np.int32)
    else:
        _status("Segmenting cells...")
        labels_3d = segment_3d(yellow, spacing, SIGMA_UM, MIN_SIZE_VOXELS, MAX_SIZE_VOXELS,
                               min_distance_um=MIN_DISTANCE_UM,
                               top_percentile=TOP_PERCENTILE,
                               debug_dir=out_dir, debug_stem=stem,
                               nucleus_seeds_2d=nucleus_labels_2d)
        n_cells = labels_3d.max()
    print(f"  Cells found: {n_cells}")
    _status(f"Found {n_cells} cells")

    # Per-layer 3D void detection using both channels
    _status("Detecting voids per layer...")
    void_mask_3d = _detect_dark_voids_3d(yellow, green_vol=green)

    # Outside-well from MIP: this boundary is fixed spatially (doesn't change per Z).
    # Use _detect_outside_well (border-touching dark regions, threshold=250 u16)
    # + _detect_dark_voids (internal bubbles/voids) from MIPs.
    # Broadcast to ALL layers so cells outside the well are always excluded,
    # even on layers where individual detection fails (dim/noisy z=6-7 etc.)
    yellow_mip_tmp = yellow.max(axis=0)
    green_mip_tmp = green.max(axis=0)
    outside_well_2d = _detect_outside_well(green_mip_tmp, WELL_MIN_UINT16)
    mip_voids_2d = _detect_dark_voids(yellow_mip_tmp, green_mip_tmp)
    combined_2d = outside_well_2d | mip_voids_2d
    # Well-border exclusion zone: dilate outside-well to catch edge artifacts
    from skimage.morphology import dilation, disk
    if outside_well_2d.any():
        well_border_zone = dilation(outside_well_2d.astype(np.uint8), disk(WELL_BORDER_MARGIN_PX)).astype(bool)
        well_border_zone = well_border_zone & ~outside_well_2d  # just the margin ring
    else:
        well_border_zone = np.zeros_like(outside_well_2d)

    if combined_2d.any():
        nz_vol = yellow.shape[0]
        for z in range(nz_vol):
            void_mask_3d[z] |= combined_2d
        ow_pct = outside_well_2d.sum() / outside_well_2d.size * 100
        mv_pct = mip_voids_2d.sum() / mip_voids_2d.size * 100
        print(f"  Outside-well (MIP→all layers): {ow_pct:.1f}%, MIP voids: {mv_pct:.1f}%")

    # For composite visualization: use the MIP-based mask (not per-layer green)
    void_mask_2d_spatial = combined_2d

    # Save MIP and diagnostic void image (shows green channel with void overlay)
    yellow_mip = yellow_mip_tmp  # reuse from above
    _save_void_layers_image(green, void_mask_3d, stem, out_dir)

    # Save nucleus diagnostic (detection was done earlier, before segment_3d)
    if has_nucleus_ch:
        _save_nucleus_diagnostic(yellow_mip_tmp, nucleus_mip, nucleus_labels_2d, stem, out_dir)

    del yellow, yellow_mip_tmp, green_mip_tmp, result
    if has_nucleus_ch:
        del nucleus
    gc.collect()

    if n_cells == 0:
        _status("BG only — measuring...")
        print(f"  No cells found — saving BG-only")
        # green already loaded from read_ims() above
        green_mip_for_art = green.max(axis=0)

        # Brightness check (use spatial-only mask, not dead-layer mask)
        is_bright, med_u16, p90_u16 = _check_bright_image(green_mip_for_art, void_mask_2d_spatial)

        # Artifact detection on green MIP (brightness-aware)
        artifact_mask_2d = _detect_artifacts_2d(green_mip_for_art)
        artifact_mask_3d = np.broadcast_to(artifact_mask_2d[np.newaxis, :, :], green.shape)
        print(f"  Artifacts: {artifact_mask_2d.sum()} px (2D)")

        # BG = entire volume minus artifacts, minus per-layer voids
        bg_3d = ~artifact_mask_3d & ~void_mask_3d
        bg_pixels = green[bg_3d] * 65535
        void_pct_3d = void_mask_3d.sum() / void_mask_3d.size * 100
        print(f"  Void mask (3D): {void_pct_3d:.1f}% excluded")
        voxel_vol = spacing[0] * spacing[1] * spacing[2]
        bg_row = {
            'file': stem, 'cell': 'BG',
            'bright_image': is_bright,
            'green_median_u16': round(med_u16, 0),
            'green_p90_u16': round(p90_u16, 0),
            'orig_voxels': 0, 'orig_volume_um3': 0,
            'expanded_voxels': int(bg_3d.sum()),
            'expanded_volume_um3': round(bg_3d.sum() * voxel_vol, 1),
            'z_start': '', 'z_end': '', 'n_z_slices': '',
            'mean_green': round(bg_pixels.mean(), 1) if bg_3d.sum() > 0 else 0,
            'median_green': round(np.median(bg_pixels), 1) if bg_3d.sum() > 0 else 0,
        }
        for gt in GREEN_THRESHOLDS:
            vox_above = (bg_pixels >= gt).sum()
            bg_row[f'pct_above_{gt}'] = round(vox_above / bg_3d.sum() * 100, 2) if bg_3d.sum() > 0 else 0
        rows = [bg_row]
        # Uncorrected CSV disabled — the Streamlit app only consumes _corrected.csv.
        # If you need the uncorrected data back, uncomment:
        # df = pd.DataFrame(rows)
        # df.to_csv(csv_path, index=False)
        # print(f"  Saved: {csv_path}")

        # Z-profile for bleach analysis (no cells to exclude in this path)
        cell_mask_empty = np.zeros(green.shape, dtype=bool)
        correction = _save_z_profile(green, cell_mask_empty, artifact_mask_2d, stem, out_dir, void_mask_3d)

        # Corrected BG-only CSV (in-place)
        nz_g = green.shape[0]
        green *= correction[:nz_g, np.newaxis, np.newaxis]
        green_corrected = green
        bg_pixels_c = green_corrected[bg_3d] * 65535
        corr_bg_row = dict(bg_row)  # copy uncorrected row
        corr_bg_row['mean_green'] = round(bg_pixels_c.mean(), 1) if bg_3d.sum() > 0 else 0
        corr_bg_row['median_green'] = round(np.median(bg_pixels_c), 1) if bg_3d.sum() > 0 else 0
        for gt in GREEN_THRESHOLDS:
            vox_above = (bg_pixels_c >= gt).sum()
            corr_bg_row[f'pct_above_{gt}'] = round(vox_above / bg_3d.sum() * 100, 2) if bg_3d.sum() > 0 else 0
        corr_csv_path = os.path.join(out_dir, f'{stem}_corrected.csv')
        pd.DataFrame([corr_bg_row]).to_csv(corr_csv_path, index=False)
        print(f"  Saved: {corr_csv_path}")
        del green_corrected; gc.collect()

        # Composite image (no cells, just red + green)
        green_mip = green.max(axis=0)
        DISPLAY_RED_LO, DISPLAY_RED_HI = 150 / 65535, 250 / 65535
        DISPLAY_GREEN_LO, DISPLAY_GREEN_HI = 200 / 65535, 4000 / 65535
        y_norm = np.clip((yellow_mip - DISPLAY_RED_LO) / (DISPLAY_RED_HI - DISPLAY_RED_LO + 1e-10), 0, 1)
        g_norm = np.clip((green_mip - DISPLAY_GREEN_LO) / (DISPLAY_GREEN_HI - DISPLAY_GREEN_LO + 1e-10), 0, 1)
        h, w = y_norm.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[:, :, 0] = y_norm
        rgb[:, :, 1] = g_norm
        if nucleus_mip is not None:
            b_norm = np.clip(
                (nucleus_mip - DISPLAY_NUCLEUS_LO) /
                (DISPLAY_NUCLEUS_HI - DISPLAY_NUCLEUS_LO + 1e-10),
                0, 1,
            )
            rgb[:, :, 2] = b_norm
        # Draw void boundary (spatial voids only — not dead layers)
        if void_mask_2d_spatial.any():
            from skimage.morphology import dilation, disk
            well_bd = find_boundaries(void_mask_2d_spatial.astype(np.int32), mode='thick')
            well_bd = dilation(well_bd, disk(2))  # thicken contour
            rgb[well_bd] = [1, 1, 1]
            rgb[void_mask_2d_spatial] *= 0.15
        # Dim artifact regions
        if artifact_mask_2d.any():
            # Semi-transparent red tint: blend 30% red over original
            rgb[artifact_mask_2d, 0] = rgb[artifact_mask_2d, 0] * 0.5 + 0.3
            rgb[artifact_mask_2d, 1] *= 0.5
            rgb[artifact_mask_2d, 2] *= 0.5
            art_bd = find_boundaries(artifact_mask_2d.astype(np.int32), mode='thick')
            rgb[art_bd] = [1, 0, 1]  # magenta contour
        img_path = os.path.join(out_dir, f'{stem}_composite.png')
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(img_path)
        print(f"  Saved: {img_path}")

        del green; gc.collect()
        return

    # Save original labels
    labels_orig = labels_3d.copy()

    # green already loaded from read_ims() above; void_mask_3d already includes dead layers

    # Substrate direction — hardcoded: substrate is 'up' (lower Z index)
    substrate_dir = 'up'

    # Expand labels in 3D, only toward substrate side
    _status(f"Expanding labels ({EXPAND_UM}um)...")
    labels_expanded, dist_from_labels = expand_labels_3d(labels_3d, EXPAND_UM, spacing, substrate_dir)
    del labels_3d; gc.collect()
    print(f"  Expanded labels by {EXPAND_UM}µm (toward '{substrate_dir}'), steps: {EXPAND_STEPS}")

    # 2D artifact detection on green MIP, then broadcast to 3D
    green_mip_for_art = green.max(axis=0)
    labels_mip_orig = labels_orig.max(axis=0)
    outside_well_pre = _detect_outside_well(green_mip_for_art, WELL_MIN_UINT16)
    is_bright, med_u16, p90_u16 = _check_bright_image(green_mip_for_art, outside_well_pre)
    artifact_mask_2d = _detect_artifacts_2d(green_mip_for_art, labels_mip_orig)
    print(f"  Artifacts: {artifact_mask_2d.sum()} px (2D)")
    artifact_mask_3d = np.broadcast_to(artifact_mask_2d[np.newaxis, :, :], green.shape)

    # void_mask_3d already computed per-layer above (spatial + dead green layers)
    # void_mask_2d_spatial for composite visualization (spatial voids only)

    # Per-cell stats — all in 3D
    _status(f"Measuring {n_cells} cells (uncorrected)...")
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    excluded_cells = {}  # cell_id → reason ('artifact', 'void', 'border', 'no_nucleus')
    rows = []
    for i in range(1, n_cells + 1):
        cell_3d = labels_expanded == i
        vox_total = cell_3d.sum()

        # Check artifact overlap on original (unexpanded) cell mask
        orig_3d = labels_orig == i
        orig_vox = orig_3d.sum()
        if orig_vox > 0 and artifact_mask_3d.any():
            overlap = (orig_3d & artifact_mask_3d).sum()
            overlap_pct = overlap / orig_vox * 100
            if overlap_pct > MAX_ARTIFACT_OVERLAP_PCT:
                excluded_cells[i] = 'artifact'
                print(f"  Cell {i} excluded: {overlap_pct:.0f}% artifact overlap")
                continue

        # Check void overlap — per-layer 3D void mask (bubbles at specific Z depths)
        if orig_vox > 0 and void_mask_3d.any():
            void_overlap = (orig_3d & void_mask_3d).sum()
            void_pct = void_overlap / orig_vox * 100
            if void_pct > 80:  # >80% of cell is in a void
                excluded_cells[i] = 'void'
                print(f"  Cell {i} excluded: {void_pct:.0f}% in dark void")
                continue

        # Check well-border proximity — bright line artifacts at well edge
        if orig_vox > 0 and well_border_zone.any():
            # Project cell to 2D and check overlap with border zone
            cell_2d = orig_3d.any(axis=0)
            border_overlap = (cell_2d & well_border_zone).sum()
            border_pct = border_overlap / cell_2d.sum() * 100
            if border_pct > 30:  # >30% of cell footprint in border zone
                excluded_cells[i] = 'border'
                print(f"  Cell {i} excluded: {border_pct:.0f}% in well-border zone")
                continue

        # Check nucleus overlap (2D MIP) — keep the cell if at least one nucleus has
        # ≥ MIN_NUCLEUS_OVERLAP_PCT of its pixels inside the cell's XY footprint.
        # If the file has a nucleus channel but NO nuclei were detected (or all were
        # dropped as oversized artifacts), reject every cell — there's no valid
        # nuclear signal to validate against.
        if has_nucleus_ch and orig_vox > 0:
            if not nucleus_sizes:
                excluded_cells[i] = 'no_nucleus'
                print(f"  Cell {i} excluded: no nuclei detected on this tile")
                continue
            cell_xy = orig_3d.any(axis=0)
            hits = nucleus_labels_2d[cell_xy]
            hits = hits[hits > 0]
            best_pct = 0.0
            if hits.size > 0:
                labels_hit, counts = np.unique(hits, return_counts=True)
                for lbl, cnt in zip(labels_hit, counts):
                    pct = cnt / nucleus_sizes[int(lbl)] * 100
                    if pct > best_pct:
                        best_pct = pct
            if best_pct < MIN_NUCLEUS_OVERLAP_PCT:
                excluded_cells[i] = 'no_nucleus'
                print(f"  Cell {i} excluded: best nucleus-in-cell {best_pct:.0f}% "
                      f"(<{MIN_NUCLEUS_OVERLAP_PCT}%)")
                continue

        # Mask out void voxels from green measurements
        cell_valid = cell_3d & ~void_mask_3d
        valid_vox = cell_valid.sum()
        cell_pixels = green[cell_valid] * 65535 if valid_vox > 0 else np.array([0.0])
        mean_all = cell_pixels.mean() if valid_vox > 0 else 0
        median_all = np.median(cell_pixels) if valid_vox > 0 else 0

        orig_vol_um3 = orig_vox * voxel_vol
        z_slices = np.where(orig_3d.any(axis=(1, 2)))[0]

        # Cell spatial info: bounding box + centroid in physical µm coordinates
        coords_zyx = np.argwhere(orig_3d)  # (N, 3) array of [z, y, x] voxel indices
        bbox_min_z, bbox_min_y, bbox_min_x = coords_zyx.min(axis=0)
        bbox_max_z, bbox_max_y, bbox_max_x = coords_zyx.max(axis=0)
        cent_z, cent_y, cent_x = coords_zyx.mean(axis=0)

        # Convert to physical coordinates (µm, absolute)
        # ext_min is [x, y, z], spacing is (z_um, y_um, x_um)
        z_um, y_um, x_um = spacing
        phys_cent_x = ext_min[0] + cent_x * x_um
        phys_cent_y = ext_min[1] + cent_y * y_um
        phys_cent_z = ext_min[2] + cent_z * z_um
        phys_bbox_min_x = ext_min[0] + bbox_min_x * x_um
        phys_bbox_max_x = ext_min[0] + bbox_max_x * x_um
        phys_bbox_min_y = ext_min[1] + bbox_min_y * y_um
        phys_bbox_max_y = ext_min[1] + bbox_max_y * y_um

        row = {
            'file': stem, 'cell': i,
            'bright_image': is_bright,
            'green_median_u16': round(med_u16, 0),
            'green_p90_u16': round(p90_u16, 0),
            'orig_voxels': int(orig_vox),
            'orig_volume_um3': round(orig_vol_um3, 1),
            'expanded_voxels': int(vox_total),
            'expanded_volume_um3': round(vox_total * voxel_vol, 1),
            'z_start': int(z_slices.min()),
            'z_end': int(z_slices.max()),
            'n_z_slices': int(z_slices.max() - z_slices.min() + 1),
            # Cell position in pixel coords
            'centroid_x_px': round(cent_x, 1),
            'centroid_y_px': round(cent_y, 1),
            'centroid_z_px': round(cent_z, 1),
            'bbox_min_x_px': int(bbox_min_x),
            'bbox_max_x_px': int(bbox_max_x),
            'bbox_min_y_px': int(bbox_min_y),
            'bbox_max_y_px': int(bbox_max_y),
            # Cell position in absolute physical coords (µm)
            'centroid_x_um': round(phys_cent_x, 1),
            'centroid_y_um': round(phys_cent_y, 1),
            'centroid_z_um': round(phys_cent_z, 1),
            'bbox_min_x_um': round(phys_bbox_min_x, 1),
            'bbox_max_x_um': round(phys_bbox_max_x, 1),
            'bbox_min_y_um': round(phys_bbox_min_y, 1),
            'bbox_max_y_um': round(phys_bbox_max_y, 1),
            # Tile info for deduplication
            'tile_ext_min_x': round(ext_min[0], 1),
            'tile_ext_min_y': round(ext_min[1], 1),
            'tile_ext_max_x': round(ext_max[0], 1),
            'tile_ext_max_y': round(ext_max[1], 1),
            'tile_recording_date': rec_date,
            'mean_green': round(mean_all, 1),
            'median_green': round(median_all, 1),
        }

        # Per-expansion-step green stats (using distance from original cell surface)
        for step_um in EXPAND_STEPS:
            step_mask = cell_3d & (dist_from_labels <= step_um) & ~void_mask_3d
            step_vox = step_mask.sum()
            if step_vox > 0:
                step_pixels = green[step_mask] * 65535
                row[f'mean_green_{step_um}um'] = round(step_pixels.mean(), 1)
                row[f'median_green_{step_um}um'] = round(np.median(step_pixels), 1)
                for gt in GREEN_THRESHOLDS:
                    vox_above = (step_pixels >= gt).sum()
                    row[f'pct_above_{gt}_at_{step_um}um'] = round(vox_above / step_vox * 100, 2)
                row.update(_skewness_metrics(step_pixels, 'cell', step_um))
            else:
                row[f'mean_green_{step_um}um'] = 0
                row[f'median_green_{step_um}um'] = 0
                for gt in GREEN_THRESHOLDS:
                    row[f'pct_above_{gt}_at_{step_um}um'] = 0
                row.update(_skewness_metrics(np.array([]), 'cell', step_um))

        rows.append(row)
    print(f"  Kept {n_cells - len(excluded_cells)}/{n_cells} cells")

    # Background in 3D — reuse per-layer void mask computed earlier
    void_pct_3d = void_mask_3d.sum() / void_mask_3d.size * 100
    print(f"  Dark void mask (3D): {void_pct_3d:.1f}% excluded")

    # All-cells mask at each step (union of all non-excluded cells within step distance)
    all_cells_expanded = (labels_expanded > 0)
    for ex_i in excluded_cells:
        all_cells_expanded = all_cells_expanded & (labels_expanded != ex_i)
    exclude_3d = artifact_mask_3d | void_mask_3d

    bg_row = {
        'file': stem, 'cell': 'BG',
        'bright_image': is_bright,
        'green_median_u16': round(med_u16, 0),
        'green_p90_u16': round(p90_u16, 0),
        'orig_voxels': 0, 'orig_volume_um3': 0,
        'expanded_voxels': '', 'expanded_volume_um3': '',
        'z_start': '', 'z_end': '', 'n_z_slices': '',
        'mean_green': '', 'median_green': '',
    }
    for step_um in EXPAND_STEPS:
        # BG at this step = outside all cells' step-radius shells, minus artifacts/well
        step_cells = all_cells_expanded & (dist_from_labels <= step_um)
        bg_step = ~step_cells & ~exclude_3d
        bg_n = bg_step.sum()
        if bg_n > 0:
            bg_px = green[bg_step] * 65535
            bg_row[f'mean_green_{step_um}um'] = round(bg_px.mean(), 1)
            bg_row[f'median_green_{step_um}um'] = round(np.median(bg_px), 1)
            for gt in GREEN_THRESHOLDS:
                vox_above = (bg_px >= gt).sum()
                bg_row[f'pct_above_{gt}_at_{step_um}um'] = round(vox_above / bg_n * 100, 2)
            bg_row.update(_skewness_metrics(bg_px, 'bg', step_um))
        else:
            bg_row[f'mean_green_{step_um}um'] = 0
            bg_row[f'median_green_{step_um}um'] = 0
            for gt in GREEN_THRESHOLDS:
                bg_row[f'pct_above_{gt}_at_{step_um}um'] = 0
            bg_row.update(_skewness_metrics(np.array([]), 'bg', step_um))
    rows.append(bg_row)

    # Uncorrected CSV disabled — the Streamlit app only consumes _corrected.csv.
    # If you need the uncorrected data back, uncomment:
    # df = pd.DataFrame(rows)
    # df.to_csv(csv_path, index=False)
    # print(f"  Saved: {csv_path}")

    _status("Bleach correction + corrected measurements...")
    # ── Z-profile + per-step bleach correction ──
    # Precompute the per-Z in-sample mask once — reused by every _compute_z_bg_means call
    nz_g = green.shape[0]
    in_sample_3d = _in_sample_mask_3d(green)

    # Save z-profile CSV using step=0 BG (original cell mask only) as reference
    step0_cells = all_cells_expanded & (dist_from_labels <= 0)
    _save_z_profile(green, step0_cells, artifact_mask_2d, stem, out_dir, void_mask_3d,
                    in_sample_3d=in_sample_3d)

    # Compute per-step correction factors so each step uses its own BG
    step_corrections = {}
    for step_um in EXPAND_STEPS:
        step_cells = all_cells_expanded & (dist_from_labels <= step_um)
        bg_means, _ = _compute_z_bg_means(green, step_cells, artifact_mask_2d, void_mask_3d,
                                          in_sample_3d=in_sample_3d)
        step_corrections[step_um] = _compute_bleach_correction(bg_means)
    print(f"  Per-step bleach corrections computed for steps: {EXPAND_STEPS}")

    # Per-cell bounding boxes: restrict work to each cell's sub-volume instead of
    # scanning the full 53M-voxel volume for every operation.
    from scipy.ndimage import find_objects
    cell_slices = find_objects(labels_expanded)  # list indexed by (label-1)

    # ── Corrected CSV: per-step correction applied independently ──
    corr_rows = []
    for i in range(1, n_cells + 1):
        if i in excluded_cells:
            continue
        slc = cell_slices[i - 1] if i - 1 < len(cell_slices) else None
        if slc is None:
            continue

        zs, ys, xs = slc
        # Sub-volumes (views, not copies)
        cell_3d_sub  = (labels_expanded[slc] == i)
        orig_3d_sub  = (labels_orig[slc] == i)
        void_sub     = void_mask_3d[slc]
        green_sub    = green[slc]
        dist_sub     = dist_from_labels[slc]

        vox_total = int(cell_3d_sub.sum())
        orig_vox  = int(orig_3d_sub.sum())
        z_present = np.where(orig_3d_sub.any(axis=(1, 2)))[0] + zs.start

        cell_valid_sub = cell_3d_sub & ~void_sub
        valid_vox = int(cell_valid_sub.sum())
        # Apply step=0 correction on-the-fly within the sub-volume
        corr0 = step_corrections[EXPAND_STEPS[0]][zs.start:zs.stop, None, None]
        if valid_vox > 0:
            cell_pixels_c = (green_sub * corr0)[cell_valid_sub] * 65535
            mean_all_c   = float(cell_pixels_c.mean())
            median_all_c = float(np.median(cell_pixels_c))
        else:
            mean_all_c = median_all_c = 0.0

        # Original-cell bounding box (not expanded). Matches the bbox emitted
        # on the uncorrected rows; needed by the viewer for overlay on the
        # composite MIP. Coordinates are global (slc offsets added back).
        orig_coords_local = np.argwhere(orig_3d_sub)
        if len(orig_coords_local):
            bbox_min_y = int(orig_coords_local[:, 1].min() + ys.start)
            bbox_max_y = int(orig_coords_local[:, 1].max() + ys.start)
            bbox_min_x = int(orig_coords_local[:, 2].min() + xs.start)
            bbox_max_x = int(orig_coords_local[:, 2].max() + xs.start)
        else:
            bbox_min_y = bbox_max_y = bbox_min_x = bbox_max_x = 0

        crow = {
            'file': stem, 'cell': i,
            'bright_image': is_bright,
            'orig_voxels': int(orig_vox),
            'orig_volume_um3': round(orig_vox * voxel_vol, 1),
            'z_start': int(z_present.min()),
            'z_end': int(z_present.max()),
            'n_z_slices': int(z_present.max() - z_present.min() + 1),
            'bbox_min_y_px': bbox_min_y,
            'bbox_max_y_px': bbox_max_y,
            'bbox_min_x_px': bbox_min_x,
            'bbox_max_x_px': bbox_max_x,
            'mean_green': round(mean_all_c, 1),
            'median_green': round(median_all_c, 1),
        }
        for step_um in EXPAND_STEPS:
            step_mask_sub = cell_3d_sub & (dist_sub <= step_um) & ~void_sub
            step_vox = int(step_mask_sub.sum())
            if step_vox > 0:
                corr_s = step_corrections[step_um][zs.start:zs.stop, None, None]
                step_px = (green_sub * corr_s)[step_mask_sub] * 65535
                crow[f'mean_green_{step_um}um'] = round(float(step_px.mean()), 1)
                crow[f'median_green_{step_um}um'] = round(float(np.median(step_px)), 1)
                for gt in GREEN_THRESHOLDS:
                    vox_above = int((step_px >= gt).sum())
                    crow[f'pct_above_{gt}_at_{step_um}um'] = round(vox_above / step_vox * 100, 2)
                crow.update(_skewness_metrics(step_px, 'cell', step_um))
            else:
                crow[f'mean_green_{step_um}um'] = 0
                crow[f'median_green_{step_um}um'] = 0
                for gt in GREEN_THRESHOLDS:
                    crow[f'pct_above_{gt}_at_{step_um}um'] = 0
                crow.update(_skewness_metrics(np.array([]), 'cell', step_um))
        corr_rows.append(crow)

    # Corrected BG — each step uses its own correction
    corr_bg_row = {
        'file': stem, 'cell': 'BG',
        'bright_image': is_bright,
        'orig_voxels': 0, 'orig_volume_um3': 0,
        'z_start': '', 'z_end': '', 'n_z_slices': '',
        'mean_green': '', 'median_green': '',
    }
    for step_um in EXPAND_STEPS:
        step_cells = all_cells_expanded & (dist_from_labels <= step_um)
        bg_step = ~step_cells & ~exclude_3d
        bg_n = bg_step.sum()
        if bg_n > 0:
            # Apply per-Z correction on-the-fly to BG pixels (avoids full-volume copy)
            corr_s = step_corrections[step_um][:nz_g, None, None]
            bg_px_c = (green * corr_s)[bg_step] * 65535
            corr_bg_row[f'mean_green_{step_um}um'] = round(bg_px_c.mean(), 1)
            corr_bg_row[f'median_green_{step_um}um'] = round(np.median(bg_px_c), 1)
            for gt in GREEN_THRESHOLDS:
                vox_above = (bg_px_c >= gt).sum()
                corr_bg_row[f'pct_above_{gt}_at_{step_um}um'] = round(vox_above / bg_n * 100, 2)
            corr_bg_row.update(_skewness_metrics(bg_px_c, 'bg', step_um))
        else:
            corr_bg_row[f'mean_green_{step_um}um'] = 0
            corr_bg_row[f'median_green_{step_um}um'] = 0
            for gt in GREEN_THRESHOLDS:
                corr_bg_row[f'pct_above_{gt}_at_{step_um}um'] = 0
            corr_bg_row.update(_skewness_metrics(np.array([]), 'bg', step_um))
    corr_rows.append(corr_bg_row)

    corr_csv_path = os.path.join(out_dir, f'{stem}_corrected.csv')
    pd.DataFrame(corr_rows).to_csv(corr_csv_path, index=False)
    print(f"  Saved: {corr_csv_path}")

    gc.collect()

    _status("Saving composite image...")
    # Save composite MIP (project labels to 2D only for visualization)
    labels_mip = labels_expanded.max(axis=0)
    green_mip = green.max(axis=0)
    del green; gc.collect()
    # yellow_mip was saved before freeing yellow
    # Use fixed display ranges (uint16 scale) — adjust these if needed
    DISPLAY_RED_LO, DISPLAY_RED_HI = 150 / 65535, 250 / 65535
    DISPLAY_GREEN_LO, DISPLAY_GREEN_HI = 200 / 65535, 4000 / 65535
    y_norm = np.clip((yellow_mip - DISPLAY_RED_LO) / (DISPLAY_RED_HI - DISPLAY_RED_LO + 1e-10), 0, 1)
    g_norm = np.clip((green_mip - DISPLAY_GREEN_LO) / (DISPLAY_GREEN_HI - DISPLAY_GREEN_LO + 1e-10), 0, 1)

    h, w = y_norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :, 0] = y_norm
    rgb[:, :, 1] = g_norm
    # Blue channel: nucleus MIP (only when nucleus channel is present)
    if nucleus_mip is not None:
        b_norm = np.clip(
            (nucleus_mip - DISPLAY_NUCLEUS_LO) /
            (DISPLAY_NUCLEUS_HI - DISPLAY_NUCLEUS_LO + 1e-10),
            0, 1,
        )
        rgb[:, :, 2] = b_norm

    # Draw void boundary (spatial voids only — not dead layers)
    if void_mask_2d_spatial.any():
        well_bd = find_boundaries(void_mask_2d_spatial.astype(np.int32), mode='thick')
        rgb[well_bd] = [1, 1, 1]  # white
        rgb[void_mask_2d_spatial] *= 0.15  # dim void areas

    if artifact_mask_2d.any():
        art_bg = artifact_mask_2d & (labels_mip == 0)
        # Semi-transparent red tint: blend 30% red over original
        rgb[art_bg, 0] = rgb[art_bg, 0] * 0.5 + 0.3
        rgb[art_bg, 1] *= 0.5
        rgb[art_bg, 2] *= 0.5
        art_bd = find_boundaries(artifact_mask_2d.astype(np.int32), mode='thick')
        rgb[art_bd] = [1, 0, 1]  # magenta contour

    # Draw well-border exclusion zone (yellow dashed-like contour)
    if well_border_zone.any():
        border_bd = find_boundaries(well_border_zone.astype(np.int32), mode='thick')
        rgb[border_bd] = [1, 1, 0]  # yellow contour for border zone

    # Project distance to 2D: for each pixel, min distance across Z within expanded label
    dist_mip = np.full((h, w), np.inf, dtype=np.float32)
    label_mip_ids = labels_mip  # which cell each 2D pixel belongs to
    for z in range(dist_from_labels.shape[0]):
        mask_z = labels_expanded[z] > 0
        closer = dist_from_labels[z] < dist_mip
        update = mask_z & closer
        dist_mip[update] = dist_from_labels[z][update]

    import matplotlib.pyplot as plt
    cmap_c = plt.colormaps['tab20']
    # Draw expansion step contours (including 0 = cell body boundary)
    for i in range(1, n_cells + 1):
        if i in excluded_cells:
            reason = excluded_cells[i]
            cell_fill = labels_mip == i

            if reason == 'no_nucleus':
                # Dashed white contour, no fill tint (keep underlying signal visible
                # so the viewer can confirm absence of blue nucleus signal)
                if cell_fill.any():
                    cell_bd = find_boundaries(cell_fill.astype(np.int32), mode='thick')
                    yy, xx = np.indices(cell_bd.shape)
                    dash_mask = ((yy + xx) % (2 * DASH_LEN)) < DASH_LEN
                    dashed_bd = cell_bd & dash_mask
                    rgb[dashed_bd] = [1, 1, 1]
            else:
                # Existing red-tint for artifact / void / border exclusions
                rgb[cell_fill, 0] = rgb[cell_fill, 0] * 0.4 + 0.6
                rgb[cell_fill, 1] *= 0.3
                rgb[cell_fill, 2] *= 0.3
            continue

        cell_2d = labels_mip == i
        color = cmap_c(i % cmap_c.N)[:3]
        max_step = max(EXPAND_STEPS) or 1  # avoid div by zero when only [0]

        for step_um in EXPAND_STEPS:
            step_region = cell_2d & (dist_mip <= step_um)
            if not step_region.any():
                continue
            bd = find_boundaries(step_region.astype(np.int32), mode='thick')
            # Fade alpha for inner rings, full color for outer
            alpha = 0.3 + 0.7 * (step_um / max_step)
            # Lighten color: blend 50% toward white
            rgb[bd] = [0.5 + 0.5 * c for c in color]

    img_path = os.path.join(out_dir, f'{stem}_composite.png')
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(img_path)
    print(f"  Saved: {img_path}")


# ── Multiprocessing worker ────────────────────────────────────────────────
def _process_worker(args):
    """Worker for multiprocessing — processes one file, captures all output."""
    import io, contextlib
    ims_path, out_dir, status_dict = args
    name = os.path.basename(ims_path)
    short = name.split('_')[-1].replace('.ims', '')  # e.g. "F002"

    def _update(msg):
        if status_dict is not None:
            status_dict[short] = msg

    _update('starting...')
    buf = io.StringIO()
    try:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            process_one(ims_path, out_dir, status_cb=_update)
        elapsed = time.perf_counter() - t0
        _update(f'done ({elapsed:.1f}s)')
        return short, elapsed, None, buf.getvalue()
    except Exception as e:
        _update(f'ERROR: {e}')
        return short, 0, str(e), buf.getvalue()


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, shutil
    from multiprocessing import Pool, Manager, cpu_count
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn,
    )

    # parent_dir = '/Volumes/Transcend/All_ims'
    # parent_dir = '/Volumes/Transcend/20260402'
    # parent_dir = '/Volumes/Transcend/V1+Thalidomide'
    parent_dir = '/Volumes/Transcend/V1+Metformin'
    out_base = '/Users/s/Desktop/demo/Imaris_test/seg3d_results_20260420'
    test_out = '/Users/s/Desktop/demo/Imaris_test/seg3d_test_new'

    N_WORKERS = min(4, cpu_count())  # 3 parallel files max (memory safe)
    console = Console()

    os.makedirs(test_out, exist_ok=True)
    shutil.rmtree(test_out)
    os.makedirs(test_out)

    # Single file or directory test:
    #   python _segment_3d_rs.py /path/to/file.ims
    #   python _segment_3d_rs.py /path/to/dir_with_ims_files/
    if len(sys.argv) > 1:
        test_path = sys.argv[1]

        if os.path.isdir(test_path):
            # Directory mode: process all .ims files in the directory
            ims_files = sorted(glob.glob(os.path.join(test_path, '*.ims')))
            if not ims_files:
                console.print(f"[red]No .ims files found in {test_path}[/red]")
                sys.exit(1)
            subdir_name = os.path.basename(os.path.normpath(test_path))
            out_dir = os.path.join(test_out, subdir_name)
            os.makedirs(out_dir, exist_ok=True)
            console.print(f"[bold]=== Processing {len(ims_files)} files in {subdir_name} ===[/bold]")
            total_t0 = time.perf_counter()
            for i, ims_file in enumerate(ims_files):
                stem = Path(ims_file).stem
                console.print(f"\n[bold][{i+1}/{len(ims_files)}] {os.path.basename(ims_file)}[/bold]")
                t0 = time.perf_counter()
                try:
                    process_one(ims_file, out_dir)
                except Exception as e:
                    console.print(f"  [red]ERROR: {e}[/red]")
                console.print(f"  Time: {time.perf_counter()-t0:.1f}s")
            console.print(f"\n[bold green]Done! {len(ims_files)} files in {time.perf_counter()-total_t0:.1f}s[/bold green]")
            sys.exit(0)
        else:
            # Single file mode
            test_file = test_path
            subdir_name = os.path.basename(os.path.dirname(test_file))
            out_dir = os.path.join(test_out, subdir_name) if subdir_name else test_out
            os.makedirs(out_dir, exist_ok=True)
            stem = Path(test_file).stem
            for f in glob.glob(os.path.join(out_dir, f'{stem}*')):
                os.remove(f)
            console.print(f"[bold]=== Test: {os.path.basename(test_file)} ===[/bold]")
            t0 = time.perf_counter()
            process_one(test_file, out_dir)
            console.print(f"  Time: {time.perf_counter()-t0:.1f}s")
            sys.exit(0)

    def _make_progress():
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[info]}[/]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True),
            console=console,
            expand=False,
        )

    # Find all subdirs containing .ims files
    subdirs = sorted(glob.glob(os.path.join(parent_dir, '*/')))
    console.print(f"[bold]Found {len(subdirs)} subdirs in {parent_dir}[/bold]")
    console.print(f"Using [cyan]{N_WORKERS}[/cyan] parallel workers\n")

    total_t0 = time.perf_counter()

    # Collect all (subdir, files) pairs
    all_work = []
    for subdir in subdirs:
        subdir_name = os.path.basename(os.path.normpath(subdir))
        if '_Field_' not in subdir_name:
            continue
        if not subdir_name.endswith(("_Field_2", "_Field_3")):
            continue
        ims_files = sorted(glob.glob(os.path.join(subdir, '*.ims')))
        if ims_files:
            all_work.append((subdir_name, ims_files))

    total_files = sum(len(files) for _, files in all_work)

    progress = _make_progress()
    with progress:
        overall_task = progress.add_task(
            "[bold blue]Overall", total=total_files, info="")

        for subdir_name, ims_files in all_work:
            out_dir = os.path.join(out_base, subdir_name)
            os.makedirs(out_dir, exist_ok=True)
            field_label = subdir_name.split('_Field_')[-1]

            field_task = progress.add_task(
                f"  [cyan]Field {field_label}",
                total=len(ims_files), info="starting...")

            # Use Manager dict for shared status updates
            manager = Manager()
            status_dict = manager.dict()
            for f in ims_files:
                short = os.path.basename(f).split('_')[-1].replace('.ims', '')
                status_dict[short] = 'pending'

            # Per-worker task rows
            worker_tasks = {}

            tasks = [(f, out_dir, status_dict) for f in ims_files]

            with Pool(N_WORKERS) as pool:
                async_results = pool.map_async(_process_worker, tasks, chunksize=1)

                prev_status = {}
                while not async_results.ready():
                    # Update field progress
                    n_done = sum(1 for v in status_dict.values()
                                 if v.startswith('done') or v.startswith('ERROR'))
                    progress.update(field_task, completed=n_done, info="")

                    # Update/create per-file task rows
                    for fname, status in sorted(status_dict.items()):
                        if status == 'pending':
                            continue
                        is_done = status.startswith('done') or status.startswith('ERROR')
                        if fname not in worker_tasks:
                            worker_tasks[fname] = progress.add_task(
                                f"    [dim]{fname}[/]", total=1, info=status)
                        if status != prev_status.get(fname):
                            progress.update(worker_tasks[fname],
                                            completed=1 if is_done else 0,
                                            info=status,
                                            visible=not is_done)
                            prev_status[fname] = status

                    time.sleep(0.2)

                results = async_results.get()

            # Final update
            progress.update(field_task, completed=len(ims_files), info="done")
            for fname, tid in worker_tasks.items():
                status = status_dict.get(fname, '?')
                progress.update(tid, completed=1, info=status)

            progress.advance(overall_task, advance=len(ims_files))

            # Hide per-file tasks (mark invisible)
            for tid in worker_tasks.values():
                progress.update(tid, visible=False)
            progress.update(field_task, visible=False)

            # Print summary for this field
            n_ok = sum(1 for _, _, e, _ in results if e is None)
            n_err = sum(1 for _, _, e, _ in results if e is not None)
            times = [e for _, e, err, _ in results if err is None]
            avg_t = np.mean(times) if times else 0
            console.print(
                f"  [bold]Field {field_label}:[/bold] {n_ok} ok, {n_err} errors, "
                f"avg {avg_t:.1f}s/file")
            for name, elapsed, err, log in results:
                if err:
                    console.print(f"    [red]x {name}: {err}[/red]")

            # Per-subdir summary CSV
            csvs = sorted(glob.glob(os.path.join(out_dir, '*.csv')))
            csvs = [c for c in csvs if not c.endswith('_summary.csv')]
            if csvs:
                dfs = [pd.read_csv(c) for c in csvs]
                summary = pd.concat(dfs, ignore_index=True)
                summary['group'] = subdir_name
                summary_path = os.path.join(out_dir, '_summary.csv')
                summary.to_csv(summary_path, index=False)

    total_elapsed = time.perf_counter() - total_t0
    console.print(
        f"\n[bold green]=== All done: {total_files} files in "
        f"{total_elapsed:.1f}s ({total_elapsed/60:.1f}min) ===[/bold green]")

    # Global summary across all subdirs
    all_summaries = sorted(glob.glob(os.path.join(out_base, '*', '_summary.csv')))
    if all_summaries:
        all_dfs = [pd.read_csv(s) for s in all_summaries]
        global_summary = pd.concat(all_dfs, ignore_index=True)
        global_path = os.path.join(out_base, '_global_summary.csv')
        global_summary.to_csv(global_path, index=False)
        console.print(f"Global summary: {global_path} ({len(global_summary)} rows)")
