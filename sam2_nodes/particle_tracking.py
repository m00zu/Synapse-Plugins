"""
particle_tracking.py — Particle Linker, Track Properties, Trajectory Plot, Track Filter.

Takes a regionprops table (frame, centroid_y, centroid_x columns) produced by
SAM2 Video Analyze or ParticleProps + BatchAccumulator, and adds track identity
(track_id) plus per-track statistics and visualizations.

Pipeline:
    SAM2 Video Analyze ──→ Particle Linker ──→ Track Properties
                                   │                  │
                                   ↓                  ↓
                           Trajectory Plot       MSD Analysis
                                   │
                                   ↓
                             Track Filter
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from data_models import ImageData, TableData
from nodes.base import PORT_COLORS
from nodes.base import BaseImageProcessNode

logger = logging.getLogger(__name__)

__all__ = [
    'ParticleLinkerNode',
    'TrackPropertiesNode',
    'TrajectoryPlotNode',
    'TrackFilterNode',
    'MSDAnalysisNode',
]

# ── Palette ──────────────────────────────────────────────────────────────────

_TRACK_PALETTE = [
    (230,  25,  75), (60,  180,  75), ( 67, 118, 232), (255, 165,   0),
    (145,  30, 180), ( 70, 240, 240), (240,  50, 230), (188, 143, 143),
    (  0, 130, 200), (245, 130,  48), (255, 225,  25), (128,   0,   0),
    (  0, 128, 128), (  0,   0, 128), (128, 128,   0), (255, 215,   0),
]

def _track_color(track_id: int) -> tuple[int, int, int]:
    return _TRACK_PALETTE[int(track_id) % len(_TRACK_PALETTE)]


def _get_table(node, port_name: str = 'table') -> Optional[pd.DataFrame]:
    port = node.inputs().get(port_name)
    if not port or not port.connected_ports():
        return None
    cp = port.connected_ports()[0]
    data = cp.node().output_values.get(cp.name())
    if isinstance(data, TableData) and data.payload is not None:
        return data.payload.copy()
    return None


def _get_image(node, port_name: str = 'image') -> Optional[np.ndarray]:
    port = node.inputs().get(port_name)
    if not port or not port.connected_ports():
        return None
    cp = port.connected_ports()[0]
    data = cp.node().output_values.get(cp.name())
    if isinstance(data, ImageData) and data.payload is not None:
        arr = data.payload
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return np.ascontiguousarray(arr)
    return None


# ── Core linking algorithm ────────────────────────────────────────────────────

def _link_particles(df: pd.DataFrame, max_disp: float, max_gap: int,
                    min_length: int, weight_area: float) -> pd.DataFrame:
    """
    Nearest-neighbor particle linking across frames with gap-closing.

    Parameters
    ----------
    df          : regionprops table with columns frame, centroid_y, centroid_x
    max_disp    : maximum pixel displacement per frame
    max_gap     : allow tracks to skip this many frames before dying
    min_length  : drop finished tracks shorter than this many frames
    weight_area : 0..1 weight given to area similarity vs position distance

    Returns
    -------
    df with added `track_id` column
    """
    from scipy.spatial import KDTree

    df = df.copy()
    df['track_id'] = -1

    frames = sorted(df['frame'].unique())
    if not frames:
        return df

    # Active tracks: track_id → dict(last_frame, last_y, last_x, last_area)
    active: dict[int, dict] = {}
    next_id = 0
    finished: list[int] = []   # track_ids that died

    def _coords(frame_df):
        return frame_df[['centroid_y', 'centroid_x']].values

    prev_frame_df = None
    prev_idxs = None  # index into df for each row in prev_frame_df

    for frame in frames:
        cur = df[df['frame'] == frame]
        cur_idxs = cur.index.tolist()
        n_cur = len(cur)

        if prev_frame_df is None or not active:
            # First frame or no active tracks — start fresh tracks
            for idx in cur_idxs:
                row = df.loc[idx]
                df.at[idx, 'track_id'] = next_id
                active[next_id] = {
                    'last_frame': frame,
                    'last_y': row['centroid_y'],
                    'last_x': row['centroid_x'],
                    'last_area': row.get('area', 0),
                }
                next_id += 1
        else:
            # Build cost matrix: rows=active tracks, cols=current detections
            active_ids = list(active.keys())
            active_pos = np.array([[active[t]['last_y'], active[t]['last_x']]
                                   for t in active_ids])
            cur_pos = _coords(cur)

            if len(cur_pos) == 0:
                # No detections this frame — age out dormant tracks
                for tid in list(active.keys()):
                    if frame - active[tid]['last_frame'] > max_gap:
                        finished.append(tid)
                        del active[tid]
                prev_frame_df = cur
                prev_idxs = cur_idxs
                continue

            # Distance cost
            from scipy.spatial.distance import cdist
            dist_cost = cdist(active_pos, cur_pos)  # (n_active, n_cur)

            # Optional area similarity cost (normalized)
            if weight_area > 0 and 'area' in df.columns:
                active_areas = np.array([active[t]['last_area'] for t in active_ids],
                                        dtype=float).reshape(-1, 1)
                cur_areas = cur['area'].values.astype(float).reshape(1, -1)
                # Ratio penalty: 0 = identical, high = very different
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(
                        (active_areas + cur_areas) > 0,
                        np.abs(active_areas - cur_areas) / np.maximum(active_areas, cur_areas),
                        1.0,
                    )
                cost = (1 - weight_area) * dist_cost + weight_area * ratio * max_disp
            else:
                cost = dist_cost

            # Mask out pairs that exceed max_disp
            cost[dist_cost > max_disp] = 1e9

            # Hungarian assignment
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)

            assigned_tracks = set()
            assigned_dets = set()

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 1e9:
                    continue
                tid = active_ids[r]
                idx = cur_idxs[c]
                row = df.loc[idx]
                df.at[idx, 'track_id'] = tid
                active[tid] = {
                    'last_frame': frame,
                    'last_y': row['centroid_y'],
                    'last_x': row['centroid_x'],
                    'last_area': row.get('area', 0),
                }
                assigned_tracks.add(tid)
                assigned_dets.add(c)

            # Unmatched detections → new tracks
            for c, idx in enumerate(cur_idxs):
                if c not in assigned_dets:
                    row = df.loc[idx]
                    df.at[idx, 'track_id'] = next_id
                    active[next_id] = {
                        'last_frame': frame,
                        'last_y': row['centroid_y'],
                        'last_x': row['centroid_x'],
                        'last_area': row.get('area', 0),
                    }
                    next_id += 1

            # Age out dormant tracks
            for tid in list(active.keys()):
                if frame - active[tid]['last_frame'] > max_gap:
                    finished.append(tid)
                    del active[tid]

        prev_frame_df = cur
        prev_idxs = cur_idxs

    # All remaining active tracks are finished
    for tid in active:
        finished.append(tid)

    # Drop tracks shorter than min_length
    if min_length > 1:
        counts = df[df['track_id'] >= 0]['track_id'].value_counts()
        drop_ids = set(counts[counts < min_length].index.tolist())
        df.loc[df['track_id'].isin(drop_ids), 'track_id'] = -1

    df['track_id'] = df['track_id'].astype(int)
    return df


# ── Node 1: Particle Linker ───────────────────────────────────────────────────

class ParticleLinkerNode(BaseImageProcessNode):
    """Link particle detections across video frames into tracks.

    Takes a regionprops table (frame, centroid_y, centroid_x columns) and
    assigns a track_id to each detection so the same physical particle
    shares one ID across all frames.

    Uses nearest-neighbor linking (Hungarian algorithm) with gap-closing.
    Unlinked detections get track_id = -1.

    Keywords: track, link, particle, trajectory, 粒子, 追蹤, 連結
    """
    __identifier__ = 'plugins.Plugins.VideoAnalysis'
    NODE_NAME      = 'Particle Linker'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('table', color=PORT_COLORS['table'])

        self._add_float_spinbox('max_displacement', 'Max Displacement (px)',
                                value=30.0, min_val=1.0, max_val=2000.0,
                                step=5.0, decimals=1)
        self._add_float_spinbox('max_gap_frames', 'Max Gap (frames)',
                                value=2.0, min_val=0.0, max_val=100.0,
                                step=1.0, decimals=0)
        self._add_float_spinbox('min_track_length', 'Min Track Length',
                                value=3.0, min_val=1.0, max_val=9999.0,
                                step=1.0, decimals=0)
        self._add_float_spinbox('area_weight', 'Area Weight',
                                value=0.0, min_val=0.0, max_val=1.0,
                                step=0.05, decimals=2)

    def evaluate(self):
        self.reset_progress()

        df = _get_table(self)
        if df is None:
            return False, "No table connected"

        required = {'frame', 'centroid_y', 'centroid_x'}
        missing = required - set(df.columns)
        if missing:
            return False, f"Table missing columns: {missing}"

        self.set_progress(10)

        max_disp  = float(self.get_property('max_displacement') or 30)
        max_gap   = int(self.get_property('max_gap_frames') or 2)
        min_len   = int(self.get_property('min_track_length') or 3)
        area_wt   = float(self.get_property('area_weight') or 0.0)

        linked = _link_particles(df, max_disp, max_gap, min_len, area_wt)

        n_tracks = linked[linked['track_id'] >= 0]['track_id'].nunique()
        n_unlinked = (linked['track_id'] < 0).sum()
        logger.info("Particle Linker: %d tracks, %d unlinked detections",
                    n_tracks, n_unlinked)

        self.set_progress(100)
        self.output_values['table'] = TableData(payload=linked)
        self.mark_clean()
        return True, None


# ── Node 2: Track Properties ──────────────────────────────────────────────────

def _compute_track_props(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-track statistics from a linked table."""
    required = {'track_id', 'frame', 'centroid_y', 'centroid_x'}
    df = df[df['track_id'] >= 0].copy()

    rows = []
    for tid, grp in df.groupby('track_id'):
        grp = grp.sort_values('frame')
        ys = grp['centroid_y'].values.astype(float)
        xs = grp['centroid_x'].values.astype(float)
        frames = grp['frame'].values

        # Step displacements
        dy = np.diff(ys)
        dx = np.diff(xs)
        steps = np.sqrt(dy**2 + dx**2)

        total_path       = float(steps.sum()) if len(steps) else 0.0
        net_displacement = float(np.sqrt((ys[-1]-ys[0])**2 + (xs[-1]-xs[0])**2))
        n_frames         = len(grp)
        duration         = int(frames[-1]) - int(frames[0]) + 1
        confinement      = net_displacement / total_path if total_path > 0 else 0.0
        mean_speed       = total_path / max(n_frames - 1, 1)
        max_speed        = float(steps.max()) if len(steps) else 0.0

        # Simple MSD-based diffusion coefficient (2D, short-time fit)
        # MSD(1) ≈ 4 * D * dt  (dt = 1 frame)
        msd1 = float(np.mean(steps**2)) if len(steps) else 0.0
        diff_coeff = msd1 / 4.0

        # Anomalous exponent: log-log slope of MSD vs lag (need ≥4 points)
        alpha = float('nan')
        if n_frames >= 5:
            max_lag = max(2, n_frames // 2)
            lags, msds = [], []
            for lag in range(1, max_lag + 1):
                sq_disps = (ys[lag:] - ys[:-lag])**2 + (xs[lag:] - xs[:-lag])**2
                if len(sq_disps) > 0:
                    lags.append(lag)
                    msds.append(float(np.mean(sq_disps)))
            if len(lags) >= 3:
                log_lag  = np.log(lags)
                log_msd  = np.log(np.maximum(msds, 1e-9))
                # Weighted linear fit (weight early lags more)
                w = 1.0 / np.arange(1, len(lags) + 1, dtype=float)
                coeffs = np.polyfit(log_lag, log_msd, 1, w=w)
                alpha  = float(coeffs[0])

        row = {
            'track_id':         int(tid),
            'n_frames':         n_frames,
            'start_frame':      int(frames[0]),
            'end_frame':        int(frames[-1]),
            'duration':         duration,
            'total_path':       round(total_path, 2),
            'net_displacement': round(net_displacement, 2),
            'confinement_ratio': round(confinement, 4),
            'mean_speed':       round(mean_speed, 3),
            'max_speed':        round(max_speed, 3),
            'diffusion_coeff':  round(diff_coeff, 4),
            'alpha':            round(alpha, 3) if not np.isnan(alpha) else float('nan'),
            'start_y':          round(float(ys[0]), 1),
            'start_x':          round(float(xs[0]), 1),
        }

        # Optional: mean area / mean intensity if columns present
        if 'area' in grp.columns:
            row['mean_area'] = round(float(grp['area'].mean()), 1)
        if 'mean_intensity' in grp.columns:
            row['mean_intensity'] = round(float(grp['mean_intensity'].mean()), 3)

        rows.append(row)

    return pd.DataFrame(rows)


class TrackPropertiesNode(BaseImageProcessNode):
    """Compute per-track statistics from a linked particle table.

    Input must have track_id, frame, centroid_y, centroid_x columns
    (output of Particle Linker).

    Output columns: track_id, n_frames, duration, total_path,
    net_displacement, confinement_ratio, mean_speed, max_speed,
    diffusion_coeff, alpha (anomalous exponent).

    Motion type interpretation of alpha:
      alpha ≈ 1  → Brownian diffusion (random walk)
      alpha < 1  → Confined / sub-diffusion (tethered, corralled)
      alpha > 1  → Directed / super-diffusion (motor-driven transport)

    Keywords: track, MSD, diffusion, velocity, trajectory, 軌跡, 擴散
    """
    __identifier__ = 'plugins.Plugins.VideoAnalysis'
    NODE_NAME      = 'Track Properties'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('table', color=PORT_COLORS['table'])

    def evaluate(self):
        self.reset_progress()

        df = _get_table(self)
        if df is None:
            return False, "No table connected"
        if 'track_id' not in df.columns:
            return False, "Table has no track_id column — run Particle Linker first"

        self.set_progress(20)
        result = _compute_track_props(df)
        self.set_progress(100)

        logger.info("Track Properties: %d tracks", len(result))
        self.output_values['table'] = TableData(payload=result)
        self.mark_clean()
        return True, None


# ── Node 3: Trajectory Plot ───────────────────────────────────────────────────

def _draw_trajectories(rgb_arr: np.ndarray, df: pd.DataFrame,
                       tail_frames: int, line_width: int,
                       show_id: bool, color_by: str) -> np.ndarray:
    """Draw particle trajectories on an RGB image."""
    img = Image.fromarray(rgb_arr, mode='RGB')
    draw = ImageDraw.Draw(img, 'RGBA')

    all_frames = sorted(df['frame'].unique())
    max_frame  = max(all_frames) if all_frames else 0
    min_frame  = min(all_frames) if all_frames else 0

    # Build per-track sorted positions
    track_pos: dict[int, list[tuple[int, float, float]]] = {}
    for tid, grp in df.groupby('track_id'):
        if int(tid) < 0:
            continue
        pts = [(int(r['frame']), float(r['centroid_x']), float(r['centroid_y']))
               for _, r in grp.sort_values('frame').iterrows()]
        track_pos[int(tid)] = pts

    # Compute per-track speed for color_by='speed'
    track_speed: dict[int, float] = {}
    if color_by == 'speed':
        for tid, pts in track_pos.items():
            if len(pts) < 2:
                track_speed[tid] = 0.0
            else:
                steps = [np.sqrt((pts[i+1][1]-pts[i][1])**2 + (pts[i+1][2]-pts[i][2])**2)
                         for i in range(len(pts)-1)]
                track_speed[tid] = float(np.mean(steps))
        max_speed = max(track_speed.values()) or 1.0

    for tid, pts in track_pos.items():
        if tail_frames > 0:
            pts = [p for p in pts if p[0] >= max_frame - tail_frames]
        if len(pts) < 2:
            continue

        # Color
        if color_by == 'speed':
            ratio = min(track_speed.get(tid, 0) / max_speed, 1.0)
            color = (int(255 * ratio), int(255 * (1 - ratio)), 80, 220)
        else:
            r, g, b = _track_color(tid)
            color = (r, g, b, 220)

        # Draw path
        xy = [(p[1], p[2]) for p in pts]
        for i in range(len(xy) - 1):
            draw.line([xy[i], xy[i+1]], fill=color, width=line_width)

        # Draw dot at last position
        lx, ly = xy[-1]
        r = line_width + 1
        draw.ellipse([lx-r, ly-r, lx+r, ly+r], fill=color)

        # Optionally draw track ID
        if show_id:
            draw.text((lx + r + 2, ly - 6), str(tid), fill=color[:3] + (255,))

    return np.asarray(img, dtype=np.uint8)


class TrajectoryPlotNode(BaseImageProcessNode):
    """Draw particle trajectories as colored paths on a background image.

    Connect a reference image (e.g. first frame or overlay) and the
    linked table from Particle Linker. Each track gets a distinct color.

    Parameters:
      Tail Frames — 0 = show full history; N = show last N frames only
      Color By    — track_id (fixed color) or speed (red=fast, green=slow)
      Show IDs    — label each track with its ID number

    Keywords: trajectory, track, visualize, path, overlay, 軌跡, 視覺化
    """
    __identifier__ = 'plugins.Plugins.VideoAnalysis'
    NODE_NAME      = 'Trajectory Plot'
    PORT_SPEC      = {'inputs': ['image', 'table'], 'outputs': ['image']}

    _UI_PROPS = frozenset({'color', 'pos', 'selected', 'name', 'progress',
                           'show_preview', 'live_preview'})

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('image', color=PORT_COLORS['image'])

        self._add_float_spinbox('tail_frames', 'Tail Frames (0=all)',
                                value=0.0, min_val=0.0, max_val=9999.0,
                                step=5.0, decimals=0)
        self._add_float_spinbox('line_width', 'Line Width',
                                value=2.0, min_val=1.0, max_val=10.0,
                                step=1.0, decimals=0)
        self.add_combo_menu('color_by', 'Color By', items=['track_id', 'speed'])
        self.add_checkbox('show_ids', 'Show IDs', state=False)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        rgb = _get_image(self)
        if rgb is None:
            return False, "No image connected"

        df = _get_table(self)
        if df is None:
            return False, "No table connected"
        if 'track_id' not in df.columns:
            return False, "Table has no track_id column — run Particle Linker first"

        required = {'frame', 'centroid_y', 'centroid_x'}
        if not required.issubset(df.columns):
            return False, f"Table missing columns: {required - set(df.columns)}"

        self.set_progress(20)

        tail   = int(self.get_property('tail_frames') or 0)
        lw     = int(self.get_property('line_width') or 2)
        cby    = self.get_property('color_by') or 'track_id'
        showid = bool(self.get_property('show_ids'))

        vis = _draw_trajectories(rgb, df, tail, lw, showid, cby)
        self.set_progress(90)

        self.output_values['image'] = ImageData(payload=vis)
        # self._update_preview(pil)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ── Node 4: Track Filter ──────────────────────────────────────────────────────

class TrackFilterNode(BaseImageProcessNode):
    """Filter a linked particle table by per-track statistics.

    Removes rows belonging to tracks that fall outside the specified
    min/max bounds. Unlinked rows (track_id = -1) are always kept unless
    'Drop Unlinked' is checked.

    Set any bound to 0 to disable that limit.

    Keywords: filter, track, trajectory, 篩選, 軌跡, 粒子
    """
    __identifier__ = 'plugins.Plugins.VideoAnalysis'
    NODE_NAME      = 'Track Filter'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('table', color=PORT_COLORS['table'])

        self._add_float_spinbox('min_length', 'Min Length (frames)',
                                value=0.0, min_val=0.0, max_val=9999.0,
                                step=1.0, decimals=0)
        self._add_float_spinbox('max_length', 'Max Length (frames)',
                                value=0.0, min_val=0.0, max_val=9999.0,
                                step=1.0, decimals=0)
        self._add_float_spinbox('min_displacement', 'Min Net Disp (px)',
                                value=0.0, min_val=0.0, max_val=9999.0,
                                step=1.0, decimals=1)
        self._add_float_spinbox('max_displacement', 'Max Net Disp (px)',
                                value=0.0, min_val=0.0, max_val=99999.0,
                                step=5.0, decimals=1)
        self._add_float_spinbox('min_speed', 'Min Mean Speed',
                                value=0.0, min_val=0.0, max_val=9999.0,
                                step=0.5, decimals=2)
        self._add_float_spinbox('max_speed', 'Max Mean Speed',
                                value=0.0, min_val=0.0, max_val=9999.0,
                                step=0.5, decimals=2)
        self.add_checkbox('drop_unlinked', 'Drop Unlinked (track_id=-1)', state=False)

    def evaluate(self):
        self.reset_progress()

        df = _get_table(self)
        if df is None:
            return False, "No table connected"
        if 'track_id' not in df.columns:
            return False, "Table has no track_id column — run Particle Linker first"

        self.set_progress(10)

        min_len  = float(self.get_property('min_length') or 0)
        max_len  = float(self.get_property('max_length') or 0)
        min_disp = float(self.get_property('min_displacement') or 0)
        max_disp = float(self.get_property('max_displacement') or 0)
        min_spd  = float(self.get_property('min_speed') or 0)
        max_spd  = float(self.get_property('max_speed') or 0)
        drop_unl = bool(self.get_property('drop_unlinked'))

        linked = df[df['track_id'] >= 0]
        unlinked = df[df['track_id'] < 0]

        # Per-track stats for filtering
        stats = _compute_track_props(linked) if len(linked) > 0 else pd.DataFrame()

        keep_ids: set[int] = set()
        for _, row in stats.iterrows():
            tid = int(row['track_id'])
            nf  = row['n_frames']
            nd  = row['net_displacement']
            ms  = row['mean_speed']
            if min_len  > 0 and nf < min_len:  continue
            if max_len  > 0 and nf > max_len:  continue
            if min_disp > 0 and nd < min_disp: continue
            if max_disp > 0 and nd > max_disp: continue
            if min_spd  > 0 and ms < min_spd:  continue
            if max_spd  > 0 and ms > max_spd:  continue
            keep_ids.add(tid)

        filtered = linked[linked['track_id'].isin(keep_ids)]
        if not drop_unl:
            filtered = pd.concat([filtered, unlinked], ignore_index=True)

        self.set_progress(100)
        logger.info("Track Filter: %d tracks kept (of %d), %d rows",
                    len(keep_ids), stats['track_id'].nunique() if len(stats) else 0,
                    len(filtered))
        self.output_values['table'] = TableData(payload=filtered)
        self.mark_clean()
        return True, None


# ── Node 5: MSD Analysis ──────────────────────────────────────────────────────

class MSDAnalysisNode(BaseImageProcessNode):
    """Compute Mean Squared Displacement (MSD) vs lag time for all tracks.

    MSD(τ) = <|r(t+τ) − r(t)|²> averaged over all track-time-origin pairs.

    Output table columns: lag, msd_ensemble, n_samples, plus per-track
    columns if 'Per Track' is checked.

    Interpretation:
      log-log slope (alpha) ≈ 1 → Brownian diffusion
      alpha < 1              → Confined motion (sub-diffusion)
      alpha > 1              → Directed transport (super-diffusion)

    Keywords: MSD, diffusion, trajectory, Brownian, 均方位移, 擴散
    """
    __identifier__ = 'plugins.Plugins.VideoAnalysis'
    NODE_NAME      = 'MSD Analysis'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['table', 'plot']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('table', color=PORT_COLORS['table'])
        self.add_output('plot',  color=PORT_COLORS['image'])

        self._add_float_spinbox('max_lag', 'Max Lag (frames, 0=auto)',
                                value=0.0, min_val=0.0, max_val=9999.0,
                                step=5.0, decimals=0)
        self.add_checkbox('per_track', 'Per-Track MSD columns', state=False)

    def evaluate(self):
        self.reset_progress()

        df = _get_table(self)
        if df is None:
            return False, "No table connected"
        if 'track_id' not in df.columns:
            return False, "Table has no track_id column — run Particle Linker first"

        required = {'frame', 'centroid_y', 'centroid_x'}
        if not required.issubset(df.columns):
            return False, f"Table missing: {required - set(df.columns)}"

        self.set_progress(10)
        df = df[df['track_id'] >= 0].copy()

        max_lag_param = int(self.get_property('max_lag') or 0)
        per_track     = bool(self.get_property('per_track'))

        # Auto max_lag: half the median track length
        if max_lag_param == 0:
            lengths = df.groupby('track_id')['frame'].count()
            max_lag = max(2, int(lengths.median() // 2))
        else:
            max_lag = max_lag_param

        # Per-track position arrays
        tracks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for tid, grp in df.groupby('track_id'):
            g = grp.sort_values('frame')
            tracks[int(tid)] = (g['centroid_y'].values.astype(float),
                                g['centroid_x'].values.astype(float))

        self.set_progress(30)

        # Ensemble MSD
        lag_range = range(1, max_lag + 1)
        ensemble_rows = []
        per_track_data: dict[int, dict[int, float]] = {tid: {} for tid in tracks}

        for lag in lag_range:
            all_sq = []
            for tid, (ys, xs) in tracks.items():
                if len(ys) <= lag:
                    continue
                sq = (ys[lag:] - ys[:-lag])**2 + (xs[lag:] - xs[:-lag])**2
                all_sq.extend(sq.tolist())
                if per_track:
                    per_track_data[tid][lag] = float(np.mean(sq))
            if all_sq:
                ensemble_rows.append({
                    'lag': lag,
                    'msd_ensemble': round(float(np.mean(all_sq)), 4),
                    'n_samples': len(all_sq),
                })

        msd_df = pd.DataFrame(ensemble_rows)

        if per_track and len(msd_df) > 0:
            for tid in tracks:
                col = [per_track_data[tid].get(lag, float('nan'))
                       for lag in msd_df['lag']]
                msd_df[f'track_{tid}'] = col

        self.set_progress(70)

        # MSD plot (matplotlib)
        plot_img = self._make_msd_plot(msd_df)

        self.set_progress(100)
        self.output_values['table'] = TableData(payload=msd_df)
        self.output_values['plot']  = ImageData(payload=plot_img)
        self.mark_clean()
        return True, None

    def _make_msd_plot(self, msd_df: pd.DataFrame) -> np.ndarray:
        """Render MSD vs lag time log-log plot. Returns uint8 RGB numpy array."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            # Return blank image if matplotlib unavailable
            return np.full((300, 400, 3), 30, dtype=np.uint8)

        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#16213e')

        if len(msd_df) > 0 and 'msd_ensemble' in msd_df.columns:
            lags = msd_df['lag'].values
            msds = msd_df['msd_ensemble'].values
            valid = msds > 0
            if valid.any():
                ax.plot(lags[valid], msds[valid], 'o-', color='#00d4ff',
                        linewidth=2, markersize=4, label='Ensemble MSD')
                ax.set_xscale('log')
                ax.set_yscale('log')

                # Fit slope (alpha)
                if valid.sum() >= 3:
                    coeffs = np.polyfit(np.log(lags[valid]), np.log(msds[valid]), 1)
                    alpha  = coeffs[0]
                    ax.text(0.05, 0.95, f'α = {alpha:.2f}',
                            transform=ax.transAxes, color='#ffd700',
                            fontsize=11, va='top',
                            bbox=dict(boxstyle='round', fc='#0f3460', ec='#ffd700', alpha=0.7))

        ax.set_xlabel('Lag (frames)', color='white', fontsize=10)
        ax.set_ylabel('MSD (px²)', color='white', fontsize=10)
        ax.set_title('Mean Squared Displacement', color='white', fontsize=11)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.legend(facecolor='#0f3460', edgecolor='#444', labelcolor='white',
                  fontsize=9)
        ax.grid(True, alpha=0.2, color='white')

        fig.tight_layout()

        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        return np.asarray(pil_img, dtype=np.uint8).copy()
