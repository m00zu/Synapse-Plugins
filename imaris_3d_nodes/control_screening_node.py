"""ControlScreeningNode -- segment all .ims in neg+pos folders, sweep grid.

Loops files internally (no Folder Iterator needed). Saves per-file CSV +
composite to output_dir/<group>/. Outputs an ImarisDatasetData reconstructed
from those saved files (small wire payload).
"""
from __future__ import annotations

from pathlib import Path

from nodes.base import BaseExecutionNode, PORT_COLORS

from . import io as imaris_io
from . import segmentation_core
from .data import ImarisDatasetData, PORT_TYPE_NAME, IMARIS_DATASET_COLOR

PORT_COLORS.setdefault(PORT_TYPE_NAME, IMARIS_DATASET_COLOR)


# Mapping: node-property name -> _segment_3d_rs_v2 module constant.
_SEG_PARAM_PROPS = {
    'sigma_um':                   'SIGMA_UM',
    'min_size_voxels':            'MIN_SIZE_VOXELS',
    'max_size_voxels':            'MAX_SIZE_VOXELS',
    'min_distance_um':            'MIN_DISTANCE_UM',
    'top_percentile':             'TOP_PERCENTILE',
    'local_bg_radius_px':         'LOCAL_BG_RADIUS_PX',
    'artifact_blur_px':           'ARTIFACT_BLUR_PX',
    'artifact_abs_uint16':        'ARTIFACT_ABS_UINT16',
    'bright_image_median_uint16': 'BRIGHT_IMAGE_MEDIAN_UINT16',
    'well_min_uint16':            'WELL_MIN_UINT16',
    'nucleus_channel_idx':        'NUCLEUS_CH_IDX',
}


def _collect_seg_params(get_prop) -> dict:
    """Read all seg-param node properties, return a dict keyed by upstream-constant name."""
    out = {}
    for prop, const in _SEG_PARAM_PROPS.items():
        v = get_prop(prop)
        # Cast properly based on type
        if prop in ('sigma_um', 'min_distance_um', 'top_percentile'):
            out[const] = float(v)
        else:
            out[const] = int(v)
    return out


def _inclusive_int_range(lo: int, hi: int, step: int) -> list[int]:
    """Inclusive integer range with step (e.g. _inclusive_int_range(8, 16, 4) == [8, 12, 16])."""
    if step <= 0:
        return [int(lo)]
    vals = []
    v = int(lo)
    while v <= int(hi):
        vals.append(v)
        v += int(step)
    return vals


class ControlScreeningNode(BaseExecutionNode):
    """Segment neg + pos controls; sweep (threshold, step_um) grid."""

    __identifier__ = 'plugins.Imaris3D.screen'
    NODE_NAME = 'Control Screening'

    PORT_SPEC = {'inputs': [], 'outputs': ['imaris_dataset']}

    _UI_PROPS = frozenset({
        'neg_folder', 'pos_folder', 'neg_label', 'pos_label',
        'output_dir', 'force_rerun',
        *_SEG_PARAM_PROPS.keys(),
        'threshold_min', 'threshold_max', 'threshold_step',
        'expand_min_um', 'expand_max_um', 'expand_step_um',
    })

    def __init__(self):
        super().__init__()
        self.add_output('imaris_dataset', color=PORT_COLORS.get(PORT_TYPE_NAME))

        # ── I/O ──────────────────────────────────────────────────────────
        # Folder paths: each on its own row (path strings can be long)
        self.add_text_input('neg_folder', 'Neg folder', text='', tab='I/O')
        self.add_text_input('pos_folder', 'Pos folder', text='', tab='I/O')
        self.add_text_input('output_dir', 'Output dir', text='', tab='I/O')
        # Group labels stay as separate text inputs (_add_row is numeric-only)
        self.add_text_input('neg_label', 'Neg label', text='neg', tab='I/O')
        self.add_text_input('pos_label', 'Pos label', text='pos', tab='I/O')
        self.add_checkbox('force_rerun', 'Force rerun',
                          text='Re-run even if CSV exists', state=False, tab='I/O')

        # ── Seg: object size limits (one row) ────────────────────────────
        self._add_row(
            'seg_sizes_row', 'Object size (voxels)',
            fields=[
                {'name': 'min_size_voxels', 'label': 'min', 'type': 'int',
                 'value': 3000, 'min_val': 1, 'max_val': 1_000_000, 'step': 100},
                {'name': 'max_size_voxels', 'label': 'max', 'type': 'int',
                 'value': 50000, 'min_val': 1, 'max_val': 1_000_000, 'step': 1000},
            ],
            tab='Seg',
        )

        # ── Seg: detection params (one row, mixed int/float) ─────────────
        self._add_row(
            'seg_detection_row', 'Detection',
            fields=[
                {'name': 'sigma_um', 'label': 'sigma (um)', 'type': 'float',
                 'value': 1.0, 'min_val': 0.1, 'max_val': 10.0, 'step': 0.1, 'decimals': 2},
                {'name': 'min_distance_um', 'label': 'min dist (um)', 'type': 'float',
                 'value': 20.0, 'min_val': 0.5, 'max_val': 100.0, 'step': 0.5, 'decimals': 2},
                {'name': 'top_percentile', 'label': 'top %', 'type': 'float',
                 'value': 99.5, 'min_val': 0.0, 'max_val': 100.0, 'step': 0.1, 'decimals': 2},
                {'name': 'local_bg_radius_px', 'label': 'BG radius (px)', 'type': 'int',
                 'value': 75, 'min_val': 1, 'max_val': 1000, 'step': 5},
            ],
            tab='Seg',
        )

        # ── Seg: nucleus channel (single field) ──────────────────────────
        self._add_int_spinbox('nucleus_channel_idx', 'Nucleus channel idx',
                              value=2, min_val=0, max_val=8, step=1, tab='Seg')

        # ── Artifact / void detection (one row) ──────────────────────────
        self._add_row(
            'artifact_row', 'Artifact / void thresholds',
            fields=[
                {'name': 'artifact_blur_px', 'label': 'blur (px)', 'type': 'int',
                 'value': 50, 'min_val': 1, 'max_val': 500, 'step': 5},
                {'name': 'artifact_abs_uint16', 'label': 'abs (u16)', 'type': 'int',
                 'value': 4000, 'min_val': 1, 'max_val': 65535, 'step': 100},
                {'name': 'bright_image_median_uint16', 'label': 'bright median', 'type': 'int',
                 'value': 1500, 'min_val': 1, 'max_val': 65535, 'step': 100},
                {'name': 'well_min_uint16', 'label': 'well min', 'type': 'int',
                 'value': 250, 'min_val': 1, 'max_val': 65535, 'step': 50},
            ],
            tab='Artifacts',
        )

        # ── Sweep grid: threshold (one row: min/max/step) ────────────────
        self._add_row(
            'threshold_grid_row', 'Threshold grid',
            fields=[
                {'name': 'threshold_min', 'label': 'min', 'type': 'int',
                 'value': 8, 'min_val': 0, 'max_val': 100, 'step': 1},
                {'name': 'threshold_max', 'label': 'max', 'type': 'int',
                 'value': 32, 'min_val': 0, 'max_val': 100, 'step': 1},
                {'name': 'threshold_step', 'label': 'step', 'type': 'int',
                 'value': 4, 'min_val': 1, 'max_val': 50, 'step': 1},
            ],
            tab='Grid',
        )

        # ── Sweep grid: expand (one row: min/max/step) ───────────────────
        self._add_row(
            'expand_grid_row', 'Expand grid (um)',
            fields=[
                {'name': 'expand_min_um', 'label': 'min', 'type': 'int',
                 'value': 0, 'min_val': 0, 'max_val': 50, 'step': 1},
                {'name': 'expand_max_um', 'label': 'max', 'type': 'int',
                 'value': 9, 'min_val': 0, 'max_val': 50, 'step': 1},
                {'name': 'expand_step_um', 'label': 'step', 'type': 'int',
                 'value': 3, 'min_val': 1, 'max_val': 25, 'step': 1},
            ],
            tab='Grid',
        )

    def _status(self, msg: str):
        """Print status (no built-in signal on BaseExecutionNode; v1 just prints)."""
        print(f'[ControlScreening] {msg}', flush=True)

    def evaluate(self) -> tuple[bool, str | None]:
        neg_folder = Path(self.get_property('neg_folder') or '')
        pos_folder = Path(self.get_property('pos_folder') or '')
        out_dir    = Path(self.get_property('output_dir') or '')

        if not neg_folder.is_dir() or not pos_folder.is_dir():
            return False, 'neg_folder or pos_folder is not a directory'
        if not out_dir:
            return False, 'output_dir is required'

        neg_label = (self.get_property('neg_label') or 'neg').strip() or 'neg'
        pos_label = (self.get_property('pos_label') or 'pos').strip() or 'pos'
        force = bool(self.get_property('force_rerun'))

        seg_params = _collect_seg_params(self.get_property)

        thr_grid = _inclusive_int_range(
            int(self.get_property('threshold_min')),
            int(self.get_property('threshold_max')),
            int(self.get_property('threshold_step')))
        exp_grid = _inclusive_int_range(
            int(self.get_property('expand_min_um')),
            int(self.get_property('expand_max_um')),
            int(self.get_property('expand_step_um')))

        seg_params['GREEN_THRESHOLDS'] = thr_grid
        seg_params['EXPAND_STEPS'] = exp_grid

        neg_out = out_dir / neg_label
        pos_out = out_dir / pos_label

        def _make_cb(group_name: str):
            def _cb(idx: int, total: int, stem: str, msg: str):
                self._status(f'[{group_name} {idx+1}/{total}] {stem}: {msg}')
            return _cb

        # Process neg
        neg_files = imaris_io.list_ims_files(neg_folder)
        if not neg_files:
            return False, f'No .ims files in {neg_folder}'
        segmentation_core.segment_batch(
            neg_files, neg_out, params=seg_params, force=force,
            progress_cb=_make_cb(neg_label),
        )

        # Process pos
        pos_files = imaris_io.list_ims_files(pos_folder)
        if not pos_files:
            return False, f'No .ims files in {pos_folder}'
        segmentation_core.segment_batch(
            pos_files, pos_out, params=seg_params, force=force,
            progress_cb=_make_cb(pos_label),
        )

        # Reconstruct dataset from saved CSVs
        pairs = imaris_io.scan_dataset_dir(out_dir)
        entries = []
        for csv_path, group in pairs:
            entry = imaris_io.load_entry_from_csv(csv_path, group)
            if entry is not None:
                entries.append(entry)

        if not entries:
            return False, 'Segmentation finished but no entries reconstructed'

        ds = ImarisDatasetData(
            entries=entries, output_dir=out_dir,
            metadata={'seg_params': seg_params,
                      'thr_grid': thr_grid, 'exp_grid': exp_grid},
        )
        self.output_values['imaris_dataset'] = ds
        self.mark_clean()
        return True, f'Segmented {len(neg_files)+len(pos_files)} files -> {len(entries)} entries'
