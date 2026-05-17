"""ApplyToGroupsNode -- segment every .ims under parent/<group>/*.ims using
the single chosen (threshold, step_um) combo from the K-Fold node."""
from __future__ import annotations

from pathlib import Path

from nodes.base import BaseExecutionNode, PORT_COLORS

from . import io as imaris_io
from . import segmentation_core
from .control_screening_node import (
    _SEG_PARAM_PROPS, _collect_seg_params,
)
from .data import ImarisDatasetData, PORT_TYPE_NAME, IMARIS_DATASET_COLOR
from ._widgets import add_dir_picker

PORT_COLORS.setdefault(PORT_TYPE_NAME, IMARIS_DATASET_COLOR)


class ApplyToGroupsNode(BaseExecutionNode):
    """Segment every group subfolder using the chosen (threshold, step_um) combo."""

    __identifier__ = 'plugins.Imaris3D.apply'
    NODE_NAME = 'Apply to Groups'

    # Use TYPE names in PORT_SPEC (matches PORT_COLORS keys) so the
    # Node Explorer tree icon shows the correct colors.
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['imaris_dataset']}

    _UI_PROPS = frozenset({
        'parent_folder', 'output_dir', 'force_rerun',
        *_SEG_PARAM_PROPS.keys(),
        'local_thresh_offset_u16',
    })

    def __init__(self):
        super().__init__()
        self.add_input('chosen_combo', color=PORT_COLORS.get('table'))
        self.add_output('imaris_dataset', color=PORT_COLORS.get(PORT_TYPE_NAME))

        # ── I/O ──────────────────────────────────────────────────────────
        add_dir_picker(self, 'parent_folder', 'Parent folder', tab='I/O')
        add_dir_picker(self, 'output_dir', 'Output dir', tab='I/O')
        self.add_checkbox('force_rerun', 'Force rerun',
                          text='Re-run even if CSV exists', state=False, tab='I/O')

        # ── Seg: object size limits (one row) ────────────────────────────
        self._add_row(
            'seg_sizes_row', 'Object size (voxels)',
            fields=[
                {'name': 'min_size_voxels', 'label': 'min', 'type': 'int',
                 'value': 3000, 'min_val': 0, 'max_val': 1_000_000, 'step': 500},
                {'name': 'max_size_voxels', 'label': 'max', 'type': 'int',
                 'value': 50000, 'min_val': 0, 'max_val': 10_000_000, 'step': 1000},
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

        # ── Seg: void + local-threshold offset (one row) ─────────────────
        self._add_row(
            'seg_void_row', 'Void / threshold offset',
            fields=[
                {'name': 'local_thresh_offset_u16', 'label': 'local thresh +u16',
                 'type': 'int', 'value': 3, 'min_val': 0, 'max_val': 5000, 'step': 1},
                {'name': 'green_void_fraction', 'label': 'green void fraction',
                 'type': 'float', 'value': 0.5, 'min_val': 0.0, 'max_val': 1.0,
                 'step': 0.05, 'decimals': 2},
            ],
            tab='Seg',
        )

        # ── Seg: nucleus channel + size limits + local thresh ────────────
        self._add_int_spinbox('nucleus_channel_idx', 'Nucleus channel idx',
                              value=2, min_val=0, max_val=8, step=1, tab='Seg')
        self._add_row(
            'nucleus_row', 'Nucleus (2D px)',
            fields=[
                {'name': 'nucleus_min_px', 'label': 'min px', 'type': 'int',
                 'value': 80, 'min_val': 0, 'max_val': 100000, 'step': 10},
                {'name': 'nucleus_max_px', 'label': 'max px', 'type': 'int',
                 'value': 2000, 'min_val': 0, 'max_val': 100000, 'step': 100},
                {'name': 'nucleus_local_thresh_offset_u16', 'label': 'local thresh +u16',
                 'type': 'int', 'value': 30, 'min_val': 0, 'max_val': 5000, 'step': 1},
            ],
            tab='Seg',
        )

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

    def _get_input(self, name: str):
        in_port = self.inputs().get(name)
        if not in_port or not in_port.connected_ports():
            return None
        upstream = in_port.connected_ports()[0]
        return upstream.node().output_values.get(upstream.name())

    def _status(self, msg: str):
        print(f'[ApplyToGroups] {msg}', flush=True)

    def evaluate(self) -> tuple[bool, str | None]:
        parent = Path(self.get_property('parent_folder') or '')
        out_dir = Path(self.get_property('output_dir') or '')
        if not parent.is_dir():
            return False, f'parent_folder is not a directory: {parent}'
        if not out_dir:
            return False, 'output_dir is required'

        chosen = self._get_input('chosen_combo')
        if chosen is None or chosen.payload.empty:
            return False, 'No chosen_combo on input (run K-Fold first)'
        thr  = int(chosen.payload['threshold'].iloc[0])
        step = int(chosen.payload['step_um'].iloc[0])

        seg_params = _collect_seg_params(self.get_property)
        seg_params['GREEN_THRESHOLDS'] = [thr]
        seg_params['EXPAND_STEPS']     = [step]
        force = bool(self.get_property('force_rerun'))

        # Find groups (one subdir per group, each with .ims files)
        groups = []
        for sub in sorted(p for p in parent.iterdir() if p.is_dir()):
            ims_files = imaris_io.list_ims_files(sub)
            if ims_files:
                groups.append((sub.name, ims_files))

        if not groups:
            return False, f'No group subdirectories with .ims under {parent}'

        for group_name, ims_files in groups:
            group_out = out_dir / group_name

            def _cb(idx: int, total: int, stem: str, msg: str, _g=group_name):
                self._status(f'[{_g} {idx+1}/{total}] {stem}: {msg}')

            segmentation_core.segment_batch(
                ims_files, group_out, params=seg_params, force=force,
                progress_cb=_cb,
            )

        # Reconstruct dataset
        pairs = imaris_io.scan_dataset_dir(out_dir)
        entries = []
        for csv_path, group in pairs:
            e = imaris_io.load_entry_from_csv(csv_path, group)
            if e is not None:
                entries.append(e)

        if not entries:
            return False, 'Segmentation finished but no entries reconstructed'

        ds = ImarisDatasetData(
            entries=entries, output_dir=out_dir,
            metadata={
                'seg_params': seg_params,
                'chosen_threshold': thr,
                'chosen_step_um':   step,
            },
        )
        self.output_values['imaris_dataset'] = ds
        self.mark_clean()
        return True, f'Segmented {len(entries)} files across {len(groups)} groups'
