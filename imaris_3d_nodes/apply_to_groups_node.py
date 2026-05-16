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

PORT_COLORS.setdefault(PORT_TYPE_NAME, IMARIS_DATASET_COLOR)


class ApplyToGroupsNode(BaseExecutionNode):
    """Segment every group subfolder using the chosen (threshold, step_um) combo."""

    __identifier__ = 'plugins.Imaris3D.apply'
    NODE_NAME = 'Apply to Groups'

    PORT_SPEC = {'inputs': ['chosen_combo'], 'outputs': ['dataset']}

    _UI_PROPS = frozenset({
        'parent_folder', 'output_dir', 'force_rerun',
        *_SEG_PARAM_PROPS.keys(),
    })

    def __init__(self):
        super().__init__()
        self.add_input('chosen_combo', color=PORT_COLORS.get('table'))
        self.add_output('dataset', color=PORT_COLORS.get(PORT_TYPE_NAME))

        # I/O
        self.add_text_input('parent_folder', 'Parent folder', text='', tab='I/O')
        self.add_text_input('output_dir', 'Output dir', text='', tab='I/O')
        self.add_checkbox('force_rerun', 'Force rerun',
                          text='Re-run even if CSV exists', state=False, tab='I/O')

        # Segmentation (duplicate of ControlScreeningNode)
        self._add_float_spinbox('sigma_um', 'Sigma (um)',
                                value=1.0, min_val=0.1, max_val=10.0,
                                step=0.1, decimals=2, tab='Seg')
        self._add_int_spinbox('min_size_voxels', 'Min size (voxels)',
                              value=3000, min_val=1, max_val=1_000_000, step=100, tab='Seg')
        self._add_int_spinbox('max_size_voxels', 'Max size (voxels)',
                              value=50000, min_val=1, max_val=1_000_000, step=1000, tab='Seg')
        self._add_float_spinbox('min_distance_um', 'Min distance (um)',
                                value=20.0, min_val=0.5, max_val=100.0,
                                step=0.5, decimals=2, tab='Seg')
        self._add_float_spinbox('top_percentile', 'Top percentile',
                                value=99.5, min_val=0.0, max_val=100.0,
                                step=0.1, decimals=2, tab='Seg')
        self._add_int_spinbox('local_bg_radius_px', 'Local BG radius (px)',
                              value=75, min_val=1, max_val=1000, step=5, tab='Seg')
        self._add_int_spinbox('nucleus_channel_idx', 'Nucleus channel idx',
                              value=2, min_val=0, max_val=8, step=1, tab='Seg')

        # Artifact / void
        self._add_int_spinbox('artifact_blur_px', 'Artifact blur (px)',
                              value=50, min_val=1, max_val=500, step=5, tab='Artifacts')
        self._add_int_spinbox('artifact_abs_uint16', 'Artifact abs (u16)',
                              value=4000, min_val=1, max_val=65535, step=100, tab='Artifacts')
        self._add_int_spinbox('bright_image_median_uint16', 'Bright image median (u16)',
                              value=1500, min_val=1, max_val=65535, step=100, tab='Artifacts')
        self._add_int_spinbox('well_min_uint16', 'Well min (u16)',
                              value=250, min_val=1, max_val=65535, step=50, tab='Artifacts')

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
        self.output_values['dataset'] = ds
        self.mark_clean()
        return True, f'Segmented {len(entries)} files across {len(groups)} groups'
