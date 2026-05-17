"""LoadImarisDatasetNode -- reconstruct an ImarisDatasetData from a saved output dir."""
from __future__ import annotations

from pathlib import Path

from nodes.base import BaseExecutionNode, PORT_COLORS

from . import io as imaris_io
from .data import ImarisDatasetData, PORT_TYPE_NAME, IMARIS_DATASET_COLOR
from ._widgets import add_dir_picker


# Ensure the port colour is registered even if data.py hasn't been imported yet.
PORT_COLORS.setdefault(PORT_TYPE_NAME, IMARIS_DATASET_COLOR)


class LoadImarisDatasetNode(BaseExecutionNode):
    """Read an ``output_dir`` written by ControlScreening / ApplyToGroups and
    output the corresponding ImarisDatasetData without re-running segmentation.

    Useful for resuming after a crash and for iterating on K-Fold / Blank
    parameters without paying the segmentation cost again.
    """

    __identifier__ = 'plugins.Imaris3D.io'
    NODE_NAME = 'Load Imaris Dataset'

    PORT_SPEC = {'inputs': [], 'outputs': ['imaris_dataset']}

    _UI_PROPS = frozenset({'dataset_dir', 'layout', 'default_group'})

    def __init__(self):
        super().__init__()
        self.add_output('imaris_dataset', color=PORT_COLORS.get(PORT_TYPE_NAME))

        # Dataset directory uses a folder-picker
        add_dir_picker(self, 'dataset_dir', 'Dataset directory', tab='Settings')
        self.add_combo_menu(
            'layout', 'Layout',
            items=['auto', 'subfolders_per_group', 'flat_single_group'],
            tab='Settings',
        )
        # default_group is a label, not a path
        self.add_text_input(
            'default_group', 'Default group (flat layout only)',
            text='', tab='Settings',
        )

    def evaluate(self) -> tuple[bool, str | None]:
        root = Path(self.get_property('dataset_dir') or '')
        if not root.exists() or not root.is_dir():
            return False, f'Directory does not exist: {root}'

        layout = self.get_property('layout')
        default_group = self.get_property('default_group') or None

        # Resolve layout
        if layout == 'auto':
            pairs = imaris_io.scan_dataset_dir(root, default_group=default_group)
        elif layout == 'subfolders_per_group':
            pairs = []
            for sub in sorted(p for p in root.iterdir() if p.is_dir()):
                for csv in sorted(sub.glob('*_corrected.csv')):
                    pairs.append((csv, sub.name))
        else:  # flat_single_group
            group = default_group or root.name
            pairs = [(csv, group) for csv in sorted(root.glob('*_corrected.csv'))]

        if not pairs:
            return False, f'No *_corrected.csv files found under {root}'

        entries = []
        for csv_path, group in pairs:
            entry = imaris_io.load_entry_from_csv(csv_path, group)
            if entry is not None:
                entries.append(entry)

        if not entries:
            return False, f'Found CSVs but none were parseable under {root}'

        ds = ImarisDatasetData(
            entries=entries,
            output_dir=root,
            metadata={'loaded_from': str(root)},
        )
        self.output_values['imaris_dataset'] = ds
        self.mark_clean()
        return True, f'Loaded {len(entries)} files in {len(ds.groups)} groups'
