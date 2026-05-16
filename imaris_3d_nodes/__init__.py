"""Imaris 3D plugin for Synapse.

Six nodes implementing the fibrosis-screening pipeline:
1. ControlScreeningNode      - segment neg+pos controls, sweep (threshold, step_um) grid
1'. LoadImarisDatasetNode    - reconstruct a dataset from a saved output directory
2. CellQCFilterNode          - click cells on composite MIPs to exclude per-file outliers
3. KFoldComboPickerNode      - K-fold CV ranking of (threshold, step_um) combos
4. ApplyToGroupsNode         - re-segment all groups using the chosen combo
5. BlankNormalizeNode        - subtract BG, normalize to a reference group, emit wide table

Plugin identifier: plugins.Imaris3D.*
"""
from __future__ import annotations

# Register the custom port type BEFORE importing any node module so
# add_input/add_output calls inside the node modules find the type.
from . import data  # noqa: F401  (side-effect: register_port_type)

# Defensive imports: the plugin should not break if a single node module
# fails to import (e.g. the user removed a file). Each successful import
# extends __all__ so the plugin loader can find them.
__all__ = []

for mod_name, cls_name in (
    ('load_dataset_node',        'LoadImarisDatasetNode'),
    ('control_screening_node',   'ControlScreeningNode'),
    ('cell_qc_node',             'CellQCFilterNode'),
    ('kfold_picker_node',        'KFoldComboPickerNode'),
    ('apply_to_groups_node',     'ApplyToGroupsNode'),
    ('blank_normalize_node',     'BlankNormalizeNode'),
):
    try:
        module = __import__(f'{__name__}.{mod_name}', fromlist=[cls_name])
        globals()[cls_name] = getattr(module, cls_name)
        __all__.append(cls_name)
    except Exception as e:
        import sys
        print(f'[imaris_3d_nodes] failed to load {cls_name}: {e}', file=sys.stderr)
