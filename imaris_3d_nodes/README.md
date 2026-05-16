# Imaris 3D Plugin

Synapse plugin for the 3D confocal fibrosis-screening pipeline (originally a
Streamlit app at `Imaris_process/`).

## Nodes

| Identifier | Display name | Purpose |
|------------|--------------|---------|
| plugins.Imaris3D.io.LoadImarisDataset | Load Imaris Dataset | Reconstruct ImarisDatasetData from a saved output directory |
| plugins.Imaris3D.screen.ControlScreening | Control Screening | Segment neg+pos controls, sweep (threshold, step_um) grid |
| plugins.Imaris3D.qc.CellQCFilter | Cell QC Filter | Click cells on composite MIPs to exclude outliers |
| plugins.Imaris3D.screen.KFoldComboPicker | K-Fold Combo Picker | K-fold CV ranking + interactive heatmap pick |
| plugins.Imaris3D.apply.ApplyToGroups | Apply to Groups | Re-segment all groups with the chosen combo |
| plugins.Imaris3D.apply.BlankNormalize | Blank + Normalize | Subtract BG, normalize to reference group, emit wide table |

## Custom data type

`ImarisDatasetData` (port type name: `imaris_dataset`, color teal `(80, 180, 200)`).
Bundles per-file `ImarisDatasetEntry` records with paths to composite PNGs, per-cell tables, BG rows, and exclusion sets.

## Runtime requirements

- `ims_reader_rs` (pre-built Rust wheel from Synapse releases)
- `image_process_3d_rs` (pre-built Rust wheel from Synapse releases)
- pandas, scipy, scikit-learn, matplotlib, Pillow (all already required by Synapse core)
