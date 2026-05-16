# Imaris 3D Plugin

Synapse plugin for the 3D confocal fibrosis-screening pipeline, originally
implemented as a Streamlit app at `Imaris_process/`. Private plugin; not
distributed publicly.

## Workflow

```
[Control Screening] --(dataset)--> [Cell QC Filter] --(dataset)--> [K-Fold Combo Picker]
                                                                          |
                                                                  (chosen_combo)
                                                                          v
                          [Apply to Groups] <--(chosen_combo)-------------+
                                |
                            (dataset)
                                v
                       [Cell QC Filter] --> [Blank + Normalize] --> [Box/Swarm/Bar Plot]
```

`Load Imaris Dataset` is a drop-in replacement for `Control Screening` /
`Apply to Groups` when you already have segmented output on disk and want to
skip re-segmentation while iterating on K-Fold/Blank parameters.

## Nodes

| Identifier | Display name | Inputs | Outputs |
|------------|--------------|--------|---------|
| `plugins.Imaris3D.io.LoadImarisDatasetNode` | Load Imaris Dataset | — | `dataset` |
| `plugins.Imaris3D.screen.ControlScreeningNode` | Control Screening | — | `dataset` |
| `plugins.Imaris3D.qc.CellQCFilterNode` | Cell QC Filter | `dataset` | `dataset` |
| `plugins.Imaris3D.screen.KFoldComboPickerNode` | K-Fold Combo Picker | `dataset` | `ranking_table`, `chosen_combo` |
| `plugins.Imaris3D.apply.ApplyToGroupsNode` | Apply to Groups | `chosen_combo` | `dataset` |
| `plugins.Imaris3D.apply.BlankNormalizeNode` | Blank + Normalize | `dataset`, `chosen_combo` | `wide_table` |

## Custom data type

`ImarisDatasetData` (port type name `imaris_dataset`, color teal `(80, 180, 200)`).
Carries `ImarisDatasetEntry` records with paths to composite PNGs, per-cell
tables, BG rows, and per-file exclusion sets. PNG pixels stay on disk; only
paths and small DataFrames travel through ports.

## Conventions

- Output directories use `output_dir/<group_name>/{stem}_corrected.csv` and
  `output_dir/<group_name>/{stem}_composite.png`.
- BG row in each CSV is identified by `cell == "BG"` (string).
- Pct-above columns: `pct_above_<threshold>_at_<step_um>um`.

## Runtime dependencies

- `ims_reader_rs` (pre-built Rust wheel from Synapse releases) — only needed
  for actual segmentation (Control Screening / Apply to Groups). Downstream
  nodes (Load, Cell QC, K-Fold, Blank+Normalize) don't need it.
- `image_process_3d_rs` (pre-built Rust wheel from Synapse releases) — same.
- pandas, scipy, scikit-learn, matplotlib, Pillow — Synapse core deps.

## Tests

```bash
cd /Users/s/Desktop/demo/Synapse-Plugins
python -m pytest imaris_3d_nodes/tests/ -v
```

Current suite: 31 tests across data type, IO helpers, K-fold ranking core,
Blank+Normalize, and the four headless-testable nodes (Load, K-Fold, Blank,
Cell QC). Segmentation nodes (Control Screening, Apply to Groups) and the
two inline widgets are smoke-import tested only — full verification is
manual via the Synapse GUI.
