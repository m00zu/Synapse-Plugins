# Synapse Plugins

Plugins for [Synapse](https://github.com/m00zu/Synapse), a visual node-graph workflow editor for scientific data analysis.

## Installing

Download the `.synpkg` file for your platform from [Releases](https://github.com/m00zu/Synapse-Plugins/releases), then in Synapse: **Plugins > Install Plugin** and select the file.

Pure Python plugins (image_analysis, statistical_analysis, etc.) are cross-platform — one file works everywhere.

## Plugins

| Plugin | Status | Description |
|--------|--------|-------------|
| image_analysis | stable | Filters, thresholding, morphology, segmentation, measurements, ROI |
| statistical_analysis | stable | t-tests, ANOVA, regression, survival analysis, PCA |
| figure_plotting | stable | Scatter, box, violin, heatmap, volcano, regression plots |
| filopodia_nodes | stable | Cell protrusion detection and measurement |
| volume_nodes | experimental | 3D volume rendering and analysis |
| sam2_nodes | stable | SAM2 segmentation, Cellpose, video tracking (ONNX) |
| rdkit_nodes | beta | RDKit chemistry, AutoDock Vina docking (Rust + ONNX) |

Pure Python plugins (image_analysis, statistical_analysis, figure_plotting, filopodia_nodes, volume_nodes) are cross-platform. SAM2 and rdkit plugins have platform-specific builds — download the one matching your OS from Releases.

## Building locally

If you want to build plugins yourself instead of using the pre-built releases:

```bash
pip install zstandard

# Pure Python plugin
python package_plugin.py image_analysis --slim

# Plugin with vendored dependencies
python package_plugin.py sam2_nodes

# Plugin with Rust (needs Rust toolchain + maturin)
cd rdkit_nodes/rust/vina_rust && maturin build --release --out ../../vendor/ && cd ../../..
python package_plugin.py rdkit_nodes
```

## License

CC BY-NC 4.0
