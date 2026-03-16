# Synapse Plugins

Plugins for [Synapse](https://github.com/Ezra-Nemo/Synapse), a visual node-graph workflow editor for scientific data analysis.

## Installing

Download the `.synpkg` file for your platform from [Releases](https://github.com/Ezra-Nemo/Synapse-Plugins/releases), then in Synapse: **Plugins > Install Plugin** and select the file.

Pure Python plugins (image_analysis, statistical_analysis, etc.) are cross-platform — one file works everywhere.

## Plugins

| Plugin | Status | Platform | Description |
|--------|--------|----------|-------------|
| image_analysis | stable | all | Filters, thresholding, morphology, segmentation, measurements, ROI |
| statistical_analysis | stable | all | t-tests, ANOVA, regression, survival analysis, PCA |
| figure_plotting | stable | all | Scatter, box, violin, heatmap, volcano, regression plots |
| filopodia_nodes | stable | all | Cell protrusion detection and measurement |
| volume_nodes | experimental | all | 3D volume rendering and analysis |
| sam2_nodes | stable | macOS, Windows, Linux | SAM2 segmentation, Cellpose, video tracking (ONNX) |
| rdkit_nodes | beta | macOS, Windows, Linux | RDKit chemistry, AutoDock Vina docking (Rust + ONNX) |

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
