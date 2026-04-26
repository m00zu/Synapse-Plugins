"""
model_io_nodes.py
=================
Save and load sklearn models (joblib format).
"""
from nodes.base import BaseExecutionNode, PORT_COLORS, NodeFileSelector, NodeFileSaver
from data_models import TableData
from .ml_data import SklearnModelData, SKLEARN_PORT_COLOR
import NodeGraphQt


class ModelSaveNode(BaseExecutionNode):
    """
    Saves a trained sklearn model to disk using joblib.

    Supports `.joblib` and `.pkl` file extensions.

    Keywords: save model, export model, joblib, pickle, ML, machine learning
    """
    __identifier__ = 'plugins.ML.IO'
    NODE_NAME = 'Model Save'
    PORT_SPEC = {'inputs': ['sklearn_model', 'path'], 'outputs': []}

    def __init__(self):
        super().__init__()
        self.add_input('model', color=SKLEARN_PORT_COLOR)
        self.add_input('file_path_in', color=PORT_COLORS.get('path'))

        file_selector = NodeFileSaver(self.view, name='file_path', label='Save Path')
        self.add_custom_widget(
            file_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties'
        )
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        import joblib

        # Get model
        port = self.inputs().get('model')
        if not port or not port.connected_ports():
            self.mark_error()
            return False, "No model connected"
        cp = port.connected_ports()[0]
        model_data = cp.node().output_values.get(cp.name())
        if not isinstance(model_data, SklearnModelData):
            self.mark_error()
            return False, "Input is not a SklearnModelData"

        self.set_progress(20)

        # Get path
        path_port = self.inputs().get('file_path_in')
        if path_port and path_port.connected_ports():
            pcp = path_port.connected_ports()[0]
            file_path = pcp.node().output_values.get(pcp.name(), None)
        else:
            file_path = self.get_property("file_path")

        if not file_path:
            self.mark_error()
            return False, "No file path specified"

        self.set_progress(50)

        # Save the full SklearnModelData (model + metadata)
        save_dict = {
            'model': model_data.payload,
            'model_type': model_data.model_type,
            'feature_names': model_data.feature_names,
            'target_name': model_data.target_name,
            'score': model_data.score,
            'task': model_data.task,
        }
        try:
            joblib.dump(save_dict, file_path)
        except Exception as e:
            self.mark_error()
            return False, f"Save failed: {e}"

        self.mark_clean()
        self.set_progress(100)
        return True, None


class ModelLoadNode(BaseExecutionNode):
    """
    Loads a trained sklearn model from a joblib file.

    Outputs a SklearnModelData that can be connected to Predict or
    Cross Validation nodes.

    Keywords: load model, import model, joblib, pickle, ML, machine learning
    """
    __identifier__ = 'plugins.ML.IO'
    NODE_NAME = 'Model Load'
    PORT_SPEC = {'inputs': ['path'], 'outputs': ['sklearn_model']}

    def __init__(self):
        super().__init__()
        self.add_input('file_path', color=PORT_COLORS.get('path'))
        self.add_output('model', color=SKLEARN_PORT_COLOR)

        file_selector = NodeFileSelector(self.view, name='file_path', label='Model Path')
        self.add_custom_widget(
            file_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties'
        )
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        import joblib
        import os

        in_port = self.inputs().get('file_path')
        if in_port and in_port.connected_ports():
            cp = in_port.connected_ports()[0]
            file_path = cp.node().output_values.get(cp.name(), None)
        else:
            file_path = self.get_property("file_path")

        if not file_path or not os.path.exists(file_path):
            self.mark_error()
            return False, f"File not found: {file_path}"

        self.set_progress(30)

        try:
            loaded = joblib.load(file_path)
        except Exception as e:
            self.mark_error()
            return False, f"Load failed: {e}"

        self.set_progress(70)

        if isinstance(loaded, dict) and 'model' in loaded:
            model_data = SklearnModelData(
                payload=loaded['model'],
                model_type=loaded.get('model_type', ''),
                feature_names=loaded.get('feature_names', []),
                target_name=loaded.get('target_name', ''),
                score=loaded.get('score'),
                task=loaded.get('task', 'classification'),
            )
        else:
            # Raw sklearn model without metadata
            model_data = SklearnModelData(
                payload=loaded,
                model_type=type(loaded).__name__,
            )

        self.output_values['model'] = model_data
        self.mark_clean()
        self.set_progress(100)
        return True, None
