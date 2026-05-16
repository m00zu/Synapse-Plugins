"""Verify that importing the plugin registers the imaris_dataset port type."""
import sys
from pathlib import Path

# Add PySide_Node to path so synapse.nodes.base is importable
_PYSIDE_NODE = Path('/Users/s/Desktop/demo/PySide_Node')
if str(_PYSIDE_NODE) not in sys.path:
    sys.path.insert(0, str(_PYSIDE_NODE))


def test_imaris_dataset_port_type_registered():
    """Importing the plugin should call register_port_type('imaris_dataset', ...)."""
    import imaris_3d_nodes.data  # noqa: F401  (side-effect import)
    from synapse.nodes.base import is_port_type_compatible

    # Self-compatibility
    assert is_port_type_compatible('imaris_dataset', 'imaris_dataset')
    # 'any' wildcards should still work
    assert is_port_type_compatible('imaris_dataset', 'any')
    assert is_port_type_compatible('any', 'imaris_dataset')
    # Incompatibility with other types
    assert not is_port_type_compatible('imaris_dataset', 'image')
    assert not is_port_type_compatible('table', 'imaris_dataset')


def test_imaris_dataset_port_color_set():
    import imaris_3d_nodes.data  # noqa: F401
    from synapse.nodes.base import PORT_COLORS

    assert PORT_COLORS.get('imaris_dataset') == (80, 180, 200)
