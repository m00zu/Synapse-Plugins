"""Shared inline-widget helpers for Imaris 3D nodes."""
from __future__ import annotations

import NodeGraphQt
from nodes.base import NodeDirSelector


def add_dir_picker(node, name: str, label: str, tab: str = 'I/O') -> None:
    """Embed a NodeDirSelector (line-edit + folder-browse button) as a property.

    Mirrors the pattern used in synapse/nodes/utility_nodes.py PathModifier.
    """
    selector = NodeDirSelector(node.view, name=name, label=label)
    node.add_custom_widget(
        selector,
        widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
        tab=tab,
    )
