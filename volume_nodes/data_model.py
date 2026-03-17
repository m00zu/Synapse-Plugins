"""
data_model.py — 3D volume data types for the volume processing plugin.
"""
from __future__ import annotations
from typing import Any
from data_models import NodeData


class VolumeData(NodeData):
    """3D grayscale volume. payload = np.ndarray shape (Z, H, W)."""
    payload: Any          # np.ndarray (Z, H, W)
    spacing: tuple = (1.0, 1.0, 1.0)  # (dz, dy, dx) voxel size


class VolumeMaskData(NodeData):
    """3D binary mask. payload = np.ndarray shape (Z, H, W), dtype bool."""
    payload: Any          # np.ndarray (Z, H, W) bool
    spacing: tuple = (1.0, 1.0, 1.0)


class VolumeColorData(NodeData):
    """3D color volume. payload = np.ndarray shape (Z, H, W, C), dtype uint8."""
    payload: Any          # np.ndarray (Z, H, W, C)  C = 3 (RGB) or 4 (RGBA)
    spacing: tuple = (1.0, 1.0, 1.0)


class VolumeLabelData(NodeData):
    """3D integer label array. payload = np.ndarray shape (Z, H, W), dtype int32."""
    payload: Any          # np.ndarray (Z, H, W) int32
    spacing: tuple = (1.0, 1.0, 1.0)
