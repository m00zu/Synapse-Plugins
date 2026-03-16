"""
image_ops_nodes.py — 3D volume image operations (color, filter, projection).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from nodes.base import PORT_COLORS
from nodes.base import BaseImageProcessNode
from data_models import ImageData
from .data_model import VolumeData, VolumeColorData, VolumeMaskData

_VC  = PORT_COLORS.get('volume', (220, 120, 50))
_VCC = PORT_COLORS.get('volume_color', (200, 80, 150))
_MC  = PORT_COLORS.get('volume_mask', (180, 90, 30))


# ── helpers ──────────────────────────────────────────────────────────────────

def _mid_slice_preview(vol: np.ndarray) -> Image.Image:
    """Middle Z-slice as PIL, auto-detects grayscale vs color."""
    mid = vol[vol.shape[0] // 2]
    if mid.dtype == bool:
        return Image.fromarray((mid.astype(np.uint8) * 255), 'L')
    if mid.ndim == 3 and mid.shape[2] in (3, 4):
        if mid.dtype != np.uint8:
            mid = np.clip(mid, 0, 255).astype(np.uint8)
        mode = 'RGB' if mid.shape[2] == 3 else 'RGBA'
        return Image.fromarray(mid, mode)
    mn, mx = float(mid.min()), float(mid.max())
    if mx > mn:
        mid = ((mid.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        mid = np.zeros(mid.shape, dtype=np.uint8)
    return Image.fromarray(mid, 'L')


def _get_volume(node, port_name='volume'):
    port = node.inputs().get(port_name)
    if not port or not port.connected_ports():
        return None
    cp = port.connected_ports()[0]
    return cp.node().output_values.get(cp.name())


# ══════════════════════════════════════════════════════════════════════════════
#  SplitChannels3DNode
# ══════════════════════════════════════════════════════════════════════════════

class SplitChannels3DNode(BaseImageProcessNode):
    """Split a 3D color volume (Z, H, W, 3) into R, G, B channel volumes.

    Keywords: split, channel, RGB, red, green, blue, 3D, 色彩分離, 通道, 體積
    """
    __identifier__ = 'nodes.Volume.Color'
    NODE_NAME      = '3D Split RGB'
    PORT_SPEC      = {'inputs': ['volume_color'],
                      'outputs': ['volume', 'volume', 'volume']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_color', color=_VCC)
        self.add_output('red',   color=(200, 50, 50))
        self.add_output('green', color=(50, 200, 50))
        self.add_output('blue',  color=(50, 50, 200))
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data = _get_volume(self, 'volume_color')
        if not isinstance(data, VolumeColorData):
            return False, "No color volume connected"

        vol = np.asarray(data.payload)
        if vol.ndim != 4 or vol.shape[3] < 3:
            return False, "Expected (Z, H, W, C) with C >= 3"

        self.set_progress(30)
        self.output_values['red']   = VolumeData(payload=vol[:, :, :, 0], spacing=data.spacing)
        self.output_values['green'] = VolumeData(payload=vol[:, :, :, 1], spacing=data.spacing)
        self.output_values['blue']  = VolumeData(payload=vol[:, :, :, 2], spacing=data.spacing)

        self.set_display(_mid_slice_preview(vol))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  MergeChannels3DNode
# ══════════════════════════════════════════════════════════════════════════════

class MergeChannels3DNode(BaseImageProcessNode):
    """Merge R, G, B grayscale volumes into a single 3D color volume.

    Unconnected channels default to zero.

    Keywords: merge, combine, channel, RGB, 3D, 合併, 通道, 色彩合成, 體積
    """
    __identifier__ = 'nodes.Volume.Color'
    NODE_NAME      = '3D Merge RGB'
    PORT_SPEC      = {'inputs': ['volume', 'volume', 'volume'],
                      'outputs': ['volume_color']}

    def __init__(self):
        super().__init__()
        self.add_input('red',   color=(200, 50, 50))
        self.add_input('green', color=(50, 200, 50))
        self.add_input('blue',  color=(50, 50, 200))
        self.add_output('volume_color', color=_VCC)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        channels = {}
        shape = None
        for ch_name in ('red', 'green', 'blue'):
            data = _get_volume(self, ch_name)
            if isinstance(data, VolumeData):
                arr = np.asarray(data.payload)
                channels[ch_name] = arr
                if shape is None:
                    shape = arr.shape
                    spacing = data.spacing

        if shape is None:
            return False, "At least one channel must be connected"

        self.set_progress(30)
        merged = np.zeros((*shape, 3), dtype=np.uint8)
        for i, ch_name in enumerate(('red', 'green', 'blue')):
            if ch_name in channels:
                arr = channels[ch_name]
                # Normalize to uint8 if needed
                if arr.dtype != np.uint8:
                    mn, mx = float(arr.min()), float(arr.max())
                    if mx > mn:
                        arr = ((arr.astype(np.float64) - mn) / (mx - mn) * 255)
                    else:
                        arr = np.zeros_like(arr, dtype=np.float64)
                    arr = arr.astype(np.uint8)
                merged[:, :, :, i] = arr

        self.set_progress(80)
        self.output_values['volume_color'] = VolumeColorData(
            payload=merged, spacing=spacing)
        self.set_display(_mid_slice_preview(merged))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  RGBToGray3DNode
# ══════════════════════════════════════════════════════════════════════════════

class RGBToGray3DNode(BaseImageProcessNode):
    """Convert a 3D color volume to grayscale.

    Methods: Luminosity (Rec.709), Average, or extract a single channel.

    Keywords: grayscale, gray, convert, luminosity, 3D, 灰階, 轉換, 體積
    """
    __identifier__ = 'nodes.Volume.Color'
    NODE_NAME      = '3D RGB to Gray'
    PORT_SPEC      = {'inputs': ['volume_color'], 'outputs': ['volume']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_color', color=_VCC)
        self.add_output('volume', color=_VC)
        self.add_combo_menu('method', 'Method',
                            items=['Luminosity (Rec.709)', 'Average',
                                   'Red channel', 'Green channel', 'Blue channel'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data = _get_volume(self, 'volume_color')
        if not isinstance(data, VolumeColorData):
            return False, "No color volume connected"

        vol = np.asarray(data.payload, dtype=np.float64)
        if vol.ndim != 4 or vol.shape[3] < 3:
            return False, "Expected (Z, H, W, C) with C >= 3"

        method = str(self.get_property('method'))
        self.set_progress(30)

        if 'Luminosity' in method:
            gray = 0.2126 * vol[:, :, :, 0] + 0.7152 * vol[:, :, :, 1] + 0.0722 * vol[:, :, :, 2]
        elif 'Average' in method:
            gray = np.mean(vol[:, :, :, :3], axis=3)
        elif 'Red' in method:
            gray = vol[:, :, :, 0]
        elif 'Green' in method:
            gray = vol[:, :, :, 1]
        else:
            gray = vol[:, :, :, 2]

        gray = gray.astype(np.uint8) if data.payload.dtype == np.uint8 else gray

        self.set_progress(80)
        self.output_values['volume'] = VolumeData(payload=gray, spacing=data.spacing)
        self.set_display(_mid_slice_preview(gray))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  GaussianBlur3DNode
# ══════════════════════════════════════════════════════════════════════════════

class GaussianBlur3DNode(BaseImageProcessNode):
    """Apply 3D Gaussian blur to a volume.

    Sigma can be set independently for Z and XY axes to account for
    anisotropic voxel spacing.

    Keywords: blur, smooth, gaussian, filter, denoise, 3D, 模糊, 平滑, 高斯, 體積
    """
    __identifier__ = 'nodes.Volume.Filters'
    NODE_NAME      = '3D Gaussian Blur'
    PORT_SPEC      = {'inputs': ['volume'], 'outputs': ['volume']}

    def __init__(self):
        super().__init__()
        self.add_input('volume', color=_VC)
        self.add_output('volume', color=_VC)
        self._add_float_spinbox('sigma_xy', 'Sigma XY', value=2.0,
                                min_val=0.1, max_val=100.0, step=0.5)
        self._add_float_spinbox('sigma_z', 'Sigma Z', value=1.0,
                                min_val=0.1, max_val=100.0, step=0.5)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from scipy.ndimage import gaussian_filter

        data = _get_volume(self, 'volume')
        if not isinstance(data, VolumeData):
            return False, "No volume connected"

        vol = np.asarray(data.payload, dtype=np.float64)
        sz = float(self.get_property('sigma_z'))
        sxy = float(self.get_property('sigma_xy'))

        self.set_progress(30)
        blurred = gaussian_filter(vol, sigma=(sz, sxy, sxy))
        self.set_progress(80)

        out = blurred.astype(data.payload.dtype)
        self.output_values['volume'] = VolumeData(payload=out, spacing=data.spacing)
        self.set_display(_mid_slice_preview(out))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  Invert3DNode
# ══════════════════════════════════════════════════════════════════════════════

class Invert3DNode(BaseImageProcessNode):
    """Invert a 3D volume (for uint8: 255 − value; for bool: logical NOT).

    Keywords: invert, negate, reverse, 3D, 反轉, 體積
    """
    __identifier__ = 'nodes.Volume.Exposure'
    NODE_NAME      = '3D Invert'
    PORT_SPEC      = {'inputs': ['volume'], 'outputs': ['volume']}

    def __init__(self):
        super().__init__()
        self.add_input('volume', color=_VC)
        self.add_output('volume', color=_VC)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data = _get_volume(self, 'volume')
        if not isinstance(data, VolumeData):
            return False, "No volume connected"

        vol = data.payload
        self.set_progress(30)

        if vol.dtype == bool:
            result = ~vol
        elif vol.dtype == np.uint8:
            result = np.uint8(255) - vol
        elif vol.dtype == np.uint16:
            result = np.uint16(65535) - vol
        else:
            mn, mx = vol.min(), vol.max()
            result = mx - vol + mn

        self.set_progress(80)
        self.output_values['volume'] = VolumeData(payload=result, spacing=data.spacing)
        self.set_display(_mid_slice_preview(result))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  InvertMask3DNode
# ══════════════════════════════════════════════════════════════════════════════

class InvertMask3DNode(BaseImageProcessNode):
    """Invert a 3D binary mask (logical NOT).

    Keywords: invert, negate, mask, NOT, 3D, 反轉, 遮罩, 體積
    """
    __identifier__ = 'nodes.Volume.Exposure'
    NODE_NAME      = '3D Invert Mask'
    PORT_SPEC      = {'inputs': ['volume_mask'], 'outputs': ['volume_mask']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_mask', color=_MC)
        self.add_output('volume_mask', color=_MC)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        port = self.inputs().get('volume_mask')
        if not port or not port.connected_ports():
            return False, "No volume mask connected"
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, VolumeMaskData):
            return False, "Input must be VolumeMaskData"

        self.set_progress(30)
        result = ~np.asarray(data.payload, dtype=bool)
        self.set_progress(80)

        self.output_values['volume_mask'] = VolumeMaskData(
            payload=result, spacing=data.spacing)
        self.set_display(_mid_slice_preview(result))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  MaxProjectionNode / MinProjectionNode / MeanProjectionNode
# ══════════════════════════════════════════════════════════════════════════════

class MaxProjectionNode(BaseImageProcessNode):
    """Maximum Intensity Projection (MIP) along an axis.

    Collapses a 3D volume to a 2D image by taking the max value per pixel.
    Commonly used in fluorescence microscopy to visualize Z-stacks.

    Keywords: MIP, max, projection, intensity, 3D, 最大, 投影, 體積
    """
    __identifier__ = 'nodes.Volume.Exposure'
    NODE_NAME      = '3D Max Projection'
    PORT_SPEC      = {'inputs': ['volume'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('volume', color=_VC)
        self.add_output('image', color=PORT_COLORS['image'])
        self.add_combo_menu('axis', 'Axis', items=['Z (top-down)', 'Y (front)', 'X (side)'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data = _get_volume(self, 'volume')
        if not isinstance(data, VolumeData):
            return False, "No volume connected"

        vol = data.payload
        axis_str = str(self.get_property('axis'))
        ax = 0
        if 'Y' in axis_str:
            ax = 1
        elif 'X' in axis_str:
            ax = 2

        self.set_progress(30)
        proj = np.max(vol, axis=ax)
        self.set_progress(70)

        # Normalize to uint8 for PIL
        if proj.dtype != np.uint8:
            mn, mx = float(proj.min()), float(proj.max())
            if mx > mn:
                proj = ((proj.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                proj = np.zeros(proj.shape, dtype=np.uint8)

        pil_img = Image.fromarray(proj, 'L')
        self.output_values['image'] = ImageData(payload=pil_img)
        self.set_display(pil_img)
        self.set_progress(100)
        return True, None


class MinProjectionNode(BaseImageProcessNode):
    """Minimum Intensity Projection along an axis.

    Keywords: min, projection, intensity, 3D, 最小, 投影, 體積
    """
    __identifier__ = 'nodes.Volume.Exposure'
    NODE_NAME      = '3D Min Projection'
    PORT_SPEC      = {'inputs': ['volume'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('volume', color=_VC)
        self.add_output('image', color=PORT_COLORS['image'])
        self.add_combo_menu('axis', 'Axis', items=['Z (top-down)', 'Y (front)', 'X (side)'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data = _get_volume(self, 'volume')
        if not isinstance(data, VolumeData):
            return False, "No volume connected"

        vol = data.payload
        axis_str = str(self.get_property('axis'))
        ax = 0
        if 'Y' in axis_str:
            ax = 1
        elif 'X' in axis_str:
            ax = 2

        self.set_progress(30)
        proj = np.min(vol, axis=ax)
        self.set_progress(70)

        if proj.dtype != np.uint8:
            mn, mx = float(proj.min()), float(proj.max())
            if mx > mn:
                proj = ((proj.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                proj = np.zeros(proj.shape, dtype=np.uint8)

        pil_img = Image.fromarray(proj, 'L')
        self.output_values['image'] = ImageData(payload=pil_img)
        self.set_display(pil_img)
        self.set_progress(100)
        return True, None


class MeanProjectionNode(BaseImageProcessNode):
    """Mean Intensity Projection along an axis.

    Keywords: mean, average, projection, intensity, 3D, 平均, 投影, 體積
    """
    __identifier__ = 'nodes.Volume.Exposure'
    NODE_NAME      = '3D Mean Projection'
    PORT_SPEC      = {'inputs': ['volume'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('volume', color=_VC)
        self.add_output('image', color=PORT_COLORS['image'])
        self.add_combo_menu('axis', 'Axis', items=['Z (top-down)', 'Y (front)', 'X (side)'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data = _get_volume(self, 'volume')
        if not isinstance(data, VolumeData):
            return False, "No volume connected"

        vol = data.payload
        axis_str = str(self.get_property('axis'))
        ax = 0
        if 'Y' in axis_str:
            ax = 1
        elif 'X' in axis_str:
            ax = 2

        self.set_progress(30)
        proj = np.mean(vol.astype(np.float64), axis=ax)
        self.set_progress(70)

        mn, mx = float(proj.min()), float(proj.max())
        if mx > mn:
            proj = ((proj - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            proj = np.zeros(proj.shape, dtype=np.uint8)

        pil_img = Image.fromarray(proj, 'L')
        self.output_values['image'] = ImageData(payload=pil_img)
        self.set_display(pil_img)
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  MaskVolume3DNode  — apply mask to volume (zero outside)
# ══════════════════════════════════════════════════════════════════════════════

class MaskVolume3DNode(BaseImageProcessNode):
    """Apply a 3D mask to a volume — zero out voxels outside the mask.

    Keywords: mask, apply, crop, extract, 3D, 遮罩, 套用, 體積
    """
    __identifier__ = 'nodes.Volume.Exposure'
    NODE_NAME      = '3D Apply Mask'
    PORT_SPEC      = {'inputs': ['volume', 'volume_mask'], 'outputs': ['volume']}

    def __init__(self):
        super().__init__()
        self.add_input('volume', color=_VC)
        self.add_input('volume_mask', color=_MC)
        self.add_output('volume', color=_VC)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        vdata = _get_volume(self, 'volume')
        if not isinstance(vdata, VolumeData):
            return False, "No volume connected"

        mport = self.inputs().get('volume_mask')
        if not mport or not mport.connected_ports():
            return False, "No mask connected"
        cp = mport.connected_ports()[0]
        mdata = cp.node().output_values.get(cp.name())
        if not isinstance(mdata, VolumeMaskData):
            return False, "Mask input must be VolumeMaskData"

        vol = np.array(vdata.payload)
        mask = np.asarray(mdata.payload, dtype=bool)

        if vol.shape != mask.shape:
            return False, f"Shape mismatch: volume {vol.shape} vs mask {mask.shape}"

        self.set_progress(30)
        vol[~mask] = 0
        self.set_progress(80)

        self.output_values['volume'] = VolumeData(payload=vol, spacing=vdata.spacing)
        self.set_display(_mid_slice_preview(vol))
        self.set_progress(100)
        return True, None
