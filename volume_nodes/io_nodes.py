"""
io_nodes.py — Load Z-stack TIFF files as 3D volumes.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from nodes.base import BaseExecutionNode, PORT_COLORS, NodeFileSelector
from nodes.base import BaseImageProcessNode
from .data_model import VolumeData, VolumeColorData


class LoadZStackNode(BaseImageProcessNode):
    """Load a multi-page TIFF file as a 3D volume.

    Each page in the TIFF becomes one Z-slice.
    Mode "Grayscale" outputs VolumeData (Z, H, W).
    Mode "Color (RGB)" outputs VolumeColorData (Z, H, W, 3).

    Keywords: load, tiff, z-stack, volume, 3D, import, 載入, 堆疊, 體積
    """
    __identifier__ = 'nodes.Volume.IO'
    NODE_NAME      = '3D Load Z-Stack'
    PORT_SPEC      = {'inputs': [], 'outputs': ['volume', 'volume_color']}

    def __init__(self):
        super().__init__()
        self.add_output('volume', color=PORT_COLORS.get('volume', (220, 120, 50)))
        self.add_output('volume_color',
                        color=PORT_COLORS.get('volume_color', (200, 80, 150)))

        self._file_widget = NodeFileSelector(
            self.view, name='file_path', label='TIFF File',
            ext_filter='TIFF Files (*.tif *.tiff);;All Files (*)')
        self.add_custom_widget(self._file_widget)
        self._add_float_spinbox('z_spacing', 'Z Spacing', value=1.0,
                                min_val=0.001, max_val=1000.0, step=0.1)
        self._add_float_spinbox('xy_spacing', 'XY Spacing', value=1.0,
                                min_val=0.001, max_val=1000.0, step=0.1)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        path = str(self.get_property('file_path') or '').strip()
        if not path:
            return False, "No TIFF file selected"

        try:
            im = Image.open(path)
        except Exception as e:
            return False, f"Cannot open file: {e}"

        frames_gray = []
        frames_color = []
        has_color = False
        try:
            i = 0
            while True:
                im.seek(i)
                arr = np.array(im)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    has_color = True
                    frames_color.append(arr[:, :, :3].astype(np.uint8))
                    gray = np.mean(arr[:, :, :3], axis=2).astype(np.uint8)
                else:
                    frames_color.append(np.stack([arr, arr, arr], axis=-1).astype(np.uint8))
                    gray = arr
                frames_gray.append(gray)
                i += 1
        except EOFError:
            pass

        if not frames_gray:
            return False, "No frames found in TIFF"

        self.set_progress(60)
        volume_gray = np.stack(frames_gray, axis=0)
        dz = float(self.get_property('z_spacing') or 1.0)
        dxy = float(self.get_property('xy_spacing') or 1.0)
        spacing = (dz, dxy, dxy)

        self.output_values['volume'] = VolumeData(
            payload=volume_gray, spacing=spacing)

        # Always output color volume (even if input is grayscale → triplicated)
        volume_color = np.stack(frames_color, axis=0)
        self.output_values['volume_color'] = VolumeColorData(
            payload=volume_color, spacing=spacing)

        # Preview: middle slice
        mid = volume_gray[volume_gray.shape[0] // 2]
        if mid.dtype != np.uint8:
            mn, mx = mid.min(), mid.max()
            if mx > mn:
                mid = ((mid - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                mid = np.zeros_like(mid, dtype=np.uint8)
        if has_color:
            mid_c = volume_color[volume_color.shape[0] // 2]
            self.set_display(Image.fromarray(mid_c, 'RGB'))
        else:
            self.set_display(Image.fromarray(mid, 'L'))
        self.set_progress(100)
        return True, None
