"""
nodes/image_process_nodes.py
============================
Nodes for image processing using skimage and scipy.ndimage.
"""
from data_models import ImageData, MaskData, LabelData
from PIL import Image
import numpy as np
import threading
from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from NodeGraphQt.nodes.base_node import NodeBaseWidget
from nodes.base import BaseExecutionNode, PORT_COLORS, NodeImageWidget, NodeIntSpinBoxWidget, NodeFloatSpinBoxWidget, BaseImageProcessNode, _arr_to_pil


# _arr_to_pil and BaseImageProcessNode are imported from nodes.base above.


class BitDepthConvertNode(BaseExecutionNode):
    """
    Changes the bit depth metadata of an image.

    The internal float [0,1] data is unchanged. This node only updates
    the bit_depth tag so that saving and display use the target range.

    Options:

    - **target_depth** — output bit depth (8, 12, 14, 16)

    Keywords: bit depth, convert, 8-bit, 16-bit, 12-bit, dynamic range
    """
    __identifier__ = 'nodes.image_process.Transform'
    NODE_NAME = 'Bit Depth Convert'
    PORT_SPEC = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'], multi_output=True)
        self.add_combo_menu('target_depth', 'Target Bit Depth', items=['8', '12', '14', '16'])
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No image connected"
        cp = in_port.connected_ports()[0]
        img_data = cp.node().output_values.get(cp.name())
        if not isinstance(img_data, ImageData):
            self.mark_error()
            return False, "Input is not an ImageData"

        target = int(self.get_property('target_depth') or 8)
        self.set_progress(50)

        kwargs = {f: getattr(img_data, f, None) for f in img_data.model_fields if f != 'payload'}
        kwargs['bit_depth'] = target
        self.output_values['image'] = ImageData(payload=img_data.payload, **kwargs)
        self.mark_clean()
        self.set_progress(100)
        return True, None


class SetScaleNode(BaseExecutionNode):
    """
    Manually set the pixel scale (µm/pixel) for an image.

    Use this when the image has no embedded calibration data (e.g. plain TIFF or PNG).

    Options:

    - **um_per_pixel** — micrometers per pixel
    - **known_distance** — a known real-world distance in µm
    - **distance_pixels** — the same distance measured in pixels

    If both known_distance and distance_pixels are set, um_per_pixel is calculated automatically.

    Keywords: set scale, calibrate, pixel size, micrometer, manual scale, 設定比例尺, 校正, 像素大小
    """
    __identifier__ = 'nodes.image_process.Visualize'
    NODE_NAME      = 'Set Scale'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'], multi_output=True)

        self._add_float_spinbox('um_per_pixel', 'µm / pixel',
                                value=1.0, min_val=0.001, max_val=10000.0, step=0.01, decimals=4)
        self._add_float_spinbox('known_distance', 'Known Distance (µm)',
                                value=0.0, min_val=0.0, max_val=100000.0, step=1.0, decimals=2)
        self._add_float_spinbox('distance_pixels', 'Distance (pixels)',
                                value=0.0, min_val=0.0, max_val=100000.0, step=1.0, decimals=1)
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No image connected"
        connected = in_port.connected_ports()[0]
        img_data = connected.node().output_values.get(connected.name())
        if not isinstance(img_data, ImageData):
            self.mark_error()
            return False, "Input is not an ImageData"
        self.set_progress(30)

        known_dist = float(self.get_property('known_distance') or 0)
        dist_px = float(self.get_property('distance_pixels') or 0)
        if known_dist > 0 and dist_px > 0:
            um_per_px = known_dist / dist_px
        else:
            um_per_px = float(self.get_property('um_per_pixel') or 1.0)
        if um_per_px <= 0:
            self.mark_error()
            return False, "Scale must be > 0"
        self.set_progress(60)

        kwargs = {f: getattr(img_data, f, None) for f in img_data.model_fields if f != 'payload'}
        kwargs['scale_um'] = um_per_px
        self.output_values['image'] = ImageData(payload=img_data.payload, **kwargs)
        self.mark_clean()
        self.set_progress(100)
        return True, None


class SplitRGBNode(BaseImageProcessNode):
    """
    Splits an RGB image into its individual Red, Green, and Blue channels.

    Each output is a single-channel grayscale image corresponding to one color plane.

    Keywords: split, channel, red, green, blue, 色彩分離, 通道, 灰階, 影像處理, 色調
    """
    __identifier__ = 'nodes.image_process.color'
    NODE_NAME = 'Split RGB'
    PORT_SPEC = {'inputs': ['image'], 'outputs': ['image', 'image', 'image']}
    _collection_aware = True

    def __init__(self):
        super(SplitRGBNode, self).__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('red', color=PORT_COLORS.get('image', (200, 50, 50)))
        self.add_output('green', color=PORT_COLORS.get('image', (50, 200, 50)))
        self.add_output('blue', color=PORT_COLORS.get('image', (50, 50, 200)))
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            return False, "No input connected"
        up_node = in_port.connected_ports()[0].node()
        data = up_node.output_values.get(in_port.connected_ports()[0].name())
        
        if not isinstance(data, ImageData):
            return False, "Input must be ImageData"
            
        arr = data.payload
        if arr.ndim == 2:
            # Grayscale — treat as all three channels identical
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        red   = arr[:, :, 0]
        green = arr[:, :, 1]
        blue  = arr[:, :, 2]

        self.set_progress(90)
        self._make_image_output(red, 'red')
        self._make_image_output(green, 'green')
        self._make_image_output(blue, 'blue')

        self.set_display(arr)
        self.set_progress(100)
        return True, None


class MergeRGBNode(BaseImageProcessNode):
    """
    Merges three single-channel grayscale images into a single RGB image.

    Any unconnected channel defaults to zero (black). All connected channels must have the same dimensions.

    Keywords: merge, combine, channel, red, green, 合併, 通道, 色彩合成, 影像處理, 色調
    """
    __identifier__ = 'nodes.image_process.color'
    NODE_NAME = 'Merge RGB'
    PORT_SPEC = {'inputs': ['image', 'image', 'image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super(MergeRGBNode, self).__init__()
        self.add_input('red', color=PORT_COLORS.get('image', (200, 50, 50)))
        self.add_input('green', color=PORT_COLORS.get('image', (50, 200, 50)))
        self.add_input('blue', color=PORT_COLORS.get('image', (50, 50, 200)))
        self.add_output('image', color=PORT_COLORS['image'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        channels = {'red': None, 'green': None, 'blue': None}
        for color in channels:
            port = self.inputs().get(color)
            if port and port.connected_ports():
                up = port.connected_ports()[0].node()
                val = up.output_values.get(port.connected_ports()[0].name())
                if isinstance(val, ImageData):
                    arr = val.payload
                    # Handle if the channel is 2D or 3D
                    if arr.ndim == 3:
                        arr = arr[:, :, 0]
                    channels[color] = arr

        # Default size based on one of the available channels
        sizes = [c.shape for c in channels.values() if c is not None]
        if not sizes:
            return False, "At least one input must be connected"
        
        shape = sizes[0]
        for color, arr in channels.items():
            if arr is None:
                channels[color] = np.zeros(shape, dtype=np.uint8)
            elif arr.shape != shape:
                return False, f"Mismatched shapes: {shape} vs {arr.shape}"
                
        self.set_progress(80)
        merged = np.stack([channels['red'], channels['green'], channels['blue']], axis=-1)

        self._make_image_output(merged)
        self.set_display(merged)
        self.set_progress(100)
        return True, None


class EqualizeAdapthistNode(BaseImageProcessNode):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast.

    **clip_limit** — controls contrast amplification; lower values produce subtler enhancement (default: 0.01).

    Keywords: CLAHE, equalize, adaptive, histogram, contrast, 均衡化, 對比, 自適應, 亮度, 影像處理
    """
    __identifier__ = 'nodes.image_process.Exposure'
    NODE_NAME = 'Equalize Adapthist'
    PORT_SPEC = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super(EqualizeAdapthistNode, self).__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])
        self._add_float_spinbox('clip_limit', 'Clip Limit', value=0.01,
                                min_val=0.0, max_val=1.0, step=0.001, decimals=3)
        self.create_preview_widgets()


    def evaluate(self):
        self.reset_progress()

        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            return False, "No input connected"
        up_node = in_port.connected_ports()[0].node()
        data = up_node.output_values.get(in_port.connected_ports()[0].name())

        if not isinstance(data, ImageData):
            return False, "Input must be ImageData"

        try:
            clip_limit = float(self.get_property('clip_limit'))
        except ValueError:
            clip_limit = 0.01

        arr = data.payload
        self.set_progress(20)

        # Try Rust first (accepts float32), fall back to skimage
        rs_ok = False
        try:
            import image_process_rs as _rs
            arr_f = arr.astype(np.float32)
            if arr_f.ndim == 2:
                out_arr = np.asarray(_rs.clahe(np.ascontiguousarray(arr_f),
                    max(arr_f.shape[0], arr_f.shape[1]) // 8, clip_limit), dtype=np.float32)
                rs_ok = True
            elif arr_f.ndim == 3:
                channels = [np.asarray(_rs.clahe(np.ascontiguousarray(arr_f[:, :, c]),
                    max(arr_f.shape[0], arr_f.shape[1]) // 8, clip_limit), dtype=np.float32)
                            for c in range(arr_f.shape[2])]
                out_arr = np.stack(channels, axis=2)
                rs_ok = True
        except Exception:
            pass

        if not rs_ok:
            from skimage.exposure import equalize_adapthist
            # Data is float [0,1] — skimage handles this directly
            res = equalize_adapthist(arr.astype(float), clip_limit=clip_limit)
            out_arr = res.astype(np.float32)

        self.set_progress(80)

        self._make_image_output(out_arr)
        self.set_display(out_arr)
        self.set_progress(100)
        return True, None


class GaussianBlurNode(BaseImageProcessNode):
    """
    Applies a Gaussian blur to smooth the image.

    **sigma** — standard deviation of the Gaussian kernel; larger values produce stronger blurring (default: 10.0).

    Keywords: blur, smooth, gaussian, filter, denoise, 模糊, 平滑, 高斯, 去噪, 影像處理
    """
    __identifier__ = 'nodes.image_process.filter'
    NODE_NAME = 'Gaussian Blur'
    PORT_SPEC = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super(GaussianBlurNode, self).__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])
        self._add_float_spinbox('sigma', 'Sigma (Blur Amount)', value=10.0,
                                min_val=0.0, max_val=500.0, step=0.5, decimals=1)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        
        from skimage.filters import gaussian
        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            return False, "No input connected"
        up_node = in_port.connected_ports()[0].node()
        data = up_node.output_values.get(in_port.connected_ports()[0].name())
        
        if not isinstance(data, ImageData):
            return False, "Input must be ImageData"
            
        try:
            sigma = float(self.get_property('sigma'))
        except ValueError:
            sigma = 10.0
            
        arr = data.payload
        self.set_progress(30)
        # skimage gaussian preserves range if preserve_range=True
        res = gaussian(arr, sigma=sigma, preserve_range=True)
        self.set_progress(90)

        out_arr = res.astype(arr.dtype)
        self._make_image_output(out_arr)
        self.set_display(out_arr)
        self.set_progress(100)
        return True, None


class ThresholdLocalNode(BaseImageProcessNode):
    """
    Applies adaptive local thresholding to produce a binary mask.

    Computes a threshold for each pixel based on its local neighbourhood, making it robust to uneven illumination.

    **block_size** — size of the local neighbourhood (must be odd; default: 25).

    Keywords: local, adaptive, threshold, segmentation, binary, 閾值, 自適應, 分割, 二值化, 影像處理
    """
    __identifier__ = 'nodes.image_process.filter'
    NODE_NAME = 'Threshold Local'
    PORT_SPEC = {'inputs': ['image'], 'outputs': ['mask']}
    _collection_aware = True

    def __init__(self):
        super(ThresholdLocalNode, self).__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self._add_int_spinbox('block_size', 'Block Size (odd)', value=25, min_val=3, max_val=999)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        
        from skimage.filters import threshold_local
        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            return False, "No input connected"
        up_node = in_port.connected_ports()[0].node()
        data = up_node.output_values.get(in_port.connected_ports()[0].name())
        
        if not isinstance(data, ImageData):
            return False, "Input must be ImageData"
            
        try:
            block_size = int(self.get_property('block_size'))
            # block size must be odd
            if block_size % 2 == 0:
                block_size += 1
        except ValueError:
            block_size = 25
            
        arr = data.payload

        self.set_progress(40)
        thresh = threshold_local(arr, block_size)
        binary = arr > thresh
        self.set_progress(90)

        # Ensure it's exactly 2D before output
        binary_sq = np.squeeze(binary)

        out_arr = (binary_sq.astype(np.uint8)) * 255
        self.output_values['mask'] = MaskData(payload=out_arr)
        self.set_display(out_arr)
        self.set_progress(100)
        return True, None


class RemoveSmallObjectsNode(BaseImageProcessNode):
    """
    Removes small connected components from a binary mask.

    Objects with area at or below the threshold are discarded. Useful for cleaning noise after thresholding.

    **max_size** — maximum object area in pixels to remove (default: 500).

    Keywords: remove, small, objects, noise, clean, 移除, 去噪, 影像處理, 粒子, 遮罩
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME = 'Remove Small Obj'
    PORT_SPEC = {'inputs': ['mask'], 'outputs': ['mask']}
    _collection_aware = True

    def __init__(self):
        super(RemoveSmallObjectsNode, self).__init__()
        self.add_input('mask', color=PORT_COLORS['mask'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self._add_int_spinbox('max_size', 'Maximum Size (px²)', value=500, min_val=0, max_val=999999)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.morphology import remove_small_objects
        
        in_port = self.inputs().get('mask')
        if not in_port or not in_port.connected_ports():
            return False, "No input connected"
        up_node = in_port.connected_ports()[0].node()
        data = up_node.output_values.get(in_port.connected_ports()[0].name())
        
        if not isinstance(data, MaskData):
            return False, "Input must be MaskData"
        try:
            max_size = int(self.get_property('max_size'))
        except ValueError:
            max_size = 500
            
        arr = data.payload
        binary = arr > 0

        self.set_progress(50)
        binary_clean = remove_small_objects(binary, max_size=max_size)
        self.set_progress(90)

        out_arr = (binary_clean.astype(np.uint8)) * 255
        self.output_values['mask'] = MaskData(payload=out_arr)
        self.set_display(out_arr)
        self.set_progress(100)
        return True, None


class RemoveSmallHolesNode(BaseImageProcessNode):
    """
    Fills small enclosed holes in a binary mask up to a given area threshold.

    Only holes (background regions fully enclosed by foreground) smaller than the threshold are filled; larger holes remain. Contrast with FillHolesNode which fills all holes regardless of size.

    Keywords: holes, fill, small, binary, interior, 填孔, 二值化, 遮罩, 閉合, 內部
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME = 'Remove Small Holes'
    PORT_SPEC = {'inputs': ['mask'], 'outputs': ['mask']}
    _collection_aware = True

    def __init__(self):
        super(RemoveSmallHolesNode, self).__init__()
        self.add_input('mask', color=PORT_COLORS['mask'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self._add_int_spinbox('max_size', 'Max Hole Size (px²)', value=50, min_val=0, max_val=999999)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.morphology import remove_small_holes
        
        in_port = self.inputs().get('mask')
        if not in_port or not in_port.connected_ports():
            return False, "No input connected"
        up_node = in_port.connected_ports()[0].node()
        data = up_node.output_values.get(in_port.connected_ports()[0].name())
        
        if not isinstance(data, MaskData):
            return False, "Input must be MaskData"
            
        try:
            max_size = int(self.get_property('max_size'))
        except ValueError:
            max_size = 50
            
        arr = data.payload
        binary = arr > 0

        self.set_progress(50)
        binary_clean = remove_small_holes(binary, max_size=max_size)
        self.set_progress(90)

        out_arr = (binary_clean.astype(np.uint8)) * 255
        self.output_values['mask'] = MaskData(payload=out_arr)
        self.set_display(out_arr)
        self.set_progress(100)
        return True, None


class KeepMaxIntensityRegionNode(BaseImageProcessNode):
    """
    Keeps the top N connected components ranked by total intensity in the reference image.

    Finds all connected components in the mask, sums each region's pixel intensities from the intensity image input, and retains only the brightest regions as a binary mask.

    **top_n** — number of highest-intensity regions to keep (default: 5).

    Keywords: brightest, intensity, max, keep, single region, 亮度, 強度, 保留, 區域, 遮罩
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME = 'Keep Max Intensity'
    PORT_SPEC = {'inputs': ['mask', 'image'], 'outputs': ['mask']}
    _collection_aware = True

    def __init__(self):
        super(KeepMaxIntensityRegionNode, self).__init__()
        self.add_input('mask', color=PORT_COLORS['mask'])
        self.add_input('intensity_image', color=PORT_COLORS['image'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self._add_int_spinbox('top_n', 'Top N Regions (by area)', value=5, min_val=1, max_val=100)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.measure import label, regionprops
        
        
        mask_port = self.inputs().get('mask')
        if not mask_port or not mask_port.connected_ports():
            return False, "Mask input not connected"
        mask_up_node = mask_port.connected_ports()[0].node()
        mask_data = mask_up_node.output_values.get(mask_port.connected_ports()[0].name())

        int_port = self.inputs().get('intensity_image')
        if not int_port or not int_port.connected_ports():
            return False, "Intensity image input not connected"
        int_up_node = int_port.connected_ports()[0].node()
        int_data = int_up_node.output_values.get(int_port.connected_ports()[0].name())
        
        if not isinstance(mask_data, MaskData):
            return False, "Mask input must be MaskData"
        if not isinstance(int_data, ImageData):
            return False, "Intensity input must be ImageData"
            
        try:
            top_n = int(self.get_property('top_n'))
        except ValueError:
            top_n = 5
            
        mask_arr = mask_data.payload > 0
        int_arr = int_data.payload
        if int_arr.ndim == 3:
            int_arr = int_arr.mean(axis=2)

        self.set_progress(20)
        # Determine cell groups/labels
        labeled = label(mask_arr)
        regions = regionprops(labeled, intensity_image=int_arr)
        self.set_progress(60)
        
        if not regions:
            # Return empty mask if no regions exist
            out_arr = np.zeros_like(mask_arr, dtype=np.uint8)
            self.output_values['mask'] = MaskData(payload=out_arr)
            self.set_display(out_arr)
            return True, None
            
        # Get top N by area
        regions.sort(key=lambda x: x.area, reverse=True)
        top_candidates = regions[:top_n]
        
        best_region = max(top_candidates, key=lambda x: np.sum(x.image * x.image_intensity))
        self.set_progress(90)
        
        out_mask = (labeled == best_region.label)
        out_arr = (out_mask.astype(np.uint8)) * 255
        self.output_values['mask'] = MaskData(payload=out_arr)
        self.set_display(out_arr)
        self.set_progress(100)
        return True, None


class DistanceRingMaskNode(BaseImageProcessNode):
    """
    Creates an annular (ring) mask by expanding foreground objects outward by a specified distance.

    Computes the Euclidean distance transform of the background and selects pixels within the given range, producing a ring-shaped region around the original mask.

    **local_distance** — maximum expansion distance in pixels (default: 200).

    Keywords: distance, ring, annular, expand, zone, 距離, 環狀, 遮罩, 膨脹, 區域
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME = 'Distance Ring Mask'
    PORT_SPEC = {'inputs': ['mask'], 'outputs': ['mask']}
    _collection_aware = True

    def __init__(self):
        super(DistanceRingMaskNode, self).__init__()
        self.add_input('mask', color=PORT_COLORS['mask'])
        self.add_output('ring_mask', color=PORT_COLORS['mask'])
        self._add_int_spinbox('local_distance', 'Local Distance (px)', value=200, min_val=0, max_val=50000)
        self.add_checkbox('include_original', '', text='Include Original Mask', state=False)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        in_port = self.inputs().get('mask')
        if not in_port or not in_port.connected_ports():
            return False, "No input connected"
        up_node = in_port.connected_ports()[0].node()
        data = up_node.output_values.get(in_port.connected_ports()[0].name())

        if not isinstance(data, MaskData):
            return False, "Input must be MaskData"

        try:
            dist = float(self.get_property('local_distance'))
        except ValueError:
            dist = 200.0

        arr = data.payload
        base_mask = arr > 0

        self.set_progress(40)
        try:
            import image_process_rs as _rs
            inv_mask = (~base_mask).astype(np.uint8)
            edt_d = _rs.distance_transform_edt(inv_mask).astype(np.float64)
        except Exception:
            from scipy.ndimage import distance_transform_edt
            edt_d = distance_transform_edt(~base_mask)
        ring_mask = (edt_d > 0) & (edt_d <= dist)

        if self.get_property('include_original'):
            ring_mask = ring_mask | base_mask

        self.set_progress(90)

        out_arr = (ring_mask.astype(np.uint8)) * 255
        self.output_values['ring_mask'] = MaskData(payload=out_arr)
        self.set_display(out_arr)
        self.set_progress(100)
        return True, None


class ImageMathNode(BaseImageProcessNode):
    """
    Performs pixel-wise arithmetic and logical operations on one or two images or masks.

    Two-input operations (connect both A and B):

    - *A + B* — add and clip to [0, 1]
    - *A - B* — subtract and clip to [0, 1]
    - *A x B (image)* — element-wise multiply
    - *A x B (apply mask)* — mask A by B: `A * (B > 0)`
    - *A AND B* — mask intersection
    - *A OR B* — mask union
    - *Max(A, B)* — element-wise maximum
    - *Min(A, B)* — element-wise minimum
    - *Blend* — weighted blend: `A*v + B*(1-v)`

    Single-input operations (only A is required):

    - *Invert A* — `1 - A`
    - *Normalize A* — stretch to [0, 1]
    - *Gamma A^v* — `A^v` where v is the scalar value
    - *Threshold A > v* — binary threshold

    Custom expression mode:

    - *Custom Expression* — type any math using A, B, v as variables.
      Available functions: abs, sqrt, log, log2, log10, exp, sin, cos, clip, max, min, mean, std.
      Example: `A ** 2 + B`, `clip(A * 2, 0, 1)`, `sqrt(A)`

    Keywords: math, add, subtract, multiply, divide, custom expression, 影像運算, 加法, 減法, 乘法, 影像處理
    """

    __identifier__ = 'nodes.image_process.Exposure'
    NODE_NAME = 'Image Math'
    PORT_SPEC = {'inputs': ['image', 'mask'], 'outputs': ['image', 'mask']}

    _TWO_INPUT_OPS = frozenset({
        'A + B', 'A - B', 'A × B (image)', 'A × B (apply mask)',
        'A AND B', 'A OR B', 'Max(A, B)', 'Min(A, B)',
        'Blend A·v + B·(1-v)',
    })
    _BINARY_OPS = frozenset({
        'A × B (apply mask)', 'A AND B', 'A OR B', 'Threshold A > v',
    })

    def __init__(self):
        super(ImageMathNode, self).__init__()
        self.add_input('A (image/mask)', color=PORT_COLORS['image'])
        self.add_input('B (mask)', color=PORT_COLORS['mask'])
        self.add_output('image', color=PORT_COLORS['image'])
        self.add_output('mask',  color=PORT_COLORS['mask'])

        self.add_combo_menu('operation', 'Operation', items=[
            'A + B',
            'A - B',
            'A × B (image)',
            'A × B (apply mask)',
            'A AND B',
            'A OR B',
            'Max(A, B)',
            'Min(A, B)',
            'Blend A·v + B·(1-v)',
            'Invert A',
            'Normalize A',
            'Gamma A^v',
            'Threshold A > v',
            'Custom Expression',
        ])

        w = NodeFloatSpinBoxWidget(self.view, name='scalar', label='Value',
                                   value=1.0, min_val=0.0, max_val=255.0,
                                   step=0.1, decimals=3)
        self.add_custom_widget(w)
        self.add_text_input('expression', 'Expression', text='A ** 2 + B')
        self.create_preview_widgets()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_arr(port) -> tuple:
        """Return (float32 ndarray, error_str | None) from a connected port."""
        if not port or not port.connected_ports():
            return None, f"Port '{port.name() if port else '?'}' not connected"
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, ImageData):
            return None, f"Port '{port.name()}' must carry ImageData or MaskData"
        return data.payload.astype(np.float32), None

    # ── evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self):
        self.reset_progress()

        op     = self.get_property('operation')
        scalar = float(self.get_property('scalar'))

        # ── input A (always required) ─────────────────────────────────────
        a_arr, err = self._get_arr(self.inputs().get('A (image/mask)'))
        if err:
            return False, err
        self.set_progress(15)

        # ── input B (only for two-input operations) ───────────────────────
        b_arr = None
        if op in self._TWO_INPUT_OPS:
            b_arr, err = self._get_arr(self.inputs().get('B (mask)'))
            if err:
                return False, f"Operation '{op}' needs B — {err}"
            if a_arr.shape[:2] != b_arr.shape[:2]:
                return False, (f"A and B must be the same size "
                               f"(A={a_arr.shape[:2]}, B={b_arr.shape[:2]})")
            # Broadcast single-channel ↔ multi-channel
            if a_arr.ndim == 3 and b_arr.ndim == 2:
                b_arr = b_arr[:, :, np.newaxis]
            elif b_arr.ndim == 3 and a_arr.ndim == 2:
                a_arr = a_arr[:, :, np.newaxis]
        self.set_progress(30)

        # ── compute ───────────────────────────────────────────────────────
        # Data is float32 [0, 1]
        if   op == 'A + B':
            result = np.clip(a_arr + b_arr, 0, 1)
        elif op == 'A - B':
            result = np.clip(a_arr - b_arr, 0, 1)
        elif op == 'A × B (image)':
            result = np.clip(a_arr * b_arr, 0, 1)
        elif op == 'A × B (apply mask)':
            result = a_arr * (b_arr > 0).astype(np.float32)
        elif op == 'A AND B':
            result = ((a_arr > 0) & (b_arr > 0)).astype(np.float32)
        elif op == 'A OR B':
            result = ((a_arr > 0) | (b_arr > 0)).astype(np.float32)
        elif op == 'Max(A, B)':
            result = np.maximum(a_arr, b_arr)
        elif op == 'Min(A, B)':
            result = np.minimum(a_arr, b_arr)
        elif op == 'Blend A·v + B·(1-v)':
            v = np.clip(scalar, 0.0, 1.0)
            result = np.clip(v * a_arr + (1.0 - v) * b_arr, 0, 1)
        elif op == 'Invert A':
            result = 1.0 - a_arr
        elif op == 'Normalize A':
            mn, mx = float(a_arr.min()), float(a_arr.max())
            result = (a_arr - mn) / (mx - mn) if mx > mn else np.zeros_like(a_arr)
        elif op == 'Gamma A^v':
            result = np.power(np.clip(a_arr, 0, 1), max(scalar, 1e-6))
        elif op == 'Threshold A > v':
            result = (a_arr > scalar / 255.0).astype(np.float32)
        elif op == 'Custom Expression':
            expr = str(self.get_property('expression') or 'A').strip()
            if not expr:
                return False, "No expression provided"
            # Safe eval with only numpy math functions
            _safe_ns = {
                'A': a_arr, 'B': b_arr, 'v': scalar,
                'np': np, 'abs': np.abs, 'sqrt': np.sqrt,
                'log': np.log, 'log2': np.log2, 'log10': np.log10,
                'exp': np.exp, 'sin': np.sin, 'cos': np.cos,
                'clip': np.clip, 'max': np.maximum, 'min': np.minimum,
                'mean': np.mean, 'std': np.std,
                'pi': np.pi, 'e': np.e,
            }
            try:
                result = eval(expr, {"__builtins__": {}}, _safe_ns)
                if not isinstance(result, np.ndarray):
                    result = np.full_like(a_arr, float(result))
                result = result.astype(np.float32)
            except Exception as exc:
                return False, f"Expression error: {exc}"
        else:
            return False, f"Unknown operation '{op}'"

        self.set_progress(70)

        # ── build PIL image ───────────────────────────────────────────────
        result = result.astype(np.float32)
        # Collapse trailing size-1 channel from broadcasting
        if result.ndim == 3 and result.shape[2] == 1:
            result = result[:, :, 0]
        mode = {2: 'L', 3: {3: 'RGB', 4: 'RGBA'}}.get(result.ndim, None)
        if isinstance(mode, dict):
            mode = mode.get(result.shape[2], None)
        if mode is None:
            return False, f"Unexpected result shape {result.shape}"

        is_binary  = op in self._BINARY_OPS
        if is_binary:
            wrapped = MaskData(payload=result)
            self.output_values['image'] = wrapped
            self.output_values['mask']  = wrapped
        else:
            self._make_image_output(result)
            self.output_values['mask'] = self.output_values['image']
        self.set_display(result)
        self.set_progress(100)
        return True, None


# ── Colormap helpers (shared by BC and Threshold nodes) ───────────────────────

_COLORMAPS = [
    'gray',     # No colour — passes through unchanged
    'viridis',  # Perceptually uniform dark→light
    'plasma',   # Perceptually uniform purple→yellow
    'inferno',  # Black→red→yellow
    'magma',    # Black→purple→white
    'hot',      # Black→red→yellow→white  (ImageJ "Fire")
    'turbo',    # Full-spectrum, perceptually improved
    'jet',      # Classic rainbow
    'Reds',     # White→red
    'Greens',   # White→green
    'Blues',    # White→blue
]


def _apply_colormap(arr_2d_uint8, cmap_name):
    """
    Apply a matplotlib colormap to a 2D uint8 luminance array.
    Returns an H×W×3 uint8 RGB ndarray.
    When cmap_name == 'gray' returns a plain 3-channel gray stack.
    """
    if cmap_name == 'gray':
        return np.stack([arr_2d_uint8] * 3, axis=-1)
    import matplotlib.cm as mcm
    cmap = mcm.get_cmap(cmap_name)
    rgba = cmap(arr_2d_uint8.astype(np.float32) / 255.0)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


class ColormapBar(QtWidgets.QWidget):
    """
    Paints a thin horizontal gradient strip for a given matplotlib colormap.

    Used as a visual legend aligned with the histogram display.
    """
    _N = 128   # Number of gradient samples

    def __init__(self, parent=None, cmap_name='gray'):
        super().__init__(parent)
        self.setFixedHeight(12)
        self._stops = []
        self.set_colormap(cmap_name)

    def set_colormap(self, name):
        if name == 'gray':
            self._stops = [(0.0, QtGui.QColor(0, 0, 0)),
                           (1.0, QtGui.QColor(255, 255, 255))]
            self.update()
            return
        try:
            import matplotlib.cm as mcm
            cmap = mcm.get_cmap(name)
        except Exception:
            self._stops = [(0.0, QtGui.QColor(0, 0, 0)),
                           (1.0, QtGui.QColor(255, 255, 255))]
            self.update()
            return
        self._stops = []
        for i in range(self._N):
            t = i / (self._N - 1)
            r, g, b, _ = cmap(t)
            self._stops.append(
                (t, QtGui.QColor(int(r * 255), int(g * 255), int(b * 255))))
        self.update()

    def paintEvent(self, event):
        if not self._stops:
            return
        painter = QtGui.QPainter(self)
        grad = QtGui.QLinearGradient(0, 0, self.width(), 0)
        for pos, color in self._stops:
            grad.setColorAt(pos, color)
        painter.fillRect(self.rect(), QtGui.QBrush(grad))
        painter.end()


# ── Brightness / Contrast ─────────────────────────────────────────────────────

class HistogramBCWidget(QtWidgets.QWidget):
    """
    Interactive histogram widget with two draggable InfiniteLines for Min/Max windowing.

    Each parameter has a slider + spinbox pair that syncs with the InfiniteLines on the plot. Dragging any control triggers a live image update immediately.

    Controls:
    - Red line / **Min** — pixels below this value become 0
    - Cyan line / **Max** — pixels above this value become 255
    - **Brightness** — shifts the centre of the window
    - **Contrast** — narrows or widens the window

    Emits `params_changed(min_val, max_val)` on any user interaction.
    """
    params_changed   = QtCore.Signal(float, float)
    colormap_changed = QtCore.Signal(str)

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full_range  = 255.0
        self._updating    = False
        self._hist_counts = None
        self._hist_edges  = None
        self._cmap_name   = 'gray'

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Plot ─────────────────────────────────────────────────────────
        self._plot = pg.PlotWidget(background='#1a1a1a')
        self._plot.setFixedHeight(110)
        self._plot.setMinimumWidth(240)
        self._plot.hideAxis('left')
        ax = self._plot.getAxis('bottom')
        ax.setPen(pg.mkPen('#555555'))
        ax.setTextPen(pg.mkPen('#aaaaaa'))
        tick_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        tick_font.setPointSize(7)
        ax.setStyle(tickFont=tick_font)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.getViewBox().setDefaultPadding(0)

        # Input histogram (gray fill, log scale)
        self._hist_curve = self._plot.plot(
            pen=pg.mkPen('#888888', width=1),
            fillLevel=0,
            brush=pg.mkBrush(120, 120, 120, 100),
            stepMode='center',
        )
        # Output histogram overlay (orange, live-updated while dragging)
        self._out_curve = self._plot.plot(
            pen=pg.mkPen('#e67e22', width=1),
            fillLevel=0,
            brush=pg.mkBrush(230, 126, 34, 60),
            stepMode='center',
        )

        # Shaded window between Min and Max
        self._region = pg.LinearRegionItem(
            values=(0, 255),
            brush=pg.mkBrush(80, 140, 255, 40),
            pen=pg.mkPen(None),
            movable=False,
        )
        self._plot.addItem(self._region)

        self._line_min = pg.InfiniteLine(
            pos=0, angle=90, movable=True,
            pen=pg.mkPen('#e74c3c', width=2),
            label='Min',
            labelOpts={'color': '#e74c3c', 'position': 0.88,
                       'anchors': [(0.5, 0), (0.5, 1)]},
        )
        self._line_max = pg.InfiniteLine(
            pos=255, angle=90, movable=True,
            pen=pg.mkPen('#3498db', width=2),
            label='Max',
            labelOpts={'color': '#3498db', 'position': 0.88,
                       'anchors': [(0.5, 0), (0.5, 1)]},
        )
        self._plot.addItem(self._line_min)
        self._plot.addItem(self._line_max)
        self._line_min.sigPositionChanged.connect(self._on_line_moved)
        self._line_max.sigPositionChanged.connect(self._on_line_moved)

        layout.addWidget(self._plot)

        # ── Colormap selector + gradient bar ─────────────────────────────
        cmap_row = QtWidgets.QHBoxLayout()
        cmap_row.setSpacing(4)
        cmap_row.setContentsMargins(0, 0, 0, 0)
        lbl_cm = QtWidgets.QLabel('Colormap')
        lbl_cm.setStyleSheet('color: #cccccc; font-size: 8pt;')
        lbl_cm.setFixedWidth(66)
        self._cmap_combo = QtWidgets.QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.setFixedWidth(90)
        self._cmap_bar = ColormapBar(cmap_name='gray')
        cmap_row.addWidget(lbl_cm)
        cmap_row.addWidget(self._cmap_combo)
        cmap_row.addWidget(self._cmap_bar, 1)
        layout.addLayout(cmap_row)

        # ── Slider + spinbox rows ─────────────────────────────────────────
        # Layout: Label(col0) | Slider(col1, stretch) | Spinbox(col2, fixed)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(3)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 1)

        # Min / Max  –  integer values in [0, full_range]
        self._sld_min = self._make_int_slider(0, 255, 0)
        self._spn_min = self._make_int_spinbox(0, 255, 0)
        self._sld_max = self._make_int_slider(0, 255, 255)
        self._spn_max = self._make_int_spinbox(0, 255, 255)

        # Brightness / Contrast  –  float in [-100, 100], slider ×10 precision
        self._sld_brightness = self._make_float_slider(-100, 100, 0.0)
        self._spn_brightness = self._make_float_spinbox(-100.0, 100.0, 0.0)
        self._sld_contrast   = self._make_float_slider(-100, 100, 0.0)
        self._spn_contrast   = self._make_float_spinbox(-100.0, 100.0, 0.0)

        _SS = 'color: #cccccc; font-size: 8pt;'
        for row, (lbl_text, sld, spn) in enumerate([
            ('Min',        self._sld_min,        self._spn_min),
            ('Max',        self._sld_max,        self._spn_max),
            ('Brightness', self._sld_brightness, self._spn_brightness),
            ('Contrast',   self._sld_contrast,   self._spn_contrast),
        ]):
            lbl = QtWidgets.QLabel(lbl_text)
            lbl.setStyleSheet(_SS)
            lbl.setFixedWidth(66)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(sld, row, 1)
            grid.addWidget(spn, row, 2)

        layout.addLayout(grid)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_auto  = QtWidgets.QPushButton('Auto')
        self._btn_reset = QtWidgets.QPushButton('Reset')
        for btn in (self._btn_auto, self._btn_reset):
            btn.setFixedHeight(22)
        btn_row.addWidget(self._btn_auto)
        btn_row.addWidget(self._btn_reset)
        layout.addLayout(btn_row)

        # ── Wire up all controls ──────────────────────────────────────────
        # Min / Max sliders and spinboxes
        self._sld_min.valueChanged.connect(lambda v: self._on_minmax_slider(v, 'min'))
        self._spn_min.valueChanged.connect(lambda v: self._on_minmax_spinbox(v, 'min'))
        self._sld_max.valueChanged.connect(lambda v: self._on_minmax_slider(v, 'max'))
        self._spn_max.valueChanged.connect(lambda v: self._on_minmax_spinbox(v, 'max'))

        # Brightness / Contrast sliders and spinboxes
        self._sld_brightness.valueChanged.connect(
            lambda v: self._on_bc_slider(v / 10.0, 'b'))
        self._spn_brightness.valueChanged.connect(
            lambda v: self._on_bc_spinbox(v, 'b'))
        self._sld_contrast.valueChanged.connect(
            lambda v: self._on_bc_slider(v / 10.0, 'c'))
        self._spn_contrast.valueChanged.connect(
            lambda v: self._on_bc_spinbox(v, 'c'))

        self._btn_auto.clicked.connect(self._auto)
        self._btn_reset.clicked.connect(self._reset)
        self._cmap_combo.currentTextChanged.connect(self._on_colormap_changed)

    # ── Widget factory helpers ─────────────────────────────────────────────────

    @staticmethod
    def _make_int_slider(lo, hi, val):
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    @staticmethod
    def _make_int_spinbox(lo, hi, val):
        s = QtWidgets.QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setFixedWidth(62)
        return s

    @staticmethod
    def _make_float_slider(lo, hi, val):
        # Range stored ×10 so we get 0.1-step precision without float sliders
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        s.setRange(int(lo * 10), int(hi * 10))
        s.setValue(int(round(val * 10)))
        return s

    @staticmethod
    def _make_float_spinbox(lo, hi, val):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(1)
        s.setSingleStep(0.5)
        s.setValue(val)
        s.setFixedWidth(62)
        return s

    # ── Public API ────────────────────────────────────────────────────────────

    def set_histogram(self, arr_flat, full_range=255):
        """Plot a log-scale histogram and update control ranges."""
        self._full_range = float(full_range)
        num_bins = min(512, int(full_range) + 1)
        counts, edges = np.histogram(arr_flat, bins=num_bins,
                                     range=(0, float(full_range)))
        self._hist_counts = counts
        self._hist_edges  = edges
        self._hist_curve.setData(x=edges, y=np.log1p(counts.astype(float)))
        self._plot.setXRange(0, full_range, padding=0)
        # Widen slider / spinbox ranges for 16-bit images
        irange = int(full_range)
        for w in (self._sld_min, self._sld_max):
            w.setRange(0, irange)
        for w in (self._spn_min, self._spn_max):
            w.setRange(0, irange)

    def set_range(self, min_val, max_val):
        """Programmatically move all handles (e.g. on node load / evaluate)."""
        self._set_minmax(min_val, max_val)

    def get_range(self):
        return float(self._line_min.value()), float(self._line_max.value())

    # ── Core atomic updater ───────────────────────────────────────────────────

    def _set_minmax(self, min_v, max_v):
        """Update every Min/Max control atomically, then refresh B&C and overlay."""
        self._updating = True
        try:
            self._line_min.setValue(min_v)
            self._line_max.setValue(max_v)
            self._region.setRegion((min_v, max_v))
            self._sld_min.setValue(int(round(min_v)))
            self._sld_max.setValue(int(round(max_v)))
            self._spn_min.setValue(int(round(min_v)))
            self._spn_max.setValue(int(round(max_v)))
            self._refresh_bc_controls(min_v, max_v)
            self._refresh_output_hist(min_v, max_v)
        finally:
            self._updating = False

    def _refresh_bc_controls(self, min_v, max_v):
        """Recompute B/C from Min/Max and push to sliders + spinboxes."""
        full   = self._full_range
        center = (min_v + max_v) / 2.0
        width  = max_v - min_v
        brightness = (center - full / 2.0) / (full / 2.0) * 100.0
        contrast   = (1.0 - width / full) * 100.0
        self._sld_brightness.setValue(int(round(brightness * 10)))
        self._spn_brightness.setValue(brightness)
        self._sld_contrast.setValue(int(round(contrast * 10)))
        self._spn_contrast.setValue(contrast)

    def _refresh_output_hist(self, min_v, max_v):
        """
        Overlay the output distribution in orange.  Because B&C is a linear
        remap, the output histogram is just the input histogram with the x-axis
        stretched from [min_v, max_v] → [0, 255].  Bins outside the window
        pile up at the extremes (clipping).
        """
        if self._hist_counts is None or self._hist_edges is None:
            return
        width = float(max_v - min_v) or 1.0
        out_edges = np.clip(
            (self._hist_edges.astype(np.float64) - min_v) / width * 255.0,
            0.0, 255.0)
        self._out_curve.setData(x=out_edges,
                                y=np.log1p(self._hist_counts.astype(float)))

    # ── Slot: InfiniteLines dragged ───────────────────────────────────────────

    def _on_line_moved(self):
        if self._updating:
            return
        min_v = float(self._line_min.value())
        max_v = float(self._line_max.value())
        if min_v >= max_v:
            return
        self._set_minmax(min_v, max_v)
        self.params_changed.emit(min_v, max_v)

    # ── Slots: Min / Max slider & spinbox ────────────────────────────────────

    def _on_minmax_slider(self, int_val, which):
        if self._updating:
            return
        val   = float(int_val)
        min_v = val if which == 'min' else float(self._line_min.value())
        max_v = val if which == 'max' else float(self._line_max.value())
        if min_v >= max_v:
            return
        self._set_minmax(min_v, max_v)
        self.params_changed.emit(min_v, max_v)

    def _on_minmax_spinbox(self, spn_val, which):
        if self._updating:
            return
        val   = float(spn_val)
        min_v = val if which == 'min' else float(self._line_min.value())
        max_v = val if which == 'max' else float(self._line_max.value())
        if min_v >= max_v:
            return
        self._set_minmax(min_v, max_v)
        self.params_changed.emit(min_v, max_v)

    # ── Slots: Brightness / Contrast slider & spinbox ────────────────────────

    def _on_bc_slider(self, float_val, which):
        """Slider moved: sync its own spinbox silently, then apply B&C."""
        if self._updating:
            return
        self._updating = True
        try:
            if which == 'b':
                self._spn_brightness.setValue(float_val)
            else:
                self._spn_contrast.setValue(float_val)
        finally:
            self._updating = False
        self._apply_bc_spinboxes()

    def _on_bc_spinbox(self, float_val, which):
        """Spinbox edited: sync its own slider silently, then apply B&C."""
        if self._updating:
            return
        self._updating = True
        try:
            if which == 'b':
                self._sld_brightness.setValue(int(round(float_val * 10)))
            else:
                self._sld_contrast.setValue(int(round(float_val * 10)))
        finally:
            self._updating = False
        self._apply_bc_spinboxes()

    def _apply_bc_spinboxes(self):
        """Read brightness/contrast spinboxes and derive new min/max."""
        brightness = self._spn_brightness.value()
        contrast   = self._spn_contrast.value()
        full       = self._full_range
        new_width  = max(1.0, (1.0 - contrast / 100.0) * full)
        center     = full / 2.0 + (brightness / 100.0) * (full / 2.0)
        new_min    = max(0.0, min(center - new_width / 2.0, full - 1.0))
        new_max    = max(1.0, min(center + new_width / 2.0, full))
        if new_min >= new_max:
            return
        self._set_minmax(new_min, new_max)
        self.params_changed.emit(new_min, new_max)

    # ── Auto / Reset ──────────────────────────────────────────────────────────

    def _auto(self):
        """Auto-stretch to 0.5 %/99.5 % of cumulative histogram."""
        if self._hist_counts is None:
            return
        cumsum = np.cumsum(self._hist_counts)
        total  = float(cumsum[-1])
        edges  = self._hist_edges
        lo_idx = int(np.clip(np.searchsorted(cumsum, total * 0.005), 0, len(edges) - 2))
        hi_idx = int(np.clip(np.searchsorted(cumsum, total * 0.995), 0, len(edges) - 2))
        new_min = float(edges[lo_idx])
        new_max = float(edges[hi_idx + 1])
        if new_min >= new_max:
            new_min, new_max = 0.0, self._full_range
        self._set_minmax(new_min, new_max)
        self.params_changed.emit(new_min, new_max)

    def _reset(self):
        self._set_minmax(0.0, self._full_range)
        self.params_changed.emit(0.0, self._full_range)

    # ── Colormap API ──────────────────────────────────────────────────────────

    def get_colormap(self):
        return self._cmap_name

    def set_colormap(self, name):
        """Restore colormap programmatically (e.g. on node load)."""
        self._cmap_name = name
        self._cmap_bar.set_colormap(name)
        self._updating = True
        idx = self._cmap_combo.findText(name)
        if idx >= 0:
            self._cmap_combo.setCurrentIndex(idx)
        self._updating = False

    def _on_colormap_changed(self, name):
        if self._updating:
            return
        self._cmap_name = name
        self._cmap_bar.set_colormap(name)
        self.colormap_changed.emit(name)


class NodeBCWidget(NodeBaseWidget):
    """NodeGraphQt proxy wrapper that embeds a HistogramBCWidget on the node surface."""
    _set_hist_sig = QtCore.Signal(object, float)
    _set_range_sig = QtCore.Signal(float, float)
    _set_cmap_sig = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent, name='bc_widget', label='')
        self._inner = HistogramBCWidget()
        self.set_custom_widget(self._inner)
        self._set_hist_sig.connect(
            self._inner.set_histogram, QtCore.Qt.ConnectionType.QueuedConnection)
        self._set_range_sig.connect(
            self._inner.set_range, QtCore.Qt.ConnectionType.QueuedConnection)
        self._set_cmap_sig.connect(
            self._inner.set_colormap, QtCore.Qt.ConnectionType.QueuedConnection)

    @property
    def inner(self):
        return self._inner

    def get_value(self):
        return list(self._inner.get_range())

    def set_value(self, value):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            self._inner.set_range(float(value[0]), float(value[1]))

    def set_histogram_threadsafe(self, arr_flat, full_range=255):
        if threading.current_thread() is threading.main_thread():
            self._inner.set_histogram(arr_flat, full_range=full_range)
        else:
            self._set_hist_sig.emit(arr_flat, float(full_range))

    def set_range_threadsafe(self, min_val, max_val):
        if threading.current_thread() is threading.main_thread():
            self._inner.set_range(float(min_val), float(max_val))
        else:
            self._set_range_sig.emit(float(min_val), float(max_val))

    def set_colormap_threadsafe(self, name):
        if threading.current_thread() is threading.main_thread():
            self._inner.set_colormap(str(name))
        else:
            self._set_cmap_sig.emit(str(name))


class BrightnessContrastNode(BaseImageProcessNode):
    """
    Adjusts brightness and contrast interactively using a histogram with draggable Min/Max lines.

    Drag the red **Min** line and cyan **Max** line on the histogram to set the display window. The output image is always 8-bit with a linear stretch: `output = clip((pixel - Min) / (Max - Min) * 255, 0, 255)`.

    Works with 8-bit and 16-bit input images. For 16-bit input the histogram spans 0-65535 and the handles can be placed anywhere in that range.

    Convenience controls:
    - **Brightness** (-100 to +100) — shifts the window centre up or down
    - **Contrast** (-100 to +100) — narrows or widens the window symmetrically

    Keywords: brightness, contrast, adjust, enhance, levels, 亮度, 對比, 調整, 增強, 影像處理
    """
    __identifier__ = 'nodes.image_process.Exposure'
    NODE_NAME      = 'Brightness / Contrast'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}
    # These properties are set silently (no mark_dirty / no auto-evaluate)
    _UI_PROPS = BaseImageProcessNode._UI_PROPS | frozenset({'bc_range', 'bc_widget'})

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('image', multi_output=True, color=PORT_COLORS['image'])

        self.create_property('bc_range',    [0.0, 255.0])
        self.create_property('bc_colormap', 'gray')

        self._bc_widget = NodeBCWidget(self.view)
        self.add_custom_widget(self._bc_widget)
        self._bc_widget.inner.params_changed.connect(self._on_params_changed)
        self._bc_widget.inner.colormap_changed.connect(self._on_colormap_changed)

        self.create_preview_widgets()
        self._cached_arr = None   # raw numpy array for fast live re-apply

    # ── Live drag callback ────────────────────────────────────────────────────

    def set_property(self, name, value, push_undo=True):
        super().set_property(name, value, push_undo)
        # Sync the interactive widget whenever bc_range is set externally
        # (e.g. WorkflowLoader). set_range uses _updating=True internally
        # so it will NOT re-emit params_changed.
        if name == 'bc_range' and hasattr(self, '_bc_widget'):
            try:
                self._bc_widget.inner.set_range(float(value[0]), float(value[1]))
            except Exception:
                pass

    def _on_params_changed(self, min_val, max_val):
        """Apply B&C immediately when the user moves a handle or edits a spinbox."""
        super(BaseExecutionNode, self).set_property(
            'bc_range', [min_val, max_val], push_undo=False)
        if self._cached_arr is not None:
            result = self._apply_bc(self._cached_arr, min_val, max_val)
            if result is not None:
                self.output_values['image'] = result
                self.set_display(self._make_display_arr(result.payload))
                # Downstream nodes must re-evaluate on next run graph execution
                for out_port in self.outputs().values():
                    for in_port in out_port.connected_ports():
                        dn = in_port.node()
                        if hasattr(dn, 'mark_dirty'):
                            dn.mark_dirty()


    def _on_colormap_changed(self, cmap_name):
        """Recolour the preview when the user picks a different colormap."""
        super(BaseExecutionNode, self).set_property(
            'bc_colormap', cmap_name, push_undo=False)
        if self._cached_arr is not None and 'image' in self.output_values:
            self.set_display(
                self._make_display_arr(self.output_values['image'].payload))

    # ── Node evaluation ───────────────────────────────────────────────────────

    def evaluate(self):
        self.reset_progress()
        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            return False, 'No image connected'
        up_node = in_port.connected_ports()[0].node()
        data    = up_node.output_values.get(in_port.connected_ports()[0].name())
        if not isinstance(data, ImageData):
            return False, 'Input must be ImageData'

        arr = data.payload
        self._cached_arr = arr
        self._cached_bit_depth = getattr(data, 'bit_depth', 8) or 8
        self.set_progress(20)

        # Map float [0,1] to original bit-depth range for slider/histogram
        max_possible = (1 << self._cached_bit_depth) - 1

        # Build flat luminance in bit-depth scale for histogram display
        flat = arr.ravel().astype(np.float32) if arr.ndim == 2 \
               else arr.mean(axis=2).ravel().astype(np.float32)
        flat_scaled = flat * max_possible
        self._bc_widget.set_histogram_threadsafe(flat_scaled, full_range=max_possible)
        self.set_progress(40)

        # Restore (and clamp) saved range
        bc = self.get_property('bc_range')
        min_val = float(np.clip(bc[0], 0, max_possible - 1))
        max_val = float(np.clip(bc[1], min_val + 1, max_possible))
        self._bc_widget.set_range_threadsafe(min_val, max_val)
        # Restore saved colormap
        self._bc_widget.set_colormap_threadsafe(self.get_property('bc_colormap'))
        self.set_progress(60)

        result = self._apply_bc(arr, min_val, max_val)
        if result is None:
            return False, 'Processing failed'

        self.output_values['image'] = result
        self.set_display(self._make_display_arr(result.payload))
        self.set_progress(100)
        return True, None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_bc(self, arr, min_val, max_val):
        """Linear stretch [min_val, max_val] to [0, 1]. min/max are in bit-depth scale."""
        try:
            bd = getattr(self, '_cached_bit_depth', 8) or 8
            max_possible = float((1 << bd) - 1)
            # Convert bit-depth-scale handles to float [0,1]
            lo = min_val / max_possible
            hi = max_val / max_possible
            width = hi - lo
            if width <= 0:
                width = 1.0 / max_possible
            out = np.clip((arr.astype(np.float32) - lo) / width, 0.0, 1.0).astype(np.float32)
            upstream = self._get_input_image_data()
            if upstream:
                kwargs = {f: getattr(upstream, f, None) for f in upstream.model_fields if f != 'payload'}
                return ImageData(payload=out, **kwargs)
            return ImageData(payload=out)
        except Exception:
            return None

    def _make_display_arr(self, arr):
        """
        Apply the selected colormap to a numpy array for preview display.
        The OUTPUT port always carries the un-coloured image; colormap is
        display-only.  For colour (RGB/RGBA) inputs, luminance is used.
        """
        cmap = self.get_property('bc_colormap')
        if cmap == 'gray':
            return arr
        gray = arr if arr.ndim == 2 else arr.mean(axis=2)
        # _apply_colormap expects uint8
        gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8) if gray.dtype in (np.float32, np.float64) else gray
        return _apply_colormap(gray_u8, cmap)

    def _display_ui(self, data):
        self._last_display_data = data
        if not self.get_property('live_preview'):
            return
        if hasattr(self, '_image_widget'):
            img = data.payload if isinstance(data, (ImageData, MaskData)) else data
            self._image_widget.set_value(img)
            if hasattr(self.view, 'draw_node'):
                self.view.draw_node()


# ── Binary Threshold ──────────────────────────────────────────────────────────

class HistogramThresholdWidget(QtWidgets.QWidget):
    """
    Interactive histogram widget with a draggable threshold line for binary segmentation.

    A slider + spinbox pair syncs with the InfiniteLine on the plot. Dragging any control triggers a live mask update.

    Controls:
    - Yellow line / **Threshold** slider — threshold value
    - Green shaded region — selected pixels
    - **Direction** combo — above or below
    - **Auto (Otsu)** — automatic threshold

    Emits `threshold_changed(value, above_threshold)` on any user interaction.
    """
    threshold_changed = QtCore.Signal(float, bool)
    colormap_changed  = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full_range  = 255.0
        self._updating    = False
        self._above       = True
        self._hist_counts = None
        self._hist_edges  = None
        self._cmap_name   = 'gray'

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Plot ─────────────────────────────────────────────────────────
        self._plot = pg.PlotWidget(background='#1a1a1a')
        self._plot.setFixedHeight(110)
        self._plot.setMinimumWidth(240)
        self._plot.hideAxis('left')
        ax = self._plot.getAxis('bottom')
        ax.setPen(pg.mkPen('#555555'))
        ax.setTextPen(pg.mkPen('#aaaaaa'))
        tick_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        tick_font.setPointSize(7)
        ax.setStyle(tickFont=tick_font)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.getViewBox().setDefaultPadding(0)

        self._hist_curve = self._plot.plot(
            pen=pg.mkPen('#888888', width=1),
            fillLevel=0,
            brush=pg.mkBrush(120, 120, 120, 100),
            stepMode='center',
        )

        self._region = pg.LinearRegionItem(
            values=(128, 255),
            brush=pg.mkBrush(80, 200, 100, 50),
            pen=pg.mkPen(None),
            movable=False,
        )
        self._plot.addItem(self._region)

        self._line = pg.InfiniteLine(
            pos=128, angle=90, movable=True,
            pen=pg.mkPen('#f1c40f', width=2),
            label='Threshold',
            labelOpts={'color': '#f1c40f', 'position': 0.88,
                       'anchors': [(0.5, 0), (0.5, 1)]},
        )
        self._plot.addItem(self._line)
        self._line.sigPositionChanged.connect(self._on_line_moved)

        layout.addWidget(self._plot)

        # ── Colormap selector + gradient bar ─────────────────────────────
        cmap_row2 = QtWidgets.QHBoxLayout()
        cmap_row2.setSpacing(4)
        cmap_row2.setContentsMargins(0, 0, 0, 0)
        lbl_cm2 = QtWidgets.QLabel('Colormap')
        lbl_cm2.setStyleSheet('color: #cccccc; font-size: 8pt;')
        lbl_cm2.setFixedWidth(66)
        self._cmap_combo = QtWidgets.QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.setFixedWidth(90)
        self._cmap_bar = ColormapBar(cmap_name='gray')
        cmap_row2.addWidget(lbl_cm2)
        cmap_row2.addWidget(self._cmap_combo)
        cmap_row2.addWidget(self._cmap_bar, 1)
        layout.addLayout(cmap_row2)

        # ── Threshold slider + spinbox row ────────────────────────────────
        thresh_row = QtWidgets.QHBoxLayout()
        thresh_row.setSpacing(4)

        lbl_t = QtWidgets.QLabel('Threshold')
        lbl_t.setStyleSheet('color: #cccccc; font-size: 8pt;')
        lbl_t.setFixedWidth(66)

        self._sld = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sld.setRange(0, 255)
        self._sld.setValue(128)

        self._spn = QtWidgets.QSpinBox()
        self._spn.setRange(0, 255)
        self._spn.setValue(128)
        self._spn.setFixedWidth(62)

        thresh_row.addWidget(lbl_t)
        thresh_row.addWidget(self._sld)
        thresh_row.addWidget(self._spn)
        layout.addLayout(thresh_row)

        # ── Direction combo ───────────────────────────────────────────────
        dir_row = QtWidgets.QHBoxLayout()
        dir_row.setSpacing(4)
        lbl_d = QtWidgets.QLabel('Direction')
        lbl_d.setStyleSheet('color: #cccccc; font-size: 8pt;')
        lbl_d.setFixedWidth(66)
        self._combo = QtWidgets.QComboBox()
        self._combo.addItems(['Above threshold (pixel > T)',
                              'Below threshold (pixel ≤ T)'])
        dir_row.addWidget(lbl_d)
        dir_row.addWidget(self._combo)
        layout.addLayout(dir_row)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_auto  = QtWidgets.QPushButton('Auto (Otsu)')
        self._btn_reset = QtWidgets.QPushButton('Reset')
        for btn in (self._btn_auto, self._btn_reset):
            btn.setFixedHeight(22)
        btn_row.addWidget(self._btn_auto)
        btn_row.addWidget(self._btn_reset)
        layout.addLayout(btn_row)

        # ── Wire up ───────────────────────────────────────────────────────
        self._sld.valueChanged.connect(self._on_slider)
        self._spn.valueChanged.connect(self._on_spinbox)
        self._combo.currentIndexChanged.connect(self._on_direction_changed)
        self._btn_auto.clicked.connect(self._auto_otsu)
        self._btn_reset.clicked.connect(self._reset)
        self._cmap_combo.currentTextChanged.connect(self._on_colormap_changed)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_histogram(self, arr_flat, full_range=255):
        self._full_range = float(full_range)
        num_bins = min(512, int(full_range) + 1)
        counts, edges = np.histogram(arr_flat, bins=num_bins,
                                     range=(0, float(full_range)))
        self._hist_counts = counts
        self._hist_edges  = edges
        self._hist_curve.setData(x=edges, y=np.log1p(counts.astype(float)))
        self._plot.setXRange(0, full_range, padding=0)
        irange = int(full_range)
        self._sld.setRange(0, irange)
        self._spn.setRange(0, irange)

    def set_threshold(self, value, above=True):
        self._updating = True
        try:
            self._line.setValue(value)
            self._sld.setValue(int(round(value)))
            self._spn.setValue(int(round(value)))
            self._above = above
            self._combo.setCurrentIndex(0 if above else 1)
            self._update_region(value)
        finally:
            self._updating = False

    def get_threshold(self):
        return float(self._line.value()), self._above

    # ── Core setter ───────────────────────────────────────────────────────────

    def _set_thresh(self, val):
        """Atomically update all threshold controls."""
        self._updating = True
        try:
            self._line.setValue(val)
            self._sld.setValue(int(round(val)))
            self._spn.setValue(int(round(val)))
            self._update_region(val)
        finally:
            self._updating = False

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_line_moved(self):
        if self._updating:
            return
        val = float(self._line.value())
        # Sync slider/spinbox/region WITHOUT calling _line.setValue — that would
        # interrupt the active pyqtgraph drag and prevent smooth tracking.
        self._updating = True
        try:
            self._sld.setValue(int(round(val)))
            self._spn.setValue(int(round(val)))
            self._update_region(val)
        finally:
            self._updating = False
        self.threshold_changed.emit(val, self._above)

    def _on_slider(self, int_val):
        if self._updating:
            return
        val = float(int_val)
        # Sync spinbox silently, then propagate
        self._updating = True
        self._spn.setValue(int_val)
        self._updating = False
        self._line.setValue(val)
        self._update_region(val)
        self.threshold_changed.emit(val, self._above)

    def _on_spinbox(self, int_val):
        if self._updating:
            return
        val = float(int_val)
        # Sync slider silently, then propagate
        self._updating = True
        self._sld.setValue(int_val)
        self._updating = False
        self._line.setValue(val)
        self._update_region(val)
        self.threshold_changed.emit(val, self._above)

    def _on_direction_changed(self, idx):
        if self._updating:
            return
        self._above = (idx == 0)
        self._update_region(float(self._line.value()))
        self.threshold_changed.emit(float(self._line.value()), self._above)

    def _update_region(self, thresh):
        if self._above:
            self._region.setRegion((thresh, self._full_range))
        else:
            self._region.setRegion((0, thresh))

    # ── Auto / Reset ──────────────────────────────────────────────────────────

    def _auto_otsu(self):
        if self._hist_counts is None:
            return
        counts = self._hist_counts.astype(float)
        total  = counts.sum()
        if total == 0:
            return
        sum_total = float(np.dot(np.arange(len(counts)), counts))
        w_bg = sum_bg = 0.0
        best_val = best_var = 0.0
        for i, c in enumerate(counts):
            w_bg += c
            w_fg  = total - w_bg
            if w_bg == 0 or w_fg == 0:
                continue
            sum_bg += i * c
            mu_bg   = sum_bg / w_bg
            mu_fg   = (sum_total - sum_bg) / w_fg
            var     = w_bg * w_fg * (mu_bg - mu_fg) ** 2
            if var > best_var:
                best_var = var
                best_val = float(self._hist_edges[i])
        self._set_thresh(best_val)
        self.threshold_changed.emit(best_val, self._above)

    def _reset(self):
        val = self._full_range / 2.0
        self._set_thresh(val)
        self._above = True
        self._combo.setCurrentIndex(0)
        self._update_region(val)
        self.threshold_changed.emit(val, True)

    # ── Colormap API ──────────────────────────────────────────────────────────

    def get_colormap(self):
        return self._cmap_name

    def set_colormap(self, name):
        self._cmap_name = name
        self._cmap_bar.set_colormap(name)
        self._updating = True
        idx = self._cmap_combo.findText(name)
        if idx >= 0:
            self._cmap_combo.setCurrentIndex(idx)
        self._updating = False

    def _on_colormap_changed(self, name):
        if self._updating:
            return
        self._cmap_name = name
        self._cmap_bar.set_colormap(name)
        self.colormap_changed.emit(name)


class NodeThresholdWidget(NodeBaseWidget):
    """NodeGraphQt proxy wrapper that embeds a HistogramThresholdWidget on the node surface."""
    _set_hist_sig = QtCore.Signal(object, float)
    _set_thresh_sig = QtCore.Signal(float, bool)
    _set_cmap_sig = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent, name='thresh_widget', label='')
        self._inner = HistogramThresholdWidget()
        self.set_custom_widget(self._inner)
        self._set_hist_sig.connect(
            self._inner.set_histogram, QtCore.Qt.ConnectionType.QueuedConnection)
        self._set_thresh_sig.connect(
            self._inner.set_threshold, QtCore.Qt.ConnectionType.QueuedConnection)
        self._set_cmap_sig.connect(
            self._inner.set_colormap, QtCore.Qt.ConnectionType.QueuedConnection)

    @property
    def inner(self):
        return self._inner

    def get_value(self):
        val, above = self._inner.get_threshold()
        return [val, int(above)]

    def set_value(self, value):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            self._inner.set_threshold(float(value[0]), bool(value[1]))

    def set_histogram_threadsafe(self, arr_flat, full_range=255):
        if threading.current_thread() is threading.main_thread():
            self._inner.set_histogram(arr_flat, full_range=full_range)
        else:
            self._set_hist_sig.emit(arr_flat, float(full_range))

    def set_threshold_threadsafe(self, value, above=True):
        if threading.current_thread() is threading.main_thread():
            self._inner.set_threshold(float(value), bool(above))
        else:
            self._set_thresh_sig.emit(float(value), bool(above))

    def set_colormap_threadsafe(self, name):
        if threading.current_thread() is threading.main_thread():
            self._inner.set_colormap(str(name))
        else:
            self._set_cmap_sig.emit(str(name))


class BinaryThresholdNode(BaseImageProcessNode):
    """
    Applies interactive global thresholding using a histogram with a draggable threshold line.

    Drag the yellow threshold line on the histogram to select pixels. The green-shaded region shows which pixels will be included in the output mask. Works with 8-bit and 16-bit input images; the threshold value is in the original pixel-value space.

    Direction modes:
    - *Above (pixel > T)* — selected pixels are brighter than the threshold
    - *Below (pixel <= T)* — selected pixels are darker than the threshold
    - *Auto (Otsu)* — automatically finds the optimal threshold using Otsu's method
    - *Auto Otsu per image* — re-computes Otsu for every new input image (useful for batch workflows with varying brightness)

    Output is binary MaskData (255 = selected, 0 = not selected).

    Keywords: threshold, binary, Otsu, segmentation, foreground, 閾值, 二值化, 分割, 影像處理, 前景
    """
    __identifier__ = 'nodes.image_process.filter'
    NODE_NAME      = 'Binary Threshold'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['mask']}
    _UI_PROPS = BaseImageProcessNode._UI_PROPS | frozenset(
        {'thresh_state', 'thresh_widget', 'auto_otsu_per_image'}
    )
    PROP_DESCRIPTIONS = {
        'thresh_state':        '[threshold, direction] — 1=keep pixels > threshold (strictly above), 0=keep pixels <= threshold (below or equal)',
        'auto_otsu_per_image': 'set to false when using an explicit thresh_state value; default true overrides manual threshold',
    }

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('mask', multi_output=True, color=PORT_COLORS['mask'])

        self.create_property('thresh_state',    [128.0, 1])  # [threshold, above]
        self.create_property('thresh_colormap', 'gray')

        self._thresh_widget = NodeThresholdWidget(self.view)
        self.add_custom_widget(self._thresh_widget)
        self._thresh_widget.inner.threshold_changed.connect(self._on_thresh_changed)
        self._thresh_widget.inner.colormap_changed.connect(self._on_colormap_changed)
        self.add_checkbox('auto_otsu_per_image', '', text='Auto Otsu per image', state=True)

        self.create_preview_widgets()
        self._cached_arr = None

    # ── Live drag callback ────────────────────────────────────────────────────

    def set_property(self, name, value, push_undo=True):
        super().set_property(name, value, push_undo)
        # Sync the interactive widget whenever thresh_state is set externally
        # (e.g. WorkflowLoader). set_threshold uses _updating=True internally
        # so it will NOT re-emit threshold_changed.
        if name == 'thresh_state' and hasattr(self, '_thresh_widget'):
            try:
                self._thresh_widget.inner.set_threshold(float(value[0]), bool(value[1]))
            except Exception:
                pass

    def _on_thresh_changed(self, value, above):
        super(BaseExecutionNode, self).set_property(
            'thresh_state', [value, int(above)], push_undo=False)
        if self._cached_arr is not None:
            result = self._apply_threshold(self._cached_arr, value, above)
            if result is not None:
                self.output_values['mask'] = result
                self._update_preview_direct(result.payload)
                # Propagate dirty to downstream nodes so they re-evaluate
                for out_port in self.outputs().values():
                    for in_port in out_port.connected_ports():
                        dn = in_port.node()
                        if hasattr(dn, 'mark_dirty'):
                            dn.mark_dirty()

    def _on_colormap_changed(self, cmap_name):
        """Recolour the preview when the user picks a different colormap."""
        super(BaseExecutionNode, self).set_property(
            'thresh_colormap', cmap_name, push_undo=False)
        if self._cached_arr is not None:
            self._update_preview_direct(self._input_display_img())

    def _update_preview_direct(self, img):
        """Push img to the preview widget bypassing the live_preview gate.
        Interactive threshold/colormap changes always show feedback immediately."""
        if img is None:
            return
        self._last_display_data = img
        if hasattr(self, '_image_widget'):
            self._image_widget.set_value(img)
            if hasattr(self.view, 'draw_node'):
                self.view.draw_node()

    # ── Node evaluation ───────────────────────────────────────────────────────

    def evaluate(self):
        self.reset_progress()
        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            return False, 'No image connected'
        up_node = in_port.connected_ports()[0].node()
        data    = up_node.output_values.get(in_port.connected_ports()[0].name())
        if not isinstance(data, ImageData):
            return False, 'Input must be ImageData'

        arr = data.payload
        self._cached_arr = arr
        self._cached_bit_depth = getattr(data, 'bit_depth', 8) or 8
        self.set_progress(20)
        is_main_thread = threading.current_thread() is threading.main_thread()

        # Map float [0,1] to original bit-depth range for slider/histogram
        max_possible = (1 << self._cached_bit_depth) - 1

        # Convert to original scale for histogram display
        flat = arr.ravel().astype(np.float32) if arr.ndim == 2 \
               else arr.mean(axis=2).ravel().astype(np.float32)
        flat_scaled = flat * max_possible  # [0,1] → [0, max_possible]
        self._thresh_widget.set_histogram_threadsafe(flat_scaled, full_range=max_possible)
        self.set_progress(40)

        state     = self.get_property('thresh_state')
        thresh    = float(np.clip(state[0], 0, max_possible))
        above     = bool(state[1])
        if bool(self.get_property('auto_otsu_per_image')):
            auto_t = self._compute_otsu_from_flat(flat_scaled, max_possible=max_possible)
            if auto_t is not None:
                thresh = float(np.clip(auto_t, 0, max_possible))
                # Keep UI-backed property mutation on main thread only.
                if is_main_thread:
                    super(BaseExecutionNode, self).set_property(
                        'thresh_state', [thresh, int(above)], push_undo=False
                    )
        self._thresh_widget.set_threshold_threadsafe(thresh, above)
        # Restore saved colormap
        self._thresh_widget.set_colormap_threadsafe(self.get_property('thresh_colormap'))
        self.set_progress(60)

        result = self._apply_threshold(arr, thresh, above)
        if result is None:
            return False, 'Processing failed'

        self.output_values['mask'] = result
        self.set_display(result.payload)
        self.set_progress(100)
        return True, None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_threshold(self, arr, thresh, above):
        """Threshold on luminance; thresh is in original bit-depth scale."""
        try:
            gray = arr.astype(np.float32) if arr.ndim == 2 \
                   else arr.mean(axis=2).astype(np.float32)
            # Convert threshold from bit-depth scale to float [0,1]
            bd = getattr(self, '_cached_bit_depth', 8) or 8
            max_val = float((1 << bd) - 1) if bd > 0 else 255.0
            thresh_float = float(thresh) / max_val
            binary = ((gray > thresh_float) if above else (gray <= thresh_float))
            out = binary.astype(np.uint8) * 255
            return MaskData(payload=out)
        except Exception:
            return None

    @staticmethod
    def _compute_otsu_from_flat(flat_vals, max_possible=255):
        """
        Compute Otsu threshold from flattened grayscale values.
        Returns threshold in original intensity scale, or None on failure.
        """
        if flat_vals is None or len(flat_vals) == 0:
            return None
        try:
            num_bins = min(512, int(max_possible) + 1)
            counts, edges = np.histogram(
                flat_vals, bins=num_bins, range=(0, float(max_possible))
            )
            counts = counts.astype(float)
            total = counts.sum()
            if total <= 0:
                return None

            sum_total = float(np.dot(np.arange(len(counts)), counts))
            w_bg = 0.0
            sum_bg = 0.0
            best_idx = 0
            best_var = -1.0
            for i, c in enumerate(counts):
                w_bg += c
                w_fg = total - w_bg
                if w_bg <= 0 or w_fg <= 0:
                    continue
                sum_bg += i * c
                mu_bg = sum_bg / w_bg
                mu_fg = (sum_total - sum_bg) / w_fg
                var = w_bg * w_fg * (mu_bg - mu_fg) ** 2
                if var > best_var:
                    best_var = var
                    best_idx = i
            return float(edges[best_idx])
        except Exception:
            return None

    def _input_display_img(self):
        """
        Build a display image from the cached INPUT array with the selected
        colormap applied.  Showing the colourised input (not the binary mask)
        lets the user see intensity levels and decide where to threshold.
        The actual output port still carries the binary mask.
        """
        if self._cached_arr is None:
            return None
        arr = self._cached_arr
        gray = arr.astype(np.float32) if arr.ndim == 2 \
               else arr.mean(axis=2).astype(np.float32)
        mn, mx = float(gray.min()), float(gray.max())
        if mx > mn:
            gray_norm = ((gray - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            gray_norm = np.zeros(gray.shape, dtype=np.uint8)
        cmap = self.get_property('thresh_colormap')
        if cmap == 'gray':
            return gray_norm
        return _apply_colormap(gray_norm, cmap)

    def _display_ui(self, data):
        self._last_display_data = data
        if not self.get_property('live_preview'):
            return
        if hasattr(self, '_image_widget'):
            img = data.payload if isinstance(data, (ImageData, MaskData)) else data
            self._image_widget.set_value(img)
            if hasattr(self.view, 'draw_node'):
                self.view.draw_node()


class GammaContrastNode(BaseImageProcessNode):
    """
    Applies gamma correction to adjust image tonality non-linearly.

    Transforms each pixel via `O = I^gamma * gain`.

    **gamma** — exponent controlling the curve shape (default: 1.0). Values below 1 brighten; values above 1 darken.

    **gain** — multiplicative scaling factor applied after gamma (default: 1.0).

    Keywords: gamma, contrast, adjust, nonlinear, exposure, 對比, 調整, 亮度, 非線性, 影像處理
    """
    __identifier__ = 'nodes.image_process.Exposure'
    NODE_NAME      = 'Gamma Contrast'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])

        self._add_float_spinbox('gamma', 'Gamma', value=1.0,
                                min_val=0.0, max_val=500.0, step=0.1, decimals=2)
        self._add_float_spinbox('gain', 'Gain', value=1.0,
                                min_val=0.0, max_val=500.0, step=0.1, decimals=2)
        self.create_preview_widgets()
    
    def evaluate(self):
        self.reset_progress()
        from skimage.exposure import adjust_gamma

        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            return False, 'No image connected'
        up_node = in_port.connected_ports()[0].node()
        data    = up_node.output_values.get(in_port.connected_ports()[0].name())
        if not isinstance(data, ImageData):
            return False, 'Input must be ImageData'
        
        arr = data.payload
        self.set_progress(20)

        gamma = self.get_property('gamma')
        gain = self.get_property('gain')
        self.set_progress(40)

        result = adjust_gamma(arr, gamma, gain)
        self.set_progress(90)

        self._make_image_output(result)
        self.set_display(result)
        self.set_progress(100)
        return True, None


class RGBToGrayNode(BaseImageProcessNode):
    """
    Converts an RGB or RGBA image to a single-channel grayscale image.

    Method options:
    - *Luminosity (Rec.601)* — standard broadcast weights (PIL default): `L = 0.299R + 0.587G + 0.114B`
    - *Luminosity (Rec.709)* — HDTV/sRGB weights used by skimage: `L = 0.2125R + 0.7154G + 0.0721B`
    - *Average* — simple mean of R, G, B
    - *Red* / *Green* / *Blue* — extracts a single colour channel as grayscale

    Output is single-channel (L-mode) ImageData.

    Keywords: grayscale, gray, RGB, convert, luminance, grey, greyscale, 灰階, 轉換, 亮度, 影像處理, 色彩
    """
    __identifier__ = 'nodes.image_process.color'
    NODE_NAME      = 'RGB to Gray'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    _METHODS = [
        'Luminosity (Rec.601)',
        'Luminosity (Rec.709)',
        'Average',
        'Red',
        'Green',
        'Blue',
    ]

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])
        self.add_combo_menu('method', 'Method', items=self._METHODS)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            return False, 'No image connected'
        up_node = in_port.connected_ports()[0].node()
        data    = up_node.output_values.get(in_port.connected_ports()[0].name())
        if not isinstance(data, ImageData):
            return False, 'Input must be ImageData'

        arr = data.payload
        # Ensure RGB for channel-based methods
        if arr.ndim == 2:
            arr_rgb = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr_rgb = arr[:, :, :3]
        else:
            arr_rgb = arr
        self.set_progress(20)

        method = self.get_property('method')
        if method == 'Luminosity (Rec.601)':
            # Standard broadcast weights: L = 0.299R + 0.587G + 0.114B
            gray = np.dot(arr_rgb[:, :, :3].astype(np.float64),
                          [0.299, 0.587, 0.114]).astype(arr.dtype)
        elif method == 'Luminosity (Rec.709)':
            from skimage.color import rgb2gray
            # Data is float [0,1] — rgb2gray handles this directly
            gray = rgb2gray(arr_rgb.astype(np.float64)).astype(np.float32)
        elif method == 'Average':
            gray = arr_rgb.mean(axis=2).astype(arr.dtype)
        elif method == 'Red':
            gray = arr_rgb[:, :, 0]
        elif method == 'Green':
            gray = arr_rgb[:, :, 1]
        elif method == 'Blue':
            gray = arr_rgb[:, :, 2]
        else:
            gray = np.dot(arr_rgb[:, :, :3].astype(np.float64),
                          [0.299, 0.587, 0.114]).astype(arr.dtype)

        self.set_progress(90)
        self._make_image_output(gray)
        self.set_display(gray)
        self.set_progress(100)
        return True, None


class ColorDeconvolutionNode(BaseImageProcessNode):
    """
    Separates staining colours in histology images using colour deconvolution.

    Stain matrices prefixed with SK are from scikit-image, CD2 matrices are
    from ImageJ's Colour Deconvolution 2 plugin.

    Output mode:

    - *Colored* — each channel retains the stain's original colour on a white background
    - *Grayscale* — intensity map where brighter = more stain (for quantification)

    Third stain completion:

    - *Ruifrok* — Ruifrok/Johnston fallback
    - *Cross Product* — stain-3 = stain-1 x stain-2
    - *Auto* — keep matrix as provided

    Keywords: color deconvolution, stain, H&E, histology, histochemistry, Masson trichrome
    """
    __identifier__ = 'nodes.image_process.color'
    NODE_NAME      = 'Color Deconvolution'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image', 'image', 'image']}
    _collection_aware = True

    # For 'skimage' kind, value is skimage.color "*_from_rgb" attribute name.
    # For 'cd2' kind, value is rgb_from_stain matrix (rows = stain OD vectors [R,G,B]).
    _MATRIX_OPTIONS: dict = {
        # ── skimage presets ────────────────────────────────────────────────
        'SK H&E':              {'kind': 'skimage', 'value': 'hed_from_rgb'},
        'SK H-DAB':            {'kind': 'skimage', 'value': 'hdx_from_rgb'},
        'SK Feulgen+LG':       {'kind': 'skimage', 'value': 'fgx_from_rgb'},
        'SK Giemsa':           {'kind': 'skimage', 'value': 'bex_from_rgb'},
        'SK FR+FB+DAB':        {'kind': 'skimage', 'value': 'rbd_from_rgb'},
        'SK MG+DAB':           {'kind': 'skimage', 'value': 'gdx_from_rgb'},
        'SK H+AEC':            {'kind': 'skimage', 'value': 'hax_from_rgb'},
        'SK BRO':              {'kind': 'skimage', 'value': 'bro_from_rgb'},
        'SK MB+Ponceau':       {'kind': 'skimage', 'value': 'bpx_from_rgb'},
        'SK Alcian+H':         {'kind': 'skimage', 'value': 'ahx_from_rgb'},
        'SK H+PAS':            {'kind': 'skimage', 'value': 'hpx_from_rgb'},
        # ── ImageJ Colour Deconvolution 2 presets (rows from MOD vectors) ─
        'CD2 H&E': {'kind': 'cd2', 'value': np.array([
            [0.644211, 0.716556, 0.266844],
            [0.092789, 0.954111, 0.283111],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 H&E2': {'kind': 'cd2', 'value': np.array([
            [0.49015734, 0.76897085, 0.41040173],
            [0.04615336, 0.8420684, 0.5373925],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 H-DAB': {'kind': 'cd2', 'value': np.array([
            [0.650, 0.704, 0.286],
            [0.268, 0.570, 0.776],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 H&E-DAB': {'kind': 'cd2', 'value': np.array([
            [0.650, 0.704, 0.286],
            [0.072, 0.990, 0.105],
            [0.268, 0.570, 0.776],
        ])},
        'CD2 NBT/BCIP-Red': {'kind': 'cd2', 'value': np.array([
            [0.62302786, 0.697869, 0.3532918],
            [0.073615186, 0.79345673, 0.6041582],
            [0.7369498, 0.0010, 0.6759475],
        ])},
        'CD2 H-DAB-NewF': {'kind': 'cd2', 'value': np.array([
            [0.5625407925, 0.70450559, 0.4308375625],
            [0.26503363, 0.68898016, 0.674584],
            [0.0777851125, 0.804293475, 0.5886050475],
        ])},
        'CD2 H-HRPG-NewF': {'kind': 'cd2', 'value': np.array([
            [0.8098939567, 0.4488181033, 0.3714423567],
            [0.0777851125, 0.804293475, 0.5886050475],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 Feulgen-LG': {'kind': 'cd2', 'value': np.array([
            [0.46420921, 0.83008335, 0.30827187],
            [0.94705542, 0.25373821, 0.19650764],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 Giemsa': {'kind': 'cd2', 'value': np.array([
            [0.834750233, 0.513556283, 0.196330403],
            [0.092789, 0.954111, 0.283111],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 FR-FB-DAB': {'kind': 'cd2', 'value': np.array([
            [0.21393921, 0.85112669, 0.47794022],
            [0.74890292, 0.60624161, 0.26731082],
            [0.268, 0.570, 0.776],
        ])},
        'CD2 MG-DAB': {'kind': 'cd2', 'value': np.array([
            [0.98003, 0.144316, 0.133146],
            [0.268, 0.570, 0.776],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 H-AEC': {'kind': 'cd2', 'value': np.array([
            [0.650, 0.704, 0.286],
            [0.2743, 0.6796, 0.6803],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 Azan-Mallory': {'kind': 'cd2', 'value': np.array([
            [0.853033, 0.508733, 0.112656],
            [0.09289875, 0.8662008, 0.49098468],
            [0.10732849, 0.36765403, 0.9237484],
        ])},
        'CD2 Masson': {'kind': 'cd2', 'value': np.array([
            [0.7995107, 0.5913521, 0.10528667],
            [0.09997159, 0.73738605, 0.6680326],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 Alcian+H': {'kind': 'cd2', 'value': np.array([
            [0.874622, 0.457711, 0.158256],
            [0.552556, 0.7544, 0.353744],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 H+PAS': {'kind': 'cd2', 'value': np.array([
            [0.644211, 0.716556, 0.266844],
            [0.175411, 0.972178, 0.154589],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 BrilliantBlue': {'kind': 'cd2', 'value': np.array([
            [0.31465548, 0.6602395, 0.68196464],
            [0.383573, 0.5271141, 0.7583024],
            [0.7433543, 0.51731443, 0.4240403],
        ])},
        'CD2 Astra-Fuchsin': {'kind': 'cd2', 'value': np.array([
            [0.92045766, 0.35425216, 0.16511545],
            [0.13336428, 0.8301452, 0.5413621],
            [0.0, 0.0, 0.0],
        ])},
        'CD2 RGB': {'kind': 'cd2', 'value': np.array([
            [0.001, 1.0, 1.0],
            [1.0, 0.001, 1.0],
            [1.0, 1.0, 0.001],
        ])},
        'CD2 CMY': {'kind': 'cd2', 'value': np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])},
    }

    @staticmethod
    def _norm_rows(m: np.ndarray) -> np.ndarray:
        out = np.array(m, dtype=np.float64, copy=True)
        for i in range(3):
            n = float(np.linalg.norm(out[i, :]))
            if n > 1e-12:
                out[i, :] /= n
        return out

    @classmethod
    def _complete_rgb_from(cls, rgb_from: np.ndarray, mode: str) -> np.ndarray:
        m = cls._norm_rows(rgb_from)

        # If second stain is unspecified, use CD2/ImageJ heuristic permutation.
        if float(np.linalg.norm(m[1, :])) <= 1e-12:
            m[1, :] = np.array([m[0, 2], m[0, 0], m[0, 1]], dtype=np.float64)
            m = cls._norm_rows(m)

        need_third = float(np.linalg.norm(m[2, :])) <= 1e-12
        if not need_third:
            return m

        v1 = m[0, :].copy()
        v2 = m[1, :].copy()

        if mode == 'ruifrok':
            v3 = np.zeros(3, dtype=np.float64)
            for c in range(3):
                s = float(v1[c] * v1[c] + v2[c] * v2[c])
                v3[c] = 0.0 if s > 1.0 else np.sqrt(max(0.0, 1.0 - s))
        else:
            v3 = np.cross(v1, v2)

        n3 = float(np.linalg.norm(v3))
        if n3 <= 1e-12:
            # Robust fallback.
            v3 = np.cross(v1 + 1e-6, v2 + 1e-6)
            n3 = float(np.linalg.norm(v3))
        if n3 > 1e-12:
            v3 /= n3
            m[2, :] = v3
        return cls._norm_rows(m)

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('ch1', color=PORT_COLORS['image'])
        self.add_output('ch2', color=PORT_COLORS['image'])
        self.add_output('ch3', color=PORT_COLORS['image'])
        self.add_combo_menu('stain_matrix', 'Stain Matrix', items=list(self._MATRIX_OPTIONS.keys()))
        self.add_combo_menu('third_stain_mode', 'Third Stain', items=[
            'Ruifrok', 'Auto (matrix)', 'Cross Product'
        ])
        self.add_checkbox('grayscale_output', '', text='Grayscale output', state=False)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            return False, 'No image connected'
        up_node = in_port.connected_ports()[0].node()
        data    = up_node.output_values.get(in_port.connected_ports()[0].name())
        if not isinstance(data, ImageData):
            return False, 'Input must be ImageData'

        import skimage.color as _skcolor
        from skimage.color import separate_stains, combine_stains

        arr_raw = data.payload
        # Ensure RGB for deconvolution
        if arr_raw.ndim == 2:
            arr_rgb = np.stack([arr_raw, arr_raw, arr_raw], axis=-1)
        elif arr_raw.ndim == 3 and arr_raw.shape[2] == 4:
            arr_rgb = arr_raw[:, :, :3]
        else:
            arr_rgb = arr_raw
        # Data is already float32 [0,1] — just convert to float64 for precision
        arr = arr_rgb.astype(np.float64)
        self.set_progress(20)

        matrix_key = self.get_property('stain_matrix')
        matrix_def = self._MATRIX_OPTIONS.get(
            matrix_key, self._MATRIX_OPTIONS['SK H&E'])
        mode_raw = str(self.get_property('third_stain_mode') or 'Auto (matrix)')
        if mode_raw.startswith('Cross'):
            third_mode = 'cross'
        elif mode_raw.startswith('Ruifrok'):
            third_mode = 'ruifrok'
        else:
            third_mode = 'ruifrok'

        if matrix_def.get('kind') == 'skimage':
            fwd_matrix = np.array(getattr(_skcolor, matrix_def['value']), dtype=np.float64)
            # Convert to rgb_from for optional third-stain forcing, then back.
            try:
                rgb_from = np.linalg.inv(fwd_matrix)
            except np.linalg.LinAlgError:
                rgb_from = np.linalg.pinv(fwd_matrix)
            if third_mode in ('cross', 'ruifrok'):
                rgb_from = self._complete_rgb_from(rgb_from, third_mode)
                try:
                    fwd_matrix = np.linalg.inv(rgb_from)
                except np.linalg.LinAlgError:
                    fwd_matrix = np.linalg.pinv(rgb_from)
        else:
            rgb_from = np.array(matrix_def['value'], dtype=np.float64)
            rgb_from = self._complete_rgb_from(rgb_from, third_mode)
            try:
                fwd_matrix = np.linalg.inv(rgb_from)
            except np.linalg.LinAlgError:
                fwd_matrix = np.linalg.pinv(rgb_from)

        stains = separate_stains(arr, fwd_matrix)  # (H, W, 3), float
        self.set_progress(60)

        grayscale = bool(self.get_property('grayscale_output'))
        channels  = []

        if grayscale:
            # Each channel = normalized concentration map (brighter = more stain)
            for i in range(3):
                ch      = np.clip(stains[:, :, i], 0, None)
                max_val = ch.max()
                ch_f    = (ch / max_val).astype(np.float32) if max_val > 0 \
                          else np.zeros(ch.shape, dtype=np.float32)
                channels.append(ch_f)
        else:
            # Reconstruct each stain with its original color using combine_stains.
            # combine_stains(stain_i_only, inv_matrix) → RGB float [0,1]
            try:
                inv_matrix = np.linalg.inv(fwd_matrix)
            except np.linalg.LinAlgError:
                inv_matrix = np.linalg.pinv(fwd_matrix)

            null = np.zeros_like(stains[:, :, 0])
            for i in range(3):
                isolated = np.stack(
                    [stains[:, :, j] if j == i else null for j in range(3)],
                    axis=-1,
                )
                ch_rgb = combine_stains(isolated, inv_matrix)   # (H, W, 3) float
                channels.append(np.clip(ch_rgb, 0.0, 1.0).astype(np.float32))

        self.set_progress(90)
        self._make_image_output(channels[0], 'ch1')
        self._make_image_output(channels[1], 'ch2')
        self._make_image_output(channels[2], 'ch3')

        self.set_display(self._make_2x2_preview(arr_rgb, channels))
        self.set_progress(100)
        return True, None

    # ------------------------------------------------------------------
    def _make_2x2_preview(self, original_arr, channels):
        """
        Returns a 2x2 composite numpy array for the preview widget:
          top-left  = Original | top-right  = ch1
          bot-left  = ch2      | bot-right  = ch3
        A dark 2-px gap separates the quadrants; each tile is 192x192 px.
        """
        from PIL import ImageDraw
        CELL = 192
        GAP  = 2

        def _arr_to_pil_local(a):
            if a.dtype in (np.float32, np.float64):
                a = np.clip(a * 255, 0, 255).astype(np.uint8)
            if a.ndim == 2:
                return Image.fromarray(a, mode='L')
            return Image.fromarray(a, mode='RGB')

        def _fit(arr):
            pil = _arr_to_pil_local(arr).convert('RGB')
            try:
                pil.thumbnail((CELL, CELL), Image.Resampling.LANCZOS)
            except AttributeError:
                pil.thumbnail((CELL, CELL), Image.LANCZOS)
            tile = Image.new('RGB', (CELL, CELL), (30, 30, 30))
            tile.paste(pil, ((CELL - pil.width) // 2, (CELL - pil.height) // 2))
            return tile

        W = H = CELL * 2 + GAP
        composite = Image.new('RGB', (W, H), (60, 60, 60))

        items = [
            (original_arr, 'Original', (0,          0)),
            (channels[0],  'ch1',      (CELL + GAP, 0)),
            (channels[1],  'ch2',      (0,          CELL + GAP)),
            (channels[2],  'ch3',      (CELL + GAP, CELL + GAP)),
        ]

        draw = ImageDraw.Draw(composite)
        for arr, label, (px, py) in items:
            composite.paste(_fit(arr), (px, py))
            draw.text((px + 4, py + 4), label, fill=(220, 220, 220))

        return np.array(composite)


# ===========================================================================
# CropNode
# ===========================================================================

# CropNode is defined in roi_nodes.py (it uses the interactive ROI drawing widget)


# ===========================================================================
# ZoomNode
# ===========================================================================

class ZoomNode(BaseImageProcessNode):
    """
    Resizes an image by a zoom factor using high-quality Lanczos resampling.

    A factor of 2.0 doubles the size; 0.5 halves it. The preview overlay shows the actual output dimensions. Works with both ImageData and MaskData inputs.

    **zoom** — scale factor (default: 1.0).

    Keywords: zoom, scale, magnify, resize, factor, 縮放, 放大, 影像處理, 比例, 尺寸
    """
    __identifier__ = 'nodes.image_process.Transform'
    NODE_NAME      = 'Zoom'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])
        self._add_float_spinbox('zoom', 'Zoom Factor', value=1.0,
                                min_val=0.01, max_val=100.0, step=0.1, decimals=3)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        port = self.inputs().get('image')
        if not port or not port.connected_ports():
            return False, "No input connected"
        cp   = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())

        if isinstance(data, ImageData):
            out_cls = ImageData
        elif isinstance(data, MaskData):
            out_cls = MaskData
        else:
            return False, "Input must be ImageData or MaskData"

        arr_in = data.payload
        H_in = arr_in.shape[0]
        W_in = arr_in.shape[1]
        factor = float(self.get_property('zoom'))
        new_W  = max(1, round(W_in * factor))
        new_H  = max(1, round(H_in * factor))

        self.set_progress(50)
        from skimage.transform import resize as _sk_resize
        resized_arr = _sk_resize(arr_in, (new_H, new_W),
                                 order=3, preserve_range=True,
                                 anti_aliasing=True).astype(arr_in.dtype)

        self._make_image_output(resized_arr)
        self.set_display(resized_arr)
        self.set_progress(100)
        return True, None


# ===========================================================================
# ResizeNode
# ===========================================================================

class ResizeNode(BaseImageProcessNode):
    """
    Resizes an image or mask to an exact pixel size (width x height).

    **resize_width** — target width in pixels (default: 300).

    **resize_height** — target height in pixels (default: 300).

    **resample** — resampling method: *lanczos*, *bilinear*, *nearest*, or *bicubic*.

    Keywords: resize, scale, dimensions, width, height, 縮放, 尺寸, 影像處理, 寬度, 高度
    """
    __identifier__ = 'nodes.image_process.Transform'
    NODE_NAME      = 'Resize'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])
        self._add_int_spinbox('resize_width',  'Width (px)',  value=300, min_val=1, max_val=16384, step=1)
        self._add_int_spinbox('resize_height', 'Height (px)', value=300, min_val=1, max_val=16384, step=1)
        self.add_combo_menu('resample', 'Resampling',
                            items=['lanczos', 'bilinear', 'nearest', 'bicubic'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        port = self.inputs().get('image')
        if not port or not port.connected_ports():
            return False, "No input connected"
        cp   = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())

        if isinstance(data, ImageData):
            out_cls = ImageData
        elif isinstance(data, MaskData):
            out_cls = MaskData
        else:
            return False, "Input must be ImageData or MaskData"

        arr_in = data.payload
        W = max(1, int(self.get_property('resize_width')))
        H = max(1, int(self.get_property('resize_height')))

        try:
            _resamplers = {
                'lanczos':  Image.Resampling.LANCZOS,
                'bilinear': Image.Resampling.BILINEAR,
                'nearest':  Image.Resampling.NEAREST,
                'bicubic':  Image.Resampling.BICUBIC,
            }
        except AttributeError:
            _resamplers = {
                'lanczos':  Image.LANCZOS,
                'bilinear': Image.BILINEAR,
                'nearest':  Image.NEAREST,
                'bicubic':  Image.BICUBIC,
            }

        _order_map = {'lanczos': 3, 'bicubic': 3, 'bilinear': 1, 'nearest': 0}
        order = _order_map.get(self.get_property('resample'), 3)
        self.set_progress(50)

        from skimage.transform import resize as _sk_resize
        resized_arr = _sk_resize(arr_in, (H, W),
                                 order=order, preserve_range=True,
                                 anti_aliasing=(order > 0)).astype(arr_in.dtype)

        self._make_image_output(resized_arr)
        self.set_display(resized_arr)
        self.set_progress(100)
        return True, None


# ===========================================================================
# RotateNode
# ===========================================================================

class RotateNode(BaseImageProcessNode):
    """
    Rotates an image counter-clockwise by a given angle in degrees.

    The canvas is expanded to fit the full rotated image; surrounding areas are filled with black. Works with both ImageData and MaskData inputs.

    **angle** — rotation angle in degrees (default: 0.0).

    Keywords: rotate, angle, orientation, transform, spin, 旋轉, 角度, 方向, 影像處理, 轉換
    """
    __identifier__ = 'nodes.image_process.Transform'
    NODE_NAME      = 'Rotate'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])
        self._add_float_spinbox('angle', 'Angle (°)', value=0.0,
                                min_val=-360.0, max_val=360.0, step=1.0, decimals=1)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        port = self.inputs().get('image')
        if not port or not port.connected_ports():
            return False, "No input connected"
        cp   = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())

        if isinstance(data, ImageData):
            out_cls = ImageData
        elif isinstance(data, MaskData):
            out_cls = MaskData
        else:
            return False, "Input must be ImageData or MaskData"

        arr_in = data.payload
        angle = float(self.get_property('angle'))

        self.set_progress(50)
        from scipy.ndimage import rotate as _nd_rotate
        rotated_arr = _nd_rotate(arr_in, angle, reshape=True, order=3,
                                 mode='constant', cval=0.0).astype(arr_in.dtype)

        self._make_image_output(rotated_arr)
        self.set_display(rotated_arr)
        self.set_progress(100)
        return True, None


# ===========================================================================
# MirrorNode
# ===========================================================================

class MirrorNode(BaseImageProcessNode):
    """
    Flips or mirrors an image or mask along one or both axes.

    Axis options:
    - *horizontal* — flip left-right (mirror across the vertical centre line)
    - *vertical* — flip top-bottom (mirror across the horizontal centre line)
    - *both* — flip both axes (equivalent to 180-degree rotation)

    Works with both ImageData and MaskData inputs.

    Keywords: mirror, flip, horizontal, vertical, reflect, 鏡像, 翻轉, 水平, 垂直, 影像處理
    """
    __identifier__ = 'nodes.image_process.Transform'
    NODE_NAME      = 'Mirror / Flip'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'], multi_output=True)
        self.add_combo_menu('axis', 'Flip Axis',
                            items=['horizontal', 'vertical', 'both'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        port = self.inputs().get('image')
        if not port or not port.connected_ports():
            return False, "No input connected"
        cp   = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())

        if isinstance(data, ImageData):
            out_cls = ImageData
        elif isinstance(data, MaskData):
            out_cls = MaskData
        else:
            return False, "Input must be ImageData or MaskData"

        arr = data.payload
        self.set_progress(30)
        axis = self.get_property('axis')
        if axis == 'horizontal':
            out_arr = np.fliplr(arr)
        elif axis == 'vertical':
            out_arr = np.flipud(arr)
        else:  # both
            out_arr = np.flipud(np.fliplr(arr))
        # Ensure contiguous for downstream consumers
        out_arr = np.ascontiguousarray(out_arr)

        self.output_values['image'] = out_cls(payload=out_arr)
        self.set_display(out_arr)
        self.set_progress(100)
        return True, None


# ===========================================================================
# RollingBallNode
# ===========================================================================

class RollingBallNode(BaseImageProcessNode):
    """
    Subtracts slowly-varying background illumination using rolling-ball estimation.

    Models the image surface as a landscape and rolls a sphere of the given radius underneath it. The sphere's path estimates the background, which is then subtracted to leave only local foreground features (cells, fibres, etc.). Works on grayscale or RGB images.

    Rule of thumb: set **radius** to slightly larger than the largest object of interest. Larger radius removes broader background gradients (default: 100.0).

    Keywords: rolling ball, background subtraction, uneven illumination, correction, subtract, 去背景, 背景扣除, 亮度, 均勻化, 影像處理
    """
    __identifier__ = 'nodes.image_process.filter'
    NODE_NAME      = 'Rolling Ball'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])
        self._add_float_spinbox('radius', 'Ball Radius (px)', value=100.0,
                                min_val=1.0, max_val=5000.0, step=10.0, decimals=1)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from PIL import Image as _PIL

        port = self.inputs().get('image')
        if not port or not port.connected_ports():
            return False, "No input connected"
        cp   = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, ImageData):
            return False, "Input must be ImageData"

        arr_in  = data.payload
        radius  = float(self.get_property('radius'))
        H, W    = arr_in.shape[0], arr_in.shape[1]

        self.set_progress(10)
        # Data is already float32 [0,1] — use directly
        arr = arr_in.astype(np.float32)

        # Try Rust (fast enough for full resolution); fall back to skimage
        # with downsampling since skimage is ~260x slower.
        rs_ok = False
        try:
            import image_process_rs as _rs
            if arr.ndim == 2:
                background = np.asarray(_rs.rolling_ball(arr, radius), dtype=np.float32)
            else:
                background = np.asarray(_rs.rolling_ball_rgb(arr, radius),
                                        dtype=np.float32).reshape(arr.shape)
            rs_ok = True
        except Exception:
            pass

        if not rs_ok:
            from skimage.restoration import rolling_ball
            from skimage.transform import resize as sk_resize
            MAX_DIM = 512
            scale   = max(1, max(H, W) // MAX_DIM)
            if scale > 1:
                sw, sh    = max(1, W // scale), max(1, H // scale)
                small_arr = sk_resize(arr, (sh, sw), preserve_range=True).astype(np.float32)
                small_bg  = rolling_ball(small_arr, radius=radius / scale).astype(np.float32)
                bg_up_arr = sk_resize(small_bg, (H, W), preserve_range=True).astype(np.float32)
                background = bg_up_arr
            else:
                background = rolling_ball(arr.astype(np.float32), radius=radius).astype(np.float32)

        self.set_progress(80)
        corrected = np.clip(arr - background, 0.0, 1.0).astype(np.float32)
        self._make_image_output(corrected)
        self.set_display(corrected)
        self.set_progress(100)
        return True, None


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Otsu Threshold
# ─────────────────────────────────────────────────────────────────────────────

class MultiOtsuNode(BaseImageProcessNode):
    """
    Splits image intensity into N classes using multi-Otsu thresholding.

    Uses `skimage.filters.threshold_multiotsu` to find optimal inter-class thresholds. Output is a LabelData integer array where each pixel is labelled 0 to N-1 (background = 0, brightest class = N-1).

    **n_classes** — number of intensity classes to separate (default: 3).

    Keywords: multi-otsu, multi otsu, threshold, classes, label, 多閾值, 多類別
    """
    __identifier__ = 'nodes.image_process.filter'
    NODE_NAME      = 'Multi-Otsu Threshold'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['label_image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('label_image', multi_output=True,
                        color=PORT_COLORS['label'])
        self._add_int_spinbox('n_classes', 'Classes', value=3, min_val=2, max_val=6)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            return False, 'No image connected'
        cp   = in_port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, ImageData):
            return False, 'Input must be ImageData'

        arr = data.payload
        gray = arr if arr.ndim == 2 else arr.mean(axis=2)
        gray = gray.astype(np.float64)
        self.set_progress(20)

        n_classes = int(self.get_property('n_classes'))
        try:
            from skimage.filters import threshold_multiotsu
            thresholds = threshold_multiotsu(gray, classes=n_classes)
        except Exception as e:
            return False, f'Multi-Otsu failed: {e}'
        self.set_progress(60)

        # np.digitize assigns each pixel to a class (0 .. n_classes-1)
        labels = np.digitize(gray, bins=thresholds).astype(np.int32)
        self.set_progress(80)

        # Build colored display image
        from skimage.color import label2rgb
        rgb = (label2rgb(labels, bg_label=-1) * 255).astype(np.uint8)

        self.output_values['label_image'] = LabelData(
            payload=labels, image=rgb)
        self.set_display(rgb)
        self.set_progress(100)
        return True, None


# ═══════════════════════════════════════════════════════════════════════
# ChannelColorizeNode — remap RGB channels to custom colors & composite
# ═══════════════════════════════════════════════════════════════════════

_COLOR_PRESETS = {
    'Red':     (255, 0, 0),
    'Green':   (0, 255, 0),
    'Blue':    (0, 0, 255),
    'Cyan':    (0, 255, 255),
    'Magenta': (255, 0, 255),
    'Yellow':  (255, 255, 0),
    'White':   (255, 255, 255),
    'Gray':    (128, 128, 128),
}


class _ColorButton(QtWidgets.QWidget):
    """Color swatch (click to pick) + small dropdown for presets."""
    color_changed = QtCore.Signal()

    def __init__(self, label, default_color, parent=None):
        super().__init__(parent)
        self._color = QtGui.QColor(*default_color)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self._swatch = QtWidgets.QPushButton(label)
        self._swatch.setFixedHeight(20)
        self._swatch.setMinimumWidth(24)
        self._swatch.clicked.connect(self._pick_color)
        layout.addWidget(self._swatch)

        self._drop = QtWidgets.QToolButton()
        self._drop.setText('\u25bc')
        self._drop.setFixedSize(14, 20)
        self._drop.setStyleSheet('font-size: 7px; border: 1px solid #555; border-radius: 2px; background: #333;')
        self._drop.clicked.connect(self._show_presets)
        layout.addWidget(self._drop)
        self._update_style()

    def _update_style(self):
        txt = '#000' if self._color.lightness() > 128 else '#fff'
        border = '#222' if self._color.lightness() > 128 else '#888'
        self._swatch.setStyleSheet(
            f"background-color: {self._color.name()}; border: 2px solid {border};"
            f" border-radius: 3px; color: {txt};"
            f" font-size: 9px; font-weight: bold;")

    def _pick_color(self):
        c = QtWidgets.QColorDialog.getColor(
            self._color, QtWidgets.QApplication.activeWindow(), "Channel Color")
        if c.isValid():
            self._color = c
            self._update_style()
            self.color_changed.emit()

    def _show_presets(self):
        menu = QtWidgets.QMenu(QtWidgets.QApplication.activeWindow())
        for name, rgb in _COLOR_PRESETS.items():
            action = menu.addAction(name)
            px = QtGui.QPixmap(12, 12)
            px.fill(QtGui.QColor(*rgb))
            action.setIcon(QtGui.QIcon(px))
            action.triggered.connect(lambda checked, r=rgb: self._set_preset(r))
        menu.exec(QtGui.QCursor.pos())

    def _set_preset(self, rgb):
        self._color = QtGui.QColor(*rgb)
        self._update_style()
        self.color_changed.emit()

    def get_color(self):
        return [self._color.red(), self._color.green(), self._color.blue()]

    def set_color(self, rgb):
        self._color = QtGui.QColor(rgb[0], rgb[1], rgb[2])
        self._update_style()


class _ChannelColorWidget(NodeBaseWidget):
    """Three color buttons in a single row: 1 2 3."""

    def __init__(self, parent=None, name='channel_colors', label=''):
        super().__init__(parent, name, label)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._btns = [
            _ColorButton('1', (255, 0, 0)),
            _ColorButton('2', (0, 255, 0)),
            _ColorButton('3', (0, 0, 255)),
        ]
        for btn in self._btns:
            btn.color_changed.connect(
                lambda: self.value_changed.emit(self.get_name(), self.get_value()))
            layout.addWidget(btn)

        self.set_custom_widget(container)

    def get_value(self):
        return [b.get_color() for b in self._btns]

    def set_value(self, value):
        if isinstance(value, list) and len(value) >= 3:
            for i, v in enumerate(value[:3]):
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    self._btns[i].set_color(v)


class ChannelColorizeNode(BaseImageProcessNode):
    """
    Remaps RGB channels to custom colors and composites them.

    Each channel can be assigned any color. The node multiplies each
    channel's grayscale intensity by its chosen color, then additively
    blends all channels into one RGB output.

    Use cases:

    - Change DAPI from blue to cyan
    - Show two channels in magenta + green for better contrast

    Keywords: colorize, LUT, pseudocolor, channel color, false color, composite, merge, 偽色, 通道顏色, 合成
    """
    __identifier__ = 'nodes.image_process.color'
    NODE_NAME      = 'Channel Colorize'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress', 'image_view',
        'show_preview', 'live_preview',
    })
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'], multi_output=True)

        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value

        w = _ChannelColorWidget(self.view, name='channel_colors', label='Channels')
        self.add_custom_widget(w, widget_type=H, tab='Properties')

        self.create_preview_widgets()
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()

        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No image connected"
        connected = in_port.connected_ports()[0]
        img_data = connected.node().output_values.get(connected.name())

        if not isinstance(img_data, ImageData):
            self.mark_error()
            return False, "Input is not an ImageData"

        self.set_progress(20)

        arr_raw = img_data.payload
        if arr_raw.ndim == 2:
            arr = np.stack([arr_raw, arr_raw, arr_raw], axis=-1).astype(np.float32)
        elif arr_raw.ndim == 3 and arr_raw.shape[2] == 4:
            arr = arr_raw[:, :, :3].astype(np.float32)
        else:
            arr = arr_raw.astype(np.float32)
        colors = self.get_property('channel_colors')
        if not colors or not isinstance(colors, list) or len(colors) < 3:
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

        self.set_progress(40)

        result = np.zeros_like(arr)
        for i, col in enumerate(colors[:3]):
            gray = arr[:, :, i]
            target = np.array(col, dtype=np.float32) / 255.0
            for c in range(3):
                result[:, :, c] += gray * target[c]

        self.set_progress(80)

        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        self._make_image_output(result)
        self.set_display(result)
        self.set_progress(100)
        return True, None


# ===========================================================================
# OIR Reader Node — reads OIR files and outputs individual channels
# ===========================================================================

class _OIRChannelColorWidget(NodeBaseWidget):
    """4 channel color buttons with per-channel grayscale checkbox."""

    def __init__(self, parent=None, name='channel_colors', label=''):
        super().__init__(parent, name, label)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        defaults = [
            ('1', (255, 0, 0)),      # Ch1: Red
            ('2', (0, 255, 0)),      # Ch2: Green
            ('3', (0, 0, 255)),      # Ch3: Blue
            ('4', (255, 0, 255)),    # Ch4: Magenta
        ]

        # Color buttons row
        color_row = QtWidgets.QHBoxLayout()
        color_row.setSpacing(3)
        self._btns = []
        for lbl, color in defaults:
            btn = _ColorButton(lbl, color)
            btn.color_changed.connect(
                lambda: self.value_changed.emit(self.get_name(), self.get_value()))
            color_row.addWidget(btn)
            self._btns.append(btn)
        layout.addLayout(color_row)

        # Grayscale checkboxes row
        gray_row = QtWidgets.QHBoxLayout()
        gray_row.setSpacing(3)
        self._gray_cbs = []
        for i in range(4):
            cb = QtWidgets.QCheckBox("Gray")
            cb.setStyleSheet("color: #999; font-size: 9px;")
            cb.setChecked(True)
            cb.toggled.connect(
                lambda _v: self.value_changed.emit(self.get_name(), self.get_value()))
            gray_row.addWidget(cb)
            self._gray_cbs.append(cb)
        layout.addLayout(gray_row)

        self.set_custom_widget(container)

    def get_value(self):
        return {
            'colors': [b.get_color() for b in self._btns],
            'grayscale': [cb.isChecked() for cb in self._gray_cbs],
        }

    def set_value(self, value):
        if isinstance(value, dict):
            colors = value.get('colors', [])
            grays = value.get('grayscale', [])
            for i, v in enumerate(colors[:len(self._btns)]):
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    self._btns[i].set_color(v)
            for i, v in enumerate(grays[:len(self._gray_cbs)]):
                self._gray_cbs[i].blockSignals(True)
                self._gray_cbs[i].setChecked(bool(v))
                self._gray_cbs[i].blockSignals(False)
        elif isinstance(value, list):
            # Backward compat: old format was just a list of colors
            for i, v in enumerate(value[:len(self._btns)]):
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    self._btns[i].set_color(v)


class OIRReaderNode(BaseImageProcessNode):
    """
    Read Olympus OIR microscopy files and output each channel separately.

    Outputs up to 4 individual grayscale channels and a colorized composite.
    Each channel can be assigned a display color (or grayscale) using the
    color buttons. Channels not present in the file output as black.

    Supports both native OIR format and TIFF files saved with .oir extension.
    Uses the Rust reader when available, with Python fallback.

    Keywords: OIR, Olympus, confocal, microscopy, multi-channel, fluorescence, 顯微鏡, 共焦, 多通道, 螢光
    """
    __identifier__ = 'nodes.image_process.IO'
    NODE_NAME      = 'OIR Reader'
    PORT_SPEC      = {
        'inputs': ['path'],
        'outputs': ['image', 'image', 'image', 'image', 'image']
    }

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress', 'image_view',
        'show_preview', 'live_preview', 'channel_colors',
    })
    _collection_aware = True

    def __init__(self):
        super().__init__()
        from nodes.base import NodeFileSelector
        import NodeGraphQt

        self.add_input('path', color=PORT_COLORS.get('path', (149, 165, 166)))
        self.add_output('ch1', color=PORT_COLORS['image'], multi_output=True)
        self.add_output('ch2', color=PORT_COLORS['image'], multi_output=True)
        self.add_output('ch3', color=PORT_COLORS['image'], multi_output=True)
        self.add_output('ch4', color=PORT_COLORS['image'], multi_output=True)
        self.add_output('composite', color=PORT_COLORS['image'], multi_output=True)

        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value

        file_w = NodeFileSelector(self.view, name='file_path', label='OIR File',
                                  ext_filter='*.oir')
        self.add_custom_widget(file_w, widget_type=H, tab='Properties')

        color_w = _OIRChannelColorWidget(self.view, name='channel_colors',
                                         label='Channel Colors')
        self.add_custom_widget(color_w, widget_type=H, tab='Properties')

        self.create_preview_widgets()
        self.output_values = {}

    def set_property(self, name, value, push_undo=True):
        super().set_property(name, value, push_undo)
        # channel_colors is in _UI_PROPS (won't auto-dirty), but user color
        # changes should trigger re-evaluation when live_preview is on
        if name == 'channel_colors' and self.get_property('live_preview'):
            self.mark_dirty()
            success, err = self.evaluate()
            if success:
                self.mark_clean()
            else:
                self.mark_error()

    def evaluate(self):
        self.reset_progress()

        # Get file path from input port or property
        file_path = None
        path_port = self.inputs().get('path')
        if path_port and path_port.connected_ports():
            cp = path_port.connected_ports()[0]
            val = cp.node().output_values.get(cp.name())
            if isinstance(val, str):
                file_path = val
            elif hasattr(val, 'payload'):
                file_path = str(val.payload)
        if not file_path:
            file_path = self.get_property('file_path')
        if not file_path:
            return False, "No OIR file selected"

        import os
        if not os.path.isfile(file_path):
            return False, f"File not found: {file_path}"

        self.set_progress(10)

        # Detect TIFF-disguised .oir files
        with open(file_path, 'rb') as f:
            magic = f.read(4)
        is_tiff = magic[:2] in (b'MM', b'II')

        try:
            if is_tiff:
                # Read as TIFF
                import tifffile
                arr = tifffile.imread(file_path)
                # tifffile returns (C, H, W) or (H, W) or (H, W, C)
                if arr.ndim == 2:
                    channels = [arr]
                elif arr.ndim == 3:
                    if arr.shape[0] <= 6:  # likely (C, H, W)
                        channels = [arr[i] for i in range(arr.shape[0])]
                    else:  # likely (H, W, C)
                        channels = [arr[:, :, i] for i in range(arr.shape[2])]
                else:
                    channels = [arr[0]] if arr.ndim > 3 else [arr]
                bit_depth = 12 if arr.dtype == np.uint16 else 8
                scale_um = None
            else:
                # Try Rust reader first, fall back to Python
                rust_ok = False
                try:
                    import oir_reader_rs
                    _name, img, _group, _isize = oir_reader_rs.read_oir_file(
                        file_path, list(range(1, 5)))
                    if img is not None:
                        # Rust reader returns (H, W) or (H, W, C) uint16
                        if img.ndim == 2:
                            channels = [img]
                        else:
                            channels = [img[:, :, i] for i in range(img.shape[2])]
                        try:
                            from synapse.nodes.io_nodes import _extract_oir_scale
                        except ImportError:
                            from nodes.io_nodes import _extract_oir_scale
                        scale_um = _extract_oir_scale(file_path)
                        from synapse.nodes.io_nodes import _guess_bit_depth
                        bit_depth = _guess_bit_depth(img)
                        rust_ok = True
                except (ImportError, Exception):
                    pass

                if not rust_ok:
                    # Python fallback
                    try:
                        from synapse.nodes.io_nodes import _py_read_single_oir
                    except ImportError:
                        from nodes.io_nodes import _py_read_single_oir
                    img, n_ch, scale_um, bit_depth = _py_read_single_oir(file_path)
                    channels = [img[:, :, i] for i in range(img.shape[2])]

        except Exception as e:
            return False, f"Failed to read OIR: {e}"

        self.set_progress(50)

        # Normalize each channel to float32 [0, 1]
        max_val = float((1 << bit_depth) - 1) if bit_depth > 8 else 255.0
        norm_channels = []
        for ch in channels:
            if ch.dtype in (np.float32, np.float64):
                norm_channels.append(ch.astype(np.float32))
            else:
                norm_channels.append(ch.astype(np.float32) / max_val)

        # Pad to 4 channels with black
        h, w = norm_channels[0].shape[:2]
        while len(norm_channels) < 4:
            norm_channels.append(np.zeros((h, w), dtype=np.float32))

        self.set_progress(70)

        # Read channel settings
        ch_settings = self.get_property('channel_colors')
        if isinstance(ch_settings, dict):
            colors = ch_settings.get('colors', [[0,255,0],[255,0,0],[0,255,255],[255,0,255]])
            grays = ch_settings.get('grayscale', [False, False, False, False])
        elif isinstance(ch_settings, list):
            colors = ch_settings
            grays = [False, False, False, False]
        else:
            colors = [[255,0,0],[0,255,0],[0,0,255],[255,0,255]]
            grays = [False, False, False, False]
        while len(colors) < 4:
            colors.append([255, 255, 255])
        while len(grays) < 4:
            grays.append(False)

        # Output individual channels (grayscale or colorized)
        scale = scale_um if not is_tiff else None
        for i, port_name in enumerate(['ch1', 'ch2', 'ch3', 'ch4']):
            ch = norm_channels[i]
            if ch.ndim == 3:
                ch = ch.mean(axis=2)
            if grays[i]:
                # Grayscale output
                out_ch = ch
            else:
                # Colorized output
                target = np.array(colors[i], dtype=np.float32) / 255.0
                out_ch = np.stack([ch * target[c] for c in range(3)], axis=-1)
                out_ch = np.clip(out_ch, 0.0, 1.0)
            self.output_values[port_name] = ImageData(
                payload=out_ch, bit_depth=bit_depth, scale_um=scale)

        # Build colorized composite (always uses colors, ignores grayscale toggle)
        composite = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(4):
            ch = norm_channels[i]
            if ch.ndim == 3:
                ch = ch.mean(axis=2)
            target = np.array(colors[i], dtype=np.float32) / 255.0
            for c in range(3):
                composite[:, :, c] += ch * target[c]
        composite = np.clip(composite, 0.0, 1.0)

        self.output_values['composite'] = ImageData(
            payload=composite, bit_depth=8, scale_um=scale)

        self.set_display(composite)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ===========================================================================
# Multi-Channel Brightness & Contrast
# ===========================================================================

class _MiniChannelBC(QtWidgets.QWidget):
    """Compact B&C controls for a single channel: mini histogram + min/max + B/C sliders."""
    params_changed = QtCore.Signal(int, float, float)  # channel_idx, min_val, max_val

    def __init__(self, ch_idx: int, default_color=(255, 255, 255), parent=None):
        super().__init__(parent)
        self._ch_idx = ch_idx
        self._full_range = 255.0
        self._updating = False
        self._color = default_color
        self._hist_counts = None
        self._hist_edges = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # ── Header: color button + Auto/Reset ────────────────────────
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(4)
        header.setContentsMargins(0, 0, 0, 0)
        self._color_btn = _ColorButton(str(ch_idx + 1), default_color)
        self._color_btn.color_changed.connect(self._on_color_changed)
        self._color_btn.setMaximumWidth(32)
        self._color_btn.setMaximumHeight(18)
        header.addWidget(self._color_btn)
        header.addStretch()
        btn_auto = QtWidgets.QPushButton('Auto')
        btn_auto.setToolTip('Auto stretch (0.5%–99.5%)')
        btn_auto.setFixedSize(32, 16)
        btn_auto.setStyleSheet('font-size: 7px; padding: 0px;')
        btn_auto.clicked.connect(self._auto)
        btn_reset = QtWidgets.QPushButton('Reset')
        btn_reset.setToolTip('Reset to full range')
        btn_reset.setFixedSize(32, 16)
        btn_reset.setStyleSheet('font-size: 7px; padding: 0px;')
        btn_reset.clicked.connect(self._reset)
        self._gray_cb = QtWidgets.QCheckBox('Gray')
        self._gray_cb.setToolTip('Output grayscale (unchecked = colorized RGB)')
        self._gray_cb.setStyleSheet('font-size: 8px; color: #ccc;')
        self._gray_cb.toggled.connect(lambda _: self.params_changed.emit(
            self._ch_idx, *self.get_range()))
        header.addWidget(btn_auto)
        header.addWidget(btn_reset)
        header.addWidget(self._gray_cb)
        layout.addLayout(header)

        # ── Mini histogram (pyqtgraph) ──────────────────────────────
        self._plot = pg.PlotWidget(background='#1a1a1a')
        self._plot.setFixedHeight(36)
        self._plot.setMaximumWidth(250)
        self._plot.hideAxis('left')
        self._plot.hideAxis('bottom')
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.getViewBox().setDefaultPadding(0)

        r, g, b = default_color
        self._hist_curve = self._plot.plot(
            pen=pg.mkPen(r, g, b, 180, width=1),
            fillLevel=0,
            brush=pg.mkBrush(r, g, b, 60),
            stepMode='center',
        )
        self._line_min = pg.InfiniteLine(pos=0, angle=90, movable=True,
                                         pen=pg.mkPen('#e74c3c', width=1.5))
        self._line_max = pg.InfiniteLine(pos=255, angle=90, movable=True,
                                         pen=pg.mkPen('#3498db', width=1.5))
        self._plot.addItem(self._line_min)
        self._plot.addItem(self._line_max)
        self._line_min.sigPositionChanged.connect(self._on_line_moved)
        self._line_max.sigPositionChanged.connect(self._on_line_moved)
        layout.addWidget(self._plot)

        # ── Min/Max + Brightness/Contrast sliders (compact grid) ────
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(1)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 1)

        _SS = 'color: #aaa; font-size: 7px;'
        self._sld_min = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sld_min.setRange(0, 255)
        self._spn_min = QtWidgets.QSpinBox()
        self._spn_min.setRange(0, 255)
        self._spn_min.setFixedWidth(52)
        self._spn_min.setStyleSheet('font-size: 7px; color: #ccc;')

        self._sld_max = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sld_max.setRange(0, 255)
        self._sld_max.setValue(255)
        self._spn_max = QtWidgets.QSpinBox()
        self._spn_max.setRange(0, 255)
        self._spn_max.setValue(255)
        self._spn_max.setFixedWidth(52)
        self._spn_max.setStyleSheet('font-size: 7px; color: #ccc;')

        self._sld_brightness = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sld_brightness.setRange(-1000, 1000)
        self._spn_brightness = QtWidgets.QDoubleSpinBox()
        self._spn_brightness.setRange(-100.0, 100.0)
        self._spn_brightness.setDecimals(1)
        self._spn_brightness.setSingleStep(0.5)
        self._spn_brightness.setFixedWidth(52)
        self._spn_brightness.setStyleSheet('font-size: 7px; color: #ccc;')

        self._sld_contrast = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sld_contrast.setRange(-1000, 1000)
        self._spn_contrast = QtWidgets.QDoubleSpinBox()
        self._spn_contrast.setRange(-100.0, 100.0)
        self._spn_contrast.setDecimals(1)
        self._spn_contrast.setSingleStep(0.5)
        self._spn_contrast.setFixedWidth(52)
        self._spn_contrast.setStyleSheet('font-size: 7px; color: #ccc;')

        for row, (lbl_text, sld, spn) in enumerate([
            ('Min', self._sld_min, self._spn_min),
            ('Max', self._sld_max, self._spn_max),
            ('B',   self._sld_brightness, self._spn_brightness),
            ('C',   self._sld_contrast, self._spn_contrast),
        ]):
            lbl = QtWidgets.QLabel(lbl_text)
            lbl.setStyleSheet(_SS)
            lbl.setFixedWidth(16)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(sld, row, 1)
            grid.addWidget(spn, row, 2)
        layout.addLayout(grid)

        # ── Wire signals ────────────────────────────────────────────
        self._sld_min.valueChanged.connect(lambda v: self._on_minmax_slider(v, 'min'))
        self._spn_min.valueChanged.connect(lambda v: self._on_minmax_spinbox(v, 'min'))
        self._sld_max.valueChanged.connect(lambda v: self._on_minmax_slider(v, 'max'))
        self._spn_max.valueChanged.connect(lambda v: self._on_minmax_spinbox(v, 'max'))
        self._sld_brightness.valueChanged.connect(lambda v: self._on_bc_slider(v / 10.0, 'b'))
        self._spn_brightness.valueChanged.connect(lambda v: self._on_bc_spinbox(v, 'b'))
        self._sld_contrast.valueChanged.connect(lambda v: self._on_bc_slider(v / 10.0, 'c'))
        self._spn_contrast.valueChanged.connect(lambda v: self._on_bc_spinbox(v, 'c'))

    # ── Public API ───────────────────────────────────────────────────

    def set_histogram(self, arr_flat, full_range=255):
        old_full = self._full_range
        self._full_range = float(full_range)
        num_bins = min(256, int(full_range) + 1)
        counts, edges = np.histogram(arr_flat, bins=num_bins, range=(0, float(full_range)))
        self._hist_counts = counts
        self._hist_edges = edges
        self._hist_curve.setData(x=edges, y=np.log1p(counts.astype(float)))
        self._plot.setXRange(0, full_range, padding=0)
        irange = int(full_range)
        for w in (self._sld_min, self._sld_max):
            w.setRange(0, irange)
        for w in (self._spn_min, self._spn_max):
            w.setRange(0, irange)
        # Reset max handle to full range when bit depth changes
        _, cur_max = self.get_range()
        if cur_max <= old_full or cur_max <= 255:
            self._set_minmax(0, float(irange))

    def set_range(self, min_val, max_val):
        self._set_minmax(min_val, max_val)

    def get_range(self):
        return float(self._line_min.value()), float(self._line_max.value())

    def get_color(self):
        return self._color_btn.get_color()

    def is_grayscale(self):
        return self._gray_cb.isChecked()

    def set_grayscale(self, val):
        self._gray_cb.blockSignals(True)
        self._gray_cb.setChecked(bool(val))
        self._gray_cb.blockSignals(False)

    def set_color(self, rgb):
        self._color_btn.set_color(rgb)
        self._on_color_changed()

    # ── Internals ────────────────────────────────────────────────────

    def _on_color_changed(self):
        c = self._color_btn.get_color()
        self._color = tuple(c)
        r, g, b = c
        self._hist_curve.setPen(pg.mkPen(r, g, b, 180, width=1))
        self._hist_curve.setBrush(pg.mkBrush(r, g, b, 60))
        # Emit so the node re-composites
        mn, mx = self.get_range()
        self.params_changed.emit(self._ch_idx, mn, mx)

    def _set_minmax(self, min_v, max_v):
        self._updating = True
        try:
            self._line_min.setValue(min_v)
            self._line_max.setValue(max_v)
            self._sld_min.setValue(int(round(min_v)))
            self._sld_max.setValue(int(round(max_v)))
            self._spn_min.setValue(int(round(min_v)))
            self._spn_max.setValue(int(round(max_v)))
            self._refresh_bc_controls(min_v, max_v)
        finally:
            self._updating = False

    def _refresh_bc_controls(self, min_v, max_v):
        full = self._full_range
        center = (min_v + max_v) / 2.0
        width = max_v - min_v
        brightness = (center - full / 2.0) / (full / 2.0) * 100.0 if full > 0 else 0
        contrast = (1.0 - width / full) * 100.0 if full > 0 else 0
        self._sld_brightness.setValue(int(round(brightness * 10)))
        self._spn_brightness.setValue(brightness)
        self._sld_contrast.setValue(int(round(contrast * 10)))
        self._spn_contrast.setValue(contrast)

    def _on_line_moved(self):
        if self._updating:
            return
        min_v = float(self._line_min.value())
        max_v = float(self._line_max.value())
        if min_v >= max_v:
            return
        self._set_minmax(min_v, max_v)
        self.params_changed.emit(self._ch_idx, min_v, max_v)

    def _on_minmax_slider(self, int_val, which):
        if self._updating:
            return
        val = float(int_val)
        min_v = val if which == 'min' else float(self._line_min.value())
        max_v = val if which == 'max' else float(self._line_max.value())
        if min_v >= max_v:
            return
        self._set_minmax(min_v, max_v)
        self.params_changed.emit(self._ch_idx, min_v, max_v)

    def _on_minmax_spinbox(self, spn_val, which):
        if self._updating:
            return
        val = float(spn_val)
        min_v = val if which == 'min' else float(self._line_min.value())
        max_v = val if which == 'max' else float(self._line_max.value())
        if min_v >= max_v:
            return
        self._set_minmax(min_v, max_v)
        self.params_changed.emit(self._ch_idx, min_v, max_v)

    def _on_bc_slider(self, float_val, which):
        if self._updating:
            return
        self._updating = True
        try:
            if which == 'b':
                self._spn_brightness.setValue(float_val)
            else:
                self._spn_contrast.setValue(float_val)
        finally:
            self._updating = False
        self._apply_bc_spinboxes()

    def _on_bc_spinbox(self, float_val, which):
        if self._updating:
            return
        self._updating = True
        try:
            if which == 'b':
                self._sld_brightness.setValue(int(round(float_val * 10)))
            else:
                self._sld_contrast.setValue(int(round(float_val * 10)))
        finally:
            self._updating = False
        self._apply_bc_spinboxes()

    def _apply_bc_spinboxes(self):
        brightness = self._spn_brightness.value()
        contrast = self._spn_contrast.value()
        full = self._full_range
        new_width = max(1.0, (1.0 - contrast / 100.0) * full)
        center = full / 2.0 + (brightness / 100.0) * (full / 2.0)
        new_min = max(0.0, min(center - new_width / 2.0, full - 1.0))
        new_max = max(1.0, min(center + new_width / 2.0, full))
        if new_min >= new_max:
            return
        self._set_minmax(new_min, new_max)
        self.params_changed.emit(self._ch_idx, new_min, new_max)

    def _auto(self):
        if self._hist_counts is None:
            return
        cumsum = np.cumsum(self._hist_counts)
        total = float(cumsum[-1])
        edges = self._hist_edges
        lo_idx = int(np.clip(np.searchsorted(cumsum, total * 0.005), 0, len(edges) - 2))
        hi_idx = int(np.clip(np.searchsorted(cumsum, total * 0.995), 0, len(edges) - 2))
        new_min = float(edges[lo_idx])
        new_max = float(edges[hi_idx + 1])
        if new_min >= new_max:
            new_min, new_max = 0.0, self._full_range
        self._set_minmax(new_min, new_max)
        self.params_changed.emit(self._ch_idx, new_min, new_max)

    def _reset(self):
        self._set_minmax(0.0, self._full_range)
        self.params_changed.emit(self._ch_idx, 0.0, self._full_range)


def _rgb_to_hex(rgb):
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'


class _NodeMultiChannelBCWidget(NodeBaseWidget):
    """Container widget embedding 4 _MiniChannelBC panels in a 2x2 grid."""
    _set_data_sig = QtCore.Signal(int, object, float)  # panel_idx, arr_flat, full_range
    _enable_sig = QtCore.Signal(int, bool)              # panel_idx, enabled

    def __init__(self, parent=None):
        super().__init__(parent, name='mc_bc_widget', label='')
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        defaults = [
            (255, 0, 0),      # Ch1: Red
            (0, 255, 0),      # Ch2: Green
            (0, 0, 255),      # Ch3: Blue
            (255, 0, 255),    # Ch4: Magenta
        ]
        self._panels: list[_MiniChannelBC] = []
        for i, color in enumerate(defaults):
            panel = _MiniChannelBC(i, default_color=color)
            panel.setEnabled(False)
            panel.setStyleSheet('QWidget { opacity: 0.3; }')
            grid.addWidget(panel, i // 2, i % 2)
            self._panels.append(panel)

        self.set_custom_widget(container)
        self._set_data_sig.connect(self._set_histogram_main,
                                   QtCore.Qt.ConnectionType.QueuedConnection)
        self._enable_sig.connect(self._set_enabled_main,
                                 QtCore.Qt.ConnectionType.QueuedConnection)

    def panel(self, idx) -> _MiniChannelBC:
        return self._panels[idx]

    def set_channel_enabled_threadsafe(self, panel_idx: int, enabled: bool):
        if threading.current_thread() is threading.main_thread():
            self._set_enabled_main(panel_idx, enabled)
        else:
            self._enable_sig.emit(panel_idx, enabled)

    def set_histogram_threadsafe(self, panel_idx, arr_flat, full_range=255):
        if threading.current_thread() is threading.main_thread():
            self._set_histogram_main(panel_idx, arr_flat, full_range)
        else:
            self._set_data_sig.emit(panel_idx, arr_flat, float(full_range))

    def _set_enabled_main(self, panel_idx, enabled):
        if 0 <= panel_idx < len(self._panels):
            p = self._panels[panel_idx]
            p.setEnabled(enabled)
            # Dim inactive panels
            p.setStyleSheet('' if enabled else 'QWidget { color: #555; }')
            if not enabled:
                # Clear histogram for inactive channel
                p._hist_curve.setData(x=[], y=[])

    def _set_histogram_main(self, panel_idx, arr_flat, full_range):
        if 0 <= panel_idx < len(self._panels):
            self._panels[panel_idx].set_histogram(arr_flat, full_range)

    def get_value(self):
        return {
            'ranges': [p.get_range() for p in self._panels],
            'colors': [p.get_color() for p in self._panels],
            'grayscale': [p.is_grayscale() for p in self._panels],
        }

    def set_value(self, value):
        if not isinstance(value, dict):
            return
        ranges = value.get('ranges', [])
        colors = value.get('colors', [])
        grays = value.get('grayscale', [])
        for i, (mn, mx) in enumerate(ranges[:4]):
            self._panels[i].set_range(float(mn), float(mx))
        for i, c in enumerate(colors[:4]):
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                self._panels[i].set_color(c)
        for i, g in enumerate(grays[:4]):
            self._panels[i].set_grayscale(g)


class MultiChannelBCNode(BaseImageProcessNode):
    """
    Per-channel brightness & contrast with live composite preview.

    Connect 1–4 grayscale channels (or a single multi-channel image) and adjust
    each channel's display window independently. Each channel has its own
    histogram with draggable min/max lines, brightness/contrast sliders, and
    a color button. The composite output blends all channels additively.

    Works as a single-channel B&C node when only one input is connected.

    Keywords: brightness, contrast, multi-channel, composite, merge, colorize, 亮度, 對比, 多通道, 合成
    """
    __identifier__ = 'nodes.image_process.Exposure'
    NODE_NAME      = 'Multi-Channel B&C'
    PORT_SPEC      = {
        'inputs': ['image', 'image', 'image', 'image', 'image'],
        'outputs': ['image', 'image', 'image', 'image', 'image'],
    }

    _UI_PROPS = BaseImageProcessNode._UI_PROPS | frozenset({'mc_bc_widget'})

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_input('ch1', color=PORT_COLORS['image'])
        self.add_input('ch2', color=PORT_COLORS['image'])
        self.add_input('ch3', color=PORT_COLORS['image'])
        self.add_input('ch4', color=PORT_COLORS['image'])

        self.add_output('ch1', color=PORT_COLORS['image'], multi_output=True)
        self.add_output('ch2', color=PORT_COLORS['image'], multi_output=True)
        self.add_output('ch3', color=PORT_COLORS['image'], multi_output=True)
        self.add_output('ch4', color=PORT_COLORS['image'], multi_output=True)
        self.add_output('composite', color=PORT_COLORS['image'], multi_output=True)

        self._mc_widget = _NodeMultiChannelBCWidget(self.view)
        self.add_custom_widget(self._mc_widget)
        for i in range(4):
            self._mc_widget.panel(i).params_changed.connect(self._on_channel_changed)

        self.create_preview_widgets()
        self._cached_channels: list[np.ndarray | None] = [None] * 4
        self._cached_channels_small: list[np.ndarray | None] = [None] * 4
        self._cached_bit_depths: list[int] = [8, 8, 8, 8]
        self._cached_scale_ums: list[float | None] = [None, None, None, None]

        # Debounce timer: fast preview first, full-res build after settling
        self._bc_debounce = QtCore.QTimer()
        self._bc_debounce.setSingleShot(True)
        self._bc_debounce.setInterval(120)
        self._bc_debounce.timeout.connect(self._on_debounce_fire)

    # Max pixels for the downsampled preview used during slider drag
    _PREVIEW_MAX_PX = 512 * 512

    def _on_channel_changed(self, ch_idx, min_val, max_val):
        """Live update when user drags a handle — debounced."""
        if self._cached_channels[ch_idx] is None:
            return
        # Instant preview on downsampled data
        preview = self._build_composite(use_small=True)
        if preview is not None:
            self.set_display(preview)
        # Restart debounce for full-res build + downstream cascade
        self._bc_debounce.start()

    def _on_debounce_fire(self):
        """Debounce expired — build full-res outputs and cascade."""
        result = self._build_outputs()
        if result:
            self.set_display(self.output_values.get('composite', result).payload)
            for out_port in self.outputs().values():
                for in_port in out_port.connected_ports():
                    dn = in_port.node()
                    if hasattr(dn, 'mark_dirty'):
                        dn.mark_dirty()

    def evaluate(self):
        self.reset_progress()

        # ── Collect channels ─────────────────────────────────────────
        channels: list[np.ndarray | None] = [None] * 4

        # Single multi-channel image input
        img_port = self.inputs().get('image')
        if img_port and img_port.connected_ports():
            cp = img_port.connected_ports()[0]
            data = cp.node().output_values.get(cp.name())
            if isinstance(data, ImageData):
                arr = data.payload
                bd = getattr(data, 'bit_depth', 8) or 8
                sc = getattr(data, 'scale_um', None)
                if arr.ndim == 3:
                    for i in range(min(4, arr.shape[2])):
                        channels[i] = arr[:, :, i]
                        self._cached_bit_depths[i] = bd
                        self._cached_scale_ums[i] = sc
                elif arr.ndim == 2:
                    channels[0] = arr
                    self._cached_bit_depths[0] = bd
                    self._cached_scale_ums[0] = sc

        # Individual channel inputs (override multi-channel)
        for i, port_name in enumerate(['ch1', 'ch2', 'ch3', 'ch4']):
            port = self.inputs().get(port_name)
            if port and port.connected_ports():
                cp = port.connected_ports()[0]
                data = cp.node().output_values.get(cp.name())
                if isinstance(data, ImageData):
                    arr = data.payload
                    if arr.ndim == 3:
                        arr = arr.mean(axis=2)
                    channels[i] = arr.astype(np.float32)
                    self._cached_bit_depths[i] = getattr(data, 'bit_depth', 8) or 8
                    self._cached_scale_ums[i] = getattr(data, 'scale_um', None)

        n_active = sum(1 for c in channels if c is not None)
        if n_active == 0:
            return False, "No channels connected"

        self._cached_channels = channels
        self._cached_channels_small = self._downsample_channels(channels)
        self.set_progress(20)

        # ── Update histograms (thread-safe) ──────────────────────────
        for i in range(4):
            if channels[i] is not None:
                max_possible = (1 << self._cached_bit_depths[i]) - 1
                flat_scaled = channels[i].ravel() * max_possible
                self._mc_widget.set_histogram_threadsafe(i, flat_scaled, full_range=max_possible)
                self._mc_widget.set_channel_enabled_threadsafe(i, True)
            else:
                self._mc_widget.set_channel_enabled_threadsafe(i, False)

        self.set_progress(50)

        # On first evaluate, widget ranges may not have updated yet (queued signals),
        # so force full-range output (no B&C clipping)
        self._build_outputs(first_eval=True)
        self.set_progress(100)
        return True, None

    def _downsample_channels(self, channels):
        """Pre-compute downsampled copies for fast preview during drag."""
        small: list[np.ndarray | None] = [None] * 4
        for i, ch in enumerate(channels):
            if ch is None:
                continue
            h, w = ch.shape[:2]
            total = h * w
            if total <= self._PREVIEW_MAX_PX:
                small[i] = ch  # already small enough, reuse
            else:
                factor = max(2, int(np.sqrt(total / self._PREVIEW_MAX_PX) + 0.5))
                small[i] = ch[::factor, ::factor]
        return small

    def _build_composite(self, use_small=False):
        """Build only the composite preview array (no output_values update)."""
        channels = self._cached_channels_small if use_small else self._cached_channels
        bds = self._cached_bit_depths
        n_active = sum(1 for c in channels if c is not None)

        h, w = 0, 0
        for c in channels:
            if c is not None:
                h, w = c.shape[:2]
                break
        if h == 0:
            return None

        composite = np.zeros((h, w, 3), dtype=np.float32)
        single_adjusted = None
        single_gray = False

        for i in range(4):
            if channels[i] is None:
                continue
            max_possible = float((1 << bds[i]) - 1)
            panel = self._mc_widget.panel(i)
            min_v, max_v = panel.get_range()
            color = panel.get_color()

            lo = min_v / max_possible
            hi = max_v / max_possible
            width = hi - lo
            if width <= 0:
                width = 1.0 / max_possible
            adjusted = np.clip((channels[i] - lo) / width, 0.0, 1.0)
            target = np.array(color, dtype=np.float32) / 255.0

            for c in range(3):
                composite[:, :, c] += adjusted * target[c]

            if n_active == 1:
                single_adjusted = adjusted
                single_gray = panel.is_grayscale()

        composite = np.clip(composite, 0.0, 1.0)
        if n_active == 1 and single_adjusted is not None and single_gray:
            return single_adjusted
        return composite

    def _build_outputs(self, first_eval=False):
        """Apply per-channel B&C and build composite.

        Args:
            first_eval: if True, use full range per channel when the widget
                        hasn't been updated yet (first evaluate).
        """
        channels = self._cached_channels
        bds = self._cached_bit_depths
        n_active = sum(1 for c in channels if c is not None)

        # Find image dimensions from first available channel
        h, w = 0, 0
        for c in channels:
            if c is not None:
                h, w = c.shape[:2]
                break
        if h == 0:
            return False

        composite = np.zeros((h, w, 3), dtype=np.float32)

        for i in range(4):
            if channels[i] is None:
                continue
            bd = bds[i]
            max_possible = float((1 << bd) - 1)
            panel = self._mc_widget.panel(i)
            min_v, max_v = panel.get_range()
            # On first load, widget may still have old 0-255 range
            if first_eval and max_v <= 255 and max_possible > 255:
                min_v, max_v = 0.0, max_possible
            color = panel.get_color()

            # Apply B&C: linear stretch [min_v, max_v] → [0, 1]
            lo = min_v / max_possible
            hi = max_v / max_possible
            width = hi - lo
            if width <= 0:
                width = 1.0 / max_possible
            adjusted = np.clip((channels[i] - lo) / width, 0.0, 1.0).astype(np.float32)

            # Color target for composite
            target = np.array(color, dtype=np.float32) / 255.0

            # Output individual channel: grayscale or colorized
            port_name = f'ch{i + 1}'
            sc = self._cached_scale_ums[i]
            if panel.is_grayscale():
                self.output_values[port_name] = ImageData(
                    payload=adjusted, bit_depth=bd, scale_um=sc)
            else:
                ch_rgb = np.stack([adjusted * target[c] for c in range(3)], axis=-1)
                ch_rgb = np.clip(ch_rgb, 0.0, 1.0)
                self.output_values[port_name] = ImageData(
                    payload=ch_rgb, bit_depth=bd, scale_um=sc)

            # Add to composite
            for c in range(3):
                composite[:, :, c] += adjusted * target[c]

        composite = np.clip(composite, 0.0, 1.0)
        active_idx = [i for i in range(4) if channels[i] is not None]
        composite_bd = max(bds[i] for i in active_idx)
        composite_sc = next((self._cached_scale_ums[i] for i in active_idx
                             if self._cached_scale_ums[i] is not None), None)

        # For single channel, output grayscale composite
        if n_active == 1:
            i = active_idx[0]
            self.output_values['composite'] = ImageData(
                payload=self.output_values[f'ch{i+1}'].payload,
                bit_depth=bds[i], scale_um=self._cached_scale_ums[i])
        else:
            self.output_values['composite'] = ImageData(
                payload=composite, bit_depth=composite_bd,
                scale_um=composite_sc)

        self.set_display(composite if n_active > 1 else
                         self.output_values['composite'].payload)
        self.mark_clean()
        return True


# ===========================================================================
# Merge Image Node — additive blend of multiple images
# ===========================================================================

class MergeImageNode(BaseImageProcessNode):
    """
    Additively blend multiple images into one output.

    Connect any number of images to the input port. The node sums all
    input pixel values and clips to [0, 1]. Useful for combining
    individually color-adjusted channels into a single composite
    (e.g. merge ch1 + ch2 + ch4, skipping DAPI).

    Works with both grayscale and RGB inputs. Grayscale inputs are
    broadcast across all 3 RGB channels.

    Keywords: merge, add, blend, combine, composite, sum, overlay, 合成, 合併, 疊加
    """
    __identifier__ = 'nodes.image_process.color'
    NODE_NAME      = 'Merge Image'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image', multi_input=True, color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'], multi_output=True)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        port = self.inputs().get('image')
        if not port or not port.connected_ports():
            return False, "No images connected"

        arrays = []
        bit_depth = 8
        scale_um = None
        for cp in port.connected_ports():
            data = cp.node().output_values.get(cp.name())
            if isinstance(data, ImageData):
                arrays.append(data.payload.astype(np.float32))
                bit_depth = max(bit_depth, getattr(data, 'bit_depth', 8) or 8)
                if scale_um is None:
                    scale_um = getattr(data, 'scale_um', None)

        if not arrays:
            return False, "No valid ImageData inputs"

        self.set_progress(30)

        # Find max dimensions
        max_h = max(a.shape[0] for a in arrays)
        max_w = max(a.shape[1] for a in arrays)

        # Determine if output should be RGB
        any_rgb = any(a.ndim == 3 for a in arrays)

        result = np.zeros((max_h, max_w, 3) if any_rgb else (max_h, max_w),
                          dtype=np.float32)

        self.set_progress(50)

        for arr in arrays:
            h, w = arr.shape[:2]
            if any_rgb and arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if any_rgb:
                result[:h, :w, :] += arr[:, :, :3]
            else:
                result[:h, :w] += arr

        result = np.clip(result, 0.0, 1.0)

        out = self._make_image_output(result)
        out.bit_depth = bit_depth
        out.scale_um = scale_um
        self.set_display(result)
        self.set_progress(100)
        self.mark_clean()
        return True, None
