"""
filopodia_nodes/cell_iteration_nodes.py
=======================================
Per-cell iteration nodes for the filopodia-per-cell workflow.

Given an image and a CollectionData of per-cell boolean masks (which may
overlap), crop each cell out with a margin, iterate cells one at a time
through the filopodia pipeline, and roll each cell's per-branch table into a
single summary row.

Nodes: CropCellsNode, CellIteratorNode, FilopodiaCellSummaryNode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_models import ImageData, MaskData, TableData, CollectionData
from nodes.base import BaseExecutionNode, PORT_COLORS

# Sibling helpers (dual pattern: package import in-app, flat import under tests
# where the plugin dir may be on sys.path directly).
try:
    from .filopodia_nodes import _read_port, _mask_to_bool
except ImportError:  # pragma: no cover - flat-load fallback
    from filopodia_nodes import _read_port, _mask_to_bool


def _crop_cell(image_arr: np.ndarray, mask_bool: np.ndarray, margin: int):
    """Crop `image_arr` to `mask_bool`'s bounding box padded by `margin` pixels.

    Returns (crop_img, crop_mask). crop_img keeps the source dtype and channel
    layout; crop_mask is boolean. No neighbour exclusion -- overlap is kept.
    Returns (None, None) when the mask has no foreground pixels.
    """
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return None, None
    h, w = mask_bool.shape
    y0 = max(0, int(ys.min()) - margin)
    x0 = max(0, int(xs.min()) - margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    crop_img = image_arr[y0:y1, x0:x1]
    crop_mask = mask_bool[y0:y1, x0:x1]
    return crop_img, crop_mask


_COLLECTION_COLOR = PORT_COLORS.get('collection', (218, 165, 32))


class CropCellsNode(BaseExecutionNode):
    """
    Crops each cell out of an image using a collection of per-cell masks.

    Input `cells` is a CollectionData of boolean masks (one per cell, may
    overlap). For each mask, the image is cropped to the mask's bounding box
    padded by **margin** pixels (so filopodia beyond the body are not clipped).
    Outputs two collections keyed identically to the input: `cell_images`
    (the crops) and `cell_masks` (the per-cell body masks, cropped to match).

    Keywords: crop, per cell, filopodia, split cells, collection, 裁切, 細胞, 絲足
    """
    __identifier__ = 'plugins.Filopodia'
    NODE_NAME      = 'Crop Cells'
    PORT_SPEC      = {'inputs': ['image', 'cells'],
                      'outputs': ['cell_images', 'cell_masks']}
    _handles_collection = True   # reads the CollectionData itself; no auto-loop

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_input('cells', color=_COLLECTION_COLOR)
        self.add_output('cell_images', color=_COLLECTION_COLOR, multi_output=True)
        self.add_output('cell_masks',  color=_COLLECTION_COLOR, multi_output=True)
        self._add_int_spinbox('margin', 'Crop Margin (px)',
                              value=50, min_val=0, max_val=2000)

    def evaluate(self):
        self.reset_progress()
        img = _read_port(self, 'image')
        cells = _read_port(self, 'cells')
        if not isinstance(img, ImageData):
            self.mark_error()
            return False, 'No image connected'
        if not isinstance(cells, CollectionData):
            self.mark_error()
            return False, 'No cell mask collection connected'

        margin = int(self.get_property('margin'))
        image_arr = img.payload
        # Carry through display/scale metadata from the source image.
        meta_fields = {f: getattr(img, f, None)
                       for f in ('bit_depth', 'scale_um', 'display_min', 'display_max')}

        cell_images: dict = {}
        cell_masks: dict = {}
        items = list(cells.payload.items())
        total = max(1, len(items))
        for idx, (key, mask_item) in enumerate(items):
            self.set_progress(int(idx / total * 100))
            if not isinstance(mask_item, MaskData):
                continue
            mask_bool = _mask_to_bool(mask_item)
            crop_img, crop_mask = _crop_cell(image_arr, mask_bool, margin)
            if crop_img is None:
                continue
            cell_images[key] = ImageData(payload=np.ascontiguousarray(crop_img),
                                         **meta_fields)
            cell_masks[key] = MaskData(
                payload=(crop_mask.astype(np.uint8) * 255),
                bit_depth=8, scale_um=meta_fields['scale_um'])

        self.output_values['cell_images'] = CollectionData(payload=cell_images)
        self.output_values['cell_masks'] = CollectionData(payload=cell_masks)
        self.set_progress(100)
        self.mark_clean()
        return True, None


class CellIteratorNode(BaseExecutionNode):
    """
    Batch iterator over per-cell crops. Emits one cell (image + mask) per
    iteration so the downstream filopodia nodes process a single cell at a
    time. Use with a Batch Gate to tune parameters per cell, then a Batch
    Accumulator to collect the per-cell results.

    Wire `cell_images` and `cell_masks` from a Crop Cells node. Run once to
    populate the collections, then Batch Run to iterate every cell.

    Keywords: iterate cells, per cell, batch, filopodia, loop, 逐一, 細胞, 批次, 絲足
    """
    __identifier__ = 'plugins.Filopodia'
    NODE_NAME      = 'Cell Iterator'
    PORT_SPEC      = {'inputs': ['cell_images', 'cell_masks'],
                      'outputs': ['image', 'cell_mask', 'cell_key']}
    _handles_collection = True   # reads the CollectionData itself; no auto-loop

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('cell_images', color=_COLLECTION_COLOR)
        self.add_input('cell_masks',  color=_COLLECTION_COLOR)
        self.add_output('image',    color=PORT_COLORS['image'])
        self.add_output('cell_mask', color=PORT_COLORS['mask'])
        self.add_output('cell_key', color=PORT_COLORS.get('path', (170, 170, 170)))
        # The batch engine writes the current item key into this property.
        self.create_property('current_file', '')

    def get_batch_items(self) -> list:
        imgs = _read_port(self, 'cell_images')
        if not isinstance(imgs, CollectionData):
            return []
        return list(imgs.payload.keys())

    def evaluate(self):
        imgs = _read_port(self, 'cell_images')
        masks = _read_port(self, 'cell_masks')
        if not isinstance(imgs, CollectionData) or not isinstance(masks, CollectionData):
            self.mark_error()
            return False, 'Connect cell_images and cell_masks from a Crop Cells node'

        keys = list(imgs.payload.keys())
        if not keys:
            self.mark_error()
            return False, 'Cell collection is empty'

        key = self.get_property('current_file')
        if not key or key not in imgs.payload:
            key = keys[0]   # single-run preview -> first cell

        img_item = imgs.payload.get(key)
        mask_item = masks.payload.get(key)
        self.output_values['image'] = img_item
        self.output_values['cell_mask'] = mask_item
        self.output_values['cell_key'] = key
        if img_item is not None and hasattr(img_item, 'payload'):
            self.set_display(img_item.payload)
        self.mark_clean()
        return True, None


class FilopodiaCellSummaryNode(BaseExecutionNode):
    """
    Rolls one cell's per-branch filopodia table into a single summary row:
    count, mean/total filopodium length, cell edge length, and density
    (filopodia per unit edge length -- micrometres when a scale is available,
    otherwise pixels).

    Place after Filopodia Analyze and before a Batch Accumulator; the
    accumulator stamps each row with the cell key (from Cell Iterator) as a
    `file` column and concatenates all cells into the final per-cell table.

    Keywords: summary, per cell, count, density, filopodia, 統計, 每個細胞, 數量, 密度, 絲足
    """
    __identifier__ = 'plugins.Filopodia'
    NODE_NAME      = 'Filopodia Cell Summary'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['summary']}
    OUTPUT_COLUMNS = {
        'summary': ['filopodia_count', 'mean_length_px', 'total_length_px',
                    'edge_length_px', 'mean_length_um', 'total_length_um',
                    'edge_length_um', 'density']
    }

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('summary', color=PORT_COLORS['table'], multi_output=True)

    def evaluate(self):
        tbl = _read_port(self, 'table')
        if not isinstance(tbl, TableData):
            self.mark_error()
            return False, 'No table connected'

        df = tbl.df
        count = int(len(df))

        def _first(col):
            return float(df[col].iloc[0]) if count > 0 and col in df.columns else float('nan')

        if count > 0:
            mean_len_px = float(df['filopodia_length_px'].mean())
            total_len_px = float(df['filopodia_length_px'].sum())
            mean_len_um = float(df['filopodia_length_um'].mean())
            total_len_um = float(df['filopodia_length_um'].sum())
        else:
            mean_len_px = mean_len_um = float('nan')
            total_len_px = total_len_um = 0.0

        edge_px = _first('edge_length_px')
        edge_um = _first('edge_length_um')

        if edge_um == edge_um and edge_um > 0:        # not NaN and positive
            density = count / edge_um
        elif edge_px == edge_px and edge_px > 0:
            density = count / edge_px
        else:
            density = 0.0

        def _r(v, n=2):
            return round(v, n) if v == v else float('nan')   # keep NaN as NaN

        row = {
            'filopodia_count': count,
            'mean_length_px':  _r(mean_len_px),
            'total_length_px': _r(total_len_px),
            'edge_length_px':  _r(edge_px),
            'mean_length_um':  _r(mean_len_um),
            'total_length_um': _r(total_len_um),
            'edge_length_um':  _r(edge_um),
            'density':         round(density, 4),
        }
        self.output_values['summary'] = TableData(payload=pd.DataFrame([row]))
        self.mark_clean()
        return True, None
