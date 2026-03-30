from pydantic import BaseModel, ConfigDict
from typing import Any, Optional, Union, Dict, List
import pandas as pd
from PIL import Image
import matplotlib.figure
# from rdkit.Chem.rdchem import Mol as RDMol

class NodeData(BaseModel):
    """Base class for all data passed between nodes."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    metadata: Dict[str, Any] = {}
    source_path: Optional[str] = None

    @classmethod
    def merge(cls, items: list) -> "NodeData":
        """Merge a list of same-typed NodeData objects into one. Override in subclasses."""
        raise NotImplementedError(f"{cls.__name__} does not support merging.")

class CollectionData(NodeData):
    """A named collection of NodeData items that flow through the pipeline as one.

    payload is a dict mapping user-defined names to NodeData instances.
    All items are typically the same type (e.g. all ImageData), but mixed
    types are allowed.

    The auto-loop in BaseExecutionNode transparently unpacks a collection,
    runs a single-item node on each entry, and repacks the results — so
    every existing node works with collections for free.
    """
    payload: Any   # dict[str, NodeData]

    @property
    def names(self) -> list[str]:
        return list(self.payload.keys())

    @property
    def inner_type(self) -> type:
        """The type of the first item (hint for port validation)."""
        first = next(iter(self.payload.values()), None)
        return type(first) if first else NodeData

    def __len__(self):
        return len(self.payload)

    def get(self, name: str):
        return self.payload.get(name)

    @classmethod
    def merge(cls, items: list):
        """Flatten collections: merge all items across iterations into one collection."""
        d = {}
        for item in items:
            meta = getattr(item, 'metadata', {}) or {}
            prefix = meta.get('batch_key') or meta.get('file') or ''
            for name, val in item.payload.items():
                key = f'{prefix}_{name}' if prefix else name
                base = key
                counter = 2
                while key in d:
                    key = f'{base}_{counter}'
                    counter += 1
                d[key] = val
        return cls(payload=d)

class TableData(NodeData):
    """Wraps a pandas DataFrame or Series."""
    payload: Union[pd.DataFrame, pd.Series]

    @property
    def df(self) -> pd.DataFrame:
        if isinstance(self.payload, pd.Series):
            return self.payload.to_frame()
        return self.payload

    @classmethod
    def merge(cls, items: list) -> "TableData":
        """Concatenate all DataFrames, injecting batch_key/frame/file from batch metadata."""
        dfs = []
        for item in items:
            df = item.df.copy()
            meta = getattr(item, 'metadata', {}) or {}
            # Inject batch context columns (don't overwrite existing)
            if 'frame' in meta and 'frame' not in df.columns:
                df.insert(0, 'frame', meta['frame'])
            if 'file' in meta and 'file' not in df.columns:
                col_pos = 1 if 'frame' in df.columns else 0
                df.insert(col_pos, 'file', meta['file'])
            elif 'batch_key' in meta and 'file' not in df.columns:
                col_pos = 1 if 'frame' in df.columns else 0
                df.insert(col_pos, 'file', meta['batch_key'])
            dfs.append(df)
        return cls(payload=pd.concat(dfs, ignore_index=True))

class StatData(TableData):
    """Wraps a pandas DataFrame with statistics."""
    pass


class ModelData(NodeData):
    """Wraps a fitted model for prediction.

    payload: a callable ``predict(X_df) -> array`` or an object with a
             ``.predict()`` method (e.g. statsmodels result, sklearn estimator).

    Attributes stored in metadata:
        model_type (str): e.g. 'linear', 'nonlinear', 'sklearn'
        x_columns (list[str]): feature column names the model expects
        y_column  (str): target column name
        degree    (int): polynomial degree (for linear regression)
        model_name (str): human-readable name (e.g. '4PL (EC50)')
    """
    payload: Any  # fitted model object


class ImageData(NodeData):
    """Wraps an image as a numpy array with bit-depth and display metadata.

    payload: numpy ndarray — shape (H, W) for grayscale, (H, W, 3) for RGB.
             dtype is preserved (uint8, uint16, float32, etc.)
    bit_depth: original source bit depth (8, 12, 14, 16).
    scale_um: micrometers per pixel (None if unknown).
    display_min: display window lower bound (None = auto from data).
    display_max: display window upper bound (None = auto from data).
    """
    payload: Any   # np.ndarray
    bit_depth: int = 8
    scale_um: Optional[float] = None
    display_min: Optional[float] = None
    display_max: Optional[float] = None

    @property
    def image(self) -> Image.Image:
        """Backward-compat: return an 8-bit PIL Image for legacy nodes."""
        return array_to_pil(self.payload, self.bit_depth,
                            self.display_min, self.display_max)

    @property
    def array(self):
        """The raw numpy array at original bit depth."""
        return self.payload

    @property
    def is_grayscale(self) -> bool:
        return self.payload.ndim == 2

    @property
    def dtype_max(self) -> float:
        """Maximum possible value for the source bit depth."""
        return float((1 << self.bit_depth) - 1) if self.bit_depth <= 16 else 255.0

    @classmethod
    def merge(cls, items: list):
        """Merge images into a CollectionData keyed by batch_key/file/frame."""
        d = {}
        for i, item in enumerate(items):
            meta = getattr(item, 'metadata', {}) or {}
            key = meta.get('batch_key') or meta.get('file') or meta.get('frame') or f'item_{i}'
            # Dedup
            base = key
            counter = 2
            while key in d:
                key = f'{base}_{counter}'
                counter += 1
            d[key] = item
        return CollectionData(payload=d)


def array_to_pil(arr, bit_depth=8, display_min=None, display_max=None):
    """Convert a numpy array of any depth/dtype to an 8-bit PIL Image for display.

    Handles: uint8, uint16, float32 [0,1], float64 [0,1], and arbitrary ranges.
    """
    import numpy as np

    # Fast path: already uint8, no windowing
    if arr.dtype == np.uint8 and display_min is None and display_max is None:
        if arr.ndim == 2:
            return Image.fromarray(arr, mode='L')
        return Image.fromarray(arr, mode='RGB')

    arr_f = arr.astype(np.float32)

    # Determine display range
    if display_min is not None and display_max is not None:
        lo, hi = float(display_min), float(display_max)
    elif arr.dtype in (np.float32, np.float64) and arr_f.max() <= 1.0 and arr_f.min() >= 0.0:
        # Float [0, 1] convention
        lo, hi = 0.0, 1.0
    elif bit_depth and bit_depth > 8:
        lo, hi = 0.0, float((1 << bit_depth) - 1)
    else:
        lo, hi = float(arr_f.min()), float(arr_f.max())

    if hi <= lo:
        hi = lo + 1.0
    display = np.clip((arr_f - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    if display.ndim == 2:
        return Image.fromarray(display, mode='L')
    return Image.fromarray(display, mode='RGB')


def array_to_qpixmap(arr, bit_depth=8, display_min=None, display_max=None):
    """Convert a numpy array of any depth/dtype to a QPixmap for Qt display."""
    import numpy as np
    from PySide6.QtGui import QImage, QPixmap

    # Normalize to 8-bit for display
    if arr.dtype != np.uint8 or display_min is not None:
        arr_f = arr.astype(np.float32)
        if display_min is not None and display_max is not None:
            lo, hi = float(display_min), float(display_max)
        elif arr.dtype in (np.float32, np.float64) and arr_f.max() <= 1.0 and arr_f.min() >= 0.0:
            lo, hi = 0.0, 1.0
        else:
            if bit_depth and bit_depth > 8:
                lo, hi = 0.0, float((1 << bit_depth) - 1)
            else:
                lo, hi = float(arr_f.min()), float(arr_f.max())
            if hi <= lo:
                hi = lo + 1.0
        display = np.clip((arr_f - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    else:
        display = arr

    display = np.ascontiguousarray(display)
    if display.ndim == 2:
        h, w = display.shape
        qimg = QImage(display.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w, c = display.shape
        if c == 4:
            qimg = QImage(display.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        else:
            qimg = QImage(display.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

class MaskData(ImageData):
    """Wraps a boolean/binary PIL Image representing a mask."""
    pass

class SkeletonData(MaskData):
    """
    A thinned 1-pixel-wide skeleton mask produced by SkeletonizeNode.
    Subclass of MaskData so it can feed any node that accepts a mask,
    but typed distinctly so SkeletonAnalysisNode can require skeleton input.
    """
    pass

class LabelData(NodeData):
    """
    Integer label array where each connected region has a unique positive integer
    value (0 = background).  payload is a numpy int32 ndarray of shape (H, W).
    image (optional) is a pre-generated RGB PIL Image for display purposes,
    produced by the source node so downstream nodes do not need to recompute it.
    """
    payload: Any   # np.ndarray dtype int32, shape (H, W)
    image:   Any = None  # PIL.Image RGB colored visualization

    @classmethod
    def merge(cls, items: list):
        """Merge label arrays into a CollectionData keyed by batch_key/file/frame."""
        d = {}
        for i, item in enumerate(items):
            meta = getattr(item, 'metadata', {}) or {}
            key = meta.get('batch_key') or meta.get('file') or meta.get('frame') or f'item_{i}'
            base = key
            counter = 2
            while key in d:
                key = f'{base}_{counter}'
                counter += 1
            d[key] = item
        return CollectionData(payload=d)

class FigureData(NodeData):
    """Wraps a matplotlib Figure."""
    payload: Optional[matplotlib.figure.Figure]
    svg_override: Optional[bytes] = None  # edited SVG from SvgEditorNode

    @property
    def fig(self) -> Optional[matplotlib.figure.Figure]:
        return self.payload

    @classmethod
    def merge(cls, items: list):
        """Merge figures into a CollectionData keyed by batch_key/file/frame."""
        d = {}
        for i, item in enumerate(items):
            if item.payload is None:
                continue
            meta = getattr(item, 'metadata', {}) or {}
            key = meta.get('batch_key') or meta.get('file') or meta.get('frame') or f'item_{i}'
            base = key
            counter = 2
            while key in d:
                key = f'{base}_{counter}'
                counter += 1
            d[key] = item
        return CollectionData(payload=d)

class ConfocalDatasetData(NodeData):
    """Wraps the specialized confocal dataset dictionary."""
    payload: Dict[str, Any]

    @classmethod
    def merge(cls, items: list):
        """Merge confocal datasets into a CollectionData keyed by batch_key/file/frame."""
        d = {}
        for i, item in enumerate(items):
            meta = getattr(item, 'metadata', {}) or {}
            key = meta.get('batch_key') or meta.get('file') or meta.get('frame') or f'item_{i}'
            base = key
            counter = 2
            while key in d:
                key = f'{base}_{counter}'
                counter += 1
            d[key] = item
        return CollectionData(payload=d)



# class RDMolData(NodeData):
#     """Wraps the specialized RDMol dictionary."""
#     payload: RDMol

#     @classmethod
#     def merge(cls, items: list) -> list:
#         """Collect RDMol into a list."""
#         return [i.payload for i in items]
