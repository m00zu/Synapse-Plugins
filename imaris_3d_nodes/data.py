"""Domain data types for the Imaris 3D plugin.

ImarisDatasetData is the wire payload that travels between nodes.  Composite
PNG pixels stay on disk (referenced by path); only small DataFrames and
metadata live in memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────
# Per-file entry
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class ImarisDatasetEntry:
    """One processed .ims file plus its outputs."""
    file_stem: str
    group: str
    csv_path: Path
    composite_path: Path
    per_cell_table: pd.DataFrame
    bg_row: pd.Series
    voxel_spacing_um: tuple[float, float, float] | None = None
    excluded_cells: set[int] = field(default_factory=set)

    @property
    def kept_table(self) -> pd.DataFrame:
        if not self.excluded_cells:
            return self.per_cell_table
        return self.per_cell_table[
            ~self.per_cell_table['cell'].astype(int).isin(self.excluded_cells)
        ]

    @property
    def n_cells(self) -> int:
        return len(self.per_cell_table)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded_cells)


# ─────────────────────────────────────────────────────────────────────────
# Top-level dataset (the wire payload)
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class ImarisDatasetData:
    """Top-level dataset object passed between nodes."""
    entries: list[ImarisDatasetEntry]
    output_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── lookups ──
    def by_stem(self, file_stem: str) -> ImarisDatasetEntry | None:
        for e in self.entries:
            if e.file_stem == file_stem:
                return e
        return None

    def iter_by_group(self) -> Iterator[tuple[str, list[ImarisDatasetEntry]]]:
        seen: dict[str, list[ImarisDatasetEntry]] = {}
        for e in self.entries:
            seen.setdefault(e.group, []).append(e)
        yield from seen.items()

    @property
    def groups(self) -> list[str]:
        seen: list[str] = []
        for e in self.entries:
            if e.group not in seen:
                seen.append(e.group)
        return seen

    # ── transforms used by downstream nodes ──
    def to_long_per_cell(self) -> pd.DataFrame:
        """Concat every entry's kept_table, adding group + file_stem cols."""
        frames = []
        for e in self.entries:
            t = e.kept_table.copy()
            t.insert(0, 'group', e.group)
            t.insert(1, 'file_stem', e.file_stem)
            frames.append(t)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def to_wide_blanked(self, threshold: int, step_um: int) -> pd.DataFrame:
        """One column per group; rows are blanked, excluded-filtered cell values."""
        col = f'pct_above_{threshold}_at_{step_um}um'
        by_group: dict[str, list[float]] = {}
        for e in self.entries:
            if col not in e.per_cell_table.columns:
                continue
            bg = float(e.bg_row.get(col, 0.0))
            kept = e.kept_table[col].astype(float) - bg
            by_group.setdefault(e.group, []).extend(kept.tolist())
        max_n = max((len(v) for v in by_group.values()), default=0)
        for g in by_group:
            by_group[g] = by_group[g] + [float('nan')] * (max_n - len(by_group[g]))
        return pd.DataFrame({k: by_group[k] for k in by_group})

    def total_cells(self) -> int:
        return sum(e.n_cells for e in self.entries)

    def total_excluded(self) -> int:
        return sum(e.n_excluded for e in self.entries)


# ─────────────────────────────────────────────────────────────────────────
# Port-type registration
# ─────────────────────────────────────────────────────────────────────────
PORT_TYPE_NAME = 'imaris_dataset'
IMARIS_DATASET_COLOR = (80, 180, 200)  # teal

try:
    from nodes.base import register_port_type, PORT_COLORS
    PORT_COLORS[PORT_TYPE_NAME] = IMARIS_DATASET_COLOR
    register_port_type(PORT_TYPE_NAME, ImarisDatasetData)
except ImportError:
    # Tests can import data.py without Synapse on the path.
    pass
