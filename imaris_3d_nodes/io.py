"""CSV loading and directory scanning helpers.

Adapted from `Imaris_process/app/pipeline/io.py` but with the BG row split out
and returned in a structured ImarisDatasetEntry.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .data import ImarisDatasetEntry

PCT_RE = re.compile(r'^pct_above_(\d+)_at_(\d+)um$')


def load_entry_from_csv(csv_path: Path, group: str) -> ImarisDatasetEntry | None:
    """Load a `{stem}_corrected.csv` file and return an ImarisDatasetEntry.

    BG row (where ``cell == "BG"``) is split out into entry.bg_row.
    Returns None if the file is unparseable, has no `cell` column, or has no BG row.
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if 'cell' not in df.columns:
        return None

    is_bg = df['cell'].astype(str) == 'BG'
    bg = df[is_bg]
    cells = df[~is_bg]
    if cells.empty or bg.empty:
        return None

    # File stem: strip the _corrected suffix
    stem = csv_path.stem
    if stem.endswith('_corrected'):
        stem = stem[: -len('_corrected')]

    composite_path = csv_path.parent / f'{stem}_composite.png'

    return ImarisDatasetEntry(
        file_stem=stem,
        group=group,
        csv_path=csv_path,
        composite_path=composite_path,
        per_cell_table=cells.reset_index(drop=True),
        bg_row=bg.iloc[0],
    )


def detect_thresholds_and_steps(csv_path: Path) -> tuple[list[int], list[int]]:
    """Inspect one CSV header to find all (threshold, step_um) combos present."""
    df = pd.read_csv(csv_path, nrows=0)
    thresholds: set[int] = set()
    steps: set[int] = set()
    for col in df.columns:
        m = PCT_RE.match(col)
        if m:
            thresholds.add(int(m.group(1)))
            steps.add(int(m.group(2)))
    return sorted(thresholds), sorted(steps)


def scan_dataset_dir(root: Path, *, default_group: str | None = None) -> list[tuple[Path, str]]:
    """Walk ``root`` and yield ``(csv_path, group_name)`` pairs.

    Auto-detect layout:
      - If any direct subdirectory contains *_corrected.csv files, treat each
        subdir as a group (group name = subdir basename).
      - Otherwise, scan ``root`` directly and assign every file to
        ``default_group`` (or ``root.name`` if not given).
    """
    root = Path(root)
    pairs: list[tuple[Path, str]] = []

    has_subdirs_with_csvs = any(
        child.is_dir() and any(child.glob('*_corrected.csv'))
        for child in root.iterdir() if child.exists()
    )

    if has_subdirs_with_csvs:
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for csv in sorted(sub.glob('*_corrected.csv')):
                pairs.append((csv, sub.name))
    else:
        group = default_group or root.name
        for csv in sorted(root.glob('*_corrected.csv')):
            pairs.append((csv, group))

    return pairs


def list_ims_files(folder: Path) -> list[Path]:
    """All `.ims` files directly inside the folder (non-recursive)."""
    return sorted(Path(folder).glob('*.ims'))
