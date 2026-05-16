"""Per-file segmentation wrapper.

Wraps the vendored `_segment_3d_rs_v2.process_one` with the module-constant
monkey-patch pattern from the upstream code: temporarily set module globals
based on a params dict, run process_one, restore originals in finally.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional


@contextlib.contextmanager
def _silence_stdio():
    """Redirect stdout+stderr to /dev/null while process_one runs."""
    devnull = open(os.devnull, 'w', encoding='utf-8', errors='replace')
    try:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        devnull.close()


def segment_file(
    ims_path: Path,
    out_dir: Path,
    params: Optional[Dict] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Run segmentation on one IMS file with module-constant overrides."""
    from . import _segment_3d_rs_v2 as seg

    params = params or {}
    old: Dict[str, object] = {k: getattr(seg, k) for k in params if hasattr(seg, k)}
    try:
        for k, v in params.items():
            if hasattr(seg, k):
                setattr(seg, k, v)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with _silence_stdio():
            seg.process_one(str(ims_path), str(out_dir), status_cb=status_cb)
    finally:
        for k, v in old.items():
            setattr(seg, k, v)


def segment_batch(
    ims_paths: Iterable[Path],
    out_dir: Path,
    params: Optional[Dict] = None,
    skip_existing: bool = True,
    force: bool = False,
    progress_cb: Optional[Callable[[int, int, str, str], None]] = None,
    stop_cb: Optional[Callable[[], bool]] = None,
) -> list[Path]:
    """Serial batch loop.

    progress_cb(index, total, stem, msg) fires for each file with msg in
    {'skip', 'start', 'done', f'error: ...'}.
    stop_cb() returning True between files aborts the loop early.
    Returns the list of `_corrected.csv` paths actually written.
    """
    paths = list(ims_paths)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    completed: list[Path] = []

    for i, p in enumerate(paths):
        if stop_cb is not None and stop_cb():
            break

        stem = p.stem
        corr = out_dir / f'{stem}_corrected.csv'
        if skip_existing and not force and corr.exists():
            if progress_cb:
                progress_cb(i, len(paths), stem, 'skip')
            completed.append(corr)
            continue

        if progress_cb:
            progress_cb(i, len(paths), stem, 'start')
        try:
            segment_file(
                p, out_dir, params=params,
                status_cb=(
                    lambda msg, _i=i, _s=stem: progress_cb(_i, len(paths), _s, msg)
                    if progress_cb else None
                ),
            )
            if progress_cb:
                progress_cb(i, len(paths), stem, 'done')
            if corr.exists():
                completed.append(corr)
        except Exception as e:
            if progress_cb:
                progress_cb(i, len(paths), stem, f'error: {type(e).__name__}: {e}')

    return completed
