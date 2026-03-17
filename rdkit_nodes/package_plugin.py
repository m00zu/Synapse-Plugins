#!/usr/bin/env python3
"""
package_plugin.py — Package the rdkit_nodes plugin as a .tar.zst archive.

Usage:
    python package_plugin.py              # full (source + vendor + data)
    python package_plugin.py --no-vendor  # source + data (no compiled libs)
    python package_plugin.py --slim       # source only (no vendor, no GNINA models)

Output: ../rdkit_nodes-<platform>-<date>.tar.zst

Install:
    python -c "
    import tarfile, zstandard, io, sys
    with open(sys.argv[1], 'rb') as f:
        raw = zstandard.ZstdDecompressor().decompress(f.read(), max_output_size=500_000_000)
    tarfile.open(fileobj=io.BytesIO(raw)).extractall('plugins/')
    " rdkit_nodes-*.tar.zst
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import sys
import tarfile
import time

import zstandard as zstd

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_NAME = 'rdkit_nodes'

EXCLUDE_ALWAYS = {
    '__pycache__', '.DS_Store', '.mypy_cache', '.pytest_cache',
    'package_plugin.py', 'package_plugin.sh', 'setup_vendor.py',
}
EXCLUDE_SUFFIXES = ('.pyc', '.pyo', '.egg-info')
EXCLUDE_VENDOR_DIRS = {'rdkit-stubs'}  # type stubs, not needed at runtime
EXCLUDE_DIST_INFO_SUFFIX = '.dist-info'


def should_exclude(name: str, *, no_vendor: bool, slim: bool) -> bool:
    """Return True if this archive member should be skipped."""
    parts = name.split('/')
    basename = parts[-1]

    if basename in EXCLUDE_ALWAYS:
        return True
    if any(basename.endswith(s) for s in EXCLUDE_SUFFIXES):
        return True

    # Relative parts after rdkit_nodes/
    rel = parts[1] if len(parts) > 1 else ''

    if slim:
        if rel == 'vendor':
            return True
        if rel == 'data' and len(parts) > 2 and parts[2] == 'gnina_models':
            return True

    if no_vendor and rel == 'vendor':
        return True

    # For full mode: skip dist-info and stubs
    if not no_vendor and not slim:
        for p in parts:
            if p.endswith(EXCLUDE_DIST_INFO_SUFFIX):
                return True
            if p in EXCLUDE_VENDOR_DIRS:
                return True

    return False


def build_archive(no_vendor: bool = False, slim: bool = False) -> str:
    date = time.strftime('%Y%m%d')
    arch = platform.machine()
    osname = platform.system().lower()
    pyver = f'cp{sys.version_info.major}{sys.version_info.minor}'

    if slim:
        tag = f'{PLUGIN_NAME}-slim-{date}'
    elif no_vendor:
        tag = f'{PLUGIN_NAME}-novendor-{date}'
    else:
        tag = f'{PLUGIN_NAME}-{osname}-{arch}-{pyver}-{date}'

    out_dir = os.path.dirname(PLUGIN_DIR)
    archive_path = os.path.join(out_dir, f'{tag}.tar.zst')

    # Build tar in memory, then compress with zstd
    tar_buf = io.BytesIO()
    n_files = 0
    with tarfile.open(fileobj=tar_buf, mode='w') as tar:
        for root, dirs, files in os.walk(PLUGIN_DIR):
            # Build archive path relative to parent of plugin dir
            rel_root = os.path.relpath(root, os.path.dirname(PLUGIN_DIR))

            # Prune excluded directories in-place
            dirs[:] = [
                d for d in dirs
                if not should_exclude(f'{rel_root}/{d}', no_vendor=no_vendor, slim=slim)
            ]

            for fname in sorted(files):
                member_name = f'{rel_root}/{fname}'
                if should_exclude(member_name, no_vendor=no_vendor, slim=slim):
                    continue
                full_path = os.path.join(root, fname)
                tar.add(full_path, arcname=member_name)
                n_files += 1

    tar_bytes = tar_buf.getvalue()

    # Compress with zstd level 18 (near-max compression)
    cctx = zstd.ZstdCompressor(level=18, threads=-1)
    compressed = cctx.compress(tar_bytes)

    with open(archive_path, 'wb') as f:
        f.write(compressed)

    tar_mb = len(tar_bytes) / 1024 / 1024
    zst_mb = len(compressed) / 1024 / 1024
    ratio = (1 - len(compressed) / len(tar_bytes)) * 100

    mode = 'slim' if slim else ('no-vendor' if no_vendor else 'full')
    print(f'Packaged {n_files} files ({mode} mode)')
    print(f'  tar:  {tar_mb:.1f} MB')
    print(f'  zst:  {zst_mb:.1f} MB  ({ratio:.0f}% compression)')
    print(f'  out:  {archive_path}')
    return archive_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Package rdkit_nodes plugin')
    parser.add_argument('--no-vendor', action='store_true',
                        help='Exclude vendor/ (compiled libs)')
    parser.add_argument('--slim', action='store_true',
                        help='Source only (no vendor, no GNINA models)')
    args = parser.parse_args()
    build_archive(no_vendor=args.no_vendor, slim=args.slim)
