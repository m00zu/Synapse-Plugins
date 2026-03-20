#!/usr/bin/env python3
"""
package_plugin.py — Package any Synapse plugin as a .synpkg archive.

A .synpkg file is a zstandard-compressed tar archive containing a plugin
directory ready to be dropped into Synapse's plugins/ folder.

Usage:
    python package_plugin.py plugins/rdkit_nodes              # full
    python package_plugin.py plugins/rdkit_nodes --no-vendor   # skip vendor/
    python package_plugin.py plugins/rdkit_nodes --slim        # source only
    python package_plugin.py plugins/rdkit_nodes -o ~/Desktop  # custom output dir

Install a .synpkg:
    python package_plugin.py --install my_plugin.synpkg
    python package_plugin.py --install my_plugin.synpkg --dest ~/custom/plugins
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

EXTENSION = '.synpkg'

# ── Exclusion rules ──────────────────────────────────────────────────────────

EXCLUDE_NAMES = {
    '__pycache__', '.DS_Store', '.mypy_cache', '.pytest_cache',
    '.git', '.gitignore', '.github', '.vscode',
    'package_plugin.py', 'package_plugin.sh', 'setup_vendor.py',
}
EXCLUDE_SUFFIXES = ('.pyc', '.pyo', '.egg-info', '.dist-info')
EXCLUDE_HIDDEN = True  # skip dotfiles/dotdirs


def should_exclude(
    rel_path: str,
    *,
    no_vendor: bool,
    slim: bool,
) -> bool:
    """Return True if this path should be skipped."""
    parts = rel_path.split('/')
    basename = parts[-1]

    # Hidden files/dirs (but keep .dylibs — needed for macOS RDKit shared libs)
    if EXCLUDE_HIDDEN and basename.startswith('.') and basename not in ('.', '.dylibs'):
        return True

    if basename in EXCLUDE_NAMES:
        return True

    if any(basename.endswith(s) for s in EXCLUDE_SUFFIXES):
        return True
    if any(p.endswith(s) for p in parts for s in EXCLUDE_SUFFIXES):
        return True

    # Relative path within the plugin (parts[0] is plugin dir name)
    rel = parts[1] if len(parts) > 1 else ''

    if (slim or no_vendor) and rel == 'vendor':
        return True

    if slim and rel == 'data':
        return True

    return False


# ── Pack ─────────────────────────────────────────────────────────────────────

def pack(
    plugin_dir: str,
    *,
    no_vendor: bool = False,
    slim: bool = False,
    output_dir: str | None = None,
    level: int = 19,
) -> str:
    """Create a .synpkg archive from a plugin directory.

    Returns the path to the created archive.
    """
    plugin_dir = os.path.abspath(plugin_dir)
    if not os.path.isdir(plugin_dir):
        sys.exit(f'Error: {plugin_dir!r} is not a directory')

    plugin_name = os.path.basename(plugin_dir)
    parent_dir = os.path.dirname(plugin_dir)

    # Build output filename (no date — same name on every build so releases replace cleanly)
    if slim:
        tag = f'{plugin_name}-slim'
    elif no_vendor:
        tag = f'{plugin_name}-novendor'
    else:
        arch = platform.machine()
        osname = platform.system().lower()
        pyver = f'cp{sys.version_info.major}{sys.version_info.minor}'
        tag = f'{plugin_name}-{osname}-{arch}-{pyver}'

    if output_dir is None:
        output_dir = parent_dir
    archive_path = os.path.join(output_dir, f'{tag}{EXTENSION}')

    # Build tar in memory
    tar_buf = io.BytesIO()
    n_files = 0
    with tarfile.open(fileobj=tar_buf, mode='w') as tar:
        for root, dirs, files in os.walk(plugin_dir):
            rel_root = os.path.relpath(root, parent_dir)

            # Prune directories in-place
            dirs[:] = sorted(
                d for d in dirs
                if not should_exclude(
                    f'{rel_root}/{d}', no_vendor=no_vendor, slim=slim
                )
            )

            for fname in sorted(files):
                member = f'{rel_root}/{fname}'
                if should_exclude(member, no_vendor=no_vendor, slim=slim):
                    continue
                tar.add(os.path.join(root, fname), arcname=member)
                n_files += 1

    tar_bytes = tar_buf.getvalue()

    # Compress with zstd
    cctx = zstd.ZstdCompressor(level=level, threads=-1)
    compressed = cctx.compress(tar_bytes)

    with open(archive_path, 'wb') as f:
        f.write(compressed)

    tar_mb = len(tar_bytes) / 1024 / 1024
    zst_mb = len(compressed) / 1024 / 1024
    ratio = (1 - len(compressed) / len(tar_bytes)) * 100 if tar_bytes else 0

    mode = 'slim' if slim else ('no-vendor' if no_vendor else 'full')
    print(f'Packaged {n_files} files ({mode} mode)')
    print(f'  tar:  {tar_mb:.1f} MB')
    print(f'  zst:  {zst_mb:.1f} MB  ({ratio:.0f}% compression)')
    print(f'  out:  {archive_path}')
    return archive_path


# ── Unpack / Install ─────────────────────────────────────────────────────────

def install(archive_path: str, dest: str = 'plugins') -> str:
    """Extract a .synpkg archive into the destination directory.

    Returns the path to the extracted plugin directory.
    """
    archive_path = os.path.abspath(archive_path)
    if not os.path.isfile(archive_path):
        sys.exit(f'Error: {archive_path!r} not found')

    with open(archive_path, 'rb') as f:
        raw = zstd.ZstdDecompressor().decompress(
            f.read(), max_output_size=500_000_000
        )

    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        names = tar.getnames()
        top = names[0].split('/')[0] if names else '??'
        tar.extractall(dest)

    plugin_path = os.path.join(dest, top)
    n = len(names)
    print(f'Installed {n} files → {plugin_path}')
    return plugin_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Package or install Synapse plugins (.synpkg)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python package_plugin.py plugins/rdkit_nodes\n'
            '  python package_plugin.py plugins/my_plugin --slim\n'
            '  python package_plugin.py --install my_plugin.synpkg\n'
        ),
    )

    # Mutual exclusion: pack vs install
    group = parser.add_mutually_exclusive_group()
    group.add_argument('plugin_dir', nargs='?',
                       help='Plugin directory to package')
    group.add_argument('--install', metavar='FILE',
                       help=f'Install a {EXTENSION} archive')

    parser.add_argument('--no-vendor', action='store_true',
                        help='Exclude vendor/ directory')
    parser.add_argument('--slim', action='store_true',
                        help='Source only (no vendor/, no data/)')
    parser.add_argument('-o', '--output', metavar='DIR',
                        help='Output directory (default: parent of plugin dir)')
    parser.add_argument('--dest', default='plugins',
                        help='Install destination directory (default: plugins/)')
    parser.add_argument('--level', type=int, default=19,
                        help='Zstandard compression level 1-22 (default: 19)')

    args = parser.parse_args()

    if args.install:
        install(args.install, dest=args.dest)
    elif args.plugin_dir:
        pack(
            args.plugin_dir,
            no_vendor=args.no_vendor,
            slim=args.slim,
            output_dir=args.output,
            level=args.level,
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
