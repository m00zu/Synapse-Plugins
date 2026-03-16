"""
setup_vendor.py — Populate plugins/rdkit_nodes/vendor/ from an RDKit wheel
===========================================================================

Usage (provide the wheel you already downloaded):
    python plugins/rdkit_nodes/setup_vendor.py /path/to/rdkit-2025.9.5-cp314-cp314-macosx_11_0_arm64.whl

The script extracts the wheel into:
    plugins/rdkit_nodes/vendor/rdkit/          ← the importable package
    plugins/rdkit_nodes/vendor/rdkit.libs/     ← bundled C++ dylibs (macOS/Linux)
    plugins/rdkit_nodes/vendor/rdkit-*.dist-info/  ← metadata (harmless)

After running, the rdkit_nodes plugin will import RDKit from vendor/ without
requiring any system-level installation or Nuitka build changes.

Packaging the plugin for distribution
--------------------------------------
To share the plugin with another machine of the same OS/Python version, zip
the entire rdkit_nodes/ directory (vendor/ included):

    cd plugins
    zip -r rdkit_nodes_macos_arm64_cp314.zip rdkit_nodes/

The recipient installs it via Plugin Manager → Install Plugin → pick the .zip.

Platform notes
--------------
A wheel is OS- and Python-version specific.  For each platform you distribute
to you need its own wheel:

  macOS ARM (Apple Silicon):  rdkit-*-cp314-cp314-macosx_11_0_arm64.whl
  macOS Intel:                rdkit-*-cp314-cp314-macosx_10_9_x86_64.whl
  Windows x64:                rdkit-*-cp314-cp314-win_amd64.whl
  Linux x86_64:               rdkit-*-cp314-cp314-manylinux_2_28_x86_64.whl

You can download wheels for all platforms at once with pip:
    pip download rdkit --only-binary=:all: --platform macosx_11_0_arm64 \\
        --python-version 314 -d /tmp/rdkit_wheels/
"""

import shutil
import sys
import zipfile
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    wheel_path = Path(sys.argv[1]).expanduser().resolve()
    if not wheel_path.exists():
        print(f"ERROR: wheel not found: {wheel_path}")
        sys.exit(1)
    if wheel_path.suffix.lower() != '.whl':
        print(f"ERROR: expected a .whl file, got: {wheel_path.name}")
        sys.exit(1)

    plugin_root = Path(__file__).parent
    vendor_dir  = plugin_root / 'vendor'

    # Remove stale rdkit content (keep other vendored packages untouched)
    for stale in vendor_dir.glob('rdkit*'):
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink()

    vendor_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {wheel_path.name} -> {vendor_dir}/")
    with zipfile.ZipFile(wheel_path) as zf:
        members = zf.namelist()
        total   = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, vendor_dir)
            if i % 200 == 0 or i == total:
                pct = i * 100 // total
                print(f"  {pct}%  ({i}/{total} files)", end='\r')
    print()

    # Sanity check
    rdkit_pkg = vendor_dir / 'rdkit'
    if not rdkit_pkg.is_dir():
        print(
            "WARNING: 'rdkit' directory not found in vendor/ after extraction.\n"
            "         The wheel may have an unexpected layout.  Contents:\n"
            + '\n'.join(f"  {p.name}" for p in sorted(vendor_dir.iterdir()))
        )
        sys.exit(1)

    print(f"Done. RDKit extracted to: {rdkit_pkg}")
    print()
    print("To verify, run:")
    print(f"  python -c \"import sys; sys.path.insert(0, '{vendor_dir}'); from rdkit import Chem; print(Chem.__file__)\"")
    print()
    print("To package for distribution:")
    print(f"  cd {plugin_root.parent}")
    print(f"  zip -r rdkit_nodes_{wheel_path.stem.split('-')[2]}.zip rdkit_nodes/")


if __name__ == '__main__':
    main()
