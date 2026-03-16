#!/usr/bin/env python3
"""prepare_dude.py — Preprocess DUD-E dataset for docking/scoring benchmarks.

For each target in the DUD-E directory:
  1. Reads actives_final.ism and decoys_final.ism
  2. Validates every SMILES with RDKit (skips failures)
  3. Writes a combined .smi file:  ACTIVE_N <smiles>  /  INACTIVE_N <smiles>
  4. Extracts crystal ligand centroid + bounding box from crystal_ligand.mol2
  5. Writes all pocket definitions to pockets.txt

Usage:
    python prepare_dude.py <dude_dir> <output_dir> [--padding 4.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

# Suppress RDKit parse warnings (noisy on large datasets)
RDLogger.DisableLog("rdApp.*")


# ── SMILES parsing ───────────────────────────────────────────────────────────

def parse_ism(ism_path: Path) -> tuple[list[str], int]:
    """Parse a DUD-E .ism file, return (valid_smiles_list, n_skipped)."""
    valid: list[str] = []
    skipped = 0
    with open(ism_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            smi = parts[0]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                skipped += 1
                continue
            valid.append(Chem.MolToSmiles(mol))
    return valid, skipped


# ── Crystal ligand mol2 reader ───────────────────────────────────────────────

def read_mol2_coords(mol2_path: Path) -> np.ndarray:
    """Read all atom coordinates from a TRIPOS .mol2 file.

    Returns (N, 3) float64 array.
    """
    coords: list[list[float]] = []
    in_atoms = False
    with open(mol2_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if stripped.startswith("@<TRIPOS>") and in_atoms:
                break
            if in_atoms and stripped:
                parts = stripped.split()
                if len(parts) >= 5:
                    try:
                        coords.append([float(parts[2]),
                                       float(parts[3]),
                                       float(parts[4])])
                    except ValueError:
                        continue
    return np.array(coords, dtype=np.float64)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DUD-E dataset for docking/scoring benchmarks.")
    parser.add_argument("dude_dir", type=Path,
                        help="Path to DUD-E parent directory (contains target subdirs)")
    parser.add_argument("output_dir", type=Path,
                        help="Output directory for .smi files and pockets.txt")
    parser.add_argument("--padding", type=float, default=4.0,
                        help="Padding (Angstroms) added per side to ligand bbox (default: 4.0)")
    args = parser.parse_args()

    dude_dir: Path = args.dude_dir
    output_dir: Path = args.output_dir
    padding: float = args.padding

    if not dude_dir.is_dir():
        print(f"Error: {dude_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    targets = sorted(d for d in dude_dir.iterdir() if d.is_dir())
    pocket_lines: list[str] = []
    total_actives = 0
    total_decoys = 0

    for target_dir in targets:
        name = target_dir.name
        actives_ism = target_dir / "actives_final.ism"
        decoys_ism = target_dir / "decoys_final.ism"
        crystal_lig = target_dir / "crystal_ligand.mol2"

        # Check required files
        missing = []
        if not actives_ism.exists():
            missing.append("actives_final.ism")
        if not decoys_ism.exists():
            missing.append("decoys_final.ism")
        if not crystal_lig.exists():
            missing.append("crystal_ligand.mol2")
        if missing:
            print(f"  SKIP {name}: missing {', '.join(missing)}")
            continue

        # ── Parse and validate SMILES ────────────────────────────────────
        actives, a_skip = parse_ism(actives_ism)
        decoys, d_skip = parse_ism(decoys_ism)
        total_actives += len(actives)
        total_decoys += len(decoys)

        # ── Write combined .smi ──────────────────────────────────────────
        smi_path = output_dir / f"{name}.smi"
        with open(smi_path, "w") as f:
            for i, smi in enumerate(actives, 1):
                f.write(f"ACTIVE_{i} {smi}\n")
            for i, smi in enumerate(decoys, 1):
                f.write(f"INACTIVE_{i} {smi}\n")

        # ── Extract pocket from crystal ligand ───────────────────────────
        coords = read_mol2_coords(crystal_lig)
        if len(coords) == 0:
            print(f"  WARNING {name}: no coordinates in crystal_ligand.mol2")
            continue

        center = coords.mean(axis=0)
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        size = (bbox_max - bbox_min) + 2.0 * padding  # padding on each side

        pocket_lines.append(
            f"{name}\t"
            f"{center[0]:.3f}\t{center[1]:.3f}\t{center[2]:.3f}\t"
            f"{size[0]:.3f}\t{size[1]:.3f}\t{size[2]:.3f}"
        )

        skip_msg = ""
        if a_skip or d_skip:
            skip_msg = f"  (skipped: {a_skip} actives, {d_skip} decoys)"
        print(f"  {name:12s}  {len(actives):5d} actives  {len(decoys):6d} decoys{skip_msg}")

    # ── Write pocket definitions ─────────────────────────────────────────
    pocket_path = output_dir / "pockets.txt"
    with open(pocket_path, "w") as f:
        f.write("# target\tcenter_x\tcenter_y\tcenter_z\tsize_x\tsize_y\tsize_z\n")
        for line in pocket_lines:
            f.write(line + "\n")

    print(f"\nDone: {len(pocket_lines)} targets processed")
    print(f"  Total: {total_actives} actives, {total_decoys} decoys")
    print(f"  .smi files:  {output_dir}/")
    print(f"  Pocket defs: {pocket_path}")


if __name__ == "__main__":
    main()
