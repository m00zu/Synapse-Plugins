#!/usr/bin/env python3
"""mp_bench.py — Compare sequential vs multiprocessing for docking & GNINA scoring.

Uses a single DUD-E target (cdk2 by default) with ~20 molecules.
Benchmarks:
  - Docking:  sequential vs ProcessPoolExecutor(2) vs ProcessPoolExecutor(4)
  - GNINA:    sequential vs ProcessPoolExecutor(2) vs ProcessPoolExecutor(4)

Usage:
    python -m plugins.rdkit_nodes.mp_bench [--target cdk2] [--n_mols 20]
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ── Ensure project root is importable ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from plugins.rdkit_nodes.protein_utils import fix_and_convert
from plugins.rdkit_nodes.docking_backend import VinaRustBackend
from plugins.rdkit_nodes.meeko_ported import MoleculePreparation, PDBQTWriterLegacy
from plugins.rdkit_nodes.gnina_scorer import (
    GNINAModel,
    parse_pdbqt_string,
    parse_pdbqt_poses,
    process_molecule,
)

PREPARED_DIR = Path("/Users/s/Desktop/demo/dude_prepared")


# ── Helpers ──────────────────────────────────────────────────────────────────

def mol_to_pdbqt(mol) -> str:
    """RDKit Mol → PDBQT string via Meeko."""
    prep = MoleculePreparation(rigid_macrocycles=False)
    mol_setups = prep.prepare(mol)
    for setup in mol_setups:
        pdbqt_str, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
        if is_ok:
            return pdbqt_str
    raise RuntimeError(f"Meeko PDBQT conversion failed: {err_msg}")


def load_molecules(target: str, n_mols: int) -> list:
    """Load up to n_mols molecules from target's actives SDF."""
    sdf_path = PREPARED_DIR / target / "actives_final.sdf.gz"
    mols = []
    with gzip.open(sdf_path) as f:
        suppl = Chem.ForwardSDMolSupplier(f, removeHs=False)
        for mol in suppl:
            if mol is None:
                continue
            # Ensure explicit H with coords
            if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
                mol = Chem.AddHs(mol, addCoords=True)
            mols.append(mol)
            if len(mols) >= n_mols:
                break
    return mols


def prepare_receptor(target: str) -> str:
    """Prepare receptor PDB → PDBQT string."""
    pdb_path = PREPARED_DIR / target / "receptor.pdb"
    result = fix_and_convert(str(pdb_path))
    if isinstance(result, tuple):
        raise RuntimeError(f"Receptor prep failed: {result[0]}")
    return result


def read_pocket(target: str):
    """Read pocket center + size from pockets.txt."""
    pocket_file = PREPARED_DIR / "pockets.txt"
    with open(pocket_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if parts[0] == target:
                cx, cy, cz = float(parts[1]), float(parts[2]), float(parts[3])
                sx, sy, sz = float(parts[4]), float(parts[5]), float(parts[6])
                return (cx, cy, cz), (sx, sy, sz)
    raise ValueError(f"Target {target} not found in pockets.txt")


# ── Docking task (for multiprocessing) ───────────────────────────────────────

def _dock_one(args):
    """Dock a single molecule. Used as ProcessPoolExecutor task."""
    rec_pdbqt, lig_pdbqt, center, size, exhaustiveness = args
    backend = VinaRustBackend("QVina2")
    # Suppress Rust stderr banner
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        poses_pdbqt, energies = backend.dock(
            rec_pdbqt, lig_pdbqt, center, size,
            exhaustiveness=exhaustiveness, n_poses=9, seed=42,
        )
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
    return poses_pdbqt, energies


# ── GNINA scoring tasks (for multiprocessing) ───────────────────────────────

# Naive approach: model loaded per task (high overhead)
def _score_one_naive(args):
    """Score one docked molecule's poses with GNINA. Loads model each time."""
    rec_coords, rec_types, poses_pdbqt = args
    model = GNINAModel(ensemble="default")
    poses = parse_pdbqt_poses(poses_pdbqt)
    if not poses:
        return None
    center = np.mean(poses[0][0], axis=0)
    results = model.score_poses(rec_coords, rec_types, poses, center=center)
    best = max(results, key=lambda r: r["CNNscore"])
    return best["CNNscore"], best["CNNaffinity"]


# Optimized approach: model loaded once per worker via initializer
_worker_model = None
_worker_rec = None


def _gnina_worker_init(ensemble, rec_coords, rec_types):
    """Initializer for pool workers — loads model + receptor once."""
    global _worker_model, _worker_rec
    _worker_model = GNINAModel(ensemble=ensemble)
    _worker_rec = (rec_coords, rec_types)


def _score_one_optimized(poses_pdbqt):
    """Score one molecule using pre-loaded model in worker process."""
    global _worker_model, _worker_rec
    rec_coords, rec_types = _worker_rec
    poses = parse_pdbqt_poses(poses_pdbqt)
    if not poses:
        return None
    center = np.mean(poses[0][0], axis=0)
    results = _worker_model.score_poses(rec_coords, rec_types, poses, center=center)
    best = max(results, key=lambda r: r["CNNscore"])
    return best["CNNscore"], best["CNNaffinity"]


# ── Benchmark runners ────────────────────────────────────────────────────────

def bench_docking_sequential(rec_pdbqt, lig_pdbqts, center, size, exhaustiveness):
    """Dock all molecules sequentially."""
    results = []
    for pdbqt in lig_pdbqts:
        r = _dock_one((rec_pdbqt, pdbqt, center, size, exhaustiveness))
        results.append(r)
    return results


def bench_docking_parallel(rec_pdbqt, lig_pdbqts, center, size, workers, exhaustiveness):
    """Dock all molecules with ProcessPoolExecutor."""
    tasks = [(rec_pdbqt, pdbqt, center, size, exhaustiveness) for pdbqt in lig_pdbqts]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_dock_one, tasks):
            results.append(r)
    return results


def bench_docking_batch(rec_pdbqt, lig_pdbqts, center, size, exhaustiveness):
    """Dock all molecules with Rust batch_dock (grid maps computed once, outer Rayon)."""
    backend = VinaRustBackend("QVina2")
    results = backend.batch_dock(
        rec_pdbqt, lig_pdbqts, center, size,
        exhaustiveness=exhaustiveness, n_poses=9, seed=42,
    )
    return results


def bench_gnina_sequential(rec_coords, rec_types, docked_pdbqts):
    """Score all docked molecules sequentially."""
    model = GNINAModel(ensemble="default")
    results = []
    for pdbqt in docked_pdbqts:
        poses = parse_pdbqt_poses(pdbqt)
        if not poses:
            results.append(None)
            continue
        center = np.mean(poses[0][0], axis=0)
        res = model.score_poses(rec_coords, rec_types, poses, center=center)
        best = max(res, key=lambda r: r["CNNscore"])
        results.append((best["CNNscore"], best["CNNaffinity"]))
    return results


def bench_gnina_parallel_naive(rec_coords, rec_types, docked_pdbqts, workers):
    """Score with ProcessPoolExecutor — model re-loaded per task (naive)."""
    tasks = [(rec_coords, rec_types, pdbqt) for pdbqt in docked_pdbqts]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_score_one_naive, tasks):
            results.append(r)
    return results


def bench_gnina_parallel_opt(rec_coords, rec_types, docked_pdbqts, workers):
    """Score with ProcessPoolExecutor — model loaded once per worker."""
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_gnina_worker_init,
        initargs=("default", rec_coords, rec_types),
    ) as pool:
        results = list(pool.map(_score_one_optimized, docked_pdbqts))
    return results


def bench_gnina_threaded(rec_coords, rec_types, docked_pdbqts, workers):
    """Score with ThreadPoolExecutor — shared model, no IPC overhead."""
    model = GNINAModel(ensemble="default")

    def _score_thread(poses_pdbqt):
        poses = parse_pdbqt_poses(poses_pdbqt)
        if not poses:
            return None
        center = np.mean(poses[0][0], axis=0)
        results = model.score_poses(rec_coords, rec_types, poses, center=center)
        best = max(results, key=lambda r: r["CNNscore"])
        return best["CNNscore"], best["CNNaffinity"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_score_thread, docked_pdbqts))
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multiprocessing benchmark")
    parser.add_argument("--target", default="cdk2", help="DUD-E target name")
    parser.add_argument("--n_mols", type=int, default=20, help="Number of molecules")
    args = parser.parse_args()

    target = args.target
    n_mols = args.n_mols
    cpu_count = os.cpu_count() or 4

    print(f"Target: {target}, Molecules: {n_mols}, CPU cores: {cpu_count}")
    print("=" * 70)

    # ── Prepare receptor ──────────────────────────────────────────────────
    print("Preparing receptor...")
    t0 = time.perf_counter()
    rec_pdbqt = prepare_receptor(target)
    print(f"  Receptor ready ({time.perf_counter() - t0:.1f}s)")

    center, size = read_pocket(target)
    print(f"  Pocket: center={center}, size={size}")

    # ── Load & convert molecules ──────────────────────────────────────────
    print(f"Loading {n_mols} molecules from {target} actives...")
    mols = load_molecules(target, n_mols)
    print(f"  Loaded {len(mols)} molecules")

    print("Converting to PDBQT...")
    lig_pdbqts = []
    for mol in mols:
        try:
            lig_pdbqts.append(mol_to_pdbqt(mol))
        except Exception as e:
            print(f"  Skip: {e}")
    n = len(lig_pdbqts)
    print(f"  {n} PDBQT strings ready")

    # ── Benchmark Docking ─────────────────────────────────────────────────
    exhaustiveness_values = [2, 4, 8, 16]

    for exh in exhaustiveness_values:
        print("\n" + "=" * 70)
        print(f"DOCKING BENCHMARK  (exhaustiveness={exh})")
        print("=" * 70)

        dock_times = {}

        # Sequential
        label = "Sequential"
        print(f"\n  {label}: docking {n} molecules...")
        t0 = time.perf_counter()
        results = bench_docking_sequential(rec_pdbqt, lig_pdbqts, center, size, exh)
        elapsed = time.perf_counter() - t0
        dock_times[label] = elapsed
        print(f"    {elapsed:.2f}s total, {elapsed / n:.2f}s/mol")

        # Pool(4 workers)
        label = "Pool(4 workers)"
        print(f"\n  {label}: docking {n} molecules...")
        t0 = time.perf_counter()
        results = bench_docking_parallel(rec_pdbqt, lig_pdbqts, center, size, 4, exh)
        elapsed = time.perf_counter() - t0
        dock_times[label] = elapsed
        print(f"    {elapsed:.2f}s total, {elapsed / n:.2f}s/mol")

        # Rust batch_dock
        label = "Rust batch_dock"
        print(f"\n  {label}: docking {n} molecules...")
        t0 = time.perf_counter()
        batch_results = bench_docking_batch(rec_pdbqt, lig_pdbqts, center, size, exh)
        elapsed = time.perf_counter() - t0
        dock_times[label] = elapsed
        print(f"    {elapsed:.2f}s total, {elapsed / n:.2f}s/mol")

        # Summary for this exhaustiveness
        print(f"\n  {'Configuration':<25s} {'Time (s)':<12s} {'Speedup':<10s}")
        print(f"  {'-' * 47}")
        seq_t = dock_times.get("Sequential", 1)
        for lbl, dt in dock_times.items():
            sp = f"{seq_t / dt:.2f}x" if lbl != "Sequential" else "-"
            print(f"  {lbl:<25s} {dt:<12.2f} {sp:<10s}")

    # # ── Benchmark GNINA Scoring ───────────────────────────────────────────
    # print("\n" + "=" * 70)
    # print("GNINA SCORING BENCHMARK")
    # print("=" * 70)

    # # Parse receptor for GNINA
    # print("  Parsing receptor for GNINA...")
    # rec_coords, rec_types = parse_pdbqt_string(rec_pdbqt)

    # gnina_times = {}

    # # Sequential (model loaded once)
    # print(f"\n  Sequential: scoring {n} docked molecules...")
    # t0 = time.perf_counter()
    # bench_gnina_sequential(rec_coords, rec_types, docked_pdbqts)
    # elapsed = time.perf_counter() - t0
    # gnina_times["Sequential"] = elapsed
    # print(f"    {elapsed:.2f}s total, {elapsed / n:.2f}s/mol")

    # # Naive parallel (model re-loaded per task)
    # for w in [2, 4]:
    #     label = f"Naive Pool({w})"
    #     print(f"\n  {label}: scoring {n} docked molecules...")
    #     t0 = time.perf_counter()
    #     bench_gnina_parallel_naive(rec_coords, rec_types, docked_pdbqts, w)
    #     elapsed = time.perf_counter() - t0
    #     gnina_times[label] = elapsed
    #     print(f"    {elapsed:.2f}s total, {elapsed / n:.2f}s/mol")

    # # Optimized parallel (model loaded once per worker via initializer)
    # for w in [2, 4]:
    #     label = f"Opt Pool({w})"
    #     print(f"\n  {label}: scoring {n} docked molecules...")
    #     t0 = time.perf_counter()
    #     bench_gnina_parallel_opt(rec_coords, rec_types, docked_pdbqts, w)
    #     elapsed = time.perf_counter() - t0
    #     gnina_times[label] = elapsed
    #     print(f"    {elapsed:.2f}s total, {elapsed / n:.2f}s/mol")

    # # ThreadPoolExecutor (shared memory, no IPC serialization)
    # for w in [2, 4]:
    #     label = f"Threads({w})"
    #     print(f"\n  {label}: scoring {n} docked molecules...")
    #     t0 = time.perf_counter()
    #     bench_gnina_threaded(rec_coords, rec_types, docked_pdbqts, w)
    #     elapsed = time.perf_counter() - t0
    #     gnina_times[label] = elapsed
    #     print(f"    {elapsed:.2f}s total, {elapsed / n:.2f}s/mol")

    # # ── Summary ───────────────────────────────────────────────────────────
    # print("\n" + "=" * 70)
    # print("SUMMARY")
    # print("=" * 70)

    # print(f"\n  DOCKING:")
    # print(f"  {'Configuration':<25s} {'Time (s)':<12s} {'Speedup':<10s}")
    # print(f"  {'-' * 47}")
    # seq_dock = dock_times.get("Sequential", 1)
    # for label, dt in dock_times.items():
    #     sp = f"{seq_dock / dt:.2f}x" if label != "Sequential" else "-"
    #     print(f"  {label:<25s} {dt:<12.2f} {sp:<10s}")

    # print(f"\n  GNINA SCORING:")
    # print(f"  {'Configuration':<25s} {'Time (s)':<12s} {'Speedup':<10s}")
    # print(f"  {'-' * 47}")
    # seq_gnina = gnina_times.get("Sequential", 1)
    # for label, gt in gnina_times.items():
    #     sp = f"{seq_gnina / gt:.2f}x" if label != "Sequential" else "-"
    #     print(f"  {label:<25s} {gt:<12.2f} {sp:<10s}")


if __name__ == "__main__":
    main()
