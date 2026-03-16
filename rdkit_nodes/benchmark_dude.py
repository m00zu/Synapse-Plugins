#!/usr/bin/env python3
"""benchmark_dude.py — Full DUD-E benchmark: docking + GNINA CNN rescoring.

For each target:
  1. Prepare receptor PDB → PDBQT
  2. Load actives/decoys from SDF (pre-computed 3D coords)
  3. Dock each conformer with QVina2/Vina/Smina
  4. Score docked poses with GNINA CNN ensemble
  5. Select best conformer per molecule
  6. Compute AUC-ROC, enrichment factors, BEDROC

Usage:
    python -m plugins.rdkit_nodes.benchmark_dude \\
        /path/to/dude_prepared /path/to/output \\
        --engine qvina2 [--targets cdk2,aa2ar] [--max_mols 50]
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import sdfrust
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

RDLogger.DisableLog("rdApp.*")

console = Console()

# Maximum number of per-ligand rows visible simultaneously in the progress
# display.  Oldest active rows are hidden when this limit is exceeded.
_MAX_VISIBLE_LIGANDS = 50


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
)


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


def read_pockets(pocket_file: Path) -> dict:
    """Read pockets.txt → {target: ((cx,cy,cz), (sx,sy,sz))}."""
    pockets = {}
    with open(pocket_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            name = parts[0]
            center = (float(parts[1]), float(parts[2]), float(parts[3]))
            size = (float(parts[4]), float(parts[5]), float(parts[6]))
            pockets[name] = (center, size)
    return pockets


def load_sdf_molecules(sdf_path: Path) -> list[tuple[str, Chem.Mol, object, str | None, list | None]]:
    """Load molecules from .sdf.gz, return [(name, rdkit_mol, rust_mol, smiles, order), ...].

    SMILES and atom order are computed before AddHs (cheap) to avoid
    expensive RemoveAllHs later.
    """
    entries = []
    rust_mols = list(sdfrust.iter_sdf_file(str(sdf_path)))
    with gzip.open(sdf_path) as f:
        suppl = Chem.ForwardSDMolSupplier(f, removeHs=False)
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"MOL_{len(entries)}"
            has_h = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())
            if not has_h:
                # Compute SMILES before AddHs (fast path)
                smi = Chem.MolToSmiles(mol, canonical=True)
                order = list(ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder')))
                mol = Chem.AddHs(mol, addCoords=True)
            else:
                mol_noH = Chem.RemoveAllHs(mol)
                smi = Chem.MolToSmiles(mol_noH, canonical=True)
                order_noH = ast.literal_eval(mol_noH.GetProp('_smilesAtomOutputOrder'))
                heavy_idx = [j for j in range(mol.GetNumAtoms())
                             if mol.GetAtomWithIdx(j).GetAtomicNum() != 1]
                order = [heavy_idx[j] for j in order_noH]
            rust_mol = rust_mols[i] if i < len(rust_mols) else None
            entries.append((name, mol, rust_mol, smi, order))
    return entries


def enrichment_factor(labels, scores, fraction):
    """Compute enrichment factor at given fraction (e.g. 0.01 for 1%)."""
    n = len(labels)
    n_actives = sum(labels)
    if n_actives == 0 or n == 0:
        return 0.0
    # Sort by score descending
    order = np.argsort(-np.array(scores))
    sorted_labels = np.array(labels)[order]
    cutoff = max(1, int(n * fraction))
    hits = sorted_labels[:cutoff].sum()
    expected = n_actives * fraction
    if expected == 0:
        return 0.0
    return hits / expected


def compute_bedroc(labels, scores, alpha=20.0):
    """Compute BEDROC using RDKit's implementation."""
    try:
        from rdkit.ML.Scoring.Scoring import CalcBEDROC
        # CalcBEDROC expects list of (score, label) sorted by score descending
        order = np.argsort(-np.array(scores))
        scored = [(float(scores[i]), int(labels[i])) for i in order]
        return CalcBEDROC(scored, col=1, alpha=alpha)
    except ImportError:
        return float("nan")


def prepare_receptor(target_dir: Path, output_dir: Path) -> str:
    """Prepare receptor PDB → PDBQT. Returns PDBQT string. Caches to file."""
    pdbqt_path = output_dir / "receptor.pdbqt"
    if pdbqt_path.exists():
        return pdbqt_path.read_text()

    pdb_path = target_dir / "receptor.pdb"
    result = fix_and_convert(str(pdb_path))
    if isinstance(result, tuple):
        raise RuntimeError(f"Receptor prep failed: {result[0]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_path.write_text(result)
    return result


def score_poses_gnina(model, rec_coords, rec_types, poses_pdbqt):
    """Score docked poses with GNINA.

    Returns (best_CNNscore, best_CNNaffinity) where each is the best
    value across all poses (independently selected).
    """
    poses = parse_pdbqt_poses(poses_pdbqt)
    if not poses:
        return None, None
    center = np.mean(poses[0][0], axis=0)
    results = model.score_poses(rec_coords, rec_types, poses, center=center)
    best_score = max(r["CNNscore"] for r in results)
    best_aff = max(r["CNNaffinity"] for r in results)
    return best_score, best_aff


# ── CSV I/O ──────────────────────────────────────────────────────────────────

RESULT_FIELDS = [
    "name", "label", "conf_idx", "vina_affinity",
    "CNNscore", "CNNaffinity", "n_poses", "status",
]


def load_existing_results(csv_path: Path) -> set[str]:
    """Load already-completed molecule keys from results.csv.
    Key = 'name_c{conf_idx}'."""
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['name']}_c{row['conf_idx']}"
            done.add(key)
    return done


def open_csv_writer(csv_path: Path, resume: bool):
    """Open CSV for appending results. Returns (file_handle, writer)."""
    if resume and csv_path.exists():
        fh = open(csv_path, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(csv_path, "w", newline="")
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()
    return fh, writer


# ── Progress helpers ─────────────────────────────────────────────────────────

def _make_progress() -> Progress:
    """Create a rich Progress bar with standard columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=25),
        TextColumn("{task.percentage:>5.1f}%"),
        TextColumn("[dim]{task.fields[info]}[/]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(compact=True),
        console=console,
        expand=False,
    )


def _make_ligand_progress() -> Progress:
    """Create a progress display for individual ligand rows."""
    return Progress(
        TextColumn("    {task.description}"),
        BarColumn(bar_width=20),
        TextColumn("{task.percentage:>5.1f}%"),
        TextColumn("[dim]{task.fields[info]}[/]"),
        console=console,
        expand=False,
    )


# ── Per-target processing ────────────────────────────────────────────────────

def process_target(
    target: str,
    prepared_dir: Path,
    output_dir: Path,
    pocket: tuple,
    engine: str,
    exhaustiveness: int,
    n_poses: int,
    max_mols: int | None,
    skip_gnina: bool,
    gnina_model: GNINAModel | None,
    resume: bool,
    progress: Progress | None = None,
):
    """Process a single DUD-E target. Returns dict of metrics or None on failure."""
    t_target_start = time.perf_counter()
    log = progress.console.print if progress else console.print
    target_dir = prepared_dir / target
    center, size = pocket
    out_dir = output_dir / f"{target}_{engine}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Prepare receptor ──────────────────────────────────────────────────
    try:
        rec_pdbqt = prepare_receptor(target_dir, out_dir)
    except Exception as e:
        log(f"  [red]x[/] {target}: receptor prep failed: {e}")
        return None

    # Parse receptor for GNINA (if needed)
    rec_coords = rec_types = None
    if not skip_gnina and gnina_model is not None:
        try:
            rec_coords, rec_types = parse_pdbqt_string(rec_pdbqt)
        except Exception as e:
            log(f"  [yellow]![/] {target}: GNINA receptor parse failed: {e}")
            skip_gnina = True

    # ── Load molecules ────────────────────────────────────────────────────
    actives_sdf = target_dir / "actives_final.sdf.gz"
    decoys_sdf = target_dir / "decoys_final.sdf.gz"

    if not actives_sdf.exists() or not decoys_sdf.exists():
        log(f"  [red]x[/] {target}: missing SDF files")
        return None

    actives = load_sdf_molecules(actives_sdf)
    decoys = load_sdf_molecules(decoys_sdf)

    # Assign labels
    mol_entries = [(name, mol, rust_mol, smi, order, 1) for name, mol, rust_mol, smi, order in actives] + \
                  [(name, mol, rust_mol, smi, order, 0) for name, mol, rust_mol, smi, order in decoys]

    if max_mols is not None:
        mol_entries = mol_entries[:max_mols]

    # Assign conformer indices (same name → incrementing conf_idx)
    name_counts: dict[str, int] = {}
    entries_with_conf = []
    for name, mol, rust_mol, smi, order, label in mol_entries:
        cidx = name_counts.get(name, 0)
        name_counts[name] = cidx + 1
        entries_with_conf.append((name, mol, rust_mol, smi, order, label, cidx))

    # ── Resume: load existing results ─────────────────────────────────────
    csv_path = out_dir / "results.csv"
    done_keys = load_existing_results(csv_path) if resume else set()
    remaining = [(n, m, rm, smi, order, l, c) for n, m, rm, smi, order, l, c in entries_with_conf
                 if f"{n}_c{c}" not in done_keys]

    total = len(entries_with_conf)
    already = total - len(remaining)
    if already > 0:
        log(f"  [dim]{target}: resuming — {already}/{total} already done, "
            f"{len(remaining)} remaining[/]")

    # ── Phase 1: Docking ──────────────────────────────────────────────────
    engine_name = engine.capitalize() if engine != "qvina2" else "QVina2"

    if not remaining:
        log(f"  [dim]{target}: all molecules already docked[/]")
    else:
        fh, writer = open_csv_writer(csv_path, resume)

        # ── Pre-convert all molecules to PDBQT (Rust batch) ────────────
        n_remaining = len(remaining)
        rdkit_mols_rem = [entry[1] for entry in remaining]
        rust_mols_rem = [entry[2] for entry in remaining]
        smi_list_rem = [entry[3] for entry in remaining]
        order_list_rem = [entry[4] for entry in remaining]

        # 1. Pre-compute RDKit aromaticity + symmetry (single-threaded —
        #    RDKit holds GIL so ProcessPool has no benefit, and avoids
        #    fork-deadlock with Rayon threads from sdfrust).
        precomp_task = None
        if progress:
            precomp_task = progress.add_task(
                f"  [cyan]{target}[/] RDKit precomp",
                total=n_remaining, info="")

        arom_list: list = [None] * n_remaining
        sym_list: list = [None] * n_remaining
        for idx, mol in enumerate(rdkit_mols_rem):
            if mol is not None:
                arom_list[idx] = [a.GetIsAromatic() for a in mol.GetAtoms()]
                sym_list[idx] = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
            if precomp_task is not None:
                progress.advance(precomp_task)

        if precomp_task is not None:
            progress.remove_task(precomp_task)

        # 2. Batch Rust PDBQT conversion with Rayon
        valid_mask = [rm is not None for rm in rust_mols_rem]
        valid_rust = [rm for rm in rust_mols_rem if rm is not None]
        valid_arom = [a for a, v in zip(arom_list, valid_mask) if v]
        valid_sym = [s for s, v in zip(sym_list, valid_mask) if v]
        valid_smi = [s for s, v in zip(smi_list_rem, valid_mask) if v]
        valid_order = [o for o, v in zip(order_list_rem, valid_mask) if v]
        n_valid_rust = len(valid_rust)

        conv_task = None
        if progress:
            conv_task = progress.add_task(
                f"  [cyan]{target}[/] PDBQT convert",
                total=n_valid_rust, info="")

        def _pdbqt_progress(i, total_mols, result):
            if conv_task is not None:
                progress.advance(conv_task)

        if valid_rust:
            batch_results = sdfrust.batch_mol_to_pdbqt(
                valid_rust,
                aromatic_atoms=valid_arom,
                symmetry_classes=valid_sym,
                smiles=valid_smi,
                smiles_atom_orders=valid_order,
                callback=_pdbqt_progress if progress else None,
            )
        else:
            batch_results = []

        if conv_task is not None:
            progress.remove_task(conv_task)

        # Map back to full list
        lig_pdbqts: list[str | None] = [None] * n_remaining
        bi = 0
        for i, v in enumerate(valid_mask):
            if v:
                lig_pdbqts[i] = batch_results[bi]
                bi += 1

        n_failed_prep = sum(1 for p in lig_pdbqts if p is None)
        if n_failed_prep:
            log(f"  [yellow]![/] {target}: {n_failed_prep} PDBQT conversions failed")

        # ── Batch docking with streaming results ─────────────────────
        valid_indices = [i for i, p in enumerate(lig_pdbqts) if p is not None]
        valid_pdbqts = [lig_pdbqts[i] for i in valid_indices]
        n_valid = len(valid_pdbqts)

        # Per-ligand display names
        batch_names: list[str] = []
        for vi in valid_indices:
            name, _, _, _, _, _, cidx = remaining[vi]
            batch_names.append(f"{name}_c{cidx}" if cidx > 0 else name)

        t_dock_start = time.perf_counter()

        if valid_pdbqts:
            backend = VinaRustBackend(engine_name)

            # ── Progress state (shared across chunks) ─────────────────
            dock_task = None
            lig_tasks: dict[int, int] = {}
            _visible_q: collections.deque[int] = collections.deque()
            _done_count = [0]
            _done_set: set[int] = set()  # ligand indices that got LigandDone

            if progress:
                for bi, lname in enumerate(batch_names):
                    tid = progress.add_task(
                        f"[dim]{lname[:24]:<24}[/]",
                        total=100, visible=False, info="queued",
                    )
                    lig_tasks[bi] = tid
                dock_task = progress.add_task(
                    f"  [cyan]{target}[/] Docking",
                    total=n_valid, info=f"0/{n_valid}",
                )

            # ── Streaming callback: save results on-the-fly ──────────
            # With stream_results=True, each LigandDone callback carries
            # result_pdbqt + result_energies.  We save to disk + CSV
            # immediately, so memory is freed right after the callback.

            def _make_dock_cb():
                def dock_cb(**kwargs):
                    stage = kwargs.get('stage', '')
                    lig_idx = int(kwargs.get('ligand_index', 0))

                    if stage == 'LigandDone':
                        _done_count[0] += 1
                        _done_set.add(lig_idx)
                        best_e = kwargs.get('best_energy', 0.0)
                        score_s = f"{best_e:.1f}" if best_e != 0.0 else "n/a"
                        poses_pdbqt = kwargs.get('result_pdbqt', '')
                        energies = kwargs.get('result_energies', [])

                        # Map callback ligand_index → remaining index
                        ri = valid_indices[lig_idx]
                        name, _, _, _, _, label, conf_idx = remaining[ri]

                        row = {
                            "name": name, "label": label,
                            "conf_idx": conf_idx,
                            "vina_affinity": "", "CNNscore": "",
                            "CNNaffinity": "", "n_poses": 0,
                            "status": "dock_failed",
                        }
                        if poses_pdbqt and energies:
                            row["vina_affinity"] = round(energies[0][0], 3)
                            row["n_poses"] = len(energies)
                            row["status"] = "docked"
                            poses_file = out_dir / f"{name}_c{conf_idx}.pdbqt"
                            poses_file.write_text(poses_pdbqt)
                        writer.writerow(row)
                        fh.flush()

                        # Progress UI
                        if progress:
                            if lig_idx in lig_tasks:
                                progress.remove_task(lig_tasks[lig_idx])
                                del lig_tasks[lig_idx]
                                if lig_idx in _visible_q:
                                    _visible_q.remove(lig_idx)
                            progress.console.print(
                                f"    [green]OK[/] {batch_names[lig_idx]}"
                                f"  {score_s} kcal/mol")
                            progress.update(
                                dock_task,
                                completed=_done_count[0],
                                info=(f"{_done_count[0]}/{n_valid}"
                                      f"  {score_s} kcal/mol"),
                            )
                    elif progress:
                        # Per-ligand intermediate progress
                        overall_pct = kwargs.get('percent_complete', 0)
                        per_lig = max(0.0, min(
                            100.0,
                            overall_pct * n_valid - lig_idx * 100))
                        if lig_idx in lig_tasks:
                            tid = lig_tasks[lig_idx]
                            if lig_idx not in _visible_q:
                                _visible_q.append(lig_idx)
                                progress.update(tid, visible=True)
                                while len(_visible_q) > _MAX_VISIBLE_LIGANDS:
                                    oldest = _visible_q.popleft()
                                    if oldest in lig_tasks:
                                        progress.update(
                                            lig_tasks[oldest], visible=False)
                            progress.update(tid, completed=int(per_lig))
                return dock_cb

            # Single batch_dock call — results streamed via callback,
            # return value is empty stubs (no memory accumulation).
            backend.batch_dock(
                rec_pdbqt, valid_pdbqts, center, size,
                exhaustiveness=exhaustiveness, n_poses=n_poses, seed=42,
                progress_callback=_make_dock_cb(),
                stream_results=True,
            )

            # Write rows for ligands that never got LigandDone (silent failure)
            valid_set = {valid_indices[bi] for bi in range(n_valid)}
            n_silent = 0
            for bi in range(n_valid):
                if bi not in _done_set:
                    ri = valid_indices[bi]
                    name, _, _, _, _, label, conf_idx = remaining[ri]
                    writer.writerow({
                        "name": name, "label": label, "conf_idx": conf_idx,
                        "vina_affinity": "", "CNNscore": "",
                        "CNNaffinity": "", "n_poses": 0,
                        "status": "dock_failed",
                    })
                    n_silent += 1
            if n_silent:
                log(f"  [yellow]![/] {target}: {n_silent} ligands silently failed docking")

            # Write rows for failed PDBQT conversions (never sent to Rust)
            for ri, (name, _, _, _, _, label, conf_idx) in enumerate(remaining):
                if ri not in valid_set:
                    writer.writerow({
                        "name": name, "label": label, "conf_idx": conf_idx,
                        "vina_affinity": "", "CNNscore": "",
                        "CNNaffinity": "", "n_poses": 0,
                        "status": "pdbqt_failed",
                    })
            fh.flush()

            # Clean up progress tasks
            if progress:
                for tid in lig_tasks.values():
                    progress.update(tid, visible=False)
                    progress.remove_task(tid)
                progress.update(dock_task, completed=n_valid,
                                info=f"{n_valid}/{n_valid} done")
                progress.remove_task(dock_task)
        else:
            # No valid PDBQT — write all as failed
            for name, _, _, _, _, label, conf_idx in remaining:
                writer.writerow({
                    "name": name, "label": label, "conf_idx": conf_idx,
                    "vina_affinity": "", "CNNscore": "", "CNNaffinity": "",
                    "n_poses": 0, "status": "pdbqt_failed",
                })
            fh.flush()

        fh.close()

        elapsed_dock = time.perf_counter() - t_dock_start
        rate = n_valid / elapsed_dock if elapsed_dock > 0 else 0
        log(f"  [dim]{target}: docked {n_valid}/{total} "
            f"in {elapsed_dock:.1f}s ({rate:.1f} mol/s)[/]")

    # ── Phase 2: GNINA scoring (independent of docking phase) ─────────
    # Scan CSV for rows with status="docked" that still need GNINA.
    # This handles both fresh runs and resumes where docking finished
    # but GNINA was interrupted.
    if not skip_gnina and gnina_model is not None:
        df_csv = pd.read_csv(csv_path)
        needs_gnina = df_csv[df_csv["status"] == "docked"]

        if not needs_gnina.empty:
            gnina_queue: list[tuple[str, str]] = []
            for _, erow in needs_gnina.iterrows():
                mol_key = f"{erow['name']}_c{erow['conf_idx']}"
                poses_file = out_dir / f"{erow['name']}_c{erow['conf_idx']}.pdbqt"
                if poses_file.exists():
                    gnina_queue.append((mol_key, poses_file.read_text()))

            if gnina_queue:
                gnina_workers = min(4, len(gnina_queue))

                def _score_one(item):
                    mol_key, pdbqt = item
                    try:
                        return mol_key, score_poses_gnina(
                            gnina_model, rec_coords, rec_types, pdbqt)
                    except Exception:
                        return mol_key, (None, None)

                gnina_task = None
                if progress:
                    gnina_task = progress.add_task(
                        f"  [cyan]{target}[/] GNINA scoring",
                        total=len(gnina_queue), info="")

                # Score → collect updates
                updates: dict[str, dict] = {}  # mol_key → {CNNscore, ...}
                scored = 0
                with ThreadPoolExecutor(max_workers=gnina_workers) as pool:
                    futures = {pool.submit(_score_one, item): item
                               for item in gnina_queue}
                    for future in as_completed(futures):
                        mol_key, (cnn_score, cnn_aff) = future.result()
                        if cnn_score is not None:
                            updates[mol_key] = {
                                "CNNscore": round(cnn_score, 4),
                                "CNNaffinity": round(cnn_aff, 2),
                                "status": "success",
                            }
                        else:
                            updates[mol_key] = {"status": "gnina_failed"}
                        scored += 1
                        if gnina_task is not None:
                            progress.update(gnina_task, completed=scored,
                                            info=mol_key[:20])
                        elif not progress:
                            print(f"\r  {target}: gnina "
                                  f"{scored}/{len(gnina_queue)}  ",
                                  end="", flush=True)

                if gnina_task is not None:
                    progress.update(gnina_task,
                                    completed=len(gnina_queue), info="done")
                    progress.remove_task(gnina_task)
                elif not progress:
                    print()

                # Apply updates and rewrite CSV
                for idx, erow in df_csv.iterrows():
                    mk = f"{erow['name']}_c{erow['conf_idx']}"
                    if mk in updates:
                        for k, v in updates[mk].items():
                            df_csv.at[idx, k] = v

                df_csv.to_csv(csv_path, index=False)

    metrics = compute_target_metrics(csv_path, target)
    if metrics is not None:
        # Accumulate elapsed time across resume runs
        timing_path = out_dir / "timing.json"
        prev_sec = 0.0
        if timing_path.exists():
            try:
                prev_sec = json.loads(timing_path.read_text()).get("elapsed_sec", 0.0)
            except (json.JSONDecodeError, KeyError):
                pass
        this_sec = time.perf_counter() - t_target_start
        total_sec = prev_sec + this_sec
        timing_path.write_text(json.dumps({"elapsed_sec": round(total_sec, 1)}))
        metrics["elapsed_min"] = round(total_sec / 60, 2)
    return metrics


def compute_target_metrics(csv_path: Path, target: str) -> dict:
    """Compute metrics from a target's results.csv."""
    df = pd.read_csv(csv_path)

    # Best-conformer selection: for each unique molecule name,
    # pick the conformer with the best (most negative) vina_affinity
    df_valid = df[df["status"].isin(["success", "docked"])].copy()
    if df_valid.empty:
        return {"target": target, "n_actives": 0, "n_decoys": 0, "n_failed": len(df)}

    # Convert numeric columns
    for col in ["vina_affinity", "CNNscore", "CNNaffinity"]:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

    # Best conformer by vina_affinity (most negative = best)
    best_vina = df_valid.loc[df_valid.groupby("name")["vina_affinity"].idxmin()]

    labels = best_vina["label"].astype(int).values
    n_actives = int(labels.sum())
    n_decoys = int(len(labels) - n_actives)
    n_failed = int(len(df) - len(df_valid))

    metrics = {
        "target": target,
        "n_actives": n_actives,
        "n_decoys": n_decoys,
        "n_failed": n_failed,
    }

    # Vina AUC (more negative = better binder, so negate for AUC)
    vina_scores = best_vina["vina_affinity"].values
    if n_actives > 0 and n_decoys > 0 and not np.isnan(vina_scores).all():
        valid_mask = ~np.isnan(vina_scores)
        if valid_mask.sum() > 1:
            metrics["vina_auc"] = round(roc_auc_score(
                labels[valid_mask], -vina_scores[valid_mask]), 4)
            for frac, key in [(0.005, "0.5"), (0.01, "1"), (0.02, "2"),
                              (0.05, "5"), (0.10, "10")]:
                metrics[f"vina_ef{key}"] = round(enrichment_factor(
                    labels[valid_mask], -vina_scores[valid_mask], frac), 2)
            metrics["vina_bedroc"] = round(compute_bedroc(
                labels[valid_mask], -vina_scores[valid_mask]), 4)

    # CNN score AUC (higher CNNscore = better binder)
    # Best conformer by CNNscore
    cnn_valid = df_valid.dropna(subset=["CNNscore"])
    if not cnn_valid.empty:
        best_cnn = cnn_valid.loc[cnn_valid.groupby("name")["CNNscore"].idxmax()]
        cnn_labels = best_cnn["label"].astype(int).values
        cnn_scores = best_cnn["CNNscore"].values
        n_act_cnn = int(cnn_labels.sum())
        n_dec_cnn = int(len(cnn_labels) - n_act_cnn)
        if n_act_cnn > 0 and n_dec_cnn > 0:
            metrics["cnn_auc"] = round(roc_auc_score(cnn_labels, cnn_scores), 4)
            for frac, key in [(0.005, "0.5"), (0.01, "1"), (0.02, "2"),
                              (0.05, "5"), (0.10, "10")]:
                metrics[f"cnn_ef{key}"] = round(enrichment_factor(
                    cnn_labels, cnn_scores, frac), 2)
            metrics["cnn_bedroc"] = round(compute_bedroc(
                cnn_labels, cnn_scores), 4)

    # CNN affinity AUC (higher pKd = better binder, like CNNscore)
    # Best conformer by highest CNNaffinity
    cnnaff_valid = df_valid.dropna(subset=["CNNaffinity"])
    if not cnnaff_valid.empty:
        best_aff = cnnaff_valid.loc[cnnaff_valid.groupby("name")["CNNaffinity"].idxmax()]
        aff_labels = best_aff["label"].astype(int).values
        aff_scores = best_aff["CNNaffinity"].values
        n_act_aff = int(aff_labels.sum())
        n_dec_aff = int(len(aff_labels) - n_act_aff)
        if n_act_aff > 0 and n_dec_aff > 0:
            metrics["cnnaff_auc"] = round(roc_auc_score(aff_labels, aff_scores), 4)
            for frac, key in [(0.005, "0.5"), (0.01, "1"), (0.02, "2"),
                              (0.05, "5"), (0.10, "10")]:
                metrics[f"cnnaff_ef{key}"] = round(enrichment_factor(
                    aff_labels, aff_scores, frac), 2)
            metrics["cnnaff_bedroc"] = round(compute_bedroc(
                aff_labels, aff_scores), 4)

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DUD-E benchmark: docking + GNINA CNN rescoring")
    parser.add_argument("prepared_dir", type=Path,
                        help="Path to prepared DUD-E directory")
    parser.add_argument("output_dir", type=Path,
                        help="Output directory for results")
    parser.add_argument("--engine", required=True,
                        choices=["qvina2", "vina", "smina"],
                        help="Docking engine")
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated target names (default: all)")
    parser.add_argument("--exhaustiveness", type=int, default=4)
    parser.add_argument("--n_poses", type=int, default=9)
    parser.add_argument("--max_mols", type=int, default=None,
                        help="Limit molecules per target (for testing)")
    parser.add_argument("--gnina_ensemble", default="default",
                        help="GNINA model ensemble name")
    parser.add_argument("--skip_gnina", action="store_true",
                        help="Skip GNINA CNN rescoring")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results")
    args = parser.parse_args()

    prepared_dir = args.prepared_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Read pockets ──────────────────────────────────────────────────────
    pockets = read_pockets(prepared_dir / "pockets.txt")

    # ── Select targets ────────────────────────────────────────────────────
    if args.targets:
        target_names = [t.strip() for t in args.targets.split(",")]
        missing = [t for t in target_names if t not in pockets]
        if missing:
            console.print(f"[yellow]WARNING:[/] targets not in pockets.txt: {missing}")
            target_names = [t for t in target_names if t in pockets]
    else:
        target_names = sorted(pockets.keys())

    console.print(
        f"[bold]DUD-E Benchmark[/]  {len(target_names)} targets, "
        f"engine={args.engine}, exhaustiveness={args.exhaustiveness}, "
        f"n_poses={args.n_poses}"
    )
    if args.max_mols:
        console.print(f"  max_mols={args.max_mols}")

    # ── Load GNINA model ──────────────────────────────────────────────────
    gnina_model = None
    if not args.skip_gnina:
        console.print("[dim]Loading GNINA models...[/]")
        gnina_model = GNINAModel(ensemble=args.gnina_ensemble)

    # ── Process targets ───────────────────────────────────────────────────
    all_metrics = []
    t_total = time.perf_counter()

    progress = _make_progress()
    with progress:
        overall = progress.add_task(
            "[bold blue]Overall", total=len(target_names), info="")

        for ti, target in enumerate(target_names):
            progress.update(
                overall,
                description=f"[bold blue]Overall [dim]({ti}/{len(target_names)})[/]",
                info=target,
            )

            metrics = process_target(
                target=target,
                prepared_dir=prepared_dir,
                output_dir=output_dir,
                pocket=pockets[target],
                engine=args.engine,
                exhaustiveness=args.exhaustiveness,
                n_poses=args.n_poses,
                max_mols=args.max_mols,
                skip_gnina=args.skip_gnina,
                gnina_model=gnina_model,
                resume=args.resume,
                progress=progress,
            )

            progress.advance(overall)

            if metrics:
                all_metrics.append(metrics)
                parts = []
                if "vina_auc" in metrics:
                    parts.append(f"Vina AUC={metrics['vina_auc']:.3f}")
                if "cnn_auc" in metrics:
                    parts.append(f"CNN AUC={metrics['cnn_auc']:.3f}")
                if "cnnaff_auc" in metrics:
                    parts.append(f"CNNaff AUC={metrics['cnnaff_auc']:.3f}")
                if "elapsed_min" in metrics:
                    parts.append(f"{metrics['elapsed_min']:.1f}min")
                result_str = ", ".join(parts) if parts else "no metrics"
                progress.console.print(
                    f"  [green]OK[/] [bold]{target}[/]: {result_str}")
            else:
                progress.console.print(
                    f"  [red]FAIL[/] [bold]{target}[/]")

        progress.update(overall, info="done")

    # ── Write summary CSV ─────────────────────────────────────────────────
    if all_metrics:
        summary_path = output_dir / "summary.csv"
        new_df = pd.DataFrame(all_metrics)
        # Merge with existing summary (update rows for re-run targets, keep others)
        if summary_path.exists():
            old_df = pd.read_csv(summary_path)
            # Remove old rows for targets we just ran, then append new
            old_df = old_df[~old_df["target"].isin(new_df["target"])]
            summary_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            summary_df = new_df
        summary_df.sort_values("target", inplace=True)
        summary_df.to_csv(summary_path, index=False)

        elapsed = time.perf_counter() - t_total
        console.print()
        console.rule("[bold]Results")
        n_total_summary = len(summary_df)
        console.print(
            f"[bold green]DONE:[/] {len(all_metrics)} targets "
            f"in {elapsed / 60:.1f} min"
            f" ({n_total_summary} total in summary)")
        console.print(f"Summary: {summary_path}")

        if "vina_auc" in summary_df.columns:
            v = summary_df["vina_auc"].dropna()
            console.print(
                f"  Vina AUC:    mean={v.mean():.3f}, median={v.median():.3f}")
        if "cnn_auc" in summary_df.columns:
            c = summary_df["cnn_auc"].dropna()
            console.print(
                f"  CNN AUC:     mean={c.mean():.3f}, median={c.median():.3f}")
        if "vina_bedroc" in summary_df.columns:
            vb = summary_df["vina_bedroc"].dropna()
            console.print(
                f"  Vina BEDROC: mean={vb.mean():.3f}, median={vb.median():.3f}")
        if "cnn_bedroc" in summary_df.columns:
            cb = summary_df["cnn_bedroc"].dropna()
            console.print(
                f"  CNN BEDROC:  mean={cb.mean():.3f}, median={cb.median():.3f}")
        if "cnnaff_auc" in summary_df.columns:
            ca = summary_df["cnnaff_auc"].dropna()
            console.print(
                f"  CNNaff AUC:  mean={ca.mean():.3f}, median={ca.median():.3f}")
        if "cnnaff_bedroc" in summary_df.columns:
            cab = summary_df["cnnaff_bedroc"].dropna()
            console.print(
                f"  CNNaff BEDROC: mean={cab.mean():.3f}, median={cab.median():.3f}")
        if "elapsed_min" in summary_df.columns:
            et = summary_df["elapsed_min"].dropna()
            console.print(
                f"  Time/target:   mean={et.mean():.1f}min, "
                f"total={et.sum():.1f}min")
    else:
        console.print("\n[red]No targets processed successfully.[/]")


if __name__ == "__main__":
    main()
