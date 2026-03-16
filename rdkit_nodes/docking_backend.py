"""
docking_backend — Unified interface for molecular docking engines.

Supports:
  - QVina2 (Rust)  — fast heuristic search, default
  - Vina (Rust)    — classic AutoDock Vina algorithm
  - Smina (Rust)   — Smina scoring/search variant

All engines are implemented in Rust via vina_rust and support Boron
atom types (like GNINA).
"""
from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple


class DockingBackend(ABC):
    """Abstract base for docking engines."""

    @abstractmethod
    def dock(
        self,
        receptor_pdbqt: str,
        ligand_pdbqt: str,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        *,
        exhaustiveness: int = 8,
        n_poses: int = 9,
        energy_range: float = 3.0,
        seed: int = 42,
        cpu: int = 0,
        scoring: str = 'vina',
        flex_pdbqt: str = '',
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[str, List[Tuple[float, ...]]]:
        """Run docking.

        Returns:
            (poses_pdbqt_string, [(affinity, dist_lb, dist_ub), ...])
        """

    @staticmethod
    def parse_vina_output_energies(pdbqt_string: str) -> List[Tuple[float, ...]]:
        """Extract energies from Vina output PDBQT."""
        energies = []
        for line in pdbqt_string.splitlines():
            if line.startswith('REMARK VINA RESULT'):
                parts = line.split()
                try:
                    energies.append(tuple(float(x) for x in parts[3:6]))
                except (ValueError, IndexError):
                    pass
        return energies


# ══════════════════════════════════════════════════════════════════════════════
#  vina_rust backend (QVina2, Vina, Smina)
# ══════════════════════════════════════════════════════════════════════════════

class VinaRustBackend(DockingBackend):
    """Use vina_rust Python bindings (Vina, QVina2, or Smina)."""

    def __init__(self, engine_class: str = 'QVina2'):
        self._engine_class = engine_class

    def dock(
        self,
        receptor_pdbqt: str,
        ligand_pdbqt: str,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        *,
        exhaustiveness: int = 8,
        n_poses: int = 9,
        energy_range: float = 3.0,
        seed: int = 42,
        cpu: int = 0,
        scoring: str = 'vina',
        flex_pdbqt: str = '',
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[str, List[Tuple[float, ...]]]:
        try:
            import vina_rust
        except ImportError as e:
            raise ImportError(
                'vina_rust is not installed.\n'
                f'Original error: {e}'
            ) from e

        engine_cls = getattr(vina_rust, self._engine_class, None)
        if engine_cls is None:
            raise ValueError(f'Unknown vina_rust engine: {self._engine_class}')

        engine = engine_cls(sf_name=scoring, cpu=cpu, seed=seed, verbosity=0)

        # Write receptor to temp file (vina_rust expects file path)
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.pdbqt', delete=False) as tmp:
            tmp.write(receptor_pdbqt)
            receptor_path = tmp.name

        try:
            engine.set_receptor(rigid_name=receptor_path)
        finally:
            os.unlink(receptor_path)

        engine.set_ligand_from_string(ligand_pdbqt)
        engine.compute_vina_maps(
            center_x=center[0], center_y=center[1], center_z=center[2],
            size_x=size[0], size_y=size[1], size_z=size[2],
        )

        # Suppress Rust engine's stderr banner (e.g. "[QVina2] tasks=…")
        _saved_stderr = os.dup(2)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 2)
        try:
            engine.global_search(
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                min_rmsd=1.0,
                max_evals=0,
                progress_callback=progress_callback,
            )
        finally:
            os.dup2(_saved_stderr, 2)
            os.close(_saved_stderr)
            os.close(_devnull)

        energies_raw = engine.get_poses_energies(
            how_many=n_poses, energy_range=energy_range)
        energies = [tuple(e[:3]) for e in energies_raw]

        poses_pdbqt = engine.get_poses(how_many=n_poses, energy_range=energy_range)
        return poses_pdbqt, energies

    def batch_dock(
        self,
        receptor_pdbqt: str,
        ligand_pdbqts: List[str],
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        *,
        exhaustiveness: int = 8,
        n_poses: int = 9,
        energy_range: float = 3.0,
        seed: int = 42,
        cpu: int = 0,
        scoring: str = 'vina',
        progress_callback: Optional[Callable] = None,
        stream_results: bool = False,
    ) -> List[Tuple[str, List[Tuple[float, ...]]]]:
        """Batch dock multiple ligands against one receptor.

        Grid maps are computed once and reused for all ligands,
        giving significant speedup over sequential dock() calls.

        When *stream_results* is True the callback receives
        ``result_pdbqt`` and ``result_energies`` kwargs on the
        ``LigandDone`` stage.  The return list will contain empty
        stubs (``("", [])``) — results should be consumed in the
        callback instead.

        Returns:
            List of (poses_pdbqt, [(affinity, intra, inter, ...), ...])
            per ligand.  Failed ligands return ("", []).
        """
        try:
            import vina_rust
        except ImportError as e:
            raise ImportError(
                'vina_rust is not installed.\n'
                f'Original error: {e}'
            ) from e

        engine_cls = getattr(vina_rust, self._engine_class, None)
        if engine_cls is None:
            raise ValueError(f'Unknown vina_rust engine: {self._engine_class}')

        engine = engine_cls(sf_name=scoring, cpu=cpu, seed=seed, verbosity=0)

        # Write receptor to temp file (vina_rust expects file path)
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.pdbqt', delete=False) as tmp:
            tmp.write(receptor_pdbqt)
            receptor_path = tmp.name

        try:
            engine.set_receptor(rigid_name=receptor_path)
        finally:
            os.unlink(receptor_path)

        # Suppress Rust engine's stderr banner
        _saved_stderr = os.dup(2)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 2)
        try:
            results = engine.batch_dock(
                ligand_strings=ligand_pdbqts,
                center_x=center[0], center_y=center[1], center_z=center[2],
                size_x=size[0], size_y=size[1], size_z=size[2],
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                energy_range=energy_range,
                how_many=n_poses,
                progress_callback=progress_callback,
                stream_results=stream_results,
            )
        finally:
            os.dup2(_saved_stderr, 2)
            os.close(_saved_stderr)
            os.close(_devnull)

        # Convert energies to tuples
        return [
            (poses, [tuple(e) for e in energies])
            for poses, energies in results
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

BACKENDS = {
    'QVina2': lambda: VinaRustBackend(engine_class='QVina2'),
    'Vina':   lambda: VinaRustBackend(engine_class='Vina'),
    'Smina':  lambda: VinaRustBackend(engine_class='Smina'),
}


def get_backend(name: str) -> DockingBackend:
    factory = BACKENDS.get(name)
    if factory is None:
        raise ValueError(f'Unknown backend: {name!r}. Choose from: {list(BACKENDS)}')
    return factory()
