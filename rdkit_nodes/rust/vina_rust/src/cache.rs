use crate::atom::*;
use crate::common::*;
use crate::grid::*;
use crate::model::Model;
use crate::precalculate::*;
use rayon::prelude::*;

// ─── IGrid Trait ───────────────────────────────────────────────────────────────

pub trait IGrid: Send + Sync {
    fn eval(&self, model: &Model, v: f64) -> f64;
    fn eval_deriv(&self, model: &mut Model, v: f64) -> f64;
}

// ─── Cache (Vina/Vinardo grid-based scoring) ───────────────────────────────────

pub struct Cache {
    pub gd: GridDims,
    pub slope: f64,
    pub grids: Vec<Grid>, // one per XS atom type
}

impl Cache {
    pub fn new(slope: f64) -> Self {
        let mut grids = Vec::with_capacity(XS_TYPE_SIZE);
        for _ in 0..XS_TYPE_SIZE {
            grids.push(Grid::new());
        }
        Cache {
            gd: [GridDim::new(); 3],
            slope,
            grids,
        }
    }

    /// Populate grids from receptor atoms using precalculated potentials.
    /// Uses Rayon to parallelize across X-slices for large grids.
    pub fn populate(
        &mut self,
        model: &Model,
        precalc: &Precalculate,
        atom_types_needed: &[usize],
        gd: &GridDims,
    ) {
        self.gd = *gd;

        let d0 = gd[0].n_voxels + 1;
        let d1 = gd[1].n_voxels + 1;
        let d2 = gd[2].n_voxels + 1;
        let total_voxels = d0 * d1 * d2;

        if total_voxels > 500_000 {
            eprintln!(
                "WARNING: Large grid ({}x{}x{} = {} voxels). Map computation may take a while. \
                 Consider increasing granularity (standard Vina uses 0.375).",
                d0, d1, d2, total_voxels
            );
        }

        // Pre-extract receptor atom types and coords for cache-friendly access
        let ga_data: Vec<(Vec3, usize)> = model.grid_atoms.iter()
            .map(|ga| (ga.coords, ga.get_type(AtomTyping::XS)))
            .filter(|(_, t)| *t < XS_TYPE_SIZE)
            .collect();
        let cutoff_sqr = precalc.cutoff_sqr;

        for &t in atom_types_needed {
            if t >= XS_TYPE_SIZE { continue; }

            self.grids[t].init(gd);

            // Compute grid values in parallel across X-slices
            let grid_ref = &self.grids[t];
            let row_data: Vec<Vec<f64>> = (0..d0)
                .into_par_iter()
                .map(|x| {
                    let mut slice = vec![0.0_f64; d1 * d2];
                    for y in 0..d1 {
                        for z in 0..d2 {
                            let loc = grid_ref.index_to_argument(x, y, z);
                            let mut val = 0.0;
                            for &(ga_coords, ga_type) in &ga_data {
                                let r2 = loc.distance_sqr(&ga_coords);
                                if r2 < cutoff_sqr {
                                    val += precalc.eval_fast(t, ga_type, r2);
                                }
                            }
                            slice[y * d2 + z] = val;
                        }
                    }
                    slice
                })
                .collect();

            // Write results back to the grid
            for x in 0..d0 {
                for y in 0..d1 {
                    for z in 0..d2 {
                        self.grids[t].m_data.set(x, y, z, row_data[x][y * d2 + z]);
                    }
                }
            }
        }
    }

    pub fn is_atom_type_initialized(&self, t: usize) -> bool {
        t < self.grids.len() && self.grids[t].initialized()
    }
}

impl IGrid for Cache {
    fn eval(&self, model: &Model, v: f64) -> f64 {
        let mut e = 0.0;
        for i in 0..model.num_movable_atoms {
            let a = &model.atoms[i];
            let t = a.xs;
            if t >= XS_TYPE_SIZE || !self.grids[t].initialized() { continue; }
            // Grid now applies curl internally (matching C++)
            e += self.grids[t].evaluate(&model.coords[i], self.slope, v);
        }
        e
    }

    fn eval_deriv(&self, model: &mut Model, v: f64) -> f64 {
        let mut e = 0.0;
        for i in 0..model.num_movable_atoms {
            let a = &model.atoms[i];
            let t = a.xs;
            if t >= XS_TYPE_SIZE || !self.grids[t].initialized() {
                continue;
            }
            // Grid now applies curl internally (matching C++)
            let (energy, deriv) = self.grids[t].evaluate_deriv(&model.coords[i], self.slope, v);
            e += energy;
            // C++ uses assignment: m.minus_forces[i] = deriv;
            // Since Rust zeros forces first, += is equivalent
            model.minus_forces[i] += deriv;
        }
        e
    }
}

// ─── NonCache (on-the-fly evaluation) ──────────────────────────────────────────

/// Spatial grid for neighbor lookup
#[derive(Clone)]
pub struct SpatialGrid {
    cells: Vec<Vec<usize>>,
    origin: Vec3,
    cell_size: f64,
    dims: [usize; 3],
}

impl SpatialGrid {
    pub fn new(atoms: &[Atom], gd: &GridDims, cutoff: f64) -> Self {
        let cell_size = cutoff;
        let origin = Vec3::new(gd[0].begin, gd[1].begin, gd[2].begin);
        let dims = [
            ((gd[0].end - gd[0].begin) / cell_size).ceil() as usize + 1,
            ((gd[1].end - gd[1].begin) / cell_size).ceil() as usize + 1,
            ((gd[2].end - gd[2].begin) / cell_size).ceil() as usize + 1,
        ];
        let total = dims[0] * dims[1] * dims[2];
        let mut cells = vec![Vec::new(); total];

        for (idx, atom) in atoms.iter().enumerate() {
            let ci = ((atom.coords[0] - origin[0]) / cell_size).floor() as isize;
            let cj = ((atom.coords[1] - origin[1]) / cell_size).floor() as isize;
            let ck = ((atom.coords[2] - origin[2]) / cell_size).floor() as isize;
            if ci >= 0 && cj >= 0 && ck >= 0 {
                let ci = ci as usize;
                let cj = cj as usize;
                let ck = ck as usize;
                if ci < dims[0] && cj < dims[1] && ck < dims[2] {
                    let cell_idx = ci + dims[0] * (cj + dims[1] * ck);
                    cells[cell_idx].push(idx);
                }
            }
        }

        SpatialGrid { cells, origin, cell_size, dims }
    }

    pub fn possibilities(&self, loc: &Vec3) -> Vec<usize> {
        let ci = ((loc[0] - self.origin[0]) / self.cell_size).floor() as isize;
        let cj = ((loc[1] - self.origin[1]) / self.cell_size).floor() as isize;
        let ck = ((loc[2] - self.origin[2]) / self.cell_size).floor() as isize;

        let mut result = Vec::new();
        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    let ni = ci + di;
                    let nj = cj + dj;
                    let nk = ck + dk;
                    if ni >= 0 && nj >= 0 && nk >= 0 {
                        let ni = ni as usize;
                        let nj = nj as usize;
                        let nk = nk as usize;
                        if ni < self.dims[0] && nj < self.dims[1] && nk < self.dims[2] {
                            let cell_idx = ni + self.dims[0] * (nj + self.dims[1] * nk);
                            result.extend_from_slice(&self.cells[cell_idx]);
                        }
                    }
                }
            }
        }
        result
    }
}

#[derive(Clone)]
pub struct NonCache {
    pub sgrid: SpatialGrid,
    pub gd: GridDims,
    pub precalc: Precalculate,
    pub slope: f64,
    pub atom_typing: AtomTyping,
    pub grid_atoms_snapshot: Vec<Atom>, // snapshot of receptor atoms for evaluation
}

impl NonCache {
    pub fn new(model: &Model, gd: &GridDims, precalc: &Precalculate, slope: f64) -> Self {
        let cutoff = precalc.max_cutoff_sqr.sqrt();
        NonCache {
            sgrid: SpatialGrid::new(&model.grid_atoms, gd, cutoff),
            gd: *gd,
            precalc: precalc.clone(),
            slope,
            atom_typing: model.atom_typing,
            grid_atoms_snapshot: model.grid_atoms.clone(),
        }
    }

    /// Check if all non-hydrogen movable atoms are within the grid bounds (matching C++ non_cache::within)
    pub fn within(&self, model: &Model, margin: f64) -> bool {
        for i in 0..model.num_movable_atoms {
            if model.atoms[i].is_hydrogen() { continue; }
            let c = &model.coords[i];
            for dim in 0..3 {
                if self.gd[dim].n_voxels > 0 {
                    if c[dim] < self.gd[dim].begin - margin || c[dim] > self.gd[dim].end + margin {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl IGrid for NonCache {
    fn eval(&self, model: &Model, v: f64) -> f64 {
        let cutoff_sqr = self.precalc.cutoff_sqr;
        let mut e = 0.0;

        for i in 0..model.num_movable_atoms {
            let a = &model.atoms[i];
            let a_coords = &model.coords[i];

            // Clamp to grid bounds
            let mut adjusted = *a_coords;
            let mut penalty = 0.0;
            for dim in 0..3 {
                if adjusted[dim] < self.gd[dim].begin {
                    penalty += (self.gd[dim].begin - adjusted[dim]).abs();
                    adjusted[dim] = self.gd[dim].begin;
                } else if adjusted[dim] > self.gd[dim].end {
                    penalty += (adjusted[dim] - self.gd[dim].end).abs();
                    adjusted[dim] = self.gd[dim].end;
                }
            }
            penalty *= self.slope;

            let t_a = a.get_type(self.atom_typing);
            if t_a >= self.precalc.dim() { continue; }
            let possibilities = self.sgrid.possibilities(&adjusted);

            let mut this_e = 0.0;
            for &j in &possibilities {
                let b = &self.grid_atoms_snapshot[j];
                let t_b = b.get_type(self.atom_typing);
                if t_b >= self.precalc.dim() { continue; }
                let r2 = adjusted.distance_sqr(&b.coords);
                if r2 < cutoff_sqr {
                    this_e += self.precalc.eval_fast(t_a, t_b, r2);
                }
            }
            curl(&mut this_e, v);
            e += this_e + penalty;
        }
        e
    }

    fn eval_deriv(&self, model: &mut Model, v: f64) -> f64 {
        let cutoff_sqr = self.precalc.cutoff_sqr;
        let mut e = 0.0;

        for i in 0..model.num_movable_atoms {
            let a = &model.atoms[i];
            let a_coords = model.coords[i];

            let mut adjusted = a_coords;
            let mut penalty = 0.0;
            let mut penalty_deriv = Vec3::ZERO;
            for dim in 0..3 {
                if adjusted[dim] < self.gd[dim].begin {
                    penalty += (self.gd[dim].begin - adjusted[dim]).abs();
                    penalty_deriv[dim] = -self.slope;
                    adjusted[dim] = self.gd[dim].begin;
                } else if adjusted[dim] > self.gd[dim].end {
                    penalty += (adjusted[dim] - self.gd[dim].end).abs();
                    penalty_deriv[dim] = self.slope;
                    adjusted[dim] = self.gd[dim].end;
                }
            }
            penalty *= self.slope;

            let t_a = a.get_type(self.atom_typing);
            if t_a >= self.precalc.dim() { continue; }
            let possibilities = self.sgrid.possibilities(&adjusted);

            let mut this_e = 0.0;
            let mut this_deriv = Vec3::ZERO;

            for &j in &possibilities {
                let b = &self.grid_atoms_snapshot[j];
                let t_b = b.get_type(self.atom_typing);
                if t_b >= self.precalc.dim() { continue; }
                let r_vec = adjusted - b.coords;
                let r2 = r_vec.norm_sqr();
                if r2 < cutoff_sqr {
                    let (energy, dor) = self.precalc.eval_deriv(t_a, t_b, r2);
                    this_e += energy;
                    this_deriv += r_vec * dor;
                }
            }
            curl_with_deriv(&mut this_e, &mut this_deriv, v);
            e += this_e + penalty;
            model.minus_forces[i] += this_deriv + penalty_deriv;
        }
        e
    }
}

// ─── Ad4Cache ──────────────────────────────────────────────────────────────────

pub struct Ad4Cache {
    pub gd: GridDims,
    pub slope: f64,
    pub grids: Vec<Grid>, // AD_TYPE_SIZE + 2 (elec + desolv)
}

impl Ad4Cache {
    pub fn new(slope: f64) -> Self {
        let mut grids = Vec::with_capacity(AD_TYPE_SIZE + 2);
        for _ in 0..AD_TYPE_SIZE + 2 {
            grids.push(Grid::new());
        }
        Ad4Cache {
            gd: [GridDim::new(); 3],
            slope,
            grids,
        }
    }
}

impl IGrid for Ad4Cache {
    fn eval(&self, model: &Model, v: f64) -> f64 {
        let mut e = 0.0;
        for i in 0..model.num_movable_atoms {
            let a = &model.atoms[i];
            let mut t = a.ad;

            // Map closure types to C
            if t >= AD_TYPE_CG0 && t <= AD_TYPE_CG3 { t = AD_TYPE_C; }
            if t >= AD_TYPE_SIZE { continue; }

            // VdW + H-bonding — grid applies curl internally
            if self.grids[t].initialized() {
                e += self.grids[t].evaluate(&model.coords[i], self.slope, v);
            }

            // Electrostatic
            let elec_idx = AD_TYPE_SIZE;
            if self.grids[elec_idx].initialized() {
                let tmp = self.grids[elec_idx].evaluate(&model.coords[i], self.slope, v);
                e += tmp * a.charge;
            }

            // Desolvation
            let desolv_idx = AD_TYPE_SIZE + 1;
            if self.grids[desolv_idx].initialized() {
                let tmp = self.grids[desolv_idx].evaluate(&model.coords[i], self.slope, v);
                e += tmp * a.charge.abs();
            }
        }
        e
    }

    fn eval_deriv(&self, model: &mut Model, v: f64) -> f64 {
        let mut e = 0.0;
        for i in 0..model.num_movable_atoms {
            let a = &model.atoms[i];
            let mut t = a.ad;
            if t >= AD_TYPE_CG0 && t <= AD_TYPE_CG3 { t = AD_TYPE_C; }
            if t >= AD_TYPE_SIZE { continue; }

            // VdW + H-bonding — grid applies curl internally
            if self.grids[t].initialized() {
                let (energy, deriv) = self.grids[t].evaluate_deriv(&model.coords[i], self.slope, v);
                e += energy;
                model.minus_forces[i] += deriv;
            }

            // Electrostatic
            let elec_idx = AD_TYPE_SIZE;
            if self.grids[elec_idx].initialized() {
                let (energy, deriv) = self.grids[elec_idx].evaluate_deriv(&model.coords[i], self.slope, v);
                e += energy * a.charge;
                model.minus_forces[i] += deriv * a.charge;
            }

            // Desolvation
            let desolv_idx = AD_TYPE_SIZE + 1;
            if self.grids[desolv_idx].initialized() {
                let (energy, deriv) = self.grids[desolv_idx].evaluate_deriv(&model.coords[i], self.slope, v);
                e += energy * a.charge.abs();
                model.minus_forces[i] += deriv * a.charge.abs();
            }
        }
        e
    }
}
