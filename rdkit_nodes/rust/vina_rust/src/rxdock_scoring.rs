//! Scoring functions for rxDock-style docking.
//!
//! Implements VDW 4-8/6-12, H-bond (polar), rotational entropy, constant
//! entropy, cavity restraint, and intramolecular scoring terms.
//! Based on published parameters from Ruiz-Carmona et al. 2014 and
//! Tripos 5.2 force field.

use crate::atom::*;
use crate::common::*;
use crate::model::Model;
use crate::rxdock_atom::*;
use crate::rxdock_cavity::Cavity;

// ─── Receptor Atom Cache ────────────────────────────────────────────────────

/// Pre-processed receptor atom for fast scoring.
/// Stores Tripos type + coordinates for VDW evaluation.
#[derive(Debug, Clone)]
pub struct RxReceptorAtom {
    pub coords: Vec3,
    pub sy: usize,
}

/// Build receptor atom cache from model grid_atoms.
pub fn build_receptor_cache(model: &Model) -> Vec<RxReceptorAtom> {
    model.grid_atoms.iter().map(|a| {
        RxReceptorAtom {
            coords: a.coords,
            sy: ad_to_tripos(a.ad),
        }
    }).collect()
}

// ─── Spatial Grid for Neighbor Lookup ───────────────────────────────────────

/// Simple spatial hash grid for fast neighbor lookups during scoring.
pub struct SpatialGrid {
    cells: Vec<Vec<usize>>,  // atom indices per cell
    origin: Vec3,
    cell_size: f64,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl SpatialGrid {
    /// Build a spatial grid from receptor atoms.
    pub fn new(atoms: &[RxReceptorAtom], cell_size: f64) -> Self {
        if atoms.is_empty() {
            return SpatialGrid {
                cells: Vec::new(),
                origin: Vec3::ZERO,
                cell_size,
                nx: 0, ny: 0, nz: 0,
            };
        }

        // Find bounding box
        let mut min_c = atoms[0].coords;
        let mut max_c = atoms[0].coords;
        for a in atoms {
            min_c = Vec3::new(min_c.x().min(a.coords.x()), min_c.y().min(a.coords.y()), min_c.z().min(a.coords.z()));
            max_c = Vec3::new(max_c.x().max(a.coords.x()), max_c.y().max(a.coords.y()), max_c.z().max(a.coords.z()));
        }

        // Add border
        let border = cell_size * 2.0;
        let origin = Vec3::new(min_c.x() - border, min_c.y() - border, min_c.z() - border);
        let extent = Vec3::new(max_c.x() + border, max_c.y() + border, max_c.z() + border) - origin;

        let nx = (extent.x() / cell_size).ceil() as usize + 1;
        let ny = (extent.y() / cell_size).ceil() as usize + 1;
        let nz = (extent.z() / cell_size).ceil() as usize + 1;

        let mut cells = vec![Vec::new(); nx * ny * nz];

        for (i, a) in atoms.iter().enumerate() {
            let ix = ((a.coords.x() - origin.x()) / cell_size) as usize;
            let iy = ((a.coords.y() - origin.y()) / cell_size) as usize;
            let iz = ((a.coords.z() - origin.z()) / cell_size) as usize;
            if ix < nx && iy < ny && iz < nz {
                cells[ix * ny * nz + iy * nz + iz].push(i);
            }
        }

        SpatialGrid { cells, origin, cell_size, nx, ny, nz }
    }

    /// Get all atom indices within `radius` of `point`.
    pub fn neighbors_within(&self, point: &Vec3, radius: f64) -> Vec<usize> {
        if self.nx == 0 { return Vec::new(); }

        let mut result = Vec::new();
        let r_cells = (radius / self.cell_size).ceil() as isize + 1;

        let cx = ((point.x() - self.origin.x()) / self.cell_size) as isize;
        let cy = ((point.y() - self.origin.y()) / self.cell_size) as isize;
        let cz = ((point.z() - self.origin.z()) / self.cell_size) as isize;

        for ix in (cx - r_cells).max(0)..=(cx + r_cells).min(self.nx as isize - 1) {
            for iy in (cy - r_cells).max(0)..=(cy + r_cells).min(self.ny as isize - 1) {
                for iz in (cz - r_cells).max(0)..=(cz + r_cells).min(self.nz as isize - 1) {
                    let idx = ix as usize * self.ny * self.nz + iy as usize * self.nz + iz as usize;
                    for &ai in &self.cells[idx] {
                        result.push(ai);
                    }
                }
            }
        }

        result
    }
}

// ─── Polar (H-bond) Scoring ─────────────────────────────────────────────────

/// Parameters for the f1 linear ramp function used in polar scoring.
#[derive(Debug, Clone, Copy)]
pub struct RampParams {
    pub dr_min: f64,   // tolerance: score=1.0 if deviation ≤ dr_min
    pub dr_max: f64,   // cutoff: score=0.0 if deviation > dr_max
}

impl RampParams {
    fn new(dr_min: f64, dr_max: f64) -> Self {
        RampParams { dr_min, dr_max }
    }
}

/// Linear ramp function: 1.0 within tolerance, linear decay, 0.0 beyond max.
#[inline(always)]
fn f1_ramp(deviation: f64, p: &RampParams) -> f64 {
    if deviation <= p.dr_min {
        1.0
    } else if deviation >= p.dr_max {
        0.0
    } else {
        1.0 - (deviation - p.dr_min) / (p.dr_max - p.dr_min)
    }
}

/// Parameters for H-bond (polar) scoring function.
#[derive(Debug, Clone)]
pub struct PolarSFParams {
    pub r12_factor: f64,     // VDW radius multiplier (1.0)
    pub r12_increment: f64,  // offset added to VDW sum
    pub dist_ramp: RampParams,
    pub angle1_ramp: RampParams,  // D-H...A angle (reference 180°)
    pub angle2_ramp: RampParams,  // H...A-X angle (reference 180°)
    pub abs_dr12: bool,      // use |DR| (true for attractive)
    pub weight: f64,
    pub range: f64,          // max interaction distance for grid lookup
}

impl PolarSFParams {
    /// C++ rxDock attractive H-bond defaults (intermolecular-indexed.json "polar")
    pub fn attractive() -> Self {
        PolarSFParams {
            r12_factor: 1.0,
            r12_increment: 0.05,
            dist_ramp: RampParams::new(0.25, 0.6),
            angle1_ramp: RampParams::new(30.0, 80.0),
            angle2_ramp: RampParams::new(60.0, 100.0),
            abs_dr12: true,
            weight: 3.4,
            range: 5.31,
        }
    }

    /// C++ rxDock repulsive polar defaults (intermolecular-indexed.json "repul")
    pub fn repulsive() -> Self {
        PolarSFParams {
            r12_factor: 1.0,
            r12_increment: 0.6,
            dist_ramp: RampParams::new(0.25, 1.1),
            angle1_ramp: RampParams::new(30.0, 60.0),
            angle2_ramp: RampParams::new(30.0, 60.0),
            abs_dr12: false,
            weight: 5.0,
            range: 5.32,
        }
    }
}

/// An interaction center for H-bond scoring.
/// For donors: atom_idx = H, parent_idx = N/O (the heavy atom bonded to H)
/// For acceptors: atom_idx = N/O, parent_idx = bonded heavy atom
#[derive(Debug, Clone)]
pub struct InteractionCenter {
    pub atom_idx: usize,
    pub parent_idx: usize,
    pub coords: Vec3,
    pub parent_coords: Vec3,
    pub vdw_radius: f64,
    pub is_donor: bool,
    /// C++ rxDock User1Value: fNeighb × sign × (1 + |charge| × 0.5)
    /// sign = -1 for HBA, +1 for HBD
    pub user1_value: f64,
}

/// Receptor interaction centers for H-bond scoring, with spatial grid.
pub struct PolarGrid {
    pub donors: Vec<InteractionCenter>,
    pub acceptors: Vec<InteractionCenter>,
    pub donor_grid: SpatialGrid,
    pub acceptor_grid: SpatialGrid,
}

impl PolarGrid {
    /// Build H-bond interaction centers from receptor atoms.
    ///
    /// Uses AD types (not Tripos) for HBD/HBA classification:
    /// - Donors: only AD_TYPE_HD (polar H bonded to N/O/S)
    /// - Acceptors: only AD_TYPE_OA, AD_TYPE_NA, AD_TYPE_SA
    ///
    /// Computes User1Value per C++ rxDock SetupPolarSF:
    /// - fNeighb = (nNeighbors_within_5A / 25.0).sqrt()
    /// - charge_factor = sign × (1.0 + |charge| × 0.5)
    /// - user1_value = fNeighb × charge_factor
    pub fn from_receptor(atoms: &[RxReceptorAtom], model: &Model) -> Self {
        let mut donors = Vec::new();
        let mut acceptors = Vec::new();

        // Build spatial grid for neighbor counting (SetupPolarSF radius=5.0)
        let rec_grid = SpatialGrid::new(atoms, 3.0);

        // Scan receptor grid_atoms using AD types for classification
        for (i, a) in model.grid_atoms.iter().enumerate() {
            if ad_is_polar_hbd(a.ad) {
                // AD_TYPE_HD: H-bond donor hydrogen
                let parent_idx = find_closest_heavy_atom(model, i, true);
                if let Some(pi) = parent_idx {
                    let sy = ad_to_tripos(a.ad);
                    // Compute fNeighb: count receptor atoms within 5.0 Å
                    let n_neighbors = rec_grid.neighbors_within(&a.coords, 5.0).len().saturating_sub(1); // exclude self
                    let f_neighb = (n_neighbors as f64 / 25.0).sqrt();
                    // HBD: sign = +1
                    let user1_value = f_neighb * (1.0 + a.charge.abs() * 0.5);

                    donors.push(InteractionCenter {
                        atom_idx: i,
                        parent_idx: pi,
                        coords: a.coords,
                        parent_coords: model.grid_atoms[pi].coords,
                        vdw_radius: TRIPOS_PARAMS[sy].radius,
                        is_donor: true,
                        user1_value,
                    });
                }
            }
            if ad_is_polar_hba(a.ad) {
                // AD_TYPE_OA/NA/SA: H-bond acceptor heavy atom
                let parent_idx = find_closest_heavy_atom(model, i, true);
                if let Some(pi) = parent_idx {
                    let sy = ad_to_tripos(a.ad);
                    let n_neighbors = rec_grid.neighbors_within(&a.coords, 5.0).len().saturating_sub(1);
                    let f_neighb = (n_neighbors as f64 / 25.0).sqrt();
                    // HBA: sign = -1
                    let user1_value = -f_neighb * (1.0 + a.charge.abs() * 0.5);

                    acceptors.push(InteractionCenter {
                        atom_idx: i,
                        parent_idx: pi,
                        coords: a.coords,
                        parent_coords: model.grid_atoms[pi].coords,
                        vdw_radius: TRIPOS_PARAMS[sy].radius,
                        is_donor: false,
                        user1_value,
                    });
                }
            }
        }

        // Build spatial grids from ICs
        let donor_atoms: Vec<RxReceptorAtom> = donors.iter()
            .map(|ic| RxReceptorAtom { coords: ic.coords, sy: 0 })
            .collect();
        let acceptor_atoms: Vec<RxReceptorAtom> = acceptors.iter()
            .map(|ic| RxReceptorAtom { coords: ic.coords, sy: 0 })
            .collect();

        PolarGrid {
            donor_grid: SpatialGrid::new(&donor_atoms, 3.0),
            acceptor_grid: SpatialGrid::new(&acceptor_atoms, 3.0),
            donors,
            acceptors,
        }
    }
}

/// Find the closest heavy atom to atom `idx` in receptor grid_atoms.
fn find_closest_heavy_atom(model: &Model, idx: usize, receptor: bool) -> Option<usize> {
    let atoms = &model.grid_atoms;
    if idx >= atoms.len() { return None; }

    let pos = atoms[idx].coords;
    let mut best_idx = None;
    let mut best_dist = f64::MAX;

    for (i, a) in atoms.iter().enumerate() {
        if i == idx { continue; }
        let sy = ad_to_tripos(a.ad);
        if sy_is_hydrogen(sy) { continue; }
        let d = pos.distance_sqr(&a.coords);
        if d < best_dist && d < 4.0 { // within ~2 Å bond distance
            best_dist = d;
            best_idx = Some(i);
        }
    }
    let _ = receptor; // suppress unused warning
    best_idx
}

/// Calculate angle (in degrees) between three points: A-B-C.
#[inline]
fn angle_abc(a: &Vec3, b: &Vec3, c: &Vec3) -> f64 {
    let ba = *a - *b;
    let bc = *c - *b;
    let dot = ba.dot(&bc);
    let mag = (ba.norm_sqr() * bc.norm_sqr()).sqrt();
    if mag < 1e-10 { return 0.0; }
    let cos_a = (dot / mag).clamp(-1.0, 1.0);
    cos_a.acos().to_degrees()
}

// ─── Scoring Terms ──────────────────────────────────────────────────────────

/// Weights for the combined scoring function.
#[derive(Debug, Clone)]
pub struct RxScoringWeights {
    pub vdw_inter: f64,       // intermolecular VDW weight
    pub vdw_intra: f64,       // intramolecular VDW weight
    pub rot_penalty: f64,     // per-rotatable-bond entropy penalty
    pub const_penalty: f64,   // fixed binding entropy penalty (ConstSF)
    pub cavity: f64,          // cavity restraint weight
    pub polar_attr: f64,      // H-bond attractive weight
    pub polar_repul: f64,     // polar repulsion weight
    pub dihedral: f64,        // dihedral strain weight (C++ rxDock = 0.5)
}

impl Default for RxScoringWeights {
    fn default() -> Self {
        RxScoringWeights {
            vdw_inter: 1.0,
            vdw_intra: 1.0,
            rot_penalty: 1.0,     // C++ rxDock RotSF weight=1.0
            const_penalty: 5.4,   // C++ rxDock ConstSF weight=5.4
            cavity: 1.0,
            polar_attr: 1.0,      // applied via PolarSFParams.weight (3.4)
            polar_repul: 1.0,     // applied via PolarSFParams.weight (5.0)
            dihedral: 0.5,        // C++ rxDock DihedralIntraSF weight=0.5
        }
    }
}

/// Combined rxDock scoring function.
pub struct RxScoringFunction {
    pub vdw_inter_table: TriposVdwTable,  // intermolecular: GOLD well depths, 6-12
    pub vdw_intra_table: TriposVdwTable,  // intramolecular: Tripos well depths, 6-12
    pub weights: RxScoringWeights,
    pub receptor_atoms: Vec<RxReceptorAtom>,
    pub receptor_grid: SpatialGrid,
    pub cavity: Option<Cavity>,
    pub polar_grid: Option<PolarGrid>,
    pub polar_attr_params: PolarSFParams,
    pub polar_repul_params: PolarSFParams,
}

impl RxScoringFunction {
    /// Create a new scoring function from receptor atoms and optional cavity.
    pub fn new(
        receptor_atoms: Vec<RxReceptorAtom>,
        cavity: Option<Cavity>,
        weights: RxScoringWeights,
    ) -> Self {
        // C++ intermolecular: 6-12, GOLD well depths, ecut=120
        let vdw_inter_table = TriposVdwTable::new_gold(false, 1.5, 120.0);
        // C++ intramolecular: 6-12, Tripos well depths, ecut=120
        let vdw_intra_table = TriposVdwTable::new(false, 1.5, 120.0);
        let receptor_grid = SpatialGrid::new(&receptor_atoms, 3.0);

        RxScoringFunction {
            vdw_inter_table,
            vdw_intra_table,
            weights,
            receptor_atoms,
            receptor_grid,
            cavity,
            polar_grid: None,
            polar_attr_params: PolarSFParams::attractive(),
            polar_repul_params: PolarSFParams::repulsive(),
        }
    }

    /// Initialize H-bond scoring from the model (call after receptor is loaded).
    pub fn init_polar(&mut self, model: &Model) {
        self.polar_grid = Some(PolarGrid::from_receptor(&self.receptor_atoms, model));
    }

    /// Score the current model conformation.
    /// Returns total score (lower = better, like Vina).
    pub fn score(&self, model: &Model) -> f64 {
        let mut total = 0.0;

        for lig in &model.ligands {
            total += self.score_vdw_inter(model, lig.begin, lig.end);
            total += self.score_vdw_intra(model, lig.begin, lig.end, &lig.pairs);
            total += self.score_rot_entropy(lig.degrees_of_freedom);
            total += self.score_const();
            total += self.score_cavity(model, lig.begin, lig.end);
            total += self.score_polar_inter(model, lig.begin, lig.end);
            total += self.score_polar_repul(model, lig.begin, lig.end);
            total += crate::rxdock_dihedral::score_dihedral_intra(model, lig.begin, lig.end)
                     * self.weights.dihedral;
        }

        total
    }

    /// Score with individual term breakdown.
    pub fn score_terms(&self, model: &Model) -> RxScoreTerms {
        let mut terms = RxScoreTerms::default();

        for lig in &model.ligands {
            terms.vdw_inter += self.score_vdw_inter(model, lig.begin, lig.end);
            terms.vdw_intra += self.score_vdw_intra(model, lig.begin, lig.end, &lig.pairs);
            terms.rot_entropy += self.score_rot_entropy(lig.degrees_of_freedom);
            terms.const_penalty += self.score_const();
            terms.cavity += self.score_cavity(model, lig.begin, lig.end);
            terms.polar_attr += self.score_polar_inter(model, lig.begin, lig.end);
            terms.polar_repul += self.score_polar_repul(model, lig.begin, lig.end);
            terms.dihedral += crate::rxdock_dihedral::score_dihedral_intra(model, lig.begin, lig.end)
                              * self.weights.dihedral;
        }

        terms.total = terms.vdw_inter + terms.vdw_intra + terms.rot_entropy
            + terms.const_penalty + terms.cavity + terms.polar_attr + terms.polar_repul
            + terms.dihedral;
        terms
    }

    // ─── Individual Scoring Terms ───────────────────────────────────────

    /// Intermolecular VDW: ligand vs receptor.
    /// C++ rxDock includes ALL atoms (including H); HBD-HBA pairs have kij=0 in the table.
    fn score_vdw_inter(&self, model: &Model, lig_begin: usize, lig_end: usize) -> f64 {
        let mut e = 0.0;
        let rmax = self.vdw_inter_table.rmax;

        for i in lig_begin..lig_end {
            let lig_atom = &model.atoms[i];
            let lig_sy = lig_atom.sy;
            let lig_coords = model.coords[i];

            let neighbors = self.receptor_grid.neighbors_within(&lig_coords, rmax);
            for &ri in &neighbors {
                let rec = &self.receptor_atoms[ri];
                let r_sq = lig_coords.distance_sqr(&rec.coords);
                e += self.vdw_inter_table.eval(lig_sy, rec.sy, r_sq);
            }
        }

        e * self.weights.vdw_inter
    }

    /// Intramolecular VDW: ligand internal (variable-mobility pairs only).
    /// C++ rxDock includes ALL atoms (including H); HBD-HBA pairs have kij=0 in the table.
    fn score_vdw_intra(&self, model: &Model, _lig_begin: usize, _lig_end: usize, pairs: &[InteractingPair]) -> f64 {
        let mut e = 0.0;
        let rmax_sq = self.vdw_intra_table.rmax * self.vdw_intra_table.rmax;

        for pair in pairs {
            let a = &model.atoms[pair.a];
            let b = &model.atoms[pair.b];

            let r_sq = model.coords[pair.a].distance_sqr(&model.coords[pair.b]);
            if r_sq < rmax_sq {
                e += self.vdw_intra_table.eval(a.sy, b.sy, r_sq);
            }
        }

        e * self.weights.vdw_intra
    }

    /// Rotational entropy penalty: 1.0 per rotatable bond (C++ rxDock RotSF).
    fn score_rot_entropy(&self, degrees_of_freedom: usize) -> f64 {
        degrees_of_freedom as f64 * self.weights.rot_penalty
    }

    /// Constant entropy penalty for ligand binding (C++ rxDock ConstSF).
    fn score_const(&self) -> f64 {
        self.weights.const_penalty
    }

    /// Cavity restraint: penalize ligand heavy atoms outside the cavity.
    fn score_cavity(&self, model: &Model, lig_begin: usize, lig_end: usize) -> f64 {
        let cavity = match &self.cavity {
            Some(c) => c,
            None => return 0.0,
        };

        let mut penalty = 0.0;

        for i in lig_begin..lig_end {
            if model.atoms[i].el == EL_TYPE_H { continue; }
            let p = model.coords[i];

            let margin = 1.5;
            let outside = p.x() < cavity.min_coord.x() - margin ||
                          p.x() > cavity.max_coord.x() + margin ||
                          p.y() < cavity.min_coord.y() - margin ||
                          p.y() > cavity.max_coord.y() + margin ||
                          p.z() < cavity.min_coord.z() - margin ||
                          p.z() > cavity.max_coord.z() + margin;

            if outside {
                let d = p - cavity.center;
                penalty += d.norm_sqr().sqrt();
            }
        }

        penalty * self.weights.cavity
    }

    /// Attractive H-bond scoring: ligand HBD ↔ receptor HBA and vice versa.
    ///
    /// C++ rxDock PolarIdxSF formula per pair:
    ///   score_pair = User1Value(lig) × User1Value(rec) × f_dist × f_angle1 × f_angle2
    ///
    /// Inside PolarScore(), result is multiplied by receptor's User1Value.
    /// In InterScore(), the accumulated sum is then multiplied by ligand's User1Value.
    fn score_polar_inter(&self, model: &Model, lig_begin: usize, lig_end: usize) -> f64 {
        let polar_grid = match &self.polar_grid {
            Some(pg) => pg,
            None => return 0.0,
        };

        let params = &self.polar_attr_params;
        let mut score = 0.0;

        for i in lig_begin..lig_end {
            let lig_atom = &model.atoms[i];
            let lig_coords = model.coords[i];

            // Ligand donor (AD_TYPE_HD) ↔ receptor acceptors
            if ad_is_polar_hbd(lig_atom.ad) {
                let parent_idx = find_closest_ligand_heavy(model, i, lig_begin, lig_end);
                if parent_idx.is_none() { continue; }
                let parent_coords = model.coords[parent_idx.unwrap()];
                // Ligand HBD User1Value: sign=+1, fNeighb=1.0 (ligand)
                let lig_user1 = 1.0 + lig_atom.charge.abs() * 0.5;

                // C++ PolarScore accumulates: Σ(rec.user1_value × f_geom)
                let mut s_for_lig = 0.0;
                let neighbors = polar_grid.acceptor_grid.neighbors_within(&lig_coords, params.range);
                for &ni in &neighbors {
                    let acc = &polar_grid.acceptors[ni];
                    let f = score_polar_pair(
                        &lig_coords, &parent_coords, TRIPOS_PARAMS[lig_atom.sy].radius,
                        &acc.coords, &acc.parent_coords, acc.vdw_radius,
                        params,
                    );
                    // C++ PolarScore: s += receptor.User1Value * f
                    s_for_lig += acc.user1_value * f;
                }
                // C++ InterScore: s *= lig.User1Value
                score += s_for_lig * lig_user1;
            }

            // Ligand acceptor ↔ receptor donor H
            if is_ligand_hba(lig_atom) {
                let parent_idx = find_closest_ligand_heavy(model, i, lig_begin, lig_end);
                if parent_idx.is_none() { continue; }
                let parent_coords = model.coords[parent_idx.unwrap()];
                // Ligand HBA User1Value: sign=-1, fNeighb=1.0
                let lig_user1 = -(1.0 + lig_atom.charge.abs() * 0.5);

                let mut s_for_lig = 0.0;
                let neighbors = polar_grid.donor_grid.neighbors_within(&lig_coords, params.range);
                for &ni in &neighbors {
                    let don = &polar_grid.donors[ni];
                    let f = score_polar_pair(
                        &don.coords, &don.parent_coords, don.vdw_radius,
                        &lig_coords, &parent_coords, TRIPOS_PARAMS[lig_atom.sy].radius,
                        params,
                    );
                    // C++ PolarScore: s += receptor.User1Value * f
                    s_for_lig += don.user1_value * f;
                }
                // C++ InterScore: s *= lig.User1Value
                score += s_for_lig * lig_user1;
            }
        }

        // Product of User1Values gives sign:
        //   donor(+) × acceptor(-) = negative (attractive, favorable)
        score * params.weight * self.weights.polar_attr
    }

    /// Repulsive polar: same-type proximity penalty (HBD↔HBD, HBA↔HBA).
    /// C++ PolarIdxSF formula: User1Value(lig) × User1Value(rec) × f_geom
    /// Same-type product is positive (penalty).
    fn score_polar_repul(&self, model: &Model, lig_begin: usize, lig_end: usize) -> f64 {
        let polar_grid = match &self.polar_grid {
            Some(pg) => pg,
            None => return 0.0,
        };

        let params = &self.polar_repul_params;
        let mut score = 0.0;

        for i in lig_begin..lig_end {
            let lig_atom = &model.atoms[i];
            let lig_coords = model.coords[i];

            // Ligand donor ↔ receptor donors (repulsive same-type)
            if ad_is_polar_hbd(lig_atom.ad) {
                let parent_idx = find_closest_ligand_heavy(model, i, lig_begin, lig_end);
                if parent_idx.is_none() { continue; }
                let parent_coords = model.coords[parent_idx.unwrap()];
                // Ligand HBD User1Value: sign=+1
                let lig_user1 = 1.0 + lig_atom.charge.abs() * 0.5;

                let mut s_for_lig = 0.0;
                let neighbors = polar_grid.donor_grid.neighbors_within(&lig_coords, params.range);
                for &ni in &neighbors {
                    let don = &polar_grid.donors[ni];
                    let f = score_polar_pair(
                        &lig_coords, &parent_coords, TRIPOS_PARAMS[lig_atom.sy].radius,
                        &don.coords, &don.parent_coords, don.vdw_radius,
                        params,
                    );
                    s_for_lig += don.user1_value * f;
                }
                score += s_for_lig * lig_user1;
            }

            // Ligand acceptor ↔ receptor acceptors (repulsive same-type)
            if is_ligand_hba(lig_atom) {
                let parent_idx = find_closest_ligand_heavy(model, i, lig_begin, lig_end);
                if parent_idx.is_none() { continue; }
                let parent_coords = model.coords[parent_idx.unwrap()];
                // Ligand HBA User1Value: sign=-1
                let lig_user1 = -(1.0 + lig_atom.charge.abs() * 0.5);

                let mut s_for_lig = 0.0;
                let neighbors = polar_grid.acceptor_grid.neighbors_within(&lig_coords, params.range);
                for &ni in &neighbors {
                    let acc = &polar_grid.acceptors[ni];
                    let f = score_polar_pair(
                        &lig_coords, &parent_coords, TRIPOS_PARAMS[lig_atom.sy].radius,
                        &acc.coords, &acc.parent_coords, acc.vdw_radius,
                        params,
                    );
                    s_for_lig += acc.user1_value * f;
                }
                score += s_for_lig * lig_user1;
            }
        }

        // Same-type: User1Value(+) × User1Value(+) or User1Value(-) × User1Value(-) = positive
        score * params.weight * self.weights.polar_repul
    }
}

/// Check if a ligand atom is an H-bond acceptor.
///
/// Uses AD types set during SDF parsing:
/// - AD_TYPE_OA: oxygen acceptor (always HBA)
/// - AD_TYPE_NA: nitrogen acceptor (aromatic/sp2 N, HBA)
/// - AD_TYPE_SA: sulfur acceptor (HBA)
/// - AD_TYPE_N: sp3 nitrogen, HBA only if <4 bonds (has lone pair)
#[inline]
fn is_ligand_hba(atom: &Atom) -> bool {
    match atom.ad {
        AD_TYPE_OA | AD_TYPE_NA | AD_TYPE_SA => true,
        AD_TYPE_N => atom.bonds.len() < 4, // sp3 amine with available lone pair
        _ => false,
    }
}

/// Find the closest heavy atom to ligand atom `idx` within the ligand range.
fn find_closest_ligand_heavy(model: &Model, idx: usize, lig_begin: usize, lig_end: usize) -> Option<usize> {
    let pos = model.coords[idx];
    let mut best_idx = None;
    let mut best_dist = f64::MAX;

    for i in lig_begin..lig_end {
        if i == idx { continue; }
        if sy_is_hydrogen(model.atoms[i].sy) { continue; }
        let d = pos.distance_sqr(&model.coords[i]);
        if d < best_dist && d < 4.0 {
            best_dist = d;
            best_idx = Some(i);
        }
    }
    best_idx
}

/// Score a single polar pair using the f1 ramp function.
/// atom1/parent1 is the "donor" side, atom2/parent2 is the "acceptor" side.
fn score_polar_pair(
    atom1: &Vec3, parent1: &Vec3, radius1: f64,
    atom2: &Vec3, parent2: &Vec3, radius2: f64,
    params: &PolarSFParams,
) -> f64 {
    // Distance scoring
    let r = (*atom1 - *atom2).norm_sqr().sqrt();
    let r12 = params.r12_factor * (radius1 + radius2) + params.r12_increment;
    let dr = if params.abs_dr12 { (r - r12).abs() } else { r - r12 };

    let f_r = f1_ramp(dr, &params.dist_ramp);
    if f_r <= 0.0 { return 0.0; }

    // Angle 1: parent1-atom1-atom2 (D-H...A angle, ideal 180°)
    let angle1 = angle_abc(parent1, atom1, atom2);
    let da1 = (angle1 - 180.0).abs();
    let f_a1 = f1_ramp(da1, &params.angle1_ramp);
    if f_a1 <= 0.0 { return 0.0; }

    // Angle 2: atom1-atom2-parent2 (H...A-X angle, ideal 180°)
    let angle2 = angle_abc(atom1, atom2, parent2);
    let da2 = (angle2 - 180.0).abs();
    let f_a2 = f1_ramp(da2, &params.angle2_ramp);
    if f_a2 <= 0.0 { return 0.0; }

    f_r * f_a1 * f_a2
}

/// Individual score term breakdown.
#[derive(Debug, Clone, Default)]
pub struct RxScoreTerms {
    pub vdw_inter: f64,
    pub vdw_intra: f64,
    pub rot_entropy: f64,
    pub const_penalty: f64,
    pub polar_attr: f64,
    pub polar_repul: f64,
    pub cavity: f64,
    pub dihedral: f64,
    pub total: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_scoring() {
        let sf = RxScoringFunction::new(Vec::new(), None, RxScoringWeights::default());
        let model = Model::new();
        let score = sf.score(&model);
        // ConstSF = 5.4, no ligands so it's still 0 (score() iterates over ligands)
        assert_eq!(score, 0.0, "Empty model should score 0");
    }

    #[test]
    fn test_rot_entropy() {
        let sf = RxScoringFunction::new(Vec::new(), None, RxScoringWeights::default());
        let penalty = sf.score_rot_entropy(5);
        assert!((penalty - 5.0).abs() < 0.001, "5 DOF * 1.0 = 5.0 (C++ rxDock weight)");
    }

    #[test]
    fn test_const_penalty() {
        let sf = RxScoringFunction::new(Vec::new(), None, RxScoringWeights::default());
        let penalty = sf.score_const();
        assert!((penalty - 5.4).abs() < 0.001, "ConstSF should be 5.4");
    }

    #[test]
    fn test_f1_ramp() {
        let p = RampParams::new(0.25, 0.6);
        assert!((f1_ramp(0.0, &p) - 1.0).abs() < 1e-10, "Below min: 1.0");
        assert!((f1_ramp(0.25, &p) - 1.0).abs() < 1e-10, "At min: 1.0");
        assert!((f1_ramp(0.6, &p) - 0.0).abs() < 1e-10, "At max: 0.0");
        assert!((f1_ramp(1.0, &p) - 0.0).abs() < 1e-10, "Beyond max: 0.0");
        // Midpoint: 0.425 → 1.0 - (0.425-0.25)/(0.6-0.25) = 1.0 - 0.5 = 0.5
        assert!((f1_ramp(0.425, &p) - 0.5).abs() < 0.001, "Midpoint: 0.5");
    }

    #[test]
    fn test_polar_pair_ideal() {
        // Ideal H-bond geometry: D-H...A linear (180°), at ideal distance
        let params = PolarSFParams::attractive();
        // H at origin, parent D at (-1, 0, 0), acceptor A at (2.05, 0, 0), acceptor parent at (3, 0, 0)
        // R = 2.05 Å, R12 = 1.0*(1.0+1.52)+0.05 = 2.57
        // DR = |2.05-2.57| = 0.52 → f1(0.52, 0.25, 0.6) = 1.0 - (0.52-0.25)/0.35 = 0.229
        // angle1 = 180° (linear) → da1 = 0 → f_a1 = 1.0
        // angle2 = 180° (linear) → da2 = 0 → f_a2 = 1.0
        let h = Vec3::new(0.0, 0.0, 0.0);
        let d = Vec3::new(-1.0, 0.0, 0.0);
        let a = Vec3::new(2.05, 0.0, 0.0);
        let ax = Vec3::new(3.0, 0.0, 0.0);
        let s = score_polar_pair(&h, &d, 1.0, &a, &ax, 1.52, &params);
        assert!(s > 0.0, "Ideal geometry should give positive score (before sign flip)");
    }

    #[test]
    fn test_spatial_grid() {
        let atoms = vec![
            RxReceptorAtom { coords: Vec3::new(0.0, 0.0, 0.0), sy: SY_C3 },
            RxReceptorAtom { coords: Vec3::new(5.0, 0.0, 0.0), sy: SY_C3 },
            RxReceptorAtom { coords: Vec3::new(20.0, 0.0, 0.0), sy: SY_C3 },
        ];
        let grid = SpatialGrid::new(&atoms, 3.0);
        let near = grid.neighbors_within(&Vec3::new(0.0, 0.0, 0.0), 4.0);
        assert!(near.contains(&0), "Atom 0 should be found");
        assert!(!near.contains(&2), "Atom 2 at 20 Å should not be found within 4.0 of origin");
    }
}
