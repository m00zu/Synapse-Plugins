use crate::atom::*;
use crate::bfgs;
use crate::cache::IGrid;
use crate::common::*;
use crate::conf::*;
use crate::precalculate::*;
use crate::tree::*;
use crate::visited::{Visited, VisitedScratch};

// ─── Ligand ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Ligand {
    pub tree: RigidBody,
    pub degrees_of_freedom: usize,
    pub pairs: Vec<InteractingPair>,
    pub begin: usize,
    pub end: usize,
    pub cont: Vec<String>, // PDBQT context lines for output
}

impl Ligand {
    pub fn new(tree: RigidBody, dof: usize) -> Self {
        Ligand {
            tree,
            degrees_of_freedom: dof,
            pairs: Vec::new(),
            begin: 0,
            end: 0,
            cont: Vec::new(),
        }
    }
}

// ─── FlexibleResidueWrapper ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FlexResidueWrapper {
    pub tree: FlexResidue,
    pub begin: usize,
    pub end: usize,
    pub cont: Vec<String>,
}

// ─── Model ─────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Model {
    // Receptor (fixed)
    pub grid_atoms: Vec<Atom>,

    // Movable atoms (ligands + flex)
    pub atoms: Vec<Atom>,
    pub internal_coords: Vec<Vec3>,  // relative coordinates for tree transforms
    pub coords: Vec<Vec3>,           // current lab coordinates
    pub minus_forces: Vec<Vec3>,     // negative forces (gradients)

    // Ligands
    pub ligands: Vec<Ligand>,

    // Flexible residues
    pub flex: Vec<FlexResidueWrapper>,

    // Interaction pairs
    pub other_pairs: Vec<InteractingPair>,   // flex-flex
    pub inter_pairs: Vec<InteractingPair>,   // ligand-flex, ligand-ligand
    pub glue_pairs: Vec<InteractingPair>,    // macrocycle closure

    pub num_movable_atoms: usize,
    pub atom_typing: AtomTyping,
    /// Smina-specific: when true, terminal heavy atoms (those with only 1 heavy
    /// neighbor, e.g. -OH oxygen) do NOT contribute their rotors to num_tors.
    /// Default false (Vina/QVina2 behavior). Set true for smina compatibility.
    pub fixed_rotable_hydrogens: bool,
}

impl Model {
    pub fn new() -> Self {
        Model {
            grid_atoms: Vec::new(),
            atoms: Vec::new(),
            internal_coords: Vec::new(),
            coords: Vec::new(),
            minus_forces: Vec::new(),
            ligands: Vec::new(),
            flex: Vec::new(),
            other_pairs: Vec::new(),
            inter_pairs: Vec::new(),
            glue_pairs: Vec::new(),
            num_movable_atoms: 0,
            atom_typing: AtomTyping::XS,
            fixed_rotable_hydrogens: false,
        }
    }

    pub fn get_size(&self) -> ConfSize {
        ConfSize {
            ligands: self.ligands.iter().map(|l| l.degrees_of_freedom).collect(),
            flex: self.flex.iter().map(|f| f.tree.children.len() + 1).collect(),
        }
    }

    /// Update coordinates from conformation via forward kinematics
    #[inline(always)]
    pub fn set(&mut self, c: &Conf) {
        for (i, lig) in self.ligands.iter_mut().enumerate() {
            lig.tree.set_conf(&self.internal_coords, &mut self.coords, &c.ligands[i]);
        }
        for (i, flx) in self.flex.iter_mut().enumerate() {
            flx.tree.set_conf(&self.internal_coords, &mut self.coords, &c.flex[i]);
        }
    }

    /// Compute gyration radius for ligand i
    pub fn gyration_radius(&self, lig_idx: usize) -> f64 {
        let lig = &self.ligands[lig_idx];
        if lig.begin >= lig.end { return 0.0; }

        let n = (lig.end - lig.begin) as f64;
        let mut center = Vec3::ZERO;
        for i in lig.begin..lig.end {
            center += self.coords[i];
        }
        center *= 1.0 / n;

        let mut sum = 0.0;
        for i in lig.begin..lig.end {
            sum += self.coords[i].distance_sqr(&center);
        }
        (sum / n).sqrt()
    }

    #[inline]
    fn num_bonded_heavy_atoms(&self, atom_idx: usize) -> usize {
        let atom = &self.atoms[atom_idx];
        let mut acc = 0usize;
        for b in &atom.bonds {
            if b.connected_atom_index.in_grid { continue; }
            let j = b.connected_atom_index.i;
            if j < self.atoms.len() && !self.atoms[j].is_hydrogen() {
                acc += 1;
            }
        }
        acc
    }

    #[inline]
    fn atom_rotors(&self, atom_idx: usize) -> usize {
        let atom = &self.atoms[atom_idx];
        let mut acc = 0usize;
        for b in &atom.bonds {
            if b.connected_atom_index.in_grid { continue; }
            let j = b.connected_atom_index.i;
            if j >= self.atoms.len() { continue; }
            let other = &self.atoms[j];
            if b.rotatable && !other.is_hydrogen() && self.num_bonded_heavy_atoms(j) > 1 {
                // Smina extra check: atom i itself must also have >1 heavy neighbor.
                // This excludes terminal heavy atoms like -OH oxygen from contributing
                // their rotors to num_tors. See smina terms.cpp atom_rotors().
                if self.fixed_rotable_hydrogens && self.num_bonded_heavy_atoms(atom_idx) <= 1 {
                    continue;
                }
                acc += 1;
            }
        }
        acc
    }

    /// Match C++ conf_independent_inputs::num_tors computation used by QVina2/Vina reporting.
    pub fn conf_independent_num_tors(&self) -> f64 {
        let mut num_tors = 0.0_f64;
        for lig in &self.ligands {
            for i in lig.begin..lig.end {
                if i >= self.atoms.len() { continue; }
                if self.atoms[i].is_hydrogen() { continue; }
                num_tors += 0.5 * self.atom_rotors(i) as f64;
            }
        }
        num_tors
    }

    /// Get heavy (non-hydrogen) atom coordinates for movable atoms
    pub fn get_heavy_atom_movable_coords(&self) -> Vec<Vec3> {
        let mut result = Vec::new();
        for i in 0..self.num_movable_atoms {
            if !self.atoms[i].is_hydrogen() {
                result.push(self.coords[i]);
            }
        }
        result
    }

    /// Evaluate intramolecular energy (within each ligand)
    pub fn eval_intra(&self, p: &PrecalculateByAtom, v: f64) -> f64 {
        let mut e = 0.0;
        let cutoff_sqr = p.cutoff_sqr();
        let nat = num_atom_types(p.atom_typing());

        // Mark ligand-owned movable atoms so we can select flex atoms quickly.
        let mut is_ligand_atom = vec![false; self.num_movable_atoms];
        for lig in &self.ligands {
            let end = lig.end.min(self.num_movable_atoms);
            for i in lig.begin.min(end)..end {
                is_ligand_atom[i] = true;
            }
        }

        // Internal for each ligand.
        for lig in &self.ligands {
            for pair in &lig.pairs {
                let r2 = self.coords[pair.a].distance_sqr(&self.coords[pair.b]);
                if r2 < cutoff_sqr {
                    let mut tmp = p.eval_fast_by_index(pair.type_pair_index, r2);
                    curl(&mut tmp, v);
                    e += tmp;
                }
            }
        }

        // Flex-rigid interactions.
        for i in 0..self.num_movable_atoms {
            if is_ligand_atom[i] {
                continue;
            }
            let a = &self.atoms[i];
            let t1 = a.get_type(p.atom_typing());
            if t1 >= nat {
                continue;
            }
            for b in &self.grid_atoms {
                let t2 = b.get_type(p.atom_typing());
                if t2 >= nat {
                    continue;
                }
                let r2 = self.coords[i].distance_sqr(&b.coords);
                if r2 < cutoff_sqr {
                    let mut tmp = p.eval_fast_by_types(t1, t2, r2);
                    curl(&mut tmp, v);
                    e += tmp;
                }
            }
        }

        // Flex-flex interactions.
        for pair in &self.other_pairs {
            let a_is_lig = pair.a < is_ligand_atom.len() && is_ligand_atom[pair.a];
            let b_is_lig = pair.b < is_ligand_atom.len() && is_ligand_atom[pair.b];
            if a_is_lig || b_is_lig {
                continue;
            }
            let r2 = self.coords[pair.a].distance_sqr(&self.coords[pair.b]);
            if r2 < cutoff_sqr {
                let mut tmp = p.eval_fast_by_index(pair.type_pair_index, r2);
                curl(&mut tmp, v);
                e += tmp;
            }
        }

        e
    }

    /// Evaluate intermolecular pair energy
    fn eval_pairs(&self, p: &PrecalculateByAtom, v: f64, pairs: &[InteractingPair], with_max_cutoff: bool) -> f64 {
        let cutoff_sqr = if with_max_cutoff { p.max_cutoff_sqr() } else { p.cutoff_sqr() };
        let mut e = 0.0;
        for pair in pairs {
            let r2 = self.coords[pair.a].distance_sqr(&self.coords[pair.b]);
            if r2 < cutoff_sqr {
                let mut tmp = p.eval_fast_by_index(pair.type_pair_index, r2);
                curl(&mut tmp, v);
                e += tmp;
            }
        }
        e
    }

    /// Evaluate total energy without derivatives (using eval_fast for all pairs).
    /// Matches C++ model::eval — used for final energy re-evaluation (eval_adjusted).
    pub fn eval(&self, p: &PrecalculateByAtom, ig: &dyn IGrid, v: &Vec3) -> f64 {
        // Grid/receptor-ligand interactions (energy only)
        let mut e = ig.eval(self, v[1]);

        // Intramolecular (within each ligand) — eval_fast
        for lig in &self.ligands {
            e += self.eval_pairs(p, v[0], &lig.pairs, false);
        }

        // Intermolecular pairs — eval_fast
        e += self.eval_pairs(p, v[2], &self.inter_pairs, false);

        // Other pairs (flex-flex) — eval_fast
        e += self.eval_pairs(p, v[2], &self.other_pairs, false);

        // Glue pairs (macrocycle, max cutoff) — eval_fast
        e += self.eval_pairs(p, v[2], &self.glue_pairs, true);

        e
    }

    /// Evaluate pair energy with derivatives (delegates to free function)
    fn eval_pairs_deriv_split(
        coords: &[Vec3],
        p: &PrecalculateByAtom,
        v: f64,
        pairs: &[InteractingPair],
        forces: &mut [Vec3],
        with_max_cutoff: bool,
    ) -> f64 {
        let cutoff_sqr = if with_max_cutoff { p.max_cutoff_sqr() } else { p.cutoff_sqr() };
        let mut e = 0.0;
        for pair in pairs {
            let r_vec = coords[pair.b] - coords[pair.a];
            let r2 = r_vec.norm_sqr();
            if r2 < cutoff_sqr {
                let (mut energy, dor) = p.eval_deriv_by_index(pair.type_pair_index, r2);
                let mut force = r_vec * dor;
                curl_with_deriv(&mut energy, &mut force, v);
                e += energy;
                forces[pair.a] -= force;
                forces[pair.b] += force;
            }
        }
        e
    }

    /// Full energy evaluation with derivatives
    pub fn eval_deriv(
        &mut self,
        p: &PrecalculateByAtom,
        ig: &dyn IGrid,
        v: &Vec3,  // [intra_weight, grid_weight, inter_weight]
        g: &mut Change,
    ) -> f64 {
        // Zero forces
        for f in self.minus_forces.iter_mut() {
            *f = Vec3::ZERO;
        }

        // Grid interactions (receptor-ligand)
        let mut e = ig.eval_deriv(self, v[1]);

        // Intramolecular (within each ligand)
        for lig in &self.ligands {
            for pair in &lig.pairs {
                let r_vec = self.coords[pair.b] - self.coords[pair.a];
                let r2 = r_vec.norm_sqr();
                if r2 < p.cutoff_sqr() {
                    let (mut energy, dor) = p.eval_deriv_by_index(pair.type_pair_index, r2);
                    let mut force = r_vec * dor;
                    curl_with_deriv(&mut energy, &mut force, v[0]);
                    e += energy;
                    self.minus_forces[pair.a] -= force;
                    self.minus_forces[pair.b] += force;
                }
            }
        }

        // Avoid cloning large pair vectors in this hot path.
        let coords = &self.coords;
        let minus_forces = &mut self.minus_forces;

        // Intermolecular pairs
        if !self.inter_pairs.is_empty() {
            e += Model::eval_pairs_deriv_split(coords, p, v[2], &self.inter_pairs, minus_forces, false);
        }

        // Other pairs (flex-flex)
        if !self.other_pairs.is_empty() {
            e += Model::eval_pairs_deriv_split(coords, p, v[2], &self.other_pairs, minus_forces, false);
        }

        // Glue pairs (macrocycle, max cutoff)
        if !self.glue_pairs.is_empty() {
            e += Model::eval_pairs_deriv_split(coords, p, v[2], &self.glue_pairs, minus_forces, true);
        }

        // Convert forces to gradient via tree derivative (reverse kinematics)
        for (i, lig) in self.ligands.iter().enumerate() {
            lig.tree.derivative(&self.coords, &self.minus_forces, &mut g.ligands[i]);
        }
        for (i, flx) in self.flex.iter().enumerate() {
            flx.tree.derivative(&self.coords, &self.minus_forces, &mut g.flex[i]);
        }

        e
    }

    /// Evaluate and optimize using quasi-newton (BFGS)
    pub fn quasi_newton_optimize(
        &mut self,
        p: &PrecalculateByAtom,
        ig: &dyn IGrid,
        out: &mut OutputType,
        g: &mut Change,
        cap: &Vec3,
        evalcount: &mut i32,
        max_steps: u32,
        scratch: &mut bfgs::BfgsScratch,
    ) {
        let v = *cap;
        let p_ref = p;
        let ig_ref = ig as *const dyn IGrid;

        // Create closure that captures model + scoring for BFGS
        let model_ptr = self as *mut Model;
        let mut eval_fn = |conf: &Conf, grad: &mut Change| -> f64 {
            unsafe {
                let model = &mut *model_ptr;
                model.set(conf);
                model.eval_deriv(p_ref, &*ig_ref, &v, grad)
            }
        };

        let res = bfgs::bfgs(&mut eval_fn, &mut out.c, g, max_steps, evalcount, scratch);
        self.set(&out.c);
        out.e = res;
    }

    /// QVina2 variant: optimize with per-thread Visited history database
    pub fn quasi_newton_optimize_qvina2(
        &mut self,
        p: &PrecalculateByAtom,
        ig: &dyn IGrid,
        out: &mut OutputType,
        g: &mut Change,
        cap: &Vec3,
        evalcount: &mut i32,
        max_steps: u32,
        visited: &mut Visited,
        visited_scratch: &mut VisitedScratch,
        scratch: &mut bfgs::BfgsScratch,
    ) {
        let v = *cap;
        let p_ref = p;
        let ig_ref = ig as *const dyn IGrid;

        let model_ptr = self as *mut Model;
        let mut eval_fn = |conf: &Conf, grad: &mut Change| -> f64 {
            unsafe {
                let model = &mut *model_ptr;
                model.set(conf);
                model.eval_deriv(p_ref, &*ig_ref, &v, grad)
            }
        };

        let res = bfgs::bfgs_qvina2(&mut eval_fn, &mut out.c, g, max_steps, evalcount, visited, visited_scratch, scratch);
        self.set(&out.c);
        out.e = res;
    }

    /// Get number of movable atoms (for QVina2 heuristics)
    pub fn num_movable_atoms(&self) -> usize {
        self.num_movable_atoms
    }

}

impl Default for Model {
    fn default() -> Self { Self::new() }
}
