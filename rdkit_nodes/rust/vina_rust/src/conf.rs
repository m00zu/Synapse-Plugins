use crate::common::*;
use rand::Rng;
use std::f64::consts::PI;

// ─── ConfSize ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ConfSize {
    pub ligands: Vec<usize>,  // torsion count per ligand
    pub flex: Vec<usize>,     // torsion count per flexible residue
}

impl ConfSize {
    pub fn num_degrees_of_freedom(&self) -> usize {
        let lig_tors: usize = self.ligands.iter().sum();
        let flex_tors: usize = self.flex.iter().sum();
        lig_tors + flex_tors + 6 * self.ligands.len()
    }
}

// ─── Scale ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Scale {
    pub position: f64,
    pub orientation: f64,
    pub torsion: f64,
}

// ─── Rigid ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RigidChange {
    pub position: Vec3,
    pub orientation: Vec3,
}

impl RigidChange {
    pub fn new() -> Self {
        RigidChange {
            position: Vec3::ZERO,
            orientation: Vec3::ZERO,
        }
    }
}

impl Default for RigidChange {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct RigidConf {
    pub position: Vec3,
    pub orientation: Quaternion,
}

impl RigidConf {
    pub fn new() -> Self {
        RigidConf {
            position: Vec3::ZERO,
            orientation: Quaternion::IDENTITY,
        }
    }

    pub fn set_to_null(&mut self) {
        self.position = Vec3::ZERO;
        self.orientation = Quaternion::IDENTITY;
    }

    pub fn increment(&mut self, c: &RigidChange, factor: f64) {
        self.position += c.position * factor;
        let rotation = c.orientation * factor;
        self.orientation.increment(&rotation);
    }

    pub fn randomize<R: Rng>(&mut self, corner1: &Vec3, corner2: &Vec3, rng: &mut R) {
        self.position = random_in_box(corner1, corner2, rng);
        self.orientation = Quaternion::random(rng);
    }

    pub fn too_close(&self, c: &RigidConf, pos_cutoff: f64, ori_cutoff: f64) -> bool {
        if self.position.distance_sqr(&c.position) > sqr(pos_cutoff) { return false; }
        let diff = Quaternion::difference(&self.orientation, &c.orientation);
        if diff.norm_sqr() > sqr(ori_cutoff) { return false; }
        true
    }

    pub fn apply(&self, input: &[Vec3], output: &mut [Vec3], begin: usize, end: usize) {
        let m = self.orientation.to_mat3();
        for i in begin..end {
            output[i] = m.mul_vec(&input[i]) + self.position;
        }
    }
}

impl Default for RigidConf {
    fn default() -> Self { Self::new() }
}

// ─── Ligand ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LigandChange {
    pub rigid: RigidChange,
    pub torsions: Vec<f64>,
}

impl LigandChange {
    pub fn new(n_torsions: usize) -> Self {
        LigandChange {
            rigid: RigidChange::new(),
            torsions: vec![0.0; n_torsions],
        }
    }
}

#[derive(Debug, Clone)]
pub struct LigandConf {
    pub rigid: RigidConf,
    pub torsions: Vec<f64>,
}

impl LigandConf {
    pub fn new(n_torsions: usize) -> Self {
        LigandConf {
            rigid: RigidConf::new(),
            torsions: vec![0.0; n_torsions],
        }
    }

    pub fn set_to_null(&mut self) {
        self.rigid.set_to_null();
        for t in &mut self.torsions { *t = 0.0; }
    }

    pub fn increment(&mut self, c: &LigandChange, factor: f64) {
        self.rigid.increment(&c.rigid, factor);
        for (t, ct) in self.torsions.iter_mut().zip(c.torsions.iter()) {
            *t += normalized_angle(factor * ct);
            normalize_angle(t);
        }
    }

    pub fn randomize<R: Rng>(&mut self, corner1: &Vec3, corner2: &Vec3, rng: &mut R) {
        self.rigid.randomize(corner1, corner2, rng);
        for t in &mut self.torsions {
            *t = random_fl(-PI, PI, rng);
        }
    }
}

// ─── Residue ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ResidueChange {
    pub torsions: Vec<f64>,
}

impl ResidueChange {
    pub fn new(n: usize) -> Self {
        ResidueChange { torsions: vec![0.0; n] }
    }
}

#[derive(Debug, Clone)]
pub struct ResidueConf {
    pub torsions: Vec<f64>,
}

impl ResidueConf {
    pub fn new(n: usize) -> Self {
        ResidueConf { torsions: vec![0.0; n] }
    }

    pub fn set_to_null(&mut self) {
        for t in &mut self.torsions { *t = 0.0; }
    }

    pub fn increment(&mut self, c: &ResidueChange, factor: f64) {
        for (t, ct) in self.torsions.iter_mut().zip(c.torsions.iter()) {
            *t += normalized_angle(factor * ct);
            normalize_angle(t);
        }
    }

    pub fn randomize<R: Rng>(&mut self, rng: &mut R) {
        for t in &mut self.torsions {
            *t = random_fl(-PI, PI, rng);
        }
    }
}

// ─── Conf ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Conf {
    pub ligands: Vec<LigandConf>,
    pub flex: Vec<ResidueConf>,
}

impl Conf {
    pub fn new(s: &ConfSize) -> Self {
        Conf {
            ligands: s.ligands.iter().map(|&n| LigandConf::new(n)).collect(),
            flex: s.flex.iter().map(|&n| ResidueConf::new(n)).collect(),
        }
    }

    pub fn set_to_null(&mut self) {
        for l in &mut self.ligands { l.set_to_null(); }
        for f in &mut self.flex { f.set_to_null(); }
    }

    pub fn increment(&mut self, c: &Change, factor: f64) {
        for (l, cl) in self.ligands.iter_mut().zip(c.ligands.iter()) {
            l.increment(cl, factor);
        }
        for (f, cf) in self.flex.iter_mut().zip(c.flex.iter()) {
            f.increment(cf, factor);
        }
    }

    /// Copy values from another Conf without reallocating buffers.
    pub fn copy_from(&mut self, other: &Conf) {
        debug_assert_eq!(self.ligands.len(), other.ligands.len());
        debug_assert_eq!(self.flex.len(), other.flex.len());

        for (dst, src) in self.ligands.iter_mut().zip(other.ligands.iter()) {
            dst.rigid.position = src.rigid.position;
            dst.rigid.orientation = src.rigid.orientation;
            debug_assert_eq!(dst.torsions.len(), src.torsions.len());
            dst.torsions.copy_from_slice(&src.torsions);
        }
        for (dst, src) in self.flex.iter_mut().zip(other.flex.iter()) {
            debug_assert_eq!(dst.torsions.len(), src.torsions.len());
            dst.torsions.copy_from_slice(&src.torsions);
        }
    }

    pub fn randomize<R: Rng>(&mut self, corner1: &Vec3, corner2: &Vec3, rng: &mut R) {
        for l in &mut self.ligands { l.randomize(corner1, corner2, rng); }
        for f in &mut self.flex { f.randomize(rng); }
    }

    /// Flatten conformation into a Vec<f64> (for QVina2 Visited database)
    /// Layout: 3 position + 4 quaternion + torsions per ligand (matches C++)
    pub fn get_v(&self) -> Vec<f64> {
        let mut v = Vec::new();
        self.write_v(&mut v);
        v
    }

    /// Flatten conformation into an existing buffer.
    /// Pushes raw quaternion (4 values) to match C++ conf::getV() layout:
    /// position(3) + quaternion(4) + torsions(n) per ligand.
    pub fn write_v(&self, out: &mut Vec<f64>) {
        out.clear();
        let n_lig: usize = self.ligands.iter().map(|l| 7 + l.torsions.len()).sum();
        let n_flex: usize = self.flex.iter().map(|f| f.torsions.len()).sum();
        out.reserve(n_lig + n_flex);

        for l in &self.ligands {
            out.push(l.rigid.position[0]);
            out.push(l.rigid.position[1]);
            out.push(l.rigid.position[2]);
            // Raw quaternion (4 components) — matches C++ qt::getV()
            out.push(l.rigid.orientation.w);
            out.push(l.rigid.orientation.x);
            out.push(l.rigid.orientation.y);
            out.push(l.rigid.orientation.z);
            out.extend_from_slice(&l.torsions);
        }
        for f in &self.flex {
            out.extend_from_slice(&f.torsions);
        }
    }
}

// ─── Change ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Change {
    pub ligands: Vec<LigandChange>,
    pub flex: Vec<ResidueChange>,
}

impl Change {
    pub fn new(s: &ConfSize) -> Self {
        Change {
            ligands: s.ligands.iter().map(|&n| LigandChange::new(n)).collect(),
            flex: s.flex.iter().map(|&n| ResidueChange::new(n)).collect(),
        }
    }

    pub fn num_floats(&self) -> usize {
        let mut tmp = 0;
        for l in &self.ligands { tmp += 6 + l.torsions.len(); }
        for f in &self.flex { tmp += f.torsions.len(); }
        tmp
    }

    /// Get float value by flat index
    pub fn get_flat(&self, mut index: usize) -> f64 {
        for l in &self.ligands {
            if index < 3 { return l.rigid.position[index]; }
            index -= 3;
            if index < 3 { return l.rigid.orientation[index]; }
            index -= 3;
            if index < l.torsions.len() { return l.torsions[index]; }
            index -= l.torsions.len();
        }
        for f in &self.flex {
            if index < f.torsions.len() { return f.torsions[index]; }
            index -= f.torsions.len();
        }
        panic!("Change::get_flat: index out of range");
    }

    /// Set float value by flat index
    pub fn set_flat(&mut self, mut index: usize, val: f64) {
        for l in &mut self.ligands {
            if index < 3 { l.rigid.position[index] = val; return; }
            index -= 3;
            if index < 3 { l.rigid.orientation[index] = val; return; }
            index -= 3;
            if index < l.torsions.len() { l.torsions[index] = val; return; }
            index -= l.torsions.len();
        }
        for f in &mut self.flex {
            if index < f.torsions.len() { f.torsions[index] = val; return; }
            index -= f.torsions.len();
        }
        panic!("Change::set_flat: index out of range");
    }

    /// Set all values to zero
    pub fn clear(&mut self) {
        for l in &mut self.ligands {
            l.rigid.position = Vec3::ZERO;
            l.rigid.orientation = Vec3::ZERO;
            for t in &mut l.torsions { *t = 0.0; }
        }
        for f in &mut self.flex {
            for t in &mut f.torsions { *t = 0.0; }
        }
    }

    /// Copy values from another Change without reallocating buffers.
    pub fn copy_from(&mut self, other: &Change) {
        debug_assert_eq!(self.ligands.len(), other.ligands.len());
        debug_assert_eq!(self.flex.len(), other.flex.len());

        for (dst, src) in self.ligands.iter_mut().zip(other.ligands.iter()) {
            dst.rigid.position = src.rigid.position;
            dst.rigid.orientation = src.rigid.orientation;
            debug_assert_eq!(dst.torsions.len(), src.torsions.len());
            dst.torsions.copy_from_slice(&src.torsions);
        }
        for (dst, src) in self.flex.iter_mut().zip(other.flex.iter()) {
            debug_assert_eq!(dst.torsions.len(), src.torsions.len());
            dst.torsions.copy_from_slice(&src.torsions);
        }
    }

    /// Flatten change into a Vec<f64> (for QVina2 Visited database)
    pub fn get_v(&self) -> Vec<f64> {
        let n = self.num_floats();
        let mut v = Vec::with_capacity(n);
        self.write_v(&mut v);
        v
    }

    /// Flatten change into an existing buffer.
    pub fn write_v(&self, out: &mut Vec<f64>) {
        out.clear();
        out.reserve(self.num_floats());
        for l in &self.ligands {
            out.push(l.rigid.position[0]);
            out.push(l.rigid.position[1]);
            out.push(l.rigid.position[2]);
            out.push(l.rigid.orientation[0]);
            out.push(l.rigid.orientation[1]);
            out.push(l.rigid.orientation[2]);
            out.extend_from_slice(&l.torsions);
        }
        for f in &self.flex {
            out.extend_from_slice(&f.torsions);
        }
    }

    /// Fill `out` with flattened values; `out.len()` must equal `num_floats()`.
    pub fn read_flat_into(&self, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.num_floats());
        let mut k = 0usize;
        for l in &self.ligands {
            out[k] = l.rigid.position[0]; k += 1;
            out[k] = l.rigid.position[1]; k += 1;
            out[k] = l.rigid.position[2]; k += 1;
            out[k] = l.rigid.orientation[0]; k += 1;
            out[k] = l.rigid.orientation[1]; k += 1;
            out[k] = l.rigid.orientation[2]; k += 1;
            let n = l.torsions.len();
            out[k..k + n].copy_from_slice(&l.torsions);
            k += n;
        }
        for f in &self.flex {
            let n = f.torsions.len();
            out[k..k + n].copy_from_slice(&f.torsions);
            k += n;
        }
    }

    /// Set flattened values from `flat`; `flat.len()` must equal `num_floats()`.
    pub fn write_flat_from(&mut self, flat: &[f64]) {
        debug_assert_eq!(flat.len(), self.num_floats());
        let mut k = 0usize;
        for l in &mut self.ligands {
            l.rigid.position[0] = flat[k]; k += 1;
            l.rigid.position[1] = flat[k]; k += 1;
            l.rigid.position[2] = flat[k]; k += 1;
            l.rigid.orientation[0] = flat[k]; k += 1;
            l.rigid.orientation[1] = flat[k]; k += 1;
            l.rigid.orientation[2] = flat[k]; k += 1;
            let n = l.torsions.len();
            l.torsions.copy_from_slice(&flat[k..k + n]);
            k += n;
        }
        for f in &mut self.flex {
            let n = f.torsions.len();
            f.torsions.copy_from_slice(&flat[k..k + n]);
            k += n;
        }
    }
}

/// Scalar product of two Change objects
pub fn change_scalar_product(a: &Change, b: &Change, n: usize) -> f64 {
    debug_assert_eq!(a.num_floats(), n);
    debug_assert_eq!(b.num_floats(), n);
    let mut s = 0.0;
    for (la, lb) in a.ligands.iter().zip(b.ligands.iter()) {
        s += la.rigid.position.dot(&lb.rigid.position);
        s += la.rigid.orientation.dot(&lb.rigid.orientation);
        for (ta, tb) in la.torsions.iter().zip(lb.torsions.iter()) {
            s += ta * tb;
        }
    }
    for (fa, fb) in a.flex.iter().zip(b.flex.iter()) {
        for (ta, tb) in fa.torsions.iter().zip(fb.torsions.iter()) {
            s += ta * tb;
        }
    }
    s
}

/// b -= a
pub fn subtract_change(b: &mut Change, a: &Change, n: usize) {
    debug_assert_eq!(a.num_floats(), n);
    debug_assert_eq!(b.num_floats(), n);
    for (lb, la) in b.ligands.iter_mut().zip(a.ligands.iter()) {
        lb.rigid.position -= la.rigid.position;
        lb.rigid.orientation -= la.rigid.orientation;
        for (tb, ta) in lb.torsions.iter_mut().zip(la.torsions.iter()) {
            *tb -= *ta;
        }
    }
    for (fb, fa) in b.flex.iter_mut().zip(a.flex.iter()) {
        for (tb, ta) in fb.torsions.iter_mut().zip(fa.torsions.iter()) {
            *tb -= *ta;
        }
    }
}

// ─── OutputType ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OutputType {
    pub c: Conf,
    pub e: f64,
    pub lb: f64,
    pub ub: f64,
    pub intra: f64,
    pub inter: f64,
    pub conf_independent: f64,
    pub unbound: f64,
    pub total: f64,
    pub coords: Vec<Vec3>,
}

impl OutputType {
    pub fn new(s: &ConfSize, e: f64) -> Self {
        OutputType {
            c: Conf::new(s),
            e,
            lb: 0.0,
            ub: 0.0,
            intra: 0.0,
            inter: 0.0,
            conf_independent: 0.0,
            unbound: 0.0,
            total: 0.0,
            coords: Vec::new(),
        }
    }

    /// Copy values from another OutputType without reallocating buffers.
    pub fn copy_from(&mut self, other: &OutputType) {
        self.c.copy_from(&other.c);
        self.e = other.e;
        self.lb = other.lb;
        self.ub = other.ub;
        self.intra = other.intra;
        self.inter = other.inter;
        self.conf_independent = other.conf_independent;
        self.unbound = other.unbound;
        self.total = other.total;
        self.coords.clear();
        self.coords.extend_from_slice(&other.coords);
    }
}

/// Output container sorted by energy
pub type OutputContainer = Vec<OutputType>;

/// Find the closest pose by RMSD (matching C++ find_closest)
fn find_closest(coords: &[Vec3], out: &OutputContainer) -> (usize, f64) {
    let mut best_idx = out.len();
    let mut best_rmsd = MAX_FL;
    for (i, existing) in out.iter().enumerate() {
        let rmsd = compute_rmsd(coords, &existing.coords);
        if i == 0 || rmsd < best_rmsd {
            best_idx = i;
            best_rmsd = rmsd;
        }
    }
    (best_idx, best_rmsd)
}

/// Add to output container with RMSD filtering (matching C++ add_to_output_container)
pub fn add_to_output_container(
    out: &mut OutputContainer,
    candidate: &OutputType,
    min_rmsd: f64,
    max_size: usize,
) {
    let (closest_idx, closest_rmsd) = find_closest(&candidate.coords, out);

    if closest_idx < out.len() && closest_rmsd < min_rmsd {
        // Have a very similar one — replace if candidate is better
        if candidate.e < out[closest_idx].e {
            out[closest_idx].copy_from(candidate);
        }
    } else {
        // Nothing similar
        if out.len() < max_size {
            out.push(candidate.clone());
        } else if !out.is_empty() && candidate.e < out.last().unwrap().e {
            // Replace worst (last after sort)
            let last = out.len() - 1;
            out[last].copy_from(candidate);
        }
    }

    out.sort_by(|a, b| a.e.partial_cmp(&b.e).unwrap_or(std::cmp::Ordering::Equal));
}

/// Compute RMSD between two coordinate sets (heavy atoms)
pub fn compute_rmsd(a: &[Vec3], b: &[Vec3]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return f64::MAX;
    }
    let n = a.len() as f64;
    let sum: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai.distance_sqr(bi)).sum();
    (sum / n).sqrt()
}
