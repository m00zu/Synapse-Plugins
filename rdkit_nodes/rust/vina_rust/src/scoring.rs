use crate::atom::*;
use crate::common::*;
use std::f64::consts::PI;

// ─── Utility functions ─────────────────────────────────────────────────────────

#[inline(always)]
pub fn slope_step(x_bad: f64, x_good: f64, x: f64) -> f64 {
    if x_bad < x_good {
        if x <= x_bad { return 0.0; }
        if x >= x_good { return 1.0; }
    } else {
        if x >= x_bad { return 0.0; }
        if x <= x_good { return 1.0; }
    }
    (x - x_bad) / (x_good - x_bad)
}

#[inline(always)]
pub fn smoothen(r: f64, rij: f64, smoothing: f64) -> f64 {
    let s = smoothing * 0.5;
    if r > rij + s {
        r - s
    } else if r < rij - s {
        r + s
    } else {
        rij
    }
}

#[inline(always)]
pub fn smooth_div(x: f64, y: f64) -> f64 {
    if x.abs() < EPSILON_FL { return 0.0; }
    if y.abs() < EPSILON_FL {
        return if x * y > 0.0 { MAX_FL } else { -MAX_FL };
    }
    x / y
}

#[inline(always)]
fn int_pow(base: f64, exp: u32) -> f64 {
    let mut result = 1.0;
    let mut b = base;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 { result *= b; }
        b *= b;
        e >>= 1;
    }
    result
}

// ─── Scoring Function Choice ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringFunctionChoice {
    Vina,
    Vinardo,
    AD42,
}

// ─── Potential Trait ────────────────────────────────────────────────────────────

pub trait Potential: Send + Sync {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64;
    fn eval_by_type(&self, t1: usize, t2: usize, r: f64) -> f64;
    fn cutoff(&self) -> f64;
}

// ─── Vina Potentials ───────────────────────────────────────────────────────────

pub struct VinaGaussian {
    pub offset: f64,
    pub width: f64,
    pub cutoff_val: f64,
}

impl VinaGaussian {
    #[inline(always)]
    fn gauss(&self, x: f64) -> f64 {
        (-sqr(x / self.width)).exp()
    }
}

impl Potential for VinaGaussian {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if a.xs >= XS_TYPE_SIZE || b.xs >= XS_TYPE_SIZE { return 0.0; }
        self.gauss(r - (optimal_distance(a.xs, b.xs) + self.offset))
    }

    fn eval_by_type(&self, t1: usize, t2: usize, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        self.gauss(r - (optimal_distance(t1, t2) + self.offset))
    }

    fn cutoff(&self) -> f64 { self.cutoff_val }
}

pub struct VinaRepulsion {
    pub offset: f64,
    pub cutoff_val: f64,
}

impl Potential for VinaRepulsion {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if a.xs >= XS_TYPE_SIZE || b.xs >= XS_TYPE_SIZE { return 0.0; }
        let d = r - (optimal_distance(a.xs, b.xs) + self.offset);
        if d > 0.0 { 0.0 } else { d * d }
    }

    fn eval_by_type(&self, t1: usize, t2: usize, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        let d = r - (optimal_distance(t1, t2) + self.offset);
        if d > 0.0 { 0.0 } else { d * d }
    }

    fn cutoff(&self) -> f64 { self.cutoff_val }
}

pub struct VinaHydrophobic {
    pub good: f64,
    pub bad: f64,
    pub cutoff_val: f64,
}

impl Potential for VinaHydrophobic {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if a.xs >= XS_TYPE_SIZE || b.xs >= XS_TYPE_SIZE { return 0.0; }
        if xs_is_hydrophobic(a.xs) && xs_is_hydrophobic(b.xs) {
            slope_step(self.bad, self.good, r - optimal_distance(a.xs, b.xs))
        } else {
            0.0
        }
    }

    fn eval_by_type(&self, t1: usize, t2: usize, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if xs_is_hydrophobic(t1) && xs_is_hydrophobic(t2) {
            slope_step(self.bad, self.good, r - optimal_distance(t1, t2))
        } else {
            0.0
        }
    }

    fn cutoff(&self) -> f64 { self.cutoff_val }
}

pub struct VinaHBond {
    pub good: f64,
    pub bad: f64,
    pub cutoff_val: f64,
}

impl Potential for VinaHBond {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if a.xs >= XS_TYPE_SIZE || b.xs >= XS_TYPE_SIZE { return 0.0; }
        if xs_h_bond_possible(a.xs, b.xs) {
            slope_step(self.bad, self.good, r - optimal_distance(a.xs, b.xs))
        } else {
            0.0
        }
    }

    fn eval_by_type(&self, t1: usize, t2: usize, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if xs_h_bond_possible(t1, t2) {
            slope_step(self.bad, self.good, r - optimal_distance(t1, t2))
        } else {
            0.0
        }
    }

    fn cutoff(&self) -> f64 { self.cutoff_val }
}

// ─── Vinardo Potentials ────────────────────────────────────────────────────────

pub struct VinardoGaussian {
    pub offset: f64,
    pub width: f64,
    pub cutoff_val: f64,
}

impl VinardoGaussian {
    #[inline(always)]
    fn gauss(&self, x: f64) -> f64 {
        (-sqr(x / self.width)).exp()
    }
}

impl Potential for VinardoGaussian {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if a.xs >= XS_TYPE_SIZE || b.xs >= XS_TYPE_SIZE { return 0.0; }
        self.gauss(r - (optimal_distance_vinardo(a.xs, b.xs) + self.offset))
    }

    fn eval_by_type(&self, t1: usize, t2: usize, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        self.gauss(r - (optimal_distance_vinardo(t1, t2) + self.offset))
    }

    fn cutoff(&self) -> f64 { self.cutoff_val }
}

pub struct VinardoRepulsion {
    pub offset: f64,
    pub cutoff_val: f64,
}

impl Potential for VinardoRepulsion {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if a.xs >= XS_TYPE_SIZE || b.xs >= XS_TYPE_SIZE { return 0.0; }
        let d = r - (optimal_distance_vinardo(a.xs, b.xs) + self.offset);
        if d > 0.0 { 0.0 } else { d * d }
    }

    fn eval_by_type(&self, t1: usize, t2: usize, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        let d = r - (optimal_distance_vinardo(t1, t2) + self.offset);
        if d > 0.0 { 0.0 } else { d * d }
    }

    fn cutoff(&self) -> f64 { self.cutoff_val }
}

pub struct VinardoHydrophobic {
    pub good: f64,
    pub bad: f64,
    pub cutoff_val: f64,
}

impl Potential for VinardoHydrophobic {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if a.xs >= XS_TYPE_SIZE || b.xs >= XS_TYPE_SIZE { return 0.0; }
        if xs_is_hydrophobic(a.xs) && xs_is_hydrophobic(b.xs) {
            slope_step(self.bad, self.good, r - optimal_distance_vinardo(a.xs, b.xs))
        } else {
            0.0
        }
    }

    fn eval_by_type(&self, t1: usize, t2: usize, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if xs_is_hydrophobic(t1) && xs_is_hydrophobic(t2) {
            slope_step(self.bad, self.good, r - optimal_distance_vinardo(t1, t2))
        } else {
            0.0
        }
    }

    fn cutoff(&self) -> f64 { self.cutoff_val }
}

pub struct VinardoHBond {
    pub good: f64,
    pub bad: f64,
    pub cutoff_val: f64,
}

impl Potential for VinardoHBond {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if a.xs >= XS_TYPE_SIZE || b.xs >= XS_TYPE_SIZE { return 0.0; }
        if xs_h_bond_possible(a.xs, b.xs) {
            slope_step(self.bad, self.good, r - optimal_distance_vinardo(a.xs, b.xs))
        } else {
            0.0
        }
    }

    fn eval_by_type(&self, t1: usize, t2: usize, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if xs_h_bond_possible(t1, t2) {
            slope_step(self.bad, self.good, r - optimal_distance_vinardo(t1, t2))
        } else {
            0.0
        }
    }

    fn cutoff(&self) -> f64 { self.cutoff_val }
}

// ─── AD4 Potentials ────────────────────────────────────────────────────────────

pub struct Ad4Vdw {
    pub smoothing: f64,
    pub cap: f64,
    pub cutoff_val: f64,
}

impl Potential for Ad4Vdw {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        let t1 = a.ad;
        let t2 = b.ad;
        let hb_depth = ad_type_property(t1).hb_depth * ad_type_property(t2).hb_depth;
        let vdw_rij = ad_type_property(t1).radius + ad_type_property(t2).radius;
        let vdw_depth = (ad_type_property(t1).depth * ad_type_property(t2).depth).sqrt();
        if hb_depth < 0.0 { return 0.0; } // interaction is hb, not vdw
        let r_sm = smoothen(r, vdw_rij, self.smoothing);
        let c_12 = int_pow(vdw_rij, 12) * vdw_depth;
        let c_6 = int_pow(vdw_rij, 6) * vdw_depth * 2.0;
        let r6 = int_pow(r_sm, 6);
        let r12 = int_pow(r_sm, 12);
        if r12 > EPSILON_FL && r6 > EPSILON_FL {
            (c_12 / r12 - c_6 / r6).min(self.cap)
        } else {
            self.cap
        }
    }

    fn eval_by_type(&self, _t1: usize, _t2: usize, _r: f64) -> f64 { 0.0 }
    fn cutoff(&self) -> f64 { self.cutoff_val }
}

pub struct Ad4Hb {
    pub smoothing: f64,
    pub cap: f64,
    pub cutoff_val: f64,
}

impl Potential for Ad4Hb {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        let t1 = a.ad;
        let t2 = b.ad;
        let hb_rij = ad_type_property(t1).hb_radius + ad_type_property(t2).hb_radius;
        let hb_depth = ad_type_property(t1).hb_depth * ad_type_property(t2).hb_depth;
        if hb_depth >= 0.0 { return 0.0; } // not hb
        let r_sm = smoothen(r, hb_rij, self.smoothing);
        let c_12 = int_pow(hb_rij, 12) * (-hb_depth) * 10.0 / 2.0;
        let c_10 = int_pow(hb_rij, 10) * (-hb_depth) * 12.0 / 2.0;
        let r10 = int_pow(r_sm, 10);
        let r12 = int_pow(r_sm, 12);
        if r12 > EPSILON_FL && r10 > EPSILON_FL {
            (c_12 / r12 - c_10 / r10).min(self.cap)
        } else {
            self.cap
        }
    }

    fn eval_by_type(&self, _t1: usize, _t2: usize, _r: f64) -> f64 { 0.0 }
    fn cutoff(&self) -> f64 { self.cutoff_val }
}

pub struct Ad4Electrostatic {
    pub cap: f64,
    pub cutoff_val: f64,
}

impl Potential for Ad4Electrostatic {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        let q1q2 = a.charge * b.charge * 332.0;
        let big_b = 78.4 + 8.5525;
        let lb = -big_b * 0.003627;
        let diel = -8.5525 + (big_b / (1.0 + 7.7839 * (lb * r).exp()));
        if r < EPSILON_FL {
            q1q2 * self.cap / diel
        } else {
            q1q2 * (1.0 / (r * diel)).min(self.cap)
        }
    }

    fn eval_by_type(&self, _t1: usize, _t2: usize, _r: f64) -> f64 { 0.0 }
    fn cutoff(&self) -> f64 { self.cutoff_val }
}

pub struct Ad4Solvation {
    pub desolvation_sigma: f64,
    pub solvation_q: f64,
    pub charge_dependent: bool,
    pub cutoff_val: f64,
}

impl Ad4Solvation {
    fn volume(&self, a: &Atom) -> f64 {
        if a.ad < AD_TYPE_SIZE {
            ad_type_property(a.ad).volume
        } else if a.xs < XS_TYPE_SIZE {
            4.0 * PI / 3.0 * int_pow(xs_radius(a.xs), 3)
        } else {
            0.0
        }
    }

    fn solvation_parameter(&self, a: &Atom) -> f64 {
        if a.ad < AD_TYPE_SIZE {
            ad_type_property(a.ad).solvation
        } else if a.xs == XS_TYPE_MET_D {
            METAL_SOLVATION_PARAMETER
        } else {
            0.0
        }
    }
}

impl Potential for Ad4Solvation {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        let q1 = a.charge;
        let q2 = b.charge;
        let solv1 = self.solvation_parameter(a);
        let solv2 = self.solvation_parameter(b);
        let vol1 = self.volume(a);
        let vol2 = self.volume(b);
        let my_solv = if self.charge_dependent { self.solvation_q } else { 0.0 };
        ((solv1 + my_solv * q1.abs()) * vol2 + (solv2 + my_solv * q2.abs()) * vol1)
            * (-0.5 * sqr(r / self.desolvation_sigma)).exp()
    }

    fn eval_by_type(&self, _t1: usize, _t2: usize, _r: f64) -> f64 { 0.0 }
    fn cutoff(&self) -> f64 { self.cutoff_val }
}

// ─── Macrocycle Linear Attraction ──────────────────────────────────────────────

pub struct LinearAttraction {
    pub cutoff_val: f64,
}

impl Potential for LinearAttraction {
    fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if is_glued(a.xs, b.xs) { r } else { 0.0 }
    }

    fn eval_by_type(&self, t1: usize, t2: usize, r: f64) -> f64 {
        if r >= self.cutoff_val { return 0.0; }
        if is_glued(t1, t2) { r } else { 0.0 }
    }

    fn cutoff(&self) -> f64 { self.cutoff_val }
}

// ─── Conformation-Independent Terms ────────────────────────────────────────────

pub trait ConfIndependentTerm: Send + Sync {
    fn eval(&self, num_tors: f64, e: f64, weight_iter: &mut std::slice::Iter<f64>) -> f64;
}

/// Torsional penalty divisor (Vina/Vinardo)
pub struct NumTorsDiv;

impl ConfIndependentTerm for NumTorsDiv {
    fn eval(&self, num_tors: f64, e: f64, weight_iter: &mut std::slice::Iter<f64>) -> f64 {
        let w = weight_iter.next().copied().unwrap_or(0.0);
        let weight = 0.1 * (w + 1.0);
        smooth_div(e, 1.0 + weight * num_tors / 5.0)
    }
}

/// Torsional additive penalty (AD4)
pub struct Ad4TorsAdd;

impl ConfIndependentTerm for Ad4TorsAdd {
    fn eval(&self, num_tors: f64, e: f64, weight_iter: &mut std::slice::Iter<f64>) -> f64 {
        let w = weight_iter.next().copied().unwrap_or(0.0);
        e + w * num_tors
    }
}

// ─── ScoringFunction ───────────────────────────────────────────────────────────

pub struct ScoringFunction {
    pub potentials: Vec<Box<dyn Potential>>,
    pub conf_independents: Vec<Box<dyn ConfIndependentTerm>>,
    pub weights: Vec<f64>,
    pub cutoff: f64,
    pub max_cutoff: f64,
    pub atom_typing: AtomTyping,
}

impl ScoringFunction {
    pub fn new(choice: ScoringFunctionChoice, weights: Vec<f64>) -> Self {
        let (potentials, conf_independents, atom_typing, cutoff, max_cutoff): (
            Vec<Box<dyn Potential>>,
            Vec<Box<dyn ConfIndependentTerm>>,
            AtomTyping,
            f64,
            f64,
        ) = match choice {
            ScoringFunctionChoice::Vina => (
                vec![
                    Box::new(VinaGaussian { offset: 0.0, width: 0.5, cutoff_val: 8.0 }),
                    Box::new(VinaGaussian { offset: 3.0, width: 2.0, cutoff_val: 8.0 }),
                    Box::new(VinaRepulsion { offset: 0.0, cutoff_val: 8.0 }),
                    Box::new(VinaHydrophobic { good: 0.5, bad: 1.5, cutoff_val: 8.0 }),
                    Box::new(VinaHBond { good: -0.7, bad: 0.0, cutoff_val: 8.0 }),
                    Box::new(LinearAttraction { cutoff_val: 20.0 }),
                ],
                vec![Box::new(NumTorsDiv)],
                AtomTyping::XS,
                8.0,
                20.0,
            ),
            ScoringFunctionChoice::Vinardo => (
                vec![
                    Box::new(VinardoGaussian { offset: 0.0, width: 0.8, cutoff_val: 8.0 }),
                    Box::new(VinardoRepulsion { offset: 0.0, cutoff_val: 8.0 }),
                    Box::new(VinardoHydrophobic { good: 0.0, bad: 2.5, cutoff_val: 8.0 }),
                    Box::new(VinardoHBond { good: -0.6, bad: 0.0, cutoff_val: 8.0 }),
                    Box::new(LinearAttraction { cutoff_val: 20.0 }),
                ],
                vec![Box::new(NumTorsDiv)],
                AtomTyping::XS,
                8.0,
                20.0,
            ),
            ScoringFunctionChoice::AD42 => (
                vec![
                    Box::new(Ad4Vdw { smoothing: 0.5, cap: 100000.0, cutoff_val: 8.0 }),
                    Box::new(Ad4Hb { smoothing: 0.5, cap: 100000.0, cutoff_val: 8.0 }),
                    Box::new(Ad4Electrostatic { cap: 100.0, cutoff_val: 20.48 }),
                    Box::new(Ad4Solvation { desolvation_sigma: 3.6, solvation_q: 0.01097, charge_dependent: true, cutoff_val: 20.48 }),
                    Box::new(LinearAttraction { cutoff_val: 20.0 }),
                ],
                vec![Box::new(Ad4TorsAdd)],
                AtomTyping::AD,
                20.48,
                20.48,
            ),
        };

        ScoringFunction {
            potentials,
            conf_independents,
            weights,
            cutoff,
            max_cutoff,
            atom_typing,
        }
    }

    /// Evaluate scoring function by atom type pair
    pub fn eval_by_type(&self, t1: usize, t2: usize, r: f64) -> f64 {
        let mut acc = 0.0;
        for (i, pot) in self.potentials.iter().enumerate() {
            acc += self.weights[i] * pot.eval_by_type(t1, t2, r);
        }
        acc
    }

    /// Evaluate scoring function with actual atom objects
    pub fn eval_by_atoms(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        let mut acc = 0.0;
        for (i, pot) in self.potentials.iter().enumerate() {
            acc += self.weights[i] * pot.eval_by_atoms(a, b, r);
        }
        acc
    }

    /// Apply conformation-independent terms (torsional penalty)
    pub fn conf_independent(&self, num_tors: f64, mut e: f64) -> f64 {
        let num_pots = self.potentials.len();
        let mut iter = self.weights[num_pots..].iter();
        for ci in &self.conf_independents {
            e = ci.eval(num_tors, e, &mut iter);
        }
        e
    }

    pub fn num_potentials(&self) -> usize { self.potentials.len() }
    pub fn num_atom_types(&self) -> usize { num_atom_types(self.atom_typing) }
}

// ─── Default Weights ───────────────────────────────────────────────────────────

pub fn default_vina_weights() -> Vec<f64> {
    // [gauss1, gauss2, repulsion, hydrophobic, HBond, LinearAttraction(macrocycle=50), NumTorsDiv]
    vec![-0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 50.0, 1.923]
}

pub fn default_vinardo_weights() -> Vec<f64> {
    vec![-0.045, 0.8, -0.035, -0.600, 50.0, 1.923]
}

pub fn default_ad4_weights() -> Vec<f64> {
    vec![0.1662, 0.1209, 0.1406, 0.1322, 50.0, 0.2983]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slope_step() {
        assert_eq!(slope_step(1.5, 0.5, 1.0), 0.5);
        assert_eq!(slope_step(1.5, 0.5, 0.5), 1.0);
        assert_eq!(slope_step(1.5, 0.5, 1.5), 0.0);
        assert_eq!(slope_step(1.5, 0.5, 0.0), 1.0);
        assert_eq!(slope_step(1.5, 0.5, 2.0), 0.0);
    }

    #[test]
    fn test_vina_gaussian() {
        let g = VinaGaussian { offset: 0.0, width: 0.5, cutoff_val: 8.0 };
        // At optimal distance (d=0), gauss should return 1.0
        let val = g.eval_by_type(XS_TYPE_C_H, XS_TYPE_C_H, 3.8); // optimal = 1.9+1.9 = 3.8
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vina_repulsion() {
        let r = VinaRepulsion { offset: 0.0, cutoff_val: 8.0 };
        // At optimal distance, d=0, repulsion=0
        assert_eq!(r.eval_by_type(XS_TYPE_C_H, XS_TYPE_C_H, 3.8), 0.0);
        // Below optimal: d<0, repulsion = d^2
        let val = r.eval_by_type(XS_TYPE_C_H, XS_TYPE_C_H, 3.0); // d = 3.0 - 3.8 = -0.8
        assert!((val - 0.64).abs() < 1e-10);
    }

    #[test]
    fn test_scoring_function_creation() {
        let sf = ScoringFunction::new(ScoringFunctionChoice::Vina, default_vina_weights());
        assert_eq!(sf.potentials.len(), 6);
        assert_eq!(sf.cutoff, 8.0);
        assert_eq!(sf.max_cutoff, 20.0);
    }
}
