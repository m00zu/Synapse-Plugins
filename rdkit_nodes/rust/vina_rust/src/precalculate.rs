use crate::atom::*;
use crate::common::*;
use crate::scoring::ScoringFunction;

// ─── PrecalcElement ────────────────────────────────────────────────────────────

/// Precalculated energy/derivative lookup table for a single atom-type pair
#[derive(Debug, Clone)]
pub struct PrecalcElement {
    pub fast: Vec<f64>,
    pub smooth: Vec<(f64, f64)>, // (energy, dE/dr)
    pub factor: f64,
}

impl PrecalcElement {
    pub fn new(n: usize, factor: f64) -> Self {
        PrecalcElement {
            fast: vec![0.0; n],
            smooth: vec![(0.0, 0.0); n],
            factor,
        }
    }

    /// Fast energy lookup (unsmoothed average)
    #[inline(always)]
    pub fn eval_fast(&self, r2: f64) -> f64 {
        let i = (self.factor * r2) as usize;
        if i < self.fast.len() { self.fast[i] } else { 0.0 }
    }

    /// Energy + derivative lookup with linear interpolation
    #[inline(always)]
    pub fn eval_deriv(&self, r2: f64) -> (f64, f64) {
        let r2_factored = self.factor * r2;
        let i1 = r2_factored as usize;
        let i2 = i1 + 1;
        if i2 >= self.smooth.len() {
            return (0.0, 0.0);
        }
        let rem = r2_factored - i1 as f64;
        let p1 = &self.smooth[i1];
        let p2 = &self.smooth[i2];
        let e = p1.0 + rem * (p2.0 - p1.0);
        let dor = p1.1 + rem * (p2.1 - p1.1);
        (e, dor)
    }

    /// Initialize derivatives and fast values from smooth energy values
    pub fn init_from_smooth_fst(&mut self, rs: &[f64]) {
        let n = self.smooth.len();
        for i in 0..n {
            // Compute derivative (central difference)
            if i == 0 || i == n - 1 {
                self.smooth[i].1 = 0.0;
            } else {
                let delta = rs[i + 1] - rs[i - 1];
                let r = rs[i];
                if r.abs() > EPSILON_FL && delta.abs() > EPSILON_FL {
                    self.smooth[i].1 = (self.smooth[i + 1].0 - self.smooth[i - 1].0) / (delta * r);
                } else {
                    self.smooth[i].1 = 0.0;
                }
            }
            // Compute fast (average of adjacent)
            let f1 = self.smooth[i].0;
            let f2 = if i + 1 < n { self.smooth[i + 1].0 } else { 0.0 };
            self.fast[i] = (f2 + f1) / 2.0;
        }
    }
}

// ─── Precalculate ──────────────────────────────────────────────────────────────

/// Precalculated scoring for all atom-type pairs (triangular matrix)
#[derive(Clone)]
pub struct Precalculate {
    data: Vec<PrecalcElement>, // triangular storage
    dim: usize,                // number of atom types
    pub factor: f64,
    pub n: usize,
    pub cutoff_sqr: f64,
    pub max_cutoff_sqr: f64,
}

impl Precalculate {
    pub fn new(sf: &ScoringFunction, v: f64, factor: f64) -> Self {
        let cutoff_sqr = sqr(sf.cutoff);
        let max_cutoff_sqr = sqr(sf.max_cutoff);
        let n = (factor * max_cutoff_sqr) as usize + 3;
        let dim = sf.num_atom_types();
        let num_pairs = dim * (dim + 1) / 2;

        let v_cap = if v < MAX_FL * 0.1 { v } else { MAX_FL };

        // Precompute rs: rs[i] = sqrt(i / factor)
        let rs: Vec<f64> = (0..n).map(|i| (i as f64 / factor).sqrt()).collect();

        let mut data = Vec::with_capacity(num_pairs);

        // Build in column-major triangular order to match index() = a + b*(b+1)/2
        for b in 0..dim {
            for a in 0..=b {
                let mut elem = PrecalcElement::new(n, factor);
                for i in 0..n {
                    let val = sf.eval_by_type(a, b, rs[i]);
                    elem.smooth[i].0 = val.min(v_cap);
                }
                elem.init_from_smooth_fst(&rs);
                data.push(elem);
            }
        }

        Precalculate {
            data,
            dim,
            factor,
            n,
            cutoff_sqr,
            max_cutoff_sqr,
        }
    }

    pub fn dim(&self) -> usize { self.dim }

    /// Index into triangular storage for types (t1, t2)
    #[inline(always)]
    fn index(&self, t1: usize, t2: usize) -> usize {
        let (a, b) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        a + b * (b + 1) / 2
    }

    pub fn eval_fast(&self, t1: usize, t2: usize, r2: f64) -> f64 {
        self.data[self.index(t1, t2)].eval_fast(r2)
    }

    pub fn eval_deriv(&self, t1: usize, t2: usize, r2: f64) -> (f64, f64) {
        self.data[self.index(t1, t2)].eval_deriv(r2)
    }

    pub fn element(&self, t1: usize, t2: usize) -> &PrecalcElement {
        &self.data[self.index(t1, t2)]
    }
}

// ─── PrecalculateByAtom ────────────────────────────────────────────────────────

/// Per-atom precalculation for fast evaluation
#[derive(Clone)]
pub struct PrecalculateByAtom {
    precalc: Precalculate,
    atom_typing: AtomTyping,
}

impl PrecalculateByAtom {
    pub fn new(sf: &ScoringFunction, factor: f64) -> Self {
        PrecalculateByAtom {
            precalc: Precalculate::new(sf, MAX_FL, factor),
            atom_typing: sf.atom_typing,
        }
    }

    pub fn cutoff_sqr(&self) -> f64 { self.precalc.cutoff_sqr }
    pub fn max_cutoff_sqr(&self) -> f64 { self.precalc.max_cutoff_sqr }

    /// Evaluate energy (fast) given atom indices and their types
    #[inline(always)]
    pub fn eval_fast_by_types(&self, t1: usize, t2: usize, r2: f64) -> f64 {
        self.precalc.eval_fast(t1, t2, r2)
    }

    /// Evaluate energy + derivative given atom types
    #[inline(always)]
    pub fn eval_deriv_by_types(&self, t1: usize, t2: usize, r2: f64) -> (f64, f64) {
        self.precalc.eval_deriv(t1, t2, r2)
    }

    /// Evaluate fast using type pair index (pre-computed for interacting pairs)
    #[inline(always)]
    pub fn eval_fast_by_index(&self, pair_index: usize, r2: f64) -> f64 {
        self.precalc.data[pair_index].eval_fast(r2)
    }

    /// Evaluate deriv using type pair index
    #[inline(always)]
    pub fn eval_deriv_by_index(&self, pair_index: usize, r2: f64) -> (f64, f64) {
        self.precalc.data[pair_index].eval_deriv(r2)
    }

    pub fn atom_typing(&self) -> AtomTyping { self.atom_typing }
    pub fn factor(&self) -> f64 { self.precalc.factor }
    pub fn precalc(&self) -> &Precalculate { &self.precalc }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::*;

    #[test]
    fn test_precalculate_creation() {
        let sf = ScoringFunction::new(ScoringFunctionChoice::Vina, default_vina_weights());
        let pc = Precalculate::new(&sf, MAX_FL, 32.0);
        assert_eq!(pc.n, (32.0 * 400.0) as usize + 3); // 12803
        assert_eq!(pc.cutoff_sqr, 64.0);
        assert_eq!(pc.max_cutoff_sqr, 400.0);
    }

    #[test]
    fn test_precalculate_eval() {
        let sf = ScoringFunction::new(ScoringFunctionChoice::Vina, default_vina_weights());
        let pc = Precalculate::new(&sf, MAX_FL, 32.0);
        // At optimal distance for C_H - C_H (3.8 Å), r² = 14.44
        let (e, _dor) = pc.eval_deriv(XS_TYPE_C_H, XS_TYPE_C_H, 14.44);
        // Energy should be finite and reasonable
        assert!(e.is_finite());
    }
}
