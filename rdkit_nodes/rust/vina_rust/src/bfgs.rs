use crate::common::*;
use crate::conf::*;
use crate::visited::{Visited, VisitedScratch};

// ─── BFGS Optimizer ────────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for BFGS to avoid per-call heap allocations.
/// Create once in the MC loop, reuse across all BFGS calls.
pub struct BfgsScratch {
    pub h: TriangularMatrix,
    pub g_new: Change,
    pub x_new: Conf,
    pub g_orig: Change,
    pub x_orig: Conf,
    pub p: Change,
    pub y: Change,
    pub minus_hy: Change,
    // Flat buffers for O(1) indexed access (avoiding O(n) get_flat)
    pub flat_a: Vec<f64>,  // general scratch (g_flat in mat-vec, mhy_flat in update)
    pub flat_b: Vec<f64>,  // general scratch (out_flat in mat-vec, p_flat in update)
    pub flat_c: Vec<f64>,  // y_flat in update
    pub flat_d: Vec<f64>,  // extra scratch for scalar products
}

impl BfgsScratch {
    pub fn new(x: &Conf, g: &Change) -> Self {
        let n = g.num_floats();
        BfgsScratch {
            h: TriangularMatrix::new(n, 0.0),
            g_new: g.clone(),
            x_new: x.clone(),
            g_orig: g.clone(),
            x_orig: x.clone(),
            p: g.clone(),
            y: g.clone(),
            minus_hy: g.clone(),
            flat_a: vec![0.0; n],
            flat_b: vec![0.0; n],
            flat_c: vec![0.0; n],
            flat_d: vec![0.0; n],
        }
    }

    /// Reset Hessian to identity matrix
    #[inline]
    fn reset_h(&mut self) {
        self.h.reset(0.0);
        let n = self.h.dim();
        for i in 0..n {
            self.h.set(i, i, 1.0);
        }
    }
}

/// Scalar product using pre-flattened arrays — O(n) with no struct traversal.
#[inline]
fn flat_scalar_product(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

/// Multiply -H * g, storing result in out. Uses pre-allocated flat buffers.
#[inline]
fn minus_mat_vec_product(
    h: &TriangularMatrix,
    g: &Change,
    out: &mut Change,
    g_flat: &mut [f64],
    out_flat: &mut [f64],
) {
    let n = h.dim();
    g.read_flat_into(g_flat);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += h.get(i, j) * g_flat[j];
        }
        out_flat[i] = -sum;
    }
    out.write_flat_from(out_flat);
}

fn set_diagonal(h: &mut TriangularMatrix, x: f64) {
    for i in 0..h.dim() {
        h.set(i, i, x);
    }
}

/// BFGS Hessian update — all O(1) flat array access, no get_flat calls.
fn bfgs_update(
    h: &mut TriangularMatrix,
    n: usize,
    alpha: f64,
    // Pre-flattened arrays (caller must flatten before calling)
    p_flat: &[f64],
    y_flat: &[f64],
    mhy_flat: &[f64],
) -> bool {
    let yp = flat_scalar_product(y_flat, p_flat, n);
    if alpha * yp < EPSILON_FL { return false; }

    let yhy = -flat_scalar_product(y_flat, mhy_flat, n);
    let r = 1.0 / (alpha * yp);
    let alpha_r = alpha * r;
    let alpha2_r2_yhy_plus_r = alpha * alpha * (r * r * yhy + r);

    for i in 0..n {
        let mhy_i = mhy_flat[i];
        let p_i = p_flat[i];
        for j in i..n {
            let update = alpha_r * (mhy_i * p_flat[j] + mhy_flat[j] * p_i)
                + alpha2_r2_yhy_plus_r * p_i * p_flat[j];
            let old = h.get(i, j);
            h.set(i, j, old + update);
        }
    }
    true
}

/// Armijo backtracking line search
pub fn line_search<F>(
    f: &mut F,
    n: usize,
    x: &Conf,
    g: &Change,
    f0: f64,
    p: &Change,
    x_new: &mut Conf,
    g_new: &mut Change,
    f1: &mut f64,
    evalcount: &mut i32,
) -> f64
where
    F: FnMut(&Conf, &mut Change) -> f64,
{
    const C0: f64 = 0.0001;
    const MAX_TRIALS: u32 = 10;
    const MULTIPLIER: f64 = 0.5;
    let mut alpha = 1.0;

    let pg = change_scalar_product(p, g, n);

    for _ in 0..MAX_TRIALS {
        x_new.copy_from(x);
        x_new.increment(p, alpha);
        *f1 = f(x_new, g_new);
        *evalcount += 1;
        if *f1 - f0 < C0 * alpha * pg {
            break;
        }
        alpha *= MULTIPLIER;
    }
    alpha
}

/// Main BFGS optimizer — uses pre-allocated scratch buffers, all flat array access
pub fn bfgs<F>(
    f: &mut F,
    x: &mut Conf,
    g: &mut Change,
    max_steps: u32,
    evalcount: &mut i32,
    scratch: &mut BfgsScratch,
) -> f64
where
    F: FnMut(&Conf, &mut Change) -> f64,
{
    let n = g.num_floats();
    if n == 0 { return 0.0; }

    scratch.reset_h();

    let mut f0 = f(x, g);
    *evalcount += 1;

    let f_orig = f0;
    scratch.g_orig.copy_from(g);
    scratch.x_orig.copy_from(x);

    for step in 0..max_steps {
        // p = -H * g (flat_a = g_flat, flat_b = p_flat after this)
        minus_mat_vec_product(&scratch.h, g, &mut scratch.p, &mut scratch.flat_a, &mut scratch.flat_b);
        let mut f1 = 0.0;
        let alpha = line_search(f, n, x, g, f0, &scratch.p, &mut scratch.x_new, &mut scratch.g_new, &mut f1, evalcount);

        // y = g_new - g
        scratch.y.copy_from(&scratch.g_new);
        subtract_change(&mut scratch.y, g, n);

        f0 = f1;
        x.copy_from(&scratch.x_new);
        g.copy_from(&scratch.g_new);

        // Convergence check on NEW gradient (matching C++ order)
        let grad_norm = change_scalar_product(g, g, n).sqrt();
        if !(grad_norm >= 1e-5) { break; }

        // Flatten y and p for the Hessian update
        scratch.y.read_flat_into(&mut scratch.flat_c);  // y_flat
        scratch.p.read_flat_into(&mut scratch.flat_b);  // p_flat

        if step == 0 {
            // Initial Hessian scaling
            let yy = flat_scalar_product(&scratch.flat_c, &scratch.flat_c, n);
            if yy.abs() > EPSILON_FL {
                let yp = flat_scalar_product(&scratch.flat_c, &scratch.flat_b, n);
                set_diagonal(&mut scratch.h, alpha * yp / yy);
            }
        }

        // minus_hy = -H * y (uses flat_a as temp g_flat, flat_d as temp out_flat)
        minus_mat_vec_product(&scratch.h, &scratch.y, &mut scratch.minus_hy,
                             &mut scratch.flat_a, &mut scratch.flat_d);
        // flat_d now has minus_hy as flat array — use it directly
        bfgs_update(&mut scratch.h, n, alpha,
                     &scratch.flat_b, &scratch.flat_c, &scratch.flat_d);
    }

    // Safety: restore if optimization made things worse
    if !(f0 <= f_orig) {
        f0 = f_orig;
        x.copy_from(&scratch.x_orig);
        g.copy_from(&scratch.g_orig);
    }

    f0
}

/// QVina2 BFGS optimizer — per-thread Visited (no locking, matching C++ design)
pub fn bfgs_qvina2<F>(
    f: &mut F,
    x: &mut Conf,
    g: &mut Change,
    max_steps: u32,
    evalcount: &mut i32,
    visited: &mut Visited,
    visited_scratch: &mut VisitedScratch,
    scratch: &mut BfgsScratch,
) -> f64
where
    F: FnMut(&Conf, &mut Change) -> f64,
{
    let n = g.num_floats();
    if n == 0 { return 0.0; }

    // Evaluate FIRST — before any work (matching C++ QVina2 structure)
    let mut f0 = f(x, g);
    *evalcount += 1;

    // QVina2: check if this conformation is worth exploring (per-thread, no lock)
    if !visited.interesting(x, f0, g, visited_scratch) {
        return f0;
    }
    visited.add(x, f0, g, visited_scratch);

    // Reset Hessian to identity (reuse existing allocation)
    scratch.reset_h();

    let f_orig = f0;
    scratch.g_orig.copy_from(g);
    scratch.x_orig.copy_from(x);

    for step in 0..max_steps {
        // p = -H * g
        minus_mat_vec_product(&scratch.h, g, &mut scratch.p, &mut scratch.flat_a, &mut scratch.flat_b);
        let mut f1 = 0.0;
        let alpha = line_search(f, n, x, g, f0, &scratch.p, &mut scratch.x_new, &mut scratch.g_new, &mut f1, evalcount);

        // y = g_new - g
        scratch.y.copy_from(&scratch.g_new);
        subtract_change(&mut scratch.y, g, n);

        f0 = f1;
        x.copy_from(&scratch.x_new);
        g.copy_from(&scratch.g_new);

        // Convergence check on NEW gradient (matching C++ order)
        let grad_norm = change_scalar_product(g, g, n).sqrt();
        if !(grad_norm >= 1e-5) { break; }

        // Flatten y and p for the Hessian update
        scratch.y.read_flat_into(&mut scratch.flat_c);  // y_flat
        scratch.p.read_flat_into(&mut scratch.flat_b);  // p_flat

        if step == 0 {
            let yy = flat_scalar_product(&scratch.flat_c, &scratch.flat_c, n);
            if yy.abs() > EPSILON_FL {
                let yp = flat_scalar_product(&scratch.flat_c, &scratch.flat_b, n);
                set_diagonal(&mut scratch.h, alpha * yp / yy);
            }
        }

        // minus_hy = -H * y
        minus_mat_vec_product(&scratch.h, &scratch.y, &mut scratch.minus_hy,
                             &mut scratch.flat_a, &mut scratch.flat_d);
        bfgs_update(&mut scratch.h, n, alpha,
                     &scratch.flat_b, &scratch.flat_c, &scratch.flat_d);

        // QVina2: record trajectory point (per-thread, no lock)
        visited.add(x, f0, g, visited_scratch);
    }

    // Safety: restore if optimization made things worse
    if !(f0 <= f_orig) {
        f0 = f_orig;
        x.copy_from(&scratch.x_orig);
        g.copy_from(&scratch.g_orig);
    }

    f0
}

/// Quasi-Newton wrapper connecting BFGS to model evaluation
#[derive(Clone)]
pub struct QuasiNewton {
    pub max_steps: u32,
}

impl QuasiNewton {
    pub fn new() -> Self {
        QuasiNewton { max_steps: 1000 }
    }
}

impl Default for QuasiNewton {
    fn default() -> Self { Self::new() }
}
