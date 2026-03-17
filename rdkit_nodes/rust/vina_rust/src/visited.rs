use crate::conf::*;

// ─── VisitedElement ─────────────────────────────────────────────────────────

/// A single recorded conformation in the visited history.
/// Uses bit-encoded gradient signs for fast basin comparison.
#[derive(Debug, Clone)]
pub struct VisitedElement {
    pub x: Vec<f64>,       // conformation as flat vector
    pub f: f64,            // energy value
    pub d_zero: i64,       // bitmask: bit i = 1 if derivative[i] is zero
    pub d_positive: i64,   // bitmask: bit i = 1 if derivative[i] is positive
}

impl VisitedElement {
    pub fn new(x: Vec<f64>, f: f64, d: &[f64]) -> Self {
        let mut d_zero: i64 = 0;
        let mut d_positive: i64 = 0;

        for (i, &di) in d.iter().enumerate() {
            let bit_mask: i64 = 1_i64 << i;
            if di == 0.0 {
                d_zero |= bit_mask;
            } else if di > 0.0 {
                d_positive |= bit_mask;
            }
        }

        VisitedElement { x, f, d_zero, d_positive }
    }

    /// Squared Euclidean distance to another point
    #[inline]
    pub fn dist2(&self, now: &[f64]) -> f64 {
        let mut out = 0.0;
        for (xi, ni) in self.x.iter().zip(now.iter()) {
            let d = xi - ni;
            out += d * d;
        }
        out
    }

    /// Check if a new point is in a different basin than this recorded point.
    /// Returns false if the new point appears to be in the same basin (not interesting).
    #[inline]
    pub fn check(&self, now_x: &[f64], now_f: f64, now_d: &[f64]) -> bool {
        let new_y_bigger = (now_f - self.f) > 0.0;

        for (i, &now_di) in now_d.iter().enumerate() {
            let bit_mask: i64 = 1_i64 << i;

            // If either derivative is zero, skip (inconclusive)
            if (self.d_zero & bit_mask) != 0 || now_di == 0.0 {
                continue;
            }

            let now_positive = now_di > 0.0;
            let d_positive = (self.d_positive & bit_mask) != 0;

            if now_positive ^ d_positive {
                // Different signs — skip (different direction)
                continue;
            } else {
                // Same sign — check if consistent with energy landscape
                let new_x_bigger = (now_x[i] - self.x[i]) > 0.0;
                if !(now_positive ^ new_x_bigger ^ new_y_bigger) {
                    // Consistent — continue checking
                } else {
                    // Inconsistent — this is the same basin
                    return false;
                }
            }
        }
        true
    }
}

// ─── VisitedScratch ────────────────────────────────────────────────────────
// Per-thread scratch buffers used by interesting(). Kept separate from
// Visited so that the shared data can be behind a RwLock while each thread
// owns its own scratch.

#[derive(Debug, Clone)]
pub struct VisitedScratch {
    pub dist_buf: Vec<f64>,
    pub not_picked_buf: Vec<bool>,
    pub conf_buf: Vec<f64>,
    pub change_buf: Vec<f64>,
}

impl VisitedScratch {
    pub fn new() -> Self {
        VisitedScratch {
            dist_buf: Vec::new(),
            not_picked_buf: Vec::new(),
            conf_buf: Vec::new(),
            change_buf: Vec::new(),
        }
    }
}

impl Default for VisitedScratch {
    fn default() -> Self { Self::new() }
}

// ─── Visited ────────────────────────────────────────────────────────────────

/// Ring buffer of visited conformations for avoiding redundant local searches.
/// Core optimization of QuickVina 2.
///
/// The data is designed to be shared across threads via `Arc<RwLock<Visited>>`.
/// Each thread keeps its own `VisitedScratch` for scratch buffers.
#[derive(Debug, Clone)]
pub struct Visited {
    pub list: Vec<VisitedElement>,
    pub n_variable: usize,
    pub pointer: usize,
    pub full: bool,
}

impl Visited {
    pub fn new() -> Self {
        Visited {
            list: Vec::new(),
            n_variable: 0,
            pointer: 0,
            full: false,
        }
    }

    #[inline]
    fn max_check(&self) -> usize {
        4 * self.n_variable
    }

    #[inline]
    fn max_size(&self) -> usize {
        5 * self.n_variable
    }

    /// Check if a new conformation is worth exploring.
    /// Returns true if the point appears to be in a new basin.
    /// Takes `&self` so it can be called behind a RwLock read guard.
    pub fn interesting(&self, conf: &Conf, f: f64, change: &Change, scratch: &mut VisitedScratch) -> bool {
        let len = self.list.len();
        if len == 0 || !self.full {
            return true;
        }

        conf.write_v(&mut scratch.conf_buf);
        change.write_v(&mut scratch.change_buf);
        let conf_v = &scratch.conf_buf;
        let change_v = &scratch.change_buf;

        // Reuse scratch buffers to avoid allocations
        scratch.dist_buf.clear();
        scratch.dist_buf.extend(self.list.iter().map(|e| e.dist2(conf_v)));

        scratch.not_picked_buf.clear();
        scratch.not_picked_buf.resize(len, true);

        let max_check = self.max_check().min(len);

        // Check nearest max_check neighbors
        for _ in 0..max_check {
            // Find closest unpicked point
            let mut min_dist = f64::MAX;
            let mut p = 0;
            for j in 0..len {
                if scratch.not_picked_buf[j] && scratch.dist_buf[j] < min_dist {
                    p = j;
                    min_dist = scratch.dist_buf[j];
                }
            }
            scratch.not_picked_buf[p] = false;

            if self.list[p].check(conf_v, f, change_v) {
                return true;
            }
        }
        false
    }

    /// Add a conformation to the visited history.
    pub fn add(&mut self, conf: &Conf, f: f64, change: &Change, scratch: &mut VisitedScratch) {
        // Reuse internal scratch buffers, then clone into element storage
        conf.write_v(&mut scratch.conf_buf);
        change.write_v(&mut scratch.change_buf);

        if self.list.is_empty() {
            self.n_variable = scratch.conf_buf.len();
        }

        let elem = VisitedElement::new(scratch.conf_buf.clone(), f, &scratch.change_buf);

        if !self.full {
            self.list.push(elem);
            if self.list.len() >= self.max_size() {
                self.full = true;
                self.pointer = 0;
            }
        } else {
            let max_size = self.max_size();
            self.list[self.pointer] = elem;
            self.pointer = (self.pointer + 1) % max_size;
        }
    }
}

impl Default for Visited {
    fn default() -> Self { Self::new() }
}
