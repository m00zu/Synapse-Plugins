use crate::common::*;

// ─── Array3D ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Array3D {
    data: Vec<f64>,
    dim0: usize,
    dim1: usize,
    dim2: usize,
}

impl Array3D {
    pub fn new(d0: usize, d1: usize, d2: usize, fill: f64) -> Self {
        Array3D {
            data: vec![fill; d0 * d1 * d2],
            dim0: d0,
            dim1: d1,
            dim2: d2,
        }
    }

    pub fn dim0(&self) -> usize { self.dim0 }
    pub fn dim1(&self) -> usize { self.dim1 }
    pub fn dim2(&self) -> usize { self.dim2 }

    #[inline(always)]
    fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dim0 * (j + self.dim1 * k)
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        // SAFETY: callers ensure indices are in-bounds (clamped in evaluate_aux)
        unsafe { *self.data.get_unchecked(self.index(i, j, k)) }
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, k: usize, val: f64) {
        let idx = self.index(i, j, k);
        self.data[idx] = val;
    }
}

// ─── GridDim ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct GridDim {
    pub n_voxels: usize,
    pub begin: f64,
    pub end: f64,
}

impl GridDim {
    pub fn new() -> Self {
        GridDim { n_voxels: 0, begin: 0.0, end: 0.0 }
    }

    pub fn span(&self) -> f64 {
        self.end - self.begin
    }
}

impl Default for GridDim {
    fn default() -> Self { Self::new() }
}

pub type GridDims = [GridDim; 3];

pub fn make_grid_dims(center: &Vec3, size: &Vec3, granularity: f64, force_even: bool) -> GridDims {
    let mut gd = [GridDim::new(); 3];
    for i in 0..3 {
        let mut n = (size[i] / granularity).ceil() as usize;
        if force_even && n % 2 == 1 { n += 1; }
        let real_span = granularity * n as f64;
        gd[i].n_voxels = n;
        gd[i].begin = center[i] - real_span / 2.0;
        gd[i].end = gd[i].begin + real_span;
    }
    gd
}

// ─── Grid ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Grid {
    pub m_init: Vec3,
    pub m_range: Vec3,
    pub m_factor: Vec3,
    pub m_factor_inv: Vec3,
    pub m_dim_fl_minus_1: Vec3,
    pub m_data: Array3D,
}

impl Grid {
    pub fn new() -> Self {
        Grid {
            m_init: Vec3::ZERO,
            m_range: Vec3::new(1.0, 1.0, 1.0),
            m_factor: Vec3::new(1.0, 1.0, 1.0),
            m_factor_inv: Vec3::new(1.0, 1.0, 1.0),
            m_dim_fl_minus_1: Vec3::new(-1.0, -1.0, -1.0),
            m_data: Array3D::new(0, 0, 0, 0.0),
        }
    }

    pub fn init(&mut self, gd: &GridDims) {
        let d0 = gd[0].n_voxels + 1;
        let d1 = gd[1].n_voxels + 1;
        let d2 = gd[2].n_voxels + 1;
        self.m_data = Array3D::new(d0, d1, d2, 0.0);

        for i in 0..3 {
            self.m_init[i] = gd[i].begin;
            self.m_range[i] = gd[i].end - gd[i].begin;
            let n = if i == 0 { d0 } else if i == 1 { d1 } else { d2 };
            if n > 1 {
                self.m_factor[i] = (n as f64 - 1.0) / self.m_range[i];
                self.m_factor_inv[i] = self.m_range[i] / (n as f64 - 1.0);
            }
            self.m_dim_fl_minus_1[i] = n as f64 - 1.0;
        }
    }

    pub fn initialized(&self) -> bool {
        self.m_data.dim0() > 0 && self.m_data.dim1() > 0 && self.m_data.dim2() > 0
    }

    pub fn index_to_argument(&self, x: usize, y: usize, z: usize) -> Vec3 {
        Vec3::new(
            self.m_init[0] + self.m_factor_inv[0] * x as f64,
            self.m_init[1] + self.m_factor_inv[1] * y as f64,
            self.m_init[2] + self.m_factor_inv[2] * z as f64,
        )
    }

    /// Trilinear interpolation without derivatives
    pub fn evaluate(&self, location: &Vec3, slope: f64, v: f64) -> f64 {
        self.evaluate_aux(location, slope, v, None)
    }

    /// Trilinear interpolation with derivatives
    pub fn evaluate_deriv(&self, location: &Vec3, slope: f64, v: f64) -> (f64, Vec3) {
        let mut deriv = Vec3::ZERO;
        let e = self.evaluate_aux(location, slope, v, Some(&mut deriv));
        (e, deriv)
    }

    fn evaluate_aux(&self, location: &Vec3, slope: f64, v: f64, deriv: Option<&mut Vec3>) -> f64 {
        let mut s = [0.0_f64; 3]; // fractional coordinates
        let mut a = [0_usize; 3];  // integer grid indices
        let mut missing = [0.0_f64; 3]; // out-of-bounds distance
        let mut region = [0_i32; 3]; // -1, 0, +1 like C++

        for dim in 0..3 {
            s[dim] = (location[dim] - self.m_init[dim]) * self.m_factor[dim];

            if s[dim] < 0.0 {
                missing[dim] = -s[dim];
                region[dim] = -1;
                a[dim] = 0;
                s[dim] = 0.0;
            } else if s[dim] >= self.m_dim_fl_minus_1[dim] {
                missing[dim] = s[dim] - self.m_dim_fl_minus_1[dim];
                region[dim] = 1;
                let max_idx = match dim {
                    0 => self.m_data.dim0().saturating_sub(2),
                    1 => self.m_data.dim1().saturating_sub(2),
                    _ => self.m_data.dim2().saturating_sub(2),
                };
                a[dim] = max_idx;
                s[dim] = 1.0;
            } else {
                region[dim] = 0;
                a[dim] = s[dim] as usize;
                s[dim] -= a[dim] as f64;
            }
        }

        // C++ uses: penalty = slope * (miss * m_factor_inv)
        // which converts fractional-space miss to real-space distance
        let penalty = slope * (
            missing[0] * self.m_factor_inv[0] +
            missing[1] * self.m_factor_inv[1] +
            missing[2] * self.m_factor_inv[2]
        );

        // Trilinear interpolation
        let sx = s[0]; let sy = s[1]; let sz = s[2];
        let mx = 1.0 - sx; let my = 1.0 - sy; let mz = 1.0 - sz;

        let f000 = self.m_data.get(a[0],   a[1],   a[2]);
        let f100 = self.m_data.get(a[0]+1, a[1],   a[2]);
        let f010 = self.m_data.get(a[0],   a[1]+1, a[2]);
        let f110 = self.m_data.get(a[0]+1, a[1]+1, a[2]);
        let f001 = self.m_data.get(a[0],   a[1],   a[2]+1);
        let f101 = self.m_data.get(a[0]+1, a[1],   a[2]+1);
        let f011 = self.m_data.get(a[0],   a[1]+1, a[2]+1);
        let f111 = self.m_data.get(a[0]+1, a[1]+1, a[2]+1);

        let mut f = mx * my * mz * f000
            + sx * my * mz * f100
            + mx * sy * mz * f010
            + sx * sy * mz * f110
            + mx * my * sz * f001
            + sx * my * sz * f101
            + mx * sy * sz * f011
            + sx * sy * sz * f111;

        if let Some(d) = deriv {
            // Raw fractional-space gradients (before m_factor scaling)
            let x_g = my * mz * (f100 - f000)
                    + sy * mz * (f110 - f010)
                    + my * sz * (f101 - f001)
                    + sy * sz * (f111 - f011);

            let y_g = mx * mz * (f010 - f000)
                    + sx * mz * (f110 - f100)
                    + mx * sz * (f011 - f001)
                    + sx * sz * (f111 - f101);

            let z_g = mx * my * (f001 - f000)
                    + sx * my * (f101 - f100)
                    + mx * sy * (f011 - f010)
                    + sx * sy * (f111 - f110);

            // Apply curl to f and gradient in fractional space (matching C++)
            let mut gradient = Vec3::new(x_g, y_g, z_g);
            curl_with_deriv(&mut f, &mut gradient, v);

            // Construct final derivative: m_factor * curled_gradient (in-bounds only) + penalty slope
            // C++: gradient_everywhere[i] = (region[i] == 0) ? gradient[i] : 0;
            //      (*deriv)[i] = m_factor[i] * gradient_everywhere[i] + slope * region[i];
            for dim in 0..3 {
                let grad_component = if region[dim] == 0 { gradient[dim] } else { 0.0 };
                d[dim] = self.m_factor[dim] * grad_component + slope * region[dim] as f64;
            }

            f + penalty
        } else {
            // Energy-only path: curl f, then add penalty (matching C++)
            curl(&mut f, v);
            f + penalty
        }
    }
}

impl Default for Grid {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array3d() {
        let mut a = Array3D::new(3, 4, 5, 0.0);
        a.set(1, 2, 3, 42.0);
        assert_eq!(a.get(1, 2, 3), 42.0);
        assert_eq!(a.get(0, 0, 0), 0.0);
    }

    #[test]
    fn test_grid_init() {
        let gd = [
            GridDim { n_voxels: 10, begin: -5.0, end: 5.0 },
            GridDim { n_voxels: 10, begin: -5.0, end: 5.0 },
            GridDim { n_voxels: 10, begin: -5.0, end: 5.0 },
        ];
        let mut g = Grid::new();
        g.init(&gd);
        assert!(g.initialized());
        assert_eq!(g.m_data.dim0(), 11);
    }
}
