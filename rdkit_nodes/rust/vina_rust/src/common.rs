use rand::Rng;
use std::f64::consts::PI;

pub const EPSILON_FL: f64 = f64::EPSILON;
pub const MAX_FL: f64 = f64::MAX;

// ─── Vec3 ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3(pub [f64; 3]);

impl Vec3 {
    pub const ZERO: Vec3 = Vec3([0.0, 0.0, 0.0]);

    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3([x, y, z])
    }

    #[inline(always)]
    pub fn x(&self) -> f64 { self.0[0] }
    #[inline(always)]
    pub fn y(&self) -> f64 { self.0[1] }
    #[inline(always)]
    pub fn z(&self) -> f64 { self.0[2] }

    #[inline(always)]
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.0[0] * other.0[0] + self.0[1] * other.0[1] + self.0[2] * other.0[2]
    }

    #[inline(always)]
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3([
            self.0[1] * other.0[2] - self.0[2] * other.0[1],
            self.0[2] * other.0[0] - self.0[0] * other.0[2],
            self.0[0] * other.0[1] - self.0[1] * other.0[0],
        ])
    }

    #[inline(always)]
    pub fn norm_sqr(&self) -> f64 {
        self.dot(self)
    }

    #[inline(always)]
    pub fn norm(&self) -> f64 {
        self.norm_sqr().sqrt()
    }

    #[inline(always)]
    pub fn scale(&self, s: f64) -> Vec3 {
        Vec3([self.0[0] * s, self.0[1] * s, self.0[2] * s])
    }

    #[inline(always)]
    pub fn distance_sqr(&self, other: &Vec3) -> f64 {
        let d = *self - *other;
        d.norm_sqr()
    }

    #[inline(always)]
    pub fn assign(&mut self, val: f64) {
        self.0 = [val, val, val];
    }

    #[inline(always)]
    pub fn normalized(&self) -> Vec3 {
        let n = self.norm();
        if n > EPSILON_FL {
            self.scale(1.0 / n)
        } else {
            Vec3::ZERO
        }
    }

    #[inline(always)]
    pub fn elementwise_product(&self, other: &Vec3) -> Vec3 {
        Vec3([
            self.0[0] * other.0[0],
            self.0[1] * other.0[1],
            self.0[2] * other.0[2],
        ])
    }
}

impl std::ops::Index<usize> for Vec3 {
    type Output = f64;
    #[inline(always)]
    fn index(&self, i: usize) -> &f64 { &self.0[i] }
}

impl std::ops::IndexMut<usize> for Vec3 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f64 { &mut self.0[i] }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1], self.0[2] + rhs.0[2]])
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1], self.0[2] - rhs.0[2]])
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn neg(self) -> Vec3 {
        Vec3([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl std::ops::AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Vec3) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
    }
}

impl std::ops::SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Vec3) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
        self.0[2] -= rhs.0[2];
    }
}

impl std::ops::MulAssign<f64> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f64) {
        self.0[0] *= s;
        self.0[1] *= s;
        self.0[2] *= s;
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, s: f64) -> Vec3 { self.scale(s) }
}

impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, v: Vec3) -> Vec3 { v.scale(self) }
}

// ─── Mat3 (column-major 3×3) ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3(pub [f64; 9]);

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    /// Access element (row i, col j) — column-major: data[i + 3*j]
    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.0[i + 3 * j]
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.0[i + 3 * j] = val;
    }

    #[inline(always)]
    pub fn mul_vec(&self, v: &Vec3) -> Vec3 {
        Vec3([
            self.0[0] * v.0[0] + self.0[3] * v.0[1] + self.0[6] * v.0[2],
            self.0[1] * v.0[0] + self.0[4] * v.0[1] + self.0[7] * v.0[2],
            self.0[2] * v.0[0] + self.0[5] * v.0[1] + self.0[8] * v.0[2],
        ])
    }
}

// ─── Quaternion ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub const IDENTITY: Quaternion = Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    #[inline(always)]
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Quaternion { w, x, y, z }
    }

    #[inline(always)]
    pub fn norm_sqr(&self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline(always)]
    pub fn norm(&self) -> f64 {
        self.norm_sqr().sqrt()
    }

    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > EPSILON_FL {
            let inv = 1.0 / n;
            self.w *= inv;
            self.x *= inv;
            self.y *= inv;
            self.z *= inv;
        }
    }

    pub fn normalized(&self) -> Quaternion {
        let mut q = *self;
        q.normalize();
        q
    }

    pub fn normalize_approx(&mut self) {
        let s = self.norm_sqr();
        if (s - 1.0).abs() > 1e-6 {
            self.normalize();
        }
    }

    pub fn is_normalized(&self) -> bool {
        (self.norm_sqr() - 1.0).abs() < 0.001
    }

    /// Multiply two quaternions: self * other
    #[inline(always)]
    pub fn mul(&self, r: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            x: self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            y: self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            z: self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        }
    }

    /// Convert quaternion to 3×3 rotation matrix (column-major)
    pub fn to_mat3(&self) -> Mat3 {
        let a = self.w;
        let b = self.x;
        let c = self.y;
        let d = self.z;

        let aa = a * a;
        let ab = a * b;
        let ac = a * c;
        let ad = a * d;
        let bb = b * b;
        let bc = b * c;
        let bd = b * d;
        let cc = c * c;
        let cd = c * d;
        let dd = d * d;

        // Column-major: data[i + 3*j]
        Mat3([
            // column 0
            aa + bb - cc - dd,
            2.0 * (bc + ad),
            2.0 * (bd - ac),
            // column 1
            2.0 * (bc - ad),
            aa - bb + cc - dd,
            2.0 * (cd + ab),
            // column 2
            2.0 * (bd + ac),
            2.0 * (cd - ab),
            aa - bb - cc + dd,
        ])
    }

    /// Create quaternion from axis-angle (rotation vector = angle * unit_axis)
    pub fn from_angle_axis(rotation: &Vec3) -> Quaternion {
        let angle = rotation.norm();
        if angle > EPSILON_FL {
            let axis = rotation.scale(1.0 / angle);
            let half = angle * 0.5;
            let s = half.sin();
            Quaternion {
                w: half.cos(),
                x: s * axis.0[0],
                y: s * axis.0[1],
                z: s * axis.0[2],
            }
        } else {
            Quaternion::IDENTITY
        }
    }

    /// Create quaternion from axis and angle separately
    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> Quaternion {
        let half = angle * 0.5;
        let s = half.sin();
        Quaternion {
            w: half.cos(),
            x: s * axis.0[0],
            y: s * axis.0[1],
            z: s * axis.0[2],
        }
    }

    /// Convert quaternion to rotation vector (angle * axis)
    pub fn to_angle(&self) -> Vec3 {
        // Ensure w is positive for shortest rotation
        let q = if self.w < 0.0 {
            Quaternion::new(-self.w, -self.x, -self.y, -self.z)
        } else {
            *self
        };
        let s = (q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
        if s > EPSILON_FL {
            let angle = 2.0 * s.atan2(q.w);
            let inv_s = angle / s;
            Vec3::new(q.x * inv_s, q.y * inv_s, q.z * inv_s)
        } else {
            Vec3::ZERO
        }
    }

    /// Increment orientation by rotation vector: q = rotation_quat * q
    pub fn increment(&mut self, rotation: &Vec3) {
        let dq = Quaternion::from_angle_axis(rotation);
        *self = dq.mul(self);
        self.normalize_approx();
    }

    /// Difference: rotation from a to b, returned as rotation vector
    pub fn difference(b: &Quaternion, a: &Quaternion) -> Vec3 {
        // q_diff = b * conj(a)
        let a_conj = Quaternion::new(a.w, -a.x, -a.y, -a.z);
        let diff = b.mul(&a_conj);
        diff.to_angle()
    }

    /// Random unit quaternion
    pub fn random<R: Rng>(rng: &mut R) -> Quaternion {
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen::<f64>() * 2.0 * PI;
        let u3: f64 = rng.gen::<f64>() * 2.0 * PI;

        let a = (1.0 - u1).sqrt();
        let b = u1.sqrt();

        Quaternion {
            w: a * u2.sin(),
            x: a * u2.cos(),
            y: b * u3.sin(),
            z: b * u3.cos(),
        }
    }
}

// ─── Utility functions ─────────────────────────────────────────────────────────

#[inline(always)]
pub fn sqr(x: f64) -> f64 {
    x * x
}

#[inline(always)]
pub fn normalize_angle(x: &mut f64) {
    // Match C++ behavior: O(1) reduction for large values using ceil,
    // then single subtraction for near-range values.
    if *x > 3.0 * PI {
        let n = (*x - PI) / (2.0 * PI);
        *x -= 2.0 * PI * n.ceil();
        normalize_angle(x); // recursive call for edge cases
    } else if *x < -3.0 * PI {
        let n = (-*x - PI) / (2.0 * PI);
        *x += 2.0 * PI * n.ceil();
        normalize_angle(x);
    } else if *x > PI {
        *x -= 2.0 * PI;
    } else if *x < -PI {
        *x += 2.0 * PI;
    }
}

#[inline(always)]
pub fn normalized_angle(x: f64) -> f64 {
    let mut a = x;
    normalize_angle(&mut a);
    a
}

#[inline(always)]
pub fn not_max(x: f64) -> bool {
    x < 0.1 * MAX_FL
}

pub fn random_inside_sphere<R: Rng>(rng: &mut R) -> Vec3 {
    loop {
        let v = Vec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        if v.norm_sqr() <= 1.0 {
            return v;
        }
    }
}

pub fn random_in_box<R: Rng>(corner1: &Vec3, corner2: &Vec3, rng: &mut R) -> Vec3 {
    Vec3::new(
        rng.gen_range(corner1[0]..corner2[0]),
        rng.gen_range(corner1[1]..corner2[1]),
        rng.gen_range(corner1[2]..corner2[2]),
    )
}

pub fn random_fl<R: Rng>(a: f64, b: f64, rng: &mut R) -> f64 {
    rng.gen_range(a..b)
}

pub fn random_int<R: Rng>(a: i32, b: i32, rng: &mut R) -> i32 {
    rng.gen_range(a..=b)
}

/// Curl function: caps energy value at v, with smooth transition
#[inline(always)]
pub fn curl(e: &mut f64, v: f64) {
    if *e > 0.0 && not_max(v) {
        *e = v * (*e / (v + *e));
    }
}

/// Curl with force adjustment
#[inline(always)]
pub fn curl_with_deriv(e: &mut f64, deriv: &mut Vec3, v: f64) {
    if *e > 0.0 && not_max(v) {
        let old_e = *e;
        *e = v * (old_e / (v + old_e));
        let factor = sqr(v / (v + old_e));
        *deriv *= factor;
    }
}

// ─── TriangularMatrix ──────────────────────────────────────────────────────────

/// Upper triangular matrix stored as flat array: n*(n+1)/2 elements
#[derive(Debug, Clone)]
pub struct TriangularMatrix {
    data: Vec<f64>,
    dim: usize,
}

impl TriangularMatrix {
    pub fn new(n: usize, fill: f64) -> Self {
        TriangularMatrix {
            data: vec![fill; n * (n + 1) / 2],
            dim: n,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Index for (i, j) where i <= j
    #[inline(always)]
    fn index(&self, i: usize, j: usize) -> usize {
        debug_assert!(i <= j);
        debug_assert!(j < self.dim);
        i + j * (j + 1) / 2
    }

    /// Permissive index: min/max applied automatically
    #[inline(always)]
    pub fn index_permissive(&self, i: usize, j: usize) -> usize {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        self.index(a, b)
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        unsafe { *self.data.get_unchecked(self.index_permissive(i, j)) }
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        let idx = self.index_permissive(i, j);
        unsafe { *self.data.get_unchecked_mut(idx) = val; }
    }

    #[inline(always)]
    pub fn get_by_idx(&self, idx: usize) -> f64 {
        self.data[idx]
    }

    #[inline(always)]
    pub fn set_by_idx(&mut self, idx: usize, val: f64) {
        self.data[idx] = val;
    }

    /// Reset all values to `fill` without reallocating.
    #[inline]
    pub fn reset(&mut self, fill: f64) {
        for v in &mut self.data {
            *v = fill;
        }
    }
}

impl std::ops::Index<(usize, usize)> for TriangularMatrix {
    type Output = f64;
    #[inline(always)]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[self.index_permissive(i, j)]
    }
}

impl std::ops::IndexMut<(usize, usize)> for TriangularMatrix {
    #[inline(always)]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        let idx = self.index_permissive(i, j);
        &mut self.data[idx]
    }
}

// ─── StrictlyTriangularMatrix ──────────────────────────────────────────────────

/// Strictly upper triangular matrix: n*(n-1)/2 elements (no diagonal)
#[derive(Debug, Clone)]
pub struct StrictlyTriangularMatrix<T: Clone + Default> {
    data: Vec<T>,
    dim: usize,
}

impl<T: Clone + Default> StrictlyTriangularMatrix<T> {
    pub fn new(n: usize, fill: T) -> Self {
        StrictlyTriangularMatrix {
            data: vec![fill; n * (n.saturating_sub(1)) / 2],
            dim: n,
        }
    }

    pub fn dim(&self) -> usize { self.dim }

    /// Index for (i, j) where i < j
    #[inline(always)]
    fn index(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < j);
        debug_assert!(j < self.dim);
        i + j * (j - 1) / 2
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> &T {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        &self.data[self.index(a, b)]
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, val: T) {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let idx = self.index(a, b);
        self.data[idx] = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(&b), 32.0);
        let c = a.cross(&b);
        assert_eq!(c, Vec3::new(-3.0, 6.0, -3.0));
    }

    #[test]
    fn test_quaternion_identity() {
        let q = Quaternion::IDENTITY;
        let m = q.to_mat3();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = m.mul_vec(&v);
        assert!((r[0] - 1.0).abs() < 1e-10);
        assert!((r[1] - 2.0).abs() < 1e-10);
        assert!((r[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_rotation() {
        // 90° around Z axis
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let q = Quaternion::from_axis_angle(&axis, PI / 2.0);
        let m = q.to_mat3();
        let v = Vec3::new(1.0, 0.0, 0.0);
        let r = m.mul_vec(&v);
        assert!((r[0]).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
        assert!((r[2]).abs() < 1e-10);
    }

    #[test]
    fn test_triangular_matrix() {
        let mut m = TriangularMatrix::new(3, 0.0);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 2.0); // same as (0,1) due to symmetry
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(1, 0), 2.0);
    }
}
