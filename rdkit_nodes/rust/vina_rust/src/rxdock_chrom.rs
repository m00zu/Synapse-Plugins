//! Chromosome representation for rxDock search.
//!
//! Maps between a flat f64 vector (for Simplex/SA) and the existing
//! Conf/LigandConf structure used by Model::set().

use crate::common::*;
use crate::conf::*;
use crate::rxdock_cavity::Cavity;
use rand::Rng;
use std::f64::consts::PI;

/// Chromosome: position [3] + orientation [3] (axis-angle) + torsions [N].
/// Total DOF = 6 + N.
#[derive(Debug, Clone)]
pub struct RxChromosome {
    /// Flat representation: [px, py, pz, ax, ay, az, t0, t1, ..., tN-1]
    pub genes: Vec<f64>,
    /// Step sizes for each gene (for SA mutation / Simplex steps).
    pub step_sizes: Vec<f64>,
    /// Number of torsion angles.
    pub n_torsions: usize,
}

impl RxChromosome {
    /// Create a new chromosome for a ligand with `n_torsions` rotatable bonds.
    pub fn new(n_torsions: usize) -> Self {
        let n = 6 + n_torsions;
        let mut step_sizes = Vec::with_capacity(n);
        // Position step sizes (Å)
        step_sizes.extend_from_slice(&[0.5, 0.5, 0.5]);
        // Orientation step sizes (radians)
        step_sizes.extend_from_slice(&[0.3, 0.3, 0.3]);
        // Torsion step sizes (radians)
        for _ in 0..n_torsions {
            step_sizes.push(0.5);
        }

        RxChromosome {
            genes: vec![0.0; n],
            step_sizes,
            n_torsions,
        }
    }

    /// Total number of genes (degrees of freedom).
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    /// Randomize the chromosome within the cavity bounds.
    pub fn randomize<R: Rng>(&mut self, cavity: &Cavity, rng: &mut R) {
        // Random position within cavity bounding box
        self.genes[0] = rng.gen_range(cavity.min_coord.x()..=cavity.max_coord.x());
        self.genes[1] = rng.gen_range(cavity.min_coord.y()..=cavity.max_coord.y());
        self.genes[2] = rng.gen_range(cavity.min_coord.z()..=cavity.max_coord.z());

        // Random orientation (random axis-angle with magnitude up to PI)
        let angle = rng.gen_range(0.0..PI);
        let u: f64 = rng.gen_range(-1.0..1.0);
        let theta: f64 = rng.gen_range(0.0..2.0 * PI);
        let sin_u = (1.0 - u * u).sqrt();
        self.genes[3] = angle * sin_u * theta.cos();
        self.genes[4] = angle * sin_u * theta.sin();
        self.genes[5] = angle * u;

        // Random torsions in [-PI, PI)
        for i in 0..self.n_torsions {
            self.genes[6 + i] = rng.gen_range(-PI..PI);
        }
    }

    /// Randomize within a search sphere (center + radius) instead of cavity bbox.
    pub fn randomize_sphere<R: Rng>(&mut self, center: &Vec3, radius: f64, rng: &mut R) {
        // Random point in sphere
        loop {
            let x = rng.gen_range(-radius..radius);
            let y = rng.gen_range(-radius..radius);
            let z = rng.gen_range(-radius..radius);
            if x * x + y * y + z * z <= radius * radius {
                self.genes[0] = center.x() + x;
                self.genes[1] = center.y() + y;
                self.genes[2] = center.z() + z;
                break;
            }
        }

        // Random orientation
        let angle = rng.gen_range(0.0..PI);
        let u: f64 = rng.gen_range(-1.0..1.0);
        let theta: f64 = rng.gen_range(0.0..2.0 * PI);
        let sin_u = (1.0 - u * u).sqrt();
        self.genes[3] = angle * sin_u * theta.cos();
        self.genes[4] = angle * sin_u * theta.sin();
        self.genes[5] = angle * u;

        for i in 0..self.n_torsions {
            self.genes[6 + i] = rng.gen_range(-PI..PI);
        }
    }

    /// Mutate: perturb each gene by step_size * gaussian.
    pub fn mutate<R: Rng>(&mut self, rng: &mut R) {
        for i in 0..self.genes.len() {
            let perturbation = gauss(rng) * self.step_sizes[i];
            self.genes[i] += perturbation;
        }
        // Wrap torsions to [-PI, PI)
        for i in 6..self.genes.len() {
            self.genes[i] = wrap_angle(self.genes[i]);
        }
    }

    /// Mutate a single random gene.
    pub fn mutate_one<R: Rng>(&mut self, rng: &mut R) {
        let idx = rng.gen_range(0..self.genes.len());
        self.genes[idx] += gauss(rng) * self.step_sizes[idx];
        if idx >= 6 {
            self.genes[idx] = wrap_angle(self.genes[idx]);
        }
    }

    /// Convert chromosome to a LigandConf for Model::set().
    pub fn to_ligand_conf(&self) -> LigandConf {
        let position = Vec3::new(self.genes[0], self.genes[1], self.genes[2]);

        // Convert axis-angle to quaternion
        let ax = self.genes[3];
        let ay = self.genes[4];
        let az = self.genes[5];
        let angle = (ax * ax + ay * ay + az * az).sqrt();
        let orientation = if angle > 1e-10 {
            Quaternion::from_axis_angle(
                &Vec3::new(ax / angle, ay / angle, az / angle),
                angle,
            )
        } else {
            Quaternion::IDENTITY
        };

        let torsions = self.genes[6..].to_vec();

        LigandConf {
            rigid: RigidConf { position, orientation },
            torsions,
        }
    }

    /// Convert chromosome to a full Conf (single-ligand case).
    pub fn to_conf(&self) -> Conf {
        Conf {
            ligands: vec![self.to_ligand_conf()],
            flex: Vec::new(),
        }
    }

    /// Read back from a LigandConf into the chromosome genes.
    pub fn from_ligand_conf(&mut self, conf: &LigandConf) {
        self.genes[0] = conf.rigid.position.x();
        self.genes[1] = conf.rigid.position.y();
        self.genes[2] = conf.rigid.position.z();

        // Convert quaternion to axis-angle vector (magnitude = angle, direction = axis)
        let aa = conf.rigid.orientation.to_angle();
        self.genes[3] = aa.x();
        self.genes[4] = aa.y();
        self.genes[5] = aa.z();

        for (i, &t) in conf.torsions.iter().enumerate() {
            self.genes[6 + i] = t;
        }
    }

    /// Scale step sizes by a factor.
    pub fn scale_steps(&mut self, factor: f64) {
        for s in &mut self.step_sizes {
            *s *= factor;
        }
    }

    /// Rectangular (uniform) mutation: each gene ± step_size * rel_step.
    pub fn rect_mutate<R: Rng>(&mut self, rel_step: f64, rng: &mut R) {
        for i in 0..self.genes.len() {
            let delta = rng.gen_range(-1.0..1.0) * self.step_sizes[i] * rel_step;
            self.genes[i] += delta;
        }
        for i in 6..self.genes.len() {
            self.genes[i] = wrap_angle(self.genes[i]);
        }
    }

    /// Cauchy mutation: heavy-tailed perturbation for occasional large jumps.
    pub fn cauchy_mutate<R: Rng>(&mut self, variance: f64, rng: &mut R) {
        let rel_step = cauchy_random(0.0, variance, rng).abs();
        self.rect_mutate(rel_step, rng);
    }

    /// 2-point crossover: swap genes between two random crossover points.
    /// Returns two children.
    pub fn crossover<R: Rng>(parent1: &RxChromosome, parent2: &RxChromosome, rng: &mut R) -> (RxChromosome, RxChromosome) {
        let n = parent1.genes.len();
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();

        if n < 2 { return (child1, child2); }

        let ix_begin = rng.gen_range(0..n);
        let ix_end = if ix_begin == 0 {
            rng.gen_range(1..n)
        } else {
            rng.gen_range(ix_begin + 1..=n)
        };

        for i in ix_begin..ix_end.min(n) {
            std::mem::swap(&mut child1.genes[i], &mut child2.genes[i]);
        }

        // Wrap torsion angles after crossover
        for i in 6..n {
            child1.genes[i] = wrap_angle(child1.genes[i]);
            child2.genes[i] = wrap_angle(child2.genes[i]);
        }

        (child1, child2)
    }

    /// Check if two chromosomes are near-duplicates (relative difference < threshold).
    pub fn is_near_duplicate(&self, other: &RxChromosome, threshold: f64) -> bool {
        if self.genes.len() != other.genes.len() { return false; }
        for i in 0..self.genes.len() {
            let denom = self.genes[i].abs().max(1.0);
            let rel_diff = (self.genes[i] - other.genes[i]).abs() / denom;
            if rel_diff > threshold {
                return false;
            }
        }
        true
    }
}

/// Wrap angle to [-PI, PI).
fn wrap_angle(a: f64) -> f64 {
    let mut a = a % (2.0 * PI);
    if a > PI { a -= 2.0 * PI; }
    if a < -PI { a += 2.0 * PI; }
    a
}

/// Simple Box-Muller Gaussian random number.
fn gauss<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen_range(0.0001..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Cauchy random number with given location and scale.
fn cauchy_random<R: Rng>(location: f64, scale: f64, rng: &mut R) -> f64 {
    let u: f64 = rng.gen_range(0.0001..0.9999);
    location + scale * (PI * (u - 0.5)).tan()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chromosome_size() {
        let chrom = RxChromosome::new(5);
        assert_eq!(chrom.len(), 11); // 6 rigid + 5 torsions
        assert_eq!(chrom.step_sizes.len(), 11);
    }

    #[test]
    fn test_to_conf_roundtrip() {
        let mut chrom = RxChromosome::new(3);
        chrom.genes = vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.5, -0.5, 1.0];

        let conf = chrom.to_ligand_conf();
        assert!((conf.rigid.position.x() - 1.0).abs() < 1e-10);
        assert_eq!(conf.torsions.len(), 3);
        assert!((conf.torsions[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_wrap_angle() {
        assert!((wrap_angle(4.0) - (4.0 - 2.0 * PI)).abs() < 1e-10);
        assert!((wrap_angle(-4.0) - (-4.0 + 2.0 * PI)).abs() < 1e-10);
        assert!((wrap_angle(1.0) - 1.0).abs() < 1e-10);
    }
}
