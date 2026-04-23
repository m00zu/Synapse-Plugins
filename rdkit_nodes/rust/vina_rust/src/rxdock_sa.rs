//! Simulated Annealing search for rxDock.
//!
//! Metropolis Monte Carlo with geometric cooling schedule.
//! Accepts uphill moves early in the search (high T), gradually
//! restricting to downhill moves as temperature decreases.

use crate::rxdock_chrom::RxChromosome;
use rand::Rng;

/// Parameters for simulated annealing.
#[derive(Debug, Clone)]
pub struct SimAnnParams {
    /// Starting temperature.
    pub start_temp: f64,
    /// Final temperature.
    pub final_temp: f64,
    /// Number of cooling blocks.
    pub num_blocks: usize,
    /// Moves per block = block_length × chromosome_length.
    pub block_length: usize,
    /// Minimum acceptance rate before step size adjustment.
    pub min_acc_rate: f64,
    /// Maximum acceptance rate before step size adjustment.
    pub max_acc_rate: f64,
    /// Step size scale factor for adjustment.
    pub step_factor: f64,
}

impl Default for SimAnnParams {
    fn default() -> Self {
        SimAnnParams {
            start_temp: 300.0,
            final_temp: 1.0,
            num_blocks: 25,
            block_length: 50,
            min_acc_rate: 0.25,
            max_acc_rate: 0.75,
            step_factor: 1.5,
        }
    }
}

/// Result of a simulated annealing run.
pub struct SimAnnResult {
    /// Best chromosome found.
    pub best_chrom: RxChromosome,
    /// Best energy found.
    pub best_energy: f64,
    /// Total number of function evaluations.
    pub n_evals: usize,
}

/// Run simulated annealing.
///
/// # Arguments
/// * `eval_fn` - Scoring function: takes &[f64] genes → f64 energy
/// * `initial` - Starting chromosome (will be cloned)
/// * `params` - SA parameters
/// * `rng` - Random number generator
/// * `progress_fn` - Optional progress callback: (block_idx, num_blocks, best_energy)
pub fn simulated_annealing<F, R, P>(
    eval_fn: &mut F,
    initial: &RxChromosome,
    params: &SimAnnParams,
    rng: &mut R,
    mut progress_fn: Option<&mut P>,
) -> SimAnnResult
where
    F: FnMut(&[f64]) -> f64,
    R: Rng,
    P: FnMut(usize, usize, f64),
{
    let n = initial.len();
    let moves_per_block = params.block_length * n;

    // Geometric cooling: T_k = T_0 * ratio^k
    let cooling_ratio = if params.num_blocks > 1 {
        (params.final_temp / params.start_temp).powf(1.0 / (params.num_blocks - 1) as f64)
    } else {
        1.0
    };

    let mut current = initial.clone();
    let mut current_energy = eval_fn(&current.genes);
    let mut best = current.clone();
    let mut best_energy = current_energy;
    let mut n_evals = 1;

    let mut temp = params.start_temp;

    for block in 0..params.num_blocks {
        let mut accepted = 0usize;
        let mut attempted = 0usize;

        for _ in 0..moves_per_block {
            // Generate trial move
            let mut trial = current.clone();
            trial.mutate_one(rng);

            let trial_energy = eval_fn(&trial.genes);
            n_evals += 1;
            attempted += 1;

            // Metropolis criterion
            let delta_e = trial_energy - current_energy;
            let accept = if delta_e <= 0.0 {
                true
            } else if temp > 1e-10 {
                let prob = (-delta_e / temp).exp();
                rng.gen::<f64>() < prob
            } else {
                false
            };

            if accept {
                current = trial;
                current_energy = trial_energy;
                accepted += 1;

                // Track best ever
                if current_energy < best_energy {
                    best = current.clone();
                    best_energy = current_energy;
                }
            }
        }

        // Adjust step sizes based on acceptance rate
        if attempted > 0 {
            let acc_rate = accepted as f64 / attempted as f64;
            if acc_rate < params.min_acc_rate {
                // Too many rejections → decrease step size
                current.scale_steps(1.0 / params.step_factor);
            } else if acc_rate > params.max_acc_rate {
                // Too many acceptances → increase step size
                current.scale_steps(params.step_factor);
            }
        }

        // Cool down
        temp *= cooling_ratio;

        // Report progress
        if let Some(ref mut cb) = progress_fn {
            cb(block, params.num_blocks, best_energy);
        }
    }

    SimAnnResult {
        best_chrom: best,
        best_energy,
        n_evals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_sa_finds_minimum() {
        // Minimize f(x,y) = (x-3)^2 + (y+2)^2
        // Minimum at (3, -2), value 0
        let mut eval = |genes: &[f64]| -> f64 {
            let x = genes[0];
            let y = genes[1];
            (x - 3.0) * (x - 3.0) + (y + 2.0) * (y + 2.0)
        };

        // Start chromosome near the origin. We only have 2 "genes" (no torsions),
        // but the chromosome structure uses 6 rigid + N torsion. For this unit test,
        // we'll work directly with a custom-sized chromosome.
        let chrom = RxChromosome {
            genes: vec![0.0, 0.0],
            step_sizes: vec![1.0, 1.0],
            n_torsions: 0,
        };

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let params = SimAnnParams {
            start_temp: 100.0,
            final_temp: 0.01,
            num_blocks: 50,
            block_length: 20,
            ..Default::default()
        };

        let result = simulated_annealing(&mut eval, &chrom, &params, &mut rng, None::<&mut fn(usize, usize, f64)>);

        assert!(result.best_energy < 1.0,
            "SA should find near-minimum, got energy {}", result.best_energy);
    }
}
