//! Genetic Algorithm search for rxDock.
//!
//! Population-based search with roulette wheel selection (sigma truncation),
//! 2-point crossover, and Cauchy/rectangular mutation.
//! Matches the C++ rxDock GATransform algorithm.

use crate::rxdock_chrom::RxChromosome;
use rand::Rng;

/// Parameters for the GA search.
#[derive(Debug, Clone)]
pub struct GAParams {
    /// Population size.
    pub pop_size: usize,
    /// Maximum generations.
    pub n_cycles: usize,
    /// Stop if best doesn't improve for this many generations.
    pub convergence_count: usize,
    /// Probability of crossover (vs mutation only).
    pub crossover_prob: f64,
    /// Apply Cauchy mutation after crossover.
    pub crossover_mutation: bool,
    /// Use Cauchy distribution for regular mutations (false = rectangular).
    pub cauchy_mutation: bool,
    /// Relative step size for mutations.
    pub step_size: f64,
    /// Threshold for near-duplicate detection.
    pub equality_threshold: f64,
}

impl Default for GAParams {
    fn default() -> Self {
        GAParams {
            pop_size: 50,
            n_cycles: 100,
            convergence_count: 6,
            crossover_prob: 0.4,
            crossover_mutation: true,
            cauchy_mutation: false,
            step_size: 1.0,
            equality_threshold: 0.1,
        }
    }
}

/// A single genome in the population.
#[derive(Debug, Clone)]
struct Genome {
    chrom: RxChromosome,
    score: f64,
    rw_fitness: f64,  // cumulative roulette-wheel fitness
}

/// GA population.
struct Population {
    genomes: Vec<Genome>,
    pop_size: usize,
}

impl Population {
    /// Create initial random population.
    fn new<F, R>(
        template: &RxChromosome,
        pop_size: usize,
        eval_fn: &mut F,
        rng: &mut R,
        cavity: &crate::rxdock_cavity::Cavity,
    ) -> Self
    where
        F: FnMut(&[f64]) -> f64,
        R: Rng,
    {
        let mut genomes = Vec::with_capacity(pop_size);

        for _ in 0..pop_size {
            let mut chrom = template.clone();
            chrom.randomize(cavity, rng);
            let score = eval_fn(&chrom.genes);
            genomes.push(Genome { chrom, score, rw_fitness: 0.0 });
        }

        let mut pop = Population { genomes, pop_size };
        pop.sort_and_evaluate();
        pop
    }

    /// Sort by score (ascending = best first) and compute roulette wheel fitness.
    fn sort_and_evaluate(&mut self) {
        self.genomes.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

        let n = self.genomes.len() as f64;
        if n < 2.0 { return; }

        // Sigma truncation: fitness = max(0, score - (mean - 2*sigma))
        // Note: we minimize score, so invert: fitness = max(0, -(score - offset))
        // where offset = mean + 2*sigma (worst reasonable score)
        let mean: f64 = self.genomes.iter().map(|g| g.score).sum::<f64>() / n;
        let var: f64 = self.genomes.iter().map(|g| (g.score - mean).powi(2)).sum::<f64>() / n;
        let sigma = var.sqrt();

        // For minimization: fitness = max(0, (mean + 2*sigma) - score)
        let offset = mean + 2.0 * sigma;

        let mut total_fitness = 0.0;
        for g in &mut self.genomes {
            let f = (offset - g.score).max(0.0);
            total_fitness += f;
            g.rw_fitness = total_fitness;
        }

        // Normalize to [0, 1]
        if total_fitness > 0.0 {
            for g in &mut self.genomes {
                g.rw_fitness /= total_fitness;
            }
        }
    }

    /// Roulette wheel selection: pick a genome based on cumulative fitness.
    fn select<R: Rng>(&self, rng: &mut R) -> usize {
        let cutoff: f64 = rng.gen();
        // Binary search on cumulative fitness
        match self.genomes.binary_search_by(|g| {
            g.rw_fitness.partial_cmp(&cutoff).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => i,
            Err(i) => i.min(self.genomes.len() - 1),
        }
    }

    /// Best score in population.
    fn best_score(&self) -> f64 {
        self.genomes.first().map(|g| g.score).unwrap_or(f64::MAX)
    }

    /// Best chromosome.
    fn best_chrom(&self) -> &RxChromosome {
        &self.genomes[0].chrom
    }

    /// Remove near-duplicates, keeping the better (lower-score) one.
    fn remove_duplicates(&mut self, threshold: f64) {
        let mut i = 0;
        while i < self.genomes.len() {
            let mut j = i + 1;
            while j < self.genomes.len() {
                if self.genomes[i].chrom.is_near_duplicate(&self.genomes[j].chrom, threshold) {
                    self.genomes.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }
}

/// Result of a GA run.
pub struct GAResult {
    pub best_chrom: RxChromosome,
    pub best_score: f64,
    pub generations: usize,
}

/// Run the genetic algorithm search.
///
/// # Arguments
/// * `eval_fn` - Scoring function: &[f64] → f64 (lower is better)
/// * `template` - Template chromosome (step sizes, n_torsions)
/// * `cavity` - Cavity for random placement
/// * `params` - GA parameters
/// * `rng` - Random number generator
pub fn ga_search<F, R>(
    eval_fn: &mut F,
    template: &RxChromosome,
    cavity: &crate::rxdock_cavity::Cavity,
    params: &GAParams,
    rng: &mut R,
) -> GAResult
where
    F: FnMut(&[f64]) -> f64,
    R: Rng,
{
    let mut pop = Population::new(template, params.pop_size, eval_fn, rng, cavity);

    let mut best_score = pop.best_score();
    let mut convergence_counter = 0;

    let n_replicates = params.pop_size / 2;
    let mut gen = 0;

    for cycle in 0..params.n_cycles {
        gen = cycle;

        if convergence_counter >= params.convergence_count {
            break;
        }

        // Generate offspring
        let mut offspring: Vec<Genome> = Vec::with_capacity(n_replicates * 2);

        for _ in 0..n_replicates {
            let mother_idx = pop.select(rng);
            let mut father_idx = pop.select(rng);
            let mut attempts = 0;
            while father_idx == mother_idx && attempts < 100 {
                father_idx = pop.select(rng);
                attempts += 1;
            }

            let (mut child1, mut child2);

            if rng.gen::<f64>() < params.crossover_prob {
                // 2-point crossover
                let (c1, c2) = RxChromosome::crossover(
                    &pop.genomes[mother_idx].chrom,
                    &pop.genomes[father_idx].chrom,
                    rng,
                );
                child1 = c1;
                child2 = c2;

                // Optional Cauchy mutation after crossover
                if params.crossover_mutation {
                    child1.cauchy_mutate(params.step_size, rng);
                    child2.cauchy_mutate(params.step_size, rng);
                }
            } else {
                // Clone and mutate
                child1 = pop.genomes[mother_idx].chrom.clone();
                child2 = pop.genomes[father_idx].chrom.clone();

                if params.cauchy_mutation {
                    child1.cauchy_mutate(params.step_size, rng);
                    child2.cauchy_mutate(params.step_size, rng);
                } else {
                    child1.rect_mutate(params.step_size, rng);
                    child2.rect_mutate(params.step_size, rng);
                }
            }

            let score1 = eval_fn(&child1.genes);
            let score2 = eval_fn(&child2.genes);

            offspring.push(Genome { chrom: child1, score: score1, rw_fitness: 0.0 });
            offspring.push(Genome { chrom: child2, score: score2, rw_fitness: 0.0 });
        }

        // Merge offspring into population
        pop.genomes.extend(offspring);
        pop.genomes.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

        // Remove near-duplicates
        pop.remove_duplicates(params.equality_threshold);

        // Keep top pop_size
        pop.genomes.truncate(params.pop_size);

        // Recalculate fitness
        pop.sort_and_evaluate();

        // Check convergence
        let new_best = pop.best_score();
        if new_best < best_score {
            best_score = new_best;
            convergence_counter = 0;
        } else {
            convergence_counter += 1;
        }
    }

    GAResult {
        best_chrom: pop.best_chrom().clone(),
        best_score: pop.best_score(),
        generations: gen,
    }
}

/// Run GA on an existing population (continuing from previous best).
/// Used in multi-stage protocol where population carries over.
pub fn ga_search_from_chrom<F, R>(
    eval_fn: &mut F,
    start_chrom: &RxChromosome,
    cavity: &crate::rxdock_cavity::Cavity,
    params: &GAParams,
    rng: &mut R,
) -> GAResult
where
    F: FnMut(&[f64]) -> f64,
    R: Rng,
{
    // Create population with start_chrom as one member, rest random
    let mut genomes = Vec::with_capacity(params.pop_size);

    let score = eval_fn(&start_chrom.genes);
    genomes.push(Genome { chrom: start_chrom.clone(), score, rw_fitness: 0.0 });

    for _ in 1..params.pop_size {
        let mut chrom = start_chrom.clone();
        chrom.randomize(cavity, rng);
        let s = eval_fn(&chrom.genes);
        genomes.push(Genome { chrom, score: s, rw_fitness: 0.0 });
    }

    let mut pop = Population { genomes, pop_size: params.pop_size };
    pop.sort_and_evaluate();

    let mut best_score = pop.best_score();
    let mut convergence_counter = 0;
    let n_replicates = params.pop_size / 2;
    let mut gen = 0;

    for cycle in 0..params.n_cycles {
        gen = cycle;
        if convergence_counter >= params.convergence_count { break; }

        let mut offspring: Vec<Genome> = Vec::with_capacity(n_replicates * 2);

        for _ in 0..n_replicates {
            let mother_idx = pop.select(rng);
            let mut father_idx = pop.select(rng);
            let mut attempts = 0;
            while father_idx == mother_idx && attempts < 100 {
                father_idx = pop.select(rng);
                attempts += 1;
            }

            let (mut child1, mut child2);

            if rng.gen::<f64>() < params.crossover_prob {
                let (c1, c2) = RxChromosome::crossover(
                    &pop.genomes[mother_idx].chrom,
                    &pop.genomes[father_idx].chrom,
                    rng,
                );
                child1 = c1;
                child2 = c2;
                if params.crossover_mutation {
                    child1.cauchy_mutate(params.step_size, rng);
                    child2.cauchy_mutate(params.step_size, rng);
                }
            } else {
                child1 = pop.genomes[mother_idx].chrom.clone();
                child2 = pop.genomes[father_idx].chrom.clone();
                if params.cauchy_mutation {
                    child1.cauchy_mutate(params.step_size, rng);
                    child2.cauchy_mutate(params.step_size, rng);
                } else {
                    child1.rect_mutate(params.step_size, rng);
                    child2.rect_mutate(params.step_size, rng);
                }
            }

            let score1 = eval_fn(&child1.genes);
            let score2 = eval_fn(&child2.genes);
            offspring.push(Genome { chrom: child1, score: score1, rw_fitness: 0.0 });
            offspring.push(Genome { chrom: child2, score: score2, rw_fitness: 0.0 });
        }

        pop.genomes.extend(offspring);
        pop.genomes.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        pop.remove_duplicates(params.equality_threshold);
        pop.genomes.truncate(params.pop_size);
        pop.sort_and_evaluate();

        let new_best = pop.best_score();
        if new_best < best_score {
            best_score = new_best;
            convergence_counter = 0;
        } else {
            convergence_counter += 1;
        }
    }

    GAResult {
        best_chrom: pop.best_chrom().clone(),
        best_score: pop.best_score(),
        generations: gen,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rxdock_cavity::Cavity;
    use crate::common::Vec3;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_ga_finds_minimum() {
        // Minimize f(genes) where we only care about genes[0..2] = position (x,y,z)
        // f = (x-3)^2 + (y+2)^2 + z^2
        // Minimum at (3, -2, 0)
        let mut eval = |genes: &[f64]| -> f64 {
            let x = genes[0];
            let y = genes[1];
            let z = genes[2];
            (x - 3.0).powi(2) + (y + 2.0).powi(2) + z * z
        };

        let cavity = Cavity {
            center: Vec3::new(3.0, -2.0, 0.0),
            min_coord: Vec3::new(-5.0, -10.0, -5.0),
            max_coord: Vec3::new(10.0, 5.0, 5.0),
            volume: 100.0,
            coords: Vec::new(),
        };

        // Use proper 6+0 chromosome (pos + orientation, no torsions)
        let template = RxChromosome::new(0);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let params = GAParams {
            pop_size: 20,
            n_cycles: 50,
            ..Default::default()
        };

        let result = ga_search(&mut eval, &template, &cavity, &params, &mut rng);
        assert!(result.best_score < 5.0, "GA should find near-minimum, got {}", result.best_score);
    }
}
