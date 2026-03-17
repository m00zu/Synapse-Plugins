//! rxDock search orchestrator: multi-stage GA + SA + Simplex protocol.
//!
//! Implements the full dock.json protocol: GA with ramping parameters
//! (slope 1→3→5→10), then SA at 10K, then Simplex refinement.
//! Runs multiple independent protocols in parallel using Rayon.

use crate::common::*;
use crate::conf::*;
use crate::model::Model;
use crate::rxdock_atom::*;
use crate::rxdock_cavity::Cavity;
use crate::rxdock_chrom::RxChromosome;
use crate::rxdock_ga::*;
use crate::rxdock_sa::*;
use crate::rxdock_scoring::*;
use crate::rxdock_simplex::*;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Parameters for the full rxDock search pipeline.
#[derive(Debug, Clone)]
pub struct RxDockParams {
    pub n_runs: usize,
    pub n_poses: usize,
    pub min_rmsd: f64,
    pub seed: u64,
    /// Use full multi-stage protocol (GA+SA+Simplex). If false, use SA+Simplex only.
    pub use_protocol: bool,
}

impl Default for RxDockParams {
    fn default() -> Self {
        RxDockParams {
            n_runs: 20,
            n_poses: 9,
            min_rmsd: 1.0,
            seed: 42,
            use_protocol: true,
        }
    }
}

/// Progress callback: (percent_complete: f64, best_energy: f64)
pub type RxProgressCallback = Box<dyn Fn(f64, f64) + Send + Sync>;

/// Result of a single docking run.
struct RunResult {
    energy: f64,
    chrom: RxChromosome,
    coords: Vec<Vec3>,
    terms: RxScoreTerms,
}

/// Stage parameters for the multi-stage protocol.
struct StageParams {
    use_4_8: bool,
    ecut: f64,
    cavity_weight: f64,
    polar_dr12_max: f64,
    polar_da1_max: f64,
    polar_da2_max: f64,
}

/// Run the full multi-stage protocol for a single starting pose.
/// Protocol: RandPop→GA(slope-1)→GA(slope-3)→GA(slope-5)→SetSlope10→SA→Simplex
fn run_protocol<R: rand::Rng>(
    model: &Model,
    scoring: &RxScoringFunction,
    cavity: &Cavity,
    n_torsions: usize,
    rng: &mut R,
) -> RunResult {
    // Protocol stages matching dock.json
    let stages = [
        // Stage 1: slope-1 (very soft, 4-8, broad angles)
        StageParams { use_4_8: true, ecut: 1.0, cavity_weight: 5.0,
            polar_dr12_max: 1.5, polar_da1_max: 180.0, polar_da2_max: 180.0 },
        // Stage 2: slope-3
        StageParams { use_4_8: true, ecut: 5.0, cavity_weight: 5.0,
            polar_dr12_max: 1.2, polar_da1_max: 140.0, polar_da2_max: 140.0 },
        // Stage 3: slope-5 (switch to 6-12)
        StageParams { use_4_8: false, ecut: 25.0, cavity_weight: 5.0,
            polar_dr12_max: 0.9, polar_da1_max: 120.0, polar_da2_max: 120.0 },
        // Stage 4: slope-10 (full strength)
        StageParams { use_4_8: false, ecut: 120.0, cavity_weight: 5.0,
            polar_dr12_max: 0.6, polar_da1_max: 80.0, polar_da2_max: 100.0 },
    ];

    // Create mutable scoring function for this run (stage-specific VDW tables)
    let mut local_model = model.clone();
    let mut current_chrom = RxChromosome::new(n_torsions);
    current_chrom.randomize(cavity, rng);

    let ga_params = GAParams {
        pop_size: 50,
        n_cycles: 100,
        convergence_count: 6,
        crossover_prob: 0.4,
        crossover_mutation: true,
        cauchy_mutation: false,
        step_size: 1.0,
        equality_threshold: 0.1,
    };

    // Run GA for each slope stage
    for stage in &stages {
        // Build stage-specific VDW tables (inter uses GOLD, intra uses Tripos)
        let vdw_inter_table = TriposVdwTable::new_gold(stage.use_4_8, 1.5, stage.ecut);
        let vdw_intra_table = TriposVdwTable::new(stage.use_4_8, 1.5, stage.ecut);

        // Build stage-specific scoring function
        let mut stage_weights = scoring.weights.clone();
        stage_weights.cavity = stage.cavity_weight;

        // Build stage-specific polar params with ramped angles
        let mut polar_attr = PolarSFParams::attractive();
        polar_attr.dist_ramp.dr_max = stage.polar_dr12_max;
        polar_attr.angle1_ramp.dr_max = stage.polar_da1_max;
        polar_attr.angle2_ramp.dr_max = stage.polar_da2_max;

        let mut polar_repul = PolarSFParams::repulsive();
        polar_repul.dist_ramp.dr_max = stage.polar_dr12_max.min(1.1); // repulsive keeps min of stage/default

        let stage_sf = RxScoringFunction {
            vdw_inter_table,
            vdw_intra_table,
            weights: stage_weights,
            receptor_atoms: scoring.receptor_atoms.clone(),
            receptor_grid: SpatialGrid::new(&scoring.receptor_atoms, 3.0),
            cavity: scoring.cavity.clone(),
            polar_grid: scoring.polar_grid.as_ref().map(|pg| {
                // Reuse existing polar grid (receptor doesn't change)
                PolarGrid {
                    donors: pg.donors.clone(),
                    acceptors: pg.acceptors.clone(),
                    donor_grid: SpatialGrid::new(
                        &pg.donors.iter().map(|ic| RxReceptorAtom { coords: ic.coords, sy: 0 }).collect::<Vec<_>>(),
                        3.0,
                    ),
                    acceptor_grid: SpatialGrid::new(
                        &pg.acceptors.iter().map(|ic| RxReceptorAtom { coords: ic.coords, sy: 0 }).collect::<Vec<_>>(),
                        3.0,
                    ),
                }
            }),
            polar_attr_params: polar_attr,
            polar_repul_params: polar_repul,
        };

        // Eval closure for this stage
        let mut eval_fn = |genes: &[f64]| -> f64 {
            let temp_chrom = RxChromosome {
                genes: genes.to_vec(),
                step_sizes: current_chrom.step_sizes.clone(),
                n_torsions,
            };
            let conf = temp_chrom.to_conf();
            local_model.set(&conf);
            stage_sf.score(&local_model)
        };

        let ga_result = ga_search_from_chrom(
            &mut eval_fn,
            &current_chrom,
            cavity,
            &ga_params,
            rng,
        );

        current_chrom = ga_result.best_chrom;
    }

    // Phase 5: Simulated Annealing at 10K (isothermal)
    let sa_params = SimAnnParams {
        start_temp: 10.0,
        final_temp: 10.0,
        num_blocks: 5,
        block_length: 50,
        min_acc_rate: 0.25,
        max_acc_rate: 0.75,
        step_factor: 0.5, // halve step size if acceptance too low
    };

    // Use final scoring function (full strength) for SA and Simplex
    let mut eval_fn = |genes: &[f64]| -> f64 {
        let temp_chrom = RxChromosome {
            genes: genes.to_vec(),
            step_sizes: current_chrom.step_sizes.clone(),
            n_torsions,
        };
        let conf = temp_chrom.to_conf();
        local_model.set(&conf);
        scoring.score(&local_model)
    };

    let sa_result = simulated_annealing(
        &mut eval_fn,
        &current_chrom,
        &sa_params,
        rng,
        None::<&mut fn(usize, usize, f64)>,
    );

    // Phase 6: Simplex refinement
    let simplex_params = SimplexParams {
        max_calls: 200,
        convergence: 0.001,
        ..Default::default()
    };

    let (refined_genes, _refined_energy) = nelder_mead_minimize(
        &mut eval_fn,
        &sa_result.best_chrom.genes,
        &sa_result.best_chrom.step_sizes,
        &simplex_params,
    );

    // Final: revert cavity weight to 1.0 and re-score
    let best_chrom = RxChromosome {
        genes: refined_genes,
        step_sizes: sa_result.best_chrom.step_sizes.clone(),
        n_torsions,
    };
    let conf = best_chrom.to_conf();
    local_model.set(&conf);

    // Score with final weights (cavity=1.0)
    let terms = scoring.score_terms(&local_model);
    let coords = local_model.get_heavy_atom_movable_coords();

    RunResult {
        energy: terms.total,
        chrom: best_chrom,
        coords,
        terms,
    }
}

/// Run simple SA → Simplex protocol (no GA, for backward compatibility).
fn run_simple<R: rand::Rng>(
    model: &Model,
    scoring: &RxScoringFunction,
    cavity: &Cavity,
    n_torsions: usize,
    rng: &mut R,
) -> RunResult {
    let mut local_model = model.clone();
    let mut chrom = RxChromosome::new(n_torsions);
    chrom.randomize(cavity, rng);

    let mut eval_fn = |genes: &[f64]| -> f64 {
        let temp_chrom = RxChromosome {
            genes: genes.to_vec(),
            step_sizes: chrom.step_sizes.clone(),
            n_torsions,
        };
        let conf = temp_chrom.to_conf();
        local_model.set(&conf);
        scoring.score(&local_model)
    };

    let sa_params = SimAnnParams {
        start_temp: 300.0,
        final_temp: 1.0,
        num_blocks: 25,
        block_length: 50,
        ..Default::default()
    };

    let sa_result = simulated_annealing(
        &mut eval_fn,
        &chrom,
        &sa_params,
        rng,
        None::<&mut fn(usize, usize, f64)>,
    );

    let simplex_params = SimplexParams {
        max_calls: 200,
        convergence: 1e-4,
        ..Default::default()
    };

    let (refined_genes, _refined_energy) = nelder_mead_minimize(
        &mut eval_fn,
        &sa_result.best_chrom.genes,
        &sa_result.best_chrom.step_sizes,
        &simplex_params,
    );

    let best_chrom = RxChromosome {
        genes: refined_genes,
        step_sizes: sa_result.best_chrom.step_sizes.clone(),
        n_torsions,
    };
    let conf = best_chrom.to_conf();
    local_model.set(&conf);

    let terms = scoring.score_terms(&local_model);
    let coords = local_model.get_heavy_atom_movable_coords();

    RunResult {
        energy: terms.total,
        chrom: best_chrom,
        coords,
        terms,
    }
}

/// Run the full rxDock search: parallel multi-stage protocol, with RMSD clustering.
///
/// Returns sorted OutputContainer (best energy first).
pub fn rxdock_search(
    model: &Model,
    scoring: &RxScoringFunction,
    cavity: &Cavity,
    params: &RxDockParams,
    progress_cb: Option<Arc<RxProgressCallback>>,
) -> OutputContainer {
    let n_runs = params.n_runs;

    let task_progress: Arc<Vec<AtomicU32>> = Arc::new(
        (0..n_runs).map(|_| AtomicU32::new(0)).collect()
    );

    if model.ligands.is_empty() {
        return Vec::new();
    }
    let n_torsions = model.ligands[0].degrees_of_freedom;

    let results: Vec<RunResult> = (0..n_runs)
        .into_par_iter()
        .map(|run_idx| {
            let mut rng = ChaCha8Rng::seed_from_u64(params.seed + run_idx as u64);

            let result = if params.use_protocol {
                run_protocol(model, scoring, cavity, n_torsions, &mut rng)
            } else {
                run_simple(model, scoring, cavity, n_torsions, &mut rng)
            };

            // Report progress
            let tp = Arc::clone(&task_progress);
            tp[run_idx].store(10000, Ordering::Relaxed);
            if let Some(ref user_cb) = progress_cb {
                let total_pct: u32 = tp.iter().map(|a| a.load(Ordering::Relaxed)).sum();
                let avg = total_pct as f64 / (n_runs as f64 * 10000.0) * 100.0;
                user_cb(avg, result.energy);
            }

            result
        })
        .collect();

    // Sort by energy and cluster by RMSD
    let mut sorted_results = results;
    sorted_results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(std::cmp::Ordering::Equal));

    let mut output: OutputContainer = Vec::new();

    for result in &sorted_results {
        let conf = result.chrom.to_conf();
        let out = OutputType {
            c: conf,
            e: result.energy,
            lb: 0.0,
            ub: 0.0,
            // Store rxDock-specific decomposition
            inter: result.terms.vdw_inter + result.terms.polar_attr + result.terms.polar_repul,
            intra: result.terms.vdw_intra + result.terms.dihedral,
            conf_independent: result.terms.const_penalty + result.terms.rot_entropy,
            unbound: result.terms.cavity,
            total: result.energy,
            coords: result.coords.clone(),
        };
        add_to_output_container(&mut output, &out, params.min_rmsd, params.n_poses);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rxdock_params_default() {
        let p = RxDockParams::default();
        assert_eq!(p.n_runs, 20);
        assert_eq!(p.n_poses, 9);
        assert!((p.min_rmsd - 1.0).abs() < 1e-10);
        assert!(p.use_protocol);
    }
}
