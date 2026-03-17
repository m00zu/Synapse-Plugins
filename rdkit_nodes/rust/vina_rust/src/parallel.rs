use crate::cache::IGrid;
use crate::common::*;
use crate::conf::*;
use crate::model::Model;
use crate::monte_carlo::*;
use crate::precalculate::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

// ─── Search Algorithm Selection ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchAlgorithm {
    Vina,
    QVina2,
}

// ─── Parallel MC (Vina) ─────────────────────────────────────────────────────

/// Run parallel Monte Carlo search using Rayon (standard Vina)
pub fn parallel_mc(
    model: &Model,
    p: &PrecalculateByAtom,
    ig: Arc<dyn IGrid>,
    corner1: &Vec3,
    corner2: &Vec3,
    params: &MonteCarloParams,
    num_tasks: usize,
    seed: u64,
    n_poses: usize,
    min_rmsd_merge: f64,
    progress_cb: Option<Arc<ProgressCallback>>,
) -> OutputContainer {
    parallel_mc_impl(model, p, ig, corner1, corner2, params, num_tasks, seed, n_poses, min_rmsd_merge, progress_cb, SearchAlgorithm::Vina)
}

/// Run parallel Monte Carlo search using Rayon (QVina2)
pub fn parallel_mc_qvina2(
    model: &Model,
    p: &PrecalculateByAtom,
    ig: Arc<dyn IGrid>,
    corner1: &Vec3,
    corner2: &Vec3,
    params: &MonteCarloParams,
    num_tasks: usize,
    seed: u64,
    n_poses: usize,
    min_rmsd_merge: f64,
    progress_cb: Option<Arc<ProgressCallback>>,
) -> OutputContainer {
    parallel_mc_impl(model, p, ig, corner1, corner2, params, num_tasks, seed, n_poses, min_rmsd_merge, progress_cb, SearchAlgorithm::QVina2)
}

fn parallel_mc_impl(
    model: &Model,
    p: &PrecalculateByAtom,
    ig: Arc<dyn IGrid>,
    corner1: &Vec3,
    corner2: &Vec3,
    params: &MonteCarloParams,
    num_tasks: usize,
    seed: u64,
    n_poses: usize,
    min_rmsd_merge: f64,
    progress_cb: Option<Arc<ProgressCallback>>,
    algorithm: SearchAlgorithm,
) -> OutputContainer {
    let p_arc = Arc::new(p.clone());
    let params_arc = Arc::new(params.clone());
    let c1 = *corner1;
    let c2 = *corner2;
    let model_base = model.clone();

    // Pre-compute QVina2 heuristics if needed
    let qvina2_heuristics = if algorithm == SearchAlgorithm::QVina2 {
        Some(Arc::new(QVina2Heuristics::from_model(&model_base)))
    } else {
        None
    };

    if algorithm == SearchAlgorithm::QVina2 {
        if let Some(h) = &qvina2_heuristics {
            eprintln!(
                "[QVina2] tasks={}, steps_per_task={}, bfgs_max_steps={}, movable_atoms={}",
                num_tasks, h.num_steps, h.max_bfgs_steps, model_base.num_movable_atoms()
            );
        }
    }

    // Shared per-task progress: each slot holds 0–10000 (0.00%–100.00%).
    // The aggregating callback averages all slots so the reported percent_complete
    // increases monotonically instead of jumping between tasks.
    let task_progress: Arc<Vec<AtomicU32>> = Arc::new(
        (0..num_tasks).map(|_| AtomicU32::new(0)).collect()
    );

    // Build a per-task callback that updates the atomic slot then fires the
    // user callback with the true aggregate percentage.
    let aggregating_cb: Option<Arc<ProgressCallback>> = progress_cb.as_ref().map(|user_cb| {
        let tp = Arc::clone(&task_progress);
        let uc = Arc::clone(user_cb);
        let nt = num_tasks;
        let cb: ProgressCallback = Box::new(move |prog: &DockingProgress| {
            // Update this task's slot
            let pct_u32 = (prog.percent_complete.clamp(0.0, 100.0) * 100.0) as u32;
            tp[prog.mc_run].store(pct_u32, Ordering::Relaxed);
            // Compute average across all tasks
            let total: u32 = tp.iter().map(|a| a.load(Ordering::Relaxed)).sum();
            let avg_pct = total as f64 / (nt as f64 * 100.0);
            // Forward with corrected percent_complete
            let mut aggregated = prog.clone();
            aggregated.percent_complete = avg_pct;
            uc(&aggregated);
        });
        Arc::new(cb)
    });

    // Run MC tasks in parallel
    let results: Vec<OutputContainer> = (0..num_tasks)
        .into_par_iter()
        .map(|task_idx| {
            let mut task_model = model_base.clone();
            let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(task_idx as u64));
            let task_p = &*p_arc;
            let task_ig = &*ig;
            let task_params = &*params_arc;

            let cb_ref = aggregating_cb.as_ref().map(|cb| cb.as_ref());

            match algorithm {
                SearchAlgorithm::Vina => {
                    monte_carlo_search(
                        &mut task_model, task_p, task_ig,
                        &c1, &c2, task_params, &mut rng,
                        task_idx, num_tasks, cb_ref,
                    )
                }
                SearchAlgorithm::QVina2 => {
                    let h = qvina2_heuristics.as_ref().unwrap();
                    monte_carlo_search_qvina2(
                        &mut task_model, task_p, task_ig,
                        &c1, &c2, task_params, h,
                        &mut rng, task_idx, num_tasks, cb_ref,
                    )
                }
            }
        })
        .collect();

    // Merge results with RMSD clustering
    // C++ overrides min_rmsd to 2.0 during cross-task merge (parallel_mc.cpp:59)
    let merge_rmsd = 2.0_f64;
    let mut merged: OutputContainer = Vec::new();
    for task_out in results {
        for pose in task_out {
            add_to_output_container(&mut merged, &pose, merge_rmsd, n_poses);
        }
    }

    // Sort by energy
    merged.sort_by(|a, b| a.e.partial_cmp(&b.e).unwrap_or(std::cmp::Ordering::Equal));

    // Report completion
    let total_steps = if algorithm == SearchAlgorithm::QVina2 {
        qvina2_heuristics.as_ref().map(|h| h.num_steps).unwrap_or(params.global_steps)
    } else {
        params.global_steps
    };

    if let Some(cb) = &progress_cb {
        cb(&DockingProgress {
            stage: DockingStage::Complete,
            mc_run: num_tasks,
            total_mc_runs: num_tasks,
            step: total_steps,
            total_steps,
            best_energy: merged.first().map(|p| p.e).unwrap_or(0.0),
            current_energy: 0.0,
            poses_found: merged.len(),
            percent_complete: 100.0,
            result_pdbqt: String::new(),
            result_energies: Vec::new(),
        });
    }

    merged
}
