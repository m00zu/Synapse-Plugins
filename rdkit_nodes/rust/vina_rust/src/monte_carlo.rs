use crate::bfgs::BfgsScratch;
use crate::cache::IGrid;
use crate::common::*;
use crate::conf::*;
use crate::model::Model;
use crate::precalculate::*;
use crate::visited::{Visited, VisitedScratch};
use rand::Rng;
use std::f64::consts::PI;

// ─── Monte Carlo Parameters ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MonteCarloParams {
    pub global_steps: u32,
    pub local_steps: u32,
    pub temperature: f64,
    pub hunt_cap: Vec3,      // BFGS search radius
    pub min_rmsd: f64,       // minimum RMSD between saved poses
    pub num_saved_mins: usize,
    pub mutation_amplitude: f64,
    pub max_evals: i32,
}

impl MonteCarloParams {
    pub fn new() -> Self {
        MonteCarloParams {
            global_steps: 2500,
            local_steps: 300, // default; overridden to (25+num_movable)/3 at call site
            temperature: 1.2,
            hunt_cap: Vec3::new(10.0, 1.5, 10.0),
            min_rmsd: 1.0,
            num_saved_mins: 20,
            mutation_amplitude: 2.0,
            max_evals: 0,
        }
    }
}

impl Default for MonteCarloParams {
    fn default() -> Self { Self::new() }
}

// ─── Progress Reporting ────────────────────────────────────────────────────────

/// Detailed progress info for each callback
#[derive(Debug, Clone)]
pub struct DockingProgress {
    pub stage: DockingStage,
    pub mc_run: usize,
    pub total_mc_runs: usize,
    pub step: u32,
    pub total_steps: u32,
    pub best_energy: f64,
    pub current_energy: f64,
    pub poses_found: usize,
    pub percent_complete: f64,
    /// Populated only for `LigandDone` in batch mode: the docked poses PDBQT.
    pub result_pdbqt: String,
    /// Populated only for `LigandDone` in batch mode: per-pose energies.
    pub result_energies: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DockingStage {
    PreparingMaps,
    Searching,
    Refining,
    Complete,
    /// Fired in batch mode after refinement + re-evaluation for a single ligand.
    LigandDone,
}

impl DockingProgress {
    pub fn searching(run: usize, total_runs: usize, step: u32, total_steps: u32, best_e: f64, cur_e: f64, poses: usize) -> Self {
        // Report this task's own progress (0–100%).
        // The parallel runner aggregates across tasks for the true overall %.
        let task_pct = step as f64 / total_steps as f64 * 100.0;
        DockingProgress {
            stage: DockingStage::Searching,
            mc_run: run,
            total_mc_runs: total_runs,
            step,
            total_steps,
            best_energy: best_e,
            current_energy: cur_e,
            poses_found: poses,
            percent_complete: task_pct,
            result_pdbqt: String::new(),
            result_energies: Vec::new(),
        }
    }
}

pub type ProgressCallback = Box<dyn Fn(&DockingProgress) + Send + Sync>;

// ─── Mutation ──────────────────────────────────────────────────────────────────

/// Count total mutable entities (position + orientation + torsions for each ligand + flex torsions)
fn count_mutable_entities(c: &Conf) -> usize {
    let mut count = 0;
    for l in &c.ligands {
        count += 2 + l.torsions.len(); // position + orientation + each torsion
    }
    for f in &c.flex {
        count += f.torsions.len();
    }
    count
}

/// Mutate one random DOF of the conformation
pub fn mutate_conf<R: Rng>(c: &mut Conf, model: &Model, amplitude: f64, rng: &mut R) {
    let mutable_num = count_mutable_entities(c);
    if mutable_num == 0 { return; }

    let which_int = random_int(0, (mutable_num - 1) as i32, rng);
    let mut which = which_int as usize;

    for (i, lig_conf) in c.ligands.iter_mut().enumerate() {
        // Position mutation
        if which == 0 {
            lig_conf.rigid.position += random_inside_sphere(rng) * amplitude;
            return;
        }
        which -= 1;

        // Orientation mutation
        if which == 0 {
            let gr = model.gyration_radius(i);
            if gr > EPSILON_FL {
                let rotation = random_inside_sphere(rng) * (amplitude / gr);
                lig_conf.rigid.orientation.increment(&rotation);
            }
            return;
        }
        which -= 1;

        // Torsion mutations
        if which < lig_conf.torsions.len() {
            lig_conf.torsions[which] = random_fl(-PI, PI, rng);
            return;
        }
        which -= lig_conf.torsions.len();
    }

    // Flex residue torsions
    for flex_conf in c.flex.iter_mut() {
        if which < flex_conf.torsions.len() {
            flex_conf.torsions[which] = random_fl(-PI, PI, rng);
            return;
        }
        which -= flex_conf.torsions.len();
    }
}

// ─── Metropolis Criterion ──────────────────────────────────────────────────────

#[inline(always)]
pub fn metropolis_accept<R: Rng>(old_f: f64, new_f: f64, temperature: f64, rng: &mut R) -> bool {
    if new_f < old_f { return true; }
    let acceptance_probability = ((old_f - new_f) / temperature).exp();
    random_fl(0.0, 1.0, rng) < acceptance_probability
}

// ─── Monte Carlo Search ────────────────────────────────────────────────────────

pub fn monte_carlo_search(
    model: &mut Model,
    p: &PrecalculateByAtom,
    ig: &dyn IGrid,
    corner1: &Vec3,
    corner2: &Vec3,
    params: &MonteCarloParams,
    rng: &mut impl Rng,
    run_index: usize,
    total_runs: usize,
    progress_cb: Option<&ProgressCallback>,
) -> OutputContainer {
    let mut out: OutputContainer = Vec::new();
    let mut evalcount: i32 = 0;

    let authentic_v = Vec3::new(1000.0, 1000.0, 1000.0);
    let s = model.get_size();
    let mut g = Change::new(&s);
    let mut tmp = OutputType::new(&s, 0.0);
    let mut candidate = OutputType::new(&s, 0.0);

    // Pre-allocate BFGS scratch buffers (reused across all BFGS calls)
    let mut scratch = BfgsScratch::new(&tmp.c, &g);

    tmp.c.randomize(corner1, corner2, rng);
    let mut best_e = MAX_FL;

    let max_steps = params.local_steps;
    let progress_interval = (params.global_steps / 100).max(1);

    for step in 0..params.global_steps {
        if params.max_evals > 0 && evalcount > params.max_evals {
            break;
        }

        candidate.copy_from(&tmp);
        mutate_conf(&mut candidate.c, model, params.mutation_amplitude, rng);

        // Local optimization with hunt_cap
        model.quasi_newton_optimize(p, ig, &mut candidate, &mut g, &params.hunt_cap, &mut evalcount, max_steps, &mut scratch);

        if step == 0 || metropolis_accept(tmp.e, candidate.e, params.temperature, rng) {
            tmp.copy_from(&candidate);

            if tmp.e < best_e || out.len() < params.num_saved_mins {
                // Full optimization for promising poses
                model.quasi_newton_optimize(p, ig, &mut tmp, &mut g, &authentic_v, &mut evalcount, max_steps, &mut scratch);
                tmp.coords = model.get_heavy_atom_movable_coords();
                add_to_output_container(&mut out, &tmp, params.min_rmsd, params.num_saved_mins);
                if tmp.e < best_e {
                    best_e = tmp.e;
                }
            }
        }

        // Report roughly every 1% to avoid callback/GIL overhead dominating runtime.
        if step % progress_interval == 0 {
            if let Some(cb) = progress_cb {
                cb(&DockingProgress::searching(
                    run_index, total_runs,
                    step, params.global_steps,
                    best_e, tmp.e, out.len(),
                ));
            }
        }
    }

    // Final progress report
    if let Some(cb) = progress_cb {
        cb(&DockingProgress::searching(
            run_index, total_runs,
            params.global_steps, params.global_steps,
            best_e, tmp.e, out.len(),
        ));
    }

    assert!(!out.is_empty());
    out
}

// ─── QVina2 Monte Carlo Search ──────────────────────────────────────────────

/// QVina2 heuristic parameters, computed from model size
pub struct QVina2Heuristics {
    pub num_steps: u32,
    pub max_bfgs_steps: u32,
    pub hunt_cap: Vec3,
    pub min_rmsd: f64,
    pub num_saved_mins: usize,
}

impl QVina2Heuristics {
    pub fn from_model(model: &Model) -> Self {
        let num_movable = model.num_movable_atoms();
        let s = model.get_size();
        let num_dof = s.num_degrees_of_freedom();
        let heuristic = num_movable + 10 * num_dof;

        QVina2Heuristics {
            num_steps: (70 * 3 * (50 + heuristic) / 2) as u32,
            max_bfgs_steps: ((25 + num_movable) / 3) as u32,
            // Match qvina-qvina2 main.cpp override: par.mc.hunt_cap = vec(10, 10, 10)
            hunt_cap: Vec3::new(10.0, 10.0, 10.0),
            min_rmsd: 1.0,
            num_saved_mins: 20,
        }
    }
}

/// QVina2 Monte Carlo search — per-thread Visited history + hunt→dock (matching C++ design)
pub fn monte_carlo_search_qvina2(
    model: &mut Model,
    p: &PrecalculateByAtom,
    ig: &dyn IGrid,
    corner1: &Vec3,
    corner2: &Vec3,
    params: &MonteCarloParams,
    heuristics: &QVina2Heuristics,
    rng: &mut impl Rng,
    run_index: usize,
    total_runs: usize,
    progress_cb: Option<&ProgressCallback>,
) -> OutputContainer {
    let mut out: OutputContainer = Vec::new();
    let mut evalcount: i32 = 0;

    let authentic_v = Vec3::new(1000.0, 1000.0, 1000.0);
    let s = model.get_size();
    let mut g = Change::new(&s);
    let mut tmp = OutputType::new(&s, 0.0);
    let mut candidate = OutputType::new(&s, 0.0);

    // Per-thread Visited + scratch (no sharing, no locking — matches C++ QVina2)
    let mut visited = Visited::new();
    let mut visited_scratch = VisitedScratch::new();
    let mut scratch = BfgsScratch::new(&tmp.c, &g);

    tmp.c.randomize(corner1, corner2, rng);
    let mut best_e = MAX_FL;

    let num_steps = heuristics.num_steps;
    let max_bfgs_steps = heuristics.max_bfgs_steps;
    let progress_interval = (num_steps / 100).max(1);

    for step in 0..num_steps {
        if params.max_evals > 0 && evalcount > params.max_evals {
            break;
        }

        candidate.copy_from(&tmp);
        mutate_conf(&mut candidate.c, model, params.mutation_amplitude, rng);

        // Hunt phase: restricted bounds with per-thread visited history
        model.quasi_newton_optimize_qvina2(
            p, ig, &mut candidate, &mut g, &heuristics.hunt_cap,
            &mut evalcount, max_bfgs_steps, &mut visited, &mut visited_scratch, &mut scratch,
        );

        if step == 0 || metropolis_accept(tmp.e, candidate.e, params.temperature, rng) {
            tmp.copy_from(&candidate);

            if tmp.e < best_e || out.len() < heuristics.num_saved_mins {
                // Dock phase: full refinement with per-thread visited history
                model.quasi_newton_optimize_qvina2(
                    p, ig, &mut tmp, &mut g, &authentic_v,
                    &mut evalcount, max_bfgs_steps, &mut visited, &mut visited_scratch, &mut scratch,
                );
                tmp.coords = model.get_heavy_atom_movable_coords();
                add_to_output_container(&mut out, &tmp, heuristics.min_rmsd, heuristics.num_saved_mins);
                if tmp.e < best_e {
                    best_e = tmp.e;
                }
            }
        }

        // Report roughly every 1% to avoid callback/GIL overhead dominating runtime.
        if step % progress_interval == 0 {
            if let Some(cb) = progress_cb {
                cb(&DockingProgress::searching(
                    run_index, total_runs,
                    step, num_steps,
                    best_e, tmp.e, out.len(),
                ));
            }
        }
    }

    // Final progress report
    if let Some(cb) = progress_cb {
        cb(&DockingProgress::searching(
            run_index, total_runs,
            num_steps, num_steps,
            best_e, tmp.e, out.len(),
        ));
    }

    if out.is_empty() {
        // Fallback: add the best we found
        out.push(tmp);
    }

    out
}
