use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::sync::Arc;

use crate::atom::*;
use crate::bfgs;
use crate::cache::*;
use crate::common::*;
use crate::conf::*;
use crate::grid::*;
use crate::model::*;
use crate::monte_carlo::*;
use crate::parallel::*;
use crate::parse_pdbqt::*;
use crate::parse_sdf;
use crate::precalculate::*;
use crate::rxdock_atom::*;
use crate::rxdock_cavity;
use crate::rxdock_scoring::*;
use crate::rxdock_search;
use crate::scoring::*;

// ─── Shared helpers ─────────────────────────────────────────────────────────

fn resolve_sf(sf_name: &str) -> PyResult<(ScoringFunctionChoice, Vec<f64>)> {
    let sf_choice = match sf_name {
        "vina" => ScoringFunctionChoice::Vina,
        "vinardo" => ScoringFunctionChoice::Vinardo,
        "ad4" => ScoringFunctionChoice::AD42,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown scoring function: {} (choices: vina, vinardo, ad4)", sf_name)
        )),
    };
    let weights = match sf_choice {
        ScoringFunctionChoice::Vina => default_vina_weights(),
        ScoringFunctionChoice::Vinardo => default_vinardo_weights(),
        ScoringFunctionChoice::AD42 => default_ad4_weights(),
    };
    Ok((sf_choice, weights))
}

fn resolve_cpu(cpu: i32) -> usize {
    if cpu <= 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        cpu as usize
    }
}

fn resolve_seed(seed: i64) -> u64 {
    if seed == 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(42)
    } else {
        seed as u64
    }
}

fn build_progress_callback(py: Python<'_>, py_cb: PyObject) -> Arc<ProgressCallback> {
    let cb = py_cb.clone_ref(py);
    Arc::new(Box::new(move |progress: &DockingProgress| {
        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            let _ = kwargs.set_item("stage", format!("{:?}", progress.stage));
            let _ = kwargs.set_item("mc_run", progress.mc_run);
            let _ = kwargs.set_item("total_mc_runs", progress.total_mc_runs);
            let _ = kwargs.set_item("step", progress.step);
            let _ = kwargs.set_item("total_steps", progress.total_steps);
            let _ = kwargs.set_item("best_energy", progress.best_energy);
            let _ = kwargs.set_item("current_energy", progress.current_energy);
            let _ = kwargs.set_item("poses_found", progress.poses_found);
            let _ = kwargs.set_item("percent_complete", progress.percent_complete);
            let _ = cb.call(py, (), Some(&kwargs));
        });
    }) as Box<dyn Fn(&DockingProgress) + Send + Sync>)
}

fn do_set_receptor(
    receptor: &mut Model,
    rigid_name: &str,
    flex_name: Option<&str>,
) -> PyResult<()> {
    *receptor = parse_receptor(rigid_name, flex_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(())
}

fn do_set_ligand(
    model: &mut Model,
    receptor: &Model,
    sf_choice: ScoringFunctionChoice,
    weights: &[f64],
    scoring_function: &mut Option<ScoringFunction>,
    precalculated: &mut Option<PrecalculateByAtom>,
    unbound_energy: &mut f64,
    source: LigandSource<'_>,
) -> PyResult<()> {
    *model = receptor.clone();
    match source {
        LigandSource::File(path) => {
            parse_ligand_file(model, path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        }
        LigandSource::String(s) => {
            parse_ligand_string(model, s)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        }
    }
    *scoring_function = Some(ScoringFunction::new(sf_choice, weights.to_vec()));
    *precalculated = Some(PrecalculateByAtom::new(scoring_function.as_ref().unwrap(), 32.0));
    // Compute unbound intramolecular energy (used in VINA RESULT = conf_independent(e - unbound))
    *unbound_energy = model.eval_intra(precalculated.as_ref().unwrap(), 1000.0);
    Ok(())
}

enum LigandSource<'a> {
    File(&'a str),
    String(&'a str),
}

/// Refine MC search results using exact pair-by-pair evaluation (NonCache).
/// Matches C++ refine_structure: escalating slope (100 → 10000 → 1e6 → 1e8 → 1e10)
/// with up to 5 passes, rejecting poses that remain outside the search box.
fn refine_poses(
    model: &mut Model,
    p: &PrecalculateByAtom,
    nc: &mut NonCache,
    poses: &mut OutputContainer,
) {
    let authentic_v = Vec3::new(1000.0, 1000.0, 1000.0);
    let s = model.get_size();
    let mut g = Change::new(&s);
    let mut evalcount: i32 = 0;
    // C++ Vina: refine_structure uses ssd_par.evals = (25 + num_movable_atoms) / 3
    let max_steps = ((25 + model.num_movable_atoms()) / 3) as u32;

    // Pre-allocate scratch once for all refinement calls
    let dummy_conf = Conf::new(&s);
    let mut scratch = bfgs::BfgsScratch::new(&dummy_conf, &g);

    let slope_orig = nc.slope;

    for pose in poses.iter_mut() {
        // C++ refine_structure: escalating slope penalty to push atoms into the box
        for pass in 0..5u32 {
            nc.slope = 100.0 * 10.0_f64.powf(2.0 * pass as f64);
            model.set(&pose.c);
            model.quasi_newton_optimize(p, nc, pose, &mut g, &authentic_v, &mut evalcount, max_steps, &mut scratch);
            model.set(&pose.c);
            if nc.within(model, 0.0001) {
                break;
            }
        }
        pose.coords = model.get_heavy_atom_movable_coords();
        // C++ rejects poses still outside the box
        if !nc.within(model, 0.0001) {
            pose.e = MAX_FL;
        }
    }

    nc.slope = slope_orig;
    poses.sort_by(|a, b| a.e.partial_cmp(&b.e).unwrap_or(std::cmp::Ordering::Equal));
}

fn do_compute_maps(
    model: &Model,
    sf: &ScoringFunction,
    no_refine: bool,
    verbosity: i32,
    center_x: f64, center_y: f64, center_z: f64,
    size_x: f64, size_y: f64, size_z: f64,
    granularity: f64,
    force_even_voxels: bool,
) -> (Cache, Option<NonCache>, GridDims) {
    let center = Vec3::new(center_x, center_y, center_z);
    let size = Vec3::new(size_x, size_y, size_z);
    let gd = make_grid_dims(&center, &size, granularity, force_even_voxels);

    let precalc = Precalculate::new(sf, MAX_FL, 32.0);
    let atom_types: Vec<usize> = (0..sf.num_atom_types()).collect();

    let mut cache = Cache::new(1e6);
    cache.populate(model, &precalc, &atom_types, &gd);

    let nc = if !no_refine {
        Some(NonCache::new(model, &gd, &precalc, 1e6))
    } else {
        None
    };

    if verbosity > 0 {
        let n = gd.iter().map(|d| d.n_voxels).collect::<Vec<_>>();
        eprintln!("Grid dimensions: {}x{}x{} (granularity: {:.3} Å)", n[0], n[1], n[2], granularity);
    }

    (cache, nc, gd)
}

fn get_poses_energies_impl(poses: &OutputContainer, how_many: usize, energy_range: f64) -> Vec<Vec<f64>> {
    let best_e = poses.first().map(|p| p.e).unwrap_or(0.0);
    poses.iter()
        .take(how_many)
        .take_while(|p| p.e - best_e <= energy_range)
        .map(|p| vec![p.e, p.intra, p.inter, p.conf_independent, p.unbound, p.total])
        .collect()
}

/// Direct receptor-ligand energy evaluation (per-atom curl, matching C++ non_cache::eval)
fn eval_receptor_ligand_direct(model: &Model, sf: &ScoringFunction, v: f64) -> f64 {
    let cutoff_sqr = sqr(sf.cutoff);
    let mut e = 0.0;

    for i in 0..model.num_movable_atoms {
        let a = &model.atoms[i];
        if a.xs >= XS_TYPE_SIZE { continue; }

        // Sum all receptor contributions for this atom, then apply curl (per-atom, like C++)
        let mut this_e = 0.0;
        for ga in &model.grid_atoms {
            if ga.xs >= XS_TYPE_SIZE { continue; }
            let r2 = model.coords[i].distance_sqr(&ga.coords);
            if r2 < cutoff_sqr {
                let r = r2.sqrt();
                this_e += sf.eval_by_atoms(a, ga, r);
            }
        }
        curl(&mut this_e, v);
        e += this_e;
    }

    e
}

/// Debug scoring: print per-term and per-atom-type breakdown
fn debug_scoring_breakdown(model: &Model, sf: &ScoringFunction, p: &PrecalculateByAtom, v: f64) -> String {
    let mut out = String::new();

    // Count ligand atom types
    out.push_str("=== Ligand Atom Types ===\n");
    let mut type_counts = vec![0usize; XS_TYPE_SIZE + 1];
    let mut lig_atoms = 0usize;
    for i in 0..model.num_movable_atoms {
        let xs = model.atoms[i].xs;
        if xs < XS_TYPE_SIZE {
            type_counts[xs] += 1;
            lig_atoms += 1;
        } else {
            type_counts[XS_TYPE_SIZE] += 1; // skipped (H)
        }
    }
    let type_names = [
        "C_H", "C_P", "N_P", "N_D", "N_A", "N_DA", "O_P", "O_D", "O_A", "O_DA",
        "S_P", "P_P", "F_H", "Cl_H", "Br_H", "I_H", "Si", "At", "Met_D",
        "C_H_CG0", "C_P_CG0", "G0", "C_H_CG1", "C_P_CG1", "G1",
        "C_H_CG2", "C_P_CG2", "G2", "C_H_CG3", "C_P_CG3", "G3", "W", "SKIP",
    ];
    for (i, &c) in type_counts.iter().enumerate() {
        if c > 0 {
            let name = if i < type_names.len() { type_names[i] } else { "?" };
            out.push_str(&format!("  {} (xs={}): {} atoms\n", name, i, c));
        }
    }
    out.push_str(&format!("  Total scoring atoms: {}, skipped: {}\n", lig_atoms, type_counts[XS_TYPE_SIZE]));

    // Count receptor atom types
    out.push_str("\n=== Receptor Atom Types ===\n");
    let mut rec_type_counts = vec![0usize; XS_TYPE_SIZE + 1];
    let mut rec_atoms = 0usize;
    for ga in &model.grid_atoms {
        if ga.xs < XS_TYPE_SIZE {
            rec_type_counts[ga.xs] += 1;
            rec_atoms += 1;
        } else {
            rec_type_counts[XS_TYPE_SIZE] += 1;
        }
    }
    for (i, &c) in rec_type_counts.iter().enumerate() {
        if c > 0 {
            let name = if i < type_names.len() { type_names[i] } else { "?" };
            out.push_str(&format!("  {} (xs={}): {} atoms\n", name, i, c));
        }
    }
    out.push_str(&format!("  Total scoring atoms: {}, skipped: {}\n", rec_atoms, rec_type_counts[XS_TYPE_SIZE]));

    // Per-potential-term breakdown
    out.push_str("\n=== Per-Term Receptor-Ligand Energy ===\n");
    let cutoff_sqr = sqr(sf.cutoff);
    let mut term_energies = vec![0.0_f64; sf.potentials.len()];
    let mut n_contacts = 0usize;

    for i in 0..model.num_movable_atoms {
        let a = &model.atoms[i];
        if a.xs >= XS_TYPE_SIZE { continue; }
        for ga in &model.grid_atoms {
            if ga.xs >= XS_TYPE_SIZE { continue; }
            let r2 = model.coords[i].distance_sqr(&ga.coords);
            if r2 < cutoff_sqr {
                let r = r2.sqrt();
                for (k, pot) in sf.potentials.iter().enumerate() {
                    term_energies[k] += sf.weights[k] * pot.eval_by_atoms(a, ga, r);
                }
                n_contacts += 1;
            }
        }
    }

    let vina_term_names = ["Gauss1", "Gauss2", "Repulsion", "Hydrophobic", "HBond", "LinAttract"];
    for (k, e) in term_energies.iter().enumerate() {
        let name = if k < vina_term_names.len() { vina_term_names[k] } else { "Unknown" };
        out.push_str(&format!("  {}: {:.4} (weight={:.6})\n", name, e, sf.weights[k]));
    }
    let inter_total: f64 = term_energies.iter().sum();
    out.push_str(&format!("  TOTAL inter: {:.4} ({} contacts)\n", inter_total, n_contacts));

    // Intramolecular
    let intra_e = model.eval_intra(p, v);
    out.push_str(&format!("\n=== Intramolecular ===\n  intra: {:.4}\n", intra_e));

    let total_raw = inter_total + intra_e;
    let num_tors = model.conf_independent_num_tors();
    let total = sf.conf_independent(num_tors, total_raw);

    out.push_str(&format!("\n=== Summary ===\n"));
    out.push_str(&format!("  Inter (receptor-ligand): {:.4}\n", inter_total));
    out.push_str(&format!("  Intra (within ligand):   {:.4}\n", intra_e));
    out.push_str(&format!("  Total raw:               {:.4}\n", total_raw));
    out.push_str(&format!("  Torsions (C++ style):    {:.3}\n", num_tors));
    out.push_str(&format!("  After conf_independent:  {:.4}\n", total));

    // Ligand center of mass
    let mut com = Vec3::ZERO;
    let mut n = 0.0;
    for i in 0..model.num_movable_atoms {
        if model.atoms[i].xs < XS_TYPE_SIZE {
            com += model.coords[i];
            n += 1.0;
        }
    }
    if n > 0.0 { com *= 1.0 / n; }
    out.push_str(&format!("\n  Ligand center: ({:.2}, {:.2}, {:.2})\n", com[0], com[1], com[2]));

    out
}

/// Debug intramolecular pair details
fn debug_intra_details(model: &Model, p: &PrecalculateByAtom) -> String {
    let mut out = String::new();
    let v = 1000.0;
    let cutoff_sqr = p.cutoff_sqr();

    for (li, lig) in model.ligands.iter().enumerate() {
        out.push_str(&format!("=== Ligand {} — {} intramolecular pairs ===\n", li, lig.pairs.len()));

        // Bond count per atom
        let mut total_bonds = 0usize;
        for i in lig.begin..lig.end {
            total_bonds += model.atoms[i].bonds.len();
        }
        out.push_str(&format!("  Total bonds (directed): {}\n", total_bonds));

        // Compute per-pair energies and find top contributors
        let mut pair_energies: Vec<(usize, usize, f64, f64)> = Vec::new();
        let mut total_e = 0.0_f64;
        let mut n_active = 0usize;

        for pair in &lig.pairs {
            let r2 = model.coords[pair.a].distance_sqr(&model.coords[pair.b]);
            if r2 < cutoff_sqr {
                let mut e = p.eval_fast_by_index(pair.type_pair_index, r2);
                curl(&mut e, v);
                total_e += e;
                n_active += 1;
                pair_energies.push((pair.a, pair.b, r2.sqrt(), e));
            }
        }

        out.push_str(&format!("  Active pairs (within cutoff): {}\n", n_active));
        out.push_str(&format!("  Total intra energy: {:.4}\n\n", total_e));

        // Sort by energy (most positive first — biggest offenders)
        pair_energies.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        out.push_str("  Top 15 most positive contributors:\n");
        for (idx, &(a, b, r, e)) in pair_energies.iter().take(15).enumerate() {
            let xs_a = model.atoms[a].xs;
            let xs_b = model.atoms[b].xs;
            out.push_str(&format!("    {:2}. atoms({},{}) xs=({},{}) r={:.3}Å e={:.4}\n",
                idx + 1, a, b, xs_a, xs_b, r, e));
        }

        // Also show top negative
        out.push_str("\n  Top 10 most negative contributors:\n");
        for (idx, &(a, b, r, e)) in pair_energies.iter().rev().take(10).enumerate() {
            let xs_a = model.atoms[a].xs;
            let xs_b = model.atoms[b].xs;
            out.push_str(&format!("    {:2}. atoms({},{}) xs=({},{}) r={:.3}Å e={:.4}\n",
                idx + 1, a, b, xs_a, xs_b, r, e));
        }
    }

    out
}

fn get_poses_coordinates_impl(poses: &OutputContainer, how_many: usize, energy_range: f64) -> Vec<Vec<f64>> {
    let best_e = poses.first().map(|p| p.e).unwrap_or(0.0);
    poses.iter()
        .take(how_many)
        .take_while(|p| p.e - best_e <= energy_range)
        .map(|p| {
            p.coords.iter()
                .flat_map(|c| vec![c[0], c[1], c[2]])
                .collect()
        })
        .collect()
}

// ─── Batch Docking Helpers ──────────────────────────────────────────────────

/// Build a progress callback for batch docking that includes ligand index info.
fn build_batch_progress_callback(
    py: Python<'_>,
    py_cb: PyObject,
    total_ligands: usize,
) -> Arc<dyn Fn(usize, &DockingProgress) + Send + Sync> {
    let cb = py_cb.clone_ref(py);
    Arc::new(move |ligand_idx: usize, progress: &DockingProgress| {
        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            let _ = kwargs.set_item("stage", format!("{:?}", progress.stage));
            let _ = kwargs.set_item("mc_run", progress.mc_run);
            let _ = kwargs.set_item("total_mc_runs", progress.total_mc_runs);
            let _ = kwargs.set_item("step", progress.step);
            let _ = kwargs.set_item("total_steps", progress.total_steps);
            let _ = kwargs.set_item("best_energy", progress.best_energy);
            let _ = kwargs.set_item("current_energy", progress.current_energy);
            let _ = kwargs.set_item("poses_found", progress.poses_found);
            let _ = kwargs.set_item("ligand_index", ligand_idx);
            let _ = kwargs.set_item("total_ligands", total_ligands);
            // Overall percent: (ligand_idx * 100 + per-ligand %) / total_ligands
            let overall_pct = (ligand_idx as f64 * 100.0 + progress.percent_complete) / total_ligands as f64;
            let _ = kwargs.set_item("percent_complete", overall_pct);
            // For LigandDone: include the actual results so Python can
            // save them immediately without waiting for batch_dock to return.
            if progress.stage == DockingStage::LigandDone {
                let _ = kwargs.set_item("result_pdbqt", &progress.result_pdbqt);
                let energies_py: Vec<Vec<f64>> = progress.result_energies.clone();
                let _ = kwargs.set_item("result_energies", energies_py);
            }
            let _ = cb.call(py, (), Some(&kwargs));
        });
    })
}

/// Batch dock multiple ligands against a single receptor.
/// Grid maps are computed once and reused for all ligands.
fn batch_dock_inner(
    receptor: &Model,
    sf_choice: ScoringFunctionChoice,
    weights: &[f64],
    no_refine: bool,
    verbosity: i32,
    seed: u64,
    ligand_strings: &[String],
    center_x: f64, center_y: f64, center_z: f64,
    size_x: f64, size_y: f64, size_z: f64,
    granularity: f64,
    exhaustiveness: usize,
    n_poses: usize,
    min_rmsd: f64,
    max_evals: i32,
    energy_range: f64,
    how_many: usize,
    is_smina: bool,
    algorithm: SearchAlgorithm,
    batch_cb: Option<Arc<dyn Fn(usize, &DockingProgress) + Send + Sync>>,
    stream_results: bool,
) -> Vec<(String, Vec<Vec<f64>>)> {
    let sf = ScoringFunction::new(sf_choice, weights.to_vec());
    let precalculated = PrecalculateByAtom::new(&sf, 32.0);

    // Compute grid maps ONCE from receptor
    let (cache, nc_opt, gd) = do_compute_maps(
        receptor, &sf, no_refine, verbosity,
        center_x, center_y, center_z,
        size_x, size_y, size_z,
        granularity, false,
    );
    let cache_arc: Arc<dyn IGrid> = Arc::new(cache);

    let corner1 = Vec3::new(gd[0].begin, gd[1].begin, gd[2].begin);
    let corner2 = Vec3::new(gd[0].end, gd[1].end, gd[2].end);

    let total_ligands = ligand_strings.len();

    if verbosity > 0 {
        eprintln!("[batch] Docking {} ligands in parallel (maps computed once)", total_ligands);
    }

    // Wrap shared read-only data in Arc for cross-thread sharing
    let sf_arc = Arc::new(sf);
    let precalc_arc = Arc::new(precalculated);

    // Parallel over ligands using Rayon
    let results: Vec<(String, Vec<Vec<f64>>)> = ligand_strings
        .par_iter()
        .enumerate()
        .map(|(lig_idx, lig_str)| {
            // Parse ligand — clone receptor and merge
            let mut model = receptor.clone();
            if parse_ligand_string(&mut model, lig_str).is_err() {
                return (String::new(), Vec::new());
            }

            if is_smina {
                model.fixed_rotable_hydrogens = true;
            }

            // MC search parameters (ligand-dependent for Vina algorithm)
            let mut params = MonteCarloParams::new();
            params.max_evals = max_evals;

            if algorithm == SearchAlgorithm::Vina {
                // C++ Vina: each MC thread runs the full step count (NOT divided by exhaustiveness)
                let num_movable = model.num_movable_atoms();
                let s = model.get_size();
                let num_dof = s.num_degrees_of_freedom();
                let heuristic = num_movable + 10 * num_dof;
                params.global_steps = (70 * 3 * (50 + heuristic) / 2) as u32;
                params.local_steps = ((25 + num_movable) / 3) as u32;
            }

            // Per-ligand progress callback wrapping the batch callback
            let per_lig_cb: Option<Arc<ProgressCallback>> = batch_cb.as_ref().map(|bcb| {
                let bcb_clone = Arc::clone(bcb);
                let li = lig_idx;
                let pcb: ProgressCallback = Box::new(move |prog: &DockingProgress| {
                    bcb_clone(li, prog);
                });
                Arc::new(pcb)
            });

            // Run MC search with shared cache (inner Rayon parallelism
            // nests fine with work-stealing)
            let mut mc_results = match algorithm {
                SearchAlgorithm::Vina => {
                    parallel_mc(
                        &model, &precalc_arc, cache_arc.clone(),
                        &corner1, &corner2, &params, exhaustiveness, seed,
                        n_poses, min_rmsd, per_lig_cb,
                    )
                }
                SearchAlgorithm::QVina2 => {
                    parallel_mc_qvina2(
                        &model, &precalc_arc, cache_arc.clone(),
                        &corner1, &corner2, &params, exhaustiveness, seed,
                        n_poses, min_rmsd, per_lig_cb,
                    )
                }
            };

            // Refine poses with BFGS — each thread gets its own NonCache
            // clone (slope is mutated during refinement)
            if let Some(ref nc) = nc_opt {
                let mut nc_clone = nc.clone();
                refine_poses(&mut model, &precalc_arc, &mut nc_clone, &mut mc_results);
            }

            // Re-evaluate with exact pair-by-pair scoring
            let num_tors = model.conf_independent_num_tors();
            for pose in mc_results.iter_mut() {
                if not_max(pose.e) {
                    model.set(&pose.c);
                    let inter_e = eval_receptor_ligand_direct(&model, &sf_arc, 1000.0);
                    pose.e = sf_arc.conf_independent(num_tors, inter_e);
                }
            }
            mc_results.sort_by(|a, b| a.e.partial_cmp(&b.e).unwrap_or(std::cmp::Ordering::Equal));

            // Extract results
            let poses_pdbqt = write_poses_pdbqt(&mut model, &mc_results, how_many, energy_range);
            let energies = get_poses_energies_impl(&mc_results, how_many, energy_range);

            let best_e = mc_results.first().map(|p| p.e).unwrap_or(f64::NAN);

            // Signal per-ligand completion with results (poses + energies).
            // When stream_results is true, move the data into the callback
            // and return empty stubs to free memory immediately.
            if let Some(ref bcb) = batch_cb {
                if stream_results {
                    let dp = DockingProgress {
                        stage: DockingStage::LigandDone,
                        mc_run: 0,
                        total_mc_runs: exhaustiveness,
                        step: 0,
                        total_steps: 0,
                        best_energy: best_e,
                        current_energy: 0.0,
                        poses_found: mc_results.len(),
                        percent_complete: 100.0,
                        result_pdbqt: poses_pdbqt,       // moved
                        result_energies: energies,         // moved
                    };
                    bcb(lig_idx, &dp);
                    // dp dropped here — memory freed
                    return (String::new(), Vec::new());
                } else {
                    bcb(lig_idx, &DockingProgress {
                        stage: DockingStage::LigandDone,
                        mc_run: 0,
                        total_mc_runs: exhaustiveness,
                        step: 0,
                        total_steps: 0,
                        best_energy: best_e,
                        current_energy: 0.0,
                        poses_found: mc_results.len(),
                        percent_complete: 100.0,
                        result_pdbqt: poses_pdbqt.clone(),
                        result_energies: energies.clone(),
                    });
                }
            }

            if verbosity > 0 {
                eprintln!("[batch {}/{}] {} poses, best={:.1} kcal/mol",
                    lig_idx + 1, total_ligands, mc_results.len(), best_e);
            }

            (poses_pdbqt, energies)
        })
        .collect();

    results
}

// ─── PyVina ─────────────────────────────────────────────────────────────────

#[pyclass(name = "Vina")]
pub struct PyVina {
    model: Model,
    receptor: Model,
    sf_choice: ScoringFunctionChoice,
    weights: Vec<f64>,
    scoring_function: Option<ScoringFunction>,
    precalculated: Option<PrecalculateByAtom>,
    cache: Option<Cache>,
    non_cache: Option<NonCache>,
    grid_dims: Option<GridDims>,
    poses: OutputContainer,
    cpu: usize,
    seed: u64,
    verbosity: i32,
    no_refine: bool,
    unbound_energy: f64,
    receptor_initialized: bool,
    ligand_initialized: bool,
    map_initialized: bool,

}

#[pymethods]
impl PyVina {
    #[new]
    #[pyo3(signature = (sf_name="vina", cpu=0, seed=0, verbosity=1, no_refine=false))]
    fn new(sf_name: &str, cpu: i32, seed: i64, verbosity: i32, no_refine: bool) -> PyResult<Self> {
        let (sf_choice, weights) = resolve_sf(sf_name)?;
        Ok(PyVina {
            model: Model::new(),
            receptor: Model::new(),
            sf_choice, weights,
            scoring_function: None,
            precalculated: None,
            cache: None,
            non_cache: None,
            grid_dims: None,
            poses: Vec::new(),
            cpu: resolve_cpu(cpu),
            seed: resolve_seed(seed),
            verbosity, no_refine,
            unbound_energy: 0.0,
            receptor_initialized: false,
            ligand_initialized: false,
            map_initialized: false,
        })
    }

    #[pyo3(signature = (rigid_name, flex_name=None))]
    fn set_receptor(&mut self, rigid_name: &str, flex_name: Option<&str>) -> PyResult<()> {
        do_set_receptor(&mut self.receptor, rigid_name, flex_name)?;
        self.receptor_initialized = true;
        self.ligand_initialized = false;
        self.map_initialized = false;
        Ok(())
    }

    fn set_ligand_from_file(&mut self, ligand_name: &str) -> PyResult<()> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Receptor not initialized. Call set_receptor() first."));
        }
        do_set_ligand(&mut self.model, &self.receptor, self.sf_choice, &self.weights,
            &mut self.scoring_function, &mut self.precalculated, &mut self.unbound_energy,
            LigandSource::File(ligand_name))?;
        self.ligand_initialized = true;
        Ok(())
    }

    fn set_ligand_from_string(&mut self, ligand_string: &str) -> PyResult<()> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Receptor not initialized. Call set_receptor() first."));
        }
        do_set_ligand(&mut self.model, &self.receptor, self.sf_choice, &self.weights,
            &mut self.scoring_function, &mut self.precalculated, &mut self.unbound_energy,
            LigandSource::String(ligand_string))?;
        self.ligand_initialized = true;
        Ok(())
    }

    #[pyo3(signature = (center_x, center_y, center_z, size_x, size_y, size_z, granularity=0.5, force_even_voxels=false))]
    fn compute_vina_maps(
        &mut self,
        center_x: f64, center_y: f64, center_z: f64,
        size_x: f64, size_y: f64, size_z: f64,
        granularity: f64, force_even_voxels: bool,
    ) -> PyResult<()> {
        if !self.ligand_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Ligand not initialized. Call set_ligand_from_file/string() first."));
        }
        let sf = self.scoring_function.as_ref().unwrap();
        let (cache, nc, gd) = do_compute_maps(
            &self.model, sf, self.no_refine, self.verbosity,
            center_x, center_y, center_z, size_x, size_y, size_z, granularity, force_even_voxels,
        );
        self.cache = Some(cache);
        self.non_cache = nc;
        self.grid_dims = Some(gd);
        self.map_initialized = true;
        Ok(())
    }

    #[pyo3(signature = (exhaustiveness=8, n_poses=20, min_rmsd=1.0, max_evals=0, progress_callback=None))]
    fn global_search(
        &mut self, py: Python<'_>,
        exhaustiveness: usize, n_poses: usize, min_rmsd: f64, max_evals: i32,
        progress_callback: Option<PyObject>,
    ) -> PyResult<()> {
        if !self.map_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Maps not initialized. Call compute_vina_maps() first."));
        }

        let gd = self.grid_dims.unwrap();
        let corner1 = Vec3::new(gd[0].begin, gd[1].begin, gd[2].begin);
        let corner2 = Vec3::new(gd[0].end, gd[1].end, gd[2].end);

        let mut params = MonteCarloParams::new();
        params.max_evals = max_evals;
        // C++ Vina: each MC thread runs the full step count (NOT divided by exhaustiveness)
        let num_movable = self.model.num_movable_atoms();
        let s = self.model.get_size();
        let num_dof = s.num_degrees_of_freedom();
        let heuristic = num_movable + 10 * num_dof;
        params.global_steps = (70 * 3 * (50 + heuristic) / 2) as u32;
        // C++ Vina: ssd_par.evals = (25 + num_movable_atoms) / 3 — used as BFGS max_steps
        params.local_steps = ((25 + num_movable) / 3) as u32;

        let p = self.precalculated.as_ref().unwrap();
        let cache: Arc<dyn IGrid> = Arc::new(self.cache.take().unwrap());
        let progress_arc = progress_callback.map(|cb| build_progress_callback(py, cb));

        let model_clone = self.model.clone();
        let seed = self.seed;

        let mut results = py.allow_threads(move || {
            parallel_mc(
                &model_clone, p, cache.clone(), &corner1, &corner2,
                &params, exhaustiveness, seed, n_poses, min_rmsd, progress_arc,
            )
        });

        // Refine poses using exact pair-by-pair evaluation (NonCache)
        if let Some(nc) = &mut self.non_cache {
            let p = self.precalculated.as_ref().unwrap();
            refine_poses(&mut self.model, p, nc, &mut results);
        }

        // Re-evaluate each pose with exact pair-by-pair scoring so that
        // pose.e == score() with zero approximation gap.
        // C++ eval_adjusted: for EACH pose, compute its own intra, subtract from total.
        let num_tors = self.model.conf_independent_num_tors();
        self.poses = results;

        for i in 0..self.poses.len() {
            if not_max(self.poses[i].e) {
                let c = self.poses[i].c.clone();
                self.model.set(&c);
                let sf = self.scoring_function.as_ref().unwrap();
                let inter_e = eval_receptor_ligand_direct(&self.model, sf, 1000.0);
                self.poses[i].e = sf.conf_independent(num_tors, inter_e);
            }
        }
        self.poses.sort_by(|a, b| a.e.partial_cmp(&b.e).unwrap_or(std::cmp::Ordering::Equal));

        // Set model to best pose
        if let Some(best) = self.poses.first() {
            self.model.set(&best.c);
        }

        if self.verbosity > 0 {
            eprintln!("Found {} poses", self.poses.len());
            if let Some(best) = self.poses.first() {
                eprintln!("Best energy: {:.1} kcal/mol", best.e);
            }
        }
        Ok(())
    }

    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses(&mut self, how_many: usize, energy_range: f64) -> String {
        write_poses_pdbqt(&mut self.model, &self.poses, how_many, energy_range)
    }

    #[pyo3(signature = (output_name, how_many=9, energy_range=3.0))]
    fn write_poses(&mut self, output_name: &str, how_many: usize, energy_range: f64) -> PyResult<()> {
        std::fs::write(output_name, self.get_poses(how_many, energy_range))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot write file: {}", e)))?;
        Ok(())
    }

    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses_energies(&self, how_many: usize, energy_range: f64) -> Vec<Vec<f64>> {
        get_poses_energies_impl(&self.poses, how_many, energy_range)
    }

    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses_coordinates(&self, how_many: usize, energy_range: f64) -> Vec<Vec<f64>> {
        get_poses_coordinates_impl(&self.poses, how_many, energy_range)
    }

    /// Evaluate energy of the best (first) pose.
    /// Returns [total, inter, intra, total_raw, grid_inter].
    /// - total: final energy after conf_independent
    /// - inter: direct receptor-ligand energy (pair evaluation)
    /// - intra: intramolecular energy
    /// - total_raw: inter + intra (before conf_independent)
    /// - grid_inter: grid-based receptor-ligand energy (for comparison)
    fn score(&mut self) -> PyResult<Vec<f64>> {
        if !self.ligand_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Ligand not initialized"));
        }

        // Always score the best (first) pose; fall back to current model state
        if let Some(best) = self.poses.first() {
            self.model.set(&best.c);
        }

        let p = self.precalculated.as_ref().unwrap();
        let sf = self.scoring_function.as_ref().unwrap();
        let v = 1000.0; // authentic curl cap

        // 1. Direct receptor-ligand interaction (pair-by-pair)
        let inter_e = eval_receptor_ligand_direct(&self.model, sf, v);

        // 2. Intramolecular
        let intra_e = self.model.eval_intra(p, v);

        // 3. Total — match C++ score_only: conf_independent(inter_only)
        // C++ subtracts intramolecular before conf_independent
        let num_tors = self.model.conf_independent_num_tors();
        let total = sf.conf_independent(num_tors, inter_e);

        // 4. Grid-based energy (for comparison, if cache available)
        let grid_e = if let Some(cache) = &self.cache {
            cache.eval(&self.model, v)
        } else {
            f64::NAN
        };

        Ok(vec![total, inter_e, intra_e, inter_e + intra_e, grid_e])
    }

    /// Debug scoring: print per-atom-type and per-term breakdown
    fn debug_score(&mut self) -> PyResult<String> {
        if !self.ligand_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Ligand not initialized"));
        }
        let sf = self.scoring_function.as_ref().unwrap();
        let p = self.precalculated.as_ref().unwrap();
        let v = 1000.0;
        Ok(debug_scoring_breakdown(&self.model, sf, p, v))
    }

    fn num_poses(&self) -> usize { self.poses.len() }
    fn seed(&self) -> u64 { self.seed }

    fn __repr__(&self) -> String {
        format!("Vina(sf={:?}, cpu={}, seed={}, receptor={}, ligand={}, maps={})",
            self.sf_choice, self.cpu, self.seed,
            self.receptor_initialized, self.ligand_initialized, self.map_initialized)
    }

    /// Batch dock multiple ligands against the current receptor.
    /// Grid maps are computed once and reused for all ligands.
    /// Returns list of (poses_pdbqt, energies) for each ligand.
    #[pyo3(signature = (
        ligand_strings,
        center_x, center_y, center_z,
        size_x, size_y, size_z,
        exhaustiveness=8,
        n_poses=20,
        min_rmsd=1.0,
        max_evals=0,
        granularity=0.5,
        energy_range=3.0,
        how_many=9,
        progress_callback=None,
        stream_results=false,
    ))]
    fn batch_dock(
        &self, py: Python<'_>,
        ligand_strings: Vec<String>,
        center_x: f64, center_y: f64, center_z: f64,
        size_x: f64, size_y: f64, size_z: f64,
        exhaustiveness: usize,
        n_poses: usize,
        min_rmsd: f64,
        max_evals: i32,
        granularity: f64,
        energy_range: f64,
        how_many: usize,
        progress_callback: Option<PyObject>,
        stream_results: bool,
    ) -> PyResult<Vec<(String, Vec<Vec<f64>>)>> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Receptor not initialized. Call set_receptor() first."
            ));
        }

        let total_ligands = ligand_strings.len();
        let batch_cb = progress_callback.map(|cb| {
            build_batch_progress_callback(py, cb, total_ligands)
        });

        let receptor = self.receptor.clone();
        let sf_choice = self.sf_choice;
        let weights = self.weights.clone();
        let no_refine = self.no_refine;
        let verbosity = self.verbosity;
        let seed = self.seed;

        let results = py.allow_threads(move || {
            batch_dock_inner(
                &receptor, sf_choice, &weights,
                no_refine, verbosity, seed,
                &ligand_strings,
                center_x, center_y, center_z,
                size_x, size_y, size_z,
                granularity,
                exhaustiveness, n_poses, min_rmsd, max_evals,
                energy_range, how_many,
                false,
                SearchAlgorithm::Vina,
                batch_cb,
                stream_results,
            )
        });

        Ok(results)
    }
}

// ─── PyQVina2 ───────────────────────────────────────────────────────────────

#[pyclass(name = "QVina2")]
pub struct PyQVina2 {
    model: Model,
    receptor: Model,
    sf_choice: ScoringFunctionChoice,
    weights: Vec<f64>,
    scoring_function: Option<ScoringFunction>,
    precalculated: Option<PrecalculateByAtom>,
    cache: Option<Cache>,
    non_cache: Option<NonCache>,
    grid_dims: Option<GridDims>,
    poses: OutputContainer,
    cpu: usize,
    seed: u64,
    verbosity: i32,
    no_refine: bool,
    unbound_energy: f64,
    receptor_initialized: bool,
    ligand_initialized: bool,
    map_initialized: bool,

}

#[pymethods]
impl PyQVina2 {
    #[new]
    #[pyo3(signature = (sf_name="vina", cpu=0, seed=0, verbosity=1, no_refine=false))]
    fn new(sf_name: &str, cpu: i32, seed: i64, verbosity: i32, no_refine: bool) -> PyResult<Self> {
        let (sf_choice, weights) = resolve_sf(sf_name)?;
        Ok(PyQVina2 {
            model: Model::new(),
            receptor: Model::new(),
            sf_choice, weights,
            scoring_function: None,
            precalculated: None,
            cache: None,
            non_cache: None,
            grid_dims: None,
            poses: Vec::new(),
            cpu: resolve_cpu(cpu),
            seed: resolve_seed(seed),
            verbosity, no_refine,
            unbound_energy: 0.0,
            receptor_initialized: false,
            ligand_initialized: false,
            map_initialized: false,
        })
    }

    #[pyo3(signature = (rigid_name, flex_name=None))]
    fn set_receptor(&mut self, rigid_name: &str, flex_name: Option<&str>) -> PyResult<()> {
        do_set_receptor(&mut self.receptor, rigid_name, flex_name)?;
        self.receptor_initialized = true;
        self.ligand_initialized = false;
        self.map_initialized = false;
        Ok(())
    }

    fn set_ligand_from_file(&mut self, ligand_name: &str) -> PyResult<()> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Receptor not initialized. Call set_receptor() first."));
        }
        do_set_ligand(&mut self.model, &self.receptor, self.sf_choice, &self.weights,
            &mut self.scoring_function, &mut self.precalculated, &mut self.unbound_energy,
            LigandSource::File(ligand_name))?;
        self.ligand_initialized = true;
        Ok(())
    }

    fn set_ligand_from_string(&mut self, ligand_string: &str) -> PyResult<()> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Receptor not initialized. Call set_receptor() first."));
        }
        do_set_ligand(&mut self.model, &self.receptor, self.sf_choice, &self.weights,
            &mut self.scoring_function, &mut self.precalculated, &mut self.unbound_energy,
            LigandSource::String(ligand_string))?;
        self.ligand_initialized = true;
        Ok(())
    }

    #[pyo3(signature = (center_x, center_y, center_z, size_x, size_y, size_z, granularity=0.5, force_even_voxels=false))]
    fn compute_vina_maps(
        &mut self,
        center_x: f64, center_y: f64, center_z: f64,
        size_x: f64, size_y: f64, size_z: f64,
        granularity: f64, force_even_voxels: bool,
    ) -> PyResult<()> {
        if !self.ligand_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Ligand not initialized. Call set_ligand_from_file/string() first."));
        }
        let sf = self.scoring_function.as_ref().unwrap();
        let (cache, nc, gd) = do_compute_maps(
            &self.model, sf, self.no_refine, self.verbosity,
            center_x, center_y, center_z, size_x, size_y, size_z, granularity, force_even_voxels,
        );
        self.cache = Some(cache);
        self.non_cache = nc;
        self.grid_dims = Some(gd);
        self.map_initialized = true;
        Ok(())
    }

    /// Run QVina2 global search — uses hunt→dock + visited history for ~20x speedup
    #[pyo3(signature = (exhaustiveness=8, n_poses=20, min_rmsd=1.0, max_evals=0, progress_callback=None))]
    fn global_search(
        &mut self, py: Python<'_>,
        exhaustiveness: usize, n_poses: usize, min_rmsd: f64, max_evals: i32,
        progress_callback: Option<PyObject>,
    ) -> PyResult<()> {
        if !self.map_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Maps not initialized. Call compute_vina_maps() first."));
        }

        let gd = self.grid_dims.unwrap();
        let corner1 = Vec3::new(gd[0].begin, gd[1].begin, gd[2].begin);
        let corner2 = Vec3::new(gd[0].end, gd[1].end, gd[2].end);

        let mut params = MonteCarloParams::new();
        params.max_evals = max_evals;

        let p = self.precalculated.as_ref().unwrap();
        let cache: Arc<dyn IGrid> = Arc::new(self.cache.take().unwrap());
        let progress_arc = progress_callback.map(|cb| build_progress_callback(py, cb));

        let model_clone = self.model.clone();
        let seed = self.seed;

        let mut results = py.allow_threads(move || {
            parallel_mc_qvina2(
                &model_clone, p, cache.clone(), &corner1, &corner2,
                &params, exhaustiveness, seed, n_poses, min_rmsd, progress_arc,
            )
        });

        // Refine poses using exact pair-by-pair evaluation (NonCache)
        if let Some(nc) = &mut self.non_cache {
            let p = self.precalculated.as_ref().unwrap();
            refine_poses(&mut self.model, p, nc, &mut results);
        }

        // Re-evaluate each pose with exact pair-by-pair scoring so that
        // pose.e == score() with zero approximation gap.
        // C++ eval_adjusted: for EACH pose, compute its own intra, subtract from total.
        let num_tors = self.model.conf_independent_num_tors();
        self.poses = results;

        for i in 0..self.poses.len() {
            if not_max(self.poses[i].e) {
                let c = self.poses[i].c.clone();
                self.model.set(&c);
                let sf = self.scoring_function.as_ref().unwrap();
                let inter_e = eval_receptor_ligand_direct(&self.model, sf, 1000.0);
                self.poses[i].e = sf.conf_independent(num_tors, inter_e);
            }
        }
        self.poses.sort_by(|a, b| a.e.partial_cmp(&b.e).unwrap_or(std::cmp::Ordering::Equal));

        // Set model to best pose so score() evaluates the best conformation
        if let Some(best) = self.poses.first() {
            self.model.set(&best.c);
        }

        if self.verbosity > 0 {
            eprintln!("[QVina2] Found {} poses | tors={:.1}", self.poses.len(), num_tors);
            if let Some(best) = self.poses.first() {
                eprintln!("[QVina2] Best energy: {:.1} kcal/mol", best.e);
            }
        }
        Ok(())
    }

    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses(&mut self, how_many: usize, energy_range: f64) -> String {
        write_poses_pdbqt(&mut self.model, &self.poses, how_many, energy_range)
    }

    #[pyo3(signature = (output_name, how_many=9, energy_range=3.0))]
    fn write_poses(&mut self, output_name: &str, how_many: usize, energy_range: f64) -> PyResult<()> {
        std::fs::write(output_name, self.get_poses(how_many, energy_range))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot write file: {}", e)))?;
        Ok(())
    }

    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses_energies(&self, how_many: usize, energy_range: f64) -> Vec<Vec<f64>> {
        get_poses_energies_impl(&self.poses, how_many, energy_range)
    }

    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses_coordinates(&self, how_many: usize, energy_range: f64) -> Vec<Vec<f64>> {
        get_poses_coordinates_impl(&self.poses, how_many, energy_range)
    }

    fn score(&mut self) -> PyResult<Vec<f64>> {
        if !self.ligand_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Ligand not initialized"));
        }

        // Always score the best (first) pose; fall back to current model state
        if let Some(best) = self.poses.first() {
            self.model.set(&best.c);
        }

        let p = self.precalculated.as_ref().unwrap();
        let sf = self.scoring_function.as_ref().unwrap();
        let v = 1000.0;
        let inter_e = eval_receptor_ligand_direct(&self.model, sf, v);
        let intra_e = self.model.eval_intra(p, v);
        let num_tors = self.model.conf_independent_num_tors();
        // Match C++ score_only: conf_independent(inter_only)
        let total = sf.conf_independent(num_tors, inter_e);
        let grid_e = if let Some(cache) = &self.cache {
            cache.eval(&self.model, v)
        } else {
            f64::NAN
        };
        Ok(vec![total, inter_e, intra_e, inter_e + intra_e, grid_e])
    }

    fn debug_score(&mut self) -> PyResult<String> {
        if !self.ligand_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Ligand not initialized"));
        }
        let sf = self.scoring_function.as_ref().unwrap();
        let p = self.precalculated.as_ref().unwrap();
        let v = 1000.0;
        Ok(debug_scoring_breakdown(&self.model, sf, p, v))
    }

    /// Debug intramolecular pairs: count, top contributors, bond info
    fn debug_intra(&self) -> PyResult<String> {
        if !self.ligand_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Ligand not initialized"));
        }
        Ok(debug_intra_details(&self.model, self.precalculated.as_ref().unwrap()))
    }

    fn num_poses(&self) -> usize { self.poses.len() }
    fn seed(&self) -> u64 { self.seed }

    fn __repr__(&self) -> String {
        format!("QVina2(sf={:?}, cpu={}, seed={}, receptor={}, ligand={}, maps={})",
            self.sf_choice, self.cpu, self.seed,
            self.receptor_initialized, self.ligand_initialized, self.map_initialized)
    }

    /// Batch dock multiple ligands against the current receptor.
    /// Grid maps are computed once and reused for all ligands.
    /// Returns list of (poses_pdbqt, energies) for each ligand.
    #[pyo3(signature = (
        ligand_strings,
        center_x, center_y, center_z,
        size_x, size_y, size_z,
        exhaustiveness=8,
        n_poses=20,
        min_rmsd=1.0,
        max_evals=0,
        granularity=0.5,
        energy_range=3.0,
        how_many=9,
        progress_callback=None,
        stream_results=false,
    ))]
    fn batch_dock(
        &self, py: Python<'_>,
        ligand_strings: Vec<String>,
        center_x: f64, center_y: f64, center_z: f64,
        size_x: f64, size_y: f64, size_z: f64,
        exhaustiveness: usize,
        n_poses: usize,
        min_rmsd: f64,
        max_evals: i32,
        granularity: f64,
        energy_range: f64,
        how_many: usize,
        progress_callback: Option<PyObject>,
        stream_results: bool,
    ) -> PyResult<Vec<(String, Vec<Vec<f64>>)>> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Receptor not initialized. Call set_receptor() first."
            ));
        }

        let total_ligands = ligand_strings.len();
        let batch_cb = progress_callback.map(|cb| {
            build_batch_progress_callback(py, cb, total_ligands)
        });

        let receptor = self.receptor.clone();
        let sf_choice = self.sf_choice;
        let weights = self.weights.clone();
        let no_refine = self.no_refine;
        let verbosity = self.verbosity;
        let seed = self.seed;

        let results = py.allow_threads(move || {
            batch_dock_inner(
                &receptor, sf_choice, &weights,
                no_refine, verbosity, seed,
                &ligand_strings,
                center_x, center_y, center_z,
                size_x, size_y, size_z,
                granularity,
                exhaustiveness, n_poses, min_rmsd, max_evals,
                energy_range, how_many,
                false,
                SearchAlgorithm::QVina2,
                batch_cb,
                stream_results,
            )
        });

        Ok(results)
    }
}

// ─── PySmina ────────────────────────────────────────────────────────────────
// Smina is a fork of AutoDock Vina that adds Boron atom support and additional
// scoring functions. The docking algorithm is identical to QVina2. The key
// difference is extended atom typing: "B" (Boron) is parsed as a hydrophobic
// atom (xs_radius=1.92 Å, matching smina defaults).
// Supported sf_name values: "vina" (default), "vinardo", "ad4".

#[pyclass(name = "Smina")]
pub struct PySmina {
    model: Model,
    receptor: Model,
    sf_choice: ScoringFunctionChoice,
    weights: Vec<f64>,
    scoring_function: Option<ScoringFunction>,
    precalculated: Option<PrecalculateByAtom>,
    cache: Option<Cache>,
    non_cache: Option<NonCache>,
    grid_dims: Option<GridDims>,
    poses: OutputContainer,
    cpu: usize,
    seed: u64,
    verbosity: i32,
    no_refine: bool,
    unbound_energy: f64,
    receptor_initialized: bool,
    ligand_initialized: bool,
    map_initialized: bool,

}

#[pymethods]
impl PySmina {
    #[new]
    #[pyo3(signature = (sf_name="vina", cpu=0, seed=0, verbosity=1, no_refine=false))]
    fn new(sf_name: &str, cpu: i32, seed: i64, verbosity: i32, no_refine: bool) -> PyResult<Self> {
        let (sf_choice, weights) = resolve_sf(sf_name)?;
        Ok(PySmina {
            model: Model::new(),
            receptor: Model::new(),
            sf_choice, weights,
            scoring_function: None,
            precalculated: None,
            cache: None,
            non_cache: None,
            grid_dims: None,
            poses: Vec::new(),
            cpu: resolve_cpu(cpu),
            seed: resolve_seed(seed),
            verbosity, no_refine,
            unbound_energy: 0.0,
            receptor_initialized: false,
            ligand_initialized: false,
            map_initialized: false,
        })
    }

    #[pyo3(signature = (rigid_name, flex_name=None))]
    fn set_receptor(&mut self, rigid_name: &str, flex_name: Option<&str>) -> PyResult<()> {
        do_set_receptor(&mut self.receptor, rigid_name, flex_name)?;
        self.receptor_initialized = true;
        self.ligand_initialized = false;
        self.map_initialized = false;
        Ok(())
    }

    fn set_ligand_from_file(&mut self, ligand_name: &str) -> PyResult<()> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Receptor not initialized. Call set_receptor() first."));
        }
        do_set_ligand(&mut self.model, &self.receptor, self.sf_choice, &self.weights,
            &mut self.scoring_function, &mut self.precalculated, &mut self.unbound_energy,
            LigandSource::File(ligand_name))?;
        // Smina: exclude terminal-H rotors from num_tors (see smina terms.cpp atom_rotors)
        self.model.fixed_rotable_hydrogens = true;
        self.ligand_initialized = true;
        Ok(())
    }

    fn set_ligand_from_string(&mut self, ligand_string: &str) -> PyResult<()> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Receptor not initialized. Call set_receptor() first."));
        }
        do_set_ligand(&mut self.model, &self.receptor, self.sf_choice, &self.weights,
            &mut self.scoring_function, &mut self.precalculated, &mut self.unbound_energy,
            LigandSource::String(ligand_string))?;
        // Smina: same fix for terminal-H rotors
        self.model.fixed_rotable_hydrogens = true;
        self.ligand_initialized = true;
        Ok(())
    }

    #[pyo3(signature = (center_x, center_y, center_z, size_x, size_y, size_z, granularity=0.5, force_even_voxels=false))]
    fn compute_vina_maps(
        &mut self,
        center_x: f64, center_y: f64, center_z: f64,
        size_x: f64, size_y: f64, size_z: f64,
        granularity: f64, force_even_voxels: bool,
    ) -> PyResult<()> {
        if !self.ligand_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Ligand not initialized. Call set_ligand_from_file/string() first."));
        }
        let sf = self.scoring_function.as_ref().unwrap();
        let (cache, nc, gd) = do_compute_maps(
            &self.model, sf, self.no_refine, self.verbosity,
            center_x, center_y, center_z, size_x, size_y, size_z, granularity, force_even_voxels,
        );
        self.cache = Some(cache);
        self.non_cache = nc;
        self.grid_dims = Some(gd);
        self.map_initialized = true;
        Ok(())
    }

    /// Run Smina global search (QVina2 algorithm + smina atom typing with Boron support).
    /// Pose energies are evaluated with exact pair-by-pair scoring, matching score().
    #[pyo3(signature = (exhaustiveness=8, n_poses=20, min_rmsd=1.0, max_evals=0, progress_callback=None))]
    fn global_search(
        &mut self, py: Python<'_>,
        exhaustiveness: usize, n_poses: usize, min_rmsd: f64, max_evals: i32,
        progress_callback: Option<PyObject>,
    ) -> PyResult<()> {
        if !self.map_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Maps not initialized. Call compute_vina_maps() first."));
        }

        let gd = self.grid_dims.unwrap();
        let corner1 = Vec3::new(gd[0].begin, gd[1].begin, gd[2].begin);
        let corner2 = Vec3::new(gd[0].end, gd[1].end, gd[2].end);

        let mut params = MonteCarloParams::new();
        params.max_evals = max_evals;

        let p = self.precalculated.as_ref().unwrap();
        let cache: Arc<dyn IGrid> = Arc::new(self.cache.take().unwrap());
        let progress_arc = progress_callback.map(|cb| build_progress_callback(py, cb));

        let model_clone = self.model.clone();
        let seed = self.seed;

        let mut results = py.allow_threads(move || {
            parallel_mc_qvina2(
                &model_clone, p, cache.clone(), &corner1, &corner2,
                &params, exhaustiveness, seed, n_poses, min_rmsd, progress_arc,
            )
        });

        // Refine poses with BFGS (NonCache)
        if let Some(nc) = &mut self.non_cache {
            let p = self.precalculated.as_ref().unwrap();
            refine_poses(&mut self.model, p, nc, &mut results);
        }

        // Re-evaluate each pose with exact pair-by-pair scoring so that
        // pose.e == score() with zero approximation gap.
        let num_tors = self.model.conf_independent_num_tors();
        self.poses = results;
        for i in 0..self.poses.len() {
            if not_max(self.poses[i].e) {
                let c = self.poses[i].c.clone();
                self.model.set(&c);
                let sf = self.scoring_function.as_ref().unwrap();
                let inter_e = eval_receptor_ligand_direct(&self.model, sf, 1000.0);
                self.poses[i].e = sf.conf_independent(num_tors, inter_e);
            }
        }
        self.poses.sort_by(|a, b| a.e.partial_cmp(&b.e).unwrap_or(std::cmp::Ordering::Equal));

        // Set model to best pose
        if let Some(best) = self.poses.first() {
            self.model.set(&best.c);
        }

        if self.verbosity > 0 {
            eprintln!("[Smina] Found {} poses | tors={:.1}", self.poses.len(), num_tors);
            if let Some(best) = self.poses.first() {
                eprintln!("[Smina] Best energy: {:.1} kcal/mol", best.e);
            }
        }
        Ok(())
    }

    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses(&mut self, how_many: usize, energy_range: f64) -> String {
        write_poses_pdbqt(&mut self.model, &self.poses, how_many, energy_range)
    }

    #[pyo3(signature = (output_name, how_many=9, energy_range=3.0))]
    fn write_poses(&mut self, output_name: &str, how_many: usize, energy_range: f64) -> PyResult<()> {
        std::fs::write(output_name, self.get_poses(how_many, energy_range))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot write file: {}", e)))?;
        Ok(())
    }

    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses_energies(&self, how_many: usize, energy_range: f64) -> Vec<Vec<f64>> {
        get_poses_energies_impl(&self.poses, how_many, energy_range)
    }

    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses_coordinates(&self, how_many: usize, energy_range: f64) -> Vec<Vec<f64>> {
        get_poses_coordinates_impl(&self.poses, how_many, energy_range)
    }

    /// Score the best pose. Returns [total, inter, intra, total_raw, grid_inter].
    /// Uses exact pair-by-pair evaluation matching smina --score_only.
    fn score(&mut self) -> PyResult<Vec<f64>> {
        if !self.ligand_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Ligand not initialized"));
        }
        // Always score the best (first) pose
        if let Some(best) = self.poses.first() {
            self.model.set(&best.c);
        }
        let p = self.precalculated.as_ref().unwrap();
        let sf = self.scoring_function.as_ref().unwrap();
        let v = 1000.0;
        let inter_e = eval_receptor_ligand_direct(&self.model, sf, v);
        let intra_e = self.model.eval_intra(p, v);
        let num_tors = self.model.conf_independent_num_tors();
        let total = sf.conf_independent(num_tors, inter_e);
        let grid_e = if let Some(cache) = &self.cache {
            cache.eval(&self.model, v)
        } else {
            f64::NAN
        };
        Ok(vec![total, inter_e, intra_e, inter_e + intra_e, grid_e])
    }

    fn num_poses(&self) -> usize { self.poses.len() }
    fn seed(&self) -> u64 { self.seed }

    fn __repr__(&self) -> String {
        format!("Smina(sf={:?}, cpu={}, seed={}, receptor={}, ligand={}, maps={})",
            self.sf_choice, self.cpu, self.seed,
            self.receptor_initialized, self.ligand_initialized, self.map_initialized)
    }

    /// Batch dock multiple ligands against the current receptor.
    /// Grid maps are computed once and reused for all ligands.
    /// Returns list of (poses_pdbqt, energies) for each ligand.
    #[pyo3(signature = (
        ligand_strings,
        center_x, center_y, center_z,
        size_x, size_y, size_z,
        exhaustiveness=8,
        n_poses=20,
        min_rmsd=1.0,
        max_evals=0,
        granularity=0.5,
        energy_range=3.0,
        how_many=9,
        progress_callback=None,
        stream_results=false,
    ))]
    fn batch_dock(
        &self, py: Python<'_>,
        ligand_strings: Vec<String>,
        center_x: f64, center_y: f64, center_z: f64,
        size_x: f64, size_y: f64, size_z: f64,
        exhaustiveness: usize,
        n_poses: usize,
        min_rmsd: f64,
        max_evals: i32,
        granularity: f64,
        energy_range: f64,
        how_many: usize,
        progress_callback: Option<PyObject>,
        stream_results: bool,
    ) -> PyResult<Vec<(String, Vec<Vec<f64>>)>> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Receptor not initialized. Call set_receptor() first."
            ));
        }

        let total_ligands = ligand_strings.len();
        let batch_cb = progress_callback.map(|cb| {
            build_batch_progress_callback(py, cb, total_ligands)
        });

        let receptor = self.receptor.clone();
        let sf_choice = self.sf_choice;
        let weights = self.weights.clone();
        let no_refine = self.no_refine;
        let verbosity = self.verbosity;
        let seed = self.seed;

        let results = py.allow_threads(move || {
            batch_dock_inner(
                &receptor, sf_choice, &weights,
                no_refine, verbosity, seed,
                &ligand_strings,
                center_x, center_y, center_z,
                size_x, size_y, size_z,
                granularity,
                exhaustiveness, n_poses, min_rmsd, max_evals,
                energy_range, how_many,
                true,
                SearchAlgorithm::QVina2,
                batch_cb,
                stream_results,
            )
        });

        Ok(results)
    }
}

// ─── PyRxDock ────────────────────────────────────────────────────────────────
// Cavity-based docking using Tripos 5.2 VDW 4-8 scoring + SA/Simplex search.
// Orthogonal to the Vina family: different scoring function, different search
// algorithm, accepts SDF input for ligands.

#[pyclass(name = "RxDock")]
pub struct PyRxDock {
    model: Model,
    receptor: Model,
    scoring: Option<RxScoringFunction>,
    cavity: Option<rxdock_cavity::Cavity>,
    poses: OutputContainer,
    cpu: usize,
    seed: u64,
    verbosity: i32,
    receptor_initialized: bool,
    ligand_initialized: bool,
    cavity_initialized: bool,
}

#[pymethods]
impl PyRxDock {
    #[new]
    #[pyo3(signature = (cpu=0, seed=0, verbosity=1))]
    fn new(cpu: i32, seed: i64, verbosity: i32) -> Self {
        PyRxDock {
            model: Model::new(),
            receptor: Model::new(),
            scoring: None,
            cavity: None,
            poses: Vec::new(),
            cpu: resolve_cpu(cpu),
            seed: resolve_seed(seed),
            verbosity,
            receptor_initialized: false,
            ligand_initialized: false,
            cavity_initialized: false,
        }
    }

    /// Set receptor from PDBQT file (same format as Vina).
    #[pyo3(signature = (rigid_name, flex_name=None))]
    fn set_receptor(&mut self, rigid_name: &str, flex_name: Option<&str>) -> PyResult<()> {
        do_set_receptor(&mut self.receptor, rigid_name, flex_name)?;
        self.receptor_initialized = true;
        self.ligand_initialized = false;
        self.cavity_initialized = false;
        self.scoring = None;
        Ok(())
    }

    /// Set ligand from SDF string (e.g., from RDKit Chem.MolToMolBlock()).
    fn set_ligand_from_sdf(&mut self, sdf_string: &str) -> PyResult<()> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Receptor not initialized. Call set_receptor() first."
            ));
        }
        self.model = self.receptor.clone();
        parse_sdf::parse_sdf_to_model(&mut self.model, sdf_string)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        self.ligand_initialized = true;
        // Rebuild scoring function with updated receptor cache
        self._rebuild_scoring();
        Ok(())
    }

    /// Set ligand from MOL2 string.
    fn set_ligand_from_mol2(&mut self, mol2_string: &str) -> PyResult<()> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Receptor not initialized. Call set_receptor() first."
            ));
        }
        self.model = self.receptor.clone();
        parse_sdf::parse_mol2_to_model(&mut self.model, mol2_string)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        self.ligand_initialized = true;
        self._rebuild_scoring();
        Ok(())
    }

    /// Set ligand from file (auto-detect format by extension: .sdf, .mol, .mol2).
    fn set_ligand_from_file(&mut self, path: &str) -> PyResult<()> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Receptor not initialized. Call set_receptor() first."
            ));
        }
        self.model = self.receptor.clone();

        let lower = path.to_lowercase();
        if lower.ends_with(".mol2") {
            let content = std::fs::read_to_string(path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot read file: {}", e)))?;
            parse_sdf::parse_mol2_to_model(&mut self.model, &content)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        } else if lower.ends_with(".sdf") || lower.ends_with(".mol") {
            parse_sdf::parse_sdf_file_to_model(&mut self.model, path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported file format: {} (use .sdf, .mol, or .mol2)", path)
            ));
        }

        self.ligand_initialized = true;
        self._rebuild_scoring();
        Ok(())
    }

    /// Detect cavities in the receptor binding site.
    /// Returns list of (center_x, center_y, center_z, volume) tuples.
    #[pyo3(signature = (center_x, center_y, center_z, radius=10.0, small_sphere=1.5, large_sphere=4.0, grid_step=0.5, min_volume=100.0))]
    fn detect_cavities(
        &mut self,
        center_x: f64, center_y: f64, center_z: f64,
        radius: f64, small_sphere: f64, large_sphere: f64,
        grid_step: f64, min_volume: f64,
    ) -> PyResult<Vec<Vec<f64>>> {
        if !self.receptor_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Receptor not initialized. Call set_receptor() first."
            ));
        }

        let params = rxdock_cavity::CavityParams {
            center: Vec3::new(center_x, center_y, center_z),
            radius,
            small_sphere,
            large_sphere,
            grid_step,
            min_volume,
            ..rxdock_cavity::CavityParams::default()
        };

        // Build receptor atoms for cavity detection (need coords + VDW radii)
        let receptor_atoms: Vec<rxdock_cavity::ReceptorAtom> = self.receptor.grid_atoms.iter()
            .map(|a| {
                let sy = ad_to_tripos(a.ad);
                let radius = TRIPOS_PARAMS[sy.min(SY_SIZE - 1)].radius;
                rxdock_cavity::ReceptorAtom {
                    coords: a.coords,
                    vdw_radius: radius,
                }
            })
            .collect();

        let cavities = rxdock_cavity::detect_cavities(&receptor_atoms, &params);

        if self.verbosity > 0 {
            eprintln!("[RxDock] Found {} cavities", cavities.len());
            for (i, c) in cavities.iter().enumerate() {
                eprintln!("  Cavity {}: center=({:.1}, {:.1}, {:.1}) volume={:.0} A^3",
                    i, c.center.x(), c.center.y(), c.center.z(), c.volume);
            }
        }

        // Store the largest cavity for docking
        if let Some(best) = cavities.first() {
            self.cavity = Some(best.clone());
            self.cavity_initialized = true;
            // Rebuild scoring with cavity restraint
            self._rebuild_scoring();
        }

        Ok(cavities.iter()
            .map(|c| vec![c.center.x(), c.center.y(), c.center.z(), c.volume])
            .collect())
    }

    /// Run rxDock search: parallel SA + Simplex.
    #[pyo3(signature = (n_runs=20, n_poses=9, min_rmsd=1.0, progress_callback=None))]
    fn dock(
        &mut self, py: Python<'_>,
        n_runs: usize, n_poses: usize, min_rmsd: f64,
        progress_callback: Option<PyObject>,
    ) -> PyResult<()> {
        if !self.ligand_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ligand not initialized. Call set_ligand_from_sdf() first."
            ));
        }
        if !self.cavity_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cavity not detected. Call detect_cavities() first."
            ));
        }

        let scoring = self.scoring.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Scoring function not initialized")
        })?;
        let cavity = self.cavity.as_ref().unwrap();

        let mut params = rxdock_search::RxDockParams::default();
        params.n_runs = n_runs;
        params.n_poses = n_poses;
        params.min_rmsd = min_rmsd;
        params.seed = self.seed;

        // Build progress callback
        let progress_arc: Option<Arc<rxdock_search::RxProgressCallback>> =
            progress_callback.map(|cb| {
                let cb_ref = Python::with_gil(|py| cb.clone_ref(py));
                Arc::new(Box::new(move |percent: f64, best_energy: f64| {
                    Python::with_gil(|py| {
                        let kwargs = PyDict::new(py);
                        let _ = kwargs.set_item("stage", "Docking");
                        let _ = kwargs.set_item("percent_complete", percent);
                        let _ = kwargs.set_item("best_energy", best_energy);
                        let _ = cb_ref.call(py, (), Some(&kwargs));
                    });
                }) as Box<dyn Fn(f64, f64) + Send + Sync>)
            });

        let model_clone = self.model.clone();
        let cavity_clone = cavity.clone();

        let results = py.allow_threads(move || {
            rxdock_search::rxdock_search(
                &model_clone,
                scoring,
                &cavity_clone,
                &params,
                progress_arc,
            )
        });

        self.poses = results;

        // Set model to best pose
        if let Some(best) = self.poses.first() {
            self.model.set(&best.c);
        }

        if self.verbosity > 0 {
            eprintln!("[RxDock] Found {} poses", self.poses.len());
            if let Some(best) = self.poses.first() {
                eprintln!("[RxDock] Best energy: {:.1} kcal/mol", best.e);
            }
        }

        Ok(())
    }

    /// Score the current (best) pose.
    /// Returns [total, vdw_inter, vdw_intra, polar_attr, polar_repul, rot_entropy, const_penalty, cavity, dihedral].
    fn score(&mut self) -> PyResult<Vec<f64>> {
        if !self.ligand_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Ligand not initialized"));
        }
        if let Some(best) = self.poses.first() {
            self.model.set(&best.c);
        }
        let scoring = self.scoring.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Scoring function not initialized")
        })?;
        let terms = scoring.score_terms(&self.model);
        Ok(vec![
            terms.total, terms.vdw_inter, terms.vdw_intra,
            terms.polar_attr, terms.polar_repul,
            terms.rot_entropy, terms.const_penalty, terms.cavity,
            terms.dihedral,
        ])
    }

    /// Get energy values for top poses.
    /// Returns list of [total, intra, inter, conf_independent, unbound, total_raw].
    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses_energies(&self, how_many: usize, energy_range: f64) -> Vec<Vec<f64>> {
        get_poses_energies_impl(&self.poses, how_many, energy_range)
    }

    /// Get heavy atom coordinates for top poses.
    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses_coordinates(&self, how_many: usize, energy_range: f64) -> Vec<Vec<f64>> {
        get_poses_coordinates_impl(&self.poses, how_many, energy_range)
    }

    /// Write poses as PDBQT (for visualization compatibility).
    #[pyo3(signature = (how_many=9, energy_range=3.0))]
    fn get_poses(&mut self, how_many: usize, energy_range: f64) -> String {
        write_poses_pdbqt(&mut self.model, &self.poses, how_many, energy_range)
    }

    #[pyo3(signature = (output_name, how_many=9, energy_range=3.0))]
    fn write_poses(&mut self, output_name: &str, how_many: usize, energy_range: f64) -> PyResult<()> {
        std::fs::write(output_name, self.get_poses(how_many, energy_range))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot write file: {}", e)))?;
        Ok(())
    }

    fn num_poses(&self) -> usize { self.poses.len() }
    fn seed(&self) -> u64 { self.seed }

    fn __repr__(&self) -> String {
        format!("RxDock(cpu={}, seed={}, receptor={}, ligand={}, cavity={})",
            self.cpu, self.seed,
            self.receptor_initialized, self.ligand_initialized, self.cavity_initialized)
    }
}

impl PyRxDock {
    /// Rebuild the scoring function from current state.
    fn _rebuild_scoring(&mut self) {
        if !self.receptor_initialized { return; }

        let receptor_cache = build_receptor_cache(&self.model);
        let mut sf = RxScoringFunction::new(
            receptor_cache,
            self.cavity.clone(),
            RxScoringWeights::default(),
        );
        // Initialize H-bond interaction centers from receptor
        sf.init_polar(&self.model);
        self.scoring = Some(sf);
    }
}
