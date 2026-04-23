//! SDF → PDBQT converter for molecular docking.
//!
//! Converts an sdfrust `Molecule` (with explicit hydrogens) into AutoDock
//! PDBQT format suitable for Vina/QVina2/Smina.
//!
//! Pipeline:
//!   1. Gasteiger partial charges
//!   2. AutoDock atom type assignment (rule-based, no SMARTS)
//!   3. Non-polar hydrogen merging (charge transfer to parent)
//!   4. Rotatable bond detection
//!   5. Torsion tree construction (rigid body decomposition)
//!   6. PDBQT output with ROOT/BRANCH/ENDBRANCH tree

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::Write;

use crate::bond::BondOrder;
use crate::descriptors::{
    all_aromatic_atoms, compute_ring_membership, gasteiger_charges, is_hydrogen,
};
use crate::error::{Result, SdfError};
use crate::molecule::Molecule;

// ─── AutoDock Atom Types ────────────────────────────────────────────────────

/// Assign AutoDock atom types to each atom.
///
/// Rules follow Meeko's defaults_json SMARTS patterns, translated to
/// element + property checks (last-match-wins order):
///
/// | Element | Default | Override → Type |
/// |---------|---------|-----------------|
/// | H       | H       | bonded to N/O/F/P/S → HD |
/// | C       | C       | aromatic → A |
/// | N       | NA      | charged(+) → N; degree=3,valence=3 + aromatic/amide neighbor → N |
/// | O       | OA      | — |
/// | S       | S       | degree=2 → SA |
/// | others  | element symbol |
fn assign_atom_types(mol: &Molecule) -> Vec<&'static str> {
    let aromatic = all_aromatic_atoms(mol);
    assign_atom_types_with_arom(mol, &aromatic)
}

/// Assign atom types using pre-computed aromaticity.
fn assign_atom_types_with_arom(mol: &Molecule, aromatic: &[bool]) -> Vec<&'static str> {
    let n = mol.atoms.len();
    let mut types = vec![""; n];

    for (i, atom) in mol.atoms.iter().enumerate() {
        let elem = atom.element.trim();
        let elem_upper: String = elem.to_uppercase();

        types[i] = match elem_upper.as_str() {
            "H" | "D" | "T" => {
                // Check if bonded to heteroatom → HD
                let is_polar = mol.neighbors(i).iter().any(|&nb| {
                    let nb_elem = mol.atoms[nb].element.trim().to_uppercase();
                    matches!(nb_elem.as_str(), "N" | "O" | "F" | "P" | "S")
                });
                if is_polar { "HD" } else { "H" }
            }
            "C" => {
                if i < aromatic.len() && aromatic[i] {
                    "A"
                } else {
                    "C"
                }
            }
            "N" => assign_nitrogen_type(mol, i, &aromatic),
            "O" => "OA",
            "S" => {
                // Meeko: [SX2] → SA (aliphatic S only; aromatic s stays S)
                let degree = mol.bonds_for_atom(i).len();
                let is_arom = i < aromatic.len() && aromatic[i];
                if degree == 2 && !is_arom { "SA" } else { "S" }
            }
            "F" => "F",
            "CL" => "Cl",
            "BR" => "Br",
            "I" => "I",
            "P" => "P",
            "B" => "B",
            "MG" => "Mg",
            "CA" => "Ca",
            "MN" => "Mn",
            "FE" => "Fe",
            "ZN" => "Zn",
            "SI" => "Si",
            "SE" => "SA", // Selenium treated like sulfur acceptor
            _ => "C",     // Fallback
        };
    }

    types
}

/// Determine nitrogen atom type: NA (acceptor) vs N (donor/amide/charged).
fn assign_nitrogen_type(mol: &Molecule, idx: usize, aromatic: &[bool]) -> &'static str {
    let atom = &mol.atoms[idx];

    // [#7+1] → N (positively charged nitrogen)
    if atom.formal_charge > 0 {
        return "N";
    }

    // Count ALL explicit bonds and bond order sum (X and v include implicit H,
    // which RDKit restores in mol_noH, so total degree == original degree)
    let bonds = mol.bonds_for_atom(idx);
    let degree = bonds.len();
    let valence: f64 = bonds.iter().map(|b| b.order.order()).sum();
    let valence_int = valence.round() as u32;

    // [#7X3v3] patterns: N with 3 connections and valence 3 (all single bonds)
    if degree == 3 && valence_int == 3 {
        let neighbors = mol.neighbors(idx);

        // [#7X3v3][a] → N (pyrrole, aniline: bonded to aromatic atom)
        for &nb in &neighbors {
            if nb < aromatic.len() && aromatic[nb] {
                return "N";
            }
        }

        // [#7X3v3][#6X3v4] → N (amide: bonded to C with degree=3, valence=4)
        for &nb in &neighbors {
            let nb_elem = mol.atoms[nb].element.trim().to_uppercase();
            if nb_elem == "C" {
                let nb_bonds = mol.bonds_for_atom(nb);
                let nb_degree = nb_bonds.len();
                let nb_valence: f64 = nb_bonds.iter().map(|b| b.order.order()).sum();
                let nb_valence_int = nb_valence.round() as u32;
                if nb_degree == 3 && nb_valence_int == 4 {
                    return "N";
                }
            }
        }
    }

    // Default: NA (nitrogen acceptor)
    "NA"
}

// ─── Hydrogen Merging ───────────────────────────────────────────────────────

/// Merge non-polar hydrogens: transfer charge to parent atom, mark as merged.
///
/// Only atoms with type "H" (non-polar) are merged.  "HD" atoms (polar H on
/// N/O/F/P/S) remain as explicit atoms in the PDBQT output.
///
/// Returns a boolean vector where `merged[i] == true` means atom i is excluded.
fn merge_hydrogens(mol: &Molecule, atom_types: &[&str], charges: &mut [f64]) -> Vec<bool> {
    let n = mol.atoms.len();
    let mut merged = vec![false; n];

    for i in 0..n {
        if atom_types[i] != "H" {
            continue;
        }
        // H must have exactly 1 neighbor to merge
        let neighbors = mol.neighbors(i);
        if neighbors.len() != 1 {
            continue;
        }
        let parent = neighbors[0];
        charges[parent] += charges[i];
        charges[i] = 0.0;
        merged[i] = true;
    }

    merged
}

// ─── Rotatable Bond Detection ───────────────────────────────────────────────

/// Detect rotatable bonds for PDBQT torsion tree.
///
/// A bond is rotatable if:
///   1. Single bond
///   2. Not in a ring
///   3. Not terminal (both atoms have heavy_degree > 1, ignoring merged atoms)
///   4. Not an amide bond (N(d3)–C(d3, has C=O or C=N), C must not be aromatic)
///   5. Neither atom is a merged hydrogen
fn find_rotatable_bonds(mol: &Molecule, merged: &[bool], aromatic: &[bool], ext_symmetry: Option<&[u32]>) -> Vec<bool> {
    let m = mol.bonds.len();
    if m == 0 {
        return vec![];
    }

    let (_, bond_in_ring) = compute_ring_membership(mol);
    // Compute heavy-atom degrees excluding merged atoms
    let heavy_deg = compute_non_merged_degrees(mol, merged);
    // Compute symmetry classes for tertiary amide equivalence check.
    // If external symmetry classes are provided (e.g. from RDKit), use them
    // (converted to u64); otherwise compute Morgan-style classes locally.
    let symmetry: Vec<u64> = if let Some(ext) = ext_symmetry {
        ext.iter().map(|&v| v as u64).collect()
    } else {
        compute_symmetry_classes(mol, aromatic)
    };

    let mut rotatable = vec![false; m];

    for (bi, bond) in mol.bonds.iter().enumerate() {
        // 1. Single bond only
        if bond.order != BondOrder::Single {
            continue;
        }

        // 2. Not in ring
        if bi < bond_in_ring.len() && bond_in_ring[bi] {
            continue;
        }

        let a1 = bond.atom1;
        let a2 = bond.atom2;

        // 5. Skip if either atom is merged
        if merged[a1] || merged[a2] {
            continue;
        }

        // Skip if either is hydrogen (non-merged HD)
        if is_hydrogen(Some(mol.atoms[a1].element.trim()))
            || is_hydrogen(Some(mol.atoms[a2].element.trim()))
        {
            continue;
        }

        // 3. Not terminal
        if heavy_deg[a1] <= 1 || heavy_deg[a2] <= 1 {
            continue;
        }

        // 4. Not amide: N(degree=3)–C(degree=3, has double bond to O or N)
        //    Skip amide check if C atom is aromatic (Kekulized C=N in aromatic
        //    rings would falsely match the amide pattern)
        //    Exception: tertiary amides with non-equivalent substituents on N
        //    are allowed to rotate (matches Meeko's BondTyperLegacy behavior)
        if is_amide_bond(mol, a1, a2, aromatic) || is_amide_bond(mol, a2, a1, aromatic) {
            let n_idx = if mol.atoms[a1].element.trim().eq_ignore_ascii_case("N") { a1 } else { a2 };
            let c_idx = if n_idx == a1 { a2 } else { a1 };
            if !is_tertiary_amide_rotatable(mol, n_idx, c_idx, merged, &symmetry) {
                continue;
            }
        }

        rotatable[bi] = true;
    }

    rotatable
}

/// Compute degree of each atom counting only non-merged neighbors.
///
/// Non-merged polar H (HD) atoms count toward degree because they appear
/// as explicit atoms in the PDBQT output. Only merged (non-polar) H are excluded.
fn compute_non_merged_degrees(mol: &Molecule, merged: &[bool]) -> Vec<usize> {
    let n = mol.atoms.len();
    let mut degrees = vec![0usize; n];
    for bond in &mol.bonds {
        let a1 = bond.atom1;
        let a2 = bond.atom2;
        if a1 >= n || a2 >= n {
            continue;
        }
        if merged[a1] || merged[a2] {
            continue;
        }
        degrees[a1] += 1;
        degrees[a2] += 1;
    }
    degrees
}

/// Check if bond a1–a2 is an amide bond where a1=N, a2=C.
/// Amide: [NX3]–[CX3]=O or [NX3]–[CX3]=N
///
/// Neither atom may be aromatic:
/// - C: in Kekulized SDF, aromatic C=N would falsely match
/// - N: aromatic N (e.g. carbazole, pyrrole) forms a different kind of
///   conjugation; RDKit's SMARTS `[NX3]` only matches aliphatic N
fn is_amide_bond(mol: &Molecule, a1: usize, a2: usize, aromatic: &[bool]) -> bool {
    let elem1 = mol.atoms[a1].element.trim().to_uppercase();
    let elem2 = mol.atoms[a2].element.trim().to_uppercase();
    if elem1 != "N" || elem2 != "C" {
        return false;
    }

    // N must not be aromatic (aromatic N like carbazole/pyrrole is not a
    // traditional amide — RDKit SMARTS [NX3] only matches aliphatic N)
    if a1 < aromatic.len() && aromatic[a1] {
        return false;
    }

    // C must not be aromatic (Kekulized C=N in aromatic rings is not amide)
    if a2 < aromatic.len() && aromatic[a2] {
        return false;
    }

    // N must have degree 3 (among all explicit bonds)
    let n_degree = mol.bonds_for_atom(a1).len();
    if n_degree != 3 {
        return false;
    }

    // C must have degree 3
    let c_bonds = mol.bonds_for_atom(a2);
    if c_bonds.len() != 3 {
        return false;
    }

    // C must have a double bond to O or N
    for b in &c_bonds {
        if b.order == BondOrder::Double {
            let other = if b.atom1 == a2 { b.atom2 } else { b.atom1 };
            let other_elem = mol.atoms[other].element.trim().to_uppercase();
            if other_elem == "O" || other_elem == "N" {
                return true;
            }
        }
    }

    false
}

/// Compute Morgan-style symmetry classes for all atoms in a molecule.
///
/// This implements an iterative refinement algorithm similar to RDKit's
/// `CanonicalRankAtoms(breakTies=False)`.  Each atom starts with an
/// invariant based on (element, degree, bond-order-sum, aromaticity,
/// ring-membership, H-count, chirality).  Then, in each iteration, the
/// invariant is updated to include the sorted list of neighbor invariants.
/// Iteration stops when the number of distinct classes stops growing.
fn compute_symmetry_classes(mol: &Molecule, aromatic: &[bool]) -> Vec<u64> {
    use std::collections::hash_map::DefaultHasher;

    let n = mol.atoms.len();
    if n == 0 {
        return vec![];
    }

    let (ring_member, _) = compute_ring_membership(mol);

    // Initial invariant per atom
    let mut ranks = vec![0u64; n];
    for i in 0..n {
        let elem = mol.atoms[i].element.trim();
        let degree = mol.bonds_for_atom(i).len();
        let bo_sum: u32 = mol.bonds_for_atom(i).iter().map(|b| match b.order {
            BondOrder::Double => 2u32,
            BondOrder::Triple => 3u32,
            _ => 1u32,
        }).sum();
        let arom = if i < aromatic.len() { aromatic[i] } else { false };
        let in_ring = if i < ring_member.len() { ring_member[i] } else { false };
        // Count explicit H neighbors
        let h_count: u32 = mol.neighbors(i).iter()
            .filter(|&&nb| is_hydrogen(Some(mol.atoms[nb].element.trim())))
            .count() as u32;
        let mut hasher = DefaultHasher::new();
        elem.hash(&mut hasher);
        degree.hash(&mut hasher);
        bo_sum.hash(&mut hasher);
        arom.hash(&mut hasher);
        in_ring.hash(&mut hasher);
        h_count.hash(&mut hasher);
        ranks[i] = hasher.finish();
    }

    // Iterative refinement
    let max_iters = n.min(100);
    for _ in 0..max_iters {
        let mut new_ranks = vec![0u64; n];
        for i in 0..n {
            let mut nb_ranks: Vec<u64> = mol.neighbors(i).iter()
                .map(|&j| ranks[j])
                .collect();
            nb_ranks.sort();

            let mut hasher = DefaultHasher::new();
            ranks[i].hash(&mut hasher);
            nb_ranks.hash(&mut hasher);
            new_ranks[i] = hasher.finish();
        }

        let old_classes = ranks.iter().collect::<HashSet<_>>().len();
        let new_classes = new_ranks.iter().collect::<HashSet<_>>().len();
        ranks = new_ranks;

        if new_classes <= old_classes {
            break; // Converged
        }
    }

    ranks
}

/// Check if a tertiary amide bond should be rotatable.
///
/// Meeko makes tertiary amides rotatable when the two non-H substituents on N
/// are non-equivalent (different symmetry classes via `CanonicalRankAtoms`).
/// We use a Morgan-style symmetry analysis to determine equivalence.
///
/// Returns true if the amide should be made ROTATABLE (i.e., it's a tertiary
/// amide with non-equivalent substituents).
fn is_tertiary_amide_rotatable(mol: &Molecule, n_idx: usize, c_idx: usize, merged: &[bool], symmetry: &[u64]) -> bool {
    // Get N's non-H, non-amide-C neighbors
    let n_neighbors = mol.neighbors(n_idx);
    let mut substituents: Vec<usize> = Vec::new();
    for &nb in &n_neighbors {
        if nb == c_idx {
            continue; // Skip the amide C
        }
        if nb < merged.len() && merged[nb] {
            continue; // Skip merged H
        }
        if is_hydrogen(Some(mol.atoms[nb].element.trim())) {
            continue; // Skip non-merged H (HD)
        }
        substituents.push(nb);
    }

    // Need exactly 2 non-H substituents for tertiary amide
    if substituents.len() != 2 {
        return false;
    }

    let r1 = substituents[0];
    let r2 = substituents[1];

    // Exclude positively-charged nitrogen (e.g. nitro [N+](=O)[O-]) from
    // the tertiary amide exception — these are not true amides.
    if mol.atoms[n_idx].formal_charge > 0 {
        return false;
    }

    // Compare symmetry classes — non-equivalent substituents make bond rotatable
    symmetry[r1] != symmetry[r2]
}

// ─── Torsion Tree ───────────────────────────────────────────────────────────

/// Rigid body graph for PDBQT BRANCH/ENDBRANCH output.
struct TorsionTree {
    /// rb_id → atom indices in that rigid body
    rigid_bodies: Vec<Vec<usize>>,
    /// rb_id → neighbor rb_ids
    graph: Vec<Vec<usize>>,
    /// (rb1, rb2) → (atom_in_rb1, atom_in_rb2) at the rotatable bond
    connectivity: HashMap<(usize, usize), (usize, usize)>,
    /// Index of root rigid body
    root: usize,
    /// Number of torsions (= rigid_bodies.len() - 1)
    torsions: usize,
}

/// Build the torsion tree from rotatable bond information.
fn build_torsion_tree(mol: &Molecule, rotatable: &[bool], merged: &[bool]) -> TorsionTree {
    let n = mol.atoms.len();

    // Build adjacency list with bond indices
    let mut adj: Vec<Vec<(usize, usize)>> = vec![vec![]; n]; // (neighbor, bond_idx)
    for (bi, bond) in mol.bonds.iter().enumerate() {
        if bond.atom1 < n && bond.atom2 < n {
            adj[bond.atom1].push((bond.atom2, bi));
            adj[bond.atom2].push((bond.atom1, bi));
        }
    }

    let mut visited = vec![false; n];
    let mut rb_index = vec![usize::MAX; n]; // atom → rigid body id
    let mut rigid_bodies: Vec<Vec<usize>> = Vec::new();
    let mut graph: Vec<Vec<usize>> = Vec::new();
    let mut connectivity: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut sprouts: Vec<(usize, usize, usize)> = Vec::new(); // (from_rb, from_atom, to_atom)

    // Find first non-merged atom as start
    let start = (0..n).find(|&i| !merged[i]).unwrap_or(0);

    // BFS to build rigid bodies
    walk_rigid_body(
        start, &adj, rotatable, merged, &mut visited, &mut rb_index,
        &mut rigid_bodies, &mut graph, &mut sprouts,
    );

    // Process sprouts recursively (rotatable bond connections)
    while let Some((from_rb, from_atom, to_atom)) = sprouts.pop() {
        if visited[to_atom] {
            continue;
        }
        let new_rb = rigid_bodies.len();
        // Record connectivity
        connectivity.insert((from_rb, new_rb), (from_atom, to_atom));
        connectivity.insert((new_rb, from_rb), (to_atom, from_atom));
        graph[from_rb].push(new_rb);
        graph.push(vec![from_rb]);
        rigid_bodies.push(Vec::new());

        walk_rigid_body(
            to_atom, &adj, rotatable, merged, &mut visited, &mut rb_index,
            &mut rigid_bodies, &mut graph, &mut sprouts,
        );
    }

    // Root selection: iteratively remove leaves until 1-2 remain
    let root = select_root(&rigid_bodies, &graph);
    let torsions = if rigid_bodies.is_empty() {
        0
    } else {
        rigid_bodies.len() - 1
    };

    TorsionTree {
        rigid_bodies,
        graph,
        connectivity,
        root,
        torsions,
    }
}

/// BFS from `start` atom, grouping all atoms reachable via non-rotatable bonds
/// into the current rigid body.
fn walk_rigid_body(
    start: usize,
    adj: &[Vec<(usize, usize)>],
    rotatable: &[bool],
    merged: &[bool],
    visited: &mut [bool],
    rb_index: &mut [usize],
    rigid_bodies: &mut Vec<Vec<usize>>,
    graph: &mut Vec<Vec<usize>>,
    sprouts: &mut Vec<(usize, usize, usize)>,
) {
    let rb_id = if rigid_bodies.last().map_or(true, |rb| !rb.is_empty() || rb_index[start] != usize::MAX) {
        // Need a new rigid body
        let id = rigid_bodies.len();
        rigid_bodies.push(Vec::new());
        graph.push(Vec::new());
        id
    } else {
        rigid_bodies.len() - 1
    };

    let mut queue = vec![start];
    visited[start] = true;
    rb_index[start] = rb_id;

    let mut head = 0;
    while head < queue.len() {
        let current = queue[head];
        head += 1;

        rigid_bodies[rb_id].push(current);

        for &(neighbor, bond_idx) in &adj[current] {
            if visited[neighbor] {
                continue;
            }
            if merged[neighbor] {
                continue;
            }

            let is_rot = bond_idx < rotatable.len() && rotatable[bond_idx];
            if is_rot {
                // Rotatable bond → sprout for new rigid body
                sprouts.push((rb_id, current, neighbor));
            } else {
                // Non-rotatable → same rigid body
                visited[neighbor] = true;
                rb_index[neighbor] = rb_id;
                queue.push(neighbor);
            }
        }
    }
}

/// Select root rigid body by iteratively removing leaves.
fn select_root(rigid_bodies: &[Vec<usize>], graph: &[Vec<usize>]) -> usize {
    let n = rigid_bodies.len();
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 0;
    }

    // Work on a copy of the degree counts
    let mut degree: Vec<usize> = graph.iter().map(|g| g.len()).collect();
    let mut removed = vec![false; n];
    let mut remaining = n;

    while remaining > 2 {
        let mut leaves = Vec::new();
        for i in 0..n {
            if !removed[i] && degree[i] <= 1 {
                leaves.push(i);
            }
        }
        if leaves.is_empty() {
            break;
        }
        for &leaf in &leaves {
            removed[leaf] = true;
            remaining -= 1;
            // Decrement neighbors' degrees
            for &nb in &graph[leaf] {
                if !removed[nb] && degree[nb] > 0 {
                    degree[nb] -= 1;
                }
            }
        }
    }

    // Pick largest remaining rigid body
    let mut best = 0;
    let mut best_size = 0;
    for i in 0..n {
        if !removed[i] && rigid_bodies[i].len() > best_size {
            best_size = rigid_bodies[i].len();
            best = i;
        }
    }

    best
}

// ─── PDBQT Writer ───────────────────────────────────────────────────────────

/// Write REMARK pairs (SMILES IDX or H PARENT) with line wrapping.
fn write_remark_pairs<W: Write>(w: &mut W, prefix: &str, pairs: &[(usize, usize)]) -> Result<()> {
    if pairs.is_empty() { return Ok(()); }
    let mut line = String::from(prefix);
    let mut pairs_on_line = 0;
    for &(a, b) in pairs {
        line.push_str(&format!(" {} {}", a, b));
        pairs_on_line += 1;
        if pairs_on_line >= 12 {
            writeln!(w, "{}", line).map_err(SdfError::Io)?;
            line = String::from(prefix);
            pairs_on_line = 0;
        }
    }
    if pairs_on_line > 0 {
        writeln!(w, "{}", line).map_err(SdfError::Io)?;
    }
    Ok(())
}

/// Write PDBQT format to a writer.
///
/// When `smiles_atom_order` is provided (list of original atom indices in SMILES
/// output order), the REMARK SMILES IDX and REMARK H PARENT lines are computed
/// automatically from the internal atom numbering.  Otherwise, pre-computed
/// `smiles_idx` and `h_parent` pairs are written directly.
fn write_pdbqt<W: Write>(
    w: &mut W,
    mol: &Molecule,
    atom_types: &[&str],
    charges: &[f64],
    merged: &[bool],
    tree: &TorsionTree,
    smiles: Option<&str>,
    smiles_idx: Option<&[(usize, usize)]>,
    h_parent: Option<&[(usize, usize)]>,
    smiles_atom_order: Option<&[usize]>,
) -> Result<()> {
    // 1. Write tree to buffer first (builds numbering map)
    let mut tree_buf = Vec::new();
    let mut numbering: HashMap<usize, usize> = HashMap::new();
    let mut count = 1usize;
    let mut elem_counters: HashMap<String, usize> = HashMap::new();

    write_tree_recursive(
        &mut tree_buf, mol, atom_types, charges, merged, tree,
        tree.root, None, &mut numbering, &mut count, &mut elem_counters,
        true, 0,
    )?;

    // 2. Remark header
    let name = if mol.name.is_empty() { "UNK" } else { &mol.name };
    writeln!(w, "REMARK  Name = {}", name).map_err(SdfError::Io)?;
    writeln!(w, "REMARK  {} active torsions", tree.torsions).map_err(SdfError::Io)?;

    // 3. REMARK SMILES
    if let Some(smi) = smiles {
        writeln!(w, "REMARK SMILES {}", smi).map_err(SdfError::Io)?;
    }

    // 4. REMARK SMILES IDX + H PARENT
    if let Some(order) = smiles_atom_order {
        // Auto-compute from numbering + smiles_atom_order
        // order[i] = original mol atom index for SMILES position i
        let inv: HashMap<usize, usize> = order.iter().enumerate()
            .map(|(i, &v)| (v, i + 1))  // mol_atom_idx -> smiles_pos (1-indexed)
            .collect();

        // SMILES IDX: (smiles_pos_1indexed, pdbqt_serial) for each SMILES atom
        let computed_idx: Vec<(usize, usize)> = order.iter().enumerate()
            .filter_map(|(smi_pos, &mol_idx)| {
                numbering.get(&mol_idx).map(|&serial| (smi_pos + 1, serial))
            })
            .collect();
        write_remark_pairs(w, "REMARK SMILES IDX", &computed_idx)?;

        // H PARENT: for each H in PDBQT (has serial) that's NOT in SMILES
        let mut computed_hp: Vec<(usize, usize)> = Vec::new();
        for (&atom_idx, &h_serial) in &numbering {
            if inv.contains_key(&atom_idx) { continue; }
            if atom_idx >= mol.atoms.len() { continue; }
            if !mol.atoms[atom_idx].element.trim().eq_ignore_ascii_case("H") { continue; }
            // Find parent heavy atom via bonds
            for bond in &mol.bonds {
                let parent = if bond.atom1 == atom_idx { bond.atom2 }
                    else if bond.atom2 == atom_idx { bond.atom1 }
                    else { continue };
                if let Some(&parent_smi_pos) = inv.get(&parent) {
                    computed_hp.push((parent_smi_pos, h_serial));
                }
                break;
            }
        }
        computed_hp.sort();
        write_remark_pairs(w, "REMARK H PARENT", &computed_hp)?;
    } else {
        // Use pre-computed pairs (existing behavior)
        if let Some(idx_pairs) = smiles_idx {
            write_remark_pairs(w, "REMARK SMILES IDX", idx_pairs)?;
        }
        if let Some(hp) = h_parent {
            write_remark_pairs(w, "REMARK H PARENT", hp)?;
        }
    }

    // 5. Write tree body
    w.write_all(&tree_buf).map_err(SdfError::Io)?;

    // 6. TORSDOF
    writeln!(w, "TORSDOF {}", tree.torsions).map_err(SdfError::Io)?;

    Ok(())
}

/// Recursively write the rigid body tree.
fn write_tree_recursive<W: Write>(
    w: &mut W,
    mol: &Molecule,
    atom_types: &[&str],
    charges: &[f64],
    merged: &[bool],
    tree: &TorsionTree,
    node: usize,
    parent_rb: Option<usize>,
    numbering: &mut HashMap<usize, usize>,
    count: &mut usize,
    elem_counters: &mut HashMap<String, usize>,
    is_root: bool,
    edge_start: usize,
) -> Result<()> {
    if is_root {
        writeln!(w, "ROOT").map_err(SdfError::Io)?;
    }

    // Get members: for root, sort; for branches, put edge_start atom first
    let members = if node < tree.rigid_bodies.len() {
        &tree.rigid_bodies[node]
    } else {
        return Ok(());
    };

    let ordered: Vec<usize> = if is_root {
        let mut m = members.clone();
        m.sort();
        m
    } else {
        // Put edge_start first, then rest
        let mut m = Vec::with_capacity(members.len());
        m.push(edge_start);
        for &atom in members {
            if atom != edge_start {
                m.push(atom);
            }
        }
        m
    };

    // Write ATOM lines for this rigid body
    for &atom_idx in &ordered {
        if merged[atom_idx] {
            continue;
        }
        write_atom_line(w, mol, atom_idx, atom_types[atom_idx], charges[atom_idx], count, numbering, elem_counters)?;
    }

    if is_root {
        writeln!(w, "ENDROOT").map_err(SdfError::Io)?;
    }

    // Recurse into child rigid bodies (skip parent)
    for &child in &tree.graph[node] {
        if Some(child) == parent_rb {
            continue;
        }
        if let Some(&(begin_atom, next_atom)) = tree.connectivity.get(&(node, child)) {
            let begin_serial = numbering[&begin_atom];
            let end_serial = *count;

            writeln!(w, "BRANCH {:>3} {:>3}", begin_serial, end_serial).map_err(SdfError::Io)?;

            write_tree_recursive(
                w, mol, atom_types, charges, merged, tree,
                child, Some(node), numbering, count, elem_counters,
                false, next_atom,
            )?;

            writeln!(w, "ENDBRANCH {:>3} {:>3}", begin_serial, end_serial).map_err(SdfError::Io)?;
        }
    }

    Ok(())
}

/// Write a single ATOM line in PDBQT format.
fn write_atom_line<W: Write>(
    w: &mut W,
    mol: &Molecule,
    atom_idx: usize,
    atom_type: &str,
    charge: f64,
    count: &mut usize,
    numbering: &mut HashMap<usize, usize>,
    elem_counters: &mut HashMap<String, usize>,
) -> Result<()> {
    let atom = &mol.atoms[atom_idx];
    let serial = *count;
    numbering.insert(atom_idx, serial);

    // Generate atom name: element + counter, padded to 4 chars
    let elem = atom.element.trim();
    let elem_key = elem.to_uppercase();
    let counter = elem_counters.entry(elem_key.clone()).or_insert(0);
    *counter += 1;
    let raw_name = format!("{}{}", elem_key, counter);
    // PDB atom name formatting: 1-char elements right-justified in cols 13-14,
    // 2-char elements left-justified
    let atom_name = if elem_key.len() == 1 {
        format!(" {:<3}", raw_name)
    } else {
        format!("{:<4}", raw_name)
    };

    // Clamp charge to finite value
    let q = if charge.is_finite() { charge } else { 0.0 };

    write!(
        w,
        "ATOM  {:>5} {:4} LIG A   1    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}    {:>6.3} {:<2}\n",
        serial,
        atom_name,
        atom.x,
        atom.y,
        atom.z,
        1.0_f64,  // occupancy
        0.0_f64,  // temp factor
        q,
        atom_type,
    )
    .map_err(SdfError::Io)?;

    *count += 1;
    Ok(())
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Convert a molecule (with explicit hydrogens) to PDBQT format string.
///
/// The molecule must have 3D coordinates and explicit hydrogens.
/// Non-polar hydrogens are merged (charge transferred to parent).
/// Polar hydrogens (on N/O/F/P/S) are kept as HD atoms.
///
/// Returns the PDBQT string with ROOT/BRANCH/ENDBRANCH torsion tree.
pub fn mol_to_pdbqt(mol: &Molecule) -> Result<String> {
    mol_to_pdbqt_ext(mol, None, None, None, None, None, None)
}

/// Convert a molecule to PDBQT format with optional REMARK lines.
///
/// # Arguments
/// * `smiles` - Optional SMILES string for REMARK SMILES line
/// * `smiles_idx` - Optional pairs of (original_atom_index, pdbqt_serial) for REMARK SMILES IDX
/// * `h_parent` - Optional pairs of (parent_pdbqt_serial, h_pdbqt_serial) for REMARK H PARENT
///
/// These are typically computed by RDKit on the Python side.
pub fn mol_to_pdbqt_with_remarks(
    mol: &Molecule,
    smiles: Option<&str>,
    smiles_idx: Option<&[(usize, usize)]>,
    h_parent: Option<&[(usize, usize)]>,
) -> Result<String> {
    mol_to_pdbqt_ext(mol, smiles, smiles_idx, h_parent, None, None, None)
}

/// Convert a molecule to PDBQT format with full options.
///
/// # Arguments
/// * `smiles` - Optional SMILES string for REMARK SMILES line
/// * `smiles_idx` - Optional pairs of (original_atom_index, pdbqt_serial) for REMARK SMILES IDX
/// * `h_parent` - Optional pairs of (parent_pdbqt_serial, h_pdbqt_serial) for REMARK H PARENT
/// * `aromatic_atoms` - Optional pre-computed aromaticity (e.g., from RDKit).
///   When provided, overrides the built-in Hückel aromaticity detection.
///   This is recommended for best accuracy with Kekulized SDF files.
/// * `symmetry_classes` - Optional pre-computed symmetry classes (e.g., from RDKit).
/// * `smiles_atom_order` - Optional list of original atom indices in SMILES output
///   order.  When provided together with `smiles`, the REMARK SMILES IDX and
///   REMARK H PARENT lines are computed automatically (overrides `smiles_idx`
///   and `h_parent`).
pub fn mol_to_pdbqt_ext(
    mol: &Molecule,
    smiles: Option<&str>,
    smiles_idx: Option<&[(usize, usize)]>,
    h_parent: Option<&[(usize, usize)]>,
    aromatic_atoms: Option<&[bool]>,
    symmetry_classes: Option<&[u32]>,
    smiles_atom_order: Option<&[usize]>,
) -> Result<String> {
    if mol.atoms.is_empty() {
        return Err(SdfError::PdbqtConversion("Empty molecule".to_string()));
    }

    // 1. Gasteiger charges
    let mut charges = gasteiger_charges(mol);

    // 2. Aromaticity: use provided or compute
    let aromatic = if let Some(arom) = aromatic_atoms {
        // Extend/truncate to match atom count
        let n = mol.atoms.len();
        let mut v = vec![false; n];
        for (i, &a) in arom.iter().enumerate() {
            if i < n {
                v[i] = a;
            }
        }
        v
    } else {
        all_aromatic_atoms(mol)
    };

    // 3. AutoDock atom types
    let atom_types = assign_atom_types_with_arom(mol, &aromatic);

    // 4. Merge non-polar hydrogens
    let merged = merge_hydrogens(mol, &atom_types, &mut charges);

    // 5. Rotatable bonds
    let rotatable = find_rotatable_bonds(mol, &merged, &aromatic, symmetry_classes);

    // 6. Torsion tree
    let tree = build_torsion_tree(mol, &rotatable, &merged);

    // 7. Write PDBQT
    let mut buf = Vec::with_capacity(4096);
    write_pdbqt(&mut buf, mol, &atom_types, &charges, &merged, &tree,
                smiles, smiles_idx, h_parent, smiles_atom_order)?;

    Ok(String::from_utf8_lossy(&buf).to_string())
}

/// Write a molecule as PDBQT to a file.
pub fn write_pdbqt_file<P: AsRef<std::path::Path>>(path: P, mol: &Molecule) -> Result<()> {
    let pdbqt = mol_to_pdbqt(mol)?;
    std::fs::write(path, pdbqt).map_err(SdfError::Io)
}

/// Write a molecule as PDBQT to a file with optional REMARK lines.
pub fn write_pdbqt_file_with_remarks<P: AsRef<std::path::Path>>(
    path: P,
    mol: &Molecule,
    smiles: Option<&str>,
    smiles_idx: Option<&[(usize, usize)]>,
    h_parent: Option<&[(usize, usize)]>,
) -> Result<()> {
    let pdbqt = mol_to_pdbqt_with_remarks(mol, smiles, smiles_idx, h_parent)?;
    std::fs::write(path, pdbqt).map_err(SdfError::Io)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::Atom;
    use crate::bond::Bond;

    /// Build a simple ethanol molecule: C-C-O with explicit H.
    fn make_ethanol() -> Molecule {
        let mut mol = Molecule::new("ethanol");
        // Heavy atoms
        mol.atoms.push(Atom::new(0, "C", 0.0, 0.0, 0.0));     // C1 (methyl)
        mol.atoms.push(Atom::new(1, "C", 1.54, 0.0, 0.0));    // C2
        mol.atoms.push(Atom::new(2, "O", 2.40, 0.93, 0.0));   // O
        // H on C1
        mol.atoms.push(Atom::new(3, "H", -0.36, 1.02, 0.0));
        mol.atoms.push(Atom::new(4, "H", -0.36, -0.51, 0.88));
        mol.atoms.push(Atom::new(5, "H", -0.36, -0.51, -0.88));
        // H on C2
        mol.atoms.push(Atom::new(6, "H", 1.90, -0.51, 0.88));
        mol.atoms.push(Atom::new(7, "H", 1.90, -0.51, -0.88));
        // H on O (polar)
        mol.atoms.push(Atom::new(8, "H", 3.36, 0.72, 0.0));

        // Bonds
        mol.bonds.push(Bond::new(0, 1, BondOrder::Single)); // C-C
        mol.bonds.push(Bond::new(1, 2, BondOrder::Single)); // C-O
        mol.bonds.push(Bond::new(0, 3, BondOrder::Single)); // C-H
        mol.bonds.push(Bond::new(0, 4, BondOrder::Single)); // C-H
        mol.bonds.push(Bond::new(0, 5, BondOrder::Single)); // C-H
        mol.bonds.push(Bond::new(1, 6, BondOrder::Single)); // C-H
        mol.bonds.push(Bond::new(1, 7, BondOrder::Single)); // C-H
        mol.bonds.push(Bond::new(2, 8, BondOrder::Single)); // O-H

        mol
    }

    /// Build benzene with explicit H.
    fn make_benzene_with_h() -> Molecule {
        let mut mol = Molecule::new("benzene");
        let coords = [
            (1.21, 0.70, 0.0), (1.21, -0.70, 0.0), (0.0, -1.40, 0.0),
            (-1.21, -0.70, 0.0), (-1.21, 0.70, 0.0), (0.0, 1.40, 0.0),
        ];
        // Aromatic carbons
        for (i, (x, y, z)) in coords.iter().enumerate() {
            mol.atoms.push(Atom::new(i, "C", *x, *y, *z));
        }
        // H atoms
        let h_scale = 1.08 / 1.40;
        for i in 0..6 {
            let (cx, cy, _) = coords[i];
            let hx = cx * (1.0 + h_scale);
            let hy = cy * (1.0 + h_scale);
            mol.atoms.push(Atom::new(6 + i, "H", hx, hy, 0.0));
        }
        // Aromatic C-C bonds
        for i in 0..6 {
            mol.bonds.push(Bond::new(i, (i + 1) % 6, BondOrder::Aromatic));
        }
        // C-H bonds
        for i in 0..6 {
            mol.bonds.push(Bond::new(i, 6 + i, BondOrder::Single));
        }
        mol
    }

    #[test]
    fn test_atom_types_ethanol() {
        let mol = make_ethanol();
        let types = assign_atom_types(&mol);
        assert_eq!(types[0], "C");   // methyl C
        assert_eq!(types[1], "C");   // C
        assert_eq!(types[2], "OA");  // oxygen
        assert_eq!(types[3], "H");   // non-polar H on C
        assert_eq!(types[8], "HD");  // polar H on O
    }

    #[test]
    fn test_atom_types_benzene() {
        let mol = make_benzene_with_h();
        let types = assign_atom_types(&mol);
        for i in 0..6 {
            assert_eq!(types[i], "A", "Carbon {} should be aromatic (A)", i);
        }
        for i in 6..12 {
            assert_eq!(types[i], "H", "H {} on aromatic C should be H (non-polar)", i);
        }
    }

    #[test]
    fn test_merge_hydrogens() {
        let mol = make_ethanol();
        let types = assign_atom_types(&mol);
        let mut charges = gasteiger_charges(&mol);
        let merged = merge_hydrogens(&mol, &types, &mut charges);

        // Non-polar H (3,4,5,6,7) should be merged
        assert!(merged[3]);
        assert!(merged[4]);
        assert!(merged[5]);
        assert!(merged[6]);
        assert!(merged[7]);
        // Polar H (8, on O) should NOT be merged
        assert!(!merged[8]);
        // Heavy atoms not merged
        assert!(!merged[0]);
        assert!(!merged[1]);
        assert!(!merged[2]);
    }

    #[test]
    fn test_ethanol_rotatable_bonds() {
        let mol = make_ethanol();
        let aromatic = all_aromatic_atoms(&mol);
        let types = assign_atom_types(&mol);
        let mut charges = gasteiger_charges(&mol);
        let merged = merge_hydrogens(&mol, &types, &mut charges);
        let rotatable = find_rotatable_bonds(&mol, &merged, &aromatic);

        // C-C bond (index 0): C1 has non-merged degree 1 (only C2, H merged)
        //   → C1 is terminal → NOT rotatable
        assert!(!rotatable[0], "C-C should not be rotatable (C1 is terminal)");

        // C-O bond (index 1): C2 has non-merged degree 2 (C1+O),
        //   O has non-merged degree 2 (C2+HD). Both > 1 → rotatable.
        //   This matches Meeko's behavior (O-HD counts as a real neighbor).
        assert!(rotatable[1], "C-O should be rotatable (O has HD neighbor)");
    }

    #[test]
    fn test_mol_to_pdbqt_ethanol() {
        let mol = make_ethanol();
        let pdbqt = mol_to_pdbqt(&mol).expect("Failed to convert ethanol");

        assert!(pdbqt.contains("ROOT"), "PDBQT must have ROOT");
        assert!(pdbqt.contains("ENDROOT"), "PDBQT must have ENDROOT");
        assert!(pdbqt.contains("TORSDOF"), "PDBQT must have TORSDOF");
        assert!(pdbqt.contains("OA"), "PDBQT must contain OA atom type");
        assert!(pdbqt.contains("HD"), "PDBQT must contain HD atom type");
        // C-O is rotatable → 1 torsion with BRANCH
        assert!(pdbqt.contains("BRANCH"), "Ethanol should have a BRANCH (C-O rotatable)");
        assert!(pdbqt.contains("TORSDOF 1"), "Ethanol should have TORSDOF 1");
        // Non-polar H should NOT appear
        let atom_lines: Vec<&str> = pdbqt.lines()
            .filter(|l| l.starts_with("ATOM"))
            .collect();
        // 3 heavy atoms + 1 HD = 4 ATOM lines
        assert_eq!(atom_lines.len(), 4, "Ethanol should have 4 ATOM lines (3 heavy + 1 HD)");
    }

    #[test]
    fn test_mol_to_pdbqt_benzene() {
        let mol = make_benzene_with_h();
        let pdbqt = mol_to_pdbqt(&mol).expect("Failed to convert benzene");

        assert!(pdbqt.contains("ROOT"));
        assert!(pdbqt.contains("ENDROOT"));
        // Benzene has no rotatable bonds → 0 torsions
        assert!(pdbqt.contains("TORSDOF 0"));
        // No BRANCH (single rigid body)
        assert!(!pdbqt.contains("BRANCH"));
        // 6 aromatic C atoms, no HD
        let atom_lines: Vec<&str> = pdbqt.lines()
            .filter(|l| l.starts_with("ATOM"))
            .collect();
        assert_eq!(atom_lines.len(), 6, "Benzene should have 6 ATOM lines (aromatic C only)");
        // All should have type "A"
        for line in &atom_lines {
            assert!(line.ends_with(" A") || line.ends_with(" A "), "Atom line should have type A: {}", line);
        }
    }
}
