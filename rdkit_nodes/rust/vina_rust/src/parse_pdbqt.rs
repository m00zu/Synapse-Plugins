use crate::atom::*;
use crate::common::*;
use crate::model::*;
use crate::tree::*;
use std::fs;

// ─── Parsed Atom ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ParsedAtom {
    pub number: usize,
    pub name: String,
    pub residue_name: String,
    pub coords: Vec3,
    pub charge: f64,
    pub ad_type_name: String,
    pub ad: usize,
    pub el: usize,
    pub xs: usize,
}

fn parse_pdbqt_atom_line(line: &str) -> Option<ParsedAtom> {
    if line.len() < 78 { return None; }

    let record = line[0..6].trim();
    if record != "ATOM" && record != "HETATM" { return None; }

    let number = line[6..11].trim().parse::<usize>().unwrap_or(0);
    let name = line[12..16].trim().to_string();
    let residue_name = line[17..20].trim().to_string();
    let x = line[30..38].trim().parse::<f64>().unwrap_or(0.0);
    let y = line[38..46].trim().parse::<f64>().unwrap_or(0.0);
    let z = line[46..54].trim().parse::<f64>().unwrap_or(0.0);

    let charge = if line.len() >= 76 {
        line[68..76].trim().parse::<f64>().unwrap_or(0.0)
    } else {
        0.0
    };

    let ad_type_name = if line.len() > 77 {
        line[77..line.len().min(79)].trim().to_string()
    } else {
        String::new()
    };

    let ad = string_to_ad_type(&ad_type_name);
    let el = if ad < AD_TYPE_SIZE { ad_type_to_el_type(ad) } else { EL_TYPE_SIZE };

    // Determine XS type from AD type, then override for non-AD metals (Cu, Na, K, etc.)
    // C++: if(is_non_ad_metal_name(name)) tmp.xs = XS_TYPE_Met_D;
    let xs = if is_non_ad_metal_name(&ad_type_name) {
        XS_TYPE_MET_D
    } else {
        ad_to_xs_type(ad, &ad_type_name)
    };

    Some(ParsedAtom {
        number,
        name,
        residue_name,
        coords: Vec3::new(x, y, z),
        charge,
        ad_type_name,
        ad,
        el,
        xs,
    })
}

/// Map AD type to XS type.
/// Initial assignment only — types are refined later by refine_xs_types() / refine_xs_types_from_bonds()
/// based on bonding neighbors (HD for donor status, heteroatom for carbon hydrophobicity).
/// Both Vina and smina do this same refinement (smina via model::assign_types → adjust_smina_type).
fn ad_to_xs_type(ad: usize, name: &str) -> usize {
    // Boron is not in the AD4 type system but is hydrophobic (xs_radius=1.92)
    if name == "B" { return XS_TYPE_B_H; }
    match ad {
        AD_TYPE_C | AD_TYPE_A => XS_TYPE_C_H, // Refined later: C_P if bonded to heteroatom
        AD_TYPE_N => XS_TYPE_N_P,
        AD_TYPE_O => XS_TYPE_O_P,
        AD_TYPE_P => XS_TYPE_P_P,
        AD_TYPE_S => XS_TYPE_S_P,
        AD_TYPE_H => XS_TYPE_SIZE, // skip non-polar H
        AD_TYPE_F => XS_TYPE_F_H,
        AD_TYPE_I => XS_TYPE_I_H,
        AD_TYPE_NA => XS_TYPE_N_A, // Refined later: N_DA if bonded to HD
        AD_TYPE_OA => XS_TYPE_O_A, // Refined later: O_DA if bonded to HD
        AD_TYPE_SA => XS_TYPE_S_P,
        AD_TYPE_HD => XS_TYPE_SIZE, // skip polar H (donor)
        AD_TYPE_CL => XS_TYPE_CL_H,
        AD_TYPE_BR => XS_TYPE_BR_H,
        AD_TYPE_SI => XS_TYPE_SI,
        AD_TYPE_AT => XS_TYPE_AT,
        AD_TYPE_MG | AD_TYPE_MN | AD_TYPE_ZN | AD_TYPE_CA | AD_TYPE_FE => XS_TYPE_MET_D,
        AD_TYPE_G0 => XS_TYPE_G0,
        AD_TYPE_G1 => XS_TYPE_G1,
        AD_TYPE_G2 => XS_TYPE_G2,
        AD_TYPE_G3 => XS_TYPE_G3,
        AD_TYPE_CG0 => XS_TYPE_C_H_CG0,
        AD_TYPE_CG1 => XS_TYPE_C_H_CG1,
        AD_TYPE_CG2 => XS_TYPE_C_H_CG2,
        AD_TYPE_CG3 => XS_TYPE_C_H_CG3,
        AD_TYPE_W => XS_TYPE_W,
        _ => XS_TYPE_SIZE,
    }
}

// ─── Parsing Tree Structure ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ParsingNode {
    atom: ParsedAtom,
    children: Vec<ParsingBranch>,
    line: String,
}

#[derive(Debug, Clone)]
struct ParsingBranch {
    from_atom: usize, // serial number of branch root in parent
    to_atom: usize,   // serial number of branch root in this branch
    nodes: Vec<ParsingNode>,
    children: Vec<ParsingBranch>,
}

/// Parse a PDBQT string into atoms + tree structure
fn parse_pdbqt_tree(content: &str) -> (Vec<ParsingNode>, Vec<ParsingBranch>) {
    let mut root_atoms = Vec::new();
    let mut root_branches = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut idx = 0;

    while idx < lines.len() {
        let line = lines[idx].trim();

        if line.starts_with("ROOT") {
            idx += 1;
            // Parse root atoms
            while idx < lines.len() {
                let l = lines[idx].trim();
                if l.starts_with("ENDROOT") { idx += 1; break; }
                if let Some(pa) = parse_pdbqt_atom_line(l) {
                    root_atoms.push(ParsingNode {
                        atom: pa,
                        children: Vec::new(),
                        line: l.to_string(),
                    });
                }
                idx += 1;
            }
        } else if line.starts_with("BRANCH") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let from = parts[1].parse::<usize>().unwrap_or(0);
                let to = parts[2].parse::<usize>().unwrap_or(0);
                idx += 1;
                let branch = parse_branch(&lines, &mut idx, from, to);
                root_branches.push(branch);
            } else {
                idx += 1;
            }
        } else if line.starts_with("ATOM") || line.starts_with("HETATM") {
            if let Some(pa) = parse_pdbqt_atom_line(line) {
                root_atoms.push(ParsingNode {
                    atom: pa,
                    children: Vec::new(),
                    line: line.to_string(),
                });
            }
            idx += 1;
        } else {
            idx += 1;
        }
    }

    (root_atoms, root_branches)
}

fn parse_branch(lines: &[&str], idx: &mut usize, from: usize, to: usize) -> ParsingBranch {
    let mut branch = ParsingBranch {
        from_atom: from,
        to_atom: to,
        nodes: Vec::new(),
        children: Vec::new(),
    };

    while *idx < lines.len() {
        let line = lines[*idx].trim();

        if line.starts_with("ENDBRANCH") {
            *idx += 1;
            break;
        } else if line.starts_with("BRANCH") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let from = parts[1].parse::<usize>().unwrap_or(0);
                let to = parts[2].parse::<usize>().unwrap_or(0);
                *idx += 1;
                let child = parse_branch(lines, idx, from, to);
                branch.children.push(child);
            } else {
                *idx += 1;
            }
        } else if line.starts_with("ATOM") || line.starts_with("HETATM") {
            if let Some(pa) = parse_pdbqt_atom_line(line) {
                branch.nodes.push(ParsingNode {
                    atom: pa,
                    children: Vec::new(),
                    line: line.to_string(),
                });
            }
            *idx += 1;
        } else {
            *idx += 1;
        }
    }

    branch
}

// ─── Build Model From Parsed Data ──────────────────────────────────────────────

/// Build Atom struct from ParsedAtom
fn make_atom(pa: &ParsedAtom) -> Atom {
    let mut a = Atom::new();
    a.coords = pa.coords;
    a.charge = pa.charge;
    a.ad = pa.ad;
    a.el = pa.el;
    a.xs = pa.xs;
    a
}

/// Refine XS types using distance-based neighbor detection.
/// Used for receptor atoms (all in single rigid group) and initial ligand refinement.
/// Both Vina and smina do this refinement (smina via model::assign_types → adjust_smina_type).
fn refine_xs_types(atoms: &mut [Atom]) {
    let n = atoms.len();

    for i in 0..n {
        let has_hetero_neighbor = {
            let mut found = false;
            for j in 0..n {
                if i == j { continue; }
                let dist = atoms[i].coords.distance_sqr(&atoms[j].coords).sqrt();
                let bond_threshold = 1.1 * (atoms[i].covalent_radius() + atoms[j].covalent_radius());
                if dist < bond_threshold && atoms[j].is_heteroatom() {
                    found = true;
                    break;
                }
            }
            found
        };

        if atoms[i].ad == AD_TYPE_C || atoms[i].ad == AD_TYPE_A {
            atoms[i].xs = if has_hetero_neighbor { XS_TYPE_C_P } else { XS_TYPE_C_H };
        }

        // C++: donor_NorO = (a.el == EL_TYPE_Met || bonded_to_HD(a))
        // Metals count as donors even without HD neighbors.
        let is_metal = atoms[i].el == EL_TYPE_MET;
        let has_donor_h = {
            let mut found = false;
            for j in 0..n {
                if i == j { continue; }
                if atoms[j].ad != AD_TYPE_HD { continue; }
                let dist = atoms[i].coords.distance_sqr(&atoms[j].coords).sqrt();
                let bond_threshold = 1.1 * (atoms[i].covalent_radius() + atoms[j].covalent_radius());
                if dist < bond_threshold {
                    found = true;
                    break;
                }
            }
            found
        };
        let donor = is_metal || has_donor_h;

        match atoms[i].ad {
            AD_TYPE_NA => atoms[i].xs = if donor { XS_TYPE_N_DA } else { XS_TYPE_N_A },
            AD_TYPE_N => atoms[i].xs = if donor { XS_TYPE_N_D } else { XS_TYPE_N_P },
            _ => {}
        }
        match atoms[i].ad {
            AD_TYPE_OA => atoms[i].xs = if donor { XS_TYPE_O_DA } else { XS_TYPE_O_A },
            AD_TYPE_O => atoms[i].xs = if donor { XS_TYPE_O_D } else { XS_TYPE_O_P },
            _ => {}
        }
    }
}

/// Refine XS types using the bond list (for ligand atoms after bond detection).
/// C++ Vina's assign_types() traverses the bond graph, not raw distances.
/// Both Vina and smina do this refinement (smina via model::assign_types → adjust_smina_type).
fn refine_xs_types_from_bonds(atoms: &mut [Atom], begin: usize, end: usize) {
    // Collect refinement info first (to avoid borrow issues)
    // C++: donor_NorO = (a.el == EL_TYPE_Met || bonded_to_HD(a))
    let mut refinements: Vec<(bool, bool, bool)> = Vec::new();
    for i in begin..end {
        let has_hetero = atoms[i].bonds.iter().any(|b| {
            let j = b.connected_atom_index.i;
            !b.connected_atom_index.in_grid && j < atoms.len() && atoms[j].is_heteroatom()
        });
        let has_donor_h = atoms[i].bonds.iter().any(|b| {
            let j = b.connected_atom_index.i;
            !b.connected_atom_index.in_grid && j < atoms.len() && atoms[j].ad == AD_TYPE_HD
        });
        let is_metal = atoms[i].el == EL_TYPE_MET;
        refinements.push((has_hetero, has_donor_h, is_metal));
    }

    for (idx, &(has_hetero, has_donor_h, is_metal)) in refinements.iter().enumerate() {
        let i = begin + idx;
        let donor = is_metal || has_donor_h;
        if atoms[i].ad == AD_TYPE_C || atoms[i].ad == AD_TYPE_A {
            atoms[i].xs = if has_hetero { XS_TYPE_C_P } else { XS_TYPE_C_H };
        }
        match atoms[i].ad {
            AD_TYPE_NA => atoms[i].xs = if donor { XS_TYPE_N_DA } else { XS_TYPE_N_A },
            AD_TYPE_N => atoms[i].xs = if donor { XS_TYPE_N_D } else { XS_TYPE_N_P },
            _ => {}
        }
        match atoms[i].ad {
            AD_TYPE_OA => atoms[i].xs = if donor { XS_TYPE_O_DA } else { XS_TYPE_O_A },
            AD_TYPE_O => atoms[i].xs = if donor { XS_TYPE_O_D } else { XS_TYPE_O_P },
            _ => {}
        }
    }
}

/// Detect bonds between atoms within a single range
fn detect_bonds(atoms: &mut [Atom], begin: usize, end: usize) {
    for i in begin..end {
        for j in (i + 1)..end {
            let dist = atoms[i].coords.distance_sqr(&atoms[j].coords).sqrt();
            let threshold = 1.1 * (atoms[i].covalent_radius() + atoms[j].covalent_radius());
            if dist < threshold {
                atoms[i].bonds.push(Bond {
                    connected_atom_index: AtomIndex { i: j, in_grid: false },
                    length: dist,
                    rotatable: false,
                });
                atoms[j].bonds.push(Bond {
                    connected_atom_index: AtomIndex { i, in_grid: false },
                    length: dist,
                    rotatable: false,
                });
            }
        }
    }
}

/// Detect bonds within each rigid segment of the tree (NOT across segments).
/// C++ Vina only detects bonds between atoms where mobility != DISTANCE_VARIABLE,
/// i.e., atoms in the same segment (DISTANCE_FIXED) or hinge pairs (DISTANCE_ROTOR).
fn detect_bonds_within_segments(atoms: &mut [Atom], tree: &RigidBody) {
    // Detect bonds within root segment
    detect_bonds(atoms, tree.atom_range.begin, tree.atom_range.end);
    // Recursively detect bonds within each child segment
    fn detect_in_subtree(atoms: &mut [Atom], seg: &Segment) {
        detect_bonds(atoms, seg.atom_range.begin, seg.atom_range.end);
        for child in &seg.children {
            detect_in_subtree(atoms, child);
        }
    }
    for child in &tree.children {
        detect_in_subtree(atoms, child);
    }
}

/// Add hinge bonds between parent and child segments (BRANCH from_atom → to_atom).
/// These are the DISTANCE_ROTOR pairs in C++ Vina.
fn add_hinge_bonds(
    atoms: &mut [Atom],
    branches: &[ParsingBranch],
    serial_to_index: &std::collections::HashMap<usize, usize>,
) {
    for branch in branches {
        if let (Some(&from_idx), Some(&to_idx)) = (
            serial_to_index.get(&branch.from_atom),
            serial_to_index.get(&branch.to_atom),
        ) {
            let dist = atoms[from_idx].coords.distance_sqr(&atoms[to_idx].coords).sqrt();
            atoms[from_idx].bonds.push(Bond {
                connected_atom_index: AtomIndex { i: to_idx, in_grid: false },
                length: dist,
                rotatable: true,
            });
            atoms[to_idx].bonds.push(Bond {
                connected_atom_index: AtomIndex { i: from_idx, in_grid: false },
                length: dist,
                rotatable: true,
            });
        }
        add_hinge_bonds(atoms, &branch.children, serial_to_index);
    }
}

/// Collect atoms reachable within `depth` bond hops from atom `start`.
/// Used to exclude 1-2, 1-3, and 1-4 bonded pairs from intramolecular scoring.
fn bonded_to(atoms: &[Atom], start: usize, depth: usize) -> Vec<usize> {
    let mut out = Vec::new();
    fn recurse(atoms: &[Atom], a: usize, n: usize, out: &mut Vec<usize>) {
        if out.contains(&a) { return; }
        out.push(a);
        if n > 0 {
            for bond in &atoms[a].bonds {
                recurse(atoms, bond.connected_atom_index.i, n - 1, out);
            }
        }
    }
    recurse(atoms, start, depth, &mut out);
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MobilityType {
    Fixed,
    Rotor,
    Variable,
}

struct MobilityMatrix {
    n: usize,
    data: Vec<MobilityType>,
}

impl MobilityMatrix {
    fn new(n: usize, default: MobilityType) -> Self {
        MobilityMatrix { n, data: vec![default; n * n] }
    }

    #[inline(always)]
    fn idx(&self, i: usize, j: usize) -> usize { i * self.n + j }

    #[inline(always)]
    fn set(&mut self, i: usize, j: usize, v: MobilityType) {
        let ij = self.idx(i, j);
        let ji = self.idx(j, i);
        self.data[ij] = v;
        self.data[ji] = v;
    }

    #[inline(always)]
    fn get(&self, i: usize, j: usize) -> MobilityType {
        self.data[self.idx(i, j)]
    }
}

#[derive(Clone, Copy)]
struct BranchInfo {
    from_serial: usize,
    to_serial: usize,
    begin: usize,
    end: usize,
}

fn build_ligand_mobility_matrix(
    atom_offset: usize,
    root_begin: usize,
    root_end: usize,
    total_end: usize,
    serial_to_index: &std::collections::HashMap<usize, usize>,
    branch_infos: &[BranchInfo],
) -> MobilityMatrix {
    let n = total_end - atom_offset;
    let mut m = MobilityMatrix::new(n, MobilityType::Variable);

    // Equivalent to C++ postprocess_branch: atoms in the same rigid node are fixed.
    for i in root_begin..root_end {
        for j in (i + 1)..root_end {
            m.set(i - atom_offset, j - atom_offset, MobilityType::Fixed);
        }
    }

    for info in branch_infos {
        for i in info.begin..info.end {
            for j in (i + 1)..info.end {
                m.set(i - atom_offset, j - atom_offset, MobilityType::Fixed);
            }
        }

        let from_idx = serial_to_index.get(&info.from_serial).copied();
        let to_idx = serial_to_index.get(&info.to_serial).copied();
        if let (Some(from_idx), Some(to_idx)) = (from_idx, to_idx) {
            // Equivalent to C++ set_rotor(axis_begin, axis_end)
            m.set(from_idx - atom_offset, to_idx - atom_offset, MobilityType::Rotor);

            // Equivalent to C++ add_bonds(axis_begin/axis_end, current rigid node)
            for i in info.begin..info.end {
                m.set(from_idx - atom_offset, i - atom_offset, MobilityType::Fixed);
                m.set(to_idx - atom_offset, i - atom_offset, MobilityType::Fixed);
            }
        }
    }

    m
}

/// Build interacting pairs for a ligand, following C++ Vina's logic:
/// - Exclude pairs in the same rigid segment (DISTANCE_FIXED)
/// - Exclude pairs within 3 bond hops (1-2, 1-3, 1-4)
/// - Exclude pairs with invalid atom types
fn build_interacting_pairs(
    atoms: &[Atom],
    begin: usize,
    end: usize,
    typing: AtomTyping,
    mobility: &MobilityMatrix,
) -> Vec<InteractingPair> {
    let mut pairs = Vec::new();
    let n_types = num_atom_types(typing);

    for i in begin..end {
        // Get atoms within 3 bond hops (excludes 1-2, 1-3, 1-4)
        let bonded = bonded_to(atoms, i, 3);

        for j in (i + 1)..end {
            // Match C++ initialize_pairs: only DISTANCE_VARIABLE pairs are considered.
            if mobility.get(i - begin, j - begin) != MobilityType::Variable { continue; }

            // Skip atoms within 3 bond hops
            if bonded.contains(&j) { continue; }

            let t1 = atoms[i].get_type(typing);
            let t2 = atoms[j].get_type(typing);
            if t1 >= n_types || t2 >= n_types { continue; }

            let (a, b) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            let type_pair_index = a + b * (b + 1) / 2;

            pairs.push(InteractingPair::new(type_pair_index, i, j));
        }
    }
    pairs
}

// ─── Public API ────────────────────────────────────────────────────────────────

/// Parse a receptor PDBQT file.
pub fn parse_receptor(rigid_path: &str, flex_path: Option<&str>) -> Result<Model, String> {
    let content = fs::read_to_string(rigid_path)
        .map_err(|e| format!("Cannot read receptor file: {}", e))?;

    let mut model = Model::new();

    // Parse receptor atoms
    for line in content.lines() {
        let l = line.trim();
        if l.starts_with("ATOM") || l.starts_with("HETATM") {
            if let Some(pa) = parse_pdbqt_atom_line(l) {
                let a = make_atom(&pa);
                model.grid_atoms.push(a);
            }
        }
    }

    // Refine receptor XS types (C_H→C_P, N→N_D, O→O_D, NA→N_DA, OA→O_DA)
    refine_xs_types(&mut model.grid_atoms);

    // Parse flexible residues if provided
    if let Some(flex) = flex_path {
        let flex_content = fs::read_to_string(flex)
            .map_err(|e| format!("Cannot read flex file: {}", e))?;
        // TODO: parse flexible residues into model.flex
        let _ = flex_content;
    }

    Ok(model)
}

/// Parse a ligand PDBQT file and add to model.
pub fn parse_ligand_file(model: &mut Model, ligand_path: &str) -> Result<(), String> {
    let content = fs::read_to_string(ligand_path)
        .map_err(|e| format!("Cannot read ligand file: {}", e))?;
    parse_ligand_string(model, &content)
}

/// Collect all parsed atoms into a serial-number → coords mapping
fn collect_atom_coords(
    root_nodes: &[ParsingNode],
    branches: &[ParsingBranch],
) -> std::collections::HashMap<usize, Vec3> {
    let mut map = std::collections::HashMap::new();
    for node in root_nodes {
        map.insert(node.atom.number, node.atom.coords);
    }
    fn collect_branch(branch: &ParsingBranch, map: &mut std::collections::HashMap<usize, Vec3>) {
        for node in &branch.nodes {
            map.insert(node.atom.number, node.atom.coords);
        }
        for child in &branch.children {
            collect_branch(child, map);
        }
    }
    for b in branches {
        collect_branch(b, &mut map);
    }
    map
}

/// Count total torsions (= number of branches recursively)
fn count_torsions(branches: &[ParsingBranch]) -> usize {
    let mut count = branches.len();
    for b in branches {
        count += count_torsions(&b.children);
    }
    count
}

/// Parse a ligand PDBQT string and add to model.
pub fn parse_ligand_string(model: &mut Model, content: &str) -> Result<(), String> {
    let (root_nodes, branches) = parse_pdbqt_tree(content);

    if root_nodes.is_empty() {
        return Err("No atoms found in ligand PDBQT".to_string());
    }

    // Build serial-number → coords map for axis/origin lookups
    let atom_coords_map = collect_atom_coords(&root_nodes, &branches);

    let atom_offset = model.atoms.len();

    // Root rigid body origin = first root atom's coordinates (matches C++ Vina)
    let root_origin = root_nodes[0].atom.coords;

    // Build serial_number → model_index mapping as we add atoms
    let mut serial_to_index = std::collections::HashMap::new();

    // Add root atoms with RELATIVE internal coords (subtract root origin)
    for node in &root_nodes {
        let idx = model.atoms.len();
        serial_to_index.insert(node.atom.number, idx);
        let a = make_atom(&node.atom);
        model.atoms.push(a);
        model.internal_coords.push(node.atom.coords - root_origin);
        model.coords.push(node.atom.coords);
        model.minus_forces.push(Vec3::ZERO);
    }

    let root_end = model.atoms.len();
    let dof = count_torsions(&branches);

    // Add branch atoms recursively with proper axes and origins
    fn add_branch_atoms(
        model: &mut Model,
        branch: &ParsingBranch,
        parent_origin: &Vec3,
        atom_coords_map: &std::collections::HashMap<usize, Vec3>,
        serial_to_index: &mut std::collections::HashMap<usize, usize>,
        branch_infos: &mut Vec<BranchInfo>,
        parent_segments: &mut Vec<Segment>,
    ) {
        // Look up from/to atom coordinates for axis computation
        let from_coords = atom_coords_map.get(&branch.from_atom)
            .copied().unwrap_or(Vec3::ZERO);
        let to_coords = atom_coords_map.get(&branch.to_atom)
            .copied().unwrap_or(Vec3::ZERO);

        // Segment origin = to_atom's coordinates (the pivot point)
        let segment_origin = to_coords;

        // Relative origin = segment_origin - parent_origin
        // (At parse time, parent has identity orientation, so no rotation needed)
        let relative_origin = segment_origin - *parent_origin;

        // Rotation axis = normalize(to_coords - from_coords)
        let axis_vec = to_coords - from_coords;
        let axis_len = axis_vec.norm();
        let relative_axis = if axis_len > EPSILON_FL {
            axis_vec * (1.0 / axis_len)
        } else {
            Vec3::new(1.0, 0.0, 0.0) // fallback for degenerate case
        };

        // Add branch atoms with RELATIVE internal coords (subtract segment origin)
        let begin = model.atoms.len();
        for node in &branch.nodes {
            let idx = model.atoms.len();
            serial_to_index.insert(node.atom.number, idx);
            let a = make_atom(&node.atom);
            model.atoms.push(a);
            model.internal_coords.push(node.atom.coords - segment_origin);
            model.coords.push(node.atom.coords);
            model.minus_forces.push(Vec3::ZERO);
        }
        let end = model.atoms.len();
        branch_infos.push(BranchInfo {
            from_serial: branch.from_atom,
            to_serial: branch.to_atom,
            begin,
            end,
        });

        let atom_range = AtomRange::new(begin, end);
        let mut segment = Segment::new(relative_axis, relative_origin, atom_range);

        for child in &branch.children {
            add_branch_atoms(model, child, &segment_origin, atom_coords_map, serial_to_index, branch_infos, &mut segment.children);
        }

        parent_segments.push(segment);
    }

    let mut segments = Vec::new();
    let mut branch_infos = Vec::new();
    for branch in &branches {
        add_branch_atoms(
            model, branch, &root_origin, &atom_coords_map,
            &mut serial_to_index, &mut branch_infos, &mut segments
        );
    }

    let total_end = model.atoms.len();
    model.num_movable_atoms = total_end;

    // Build rigid body tree (needed before bond detection to define segments)
    let atom_range = AtomRange::new(atom_offset, root_end);
    let mut tree = RigidBody::new(atom_range);
    tree.children = segments;

    // Detect bonds ONLY within each rigid segment (matching C++ Vina's assign_bonds logic)
    // C++ Vina only detects bonds between atoms where mobility != DISTANCE_VARIABLE
    detect_bonds_within_segments(&mut model.atoms, &tree);

    // Add hinge bonds from BRANCH definitions (DISTANCE_ROTOR pairs)
    add_hinge_bonds(&mut model.atoms, &branches, &serial_to_index);

    // Refine XS types using bond list (C++ Vina: assign_bonds → assign_types)
    refine_xs_types_from_bonds(&mut model.atoms, atom_offset, total_end);

    // Build mobility matrix and interacting pairs (C++-style DISTANCE_VARIABLE logic).
    let mobility = build_ligand_mobility_matrix(
        atom_offset, atom_offset, root_end, total_end, &serial_to_index, &branch_infos
    );
    let pairs = build_interacting_pairs(&model.atoms, atom_offset, total_end, model.atom_typing, &mobility);

    // Create ligand
    let mut ligand = Ligand::new(tree, dof);
    ligand.begin = atom_offset;
    ligand.end = total_end;
    ligand.pairs = pairs;

    // Store context lines for output
    for line in content.lines() {
        ligand.cont.push(line.to_string());
    }

    model.ligands.push(ligand);

    Ok(())
}

/// Write poses to PDBQT format
pub fn write_poses_pdbqt(
    model: &mut Model,
    poses: &[crate::conf::OutputType],
    how_many: usize,
    energy_range: f64,
) -> String {
    let mut output = String::new();

    if poses.is_empty() { return output; }

    let best_e = poses[0].e;

    for (idx, pose) in poses.iter().enumerate() {
        if idx >= how_many { break; }
        if pose.e - best_e > energy_range { break; }

        // Set model to this pose's conformation (all atoms, including H)
        model.set(&pose.c);

        output.push_str(&format!("MODEL {}\n", idx + 1));
        output.push_str(&format!(
            "REMARK VINA RESULT:    {:.1}      {:.3}      {:.3}\n",
            pose.e, 0.0, 0.0 // TODO: compute lb/ub
        ));

        // Write atom coordinates from model.coords (correct for all atoms incl. H)
        for lig in &model.ligands {
            let mut atom_idx = lig.begin;
            for line in &lig.cont {
                let l = line.trim();
                if l.starts_with("ATOM") || l.starts_with("HETATM") {
                    if atom_idx < model.coords.len() && l.len() >= 54 {
                        let coord = model.coords[atom_idx];
                        let new_line = format!(
                            "{}{:8.3}{:8.3}{:8.3}{}",
                            &l[..30],
                            coord[0], coord[1], coord[2],
                            if l.len() > 54 { &l[54..] } else { "" }
                        );
                        output.push_str(&new_line);
                        output.push('\n');
                    } else {
                        output.push_str(l);
                        output.push('\n');
                    }
                    atom_idx += 1;
                } else {
                    output.push_str(l);
                    output.push('\n');
                }
            }
        }

        output.push_str("ENDMDL\n");
    }

    output
}
