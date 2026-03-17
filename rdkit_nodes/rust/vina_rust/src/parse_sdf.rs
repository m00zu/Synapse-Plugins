//! SDF/MOL parser adapter for rxDock-style docking.
//!
//! Uses the `sdfrust` crate for SDF/MOL2 parsing, then builds a torsion tree
//! compatible with the existing Model/RigidBody/Segment infrastructure and
//! assigns Tripos (Sybyl) atom types for rxDock scoring.

use crate::atom::*;
use crate::common::*;
use crate::model::*;
use crate::rxdock_atom::*;
use crate::tree::*;

use sdfrust::{Molecule, BondOrder};
use sdfrust::descriptors::aromaticity::all_aromatic_atoms;

// ─── Element / Type Mapping ────────────────────────────────────────────────

fn element_to_el_type(s: &str) -> usize {
    match s {
        "H" | "D" => EL_TYPE_H,
        "C" => EL_TYPE_C,
        "N" => EL_TYPE_N,
        "O" => EL_TYPE_O,
        "S" => EL_TYPE_S,
        "P" => EL_TYPE_P,
        "F" => EL_TYPE_F,
        "Cl" => EL_TYPE_CL,
        "Br" => EL_TYPE_BR,
        "I" => EL_TYPE_I,
        "Si" => EL_TYPE_SI,
        "At" => EL_TYPE_AT,
        "Fe" | "Zn" | "Mg" | "Mn" | "Ca" | "Cu" | "Na" | "K" | "Co" | "Ni" => EL_TYPE_MET,
        _ => EL_TYPE_DUMMY,
    }
}

fn el_to_default_ad_type(el: usize) -> usize {
    match el {
        EL_TYPE_H => AD_TYPE_H,
        EL_TYPE_C => AD_TYPE_C,
        EL_TYPE_N => AD_TYPE_N,
        EL_TYPE_O => AD_TYPE_O,
        EL_TYPE_S => AD_TYPE_S,
        EL_TYPE_P => AD_TYPE_P,
        EL_TYPE_F => AD_TYPE_F,
        EL_TYPE_CL => AD_TYPE_CL,
        EL_TYPE_BR => AD_TYPE_BR,
        EL_TYPE_I => AD_TYPE_I,
        EL_TYPE_SI => AD_TYPE_SI,
        EL_TYPE_MET => AD_TYPE_ZN,
        _ => AD_TYPE_SIZE,
    }
}

fn ad_to_xs_initial(ad: usize, element: &str) -> usize {
    if element == "B" { return XS_TYPE_B_H; }
    match ad {
        AD_TYPE_C => XS_TYPE_C_H,
        AD_TYPE_N => XS_TYPE_N_P,
        AD_TYPE_O => XS_TYPE_O_P,
        AD_TYPE_P => XS_TYPE_P_P,
        AD_TYPE_S => XS_TYPE_S_P,
        AD_TYPE_H => XS_TYPE_SIZE,
        AD_TYPE_F => XS_TYPE_F_H,
        AD_TYPE_I => XS_TYPE_I_H,
        AD_TYPE_CL => XS_TYPE_CL_H,
        AD_TYPE_BR => XS_TYPE_BR_H,
        AD_TYPE_SI => XS_TYPE_SI,
        AD_TYPE_ZN | AD_TYPE_FE | AD_TYPE_MG | AD_TYPE_MN | AD_TYPE_CA => XS_TYPE_MET_D,
        _ => XS_TYPE_SIZE,
    }
}

fn is_heteroatom_el(el: usize) -> bool {
    matches!(el, EL_TYPE_N | EL_TYPE_O | EL_TYPE_S | EL_TYPE_P | EL_TYPE_F |
                 EL_TYPE_CL | EL_TYPE_BR | EL_TYPE_I)
}

fn bond_order_to_u8(order: &BondOrder) -> u8 {
    match order {
        BondOrder::Single => 1,
        BondOrder::Double => 2,
        BondOrder::Triple => 3,
        BondOrder::Aromatic => 4,
        _ => 1,
    }
}

// ─── Converted Atom (intermediate) ─────────────────────────────────────────

struct ConvertedAtom {
    coords: Vec3,
    el: usize,
    ad: usize,
    xs: usize,
    sy: usize,
    charge: f64,
}

/// Convert sdfrust Molecule atoms to our intermediate representation with all types assigned.
fn convert_atoms(mol: &Molecule) -> Vec<ConvertedAtom> {
    let n = mol.atom_count();
    let aromatic = all_aromatic_atoms(mol);

    // Build per-atom bond topology info
    let mut n_heavy_neighbors = vec![0usize; n];
    let mut max_bond_order = vec![0u8; n];
    let mut total_bond_order_sum = vec![0u8; n];
    let mut has_h_neighbor = vec![false; n];
    let mut bonded_to_heteroatom = vec![false; n];

    for bond in &mol.bonds {
        let a1 = bond.atom1;
        let a2 = bond.atom2;
        let order = bond_order_to_u8(&bond.order);
        let effective = if bond.is_aromatic() { 1 } else { order };

        let el1 = element_to_el_type(&mol.atoms[a1].element);
        let el2 = element_to_el_type(&mol.atoms[a2].element);

        if el1 != EL_TYPE_H { n_heavy_neighbors[a2] += 1; } else { has_h_neighbor[a2] = true; }
        if el2 != EL_TYPE_H { n_heavy_neighbors[a1] += 1; } else { has_h_neighbor[a1] = true; }

        if order > max_bond_order[a1] { max_bond_order[a1] = order; }
        if order > max_bond_order[a2] { max_bond_order[a2] = order; }
        total_bond_order_sum[a1] = total_bond_order_sum[a1].saturating_add(effective);
        total_bond_order_sum[a2] = total_bond_order_sum[a2].saturating_add(effective);

        if is_heteroatom_el(el1) { bonded_to_heteroatom[a2] = true; }
        if is_heteroatom_el(el2) { bonded_to_heteroatom[a1] = true; }
    }

    // Check which H atoms are bonded to heteroatoms (polar H)
    let mut h_is_polar = vec![false; n];
    for bond in &mol.bonds {
        let el1 = element_to_el_type(&mol.atoms[bond.atom1].element);
        let el2 = element_to_el_type(&mol.atoms[bond.atom2].element);
        if el1 == EL_TYPE_H && is_heteroatom_el(el2) { h_is_polar[bond.atom1] = true; }
        if el2 == EL_TYPE_H && is_heteroatom_el(el1) { h_is_polar[bond.atom2] = true; }
    }

    let mut atoms = Vec::with_capacity(n);
    for i in 0..n {
        let sa = &mol.atoms[i];
        let el = element_to_el_type(&sa.element);
        let mut ad = el_to_default_ad_type(el);
        let mut xs = ad_to_xs_initial(ad, &sa.element);

        // Assign Tripos type
        let sy = if el == EL_TYPE_H && h_is_polar[i] {
            ad = AD_TYPE_HD;
            SY_HP
        } else {
            assign_tripos_type(
                el, aromatic[i], n_heavy_neighbors[i],
                max_bond_order[i], has_h_neighbor[i], total_bond_order_sum[i],
            )
        };

        // Refine AD/XS types
        match el {
            EL_TYPE_C => {
                if aromatic[i] { ad = AD_TYPE_A; }
                if bonded_to_heteroatom[i] && xs == XS_TYPE_C_H { xs = XS_TYPE_C_P; }
            }
            EL_TYPE_N => {
                if max_bond_order[i] >= 2 || aromatic[i] {
                    ad = AD_TYPE_NA;
                    xs = if has_h_neighbor[i] { XS_TYPE_N_DA } else { XS_TYPE_N_A };
                } else if has_h_neighbor[i] {
                    xs = XS_TYPE_N_D;
                }
            }
            EL_TYPE_O => {
                ad = AD_TYPE_OA;
                if max_bond_order[i] >= 2 {
                    xs = XS_TYPE_O_A;
                } else if has_h_neighbor[i] {
                    xs = XS_TYPE_O_DA;
                } else {
                    xs = XS_TYPE_O_A;
                }
            }
            EL_TYPE_S => {
                if max_bond_order[i] >= 2 { ad = AD_TYPE_SA; }
            }
            _ => {}
        }

        atoms.push(ConvertedAtom {
            coords: Vec3::new(sa.x, sa.y, sa.z),
            el, ad, xs, sy,
            charge: sa.formal_charge as f64,
        });
    }

    atoms
}

// ─── Rotatable Bond Detection ───────────────────────────────────────────────

/// Determine which bonds in the molecule are rotatable for torsion tree building.
fn find_rotatable_bonds(mol: &Molecule, atoms: &[ConvertedAtom]) -> Vec<bool> {
    // Use sdfrust's SSSR ring detection to identify ring bonds
    let rings = sdfrust::descriptors::rings::sssr(mol);
    let mut in_ring = vec![false; mol.bond_count()];
    for ring in &rings {
        for &bi in &ring.bonds {
            in_ring[bi] = true;
        }
    }

    // Build adjacency for neighbor counting
    let adj = sdfrust::AdjacencyList::from_molecule(mol);

    let mut rotatable = vec![false; mol.bond_count()];
    for (bi, bond) in mol.bonds.iter().enumerate() {
        // Must be single bond
        if bond.order != BondOrder::Single { continue; }
        // Must not be in a ring
        if in_ring[bi] { continue; }
        // Neither atom is hydrogen
        if atoms[bond.atom1].el == EL_TYPE_H || atoms[bond.atom2].el == EL_TYPE_H { continue; }
        // Neither endpoint is terminal (only 1 heavy neighbor)
        let heavy1 = adj.neighbor_atoms(bond.atom1).iter()
            .filter(|&&n| atoms[n].el != EL_TYPE_H).count();
        let heavy2 = adj.neighbor_atoms(bond.atom2).iter()
            .filter(|&&n| atoms[n].el != EL_TYPE_H).count();
        if heavy1 <= 1 || heavy2 <= 1 { continue; }

        // Skip amide C-N bonds (partial double bond character)
        let is_amide = (atoms[bond.atom1].el == EL_TYPE_C && atoms[bond.atom2].el == EL_TYPE_N &&
            adj.neighbor_atoms(bond.atom1).iter().any(|&n| {
                atoms[n].el == EL_TYPE_O && mol.bonds.iter().any(|b|
                    ((b.atom1 == bond.atom1 && b.atom2 == n) || (b.atom2 == bond.atom1 && b.atom1 == n))
                    && bond_order_to_u8(&b.order) >= 2)
            })) ||
            (atoms[bond.atom2].el == EL_TYPE_C && atoms[bond.atom1].el == EL_TYPE_N &&
            adj.neighbor_atoms(bond.atom2).iter().any(|&n| {
                atoms[n].el == EL_TYPE_O && mol.bonds.iter().any(|b|
                    ((b.atom1 == bond.atom2 && b.atom2 == n) || (b.atom2 == bond.atom2 && b.atom1 == n))
                    && bond_order_to_u8(&b.order) >= 2)
            }));
        if is_amide { continue; }

        rotatable[bi] = true;
    }

    rotatable
}

// ─── Torsion Tree Building ──────────────────────────────────────────────────

/// Intermediate tree node used during construction.
struct SegNode {
    from_atom: usize,  // atom in parent segment (axis start)
    to_atom: usize,    // atom in this segment (axis end / pivot)
    atoms: Vec<usize>, // sdf atom indices in this segment
    children: Vec<SegNode>,
}

/// Build the torsion tree structure from molecule + rotatable bond info.
fn build_seg_tree(
    mol: &Molecule,
    atoms: &[ConvertedAtom],
    rotatable: &[bool],
) -> (Vec<usize>, Vec<SegNode>, usize) {
    let n = mol.atom_count();

    // Build adjacency list
    let mut adj: Vec<Vec<(usize, usize)>> = vec![vec![]; n]; // (neighbor, bond_index)
    for (bi, bond) in mol.bonds.iter().enumerate() {
        adj[bond.atom1].push((bond.atom2, bi));
        adj[bond.atom2].push((bond.atom1, bi));
    }

    // Choose root: heaviest non-H atom with most heavy connections
    let root = (0..n)
        .filter(|&i| atoms[i].el != EL_TYPE_H)
        .max_by_key(|&i| adj[i].iter().filter(|&&(nb, _)| atoms[nb].el != EL_TYPE_H).count())
        .unwrap_or(0);

    let mut visited = vec![false; n];
    let mut root_atoms = Vec::new();

    // BFS for root segment (don't cross rotatable bonds)
    collect_segment(root, &adj, rotatable, &mut visited, &mut root_atoms);

    let mut torsion_count = 0;
    let mut children = Vec::new();

    // Find rotatable bonds from root segment to unvisited atoms
    for (bi, bond) in mol.bonds.iter().enumerate() {
        if !rotatable[bi] { continue; }
        let (from, to) = if root_atoms.contains(&bond.atom1) && !visited[bond.atom2] {
            (bond.atom1, bond.atom2)
        } else if root_atoms.contains(&bond.atom2) && !visited[bond.atom1] {
            (bond.atom2, bond.atom1)
        } else {
            continue;
        };
        children.push(build_seg_node(from, to, mol, &adj, rotatable, &mut visited, &mut torsion_count));
    }

    (root_atoms, children, torsion_count)
}

/// BFS to collect atoms in a rigid segment (don't cross rotatable bonds).
fn collect_segment(
    start: usize,
    adj: &[Vec<(usize, usize)>],
    rotatable: &[bool],
    visited: &mut [bool],
    segment_atoms: &mut Vec<usize>,
) {
    let mut queue = std::collections::VecDeque::new();
    visited[start] = true;
    queue.push_back(start);

    while let Some(curr) = queue.pop_front() {
        segment_atoms.push(curr);
        for &(neighbor, bond_idx) in &adj[curr] {
            if visited[neighbor] || rotatable[bond_idx] { continue; }
            visited[neighbor] = true;
            queue.push_back(neighbor);
        }
    }
}

/// Recursively build a SegNode from a rotatable bond.
fn build_seg_node(
    from_atom: usize,
    to_atom: usize,
    mol: &Molecule,
    adj: &[Vec<(usize, usize)>],
    rotatable: &[bool],
    visited: &mut [bool],
    torsion_count: &mut usize,
) -> SegNode {
    *torsion_count += 1;
    let mut seg_atoms = Vec::new();
    collect_segment(to_atom, adj, rotatable, visited, &mut seg_atoms);

    let mut children = Vec::new();
    for (bi, bond) in mol.bonds.iter().enumerate() {
        if !rotatable[bi] { continue; }
        let (child_from, child_to) = if seg_atoms.contains(&bond.atom1) && !visited[bond.atom2] {
            (bond.atom1, bond.atom2)
        } else if seg_atoms.contains(&bond.atom2) && !visited[bond.atom1] {
            (bond.atom2, bond.atom1)
        } else {
            continue;
        };
        children.push(build_seg_node(child_from, child_to, mol, adj, rotatable, visited, torsion_count));
    }

    SegNode { from_atom, to_atom, atoms: seg_atoms, children }
}

// ─── Model Population ───────────────────────────────────────────────────────

/// Recursively add segment atoms to Model and build Segment tree.
fn add_seg_nodes_to_model(
    seg_nodes: &[SegNode],
    conv_atoms: &[ConvertedAtom],
    model: &mut Model,
    sdf_to_model: &mut [usize],
    branch_infos: &mut Vec<(usize, usize, usize, usize)>,  // (from_sdf, to_sdf, begin, end)
    parent_origin: &Vec3,
) -> Vec<Segment> {
    let mut segments = Vec::new();

    for node in seg_nodes {
        let segment_origin = conv_atoms[node.to_atom].coords;
        let from_coords = conv_atoms[node.from_atom].coords;

        let relative_origin = segment_origin - *parent_origin;

        let axis_vec = segment_origin - from_coords;
        let axis_len = axis_vec.norm();
        let relative_axis = if axis_len > EPSILON_FL {
            axis_vec * (1.0 / axis_len)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };

        let begin = model.atoms.len();
        for &sdf_idx in &node.atoms {
            let model_idx = model.atoms.len();
            sdf_to_model[sdf_idx] = model_idx;

            let ca = &conv_atoms[sdf_idx];
            let mut a = Atom::new();
            a.coords = ca.coords;
            a.charge = ca.charge;
            a.el = ca.el;
            a.ad = ca.ad;
            a.xs = ca.xs;
            a.sy = ca.sy;
            model.atoms.push(a);
            model.internal_coords.push(ca.coords - segment_origin);
            model.coords.push(ca.coords);
            model.minus_forces.push(Vec3::ZERO);
        }
        let end = model.atoms.len();

        branch_infos.push((node.from_atom, node.to_atom, begin, end));

        let atom_range = AtomRange::new(begin, end);
        let child_segments = add_seg_nodes_to_model(
            &node.children, conv_atoms, model, sdf_to_model,
            branch_infos, &segment_origin,
        );

        let mut segment = Segment::new(relative_axis, relative_origin, atom_range);
        segment.children = child_segments;
        segments.push(segment);
    }

    segments
}

/// Add bonds to model atoms from sdfrust bond table.
fn add_bonds_to_model(
    model: &mut Model,
    mol: &Molecule,
    sdf_to_model: &[usize],
    branch_infos: &[(usize, usize, usize, usize)],
) {
    for bond in &mol.bonds {
        let mi = sdf_to_model[bond.atom1];
        let mj = sdf_to_model[bond.atom2];
        let dist = model.atoms[mi].coords.distance_sqr(&model.atoms[mj].coords).sqrt();

        let is_hinge = branch_infos.iter().any(|&(from_sdf, to_sdf, _, _)| {
            (bond.atom1 == from_sdf && bond.atom2 == to_sdf) ||
            (bond.atom2 == from_sdf && bond.atom1 == to_sdf)
        });

        model.atoms[mi].bonds.push(Bond {
            connected_atom_index: AtomIndex { i: mj, in_grid: false },
            length: dist,
            rotatable: is_hinge,
        });
        model.atoms[mj].bonds.push(Bond {
            connected_atom_index: AtomIndex { i: mi, in_grid: false },
            length: dist,
            rotatable: is_hinge,
        });
    }
}

/// Get atoms within `depth` bond hops.
fn bonded_within(atoms: &[Atom], start: usize, depth: usize) -> Vec<usize> {
    let mut out = Vec::new();
    fn recurse(atoms: &[Atom], a: usize, n: usize, out: &mut Vec<usize>) {
        if out.contains(&a) { return; }
        out.push(a);
        if n > 0 {
            for bond in &atoms[a].bonds {
                if !bond.connected_atom_index.in_grid {
                    recurse(atoms, bond.connected_atom_index.i, n - 1, out);
                }
            }
        }
    }
    recurse(atoms, start, depth, &mut out);
    out
}

#[derive(Clone, Copy, PartialEq)]
enum MobType { Fixed, Rotor, Variable }

/// Build interacting pairs for the ligand.
fn build_pairs(
    atoms: &[Atom],
    begin: usize,
    end: usize,
    typing: AtomTyping,
    mobility: &[MobType],
    n_lig: usize,
) -> Vec<InteractingPair> {
    let mut pairs = Vec::new();
    let n_types = num_atom_types(typing);

    for i in begin..end {
        let bonded = bonded_within(atoms, i, 3);
        for j in (i + 1)..end {
            let (li, lj) = (i - begin, j - begin);
            if mobility[li * n_lig + lj] != MobType::Variable { continue; }
            if bonded.contains(&j) { continue; }

            let t1 = atoms[i].get_type(typing);
            let t2 = atoms[j].get_type(typing);
            if t1 >= n_types || t2 >= n_types { continue; }

            let (a, b) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            pairs.push(InteractingPair::new(a + b * (b + 1) / 2, i, j));
        }
    }
    pairs
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Parse an SDF string and add the ligand to the model.
///
/// Uses `sdfrust` for robust SDF/MOL parsing, then builds a torsion tree
/// compatible with the existing forward/reverse kinematics infrastructure.
pub fn parse_sdf_to_model(model: &mut Model, sdf: &str) -> Result<(), String> {
    // Parse with sdfrust
    let mol = sdfrust::parse_sdf_string(sdf)
        .map_err(|e| format!("SDF parse error: {}", e))?;

    add_molecule_to_model(model, &mol)
}

/// Parse an SDF file and add the first molecule to the model.
pub fn parse_sdf_file_to_model(model: &mut Model, path: &str) -> Result<(), String> {
    let mol = sdfrust::parse_sdf_file(path)
        .map_err(|e| format!("SDF file error: {}", e))?;

    add_molecule_to_model(model, &mol)
}

/// Parse a MOL2 string and add the ligand to the model.
pub fn parse_mol2_to_model(model: &mut Model, mol2: &str) -> Result<(), String> {
    let mol = sdfrust::parse_mol2_string(mol2)
        .map_err(|e| format!("MOL2 parse error: {}", e))?;

    add_molecule_to_model(model, &mol)
}

/// Add a sdfrust Molecule to the model as a ligand.
pub fn add_molecule_to_model(model: &mut Model, mol: &Molecule) -> Result<(), String> {
    if mol.atom_count() == 0 {
        return Err("No atoms in molecule".into());
    }

    // Convert atoms with type assignment
    let conv_atoms = convert_atoms(mol);

    // Find rotatable bonds
    let rotatable = find_rotatable_bonds(mol, &conv_atoms);

    // Build torsion tree structure
    let (root_atom_indices, seg_children, torsion_count) = build_seg_tree(mol, &conv_atoms, &rotatable);

    let atom_offset = model.atoms.len();

    // Root origin = geometric center of root atoms
    let root_origin = if root_atom_indices.is_empty() {
        conv_atoms[0].coords
    } else {
        let sum: Vec3 = root_atom_indices.iter()
            .map(|&i| conv_atoms[i].coords)
            .fold(Vec3::ZERO, |a, b| a + b);
        sum * (1.0 / root_atom_indices.len() as f64)
    };

    let mut sdf_to_model = vec![0usize; mol.atom_count()];

    // Add root atoms
    for &sdf_idx in &root_atom_indices {
        let model_idx = model.atoms.len();
        sdf_to_model[sdf_idx] = model_idx;

        let ca = &conv_atoms[sdf_idx];
        let mut a = Atom::new();
        a.coords = ca.coords;
        a.charge = ca.charge;
        a.el = ca.el;
        a.ad = ca.ad;
        a.xs = ca.xs;
        a.sy = ca.sy;
        model.atoms.push(a);
        model.internal_coords.push(ca.coords - root_origin);
        model.coords.push(ca.coords);
        model.minus_forces.push(Vec3::ZERO);
    }
    let root_end = model.atoms.len();

    // Add segment atoms recursively
    let mut branch_infos = Vec::new();
    let segments = add_seg_nodes_to_model(
        &seg_children, &conv_atoms, model, &mut sdf_to_model,
        &mut branch_infos, &root_origin,
    );

    let total_end = model.atoms.len();
    model.num_movable_atoms = total_end;

    // Build tree
    let atom_range = AtomRange::new(atom_offset, root_end);
    let mut tree = RigidBody::new(atom_range);
    tree.children = segments;

    // Add bonds
    add_bonds_to_model(model, mol, &sdf_to_model, &branch_infos);

    // Build mobility matrix + interacting pairs
    let n_lig = total_end - atom_offset;
    let mut mobility = vec![MobType::Variable; n_lig * n_lig];

    // Root: all fixed to each other
    for i in atom_offset..root_end {
        for j in (i + 1)..root_end {
            let (li, lj) = (i - atom_offset, j - atom_offset);
            mobility[li * n_lig + lj] = MobType::Fixed;
            mobility[lj * n_lig + li] = MobType::Fixed;
        }
    }

    // Branches: atoms fixed within segment, rotor at hinge
    for &(from_sdf, to_sdf, begin, end) in &branch_infos {
        // Atoms within segment are fixed
        for i in begin..end {
            for j in (i + 1)..end {
                let (li, lj) = (i - atom_offset, j - atom_offset);
                mobility[li * n_lig + lj] = MobType::Fixed;
                mobility[lj * n_lig + li] = MobType::Fixed;
            }
        }

        // Hinge bond = Rotor
        let from_model = sdf_to_model[from_sdf];
        let to_model = sdf_to_model[to_sdf];
        let (lf, lt) = (from_model - atom_offset, to_model - atom_offset);
        if lf < n_lig && lt < n_lig {
            mobility[lf * n_lig + lt] = MobType::Rotor;
            mobility[lt * n_lig + lf] = MobType::Rotor;

            // Hinge atoms fixed to their segment
            for i in begin..end {
                let li = i - atom_offset;
                mobility[lf * n_lig + li] = MobType::Fixed;
                mobility[li * n_lig + lf] = MobType::Fixed;
                mobility[lt * n_lig + li] = MobType::Fixed;
                mobility[li * n_lig + lt] = MobType::Fixed;
            }
        }
    }

    let pairs = build_pairs(&model.atoms, atom_offset, total_end, model.atom_typing, &mobility, n_lig);

    let mut ligand = Ligand::new(tree, torsion_count);
    ligand.begin = atom_offset;
    ligand.end = total_end;
    ligand.pairs = pairs;

    model.ligands.push(ligand);
    Ok(())
}

/// Get the number of rotatable bonds in an SDF molecule.
pub fn count_rotatable_bonds(sdf: &str) -> Result<usize, String> {
    let mol = sdfrust::parse_sdf_string(sdf)
        .map_err(|e| format!("SDF parse error: {}", e))?;
    let conv = convert_atoms(&mol);
    let rot = find_rotatable_bonds(&mol, &conv);
    Ok(rot.iter().filter(|&&r| r).count())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ethane_sdf() -> &'static str {
        "ethane\n\n\n  8  7  0  0  0  0  0  0  0  0999 V2000\n\
         \x20   0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20  -0.3900    0.9300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20  -0.3900   -0.4700    0.8200 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20  -0.3900   -0.4700   -0.8200 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   1.9300    0.9300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   1.9300   -0.4700    0.8200 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   1.9300   -0.4700   -0.8200 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20 1  2  1  0\n\
         \x20 1  3  1  0\n\
         \x20 1  4  1  0\n\
         \x20 1  5  1  0\n\
         \x20 2  6  1  0\n\
         \x20 2  7  1  0\n\
         \x20 2  8  1  0\n\
         M  END\n$$$$"
    }

    fn butane_sdf() -> &'static str {
        "butane\n\n\n 14 13  0  0  0  0  0  0  0  0999 V2000\n\
         \x20   0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   2.3100    1.3300    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   3.8500    1.3300    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20  -0.3900    0.9300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20  -0.3900   -0.4700    0.8200 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20  -0.3900   -0.4700   -0.8200 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   1.9300    0.9300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   1.9300   -0.9300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   1.9200    2.2600    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   1.9200    0.3900   -0.9000 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   4.2400    2.2600    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   4.2400    0.8600    0.8200 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20   4.2400    0.8600   -0.8200 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
         \x20 1  2  1  0\n\
         \x20 2  3  1  0\n\
         \x20 3  4  1  0\n\
         \x20 1  5  1  0\n\
         \x20 1  6  1  0\n\
         \x20 1  7  1  0\n\
         \x20 2  8  1  0\n\
         \x20 2  9  1  0\n\
         \x20 3 10  1  0\n\
         \x20 3 11  1  0\n\
         \x20 4 12  1  0\n\
         \x20 4 13  1  0\n\
         \x20 4 14  1  0\n\
         M  END\n$$$$"
    }

    #[test]
    fn test_parse_ethane() {
        let mol = sdfrust::parse_sdf_string(ethane_sdf()).unwrap();
        assert_eq!(mol.atom_count(), 8);
        assert_eq!(mol.bond_count(), 7);
    }

    #[test]
    fn test_ethane_no_rotatable() {
        let count = count_rotatable_bonds(ethane_sdf()).unwrap();
        assert_eq!(count, 0, "Ethane has no rotatable bonds (terminal C-C)");
    }

    #[test]
    fn test_ethane_model() {
        let mut model = Model::new();
        parse_sdf_to_model(&mut model, ethane_sdf()).unwrap();
        assert_eq!(model.atoms.len(), 8);
        assert_eq!(model.ligands.len(), 1);
        assert_eq!(model.ligands[0].degrees_of_freedom, 0);
    }

    #[test]
    fn test_butane_rotatable() {
        let count = count_rotatable_bonds(butane_sdf()).unwrap();
        assert_eq!(count, 1, "Butane has 1 rotatable bond (C2-C3)");
    }

    #[test]
    fn test_butane_model() {
        let mut model = Model::new();
        parse_sdf_to_model(&mut model, butane_sdf()).unwrap();
        assert_eq!(model.atoms.len(), 14);
        assert_eq!(model.ligands[0].degrees_of_freedom, 1);
    }

    #[test]
    fn test_tripos_types_assigned() {
        let mut model = Model::new();
        parse_sdf_to_model(&mut model, ethane_sdf()).unwrap();
        // Carbons should be C.3 (sp3)
        let carbons: Vec<_> = model.atoms.iter().filter(|a| a.el == EL_TYPE_C).collect();
        assert_eq!(carbons.len(), 2);
        for c in &carbons {
            assert_eq!(c.sy, SY_C3, "Ethane carbons should be SY_C3");
        }
        // Hydrogens should be SY_H (non-polar, bonded to C)
        let hydrogens: Vec<_> = model.atoms.iter().filter(|a| a.el == EL_TYPE_H).collect();
        assert_eq!(hydrogens.len(), 6);
        for h in &hydrogens {
            assert_eq!(h.sy, SY_H, "Ethane H atoms should be SY_H");
        }
    }
}
