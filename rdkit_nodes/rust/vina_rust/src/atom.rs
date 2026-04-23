use crate::common::Vec3;

// ─── Element Types ─────────────────────────────────────────────────────────────

pub const EL_TYPE_H: usize = 0;
pub const EL_TYPE_C: usize = 1;
pub const EL_TYPE_N: usize = 2;
pub const EL_TYPE_O: usize = 3;
pub const EL_TYPE_S: usize = 4;
pub const EL_TYPE_P: usize = 5;
pub const EL_TYPE_F: usize = 6;
pub const EL_TYPE_CL: usize = 7;
pub const EL_TYPE_BR: usize = 8;
pub const EL_TYPE_I: usize = 9;
pub const EL_TYPE_SI: usize = 10;
pub const EL_TYPE_AT: usize = 11;
pub const EL_TYPE_MET: usize = 12;
pub const EL_TYPE_DUMMY: usize = 13;
pub const EL_TYPE_SIZE: usize = 14;

// ─── AutoDock4 Types ───────────────────────────────────────────────────────────

pub const AD_TYPE_C: usize = 0;
pub const AD_TYPE_A: usize = 1;
pub const AD_TYPE_N: usize = 2;
pub const AD_TYPE_O: usize = 3;
pub const AD_TYPE_P: usize = 4;
pub const AD_TYPE_S: usize = 5;
pub const AD_TYPE_H: usize = 6;
pub const AD_TYPE_F: usize = 7;
pub const AD_TYPE_I: usize = 8;
pub const AD_TYPE_NA: usize = 9;
pub const AD_TYPE_OA: usize = 10;
pub const AD_TYPE_SA: usize = 11;
pub const AD_TYPE_HD: usize = 12;
pub const AD_TYPE_MG: usize = 13;
pub const AD_TYPE_MN: usize = 14;
pub const AD_TYPE_ZN: usize = 15;
pub const AD_TYPE_CA: usize = 16;
pub const AD_TYPE_FE: usize = 17;
pub const AD_TYPE_CL: usize = 18;
pub const AD_TYPE_BR: usize = 19;
pub const AD_TYPE_SI: usize = 20;
pub const AD_TYPE_AT: usize = 21;
pub const AD_TYPE_G0: usize = 22;
pub const AD_TYPE_G1: usize = 23;
pub const AD_TYPE_G2: usize = 24;
pub const AD_TYPE_G3: usize = 25;
pub const AD_TYPE_CG0: usize = 26;
pub const AD_TYPE_CG1: usize = 27;
pub const AD_TYPE_CG2: usize = 28;
pub const AD_TYPE_CG3: usize = 29;
pub const AD_TYPE_W: usize = 30;
pub const AD_TYPE_SIZE: usize = 31;

// ─── X-Score Types ─────────────────────────────────────────────────────────────

pub const XS_TYPE_C_H: usize = 0;
pub const XS_TYPE_C_P: usize = 1;
pub const XS_TYPE_N_P: usize = 2;
pub const XS_TYPE_N_D: usize = 3;
pub const XS_TYPE_N_A: usize = 4;
pub const XS_TYPE_N_DA: usize = 5;
pub const XS_TYPE_O_P: usize = 6;
pub const XS_TYPE_O_D: usize = 7;
pub const XS_TYPE_O_A: usize = 8;
pub const XS_TYPE_O_DA: usize = 9;
pub const XS_TYPE_S_P: usize = 10;
pub const XS_TYPE_P_P: usize = 11;
pub const XS_TYPE_F_H: usize = 12;
pub const XS_TYPE_CL_H: usize = 13;
pub const XS_TYPE_BR_H: usize = 14;
pub const XS_TYPE_I_H: usize = 15;
pub const XS_TYPE_SI: usize = 16;
pub const XS_TYPE_AT: usize = 17;
pub const XS_TYPE_MET_D: usize = 18;
pub const XS_TYPE_C_H_CG0: usize = 19;
pub const XS_TYPE_C_P_CG0: usize = 20;
pub const XS_TYPE_G0: usize = 21;
pub const XS_TYPE_C_H_CG1: usize = 22;
pub const XS_TYPE_C_P_CG1: usize = 23;
pub const XS_TYPE_G1: usize = 24;
pub const XS_TYPE_C_H_CG2: usize = 25;
pub const XS_TYPE_C_P_CG2: usize = 26;
pub const XS_TYPE_G2: usize = 27;
pub const XS_TYPE_C_H_CG3: usize = 28;
pub const XS_TYPE_C_P_CG3: usize = 29;
pub const XS_TYPE_G3: usize = 30;
pub const XS_TYPE_W: usize = 31;
pub const XS_TYPE_B_H: usize = 32;  // Boron (smina extension) — hydrophobic
pub const XS_TYPE_SIZE: usize = 33;

// ─── Atom Kind Data ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AtomKind {
    pub name: &'static str,
    pub radius: f64,
    pub depth: f64,
    pub hb_depth: f64,
    pub hb_radius: f64,
    pub solvation: f64,
    pub volume: f64,
    pub covalent_radius: f64,
}

pub static ATOM_KIND_DATA: [AtomKind; AD_TYPE_SIZE] = [
    AtomKind { name: "C",   radius: 2.00000, depth: 0.15000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00143, volume: 33.51030, covalent_radius: 0.77 },
    AtomKind { name: "A",   radius: 2.00000, depth: 0.15000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00052, volume: 33.51030, covalent_radius: 0.77 },
    AtomKind { name: "N",   radius: 1.75000, depth: 0.16000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00162, volume: 22.44930, covalent_radius: 0.75 },
    AtomKind { name: "O",   radius: 1.60000, depth: 0.20000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00251, volume: 17.15730, covalent_radius: 0.73 },
    AtomKind { name: "P",   radius: 2.10000, depth: 0.20000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00110, volume: 38.79240, covalent_radius: 1.06 },
    AtomKind { name: "S",   radius: 2.00000, depth: 0.20000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00214, volume: 33.51030, covalent_radius: 1.02 },
    AtomKind { name: "H",   radius: 1.00000, depth: 0.02000, hb_depth:  0.0, hb_radius: 0.0, solvation:  0.00051, volume:  0.00000, covalent_radius: 0.37 },
    AtomKind { name: "F",   radius: 1.54500, depth: 0.08000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00110, volume: 15.44800, covalent_radius: 0.71 },
    AtomKind { name: "I",   radius: 2.36000, depth: 0.55000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00110, volume: 55.05850, covalent_radius: 1.33 },
    AtomKind { name: "NA",  radius: 1.75000, depth: 0.16000, hb_depth: -5.0, hb_radius: 1.9, solvation: -0.00162, volume: 22.44930, covalent_radius: 0.75 },
    AtomKind { name: "OA",  radius: 1.60000, depth: 0.20000, hb_depth: -5.0, hb_radius: 1.9, solvation: -0.00251, volume: 17.15730, covalent_radius: 0.73 },
    AtomKind { name: "SA",  radius: 2.00000, depth: 0.20000, hb_depth: -1.0, hb_radius: 2.5, solvation: -0.00214, volume: 33.51030, covalent_radius: 1.02 },
    AtomKind { name: "HD",  radius: 1.00000, depth: 0.02000, hb_depth:  1.0, hb_radius: 0.0, solvation:  0.00051, volume:  0.00000, covalent_radius: 0.37 },
    AtomKind { name: "Mg",  radius: 0.65000, depth: 0.87500, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00110, volume:  1.56000, covalent_radius: 1.30 },
    AtomKind { name: "Mn",  radius: 0.65000, depth: 0.87500, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00110, volume:  2.14000, covalent_radius: 1.39 },
    AtomKind { name: "Zn",  radius: 0.74000, depth: 0.55000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00110, volume:  1.70000, covalent_radius: 1.31 },
    AtomKind { name: "Ca",  radius: 0.99000, depth: 0.55000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00110, volume:  2.77000, covalent_radius: 1.74 },
    AtomKind { name: "Fe",  radius: 0.65000, depth: 0.01000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00110, volume:  1.84000, covalent_radius: 1.25 },
    AtomKind { name: "Cl",  radius: 2.04500, depth: 0.27600, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00110, volume: 35.82350, covalent_radius: 0.99 },
    AtomKind { name: "Br",  radius: 2.16500, depth: 0.38900, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00110, volume: 42.56610, covalent_radius: 1.14 },
    AtomKind { name: "Si",  radius: 2.30000, depth: 0.20000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00143, volume: 50.96500, covalent_radius: 1.11 },
    AtomKind { name: "At",  radius: 2.40000, depth: 0.55000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00110, volume: 57.90580, covalent_radius: 1.44 },
    AtomKind { name: "G0",  radius: 0.00000, depth: 0.00000, hb_depth:  0.0, hb_radius: 0.0, solvation:  0.00000, volume:  0.00000, covalent_radius: 0.77 },
    AtomKind { name: "G1",  radius: 0.00000, depth: 0.00000, hb_depth:  0.0, hb_radius: 0.0, solvation:  0.00000, volume:  0.00000, covalent_radius: 0.77 },
    AtomKind { name: "G2",  radius: 0.00000, depth: 0.00000, hb_depth:  0.0, hb_radius: 0.0, solvation:  0.00000, volume:  0.00000, covalent_radius: 0.77 },
    AtomKind { name: "G3",  radius: 0.00000, depth: 0.00000, hb_depth:  0.0, hb_radius: 0.0, solvation:  0.00000, volume:  0.00000, covalent_radius: 0.77 },
    AtomKind { name: "CG0", radius: 2.00000, depth: 0.15000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00143, volume: 33.51030, covalent_radius: 0.77 },
    AtomKind { name: "CG1", radius: 2.00000, depth: 0.15000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00143, volume: 33.51030, covalent_radius: 0.77 },
    AtomKind { name: "CG2", radius: 2.00000, depth: 0.15000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00143, volume: 33.51030, covalent_radius: 0.77 },
    AtomKind { name: "CG3", radius: 2.00000, depth: 0.15000, hb_depth:  0.0, hb_radius: 0.0, solvation: -0.00143, volume: 33.51030, covalent_radius: 0.77 },
    AtomKind { name: "W",   radius: 0.00000, depth: 0.00000, hb_depth:  0.0, hb_radius: 0.0, solvation:  0.00000, volume:  0.00000, covalent_radius: 0.00 },
];

pub const METAL_SOLVATION_PARAMETER: f64 = -0.00110;
pub const METAL_COVALENT_RADIUS: f64 = 1.75;

// ─── VDW Radii ─────────────────────────────────────────────────────────────────

pub static XS_VDW_RADII: [f64; XS_TYPE_SIZE] = [
    1.9, // C_H
    1.9, // C_P
    1.8, // N_P
    1.8, // N_D
    1.8, // N_A
    1.8, // N_DA
    1.7, // O_P
    1.7, // O_D
    1.7, // O_A
    1.7, // O_DA
    2.0, // S_P
    2.1, // P_P
    1.5, // F_H
    1.8, // Cl_H
    2.0, // Br_H
    2.2, // I_H
    2.2, // Si
    2.3, // At
    1.2, // Met_D
    1.9, // C_H_CG0
    1.9, // C_P_CG0
    0.0, // G0
    1.9, // C_H_CG1
    1.9, // C_P_CG1
    0.0, // G1
    1.9, // C_H_CG2
    1.9, // C_P_CG2
    0.0, // G2
    1.9, // C_H_CG3
    1.9, // C_P_CG3
    0.0, // G3
    0.0, // W
    1.92, // B_H (Boron, smina)
];

pub static XS_VINARDO_VDW_RADII: [f64; XS_TYPE_SIZE] = [
    2.0, // C_H
    2.0, // C_P
    1.7, // N_P
    1.7, // N_D
    1.7, // N_A
    1.7, // N_DA
    1.6, // O_P
    1.6, // O_D
    1.6, // O_A
    1.6, // O_DA
    2.0, // S_P
    2.1, // P_P
    1.5, // F_H
    1.8, // Cl_H
    2.0, // Br_H
    2.2, // I_H
    2.2, // Si
    2.3, // At
    1.2, // Met_D
    2.0, // C_H_CG0
    2.0, // C_P_CG0
    0.0, // G0
    2.0, // C_H_CG1
    2.0, // C_P_CG1
    0.0, // G1
    2.0, // C_H_CG2
    2.0, // C_P_CG2
    0.0, // G2
    2.0, // C_H_CG3
    2.0, // C_P_CG3
    0.0, // G3
    0.0, // W
    1.92, // B_H (Boron, smina)
];

// ─── Helper Functions ──────────────────────────────────────────────────────────

#[inline(always)]
pub fn xs_radius(t: usize) -> f64 {
    debug_assert!(t < XS_TYPE_SIZE);
    XS_VDW_RADII[t]
}

#[inline(always)]
pub fn xs_vinardo_radius(t: usize) -> f64 {
    debug_assert!(t < XS_TYPE_SIZE);
    XS_VINARDO_VDW_RADII[t]
}

#[inline(always)]
pub fn xs_is_hydrophobic(xs: usize) -> bool {
    xs == XS_TYPE_C_H || xs == XS_TYPE_F_H || xs == XS_TYPE_CL_H || xs == XS_TYPE_BR_H || xs == XS_TYPE_I_H
        || xs == XS_TYPE_B_H
}

#[inline(always)]
pub fn xs_is_acceptor(xs: usize) -> bool {
    xs == XS_TYPE_N_A || xs == XS_TYPE_N_DA || xs == XS_TYPE_O_A || xs == XS_TYPE_O_DA
}

#[inline(always)]
pub fn xs_is_donor(xs: usize) -> bool {
    xs == XS_TYPE_N_D || xs == XS_TYPE_N_DA || xs == XS_TYPE_O_D || xs == XS_TYPE_O_DA || xs == XS_TYPE_MET_D
}

#[inline(always)]
pub fn xs_h_bond_possible(t1: usize, t2: usize) -> bool {
    (xs_is_donor(t1) && xs_is_acceptor(t2)) || (xs_is_donor(t2) && xs_is_acceptor(t1))
}

#[inline(always)]
pub fn is_glue_type(xs_t: usize) -> bool {
    xs_t == XS_TYPE_G0 || xs_t == XS_TYPE_G1 || xs_t == XS_TYPE_G2 || xs_t == XS_TYPE_G3
}

pub fn is_glued(xs_t1: usize, xs_t2: usize) -> bool {
    (xs_t1 == XS_TYPE_G0 && (xs_t2 == XS_TYPE_C_H_CG0 || xs_t2 == XS_TYPE_C_P_CG0)) ||
    (xs_t2 == XS_TYPE_G0 && (xs_t1 == XS_TYPE_C_H_CG0 || xs_t1 == XS_TYPE_C_P_CG0)) ||
    (xs_t1 == XS_TYPE_G1 && (xs_t2 == XS_TYPE_C_H_CG1 || xs_t2 == XS_TYPE_C_P_CG1)) ||
    (xs_t2 == XS_TYPE_G1 && (xs_t1 == XS_TYPE_C_H_CG1 || xs_t1 == XS_TYPE_C_P_CG1)) ||
    (xs_t1 == XS_TYPE_G2 && (xs_t2 == XS_TYPE_C_H_CG2 || xs_t2 == XS_TYPE_C_P_CG2)) ||
    (xs_t2 == XS_TYPE_G2 && (xs_t1 == XS_TYPE_C_H_CG2 || xs_t1 == XS_TYPE_C_P_CG2)) ||
    (xs_t1 == XS_TYPE_G3 && (xs_t2 == XS_TYPE_C_H_CG3 || xs_t2 == XS_TYPE_C_P_CG3)) ||
    (xs_t2 == XS_TYPE_G3 && (xs_t1 == XS_TYPE_C_H_CG3 || xs_t1 == XS_TYPE_C_P_CG3))
}

#[inline(always)]
pub fn optimal_distance(xs_t1: usize, xs_t2: usize) -> f64 {
    if is_glue_type(xs_t1) || is_glue_type(xs_t2) { return 0.0; }
    xs_radius(xs_t1) + xs_radius(xs_t2)
}

#[inline(always)]
pub fn optimal_distance_vinardo(xs_t1: usize, xs_t2: usize) -> f64 {
    if is_glue_type(xs_t1) || is_glue_type(xs_t2) { return 0.0; }
    xs_vinardo_radius(xs_t1) + xs_vinardo_radius(xs_t2)
}

pub fn ad_is_hydrogen(ad: usize) -> bool {
    ad == AD_TYPE_H || ad == AD_TYPE_HD
}

pub fn ad_is_heteroatom(ad: usize) -> bool {
    if ad >= AD_TYPE_SIZE { return false; }
    // Match C++ bonded_to_heteroatom: neighbor.el != EL_TYPE_H && neighbor.el != EL_TYPE_C
    let el = ad_type_to_el_type(ad);
    el != EL_TYPE_H && el != EL_TYPE_C
}

pub fn ad_type_to_el_type(t: usize) -> usize {
    match t {
        AD_TYPE_C | AD_TYPE_A => EL_TYPE_C,
        AD_TYPE_N => EL_TYPE_N,
        AD_TYPE_O => EL_TYPE_O,
        AD_TYPE_P => EL_TYPE_P,
        AD_TYPE_S => EL_TYPE_S,
        AD_TYPE_H => EL_TYPE_H,
        AD_TYPE_F => EL_TYPE_F,
        AD_TYPE_I => EL_TYPE_I,
        AD_TYPE_NA => EL_TYPE_N,
        AD_TYPE_OA => EL_TYPE_O,
        AD_TYPE_SA => EL_TYPE_S,
        AD_TYPE_HD => EL_TYPE_H,
        AD_TYPE_MG | AD_TYPE_MN | AD_TYPE_ZN | AD_TYPE_CA | AD_TYPE_FE => EL_TYPE_MET,
        AD_TYPE_CL => EL_TYPE_CL,
        AD_TYPE_BR => EL_TYPE_BR,
        AD_TYPE_SI => EL_TYPE_SI,
        AD_TYPE_AT => EL_TYPE_AT,
        AD_TYPE_CG0 | AD_TYPE_CG1 | AD_TYPE_CG2 | AD_TYPE_CG3 => EL_TYPE_C,
        AD_TYPE_G0 | AD_TYPE_G1 | AD_TYPE_G2 | AD_TYPE_G3 | AD_TYPE_W => EL_TYPE_DUMMY,
        _ => EL_TYPE_SIZE,
    }
}

pub fn string_to_ad_type(name: &str) -> usize {
    for (i, kind) in ATOM_KIND_DATA.iter().enumerate() {
        if kind.name == name {
            return i;
        }
    }
    // Atom equivalences
    if name == "Se" { return string_to_ad_type("S"); }
    AD_TYPE_SIZE
}

pub fn ad_type_property(i: usize) -> &'static AtomKind {
    debug_assert!(i < AD_TYPE_SIZE);
    &ATOM_KIND_DATA[i]
}

pub fn max_covalent_radius() -> f64 {
    ATOM_KIND_DATA.iter().map(|k| k.covalent_radius).fold(0.0_f64, f64::max)
}

pub static NON_AD_METAL_NAMES: [&str; 9] = ["Cu", "Fe", "Na", "K", "Hg", "Co", "U", "Cd", "Ni"];

pub fn is_non_ad_metal_name(name: &str) -> bool {
    NON_AD_METAL_NAMES.contains(&name)
}

/// Atom typing scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomTyping {
    EL,
    AD,
    XS,
    SY,
}

pub fn num_atom_types(typing: AtomTyping) -> usize {
    match typing {
        AtomTyping::EL => EL_TYPE_SIZE,
        AtomTyping::AD => AD_TYPE_SIZE,
        AtomTyping::XS => XS_TYPE_SIZE,
        AtomTyping::SY => 18,
    }
}

// ─── Atom Structs ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomIndex {
    pub i: usize,
    pub in_grid: bool,
}

#[derive(Debug, Clone)]
pub struct Bond {
    pub connected_atom_index: AtomIndex,
    pub length: f64,
    pub rotatable: bool,
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub el: usize,
    pub ad: usize,
    pub xs: usize,
    pub sy: usize,
    pub charge: f64,
    pub coords: Vec3,
    pub bonds: Vec<Bond>,
}

impl Atom {
    pub fn new() -> Self {
        Atom {
            el: EL_TYPE_SIZE,
            ad: AD_TYPE_SIZE,
            xs: XS_TYPE_SIZE,
            sy: 18,
            charge: 0.0,
            coords: Vec3::ZERO,
            bonds: Vec::new(),
        }
    }

    pub fn get_type(&self, typing: AtomTyping) -> usize {
        match typing {
            AtomTyping::EL => self.el,
            AtomTyping::AD => self.ad,
            AtomTyping::XS => self.xs,
            AtomTyping::SY => self.sy,
        }
    }

    pub fn is_hydrogen(&self) -> bool {
        self.el == EL_TYPE_H
    }

    pub fn is_heteroatom(&self) -> bool {
        (self.ad < AD_TYPE_SIZE && ad_is_heteroatom(self.ad)) || self.xs == XS_TYPE_MET_D
    }

    pub fn covalent_radius(&self) -> f64 {
        if self.ad < AD_TYPE_SIZE {
            ATOM_KIND_DATA[self.ad].covalent_radius
        } else {
            METAL_COVALENT_RADIUS
        }
    }
}

impl Default for Atom {
    fn default() -> Self { Atom::new() }
}

// ─── InteractingPair ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct InteractingPair {
    pub type_pair_index: usize,
    pub a: usize,
    pub b: usize,
}

impl InteractingPair {
    pub fn new(type_pair_index: usize, a: usize, b: usize) -> Self {
        InteractingPair { type_pair_index, a, b }
    }
}

/// Get type pair index for triangular matrix indexing
pub fn get_type_pair_index(typing: AtomTyping, a: &Atom, b: &Atom) -> usize {
    let t1 = a.get_type(typing);
    let t2 = b.get_type(typing);
    let (i, j) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
    i + j * (j + 1) / 2
}

/// Acceptor kind (for AD4)
pub struct AcceptorKind {
    pub ad_type: usize,
    pub radius: f64,
    pub depth: f64,
}

pub static ACCEPTOR_KIND_DATA: [AcceptorKind; 3] = [
    AcceptorKind { ad_type: AD_TYPE_NA, radius: 1.9, depth: 5.0 },
    AcceptorKind { ad_type: AD_TYPE_OA, radius: 1.9, depth: 5.0 },
    AcceptorKind { ad_type: AD_TYPE_SA, radius: 2.5, depth: 1.0 },
];
