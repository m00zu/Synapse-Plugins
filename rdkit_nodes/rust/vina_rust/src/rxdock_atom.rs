//! Tripos/Sybyl atom types and VDW parameters for rxDock-style scoring.
//!
//! Clean-room implementation from published Tripos 5.2 force field parameters
//! (Clark, Cramer & Van Opdenbosch, J. Comput. Chem. 1989, 10, 982–1012).

// ─── Tripos/Sybyl Atom Type Constants ─────────────────────────────────────────

pub const SY_C3: usize = 0;       // sp3 carbon
pub const SY_C2: usize = 1;       // sp2 carbon
pub const SY_C1: usize = 2;       // sp carbon
pub const SY_CAR: usize = 3;      // aromatic carbon
pub const SY_CCAT: usize = 4;     // cationic carbon (guanidinium)
pub const SY_N3: usize = 5;       // sp3 nitrogen
pub const SY_N2: usize = 6;       // sp2 nitrogen
pub const SY_N1: usize = 7;       // sp nitrogen
pub const SY_NAR: usize = 8;      // aromatic nitrogen
pub const SY_NAM: usize = 9;      // amide nitrogen
pub const SY_NPL3: usize = 10;    // trigonal planar nitrogen
pub const SY_O3: usize = 11;      // sp3 oxygen (hydroxyl, ether)
pub const SY_O2: usize = 12;      // sp2 oxygen (carbonyl)
pub const SY_OCO2: usize = 13;    // carboxylate oxygen
pub const SY_S3: usize = 14;      // sp3 sulfur
pub const SY_S2: usize = 15;      // sp2 sulfur
pub const SY_P3: usize = 16;      // sp3 phosphorus
pub const SY_F: usize = 17;       // fluorine
pub const SY_CL: usize = 18;      // chlorine
pub const SY_BR: usize = 19;      // bromine
pub const SY_I: usize = 20;       // iodine
pub const SY_H: usize = 21;       // non-polar hydrogen
pub const SY_HP: usize = 22;      // polar hydrogen (H-bond donor)
pub const SY_MET: usize = 23;     // metal (Zn, Fe, Mg, etc.)
pub const SY_SIZE: usize = 24;

// ─── VDW Parameters (Tripos 5.2 published values) ────────────────────────────

/// Per-type VDW info: radius (Å), well depth ε (kcal/mol), ionisation potential,
/// and polarisability (for GOLD well depth formula).
#[derive(Debug, Clone, Copy)]
pub struct TriposTypeInfo {
    pub name: &'static str,
    pub radius: f64,
    pub depth: f64,
    pub ip: f64,     // ionisation potential (eV); 0 = not available
    pub pol: f64,    // polarisability (Å³); 0 = not available
    pub is_hbd: bool,
    pub is_hba: bool,
}

/// Published Tripos 5.2 + GOLD parameters from tripos-5-2-vdw.json.
pub static TRIPOS_PARAMS: [TriposTypeInfo; SY_SIZE] = [
    // SY_C3 = 0
    TriposTypeInfo { name: "C.3",   radius: 1.70, depth: 0.107, ip: 14.61, pol: 13.8, is_hbd: false, is_hba: false },
    // SY_C2 = 1
    TriposTypeInfo { name: "C.2",   radius: 1.70, depth: 0.107, ip: 15.62, pol: 13.8, is_hbd: false, is_hba: false },
    // SY_C1 = 2
    TriposTypeInfo { name: "C.1",   radius: 1.70, depth: 0.107, ip: 17.47, pol: 13.8, is_hbd: false, is_hba: false },
    // SY_CAR = 3
    TriposTypeInfo { name: "C.ar",  radius: 1.70, depth: 0.107, ip: 15.62, pol: 13.8, is_hbd: false, is_hba: false },
    // SY_CCAT = 4
    TriposTypeInfo { name: "C.cat", radius: 1.70, depth: 0.107, ip: 15.62, pol: 13.8, is_hbd: false, is_hba: false },
    // SY_N3 = 5
    TriposTypeInfo { name: "N.3",   radius: 1.55, depth: 0.095, ip: 18.93, pol: 8.4, is_hbd: false, is_hba: true },
    // SY_N2 = 6
    TriposTypeInfo { name: "N.2",   radius: 1.55, depth: 0.095, ip: 22.10, pol: 8.4, is_hbd: false, is_hba: true },
    // SY_N1 = 7
    TriposTypeInfo { name: "N.1",   radius: 1.55, depth: 0.095, ip: 23.91, pol: 8.4, is_hbd: false, is_hba: true },
    // SY_NAR = 8
    TriposTypeInfo { name: "N.ar",  radius: 1.55, depth: 0.095, ip: 22.10, pol: 8.4, is_hbd: false, is_hba: true },
    // SY_NAM = 9
    TriposTypeInfo { name: "N.am",  radius: 1.55, depth: 0.095, ip: 19.72, pol: 8.4, is_hbd: false, is_hba: false },
    // SY_NPL3 = 10
    TriposTypeInfo { name: "N.pl3", radius: 1.55, depth: 0.095, ip: 19.72, pol: 8.4, is_hbd: false, is_hba: false },
    // SY_O3 = 11
    TriposTypeInfo { name: "O.3",   radius: 1.52, depth: 0.116, ip: 24.39, pol: 5.4, is_hbd: false, is_hba: true },
    // SY_O2 = 12
    TriposTypeInfo { name: "O.2",   radius: 1.52, depth: 0.116, ip: 26.65, pol: 5.4, is_hbd: false, is_hba: true },
    // SY_OCO2 = 13
    TriposTypeInfo { name: "O.co2", radius: 1.52, depth: 0.116, ip: 35.12, pol: 5.4, is_hbd: false, is_hba: true },
    // SY_S3 = 14
    TriposTypeInfo { name: "S.3",   radius: 1.80, depth: 0.314, ip: 15.50, pol: 29.4, is_hbd: false, is_hba: false },
    // SY_S2 = 15
    TriposTypeInfo { name: "S.2",   radius: 1.80, depth: 0.314, ip: 17.78, pol: 29.4, is_hbd: false, is_hba: false },
    // SY_P3 = 16
    TriposTypeInfo { name: "P.3",   radius: 1.80, depth: 0.314, ip: 16.78, pol: 40.6, is_hbd: false, is_hba: false },
    // SY_F = 17
    TriposTypeInfo { name: "F",     radius: 1.47, depth: 0.109, ip: 20.86, pol: 3.7, is_hbd: false, is_hba: false },
    // SY_CL = 18
    TriposTypeInfo { name: "Cl",    radius: 1.75, depth: 0.314, ip: 15.03, pol: 21.8, is_hbd: false, is_hba: false },
    // SY_BR = 19
    TriposTypeInfo { name: "Br",    radius: 1.85, depth: 0.434, ip: 13.10, pol: 31.2, is_hbd: false, is_hba: false },
    // SY_I = 20
    TriposTypeInfo { name: "I",     radius: 1.98, depth: 0.623, ip: 12.67, pol: 49.0, is_hbd: false, is_hba: false },
    // SY_H = 21  — C++ rxDock: r=1.5
    TriposTypeInfo { name: "H",     radius: 1.50, depth: 0.042, ip: 13.60, pol: 4.0, is_hbd: false, is_hba: false },
    // SY_HP = 22  — C++ rxDock: r=1.2
    TriposTypeInfo { name: "H.P",   radius: 1.20, depth: 0.042, ip: 13.60, pol: 4.0, is_hbd: true, is_hba: false },
    // SY_MET = 23
    TriposTypeInfo { name: "Met",   radius: 1.20, depth: 0.875, ip: 0.0, pol: 0.0, is_hbd: false, is_hba: false },
];

// ─── Pairwise VDW Parameters ─────────────────────────────────────────────────

/// Precomputed pairwise VDW parameters for a single atom-type pair.
#[derive(Debug, Clone, Copy)]
pub struct VdwPairParams {
    pub rmin: f64,          // equilibrium distance = r_i + r_j
    pub kij: f64,           // combined well depth = sqrt(eps_i * eps_j)
    pub a_coeff: f64,       // repulsive coefficient
    pub b_coeff: f64,       // attractive coefficient
    pub rmax_sq: f64,       // cutoff distance squared (beyond = 0)
    pub rswitch_sq: f64,    // short-range quadratic switch distance²
    pub e_at_switch: f64,   // energy value at the switch distance
    pub slope: f64,         // quadratic slope for short-range cap
}

/// Precomputed VDW lookup table for all Tripos type pairs.
pub struct TriposVdwTable {
    /// Triangular storage: index(t1, t2) = min * SY_SIZE + max
    params: Vec<VdwPairParams>,
    pub rmax: f64,
    pub use_4_8: bool,
}

impl TriposVdwTable {
    /// Build the pairwise VDW table using Tripos combination rules (kij = sqrt(Ki*Kj)).
    ///
    /// - `use_4_8`: if true, use 4-8 potential (softer); if false, use 6-12 (standard LJ)
    /// - `rmax_factor`: cutoff as multiple of rmin (default 1.5, per C++ rxDock)
    /// - `ecut`: energy cap as multiple of kij (default 120, per C++ rxDock)
    pub fn new(use_4_8: bool, rmax_factor: f64, ecut: f64) -> Self {
        Self::build(use_4_8, rmax_factor, ecut, 1.5, true)
    }

    /// Build with GOLD well depths (for intermolecular VDW).
    /// Falls back to Tripos for pairs where IP/POL not available.
    pub fn new_gold(use_4_8: bool, rmax_factor: f64, ecut: f64) -> Self {
        Self::build(use_4_8, rmax_factor, ecut, 1.5, false)
    }

    fn build(use_4_8: bool, rmax_factor: f64, ecut: f64, e0: f64, use_tripos: bool) -> Self {
        let mut params = vec![VdwPairParams {
            rmin: 0.0, kij: 0.0, a_coeff: 0.0, b_coeff: 0.0,
            rmax_sq: 0.0, rswitch_sq: 0.0, e_at_switch: 0.0, slope: 0.0,
        }; SY_SIZE * SY_SIZE];

        let mut max_rmax: f64 = 0.0;

        // C++ analytic cutoff: x = 1 + sqrt(1 + ecut), c = 1 / x^(1/n)
        let x = 1.0 + (1.0 + ecut).sqrt();
        let n = if use_4_8 { 4.0 } else { 6.0 };
        let c = 1.0 / x.powf(1.0 / n);

        for i in 0..SY_SIZE {
            for j in 0..SY_SIZE {
                let pi = &TRIPOS_PARAMS[i];
                let pj = &TRIPOS_PARAMS[j];

                if pi.radius <= 0.0 || pj.radius <= 0.0 || pi.depth <= 0.0 || pj.depth <= 0.0 {
                    continue;
                }

                let rmin = pi.radius + pj.radius;

                // Zero VDW between H-bond donor H and H-bond acceptors
                // (H-bond interactions handled by PolarSF)
                if (pi.is_hbd && pj.is_hba) || (pi.is_hba && pj.is_hbd) {
                    // kij = 0, skip
                    let rmax_dist = rmin * rmax_factor;
                    if rmax_dist > max_rmax { max_rmax = rmax_dist; }
                    continue;
                }

                // Well depth: GOLD formula or Tripos combination rule
                let kij = if use_tripos || pi.ip <= 0.0 || pj.ip <= 0.0 {
                    // Tripos: kij = sqrt(Ki * Kj)
                    (pi.depth * pj.depth).sqrt()
                } else {
                    // GOLD: kij = D² / (4C) where D, C from IP and POL
                    let d_val = 0.345 * pi.ip * pj.ip * pi.pol * pj.pol / (pi.ip + pj.ip);
                    let c_val = 0.5 * d_val * rmin.powi(6);
                    if c_val > 0.0 { d_val * d_val / (4.0 * c_val) } else { (pi.depth * pj.depth).sqrt() }
                };

                // Per-pair cutoff distance: rmin * rmax_factor
                let rmax_dist = rmin * rmax_factor;
                let rmax_sq = rmax_dist * rmax_dist;
                if rmax_dist > max_rmax {
                    max_rmax = rmax_dist;
                }

                let (a, b) = if use_4_8 {
                    let rmin4 = rmin * rmin * rmin * rmin;
                    let rmin8 = rmin4 * rmin4;
                    (kij * rmin8, 2.0 * kij * rmin4)
                } else {
                    let rmin6 = rmin.powi(6);
                    let rmin12 = rmin6 * rmin6;
                    (kij * rmin12, 2.0 * kij * rmin6)
                };

                // Per-pair energy cap: kij * ecut
                let ecutoff = kij * ecut;
                let e0_energy = ecutoff * e0;

                // C++ analytic switch distance: rswitch = rmin * c
                let rswitch = rmin * c;
                let rswitch_sq = rswitch * rswitch;

                // Quadratic cap: E(r²) = e0_energy - slope * r²
                // At r=rswitch: E = ecutoff, so slope = (e0_energy - ecutoff) / rswitch_sq
                let slope = if rswitch_sq > 0.0 { (e0_energy - ecutoff) / rswitch_sq } else { 0.0 };

                params[i * SY_SIZE + j] = VdwPairParams {
                    rmin, kij, a_coeff: a, b_coeff: b,
                    rmax_sq, rswitch_sq,
                    e_at_switch: ecutoff,
                    slope,
                };
            }
        }

        TriposVdwTable { params, rmax: max_rmax, use_4_8 }
    }

    /// Evaluate VDW energy for atom types t1, t2 at squared distance r_sq.
    #[inline(always)]
    pub fn eval(&self, t1: usize, t2: usize, r_sq: f64) -> f64 {
        debug_assert!(t1 < SY_SIZE && t2 < SY_SIZE);
        let p = &self.params[t1 * SY_SIZE + t2];
        if p.kij <= 0.0 || r_sq >= p.rmax_sq {
            return 0.0;
        }
        if r_sq < p.rswitch_sq {
            // Quadratic cap: E = e0 - slope * r²
            // e0 = e_at_switch + slope * rswitch_sq (so it's continuous at rswitch)
            return p.e_at_switch + p.slope * (p.rswitch_sq - r_sq);
        }
        if self.use_4_8 {
            let r4 = r_sq * r_sq;
            let r8 = r4 * r4;
            p.a_coeff / r8 - p.b_coeff / r4
        } else {
            let r6 = r_sq * r_sq * r_sq;
            let r12 = r6 * r6;
            p.a_coeff / r12 - p.b_coeff / r6
        }
    }

    /// Get the VDW radius for a Tripos type (for cavity detection).
    #[inline(always)]
    pub fn radius(t: usize) -> f64 {
        if t < SY_SIZE { TRIPOS_PARAMS[t].radius } else { 0.0 }
    }
}

// Switch distance is now computed analytically in TriposVdwTable::build()
// using C++ rxDock's formula: rswitch = rmin / (1 + sqrt(1 + ecut))^(1/n)

// ─── SDF → Tripos Type Assignment ────────────────────────────────────────────

use crate::atom::*;

/// Assign Tripos (Sybyl) type from element type and bond topology.
///
/// This is used when parsing SDF files, where we have explicit bond orders
/// and can infer hybridization.
pub fn assign_tripos_type(
    el: usize,
    is_aromatic: bool,
    n_heavy_neighbors: usize,
    max_bond_order: u8,
    has_h_neighbor: bool,
    total_bond_order_sum: u8,
) -> usize {
    match el {
        EL_TYPE_C => {
            if is_aromatic { return SY_CAR; }
            match max_bond_order {
                3 => SY_C1,
                2 => SY_C2,
                _ => SY_C3,
            }
        }
        EL_TYPE_N => {
            if is_aromatic { return SY_NAR; }
            match max_bond_order {
                3 => SY_N1,
                2 => {
                    // amide N: sp2 nitrogen bonded to C=O
                    // For simplicity, if total bond order > 3, it's likely N.pl3
                    if total_bond_order_sum >= 4 { SY_NPL3 } else { SY_N2 }
                }
                _ => {
                    if n_heavy_neighbors <= 2 && total_bond_order_sum <= 3 {
                        SY_N3
                    } else {
                        SY_NPL3
                    }
                }
            }
        }
        EL_TYPE_O => {
            if max_bond_order >= 2 {
                SY_O2
            } else if n_heavy_neighbors >= 2 || (!has_h_neighbor && n_heavy_neighbors >= 1) {
                SY_O3
            } else {
                SY_O3
            }
        }
        EL_TYPE_S => {
            if max_bond_order >= 2 { SY_S2 } else { SY_S3 }
        }
        EL_TYPE_P => SY_P3,
        EL_TYPE_F => SY_F,
        EL_TYPE_CL => SY_CL,
        EL_TYPE_BR => SY_BR,
        EL_TYPE_I => SY_I,
        EL_TYPE_H => {
            if has_h_neighbor { SY_HP } else { SY_H }
        }
        EL_TYPE_MET | EL_TYPE_SI => SY_MET,
        _ => SY_C3, // fallback
    }
}

/// Assign Tripos type to an atom based on its AD type (for PDBQT receptor atoms).
pub fn ad_to_tripos(ad: usize) -> usize {
    match ad {
        AD_TYPE_C => SY_C3,
        AD_TYPE_A => SY_CAR,    // aromatic C in PDBQT
        AD_TYPE_N => SY_N3,
        AD_TYPE_O => SY_O3,
        AD_TYPE_P => SY_P3,
        AD_TYPE_S => SY_S3,
        AD_TYPE_H => SY_H,
        AD_TYPE_F => SY_F,
        AD_TYPE_I => SY_I,
        AD_TYPE_NA => SY_NAR,   // nitrogen acceptor → aromatic/sp2
        AD_TYPE_OA => SY_O2,    // oxygen acceptor → sp2 (carbonyl)
        AD_TYPE_SA => SY_S2,    // sulfur acceptor → sp2
        AD_TYPE_HD => SY_HP,    // polar hydrogen
        AD_TYPE_CL => SY_CL,
        AD_TYPE_BR => SY_BR,
        AD_TYPE_MG | AD_TYPE_MN | AD_TYPE_ZN | AD_TYPE_CA | AD_TYPE_FE => SY_MET,
        _ => SY_C3,
    }
}

/// H-bond donor detection based on Tripos type.
#[inline(always)]
pub fn sy_is_hbd(sy: usize) -> bool {
    sy == SY_HP
}

/// H-bond acceptor detection based on Tripos type.
#[inline(always)]
pub fn sy_is_hba(sy: usize) -> bool {
    matches!(sy, SY_N3 | SY_N2 | SY_N1 | SY_NAR | SY_NAM | SY_NPL3 |
                 SY_O3 | SY_O2 | SY_OCO2 | SY_S3 | SY_S2 | SY_F)
}

/// H-bond donor detection based on AutoDock type (for PDBQT receptor atoms).
/// Only AD_TYPE_HD (polar hydrogen bonded to N/O/S) is a donor.
#[inline(always)]
pub fn ad_is_polar_hbd(ad: usize) -> bool {
    ad == AD_TYPE_HD
}

/// H-bond acceptor detection based on AutoDock type (for PDBQT receptor atoms).
/// Only AD_TYPE_OA (oxygen acceptor), AD_TYPE_NA (nitrogen acceptor),
/// AD_TYPE_SA (sulfur acceptor) are acceptors.
#[inline(always)]
pub fn ad_is_polar_hba(ad: usize) -> bool {
    matches!(ad, AD_TYPE_OA | AD_TYPE_NA | AD_TYPE_SA)
}

/// Check if a Tripos type is hydrogen (for exclusion from VDW scoring).
#[inline(always)]
pub fn sy_is_hydrogen(sy: usize) -> bool {
    sy == SY_H || sy == SY_HP
}

/// Normalize Tripos type for dihedral parameter lookup.
#[inline(always)]
pub fn sy_normalize_dihedral(sy: usize) -> usize {
    match sy {
        SY_NAR => SY_N2,    // aromatic N → N.2
        SY_HP => SY_H,      // polar H → H
        SY_OCO2 => SY_O2,   // carboxylate O → O.2
        _ => sy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vdw_table_4_8() {
        // rmax_factor=1.5, ecut=120 (C++ rxDock defaults)
        let table = TriposVdwTable::new(true, 1.5, 120.0);

        // At equilibrium distance (rmin = ri + rj), energy should be -kij = -sqrt(ei*ej)
        let ri = TRIPOS_PARAMS[SY_C3].radius;
        let rj = TRIPOS_PARAMS[SY_C3].radius;
        let rmin = ri + rj; // 3.4 Å
        let kij = TRIPOS_PARAMS[SY_C3].depth; // sqrt(0.107 * 0.107) = 0.107
        let r_sq = rmin * rmin;

        let e = table.eval(SY_C3, SY_C3, r_sq);
        assert!((e + kij).abs() < 0.001, "E at rmin should be -{kij}: got {e}");
    }

    #[test]
    fn test_vdw_beyond_cutoff() {
        let table = TriposVdwTable::new(true, 1.5, 120.0);
        // Beyond rmin * rmax_factor = 3.4 * 1.5 = 5.1 → 6.0² = 36 is beyond
        let e = table.eval(SY_C3, SY_C3, 6.0 * 6.0);
        assert_eq!(e, 0.0, "Energy beyond cutoff should be 0");
    }

    #[test]
    fn test_vdw_short_range_cap() {
        let table = TriposVdwTable::new(true, 1.5, 120.0);
        // Very close distance — should be capped, not infinite
        let e = table.eval(SY_C3, SY_C3, 0.5 * 0.5);
        assert!(e.is_finite(), "Energy at very short range should be finite");
        assert!(e > 0.0, "Energy at very short range should be positive (repulsive)");
    }

    #[test]
    fn test_vdw_ecut_per_pair() {
        let table = TriposVdwTable::new(true, 1.5, 120.0);
        // Different atom type pairs should have different ecut values
        let p_cc = &table.params[SY_C3 * SY_SIZE + SY_C3];
        let p_oo = &table.params[SY_O3 * SY_SIZE + SY_O3];
        // e_at_switch = kij * ecut, kij differs for C-C vs O-O
        assert!((p_cc.e_at_switch - p_cc.kij * 120.0).abs() < 0.01,
            "C-C ecut should be kij*120");
        assert!((p_oo.e_at_switch - p_oo.kij * 120.0).abs() < 0.01,
            "O-O ecut should be kij*120");
        assert!((p_cc.e_at_switch - p_oo.e_at_switch).abs() > 0.01,
            "Different pairs should have different ecut values");
    }

    #[test]
    fn test_hbd_hba_exclusion() {
        let table = TriposVdwTable::new(true, 1.5, 120.0);
        // HBD (SY_HP=22) to HBA (SY_O3=11) should have kij=0
        let p = &table.params[SY_HP * SY_SIZE + SY_O3];
        assert_eq!(p.kij, 0.0, "HBD-HBA pair should have zero VDW");
        let e = table.eval(SY_HP, SY_O3, 2.0 * 2.0);
        assert_eq!(e, 0.0, "HBD-HBA VDW energy should be 0");
    }

    #[test]
    fn test_ad_to_tripos() {
        assert_eq!(ad_to_tripos(AD_TYPE_C), SY_C3);
        assert_eq!(ad_to_tripos(AD_TYPE_A), SY_CAR);
        assert_eq!(ad_to_tripos(AD_TYPE_HD), SY_HP);
        assert_eq!(ad_to_tripos(AD_TYPE_ZN), SY_MET);
    }
}
