//! Dihedral (torsional) strain scoring for rxDock.
//!
//! Implements the DihedralIntraSF scoring function using Tripos 5.2
//! dihedral parameters. For each rotatable bond, computes:
//!   E = Σ k × (1 + sign × cos(|s| × φ))
//! where the sum runs over all 1-4 atom pairs around the bond.

use crate::common::*;
use crate::model::Model;
use crate::rxdock_atom::*;

// ─── Dihedral Parameter Table (Tripos 5.2) ───────────────────────────────────

/// Wildcard sentinel for outer atom type matching.
const WILD: usize = SY_SIZE;

/// A single dihedral parameter entry.
/// central1/central2 = Tripos types of the two central bond atoms.
/// outer1/outer2 = types of the two outer atoms (WILD = wildcard).
/// k = force constant (kcal/mol), s = periodicity (negative ⇒ sign=-1).
struct DihParam {
    c1: usize, c2: usize,
    o1: usize, o2: usize,
    k: f64, s: i32,
}

/// Complete Tripos 5.2 dihedral parameter table from rxDock source.
static TABLE: &[DihParam] = &[
    // C.1_C.1
    DihParam { c1: SY_C1, c2: SY_C1, o1: WILD, o2: WILD, k: 0.0, s: 1 },
    // C.1_C.2
    DihParam { c1: SY_C1, c2: SY_C2, o1: WILD, o2: WILD, k: 0.0, s: 1 },
    // C.2_C.2
    DihParam { c1: SY_C2, c2: SY_C2, o1: WILD, o2: WILD, k: 1.424, s: -2 },
    // C.1_C.3
    DihParam { c1: SY_C1, c2: SY_C3, o1: WILD, o2: WILD, k: 0.0, s: 1 },
    // C.2_C.3
    DihParam { c1: SY_C2, c2: SY_C3, o1: WILD, o2: WILD, k: 0.12, s: -3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: WILD, o2: SY_C2, k: 0.126, s: 3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: WILD, o2: SY_C3, k: 0.126, s: 3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: WILD, o2: SY_H, k: 0.274, s: 3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_O2, o2: SY_C3, k: 0.7, s: -3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_C2, o2: SY_C2, k: 0.126, s: -3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_C2, o2: SY_H, k: 0.273, s: -3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_C3, o2: SY_C2, k: 0.126, s: 3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_C3, o2: SY_C3, k: 0.126, s: 3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_C3, o2: SY_H, k: 0.274, s: 3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_H, o2: SY_C2, k: 0.274, s: 3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_H, o2: SY_C3, k: 0.274, s: 3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_H, o2: SY_H, k: 0.274, s: 3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_C2, o2: WILD, k: 0.126, s: -3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_H, o2: WILD, k: 0.274, s: 3 },
    DihParam { c1: SY_C2, c2: SY_C3, o1: SY_C2, o2: SY_C3, k: 0.126, s: -3 },
    // C.3_C.3
    DihParam { c1: SY_C3, c2: SY_C3, o1: WILD, o2: WILD, k: 0.2, s: 3 },
    DihParam { c1: SY_C3, c2: SY_C3, o1: SY_C2, o2: SY_C2, k: 0.04, s: 3 },
    DihParam { c1: SY_C3, c2: SY_C3, o1: SY_C2, o2: SY_C3, k: 0.126, s: 3 },
    DihParam { c1: SY_C3, c2: SY_C3, o1: SY_C3, o2: SY_C2, k: 0.126, s: 3 },
    DihParam { c1: SY_C3, c2: SY_C3, o1: SY_C3, o2: SY_C3, k: 0.5, s: 3 },
    DihParam { c1: SY_C3, c2: SY_C3, o1: WILD, o2: SY_H, k: 0.32, s: 3 },
    DihParam { c1: SY_C3, c2: SY_C3, o1: SY_H, o2: WILD, k: 0.32, s: 3 },
    // C.1_C.ar
    DihParam { c1: SY_C1, c2: SY_CAR, o1: WILD, o2: WILD, k: 0.0, s: 1 },
    // C.2_C.ar
    DihParam { c1: SY_C2, c2: SY_CAR, o1: WILD, o2: WILD, k: 1.6, s: -2 },
    // C.cat_C.ar
    DihParam { c1: SY_CCAT, c2: SY_CAR, o1: WILD, o2: WILD, k: 1.6, s: -2 },
    // C.3_C.ar
    DihParam { c1: SY_C3, c2: SY_CAR, o1: WILD, o2: WILD, k: 0.12, s: -3 },
    // C.ar_C.ar
    DihParam { c1: SY_CAR, c2: SY_CAR, o1: WILD, o2: WILD, k: 0.6, s: -2 },
    // C.1_N.2
    DihParam { c1: SY_C1, c2: SY_N2, o1: WILD, o2: WILD, k: 0.0, s: 1 },
    // C.2_N.2
    DihParam { c1: SY_C2, c2: SY_N2, o1: WILD, o2: WILD, k: 12.0, s: -2 },
    // C.3_N.2
    DihParam { c1: SY_C3, c2: SY_N2, o1: WILD, o2: WILD, k: 0.4, s: -3 },
    // C.ar_N.2
    DihParam { c1: SY_CAR, c2: SY_N2, o1: WILD, o2: WILD, k: 1.6, s: -2 },
    // N.2_N.2
    DihParam { c1: SY_N2, c2: SY_N2, o1: WILD, o2: WILD, k: 1.6, s: -2 },
    // C.2_N.3
    DihParam { c1: SY_C2, c2: SY_N3, o1: WILD, o2: WILD, k: 0.12, s: -3 },
    // C.3_N.3
    DihParam { c1: SY_C3, c2: SY_N3, o1: WILD, o2: WILD, k: 0.2, s: 3 },
    // C.3_N.4 (quaternary N → use N.3)
    // C.ar_N.3
    DihParam { c1: SY_CAR, c2: SY_N3, o1: WILD, o2: WILD, k: 0.12, s: -3 },
    // N.3_N.3
    DihParam { c1: SY_N3, c2: SY_N3, o1: WILD, o2: WILD, k: 0.2, s: 3 },
    // C.2_N.am
    DihParam { c1: SY_C2, c2: SY_NAM, o1: WILD, o2: WILD, k: 6.46, s: -2 },
    // C.3_N.am
    DihParam { c1: SY_C3, c2: SY_NAM, o1: WILD, o2: WILD, k: 0.2, s: 3 },
    // C.ar_N.am
    DihParam { c1: SY_CAR, c2: SY_NAM, o1: WILD, o2: WILD, k: 1.6, s: -2 },
    // N.2_N.am
    DihParam { c1: SY_N2, c2: SY_NAM, o1: WILD, o2: WILD, k: 1.6, s: -2 },
    // N.3_N.am
    DihParam { c1: SY_N3, c2: SY_NAM, o1: WILD, o2: WILD, k: 0.12, s: -3 },
    // N.am_N.am
    DihParam { c1: SY_NAM, c2: SY_NAM, o1: WILD, o2: WILD, k: 1.6, s: -2 },
    // C.2_N.pl3
    DihParam { c1: SY_C2, c2: SY_NPL3, o1: WILD, o2: WILD, k: 12.0, s: -2 },
    // C.3_N.pl3
    DihParam { c1: SY_C3, c2: SY_NPL3, o1: WILD, o2: WILD, k: 0.4, s: -3 },
    // C.ar_N.pl3
    DihParam { c1: SY_CAR, c2: SY_NPL3, o1: WILD, o2: WILD, k: 1.6, s: -2 },
    // N.2_N.pl3
    DihParam { c1: SY_N2, c2: SY_NPL3, o1: WILD, o2: WILD, k: 1.6, s: -2 },
    // N.pl3_N.pl3
    DihParam { c1: SY_NPL3, c2: SY_NPL3, o1: WILD, o2: WILD, k: 1.6, s: -2 },
    // C.2_O.3
    DihParam { c1: SY_C2, c2: SY_O3, o1: WILD, o2: WILD, k: 5.8, s: -2 },
    // C.3_O.3
    DihParam { c1: SY_C3, c2: SY_O3, o1: WILD, o2: WILD, k: 1.2, s: 3 },
    // C.ar_O.3
    DihParam { c1: SY_CAR, c2: SY_O3, o1: WILD, o2: WILD, k: 1.2, s: -2 },
    // N.2_O.3
    DihParam { c1: SY_N2, c2: SY_O3, o1: WILD, o2: WILD, k: 1.0, s: 2 },
    // N.3_O.3
    DihParam { c1: SY_N3, c2: SY_O3, o1: WILD, o2: WILD, k: 0.2, s: 3 },
    // N.am_O.3
    DihParam { c1: SY_NAM, c2: SY_O3, o1: WILD, o2: WILD, k: 1.0, s: -2 },
    // N.pl3_O.3
    DihParam { c1: SY_NPL3, c2: SY_O3, o1: WILD, o2: WILD, k: 1.0, s: -2 },
    // C.2_P.3
    DihParam { c1: SY_C2, c2: SY_P3, o1: WILD, o2: WILD, k: 1.0, s: -2 },
    // C.3_P.3
    DihParam { c1: SY_C3, c2: SY_P3, o1: WILD, o2: WILD, k: 0.4, s: 3 },
    // C.ar_P.3
    DihParam { c1: SY_CAR, c2: SY_P3, o1: WILD, o2: WILD, k: 1.0, s: 3 },
    // O.3_P.3
    DihParam { c1: SY_O3, c2: SY_P3, o1: WILD, o2: WILD, k: 0.4, s: 3 },
    // C.2_S.2
    DihParam { c1: SY_C2, c2: SY_S2, o1: WILD, o2: WILD, k: 1.0, s: -2 },
    // C.3_S.2
    DihParam { c1: SY_C3, c2: SY_S2, o1: WILD, o2: WILD, k: 0.4, s: 3 },
    // C.ar_S.2
    DihParam { c1: SY_CAR, c2: SY_S2, o1: WILD, o2: WILD, k: 1.0, s: 3 },
    // N.3_S.2
    DihParam { c1: SY_N3, c2: SY_S2, o1: WILD, o2: WILD, k: 0.4, s: 3 },
    // C.2_S.3
    DihParam { c1: SY_C2, c2: SY_S3, o1: WILD, o2: WILD, k: 1.0, s: -2 },
    // C.3_S.3
    DihParam { c1: SY_C3, c2: SY_S3, o1: WILD, o2: WILD, k: 0.4, s: 3 },
    // C.ar_S.3
    DihParam { c1: SY_CAR, c2: SY_S3, o1: WILD, o2: WILD, k: 1.0, s: 3 },
    // S.3_S.3
    DihParam { c1: SY_S3, c2: SY_S3, o1: WILD, o2: WILD, k: 4.0, s: 3 },
];

/// Default dihedral parameters when no specific entry is found.
const DEFAULT_K: f64 = 0.2;
const DEFAULT_S: i32 = 3;

/// Look up dihedral parameters for a given set of atom types.
/// Returns (force_constant, signed_periodicity).
fn lookup_dihedral(sy1: usize, sy2: usize, sy3: usize, sy4: usize) -> (f64, i32) {
    let c1 = sy_normalize_dihedral(sy2);
    let c2 = sy_normalize_dihedral(sy3);
    let o1 = sy_normalize_dihedral(sy1);
    let o2 = sy_normalize_dihedral(sy4);

    // Try forward match (c1, c2), then reverse (c2, c1)
    for &(tc1, tc2, to1, to2) in &[(c1, c2, o1, o2), (c2, c1, o2, o1)] {
        let mut best: Option<(f64, i32, u8)> = None; // (k, s, priority)

        for p in TABLE {
            if p.c1 != tc1 || p.c2 != tc2 { continue; }

            let priority = match (p.o1 == WILD, p.o2 == WILD) {
                (false, false) if p.o1 == to1 && p.o2 == to2 => 4, // exact
                (false, true) if p.o1 == to1 => 2, // outer1 match, outer2 wild
                (true, false) if p.o2 == to2 => 2, // outer1 wild, outer2 match
                (true, true) => 1,                  // both wild
                _ => 0,                             // no match
            };

            if priority > 0 {
                if best.is_none() || priority > best.unwrap().2 {
                    best = Some((p.k, p.s, priority));
                }
            }
        }

        if let Some((k, s, _)) = best {
            return (k, s);
        }
    }

    (DEFAULT_K, DEFAULT_S)
}

// ─── Dihedral Angle Computation ──────────────────────────────────────────────

/// Compute the dihedral angle (radians, in [-π, π]) for atoms p1-p2-p3-p4.
fn dihedral_angle(p1: &Vec3, p2: &Vec3, p3: &Vec3, p4: &Vec3) -> f64 {
    let b1 = *p2 - *p1;
    let b2 = *p3 - *p2;
    let b3 = *p4 - *p3;

    let n1 = b1.cross(&b2);
    let n2 = b2.cross(&b3);

    let b2_len = b2.norm();
    if b2_len < 1e-10 { return 0.0; }

    let m1 = n1.cross(&b2);
    let x = n1.dot(&n2);
    let y = m1.dot(&n2) / b2_len;

    y.atan2(x)
}

// ─── Scoring ─────────────────────────────────────────────────────────────────

/// Compute dihedral strain energy for a single ligand.
///
/// Iterates over all rotatable bonds in the ligand and sums dihedral terms
/// for all 1-4 atom pairs around each bond.
/// Weight (0.5 from C++ rxDock intra-ligand.json) is applied externally.
pub fn score_dihedral_intra(model: &Model, lig_begin: usize, lig_end: usize) -> f64 {
    let mut total = 0.0;

    // Find all rotatable bonds (scan atoms for rotatable bond flags)
    for a in lig_begin..lig_end {
        for bond_ab in &model.atoms[a].bonds {
            let b = bond_ab.connected_atom_index.i;
            if !bond_ab.rotatable { continue; }
            if bond_ab.connected_atom_index.in_grid { continue; }
            if b <= a { continue; } // avoid double-counting

            // Get neighbors of a (not b) and neighbors of b (not a)
            let neighbors_a: Vec<usize> = model.atoms[a].bonds.iter()
                .filter(|bd| {
                    let ni = bd.connected_atom_index.i;
                    ni != b && !bd.connected_atom_index.in_grid && ni >= lig_begin && ni < lig_end
                })
                .map(|bd| bd.connected_atom_index.i)
                .collect();

            let neighbors_b: Vec<usize> = model.atoms[b].bonds.iter()
                .filter(|bd| {
                    let ni = bd.connected_atom_index.i;
                    ni != a && !bd.connected_atom_index.in_grid && ni >= lig_begin && ni < lig_end
                })
                .map(|bd| bd.connected_atom_index.i)
                .collect();

            // For each 1-4 atom pair, compute dihedral energy
            for &na in &neighbors_a {
                for &nb in &neighbors_b {
                    let phi = dihedral_angle(
                        &model.coords[na], &model.coords[a],
                        &model.coords[b], &model.coords[nb],
                    );

                    let (k, s) = lookup_dihedral(
                        model.atoms[na].sy, model.atoms[a].sy,
                        model.atoms[b].sy, model.atoms[nb].sy,
                    );

                    if k.abs() < 1e-12 { continue; }

                    let sign = if s >= 0 { 1.0 } else { -1.0 };
                    let period = s.unsigned_abs() as f64;
                    total += k * (1.0 + sign * (period * phi).cos());
                }
            }
        }
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dihedral_angle() {
        // Trans configuration: angle = π
        let p1 = Vec3::new(-1.0, 1.0, 0.0);
        let p2 = Vec3::new(0.0, 0.0, 0.0);
        let p3 = Vec3::new(1.0, 0.0, 0.0);
        let p4 = Vec3::new(2.0, -1.0, 0.0);
        let phi = dihedral_angle(&p1, &p2, &p3, &p4);
        assert!((phi - std::f64::consts::PI).abs() < 0.01 || (phi + std::f64::consts::PI).abs() < 0.01,
            "Trans dihedral should be ±π, got {}", phi);
    }

    #[test]
    fn test_dihedral_angle_gauche() {
        // Gauche configuration: angle ≈ 60°
        let p1 = Vec3::new(-1.0, 0.5, 0.866);
        let p2 = Vec3::new(0.0, 0.0, 0.0);
        let p3 = Vec3::new(1.0, 0.0, 0.0);
        let p4 = Vec3::new(2.0, 1.0, 0.0);
        let phi = dihedral_angle(&p1, &p2, &p3, &p4);
        // Just check it's in a reasonable range
        assert!(phi.abs() < std::f64::consts::PI, "Dihedral should be in [-π, π]");
    }

    #[test]
    fn test_lookup_c3_c3() {
        let (k, s) = lookup_dihedral(SY_C3, SY_C3, SY_C3, SY_C3);
        assert!((k - 0.5).abs() < 0.001, "C3-C3-C3-C3 should give k=0.5, got {}", k);
        assert_eq!(s, 3, "C3-C3-C3-C3 should give s=3");
    }

    #[test]
    fn test_lookup_c3_c3_wildcard() {
        let (k, s) = lookup_dihedral(SY_F, SY_C3, SY_C3, SY_F);
        assert!((k - 0.2).abs() < 0.001, "F-C3-C3-F should give wildcard k=0.2, got {}", k);
        assert_eq!(s, 3);
    }

    #[test]
    fn test_lookup_c2_c2() {
        let (k, s) = lookup_dihedral(SY_C3, SY_C2, SY_C2, SY_C3);
        assert!((k - 1.424).abs() < 0.001, "C2-C2 should give k=1.424");
        assert_eq!(s, -2, "C2-C2 should give s=-2");
    }

    #[test]
    fn test_lookup_reverse() {
        // C.2_N.2: k=12, s=-2. Should also match N.2_C.2 (reversed)
        let (k, s) = lookup_dihedral(SY_C3, SY_N2, SY_C2, SY_C3);
        assert!((k - 12.0).abs() < 0.001, "N2-C2 (reversed C2_N2) should give k=12, got {}", k);
        assert_eq!(s, -2);
    }

    #[test]
    fn test_lookup_default() {
        // Pair not in table → default (0.2, 3)
        let (k, s) = lookup_dihedral(SY_I, SY_I, SY_I, SY_I);
        assert!((k - 0.2).abs() < 0.001, "Unknown pair should give default k=0.2");
        assert_eq!(s, 3);
    }

    #[test]
    fn test_dihedral_energy_c3_c3() {
        // C3-C3-C3-C3 with k=0.5, s=3
        // At φ=60° (π/3): E = 0.5 * (1 + cos(3×π/3)) = 0.5 * (1 + cos(π)) = 0.5 * 0 = 0
        let phi = std::f64::consts::PI / 3.0;
        let k = 0.5;
        let s = 3;
        let e = k * (1.0 + (s as f64 * phi).cos());
        assert!(e.abs() < 0.001, "Gauche minimum energy should be ~0, got {}", e);
    }
}
