//! Cavity detection for rxDock-style docking.
//!
//! Implements the SphereSiteMapper "two-sphere" algorithm as described in
//! the rxDock/rDock reference implementation (Ruiz-Carmona et al. 2014).
//!
//! The algorithm:
//! 1. Build a 3D grid covering the search sphere plus a border region
//! 2. Initialize three zones: excluded (outside), border (ring), empty (inside)
//! 3. Exclude receptor volume (atoms + VDW radii)
//! 4. Rolling-ball test with large probe: mark regions where the large probe fits
//! 5. Collapse all non-empty to receptor, then rolling-ball with small probe
//! 6. Connected components of cavity cells → filter by minimum volume → return sorted

use crate::common::Vec3;

// ─── Parameters ──────────────────────────────────────────────────────────────

/// Configuration for cavity detection.
#[derive(Debug, Clone)]
pub struct CavityParams {
    /// Center of the search sphere.
    pub center: Vec3,
    /// Radius of the search sphere (Å). Default 10.0.
    pub radius: f64,
    /// Added to VDW radii when excluding receptor volume (Å). Default 0.0.
    pub vol_increment: f64,
    /// Small probe radius — defines cavity accessibility (Å). Default 1.5.
    pub small_sphere: f64,
    /// Large probe radius — marks open/solvent-accessible regions (Å). Default 4.0.
    pub large_sphere: f64,
    /// Grid spacing (Å). Default 0.5.
    pub grid_step: f64,
    /// Minimum cavity volume (ų). Default 100.0.
    pub min_volume: f64,
    /// Maximum number of cavities to return. Default 10.
    pub max_cavities: usize,
}

impl Default for CavityParams {
    fn default() -> Self {
        CavityParams {
            center: Vec3::ZERO,
            radius: 10.0,
            vol_increment: 0.0,
            small_sphere: 1.5,
            large_sphere: 4.0,
            grid_step: 0.5,
            min_volume: 100.0,
            max_cavities: 10,
        }
    }
}

// ─── Cavity Result ───────────────────────────────────────────────────────────

/// A detected cavity region.
#[derive(Debug, Clone)]
pub struct Cavity {
    /// Grid point coordinates belonging to this cavity.
    pub coords: Vec<Vec3>,
    /// Volume in ų.
    pub volume: f64,
    /// Geometric center of the cavity.
    pub center: Vec3,
    /// Bounding box minimum corner.
    pub min_coord: Vec3,
    /// Bounding box maximum corner.
    pub max_coord: Vec3,
}

// ─── 3D Grid ─────────────────────────────────────────────────────────────────

/// Grid cell states during cavity detection (matching C++ rxDock values).
const GRID_EMPTY: f64 = 0.0;
const GRID_RECEPTOR: f64 = -1.0;
const GRID_LARGE_ACCESSIBLE: f64 = -0.75;
const GRID_EXCLUDED: f64 = -0.5;
const GRID_BORDER: f64 = -0.25;
const GRID_CAVITY: f64 = 1.0;

/// Tolerance for float comparison of grid values.
const GRID_TOL: f64 = 0.01;

#[inline(always)]
fn grid_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < GRID_TOL
}

/// A 3D grid for cavity detection.
struct CavityGrid {
    data: Vec<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    origin: Vec3,
    step: f64,
}

impl CavityGrid {
    fn new(origin: Vec3, step: f64, nx: usize, ny: usize, nz: usize) -> Self {
        CavityGrid {
            data: vec![GRID_EMPTY; nx * ny * nz],
            nx, ny, nz,
            origin,
            step,
        }
    }

    #[inline(always)]
    fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix * self.ny * self.nz + iy * self.nz + iz
    }

    #[inline(always)]
    fn get(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        self.data[self.idx(ix, iy, iz)]
    }

    #[inline(always)]
    fn set(&mut self, ix: usize, iy: usize, iz: usize, val: f64) {
        let i = self.idx(ix, iy, iz);
        self.data[i] = val;
    }

    /// Convert grid indices to world coordinates.
    #[inline(always)]
    fn grid_to_world(&self, ix: usize, iy: usize, iz: usize) -> Vec3 {
        Vec3::new(
            self.origin.x() + ix as f64 * self.step,
            self.origin.y() + iy as f64 * self.step,
            self.origin.z() + iz as f64 * self.step,
        )
    }

    /// Mark all grid points within `radius` of `center` to `value`.
    /// If `overwrite` is true, overwrites any value. Otherwise only overwrites
    /// cells whose current value is greater than `value`.
    fn set_sphere(&mut self, center: &Vec3, radius: f64, value: f64, overwrite: bool) {
        let r_sq = radius * radius;
        let r_grid = (radius / self.step).ceil() as isize + 1;

        let cx = ((center.x() - self.origin.x()) / self.step).round() as isize;
        let cy = ((center.y() - self.origin.y()) / self.step).round() as isize;
        let cz = ((center.z() - self.origin.z()) / self.step).round() as isize;

        let ix_lo = (cx - r_grid).max(0) as usize;
        let ix_hi = ((cx + r_grid) as usize).min(self.nx - 1);
        let iy_lo = (cy - r_grid).max(0) as usize;
        let iy_hi = ((cy + r_grid) as usize).min(self.ny - 1);
        let iz_lo = (cz - r_grid).max(0) as usize;
        let iz_hi = ((cz + r_grid) as usize).min(self.nz - 1);

        for ix in ix_lo..=ix_hi {
            let dx = self.origin.x() + ix as f64 * self.step - center.x();
            let dx2 = dx * dx;
            if dx2 > r_sq { continue; }
            for iy in iy_lo..=iy_hi {
                let dy = self.origin.y() + iy as f64 * self.step - center.y();
                let dy2 = dy * dy;
                if dx2 + dy2 > r_sq { continue; }
                for iz in iz_lo..=iz_hi {
                    let dz = self.origin.z() + iz as f64 * self.step - center.z();
                    if dx2 + dy2 + dz * dz <= r_sq {
                        if overwrite {
                            self.set(ix, iy, iz, value);
                        } else {
                            let cur = self.get(ix, iy, iz);
                            if cur > value {
                                self.set(ix, iy, iz, value);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Rolling-ball accessibility test (matches C++ RealGrid::SetAccessible).
    ///
    /// For each grid point with value `old_val`:
    ///   - Check all grid cells within `radius` of this point
    ///   - If NONE have value `adj_val` → mark ALL cells in the sphere as `new_val`
    fn set_accessible(&mut self, radius: f64, old_val: f64, adj_val: f64, new_val: f64) {
        let r_grid = (radius / self.step).ceil() as isize;
        let r_sq = radius * radius;

        for ix in 0..self.nx {
            for iy in 0..self.ny {
                for iz in 0..self.nz {
                    if !grid_eq(self.get(ix, iy, iz), old_val) { continue; }

                    // Get all grid indices within sphere of `radius`
                    let sx_lo = (ix as isize - r_grid).max(0) as usize;
                    let sx_hi = (ix as isize + r_grid).min(self.nx as isize - 1) as usize;
                    let sy_lo = (iy as isize - r_grid).max(0) as usize;
                    let sy_hi = (iy as isize + r_grid).min(self.ny as isize - 1) as usize;
                    let sz_lo = (iz as isize - r_grid).max(0) as usize;
                    let sz_hi = (iz as isize + r_grid).min(self.nz as isize - 1) as usize;

                    let mut has_adj = false;
                    let mut sphere_cells: Vec<(usize, usize, usize)> = Vec::new();

                    for sx in sx_lo..=sx_hi {
                        let ddx = (sx as f64 - ix as f64) * self.step;
                        let ddx2 = ddx * ddx;
                        if ddx2 > r_sq { continue; }
                        for sy in sy_lo..=sy_hi {
                            let ddy = (sy as f64 - iy as f64) * self.step;
                            let ddy2 = ddy * ddy;
                            if ddx2 + ddy2 > r_sq { continue; }
                            for sz in sz_lo..=sz_hi {
                                let ddz = (sz as f64 - iz as f64) * self.step;
                                if ddx2 + ddy2 + ddz * ddz <= r_sq {
                                    sphere_cells.push((sx, sy, sz));
                                    if grid_eq(self.get(sx, sy, sz), adj_val) {
                                        has_adj = true;
                                    }
                                }
                            }
                        }
                    }

                    if !has_adj {
                        // Mark ALL cells in the sphere as new_val
                        for (sx, sy, sz) in sphere_cells {
                            self.set(sx, sy, sz, new_val);
                        }
                    }
                }
            }
        }
    }

    /// Count grid cells with a specific value.
    fn count(&self, value: f64) -> usize {
        self.data.iter().filter(|&&v| grid_eq(v, value)).count()
    }

    /// Replace all cells with value `old_val` to `new_val`.
    fn replace_value(&mut self, old_val: f64, new_val: f64) {
        for v in &mut self.data {
            if grid_eq(*v, old_val) {
                *v = new_val;
            }
        }
    }

    /// Find connected components of cells with the given value.
    /// Returns groups of grid indices, sorted by size descending.
    fn find_connected_components(&self, value: f64) -> Vec<Vec<(usize, usize, usize)>> {
        let mut visited = vec![false; self.data.len()];
        let mut components = Vec::new();

        for ix in 0..self.nx {
            for iy in 0..self.ny {
                for iz in 0..self.nz {
                    let i = self.idx(ix, iy, iz);
                    if !grid_eq(self.data[i], value) || visited[i] { continue; }

                    let mut component = Vec::new();
                    let mut queue = std::collections::VecDeque::new();
                    visited[i] = true;
                    queue.push_back((ix, iy, iz));

                    while let Some((cx, cy, cz)) = queue.pop_front() {
                        component.push((cx, cy, cz));
                        let neighbors: [(isize, isize, isize); 6] = [
                            (-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1),
                        ];
                        for (dx, dy, dz) in &neighbors {
                            let nx = cx as isize + dx;
                            let ny = cy as isize + dy;
                            let nz = cz as isize + dz;
                            if nx < 0 || ny < 0 || nz < 0 { continue; }
                            let nx = nx as usize;
                            let ny = ny as usize;
                            let nz = nz as usize;
                            if nx >= self.nx || ny >= self.ny || nz >= self.nz { continue; }

                            let ni = self.idx(nx, ny, nz);
                            if grid_eq(self.data[ni], value) && !visited[ni] {
                                visited[ni] = true;
                                queue.push_back((nx, ny, nz));
                            }
                        }
                    }

                    components.push(component);
                }
            }
        }

        components.sort_by(|a, b| b.len().cmp(&a.len()));
        components
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Receptor atom for cavity detection: just coordinates and VDW radius.
pub struct ReceptorAtom {
    pub coords: Vec3,
    pub vdw_radius: f64,
}

/// Detect cavities in the receptor using the two-sphere site mapper algorithm.
///
/// This faithfully reimplements the C++ rxDock SphereSiteMapper algorithm:
/// 1. Create grid with excluded/border/empty zones
/// 2. Mark receptor VDW volume
/// 3. Rolling-ball test with large probe (border region, then inner region)
/// 4. Collapse non-receptor non-empty to receptor, rolling-ball with small probe
/// 5. Connected components → filter by volume
pub fn detect_cavities(receptor_atoms: &[ReceptorAtom], params: &CavityParams) -> Vec<Cavity> {
    let step = params.grid_step;
    let large_r = params.large_sphere;
    let small_r = params.small_sphere;
    let border = 2.0 * (large_r + step);

    // Grid extents: search sphere + border on each side
    let min_coord = Vec3::new(
        params.center.x() - params.radius - border,
        params.center.y() - params.radius - border,
        params.center.z() - params.radius - border,
    );
    let max_coord = Vec3::new(
        params.center.x() + params.radius + border,
        params.center.y() + params.radius + border,
        params.center.z() + params.radius + border,
    );

    let extent = max_coord - min_coord;
    let nx = (extent.x() / step) as usize + 1;
    let ny = (extent.y() / step) as usize + 1;
    let nz = (extent.z() / step) as usize + 1;

    let mut grid = CavityGrid::new(min_coord, step, nx, ny, nz);

    // Step 1: Initialize grid zones
    // Start everything as excluded
    for v in &mut grid.data {
        *v = GRID_EXCLUDED;
    }

    // The actual grid center may differ slightly from params.center due to discretization
    let grid_center = Vec3::new(
        min_coord.x() + (nx as f64 - 1.0) * step * 0.5,
        min_coord.y() + (ny as f64 - 1.0) * step * 0.5,
        min_coord.z() + (nz as f64 - 1.0) * step * 0.5,
    );

    // Set border region: sphere of radius + large_r around center
    grid.set_sphere(&grid_center, params.radius + large_r, GRID_BORDER, true);
    // Set inner region: sphere of radius around center
    grid.set_sphere(&grid_center, params.radius, GRID_EMPTY, true);

    // Step 2: Mark receptor volume
    // Include ALL receptor atoms (not just those inside search sphere) since
    // atoms outside may have VDW volumes that overlap the active site
    for atom in receptor_atoms {
        let r = atom.vdw_radius + params.vol_increment;
        if r > 0.0 {
            grid.set_sphere(&atom.coords, r, GRID_RECEPTOR, true);
        }
    }

    // Step 3: Large sphere accessibility — border region first (edge effect prevention)
    grid.set_accessible(large_r, GRID_BORDER, GRID_RECEPTOR, GRID_LARGE_ACCESSIBLE);

    // Step 3b: Large sphere accessibility — inner (unallocated) region
    grid.set_accessible(large_r, GRID_EMPTY, GRID_RECEPTOR, GRID_LARGE_ACCESSIBLE);

    // Step 4: Collapse everything non-zero to receptor value
    // This is the key step that prevents edge effects in the small sphere pass
    grid.replace_value(GRID_BORDER, GRID_RECEPTOR);
    grid.replace_value(GRID_EXCLUDED, GRID_RECEPTOR);
    grid.replace_value(GRID_LARGE_ACCESSIBLE, GRID_RECEPTOR);

    // Step 5: Small sphere accessibility on remaining empty cells → cavity
    grid.set_accessible(small_r, GRID_EMPTY, GRID_RECEPTOR, GRID_CAVITY);

    // Step 6: Find connected components of cavity points
    let min_size = (params.min_volume / (step * step * step)) as usize;
    let components = grid.find_connected_components(GRID_CAVITY);

    // Step 7: Convert to Cavity structs, filter by minimum volume
    let voxel_volume = step * step * step;
    let mut cavities = Vec::new();

    for component in &components {
        if component.len() < min_size { continue; }
        if cavities.len() >= params.max_cavities { break; }

        let volume = component.len() as f64 * voxel_volume;

        let coords: Vec<Vec3> = component.iter()
            .map(|&(ix, iy, iz)| grid.grid_to_world(ix, iy, iz))
            .collect();

        let mut center = Vec3::ZERO;
        let mut min_c = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max_c = Vec3::new(f64::MIN, f64::MIN, f64::MIN);

        for pt in &coords {
            center = center + *pt;
            min_c = Vec3::new(
                min_c.x().min(pt.x()),
                min_c.y().min(pt.y()),
                min_c.z().min(pt.z()),
            );
            max_c = Vec3::new(
                max_c.x().max(pt.x()),
                max_c.y().max(pt.y()),
                max_c.z().max(pt.z()),
            );
        }

        let n = coords.len() as f64;
        center = center * (1.0 / n);

        cavities.push(Cavity {
            coords,
            volume,
            center,
            min_coord: min_c,
            max_coord: max_c,
        });
    }

    cavities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_receptor_no_cavities() {
        let params = CavityParams {
            center: Vec3::ZERO,
            radius: 5.0,
            min_volume: 1.0,
            ..Default::default()
        };
        let cavities = detect_cavities(&[], &params);
        // No receptor → no enclosed space → no cavities
        assert!(cavities.is_empty());
    }

    #[test]
    fn test_hollow_sphere_has_cavity() {
        // Create a hollow sphere of atoms: shell at radius 5.0
        let mut atoms = Vec::new();
        let r = 5.0;
        let n = 30;
        for i in 0..n {
            let phi = std::f64::consts::PI * i as f64 / (n - 1) as f64;
            let n_ring = (n as f64 * phi.sin()).max(1.0) as usize;
            for j in 0..n_ring {
                let theta = 2.0 * std::f64::consts::PI * j as f64 / n_ring as f64;
                atoms.push(ReceptorAtom {
                    coords: Vec3::new(
                        r * phi.sin() * theta.cos(),
                        r * phi.sin() * theta.sin(),
                        r * phi.cos(),
                    ),
                    vdw_radius: 1.5,
                });
            }
        }

        let params = CavityParams {
            center: Vec3::ZERO,
            radius: 8.0,
            min_volume: 5.0,
            grid_step: 0.8,
            ..Default::default()
        };

        let cavities = detect_cavities(&atoms, &params);
        assert!(!cavities.is_empty(), "Hollow sphere should have a cavity");
        assert!(cavities[0].volume > 5.0, "Cavity should have meaningful volume");
    }
}
