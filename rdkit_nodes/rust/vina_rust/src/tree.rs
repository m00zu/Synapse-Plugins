use crate::common::*;
use crate::conf::*;

// ─── Frame ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Frame {
    pub origin: Vec3,
    pub orientation_q: Quaternion,
    pub orientation_m: Mat3,
}

impl Frame {
    pub fn new() -> Self {
        Frame {
            origin: Vec3::ZERO,
            orientation_q: Quaternion::IDENTITY,
            orientation_m: Mat3::IDENTITY,
        }
    }

    #[inline(always)]
    pub fn local_to_lab(&self, local_coords: &Vec3) -> Vec3 {
        self.origin + self.orientation_m.mul_vec(local_coords)
    }

    #[inline(always)]
    pub fn local_to_lab_direction(&self, local_dir: &Vec3) -> Vec3 {
        self.orientation_m.mul_vec(local_dir)
    }
}

impl Default for Frame {
    fn default() -> Self { Self::new() }
}

// ─── AtomRange ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct AtomRange {
    pub begin: usize,
    pub end: usize,
}

impl AtomRange {
    pub fn new(begin: usize, end: usize) -> Self {
        AtomRange { begin, end }
    }

    pub fn empty() -> Self {
        AtomRange { begin: 0, end: 0 }
    }
}

// ─── Segment (rotatable bond node) ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Segment {
    pub frame: Frame,
    pub atom_range: AtomRange,
    pub relative_axis: Vec3,
    pub relative_origin: Vec3,
    pub axis: Vec3,         // in lab frame
    pub children: Vec<Segment>,
}

impl Segment {
    pub fn new(relative_axis: Vec3, relative_origin: Vec3, atom_range: AtomRange) -> Self {
        Segment {
            frame: Frame::new(),
            atom_range,
            relative_axis,
            relative_origin,
            axis: Vec3::ZERO,
            children: Vec::new(),
        }
    }

    /// Forward kinematics: set coordinates from parent frame + torsion angle
    pub fn set_conf(
        &mut self,
        parent: &Frame,
        atoms: &[Vec3],     // internal (relative) coordinates
        coords: &mut [Vec3], // output lab coordinates
        torsion_iter: &mut std::slice::Iter<f64>,
    ) {
        let torsion = *torsion_iter.next().unwrap();

        // Transform to lab frame
        self.frame.origin = parent.local_to_lab(&self.relative_origin);
        self.axis = parent.local_to_lab_direction(&self.relative_axis);

        // Apply rotation about axis
        let rotation_q = Quaternion::from_axis_angle(&self.axis, torsion);
        self.frame.orientation_q = rotation_q.mul(&parent.orientation_q);
        self.frame.orientation_q.normalize_approx();
        self.frame.orientation_m = self.frame.orientation_q.to_mat3();

        // Set atom coordinates
        for i in self.atom_range.begin..self.atom_range.end {
            coords[i] = self.frame.local_to_lab(&atoms[i]);
        }

        // Recurse into children
        for child in &mut self.children {
            child.set_conf(&self.frame, atoms, coords, torsion_iter);
        }
    }

    /// Reverse kinematics: compute force/torque and torsion derivatives
    pub fn derivative(
        &self,
        coords: &[Vec3],
        forces: &[Vec3],
        torsion_deriv_iter: &mut std::slice::IterMut<f64>,
    ) -> (Vec3, Vec3) {
        // Sum force and torque from children first
        let mut total_force = Vec3::ZERO;
        let mut total_torque = Vec3::ZERO;

        for child in &self.children {
            let (child_force, child_torque) = child.derivative(coords, forces, torsion_deriv_iter);
            total_force += child_force;
            // Propagate torque: add child torque + (child_origin - our_origin) × child_force
            let r = child.frame.origin - self.frame.origin;
            total_torque += child_torque + r.cross(&child_force);
        }

        // Add force/torque from our own atoms
        for i in self.atom_range.begin..self.atom_range.end {
            let force = forces[i];
            total_force += force;
            let r = coords[i] - self.frame.origin;
            total_torque += r.cross(&force);
        }

        // Torsion derivative = torque projected onto axis
        let torsion_deriv = torsion_deriv_iter.next().unwrap();
        *torsion_deriv = total_torque.dot(&self.axis);

        (total_force, total_torque)
    }
}

// ─── RigidBody (root frame) ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RigidBody {
    pub frame: Frame,
    pub atom_range: AtomRange,
    pub children: Vec<Segment>,
}

impl RigidBody {
    pub fn new(atom_range: AtomRange) -> Self {
        RigidBody {
            frame: Frame::new(),
            atom_range,
            children: Vec::new(),
        }
    }

    /// Forward kinematics: set coordinates from conformation
    pub fn set_conf(
        &mut self,
        atoms: &[Vec3],      // internal (relative) coordinates
        coords: &mut [Vec3],  // output lab coordinates
        conf: &LigandConf,
    ) {
        // Set root frame from rigid conformation
        self.frame.origin = conf.rigid.position;
        self.frame.orientation_q = conf.rigid.orientation;
        self.frame.orientation_m = conf.rigid.orientation.to_mat3();

        // Set atom coordinates
        for i in self.atom_range.begin..self.atom_range.end {
            coords[i] = self.frame.local_to_lab(&atoms[i]);
        }

        // Set children torsions
        let mut torsion_iter = conf.torsions.iter();
        for child in &mut self.children {
            child.set_conf(&self.frame, atoms, coords, &mut torsion_iter);
        }
    }

    /// Reverse kinematics: compute gradient from forces
    pub fn derivative(
        &self,
        coords: &[Vec3],
        forces: &[Vec3],
        change: &mut LigandChange,
    ) {
        let mut total_force = Vec3::ZERO;
        let mut total_torque = Vec3::ZERO;

        // Process children — collect torsion derivatives into the torsions slice
        let mut torsion_offset = change.torsions.len();
        for child in self.children.iter().rev() {
            fn deriv_segment(seg: &Segment, coords: &[Vec3], forces: &[Vec3], torsions: &mut [f64], offset: &mut usize) -> (Vec3, Vec3) {
                let mut tf = Vec3::ZERO;
                let mut tt = Vec3::ZERO;
                for c in seg.children.iter().rev() {
                    let (cf, ct) = deriv_segment(c, coords, forces, torsions, offset);
                    tf += cf;
                    let r = c.frame.origin - seg.frame.origin;
                    tt += ct + r.cross(&cf);
                }
                for i in seg.atom_range.begin..seg.atom_range.end {
                    let f = forces[i];
                    tf += f;
                    let r = coords[i] - seg.frame.origin;
                    tt += r.cross(&f);
                }
                *offset -= 1;
                torsions[*offset] = tt.dot(&seg.axis);
                (tf, tt)
            }
            let (cf, ct) = deriv_segment(child, coords, forces, &mut change.torsions, &mut torsion_offset);
            total_force += cf;
            let r = child.frame.origin - self.frame.origin;
            total_torque += ct + r.cross(&cf);
        }

        // Add force/torque from root atoms
        for i in self.atom_range.begin..self.atom_range.end {
            let force = forces[i];
            total_force += force;
            let r = coords[i] - self.frame.origin;
            total_torque += r.cross(&force);
        }

        // Position derivative = total force
        change.rigid.position = total_force;
        // Orientation derivative = total torque
        change.rigid.orientation = total_torque;
    }
}

// ─── FlexibleResidue (first_segment root) ──────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FlexResidue {
    pub frame: Frame,
    pub atom_range: AtomRange,
    pub axis: Vec3,
    pub children: Vec<Segment>,
}

impl FlexResidue {
    pub fn new(axis: Vec3, atom_range: AtomRange) -> Self {
        FlexResidue {
            frame: Frame::new(),
            atom_range,
            axis,
            children: Vec::new(),
        }
    }

    pub fn set_conf(
        &mut self,
        atoms: &[Vec3],
        coords: &mut [Vec3],
        conf: &ResidueConf,
    ) {
        let mut torsion_iter = conf.torsions.iter();
        let first_torsion = *torsion_iter.next().unwrap_or(&0.0);

        // First segment: rotation about fixed axis from origin
        self.frame.orientation_q = Quaternion::from_axis_angle(&self.axis, first_torsion);
        self.frame.orientation_m = self.frame.orientation_q.to_mat3();

        for i in self.atom_range.begin..self.atom_range.end {
            coords[i] = self.frame.local_to_lab(&atoms[i]);
        }

        for child in &mut self.children {
            child.set_conf(&self.frame, atoms, coords, &mut torsion_iter);
        }
    }

    pub fn derivative(
        &self,
        coords: &[Vec3],
        forces: &[Vec3],
        change: &mut ResidueChange,
    ) {
        let mut total_force = Vec3::ZERO;
        let mut total_torque = Vec3::ZERO;

        // Use indexed approach (same pattern as RigidBody::derivative)
        let mut torsion_offset = change.torsions.len();
        for child in self.children.iter().rev() {
            fn deriv_seg(seg: &Segment, coords: &[Vec3], forces: &[Vec3], torsions: &mut [f64], offset: &mut usize) -> (Vec3, Vec3) {
                let mut tf = Vec3::ZERO;
                let mut tt = Vec3::ZERO;
                for c in seg.children.iter().rev() {
                    let (cf, ct) = deriv_seg(c, coords, forces, torsions, offset);
                    tf += cf;
                    let r = c.frame.origin - seg.frame.origin;
                    tt += ct + r.cross(&cf);
                }
                for i in seg.atom_range.begin..seg.atom_range.end {
                    let f = forces[i];
                    tf += f;
                    let r = coords[i] - seg.frame.origin;
                    tt += r.cross(&f);
                }
                *offset -= 1;
                torsions[*offset] = tt.dot(&seg.axis);
                (tf, tt)
            }
            let (cf, ct) = deriv_seg(child, coords, forces, &mut change.torsions[1..], &mut torsion_offset);
            total_force += cf;
            let r = child.frame.origin - self.frame.origin;
            total_torque += ct + r.cross(&cf);
        }

        for i in self.atom_range.begin..self.atom_range.end {
            let force = forces[i];
            total_force += force;
            let r = coords[i] - self.frame.origin;
            total_torque += r.cross(&force);
        }

        // First torsion derivative
        if !change.torsions.is_empty() {
            change.torsions[0] = total_torque.dot(&self.axis);
        }
    }
}
