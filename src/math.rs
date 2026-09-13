//! Minimal linear algebra types (vectors, 4x4 matrices) for the software renderer.
//!
//! No external crate dependency — small, self-contained, column-major matrices.
use std::ops::{Add, Sub, Mul, Neg};

// ─── Vec2 ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, rhs: Vec2) -> Vec2 { Vec2::new(self.x + rhs.x, self.y + rhs.y) }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, rhs: Vec2) -> Vec2 { Vec2::new(self.x - rhs.x, self.y - rhs.y) }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: f32) -> Vec2 { Vec2::new(self.x * rhs, self.y * rhs) }
}

// ─── Vec3 ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    pub const X: Vec3 = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
    pub const Y: Vec3 = Vec3 { x: 0.0, y: 1.0, z: 0.0 };
    pub const Z: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 1.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    pub fn dot(self, rhs: Vec3) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Vec3) -> Vec3 {
        Vec3::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns a normalized copy, or `self` unchanged when its length is
    /// too small to normalize safely.
    pub fn normalize(self) -> Vec3 {
        let len = self.length();
        if len < 1e-8 {
            self
        } else {
            self * (1.0 / len)
        }
    }

    pub fn min(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    pub fn max(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }

    pub fn abs(self) -> Vec3 {
        Vec3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 { Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z) }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 { Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z) }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Vec3 { Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs) }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 { Vec3::new(-self.x, -self.y, -self.z) }
}

// ─── Vec4 ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }

    pub fn xyz(self) -> Vec3 { Vec3::new(self.x, self.y, self.z) }
}

impl Add for Vec4 {
    type Output = Vec4;
    fn add(self, rhs: Vec4) -> Vec4 { Vec4::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w) }
}

impl Sub for Vec4 {
    type Output = Vec4;
    fn sub(self, rhs: Vec4) -> Vec4 { Vec4::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w) }
}

impl Mul<f32> for Vec4 {
    type Output = Vec4;
    fn mul(self, rhs: f32) -> Vec4 { Vec4::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs) }
}

// ─── Mat4 ───────────────────────────────────────────────────────────────────

/// A 4x4 matrix stored column-major, matching OpenGL conventions:
/// `m[col * 4 + row]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4(pub [f32; 16]);

impl Mat4 {
    pub fn identity() -> Mat4 {
        let mut m = [0.0f32; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        Mat4(m)
    }

    #[inline]
    fn get(&self, col: usize, row: usize) -> f32 {
        self.0[col * 4 + row]
    }

    /// Returns `self * rhs`.
    pub fn mul(&self, rhs: &Mat4) -> Mat4 {
        let mut out = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += self.get(k, row) * rhs.get(col, k);
                }
                out[col * 4 + row] = sum;
            }
        }
        Mat4(out)
    }

    /// Right-handed look-at matrix (view matrix), mapping world space to
    /// eye/view space with the camera looking down -Z.
    pub fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let f = (target - eye).normalize(); // forward
        let s = f.cross(up).normalize();    // right
        let u = s.cross(f);                 // recomputed up

        let mut m = [0.0f32; 16];
        // Column 0
        m[0] = s.x;
        m[1] = u.x;
        m[2] = -f.x;
        m[3] = 0.0;
        // Column 1
        m[4] = s.y;
        m[5] = u.y;
        m[6] = -f.y;
        m[7] = 0.0;
        // Column 2
        m[8] = s.z;
        m[9] = u.z;
        m[10] = -f.z;
        m[11] = 0.0;
        // Column 3 (translation)
        m[12] = -s.dot(eye);
        m[13] = -u.dot(eye);
        m[14] = f.dot(eye);
        m[15] = 1.0;

        Mat4(m)
    }

    /// OpenGL-style right-handed perspective projection with NDC z in [-1, 1].
    pub fn perspective_rh(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let tan_half_fov = (fov_y_radians * 0.5).tan();
        let mut m = [0.0f32; 16];

        m[0] = 1.0 / (aspect * tan_half_fov);
        m[5] = 1.0 / tan_half_fov;
        m[10] = -(far + near) / (far - near);
        m[11] = -1.0;
        m[14] = -(2.0 * far * near) / (far - near);

        Mat4(m)
    }

    /// Transforms a point (implicit w = 1), returning the full homogeneous
    /// result (caller performs the perspective divide if needed).
    pub fn transform_point(&self, p: Vec3) -> Vec4 {
        Vec4::new(
            self.get(0, 0) * p.x + self.get(1, 0) * p.y + self.get(2, 0) * p.z + self.get(3, 0),
            self.get(0, 1) * p.x + self.get(1, 1) * p.y + self.get(2, 1) * p.z + self.get(3, 1),
            self.get(0, 2) * p.x + self.get(1, 2) * p.y + self.get(2, 2) * p.z + self.get(3, 2),
            self.get(0, 3) * p.x + self.get(1, 3) * p.y + self.get(2, 3) * p.z + self.get(3, 3),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_x_y_is_z() {
        assert_eq!(Vec3::X.cross(Vec3::Y), Vec3::Z);
    }

    #[test]
    fn normalize_has_unit_length() {
        let v = Vec3::new(3.0, 4.0, 0.0).normalize();
        assert!((v.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_zero_vector_unchanged() {
        let v = Vec3::ZERO.normalize();
        assert_eq!(v, Vec3::ZERO);
    }

    #[test]
    fn identity_times_vector_is_unchanged() {
        let m = Mat4::identity();
        let p = Vec3::new(1.0, 2.0, 3.0);
        let r = m.transform_point(p);
        assert_eq!(r, Vec4::new(1.0, 2.0, 3.0, 1.0));
    }

    #[test]
    fn perspective_maps_near_and_far_to_ndc_bounds() {
        let near = 0.1f32;
        let far = 100.0f32;
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far);

        let p_near = proj.transform_point(Vec3::new(0.0, 0.0, -near));
        let ndc_near_z = p_near.z / p_near.w;
        assert!((ndc_near_z - (-1.0)).abs() < 1e-4, "near ndc z = {ndc_near_z}");

        let p_far = proj.transform_point(Vec3::new(0.0, 0.0, -far));
        let ndc_far_z = p_far.z / p_far.w;
        assert!((ndc_far_z - 1.0).abs() < 1e-4, "far ndc z = {ndc_far_z}");
    }

    #[test]
    fn look_at_transforms_origin_into_view_space() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, -5.0, 0.0), Vec3::ZERO, Vec3::Z);
        let r = view.transform_point(Vec3::ZERO);
        assert!((r.x - 0.0).abs() < 1e-5, "x = {}", r.x);
        assert!((r.y - 0.0).abs() < 1e-5, "y = {}", r.y);
        assert!((r.z - (-5.0)).abs() < 1e-5, "z = {}", r.z);
    }
}
