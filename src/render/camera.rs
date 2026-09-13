//! Camera placement: fits a drawable's bounds into one of the fixed views.

use crate::math::{Mat4, Vec3};
use crate::render::View;
use crate::ydd::DrawableBounds;

/// Direction the light travels *toward* — i.e. the vector used in `dot(n, l)`.
const LIGHT_DIR: Vec3 = Vec3 { x: 0.4, y: -0.6, z: 0.8 };

/// Builds the view-projection matrix that frames `bounds` from `view`.
///
/// The world is Z-up with +Y forward. Returns the combined matrix, the eye
/// position and the normalized light direction.
pub(crate) fn camera_for(
    bounds: &DrawableBounds,
    view: View,
    aspect: f32,
    fov_deg: f32,
    margin: f32,
) -> (Mat4, Vec3, Vec3) {
    let aspect = if aspect.is_finite() && aspect > 1e-4 { aspect } else { 1.0 };
    let margin = if margin.is_finite() && margin > 0.0 { margin } else { 1.0 };
    let fov = fov_deg.clamp(1.0, 170.0).to_radians();

    let center = bounds.center;
    let half_diagonal = (bounds.box_max - bounds.box_min).length() * 0.5;
    let mut radius = bounds.sphere_radius.max(half_diagonal).max(1e-3);
    if !radius.is_finite() {
        radius = 1.0;
    }

    let mut distance = radius / (fov * 0.5).sin() * margin;
    if aspect < 1.0 {
        distance /= aspect;
    }

    let direction = match view {
        View::Front => Vec3::new(0.0, 1.0, 0.0),
        View::Back => Vec3::new(0.0, -1.0, 0.0),
        View::Left => Vec3::new(-1.0, 0.0, 0.0),
        View::Right => Vec3::new(1.0, 0.0, 0.0),
        View::Top => Vec3::new(0.0, 0.0, 1.0),
        View::Iso => Vec3::new(1.0, -1.0, 1.0),
    }
    .normalize();

    let eye = center + direction * distance;
    let up = if view == View::Top { Vec3::Y } else { Vec3::Z };

    let near = (distance - 2.0 * radius).max(0.01);
    let far = distance + 2.0 * radius;

    let projection = Mat4::perspective_rh(fov, aspect, near, far);
    let look = Mat4::look_at_rh(eye, center, up);

    (projection.mul(&look), eye, LIGHT_DIR.normalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec4;

    fn bounds() -> DrawableBounds {
        DrawableBounds {
            center: Vec3::ZERO,
            sphere_radius: 1.0,
            box_min: Vec3::new(-1.0, -1.0, -1.0),
            box_max: Vec3::new(1.0, 1.0, 1.0),
        }
    }

    fn ndc(m: &Mat4, p: Vec3) -> Vec4 {
        let c = m.transform_point(p);
        Vec4::new(c.x / c.w, c.y / c.w, c.z / c.w, c.w)
    }

    #[test]
    fn every_view_places_the_eye_outside_the_bounds() {
        for view in View::ALL {
            let (_, eye, _) = camera_for(&bounds(), view, 1.0, 40.0, 1.1);
            assert!(eye.length() > 1.0, "{view}: eye inside bounds at {eye:?}");
        }
    }

    #[test]
    fn center_projects_in_front_of_the_camera_and_inside_ndc() {
        for view in View::ALL {
            let (view_proj, _, _) = camera_for(&bounds(), view, 1.0, 40.0, 1.1);
            let p = ndc(&view_proj, Vec3::ZERO);
            assert!(p.w > 0.0, "{view}: centre behind the camera");
            assert!(p.x.abs() < 1e-4 && p.y.abs() < 1e-4, "{view}: centre off-screen {p:?}");
            assert!(p.z > -1.0 && p.z < 1.0, "{view}: centre outside the depth range");
        }
    }

    #[test]
    fn bounds_fit_inside_the_frustum_with_margin() {
        for view in View::ALL {
            let (view_proj, _, _) = camera_for(&bounds(), view, 1.0, 40.0, 1.1);
            for corner in [
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(-1.0, 1.0, -1.0),
                Vec3::new(1.0, 1.0, -1.0),
                Vec3::new(-1.0, -1.0, 1.0),
                Vec3::new(1.0, -1.0, 1.0),
                Vec3::new(-1.0, 1.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
            ] {
                let p = ndc(&view_proj, corner);
                assert!(p.w > 0.0, "{view}: corner {corner:?} behind the camera");
                assert!(p.x.abs() <= 1.0 && p.y.abs() <= 1.0, "{view}: corner clipped {p:?}");
            }
        }
    }

    #[test]
    fn narrow_aspect_pulls_the_camera_back() {
        let (_, wide_eye, _) = camera_for(&bounds(), View::Front, 1.0, 40.0, 1.0);
        let (_, tall_eye, _) = camera_for(&bounds(), View::Front, 0.5, 40.0, 1.0);
        assert!(tall_eye.length() > wide_eye.length());
    }

    #[test]
    fn degenerate_bounds_still_produce_a_finite_camera() {
        let degenerate = DrawableBounds {
            center: Vec3::ZERO,
            sphere_radius: 0.0,
            box_min: Vec3::ZERO,
            box_max: Vec3::ZERO,
        };
        let (view_proj, eye, light) = camera_for(&degenerate, View::Iso, 1.0, 40.0, 1.1);
        assert!(view_proj.0.iter().all(|v| v.is_finite()));
        assert!(eye.length().is_finite() && eye.length() > 0.0);
        assert!((light.length() - 1.0).abs() < 1e-5);
    }
}
