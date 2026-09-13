//! Triangle rasterization into an RGBA framebuffer with a depth buffer.

use image::{Rgba, RgbaImage};

use crate::math::{Mat4, Vec2, Vec3};
use crate::render::mesh::PreparedGeometry;
use crate::render::RenderOptions;

/// Colour used where a geometry has no diffuse texture.
const FLAT_GREY: [u8; 4] = [153, 153, 153, 255];
/// Sampled alpha below this is discarded on cutout geometry.
const ALPHA_CUTOFF: f32 = 128.0;

pub(crate) struct Framebuffer {
    pub color: RgbaImage,
    pub depth: Vec<f32>,
}

impl Framebuffer {
    pub(crate) fn new(width: u32, height: u32, background: [u8; 4]) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        Self {
            color: RgbaImage::from_pixel(width, height, Rgba(background)),
            depth: vec![f32::INFINITY; (width as usize) * (height as usize)],
        }
    }
}

/// One vertex after projection: screen position, NDC depth, 1/w and the
/// attributes the shading step needs.
#[derive(Clone, Copy)]
struct Projected {
    x: f32,
    y: f32,
    z: f32,
    inv_w: f32,
    world: Vec3,
    uv: Vec2,
    normal: Vec3,
    color: [f32; 3],
}

/// Rasterizes every triangle of `g` into `fb`.
pub(crate) fn draw_geometry(
    fb: &mut Framebuffer,
    g: &PreparedGeometry,
    view_proj: &Mat4,
    light: Vec3,
    o: &RenderOptions,
) {
    let width = fb.color.width() as f32;
    let height = fb.color.height() as f32;
    let light = light.normalize();

    for triangle in g.indices.chunks_exact(3) {
        let mut points = [None; 3];
        for (slot, index) in triangle.iter().enumerate() {
            let Some(vertex) = g.verts.get(*index as usize) else { break };
            let clip = view_proj.transform_point(vertex.position);
            // The near plane sits in front of the whole model, so anything at
            // or behind the eye is a degenerate case, not something to clip.
            if clip.w.is_nan() || clip.w <= 1e-4 {
                points[slot] = None;
                break;
            }
            let inv_w = 1.0 / clip.w;
            points[slot] = Some(Projected {
                x: (clip.x * inv_w + 1.0) * 0.5 * width,
                y: (1.0 - clip.y * inv_w) * 0.5 * height,
                z: clip.z * inv_w,
                inv_w,
                world: vertex.position,
                uv: vertex.texcoord0,
                normal: vertex.normal,
                color: [
                    vertex.color0[0] as f32 / 255.0,
                    vertex.color0[1] as f32 / 255.0,
                    vertex.color0[2] as f32 / 255.0,
                ],
            });
        }

        let [Some(a), Some(b), Some(c)] = points else { continue };

        // Screen space has y growing downward, so a counter-clockwise (front
        // facing) triangle in NDC has a negative signed area here.
        let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if !area.is_finite() || area.abs() < 1e-12 {
            continue;
        }
        if o.backface_cull && area > 0.0 {
            continue;
        }

        // Normalize the winding so the edge functions below are positive
        // inside the triangle, keeping one fill rule for both orientations.
        let (v0, v1, v2) = if area < 0.0 { (a, c, b) } else { (a, b, c) };

        // Only |dot(n, light)| is used, so the face normal's sign is irrelevant.
        let face_normal = (b.world - a.world).cross(c.world - a.world).normalize();

        raster_triangle(fb, g, o, [v0, v1, v2], area.abs(), face_normal, light);
    }
}

#[allow(clippy::too_many_arguments)]
fn raster_triangle(
    fb: &mut Framebuffer,
    g: &PreparedGeometry,
    o: &RenderOptions,
    v: [Projected; 3],
    area: f32,
    face_normal: Vec3,
    light: Vec3,
) {
    let width = fb.color.width() as i64;
    let height = fb.color.height() as i64;

    let min_x = v.iter().fold(f32::INFINITY, |m, p| m.min(p.x)).floor() as i64;
    let max_x = v.iter().fold(f32::NEG_INFINITY, |m, p| m.max(p.x)).ceil() as i64;
    let min_y = v.iter().fold(f32::INFINITY, |m, p| m.min(p.y)).floor() as i64;
    let max_y = v.iter().fold(f32::NEG_INFINITY, |m, p| m.max(p.y)).ceil() as i64;

    let x0 = min_x.clamp(0, width);
    let x1 = max_x.clamp(0, width);
    let y0 = min_y.clamp(0, height);
    let y1 = max_y.clamp(0, height);
    if x0 >= x1 || y0 >= y1 {
        return;
    }

    let bias = [
        top_left_bias(v[1], v[2]),
        top_left_bias(v[2], v[0]),
        top_left_bias(v[0], v[1]),
    ];
    let inv_area = 1.0 / area;

    for py in y0..y1 {
        for px in x0..x1 {
            let sx = px as f32 + 0.5;
            let sy = py as f32 + 0.5;

            let e0 = edge(v[1], v[2], sx, sy);
            let e1 = edge(v[2], v[0], sx, sy);
            let e2 = edge(v[0], v[1], sx, sy);
            let inside = [(e0, bias[0]), (e1, bias[1]), (e2, bias[2])]
                .iter()
                .all(|(e, b)| *e > 0.0 || (*e == 0.0 && *b));
            if !inside {
                continue;
            }

            let l = [e0 * inv_area, e1 * inv_area, e2 * inv_area];
            let depth = l[0] * v[0].z + l[1] * v[1].z + l[2] * v[2].z;
            let offset = (py as usize) * (width as usize) + px as usize;
            if depth.is_nan() || depth >= fb.depth[offset] {
                continue;
            }

            // Perspective-correct UVs; colour and normal are interpolated
            // linearly, which is plenty for a preview rasterizer.
            let inv_w = l[0] * v[0].inv_w + l[1] * v[1].inv_w + l[2] * v[2].inv_w;
            let uv = if inv_w.abs() > 1e-12 {
                Vec2::new(
                    (l[0] * v[0].uv.x * v[0].inv_w
                        + l[1] * v[1].uv.x * v[1].inv_w
                        + l[2] * v[2].uv.x * v[2].inv_w)
                        / inv_w,
                    (l[0] * v[0].uv.y * v[0].inv_w
                        + l[1] * v[1].uv.y * v[1].inv_w
                        + l[2] * v[2].uv.y * v[2].inv_w)
                        / inv_w,
                )
            } else {
                v[0].uv
            };

            let mut texel = match g.texture {
                Some(image) => sample_bilinear(image, uv),
                None => [
                    FLAT_GREY[0] as f32,
                    FLAT_GREY[1] as f32,
                    FLAT_GREY[2] as f32,
                    FLAT_GREY[3] as f32,
                ],
            };
            if g.alpha_cutout && texel[3] < ALPHA_CUTOFF {
                continue;
            }

            if o.lighting {
                let normal = if g.has_normals {
                    Vec3::new(
                        l[0] * v[0].normal.x + l[1] * v[1].normal.x + l[2] * v[2].normal.x,
                        l[0] * v[0].normal.y + l[1] * v[1].normal.y + l[2] * v[2].normal.y,
                        l[0] * v[0].normal.z + l[1] * v[1].normal.z + l[2] * v[2].normal.z,
                    )
                    .normalize()
                } else {
                    face_normal
                };
                let k = 0.45 + 0.55 * normal.dot(light).abs();
                texel[0] *= k;
                texel[1] *= k;
                texel[2] *= k;
            }

            if o.vertex_colors {
                let tint = [0usize, 1, 2].map(|channel| {
                    l[0] * v[0].color[channel]
                        + l[1] * v[1].color[channel]
                        + l[2] * v[2].color[channel]
                });
                for (value, tint) in texel.iter_mut().zip(tint) {
                    *value *= tint;
                }
            }

            fb.depth[offset] = depth;
            fb.color.put_pixel(
                px as u32,
                py as u32,
                Rgba([
                    to_u8(texel[0]),
                    to_u8(texel[1]),
                    to_u8(texel[2]),
                    255,
                ]),
            );
        }
    }
}

#[inline]
fn edge(a: Projected, b: Projected, x: f32, y: f32) -> f32 {
    (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x)
}

/// Pixels exactly on an edge belong to the triangle only when that edge is a
/// top or left edge, so shared edges are drawn once.
#[inline]
fn top_left_bias(a: Projected, b: Projected) -> bool {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    dy > 0.0 || (dy == 0.0 && dx < 0.0)
}

#[inline]
fn to_u8(value: f32) -> u8 {
    if value.is_nan() {
        return 0;
    }
    value.clamp(0.0, 255.0).round() as u8
}

#[inline]
fn wrap(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    let fract = value - value.floor();
    if fract < 0.0 { fract + 1.0 } else { fract }
}

/// Bilinear sample with wrap addressing, returning straight 0..255 channels.
fn sample_bilinear(image: &RgbaImage, uv: Vec2) -> [f32; 4] {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return [
            FLAT_GREY[0] as f32,
            FLAT_GREY[1] as f32,
            FLAT_GREY[2] as f32,
            FLAT_GREY[3] as f32,
        ];
    }

    let fx = wrap(uv.x) * width as f32 - 0.5;
    let fy = wrap(uv.y) * height as f32 - 0.5;
    let ix = fx.floor();
    let iy = fy.floor();
    let tx = fx - ix;
    let ty = fy - iy;

    let wrap_index = |value: f32, size: u32| -> u32 {
        let size = size as i64;
        let index = (value as i64).rem_euclid(size);
        index as u32
    };

    let x0 = wrap_index(ix, width);
    let x1 = wrap_index(ix + 1.0, width);
    let y0 = wrap_index(iy, height);
    let y1 = wrap_index(iy + 1.0, height);

    let p00 = image.get_pixel(x0, y0).0;
    let p10 = image.get_pixel(x1, y0).0;
    let p01 = image.get_pixel(x0, y1).0;
    let p11 = image.get_pixel(x1, y1).0;

    let mut out = [0.0f32; 4];
    for channel in 0..4 {
        let top = p00[channel] as f32 * (1.0 - tx) + p10[channel] as f32 * tx;
        let bottom = p01[channel] as f32 * (1.0 - tx) + p11[channel] as f32 * tx;
        out[channel] = top * (1.0 - ty) + bottom * ty;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Mat4, Vec2, Vec3};
    use crate::render::camera::camera_for;
    use crate::render::mesh::PreparedGeometry;
    use crate::render::{RenderOptions, View};
    use crate::ydd::{DrawableBounds, UnifiedVertex};
    use image::RgbaImage;

    const BACKGROUND: [u8; 4] = [230, 230, 230, 255];

    fn vertex(position: Vec3, uv: Vec2) -> UnifiedVertex {
        UnifiedVertex {
            position,
            normal: Vec3::new(0.0, -1.0, 0.0),
            color0: [255, 255, 255, 255],
            color1: [255, 255, 255, 255],
            texcoord0: uv,
            texcoord1: Vec2::new(0.0, 0.0),
            tangent: crate::math::Vec4::new(1.0, 0.0, 0.0, 1.0),
            blend_weights: crate::math::Vec4::new(0.0, 0.0, 0.0, 0.0),
            blend_indices: [0; 4],
        }
    }

    fn flat_options() -> RenderOptions {
        RenderOptions {
            width: 64,
            height: 64,
            background: BACKGROUND,
            backface_cull: false,
            lighting: false,
            vertex_colors: false,
            ..RenderOptions::default()
        }
    }

    fn solid(colour: [u8; 4]) -> RgbaImage {
        RgbaImage::from_pixel(1, 1, image::Rgba(colour))
    }

    /// A screen-filling, counter-clockwise (front-facing) quad in NDC, used
    /// with an identity view-projection. `v` grows downward on screen.
    fn ndc_quad(z: f32) -> (Vec<UnifiedVertex>, Vec<u32>) {
        let verts = vec![
            vertex(Vec3::new(-1.0, -1.0, z), Vec2::new(0.0, 1.0)),
            vertex(Vec3::new(1.0, -1.0, z), Vec2::new(1.0, 1.0)),
            vertex(Vec3::new(1.0, 1.0, z), Vec2::new(1.0, 0.0)),
            vertex(Vec3::new(-1.0, 1.0, z), Vec2::new(0.0, 0.0)),
        ];
        (verts, vec![0, 1, 2, 0, 2, 3])
    }

    fn bounds() -> DrawableBounds {
        DrawableBounds {
            center: Vec3::ZERO,
            sphere_radius: 1.5,
            box_min: Vec3::new(-1.0, -1.0, -1.0),
            box_max: Vec3::new(1.0, 1.0, 1.0),
        }
    }

    #[test]
    fn triangle_covers_expected_pixels() {
        let options = flat_options();
        let (view_proj, _eye, light) =
            camera_for(&bounds(), View::Front, 1.0, options.fov_deg, options.margin);

        let verts = vec![
            vertex(Vec3::new(-1.0, 0.0, -1.0), Vec2::new(0.0, 1.0)),
            vertex(Vec3::new(1.0, 0.0, -1.0), Vec2::new(1.0, 1.0)),
            vertex(Vec3::new(0.0, 0.0, 1.0), Vec2::new(0.5, 0.0)),
        ];
        let geometry = PreparedGeometry {
            verts,
            indices: vec![0, 1, 2],
            texture: None,
            has_normals: false,
            alpha_cutout: false,
        };

        let mut fb = Framebuffer::new(options.width, options.height, options.background);
        draw_geometry(&mut fb, &geometry, &view_proj, light, &options);

        assert_ne!(fb.color.get_pixel(32, 34).0, BACKGROUND, "centre should be covered");
        assert_eq!(fb.color.get_pixel(0, 0).0, BACKGROUND, "corner should stay background");
    }

    #[test]
    fn nearer_triangle_wins_depth() {
        let options = flat_options();
        let red = solid([255, 0, 0, 255]);
        let blue = solid([0, 0, 255, 255]);

        // Identity view-projection: smaller z is nearer.
        let (near_verts, indices) = ndc_quad(-0.5);
        let (far_verts, _) = ndc_quad(0.5);

        let near = PreparedGeometry {
            verts: near_verts,
            indices: indices.clone(),
            texture: Some(&red),
            has_normals: false,
            alpha_cutout: false,
        };
        let far = PreparedGeometry {
            verts: far_verts,
            indices,
            texture: Some(&blue),
            has_normals: false,
            alpha_cutout: false,
        };

        let identity = Mat4::identity();
        let light = Vec3::new(0.0, 0.0, 1.0);

        // Draw the nearer one first, so only the depth test can keep it.
        let mut fb = Framebuffer::new(options.width, options.height, options.background);
        draw_geometry(&mut fb, &near, &identity, light, &options);
        draw_geometry(&mut fb, &far, &identity, light, &options);
        assert_eq!(fb.color.get_pixel(32, 32).0, [255, 0, 0, 255]);

        // And the other order, for good measure.
        let mut fb = Framebuffer::new(options.width, options.height, options.background);
        draw_geometry(&mut fb, &far, &identity, light, &options);
        draw_geometry(&mut fb, &near, &identity, light, &options);
        assert_eq!(fb.color.get_pixel(32, 32).0, [255, 0, 0, 255]);
    }

    #[test]
    fn uv_sampling_from_2x2_texture() {
        let options = flat_options();
        let mut texture = RgbaImage::new(2, 2);
        texture.put_pixel(0, 0, image::Rgba([255, 0, 0, 255])); // red
        texture.put_pixel(1, 0, image::Rgba([0, 255, 0, 255])); // green
        texture.put_pixel(0, 1, image::Rgba([0, 0, 255, 255])); // blue
        texture.put_pixel(1, 1, image::Rgba([255, 255, 255, 255])); // white

        let (verts, indices) = ndc_quad(0.0);
        let geometry = PreparedGeometry {
            verts,
            indices,
            texture: Some(&texture),
            has_normals: false,
            alpha_cutout: false,
        };

        let mut fb = Framebuffer::new(options.width, options.height, options.background);
        draw_geometry(&mut fb, &geometry, &Mat4::identity(), Vec3::Z, &options);

        let top_left = fb.color.get_pixel(16, 16).0;
        assert!(
            top_left[0] > 200 && top_left[1] < 60 && top_left[2] < 60,
            "expected red-ish, got {top_left:?}"
        );

        let bottom_right = fb.color.get_pixel(48, 48).0;
        assert!(
            bottom_right[0] > 200 && bottom_right[1] > 200 && bottom_right[2] > 200,
            "expected white-ish, got {bottom_right:?}"
        );
    }

    #[test]
    fn alpha_cutout_discards() {
        let options = flat_options();
        let transparent = solid([255, 0, 0, 0]);
        let (verts, indices) = ndc_quad(0.0);
        let geometry = PreparedGeometry {
            verts,
            indices,
            texture: Some(&transparent),
            has_normals: false,
            alpha_cutout: true,
        };

        let mut fb = Framebuffer::new(options.width, options.height, options.background);
        draw_geometry(&mut fb, &geometry, &Mat4::identity(), Vec3::Z, &options);

        assert!(fb.color.pixels().all(|p| p.0 == BACKGROUND));
    }

    #[test]
    fn backface_culling_drops_reversed_winding() {
        let options = RenderOptions { backface_cull: true, ..flat_options() };
        let (verts, mut indices) = ndc_quad(0.0);
        indices.reverse();

        let geometry = PreparedGeometry {
            verts,
            indices,
            texture: None,
            has_normals: false,
            alpha_cutout: false,
        };

        let mut fb = Framebuffer::new(options.width, options.height, options.background);
        draw_geometry(&mut fb, &geometry, &Mat4::identity(), Vec3::Z, &options);
        assert!(fb.color.pixels().all(|p| p.0 == BACKGROUND));
    }
}
