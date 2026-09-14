//! Turns a drawable LOD into renderer-ready geometry: decoded vertices,
//! validated indices and a resolved diffuse texture.

use image::RgbaImage;

use crate::math::Mat4;
use crate::render::{RenderReport, TextureSet};
use crate::ydd::{Drawable, DrawableLod, UnifiedVertex, VertexSemantic};

/// How a geometry's diffuse alpha is applied, taken from the RAGE render
/// bucket its shader is assigned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlendMode {
    /// Bucket 0: alpha is ignored.
    Opaque,
    /// Bucket 3: alpha-tested, below half is discarded.
    Cutout,
    /// Bucket 1: alpha-blended over what is already drawn.
    Blend,
    /// Bucket 2: blended like `Blend`; decals sit on other surfaces.
    Decal,
}

impl BlendMode {
    pub(crate) fn from_render_bucket(bucket: u8) -> Self {
        match bucket {
            1 => BlendMode::Blend,
            2 => BlendMode::Decal,
            3 => BlendMode::Cutout,
            _ => BlendMode::Opaque,
        }
    }

    /// True for the modes that mix with the framebuffer instead of
    /// replacing it, and so have to be drawn after everything solid.
    pub(crate) fn is_translucent(self) -> bool {
        matches!(self, BlendMode::Blend | BlendMode::Decal)
    }
}

/// One geometry, ready to rasterize.
pub(crate) struct PreparedGeometry<'a> {
    pub verts: Vec<UnifiedVertex>,
    pub indices: Vec<u32>,
    pub texture: Option<&'a RgbaImage>,
    pub has_normals: bool,
    pub blend: BlendMode,
    /// Per-channel multiplier (0..=1) applied to the sampled diffuse.
    pub tint: Option<[f32; 3]>,
}

/// Decodes every geometry of `lod`, resolving diffuse textures against `tex`
/// and recording what was found in `report`.
///
/// Every model is moved by `transform`, preceded by the pose matrix its bone
/// index selects from `bone_transforms` (when it has one and the model is
/// not skinned). Positions and normals are transformed in place so the
/// rasterizer never needs to know.
///
/// Geometries without a vertex or index buffer — and triangles whose indices
/// fall outside the vertex buffer — are dropped rather than failing the render.
pub(crate) fn prepare<'a>(
    d: &Drawable,
    lod: &DrawableLod,
    transform: &Mat4,
    bone_transforms: &[Mat4],
    tex: &'a TextureSet,
    paint: Option<[u8; 3]>,
    report: &mut RenderReport,
) -> Vec<PreparedGeometry<'a>> {
    let paint_tint = paint.map(|rgb| rgb.map(|channel| channel as f32 / 255.0));
    let mut prepared = Vec::new();

    for model in &lod.models {
        let pose = if model.is_skinned() { None } else { bone_transforms.get(model.bone_index()) };
        let matrix = match pose {
            Some(pose) => transform.mul(pose),
            None => *transform,
        };
        let place = !matrix.is_identity();

        for geometry in &model.geometries {
            let (Some(buffer), Some(index_buffer)) =
                (&geometry.vertex_buffer, &geometry.index_buffer)
            else {
                continue;
            };
            let Ok(mut verts) = buffer.to_unified_vertices() else { continue };
            if verts.is_empty() {
                continue;
            }
            if place {
                for vertex in &mut verts {
                    vertex.position = matrix.transform_point(vertex.position).xyz();
                    vertex.normal = matrix.transform_vector(vertex.normal).normalize();
                }
            }

            let limit = verts.len() as u32;
            let mut indices = Vec::with_capacity(index_buffer.indices.len());
            for triangle in index_buffer.indices.chunks_exact(3) {
                if triangle.iter().all(|index| *index < limit) {
                    indices.extend_from_slice(triangle);
                }
            }
            if indices.is_empty() {
                continue;
            }

            let texture = match d.diffuse_texture_name(geometry.shader_id) {
                Some(name) => match tex.get(name) {
                    Some(image) => Some(image),
                    None => {
                        report.missing_textures.push(name.to_string());
                        None
                    }
                },
                None => {
                    report.geometries_without_diffuse += 1;
                    None
                }
            };
            if texture.is_none() {
                report.untextured_geometries += 1;
            }

            let shader = d.shader(geometry.shader_id);
            let blend = shader
                .map(|shader| BlendMode::from_render_bucket(shader.render_bucket))
                .unwrap_or(BlendMode::Opaque);
            let tint = match shader {
                Some(shader) if super::is_vehicle_paint_shader(shader.file_name_hash) => paint_tint,
                _ => None,
            };

            let declares_normals = buffer
                .declaration
                .as_ref()
                .is_some_and(|declaration| {
                    declaration
                        .components
                        .iter()
                        .any(|component| component.semantic == VertexSemantic::Normal)
                });
            let has_normals = declares_normals
                && verts.iter().any(|vertex| vertex.normal.length() > 1e-6);

            report.triangles += indices.len() / 3;
            report.geometries += 1;

            prepared.push(PreparedGeometry { verts, indices, texture, has_normals, blend, tint });
        }
    }

    prepared
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Mat4, Vec2, Vec3};
    use crate::ydd::{
        DrawableBounds, DrawableGeometry, DrawableLod, DrawableModel, IndexBuffer, LodLevel,
        VertexBuffer, VertexBufferLayout, VertexComponent, VertexComponentType, VertexDeclaration,
        VertexSemantic,
    };

    /// One triangle with Position/Normal (stride 24), normals along +Y.
    fn triangle_drawable(skeleton_binding: u32) -> Drawable {
        let declaration = VertexDeclaration {
            flags: 0,
            stride: 24,
            unknown_6h: 0,
            count: 2,
            types: 0,
            components: vec![
                VertexComponent {
                    semantic: VertexSemantic::Position,
                    semantic_index: 0,
                    component_type: VertexComponentType::Float3,
                    offset: 0,
                    size: 12,
                    component_count: 3,
                },
                VertexComponent {
                    semantic: VertexSemantic::Normal,
                    semantic_index: 0,
                    component_type: VertexComponentType::Float3,
                    offset: 12,
                    size: 12,
                    component_count: 3,
                },
            ],
        };
        let mut data = Vec::new();
        for position in [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 2.0)] {
            for value in [position.x, position.y, position.z, 0.0, 1.0, 0.0] {
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        let geometry = DrawableGeometry {
            shader_id: 0,
            indices_count: 3,
            triangles_count: 1,
            vertices_count: 3,
            vertex_stride: 24,
            vertex_buffer: Some(VertexBuffer {
                vertex_stride: 24,
                vertex_count: 3,
                data_pointer: 0,
                info_pointer: 0,
                declaration: Some(declaration),
                data,
                layout: VertexBufferLayout::Legacy,
            }),
            index_buffer: Some(IndexBuffer { indices_count: 3, indices_pointer: 0, indices: vec![0, 1, 2] }),
        };
        Drawable {
            name: "tri".to_string(),
            name_hash: 0,
            bounds: DrawableBounds { center: Vec3::ZERO, sphere_radius: 0.0, box_min: Vec3::ZERO, box_max: Vec3::ZERO },
            lod_distances: [0.0; 4],
            render_masks: [0; 4],
            shader_group: None,
            lods: vec![DrawableLod {
                level: LodLevel::High,
                models: vec![DrawableModel { skeleton_binding, render_mask_flags: 0, shader_mapping: vec![0], geometries: vec![geometry] }],
            }],
        }
    }

    /// A mirrored wheel (X and Z flipped, translated) moves the positions
    /// and flips the normals, leaving texture coordinates alone.
    #[test]
    fn prepare_transforms_positions_and_normals() {
        let d = triangle_drawable(0);
        let mut flip = Mat4::identity();
        flip.0[0] = -1.0;
        flip.0[10] = -1.0;
        let transform = flip.with_translation(Vec3::new(10.0, 0.0, 0.0));
        let mut report = RenderReport::default();
        let textures = TextureSet::new();

        let prepared = prepare(&d, &d.lods[0], &transform, &[], &textures, None, &mut report);

        assert_eq!(prepared.len(), 1);
        let verts = &prepared[0].verts;
        assert_eq!(verts[0].position, Vec3::new(9.0, 0.0, 0.0));
        assert_eq!(verts[1].position, Vec3::new(10.0, 0.0, -1.0));
        assert_eq!(verts[0].normal, Vec3::new(0.0, 1.0, 0.0), "Y is untouched by an XZ flip");
        assert_eq!(verts[0].texcoord0, Vec2::new(0.0, 0.0));
        assert!(prepared[0].has_normals);
    }

    /// A model's bone index (top byte of its skeleton binding) selects the
    /// pose matrix; out-of-range indices fall back to the part transform.
    #[test]
    fn prepare_poses_models_by_bone_index() {
        let pose = [Mat4::identity(), Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0))];
        let mut report = RenderReport::default();
        let textures = TextureSet::new();

        let bound_to_one = triangle_drawable(1 << 24);
        let prepared = prepare(&bound_to_one, &bound_to_one.lods[0], &Mat4::identity(), &pose, &textures, None, &mut report);
        assert_eq!(prepared[0].verts[0].position, Vec3::new(1.0, 5.0, 0.0));

        let out_of_range = triangle_drawable(7 << 24);
        let prepared = prepare(&out_of_range, &out_of_range.lods[0], &Mat4::identity(), &pose, &textures, None, &mut report);
        assert_eq!(prepared[0].verts[0].position, Vec3::new(1.0, 0.0, 0.0));
    }
}
