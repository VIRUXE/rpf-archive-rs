//! Turns a drawable LOD into renderer-ready geometry: decoded vertices,
//! validated indices and a resolved diffuse texture.

use image::RgbaImage;

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
/// Geometries without a vertex or index buffer — and triangles whose indices
/// fall outside the vertex buffer — are dropped rather than failing the render.
pub(crate) fn prepare<'a>(
    d: &Drawable,
    lod: &DrawableLod,
    tex: &'a TextureSet,
    paint: Option<[u8; 3]>,
    report: &mut RenderReport,
) -> Vec<PreparedGeometry<'a>> {
    let paint_tint = paint.map(|rgb| rgb.map(|channel| channel as f32 / 255.0));
    let mut prepared = Vec::new();

    for model in &lod.models {
        for geometry in &model.geometries {
            let (Some(buffer), Some(index_buffer)) =
                (&geometry.vertex_buffer, &geometry.index_buffer)
            else {
                continue;
            };
            let Ok(verts) = buffer.to_unified_vertices() else { continue };
            if verts.is_empty() {
                continue;
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
                None => None,
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
