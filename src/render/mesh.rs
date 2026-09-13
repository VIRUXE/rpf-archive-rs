//! Turns a drawable LOD into renderer-ready geometry: decoded vertices,
//! validated indices and a resolved diffuse texture.

use image::RgbaImage;

use crate::render::{RenderReport, TextureSet};
use crate::ydd::{Drawable, DrawableLod, UnifiedVertex, VertexSemantic};

/// One geometry, ready to rasterize.
pub(crate) struct PreparedGeometry<'a> {
    pub verts: Vec<UnifiedVertex>,
    pub indices: Vec<u32>,
    pub texture: Option<&'a RgbaImage>,
    pub has_normals: bool,
    pub alpha_cutout: bool,
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
    report: &mut RenderReport,
) -> Vec<PreparedGeometry<'a>> {
    let mut prepared = Vec::new();
    // Scanning a texture for translucent pixels is expensive; each distinct
    // image is only scanned once per prepare pass.
    let mut cutout_cache: Vec<(*const RgbaImage, bool)> = Vec::new();

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

            let alpha_cutout = match texture {
                Some(image) => {
                    let key = image as *const RgbaImage;
                    match cutout_cache.iter().find(|(cached, _)| *cached == key) {
                        Some((_, cutout)) => *cutout,
                        None => {
                            let cutout = image.pixels().any(|pixel| pixel.0[3] < 255);
                            cutout_cache.push((key, cutout));
                            cutout
                        }
                    }
                }
                None => false,
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

            prepared.push(PreparedGeometry { verts, indices, texture, has_normals, alpha_cutout });
        }
    }

    prepared
}
