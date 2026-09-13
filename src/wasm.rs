use wasm_bindgen::prelude::*;
use crate::ydd::{parse_drawables, DrawableEntry, DrawableKind};
use crate::ytd::{parse_ytd, YtdTexture};
use anyhow::Result;

#[wasm_bindgen]
pub fn convert_to_gltf(ydd_bytes: &[u8], ytd_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    console_error_panic_hook::set_once();

    let glb = inner_convert(ydd_bytes, ytd_bytes)
        .map_err(|e| JsValue::from_str(&format!("Conversion error: {:?}", e)))?;

    Ok(glb)
}

fn inner_convert(ydd_bytes: &[u8], ytd_bytes: &[u8]) -> Result<Vec<u8>> {
    let entries = parse_drawables(ydd_bytes, DrawableKind::Ydd)
        .or_else(|_| parse_drawables(ydd_bytes, DrawableKind::Ydr))?;

    // An empty ytd slice means "no external textures", not a parse failure.
    let external_textures = if ytd_bytes.is_empty() {
        Vec::new()
    } else {
        parse_ytd(ytd_bytes)?
    };

    let drawables = build_drawables_json(&entries);
    let textures = build_textures_json(&entries, &external_textures);

    let root = json::object! {
        "asset": {
            "version": "2.0",
            "generator": "rpf-archive-wasm"
        },
        "format": "unified_v2",
        "drawables": drawables,
        "textures": textures,
    };

    Ok(root.dump().into_bytes())
}

fn build_drawables_json(entries: &[DrawableEntry]) -> json::JsonValue {
    let mut drawables = json::JsonValue::new_array();

    for entry in entries {
        let drawable = &entry.drawable;
        let best_lod = drawable.best_lod();

        let (bounds, lod_name) = match best_lod {
            Some(lod) => (drawable.bounds_or_computed(lod).0, lod.level.as_str()),
            None => (drawable.bounds.clone(), "none"),
        };

        let bounds_json = json::object! {
            "min": vec![bounds.box_min.x, bounds.box_min.y, bounds.box_min.z],
            "max": vec![bounds.box_max.x, bounds.box_max.y, bounds.box_max.z],
            "center": vec![bounds.center.x, bounds.center.y, bounds.center.z],
            "radius": bounds.sphere_radius,
        };

        let mut meshes = json::JsonValue::new_array();

        if let Some(lod) = best_lod {
            for (model_index, model) in lod.models.iter().enumerate() {
                for (geometry_index, geometry) in model.geometries.iter().enumerate() {
                    let Some(vertex_buffer) = &geometry.vertex_buffer else { continue };
                    let Some(index_buffer) = &geometry.index_buffer else { continue };
                    let Ok(vertices) = vertex_buffer.to_unified_vertices() else { continue };

                    if vertices.is_empty() || index_buffer.indices.is_empty() {
                        continue;
                    }

                    let mut positions = Vec::with_capacity(vertices.len() * 3);
                    let mut normals = Vec::with_capacity(vertices.len() * 3);
                    let mut uvs = Vec::with_capacity(vertices.len() * 2);
                    let mut colors = Vec::with_capacity(vertices.len() * 4);

                    for vertex in &vertices {
                        positions.extend([vertex.position.x, vertex.position.y, vertex.position.z]);
                        normals.extend([vertex.normal.x, vertex.normal.y, vertex.normal.z]);
                        uvs.extend([vertex.texcoord0.x, vertex.texcoord0.y]);
                        colors.extend(vertex.color0);
                    }

                    let shader_hash = drawable
                        .shader(geometry.shader_id)
                        .map(|shader| shader.name_hash)
                        .unwrap_or(0);
                    let diffuse_texture = drawable
                        .diffuse_texture_name(geometry.shader_id)
                        .map(|name| name.to_string());

                    let _ = meshes.push(json::object! {
                        "model_index": model_index,
                        "geometry_index": geometry_index,
                        "shader_hash": shader_hash,
                        "diffuse_texture": diffuse_texture,
                        "positions": positions,
                        "normals": normals,
                        "uvs": uvs,
                        "colors": colors,
                        "indices": index_buffer.indices.clone(),
                    });
                }
            }
        }

        let _ = drawables.push(json::object! {
            "name": entry.name.clone(),
            "hash": entry.hash,
            "bounds": bounds_json,
            "lod": lod_name,
            "meshes": meshes,
        });
    }

    drawables
}

fn build_textures_json(entries: &[DrawableEntry], external: &[YtdTexture]) -> json::JsonValue {
    let mut textures = json::JsonValue::new_array();

    for entry in entries {
        let Some(group) = &entry.drawable.shader_group else { continue };
        for texture in &group.textures {
            let _ = textures.push(texture_json(texture));
        }
    }

    for texture in external {
        let _ = textures.push(texture_json(texture));
    }

    textures
}

fn texture_json(texture: &YtdTexture) -> json::JsonValue {
    json::object! {
        "name": texture.name.clone(),
        "width": texture.width,
        "height": texture.height,
        "format": texture.format.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::RSC7_MAGIC;
    use crate::ydd::tests::minimal_ydr_sections;
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write;

    /// Wraps hand-built system/graphics sections into an RSC7 container that
    /// `prepare_rsc7` accepts: a 16-byte header (magic, version, system flags,
    /// graphics flags) followed by the deflated system+graphics bytes.
    fn wrap_rsc7(system: &[u8], graphics: &[u8]) -> Vec<u8> {
        let system_flags = size_to_flags(system.len());
        let graphics_flags = size_to_flags(graphics.len());

        let mut body = Vec::with_capacity(system.len() + graphics.len());
        body.extend_from_slice(system);
        body.extend_from_slice(graphics);

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&body).expect("deflate write");
        let compressed = encoder.finish().expect("deflate finish");

        let mut out = Vec::with_capacity(16 + compressed.len());
        out.extend_from_slice(&RSC7_MAGIC.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // version, unused by prepare_rsc7
        out.extend_from_slice(&system_flags.to_le_bytes());
        out.extend_from_slice(&graphics_flags.to_le_bytes());
        out.extend_from_slice(&compressed);
        out
    }

    /// Encodes a resource size as a page-flags value that `resource_size_from_flags`
    /// decodes back to exactly `len` bytes. With the base-size shift left at 0
    /// (0x200-byte chunks), `resource_size_from_flags` reads single bits at
    /// 27/26/25/24 as the 1/2/4/8 binary digits of the chunk count, so any
    /// `len` that is a multiple of 0x200 up to 15 chunks (7680 bytes) can be
    /// encoded directly — comfortably more than this fixture needs.
    fn size_to_flags(len: usize) -> u32 {
        assert_eq!(len % 0x200, 0, "fixture sizes must be page-aligned");
        let chunks = (len / 0x200) as u32;
        assert!(chunks <= 0xF, "fixture too large for this simple flag encoding");
        let mut flags = 0u32;
        if chunks & 0x1 != 0 { flags |= 1 << 27; }
        if chunks & 0x2 != 0 { flags |= 1 << 26; }
        if chunks & 0x4 != 0 { flags |= 1 << 25; }
        if chunks & 0x8 != 0 { flags |= 1 << 24; }
        flags
    }

    #[test]
    fn convert_to_gltf_emits_unified_v2_json() {
        let (system, graphics) = minimal_ydr_sections(false);
        let ydr_bytes = wrap_rsc7(&system, &graphics);

        let output = inner_convert(&ydr_bytes, &[]).expect("conversion should succeed");
        let root = json::parse(&String::from_utf8(output).expect("utf8 json")).expect("valid json");

        assert_eq!(root["format"], "unified_v2");
        assert_eq!(root["drawables"].len(), 1);

        let drawable = &root["drawables"][0];
        assert_eq!(drawable["name"], "test_drawable");
        assert_eq!(drawable["meshes"].len(), 1);

        let mesh = &drawable["meshes"][0];
        assert_eq!(mesh["positions"].len(), 9);
        assert_eq!(mesh["indices"].len(), 3);
    }
}
