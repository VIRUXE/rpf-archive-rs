use wasm_bindgen::prelude::*;
use crate::ydd::{parse_ydd, DrawableEntry};
use crate::ytd::{parse_ytd};
use anyhow::{Result, Context};

#[wasm_bindgen]
pub fn convert_to_gltf(ydd_bytes: &[u8], ytd_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    console_error_panic_hook::set_once();

    let glb = inner_convert(ydd_bytes, ytd_bytes)
        .map_err(|e| JsValue::from_str(&format!("Conversion error: {:?}", e)))?;

    Ok(glb)
}

fn inner_convert(ydd_bytes: &[u8], ytd_bytes: &[u8]) -> Result<Vec<u8>> {
    let entries = parse_ydd(ydd_bytes).context("Failed to parse YDD")?;
    let textures = parse_ytd(ytd_bytes).context("Failed to parse YTD")?;
    let geometry = build_geometry_json(&entries);

    // Create a very basic GLTF structure as a JSON string
    // In a full implementation, we would use the `gltf` crate to build a proper GLB.
    let root = json::object! {
        "asset": {
            "version": "2.0",
            "generator": "rpf-archive-wasm"
        },
        "extensions": {
            "gta_metadata": {
                "name": entries.first().map(|e| e.name.clone()).unwrap_or_default(),
                "drawable_count": entries.len(),
                "texture_count": textures.len(),
                "texture_names": textures.iter().map(|t| t.name.clone()).collect::<Vec<_>>()
            },
            "gta_geometry": geometry
        }
    };

    Ok(root.dump().into_bytes())
}

fn build_geometry_json(entries: &[DrawableEntry]) -> json::JsonValue {
    let mut meshes = json::JsonValue::new_array();

    for entry in entries {
        let Some(lod) = entry.drawable.best_lod() else { continue };

        for (model_index, model) in lod.models.iter().enumerate() {
            for (geometry_index, geometry) in model.geometries.iter().enumerate() {
                let Some(vertex_buffer) = &geometry.vertex_buffer else { continue };
                let Ok(vertices) = vertex_buffer.to_unified_vertices() else { continue };

                let positions = vertices
                    .iter()
                    .flat_map(|v| [v.position.x, v.position.y, v.position.z])
                    .collect::<Vec<f32>>();
                let indices = geometry
                    .index_buffer
                    .as_ref()
                    .map(|buffer| buffer.indices.clone())
                    .unwrap_or_default();

                if positions.is_empty() || indices.is_empty() {
                    continue;
                }

                let _ = meshes.push(json::object! {
                    "drawable": entry.name.clone(),
                    "model_index": model_index,
                    "geometry_index": geometry_index,
                    "vertex_count": vertices.len(),
                    "index_count": indices.len(),
                    "stride": vertex_buffer.vertex_stride,
                    "positions": positions,
                    "indices": indices,
                });
            }
        }
    }

    json::object! {
        "format": "raw_position_index_v1",
        "position_layout": "float32x3_at_vertex_offset_0",
        "meshes": meshes,
    }
}
