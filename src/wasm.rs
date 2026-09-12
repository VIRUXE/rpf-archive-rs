use wasm_bindgen::prelude::*;
use crate::ydd::{parse_ydd, Drawable};
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
    let drawable = parse_ydd(ydd_bytes).context("Failed to parse YDD")?;
    let textures = parse_ytd(ytd_bytes).context("Failed to parse YTD")?;
    let geometry = build_geometry_json(&drawable);
    
    // Create a very basic GLTF structure as a JSON string
    // In a full implementation, we would use the `gltf` crate to build a proper GLB.
    let root = json::object! {
        "asset": {
            "version": "2.0",
            "generator": "rpf-rs-wasm"
        },
        "extensions": {
            "gta_metadata": {
                "name": drawable.name,
                "model_count": drawable.models.len(),
                "texture_count": textures.len(),
                "texture_names": textures.iter().map(|t| t.name.clone()).collect::<Vec<_>>()
            },
            "gta_geometry": geometry
        }
    };
    
    Ok(root.dump().into_bytes())
}

fn build_geometry_json(drawable: &Drawable) -> json::JsonValue {
    let mut meshes = json::JsonValue::new_array();

    for (model_index, model) in drawable.models.iter().enumerate() {
        for (geometry_index, geometry) in model.geometries.iter().enumerate() {
            let positions = extract_positions(&geometry.vertex_buffer.data, geometry.vertex_buffer.count, geometry.vertex_buffer.stride);
            let indices = extract_indices(&geometry.index_buffer.data, geometry.index_buffer.count);

            if positions.is_empty() || indices.is_empty() {
                continue;
            }

            let _ = meshes.push(json::object! {
                "model_index": model_index,
                "geometry_index": geometry_index,
                "vertex_count": geometry.vertex_buffer.count,
                "index_count": geometry.index_buffer.count,
                "stride": geometry.vertex_buffer.stride,
                "positions": positions,
                "indices": indices,
            });
        }
    }

    json::object! {
        "format": "raw_position_index_v1",
        "position_layout": "float32x3_at_vertex_offset_0",
        "meshes": meshes,
    }
}

fn extract_positions(data: &[u8], count: u32, stride: u32) -> Vec<f32> {
    if stride < 12 {
        return vec![];
    }

    let mut positions = Vec::with_capacity(count as usize * 3);

    for i in 0..count as usize {
        let offset = i.saturating_mul(stride as usize);
        if offset + 12 > data.len() {
            break;
        }

        positions.push(f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()));
        positions.push(f32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()));
        positions.push(f32::from_le_bytes(data[offset + 8..offset + 12].try_into().unwrap()));
    }

    positions
}

fn extract_indices(data: &[u8], count: u32) -> Vec<u32> {
    let mut indices = Vec::with_capacity(count as usize);

    for i in 0..count as usize {
        let offset = i.saturating_mul(2);
        if offset + 2 > data.len() {
            break;
        }

        indices.push(u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as u32);
    }

    indices
}
