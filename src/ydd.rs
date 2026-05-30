/// YDD (Drawable Dictionary) parser for GTA V.
use anyhow::{Result, Context};
use crate::resource::{ResReader, prepare_rsc7, u16_le, u32_le, u64_le};

// ─── Structs ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Drawable {
    pub name: String,
    pub shader_group: ShaderGroup,
    pub skeleton: Option<Skeleton>,
    pub models: Vec<DrawableModel>,
}

#[derive(Debug)]
pub struct ShaderGroup {
    pub textures: Vec<String>,
}

#[derive(Debug)]
pub struct Skeleton {
    // Basic skeletal info could go here
}

#[derive(Debug)]
pub struct DrawableModel {
    pub geometries: Vec<Geometry>,
}

#[derive(Debug)]
pub struct Geometry {
    pub vertex_buffer: VertexBuffer,
    pub index_buffer: IndexBuffer,
}

#[derive(Debug)]
pub struct VertexBuffer {
    pub data: Vec<u8>,
    pub count: u32,
    pub stride: u32,
}

#[derive(Debug)]
pub struct IndexBuffer {
    pub data: Vec<u8>,
    pub count: u32,
}

// ─── Implementation ───────────────────────────────────────────────────────────

pub fn parse_ydd(data: &[u8]) -> Result<Drawable> {
    let (system, graphics) = prepare_rsc7(data)?;
    let reader = ResReader { system: &system, graphics: &graphics };

    let sys = reader.system;

    if u32_le(sys, 0) == 0x4344_5244 || looks_like_drawable_dictionary(&reader) {
        return parse_drawable_dictionary(&reader);
    }

    parse_drawable(0x5000_0000, &reader)
}

fn looks_like_drawable_dictionary(reader: &ResReader<'_>) -> bool {
    let sys = reader.system;

    if sys.len() < 0x40 {
        return false;
    }

    let drawables_ptr = u64_le(sys, 0x30);
    let drawables_count = u16_le(sys, 0x38) as usize;
    let drawables_capacity = u16_le(sys, 0x3A) as usize;

    if drawables_count == 0 || drawables_count > drawables_capacity || drawables_count > 1024 {
        return false;
    }

    let Some(ptr_data) = reader.resolve(drawables_ptr, drawables_capacity * 8) else {
        return false;
    };

    (0..drawables_count).any(|i| {
        let drawable_va = u64_le(ptr_data, i * 8);
        reader.resolve(drawable_va, 0xD0).is_some()
    })
}

fn parse_drawable_dictionary(reader: &ResReader<'_>) -> Result<Drawable> {
    let sys = reader.system;
    let drawables_ptr = u64_le(sys, 0x30);
    let drawables_count = u16_le(sys, 0x38) as usize;
    let drawables_capacity = u16_le(sys, 0x3A) as usize;
    let pointer_count = drawables_capacity.max(drawables_count);
    let ptr_data = reader.resolve(drawables_ptr, pointer_count * 8)
        .context("Drawable dictionary pointer array out of bounds")?;

    for i in 0..drawables_count {
        let drawable_va = u64_le(ptr_data, i * 8);
        if drawable_va != 0 {
            return parse_drawable(drawable_va, reader);
        }
    }

    Err(anyhow::anyhow!("Drawable dictionary contains no drawables"))
}

fn parse_drawable(ptr: u64, reader: &ResReader<'_>) -> Result<Drawable> {
    let sys = reader.resolve(ptr, 0xB0).context("Drawable out of bounds")?;

    // ResourceFileBase is 16 bytes. DrawableBase fields follow.
    // DrawableBase (168 bytes total, but we only need pointers)
    let shader_group_ptr = u64_le(sys, 0x10);
    let skeleton_ptr = u64_le(sys, 0x18);
    let high_models_ptr = u64_le(sys, 0x50); // DrawableModelsHighPointer
    let models_ptr = match u64_le(sys, 0xA0) { // DrawableModelsPointer
        0 => high_models_ptr,
        ptr => ptr,
    };
    
    // Drawable (extends DrawableBase)
    let name_ptr = u64_le(sys, 0xA8); // Offset 168 (0xA8)
    
    let name = reader.string_at(name_ptr).unwrap_or_else(|| "unknown".to_string());
    
    let shader_group = parse_shader_group(shader_group_ptr, &reader)?;
    let skeleton = if skeleton_ptr != 0 { Some(Skeleton {}) } else { None };
    let models = parse_models(models_ptr, &reader)?;
    
    Ok(Drawable {
        name,
        shader_group,
        skeleton,
        models,
    })
}

fn parse_shader_group(ptr: u64, reader: &ResReader<'_>) -> Result<ShaderGroup> {
    if ptr == 0 {
        return Ok(ShaderGroup { textures: vec![] });
    }
    
    let raw = reader.resolve(ptr, 0x40)
        .context("ShaderGroup out of bounds")?;
    
    // ShaderGroup stores a pointer to its embedded TextureDictionary at 0x08.
    let tex_dict_va = u64_le(raw, 0x08);
    if tex_dict_va == 0 {
        return Ok(ShaderGroup { textures: vec![] });
    }

    let tex_dict_raw = reader.resolve(tex_dict_va, 0x40).context("TextureDictionary in ShaderGroup out of bounds")?;
    
    // Same offsets as in ytd.rs parse_texture_dict
    let tex_ptr_array = u64_le(tex_dict_raw, 0x30);
    let tex_count = u16_le(tex_dict_raw, 0x38) as usize;
    
    let mut textures = vec![];
    if tex_count > 0 {
        let ptr_data = reader.resolve(tex_ptr_array, tex_count * 8).context("ShaderGroup texture pointers out of bounds")?;
        for i in 0..tex_count {
            let tex_va = u64_le(ptr_data, i * 8);
            if tex_va == 0 { continue; }
            let tex_raw = reader.resolve(tex_va, 0x30).context("Texture in ShaderGroup out of bounds")?;
            let name_ptr = u64_le(tex_raw, 0x28);
            if let Some(name) = reader.string_at(name_ptr) {
                textures.push(name);
            }
        }
    }
    
    Ok(ShaderGroup { textures })
}

fn parse_models(ptr: u64, reader: &ResReader<'_>) -> Result<Vec<DrawableModel>> {
    if ptr == 0 {
        return Ok(vec![]);
    }
    
    // ResourcePointerList64<DrawableModel>
    let raw = reader.resolve(ptr, 16).context("Models pointer list out of bounds")?;
    let data_ptr = u64_le(raw, 0);
    let count = u16_le(raw, 8) as usize;
    
    let mut models = vec![];
    if count > 0 && data_ptr != 0 {
        let ptr_data = reader.resolve(data_ptr, count * 8).context("Model pointers out of bounds")?;
        for i in 0..count {
            let model_va = u64_le(ptr_data, i * 8);
            if model_va == 0 { continue; }
            models.push(parse_model(model_va, reader)?);
        }
    }
    
    Ok(models)
}

fn parse_model(ptr: u64, reader: &ResReader<'_>) -> Result<DrawableModel> {
    let raw = reader.resolve(ptr, 0x30).context("DrawableModel out of bounds")?;
    
    // DrawableModel stores the geometry pointer array at 0x08.
    let geoms_ptr = u64_le(raw, 0x08);
    let geoms_count = u16_le(raw, 0x10) as usize;
    
    let mut geometries = vec![];
    if geoms_count > 0 && geoms_ptr != 0 {
        let ptr_data = reader.resolve(geoms_ptr, geoms_count * 8).context("Geometry pointers out of bounds")?;
        for i in 0..geoms_count {
            let geom_va = u64_le(ptr_data, i * 8);
            if geom_va == 0 { continue; }
            geometries.push(parse_geometry(geom_va, reader)?);
        }
    }
    
    Ok(DrawableModel { geometries })
}

fn parse_geometry(ptr: u64, reader: &ResReader<'_>) -> Result<Geometry> {
    let raw = reader.resolve(ptr, 0x98).context("DrawableGeometry out of bounds")?;
    
    let vb_ptr = u64_le(raw, 0x18);
    let ib_ptr = u64_le(raw, 0x38);
    let vertices_count = u16_le(raw, 0x60) as u32;
    let vertex_stride = u16_le(raw, 0x70) as u32;
    let vertex_data_ptr = u64_le(raw, 0x78);
    
    let vertex_buffer = parse_vertex_buffer(vb_ptr, reader)
        .or_else(|_| parse_geometry_vertex_data(vertex_data_ptr, vertices_count, vertex_stride, reader))?;
    let index_buffer = parse_index_buffer(ib_ptr, reader)?;
    
    Ok(Geometry {
        vertex_buffer,
        index_buffer,
    })
}

fn parse_vertex_buffer(ptr: u64, reader: &ResReader<'_>) -> Result<VertexBuffer> {
    if ptr == 0 {
        return Err(anyhow::anyhow!("Missing vertex buffer"));
    }
    
    let raw = reader.resolve(ptr, 0x40).context("VertexBuffer out of bounds")?;

    // Legacy VertexBuffer layout.
    if let Some(buffer) = try_vertex_buffer(
        reader,
        u64_le(raw, 0x10),
        u32_le(raw, 0x18),
        u16_le(raw, 0x08) as u32,
    ) {
        return Ok(buffer);
    }

    // Some modded legacy files leave DataPointer1 empty and store the usable
    // stream in DataPointer2. CodeWalker uses Data1 ?? Data2 for geometry data.
    if let Some(buffer) = try_vertex_buffer(
        reader,
        u64_le(raw, 0x20),
        u32_le(raw, 0x18),
        u16_le(raw, 0x08) as u32,
    ) {
        return Ok(buffer);
    }

    // Gen9 VertexBuffer layout, as used by newer clothing resources.
    if let Some(buffer) = try_vertex_buffer(
        reader,
        u64_le(raw, 0x18),
        u32_le(raw, 0x08),
        u16_le(raw, 0x0C) as u32,
    ) {
        return Ok(buffer);
    }

    Err(anyhow::anyhow!("Vertex data out of bounds"))
}

fn parse_geometry_vertex_data(
    data_ptr: u64,
    count: u32,
    stride: u32,
    reader: &ResReader<'_>,
) -> Result<VertexBuffer> {
    try_vertex_buffer(reader, data_ptr, count, stride)
        .context("Geometry vertex data out of bounds")
}

fn parse_index_buffer(ptr: u64, reader: &ResReader<'_>) -> Result<IndexBuffer> {
    if ptr == 0 {
        return Err(anyhow::anyhow!("Missing index buffer"));
    }
    
    let raw = reader.resolve(ptr, 0x40).context("IndexBuffer out of bounds")?;

    let count = u32_le(raw, 0x08);

    // Legacy IndexBuffer layout.
    if let Some(buffer) = try_index_buffer(reader, u64_le(raw, 0x10), count, 2) {
        return Ok(buffer);
    }

    // Gen9 IndexBuffer layout.
    let gen9_index_size = u16_le(raw, 0x0C) as u32;
    if let Some(buffer) = try_index_buffer(reader, u64_le(raw, 0x18), count, gen9_index_size) {
        return Ok(buffer);
    }

    Err(anyhow::anyhow!("Index data out of bounds"))
}

fn try_vertex_buffer(
    reader: &ResReader<'_>,
    data_ptr: u64,
    count: u32,
    stride: u32,
) -> Option<VertexBuffer> {
    if data_ptr == 0 || count == 0 || stride == 0 {
        return None;
    }

    let data_len = (count as usize).checked_mul(stride as usize)?;
    let data = reader.resolve(data_ptr, data_len)?.to_vec();

    Some(VertexBuffer { data, count, stride })
}

fn try_index_buffer(
    reader: &ResReader<'_>,
    data_ptr: u64,
    count: u32,
    index_size: u32,
) -> Option<IndexBuffer> {
    if data_ptr == 0 || count == 0 || index_size == 0 {
        return None;
    }

    let data_len = (count as usize).checked_mul(index_size as usize)?;
    let data = reader.resolve(data_ptr, data_len)?.to_vec();

    Some(IndexBuffer { data, count })
}
