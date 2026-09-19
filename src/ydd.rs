//! Drawable parsing for GTA V resources: YDR (a single drawable), YDD (a
//! drawable dictionary) and the drawables carried inside a YFT fragment.
//!
//! Everything here works on the two virtual-memory sections produced by
//! [`crate::resource::prepare_rsc7`], addressed through [`ResReader`].

use anyhow::{Context, Result};

use crate::math::{Vec3, Vec4};
use crate::resource::{
    f32_le, u16_le, u32_le, u64_le, vec3_le, vec4_le, prepare_rsc7, ResReader, SYSTEM_BASE,
};
use crate::vertex::parse_vertex_buffer_at;
use crate::writer::rage_joaat;
use crate::ytd::{parse_texture_dict_at, YtdTexture};

/// `rage_joaat("diffusesampler")` — the shader parameter naming the albedo map.
pub const DIFFUSE_SAMPLER: u32 = 0xF1FE_2B71;
/// `rage_joaat("bumpsampler")` — the normal map parameter.
pub const BUMP_SAMPLER: u32 = 0x46B7_C64F;
/// `rage_joaat("specsampler")` — the specular map parameter.
pub const SPEC_SAMPLER: u32 = 0x6087_99C6;
/// `rage_joaat("texturesampler")` — the albedo parameter on shaders carried
/// over from the GTA IV / Max Payne 3 pipelines, used when a shader has no
/// `DiffuseSampler`.
pub const TEXTURE_SAMPLER: u32 = 0x2B51_70FD;

// ─── Public types ─────────────────────────────────────────────────────────────

/// Which kind of resource a set of drawables was read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawableKind {
    Ydr,
    Ydd,
    Yft,
}

impl DrawableKind {
    /// Maps a file extension (case-insensitive, with or without a leading dot)
    /// onto a drawable kind. Returns `None` for anything else.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.trim_start_matches('.').to_ascii_lowercase().as_str() {
            "ydr" => Some(Self::Ydr),
            "ydd" => Some(Self::Ydd),
            "yft" => Some(Self::Yft),
            _ => None,
        }
    }
}

/// One named drawable, as listed by a dictionary or fragment.
#[derive(Debug, Clone)]
pub struct DrawableEntry {
    pub hash: u32,
    pub name: String,
    pub drawable: Drawable,
}

#[derive(Debug, Clone)]
pub struct Drawable {
    pub name: String,
    pub name_hash: u32,
    pub bounds: DrawableBounds,
    pub lod_distances: [f32; 4],
    pub render_masks: [u32; 4],
    pub shader_group: Option<ShaderGroup>,
    pub lods: Vec<DrawableLod>,
}

#[derive(Debug, Clone)]
pub struct DrawableBounds {
    pub center: Vec3,
    pub sphere_radius: f32,
    pub box_min: Vec3,
    pub box_max: Vec3,
}

/// One geometry's own vertex extent — what [`Drawable::bounds_or_computed`]
/// folds away into a single box. See [`Drawable::geometry_bounds`].
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryBounds {
    pub model: usize,
    pub geometry: usize,
    pub shader_id: u16,
    pub vertices: usize,
    pub triangles: usize,
    pub min: Vec3,
    pub max: Vec3,
    /// Mean of the finite vertex positions — not the box centre; the two
    /// disagreeing says where the mesh's mass actually sits.
    pub centroid: Vec3,
}

#[derive(Debug, Clone)]
pub struct ShaderGroup {
    pub textures: Vec<YtdTexture>,
    pub shaders: Vec<ShaderFx>,
}

#[derive(Debug, Clone)]
pub struct ShaderFx {
    pub name_hash: u32,
    pub file_name_hash: u32,
    pub render_bucket: u8,
    pub render_bucket_mask: u32,
    pub parameter_count: u8,
    pub texture_parameter_count: u8,
    pub parameters: Vec<ShaderParameter>,
}

#[derive(Debug, Clone)]
pub struct ShaderParameter {
    pub name_hash: u32,
    pub data_type: u8,
    pub data_pointer: u64,
    pub value: ShaderParameterValue,
}

#[derive(Debug, Clone)]
pub enum ShaderParameterValue {
    Texture { name: String, name_hash: u32 },
    Vectors(Vec<Vec4>),
    Pointer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodLevel {
    High,
    Medium,
    Low,
    VeryLow,
}

impl LodLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Medium => "medium",
            Self::Low => "low",
            Self::VeryLow => "verylow",
        }
    }
}

impl std::str::FromStr for LodLevel {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "high" => Ok(Self::High),
            "medium" => Ok(Self::Medium),
            "low" => Ok(Self::Low),
            "verylow" => Ok(Self::VeryLow),
            other => anyhow::bail!("unknown LOD level '{other}'"),
        }
    }
}

impl std::fmt::Display for LodLevel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct DrawableLod {
    pub level: LodLevel,
    pub models: Vec<DrawableModel>,
}

#[derive(Debug, Clone)]
pub struct DrawableModel {
    pub skeleton_binding: u32,
    pub render_mask_flags: u16,
    pub shader_mapping: Vec<u16>,
    pub geometries: Vec<DrawableGeometry>,
}

#[derive(Debug, Clone)]
pub struct DrawableGeometry {
    pub shader_id: u16,
    pub indices_count: u32,
    pub triangles_count: u32,
    pub vertices_count: u16,
    pub vertex_stride: u16,
    pub vertex_buffer: Option<VertexBuffer>,
    pub index_buffer: Option<IndexBuffer>,
}

pub use crate::vertex::{
    UnifiedVertex, VertexAttribute, VertexAttributeValue, VertexBuffer, VertexBufferLayout,
    VertexComponent, VertexComponentType, VertexDeclaration, VertexSemantic,
};

#[derive(Debug, Clone)]
pub struct IndexBuffer {
    pub indices_count: u32,
    pub indices_pointer: u64,
    pub indices: Vec<u32>,
}

// ─── Entry points ─────────────────────────────────────────────────────────────

/// Parses a YDR resource: a single drawable at the start of the system section.
pub fn parse_ydr(data: &[u8]) -> Result<Drawable> {
    let (system, graphics) = prepare_rsc7(data)?;
    let reader = ResReader { system: &system, graphics: &graphics };
    parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None)
}

/// Parses a YDD resource: a dictionary of hashed, named drawables.
pub fn parse_ydd(data: &[u8]) -> Result<Vec<DrawableEntry>> {
    let (system, graphics) = prepare_rsc7(data)?;
    let reader = ResReader { system: &system, graphics: &graphics };
    parse_ydd_from_reader(&reader)
}

/// Parses any drawable-bearing resource into a flat list of named entries.
///
/// `kind` is only a hint: a YDR whose header actually looks like a dictionary
/// (and the reverse) is parsed the way its bytes say, not the way it is named.
pub fn parse_drawables(data: &[u8], kind: DrawableKind) -> Result<Vec<DrawableEntry>> {
    if kind == DrawableKind::Yft {
        let fragment = crate::yft::parse_yft(data)?;
        return Ok(assemble_fragment_entries(fragment));
    }

    let (system, graphics) = prepare_rsc7(data)?;
    let reader = ResReader { system: &system, graphics: &graphics };
    let is_dictionary = looks_like_drawable_dictionary(&reader);

    let as_dictionary = match kind {
        DrawableKind::Ydd => is_dictionary || !looks_like_single_drawable(&reader),
        DrawableKind::Ydr => is_dictionary,
        DrawableKind::Yft => unreachable!("handled above"),
    };

    if as_dictionary {
        return parse_ydd_from_reader(&reader);
    }

    let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None)?;
    let hash = drawable.name_hash;
    Ok(vec![DrawableEntry { hash, name: entry_name(&drawable.name, hash), drawable }])
}

/// Assembles a fragment's main drawable and its extras into one flat, named
/// list. A fragment's extra drawables (damaged variants, attached props) are
/// pushed by [`crate::yft::parse_yft`] verbatim; when the resource has no
/// names array (or an empty one), every extra falls back to the same
/// resource-level name and hash — the same collision `make_names_unique`
/// already fixes for `.ydd` dictionaries — so it's run here too before the
/// list is handed back.
fn assemble_fragment_entries(fragment: crate::yft::Fragment) -> Vec<DrawableEntry> {
    let mut entries = Vec::new();
    if let Some(drawable) = fragment.drawable {
        let name = if fragment.name.is_empty() { drawable.name.clone() } else { fragment.name.clone() };
        let hash = drawable.name_hash;
        entries.push(DrawableEntry { hash, name: entry_name(&name, hash), drawable });
    }
    for extra in fragment.extra_drawables {
        let name = entry_name(&extra.name, extra.hash);
        entries.push(DrawableEntry { hash: extra.hash, name, drawable: extra.drawable });
    }

    let mut names: Vec<String> = entries.iter().map(|entry| entry.name.clone()).collect();
    let hashes: Vec<u32> = entries.iter().map(|entry| entry.hash).collect();
    make_names_unique(&mut names, &hashes);
    for (entry, name) in entries.iter_mut().zip(names) {
        entry.name = name;
    }

    entries
}

fn entry_name(name: &str, hash: u32) -> String {
    if name.is_empty() || name.eq_ignore_ascii_case("unknown") {
        format!("0x{hash:08X}")
    } else {
        name.to_string()
    }
}

pub(crate) fn parse_ydd_from_reader(reader: &ResReader<'_>) -> Result<Vec<DrawableEntry>> {
    let raw = reader
        .resolve(SYSTEM_BASE, 0x40)
        .context("system section too small for a drawable dictionary")?;

    let hashes_pointer = u64_le(raw, 0x20);
    let hashes_count = u16_le(raw, 0x28) as usize;
    let drawables_pointer = u64_le(raw, 0x30);
    let drawables_count = u16_le(raw, 0x38) as usize;

    let hashes = reader.read_u32_list(hashes_pointer, hashes_count).unwrap_or_default();
    let pointers = reader
        .read_u64_list(drawables_pointer, drawables_count)
        .context("drawable dictionary pointer array out of bounds")?;

    let mut entries = Vec::with_capacity(pointers.len());
    for (index, pointer) in pointers.iter().copied().enumerate() {
        if pointer == 0 {
            continue;
        }

        let hash = hashes.get(index).copied().unwrap_or(0);
        let drawable = parse_drawable_at(reader, pointer, 0xA8, 0xD0, Some(hash))?;
        let name = entry_name(&drawable.name, hash);
        entries.push(DrawableEntry { hash, name, drawable });
    }

    let mut names: Vec<String> = entries.iter().map(|entry| entry.name.clone()).collect();
    let entry_hashes: Vec<u32> = entries.iter().map(|entry| entry.hash).collect();
    make_names_unique(&mut names, &entry_hashes);
    for (entry, name) in entries.iter_mut().zip(names) {
        entry.name = name;
    }

    Ok(entries)
}

/// Drawables stored in a dictionary usually carry the *resource's* own name
/// string rather than their own, so several entries come back sharing one
/// name — which makes them indistinguishable to callers that name files after
/// them. Identity inside a dictionary is the hash, so every name held by more
/// than one entry is replaced with that entry's `0x…` hash (and, if even the
/// hashes collide, with the hash plus the entry's index). Names that are
/// already unique are left alone.
fn make_names_unique(names: &mut [String], hashes: &[u32]) {
    use std::collections::{HashMap, HashSet};

    let mut counts: HashMap<String, usize> = HashMap::new();
    for name in names.iter() {
        *counts.entry(name.to_lowercase()).or_insert(0) += 1;
    }

    let duplicated: HashSet<String> =
        counts.into_iter().filter(|(_, n)| *n > 1).map(|(name, _)| name).collect();
    if duplicated.is_empty() {
        return;
    }

    let mut used: HashSet<String> = names
        .iter()
        .map(|name| name.to_lowercase())
        .filter(|name| !duplicated.contains(name))
        .collect();

    for (index, name) in names.iter_mut().enumerate() {
        if !duplicated.contains(&name.to_lowercase()) {
            continue;
        }

        let hash = hashes.get(index).copied().unwrap_or(0);
        let mut candidate = format!("0x{hash:08X}");
        if !used.insert(candidate.to_lowercase()) {
            candidate = format!("0x{hash:08X}_{index}");
            used.insert(candidate.to_lowercase());
        }
        *name = candidate;
    }
}

/// True when the system section's header reads like a drawable dictionary:
/// a plausible pointer array whose entries actually resolve to drawables.
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

/// True when the system section's header reads like a lone drawable: it has to
/// be big enough, and at least one LOD list has to resolve.
fn looks_like_single_drawable(reader: &ResReader<'_>) -> bool {
    if reader.resolve(SYSTEM_BASE, 0xD0).is_none() {
        return false;
    }

    [0x50u64, 0x58, 0x60, 0x68, 0xA0].iter().any(|offset| {
        let pointer = u64_le(reader.system, *offset as usize);
        pointer != 0 && reader.read_pointer_list_header(pointer).is_some()
    })
}

// ─── Drawable parsing ─────────────────────────────────────────────────────────

/// Parses a drawable (or fragment drawable) at `va`.
///
/// `name_ptr_offset` and `struct_len` differ between a plain `rmcDrawable`
/// (0xA8 / 0xD0) and a `fragDrawable` (0x130 / 0x150), which keeps its name
/// pointer past the extra physics fields.
pub(crate) fn parse_drawable_at(
    reader: &ResReader<'_>,
    va: u64,
    name_ptr_offset: usize,
    struct_len: usize,
    fallback_hash: Option<u32>,
) -> Result<Drawable> {
    let raw = reader
        .resolve(va, struct_len)
        .with_context(|| format!("drawable out of bounds (va=0x{va:X})"))?;

    let shader_group_pointer = u64_le(raw, 0x10);
    let bounds = DrawableBounds {
        center: vec3_le(raw, 0x20),
        sphere_radius: f32_le(raw, 0x2C),
        box_min: vec3_le(raw, 0x30),
        box_max: vec3_le(raw, 0x40),
    };

    // Some resources leave the High LOD list empty and point at the models
    // through DrawableModelsPointer instead.
    let high_pointer = match u64_le(raw, 0x50) {
        0 => u64_le(raw, 0xA0),
        pointer => pointer,
    };
    let lod_pointers = [
        (LodLevel::High, high_pointer),
        (LodLevel::Medium, u64_le(raw, 0x58)),
        (LodLevel::Low, u64_le(raw, 0x60)),
        (LodLevel::VeryLow, u64_le(raw, 0x68)),
    ];
    let lod_distances = [
        f32_le(raw, 0x70),
        f32_le(raw, 0x74),
        f32_le(raw, 0x78),
        f32_le(raw, 0x7C),
    ];
    let render_masks = [
        u32_le(raw, 0x80),
        u32_le(raw, 0x84),
        u32_le(raw, 0x88),
        u32_le(raw, 0x8C),
    ];

    let name = reader.string_at(u64_le(raw, name_ptr_offset)).unwrap_or_default();
    let name_hash = fallback_hash.unwrap_or_else(|| rage_joaat(&name.to_ascii_lowercase()));

    let shader_group = if shader_group_pointer == 0 {
        None
    } else {
        Some(parse_shader_group_at(reader, shader_group_pointer)?)
    };

    let mut lods = Vec::new();
    for (level, lod_pointer) in lod_pointers {
        if lod_pointer == 0 {
            continue;
        }

        let lod = parse_lod_models_at(reader, level, lod_pointer)?;
        if !lod.models.is_empty() {
            lods.push(lod);
        }
    }

    Ok(Drawable {
        name,
        name_hash,
        bounds,
        lod_distances,
        render_masks,
        shader_group,
        lods,
    })
}

fn parse_shader_group_at(reader: &ResReader<'_>, va: u64) -> Result<ShaderGroup> {
    let raw = reader
        .resolve(va, 0x40)
        .with_context(|| format!("shader group out of bounds (va=0x{va:X})"))?;

    let texture_dictionary_pointer = u64_le(raw, 0x08);
    let shaders_pointer = u64_le(raw, 0x10);
    let shaders_count = u16_le(raw, 0x18) as usize;

    let textures = if texture_dictionary_pointer == 0 {
        Vec::new()
    } else {
        parse_texture_dict_at(reader, texture_dictionary_pointer)
            .context("embedded texture dictionary")?
    };

    let shader_pointers = reader
        .read_u64_list(shaders_pointer, shaders_count)
        .context("shader pointer array out of bounds")?;
    let mut shaders = Vec::with_capacity(shader_pointers.len());
    for shader_pointer in shader_pointers {
        if shader_pointer != 0 {
            shaders.push(parse_shader_fx_at(reader, shader_pointer)?);
        }
    }

    Ok(ShaderGroup { textures, shaders })
}

fn parse_shader_fx_at(reader: &ResReader<'_>, va: u64) -> Result<ShaderFx> {
    let raw = reader
        .resolve(va, 0x30)
        .with_context(|| format!("shader out of bounds (va=0x{va:X})"))?;

    let parameters_pointer = u64_le(raw, 0x00);
    let parameter_count = raw[0x10];

    Ok(ShaderFx {
        name_hash: u32_le(raw, 0x08),
        file_name_hash: u32_le(raw, 0x18),
        render_bucket: raw[0x11],
        render_bucket_mask: u32_le(raw, 0x20),
        parameter_count,
        texture_parameter_count: raw[0x27],
        parameters: parse_shader_parameters_at(reader, parameters_pointer, parameter_count),
    })
}

/// Shader parameters are stored as a run of 16-byte entries, then the value
/// block they point into, then one `u32` name hash per parameter.
fn parse_shader_parameters_at(
    reader: &ResReader<'_>,
    va: u64,
    count: u8,
) -> Vec<ShaderParameter> {
    let count = count as usize;
    if va == 0 || count == 0 {
        return Vec::new();
    }

    let Some(raw) = reader.resolve(va, count * 16) else {
        return Vec::new();
    };

    let mut params = Vec::with_capacity(count);
    let mut value_bytes_len = 0usize;

    for index in 0..count {
        let offset = index * 16;
        let data_type = raw[offset];
        let data_pointer = u64_le(raw, offset + 8);
        value_bytes_len += data_type as usize * 16;

        params.push(ShaderParameter {
            name_hash: 0,
            data_type,
            data_pointer,
            value: parse_shader_parameter_value(reader, data_type, data_pointer),
        });
    }

    let hashes_pointer = va + count as u64 * 16 + value_bytes_len as u64;
    let hashes = reader.read_u32_list(hashes_pointer, count).unwrap_or_default();
    for (param, hash) in params.iter_mut().zip(hashes) {
        param.name_hash = hash;
    }

    params
}

fn parse_shader_parameter_value(
    reader: &ResReader<'_>,
    data_type: u8,
    data_pointer: u64,
) -> ShaderParameterValue {
    if data_pointer == 0 {
        return ShaderParameterValue::Pointer;
    }

    if data_type == 0 {
        // A texture parameter points at a grcTexture whose name pointer is 0x28 in.
        if let Some(raw) = reader.resolve(data_pointer, 0x50) {
            let name = reader.string_at(u64_le(raw, 0x28)).unwrap_or_default();
            let name_hash = rage_joaat(&name.to_ascii_lowercase());
            return ShaderParameterValue::Texture { name, name_hash };
        }
        return ShaderParameterValue::Pointer;
    }

    let len = data_type as usize * 16;
    if let Some(raw) = reader.resolve(data_pointer, len) {
        let vectors = (0..data_type as usize).map(|i| vec4_le(raw, i * 16)).collect();
        return ShaderParameterValue::Vectors(vectors);
    }

    ShaderParameterValue::Pointer
}

fn parse_lod_models_at(
    reader: &ResReader<'_>,
    level: LodLevel,
    va: u64,
) -> Result<DrawableLod> {
    let Some(header) = reader.read_pointer_list_header(va) else {
        return Ok(DrawableLod { level, models: Vec::new() });
    };

    let pointer_count = if header.count == 0 { header.capacity } else { header.count };
    let model_pointers = reader
        .read_u64_list(header.pointer, pointer_count as usize)
        .context("model pointer array out of bounds")?;

    let mut models = Vec::with_capacity(model_pointers.len());
    for model_pointer in model_pointers {
        if model_pointer != 0 {
            models.push(parse_model_at(reader, model_pointer)?);
        }
    }

    Ok(DrawableLod { level, models })
}

fn parse_model_at(reader: &ResReader<'_>, va: u64) -> Result<DrawableModel> {
    let raw = reader
        .resolve(va, 0x30)
        .with_context(|| format!("drawable model out of bounds (va=0x{va:X})"))?;

    let geometries_pointer = u64_le(raw, 0x08);
    let geometries_count = u16_le(raw, 0x10) as usize;
    let shader_mapping_pointer = u64_le(raw, 0x20);
    let skeleton_binding = u32_le(raw, 0x28);
    let render_mask_flags = u16_le(raw, 0x2C);

    let shader_mapping = reader
        .read_u16_list(shader_mapping_pointer, geometries_count)
        .unwrap_or_default();
    let geometry_pointers = reader
        .read_u64_list(geometries_pointer, geometries_count)
        .context("geometry pointer array out of bounds")?;

    let mut geometries = Vec::with_capacity(geometry_pointers.len());
    for (index, geometry_pointer) in geometry_pointers.iter().copied().enumerate() {
        if geometry_pointer == 0 {
            continue;
        }

        let shader_id = shader_mapping.get(index).copied().unwrap_or(0);
        geometries.push(parse_geometry_at(reader, geometry_pointer, shader_id)?);
    }

    Ok(DrawableModel {
        skeleton_binding,
        render_mask_flags,
        shader_mapping,
        geometries,
    })
}

fn parse_geometry_at(
    reader: &ResReader<'_>,
    va: u64,
    shader_id: u16,
) -> Result<DrawableGeometry> {
    let raw = reader
        .resolve(va, 0x98)
        .with_context(|| format!("drawable geometry out of bounds (va=0x{va:X})"))?;

    let vertex_buffer_pointer = u64_le(raw, 0x18);
    let index_buffer_pointer = u64_le(raw, 0x38);
    let indices_count = u32_le(raw, 0x58);
    let triangles_count = u32_le(raw, 0x5C);
    let mut vertices_count = u16_le(raw, 0x60);
    let vertex_stride = u16_le(raw, 0x70);
    let inline_data_pointer = u64_le(raw, 0x78);

    let vertex_buffer = parse_vertex_buffer_at(
        reader,
        vertex_buffer_pointer,
        inline_data_pointer,
        vertices_count as u32,
        vertex_stride,
    );
    let index_buffer = parse_index_buffer_at(reader, index_buffer_pointer);

    if vertices_count == 0 {
        if let Some(buffer) = &vertex_buffer {
            vertices_count = buffer.vertex_count.min(u16::MAX as u32) as u16;
        }
    }

    Ok(DrawableGeometry {
        shader_id,
        indices_count,
        triangles_count,
        vertices_count,
        vertex_stride,
        vertex_buffer,
        index_buffer,
    })
}

// ─── Index buffers ────────────────────────────────────────────────────────────

fn parse_index_buffer_at(reader: &ResReader<'_>, va: u64) -> Option<IndexBuffer> {
    let raw = reader.resolve(va, 0x60)?;
    let indices_count = u32_le(raw, 0x08);

    // Legacy layout: 16-bit indices behind the pointer at 0x10.
    let legacy_pointer = u64_le(raw, 0x10);
    if let Some(indices) = read_indices(reader, legacy_pointer, indices_count, 2) {
        return Some(IndexBuffer { indices_count, indices_pointer: legacy_pointer, indices });
    }

    // Gen9 layout: an explicit index size, and the data one field further on.
    let gen9_pointer = u64_le(raw, 0x18);
    let gen9_index_size = u16_le(raw, 0x0C);
    if let Some(indices) = read_indices(reader, gen9_pointer, indices_count, gen9_index_size) {
        return Some(IndexBuffer { indices_count, indices_pointer: gen9_pointer, indices });
    }

    Some(IndexBuffer {
        indices_count,
        indices_pointer: legacy_pointer,
        indices: Vec::new(),
    })
}

fn read_indices(
    reader: &ResReader<'_>,
    data_pointer: u64,
    count: u32,
    index_size: u16,
) -> Option<Vec<u32>> {
    if data_pointer == 0 || count == 0 {
        return None;
    }

    match index_size {
        2 => reader
            .read_u16_list(data_pointer, count as usize)
            .map(|list| list.into_iter().map(u32::from).collect()),
        4 => reader.read_u32_list(data_pointer, count as usize),
        _ => None,
    }
}

// ─── Drawable helpers ─────────────────────────────────────────────────────────

impl DrawableModel {
    /// Index of the bone this model hangs off: the top byte of the
    /// skeleton binding.
    pub fn bone_index(&self) -> usize {
        ((self.skeleton_binding >> 24) & 0xFF) as usize
    }

    /// True for skinned meshes, whose vertices are already in skeleton
    /// space and must not be moved with a single bone.
    pub fn is_skinned(&self) -> bool {
        (self.skeleton_binding >> 8) & 0xFF != 0
    }
}

impl Drawable {
    pub fn lod(&self, level: LodLevel) -> Option<&DrawableLod> {
        self.lods.iter().find(|lod| lod.level == level)
    }

    /// The LOD to render: High when it has models, otherwise the first
    /// non-empty one.
    pub fn best_lod(&self) -> Option<&DrawableLod> {
        self.lod(LodLevel::High)
            .filter(|lod| !lod.models.is_empty())
            .or_else(|| self.lods.iter().find(|lod| !lod.models.is_empty()))
    }

    pub fn shader(&self, id: u16) -> Option<&ShaderFx> {
        self.shader_group.as_ref()?.shaders.get(id as usize)
    }

    /// The texture bound to `param_hash` on a shader, if that parameter is a
    /// texture parameter with a name.
    pub fn texture_parameter(&self, shader_id: u16, param_hash: u32) -> Option<&str> {
        let shader = self.shader(shader_id)?;
        shader.parameters.iter().find_map(|param| match &param.value {
            ShaderParameterValue::Texture { name, .. }
                if param.name_hash == param_hash && !name.is_empty() =>
            {
                Some(name.as_str())
            }
            _ => None,
        })
    }

    /// The albedo texture of a shader: its `DiffuseSampler`, or failing that
    /// the older `TextureSampler` name.
    pub fn diffuse_texture_name(&self, shader_id: u16) -> Option<&str> {
        self.texture_parameter(shader_id, DIFFUSE_SAMPLER)
            .or_else(|| self.texture_parameter(shader_id, TEXTURE_SAMPLER))
    }

    /// The drawable's stored bounds when they actually describe the LOD's
    /// vertices, and bounds rebuilt from those vertices when they do not. The
    /// flag says which of the two was returned.
    ///
    /// Stored bounds cannot simply be trusted: on skinned drawables (ped
    /// components, for one) they are the *skeleton's* bounds, several times
    /// the size of the part itself and centred elsewhere, so a camera fitted
    /// to them shrinks the model to a speck or misses it altogether. Nor can
    /// they simply be discarded, since replacing them also replaces the stored
    /// centre and sphere radius, which the camera uses. So they are checked
    /// against the vertices and kept whole whenever they agree.
    pub fn bounds_or_computed(&self, lod: &DrawableLod) -> (DrawableBounds, bool) {
        let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut seen = false;

        for model in &lod.models {
            for geometry in &model.geometries {
                let Some(buffer) = &geometry.vertex_buffer else { continue };
                let Ok(vertices) = buffer.to_unified_vertices() else { continue };
                for vertex in vertices {
                    let p = vertex.position;
                    if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
                        continue;
                    }
                    min = min.min(p);
                    max = max.max(p);
                    seen = true;
                }
            }
        }

        if !seen || min == max || bounds_describe(&self.bounds, min, max) {
            return (self.bounds.clone(), false);
        }

        let center = (min + max) * 0.5;
        let radius = (max - min).length() * 0.5;
        (
            DrawableBounds { center, sphere_radius: radius, box_min: min, box_max: max },
            true,
        )
    }

    /// Per-geometry vertex bounds for `lod`, in model then geometry order.
    ///
    /// `bounds_or_computed` folds every geometry's vertices into one box; this
    /// keeps them separate, which is what actually shows a drawable authored
    /// as several pieces scattered far apart (see `render::cluster`) instead
    /// of one compact mesh. Geometries with no decodable vertex buffer, or
    /// with no finite vertex, are skipped.
    pub fn geometry_bounds(&self, lod: &DrawableLod) -> Vec<GeometryBounds> {
        let mut out = Vec::new();

        for (model_index, model) in lod.models.iter().enumerate() {
            for (geometry_index, geometry) in model.geometries.iter().enumerate() {
                let Some(buffer) = &geometry.vertex_buffer else { continue };
                let Ok(vertices) = buffer.to_unified_vertices() else { continue };

                let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
                let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
                let mut sum = (0.0f64, 0.0f64, 0.0f64);
                let mut seen = 0usize;

                for vertex in &vertices {
                    let p = vertex.position;
                    if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
                        continue;
                    }
                    min = min.min(p);
                    max = max.max(p);
                    sum.0 += p.x as f64;
                    sum.1 += p.y as f64;
                    sum.2 += p.z as f64;
                    seen += 1;
                }

                if seen == 0 {
                    continue;
                }

                let triangles = if geometry.indices_count > 0 {
                    geometry.indices_count as usize / 3
                } else {
                    geometry.index_buffer.as_ref().map(|ib| ib.indices.len()).unwrap_or(0) / 3
                };

                out.push(GeometryBounds {
                    model: model_index,
                    geometry: geometry_index,
                    shader_id: geometry.shader_id,
                    vertices: seen,
                    triangles,
                    min,
                    max,
                    centroid: Vec3::new(
                        (sum.0 / seen as f64) as f32,
                        (sum.1 / seen as f64) as f32,
                        (sum.2 / seen as f64) as f32,
                    ),
                });
            }
        }

        out
    }

    /// Total triangles across a LOD, derived from the geometries' index counts.
    pub fn triangle_count(&self, lod: &DrawableLod) -> usize {
        lod.models
            .iter()
            .flat_map(|model| &model.geometries)
            .map(|geometry| {
                let indices = if geometry.indices_count > 0 {
                    geometry.indices_count as usize
                } else {
                    geometry.index_buffer.as_ref().map(|ib| ib.indices.len()).unwrap_or(0)
                };
                indices / 3
            })
            .sum()
    }

    pub fn shader_count(&self) -> usize {
        self.shader_group.as_ref().map(|group| group.shaders.len()).unwrap_or(0)
    }

    pub fn embedded_texture_count(&self) -> usize {
        self.shader_group.as_ref().map(|group| group.textures.len()).unwrap_or(0)
    }

    pub fn model_count(&self) -> usize {
        self.lods.iter().map(|lod| lod.models.len()).sum()
    }

    pub fn geometry_count(&self) -> usize {
        self.lods
            .iter()
            .flat_map(|lod| &lod.models)
            .map(|model| model.geometries.len())
            .sum()
    }
}

/// True when `bounds` is a usable description of the box between `min` and
/// `max`: finite, with a positive radius, and with both corners agreeing to
/// within a thousandth of the box's diagonal.
///
/// Honest bounds land far inside that: a prop and a car measured out of the
/// retail archives disagree with their own vertices by 1.3e-4 and 4.2e-5 of
/// the diagonal, while the skeleton bounds stored on ped components are out by
/// 0.65 to 2.7 — three orders of magnitude either side of the line.
fn bounds_describe(bounds: &DrawableBounds, min: Vec3, max: Vec3) -> bool {
    let finite = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();

    if !finite(bounds.box_min)
        || !finite(bounds.box_max)
        || !finite(bounds.center)
        || !bounds.sphere_radius.is_finite()
        || bounds.sphere_radius <= 0.0
    {
        return false;
    }

    let tolerance = (max - min).length() * 1e-3;
    let close = |a: Vec3, b: Vec3| {
        (a.x - b.x).abs() <= tolerance
            && (a.y - b.y).abs() <= tolerance
            && (a.z - b.z).abs() <= tolerance
    };

    close(bounds.box_min, min) && close(bounds.box_max, max)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::resource::{GRAPHICS_BASE, SYSTEM_BASE};
    use crate::writer::rage_joaat;

    fn sections_reader<'a>(system: &'a [u8], graphics: &'a [u8]) -> ResReader<'a> {
        ResReader { system, graphics }
    }

    fn stub_drawable(name: &str) -> Drawable {
        Drawable {
            name: name.to_string(),
            name_hash: rage_joaat(name),
            bounds: DrawableBounds {
                center: Vec3::new(0.0, 0.0, 0.0),
                sphere_radius: 0.0,
                box_min: Vec3::new(0.0, 0.0, 0.0),
                box_max: Vec3::new(0.0, 0.0, 0.0),
            },
            lod_distances: [0.0; 4],
            render_masks: [0; 4],
            shader_group: None,
            lods: Vec::new(),
        }
    }

    /// The skeleton binding packs the bone index in its top byte and a
    /// skin flag in its second byte (CodeWalker's `Renderable.Init`).
    #[test]
    fn model_bone_index_and_skin_flag_come_from_the_binding() {
        let model = |skeleton_binding| DrawableModel {
            skeleton_binding,
            render_mask_flags: 0,
            shader_mapping: Vec::new(),
            geometries: Vec::new(),
        };
        assert_eq!(model(0).bone_index(), 0);
        assert_eq!(model(5 << 24).bone_index(), 5);
        assert!(!model(5 << 24).is_skinned());
        assert!(model((5 << 24) | (1 << 8)).is_skinned());
        assert_eq!(model((5 << 24) | (1 << 8)).bone_index(), 5);
    }

    #[test]
    fn assemble_fragment_entries_gives_unnamed_extras_distinct_names() {
        // A fragment whose extras have no names array: `parse_yft` falls back
        // to the resource-level name for every extra, so without
        // `make_names_unique` both would come back identical and one would
        // silently overwrite the other when written out as files.
        let fragment = crate::yft::Fragment {
            name: "frag".to_string(),
            bound_center: Vec3::new(0.0, 0.0, 0.0),
            bound_radius: 0.0,
            drawable: None,
            extra_drawables: vec![
                DrawableEntry { hash: rage_joaat("frag"), name: "frag".to_string(), drawable: stub_drawable("frag") },
                DrawableEntry { hash: rage_joaat("frag"), name: "frag".to_string(), drawable: stub_drawable("frag") },
            ],
            bone_transforms: Vec::new(),
            children: Vec::new(),
        };

        let entries = assemble_fragment_entries(fragment);

        assert_eq!(entries.len(), 2);
        assert_ne!(entries[0].name, entries[1].name);
    }

    #[test]
    fn parses_minimal_ydr() {
        let (system, graphics) = minimal_ydr_sections(false);
        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None)
            .expect("fixture should parse");

        assert_eq!(drawable.name, "test_drawable");
        assert_eq!(drawable.name_hash, rage_joaat("test_drawable"));
        assert_eq!(drawable.bounds.sphere_radius, 10.0);
        assert_eq!(drawable.lod_distances, [100.0, 50.0, 25.0, 12.5]);

        let group = drawable.shader_group.as_ref().expect("shader group");
        assert_eq!(group.shaders.len(), 1);
        let shader = &group.shaders[0];
        assert_eq!(shader.name_hash, 0x1111_1111);
        assert_eq!(shader.file_name_hash, 0x2222_2222);
        assert_eq!(shader.parameter_count, 1);
        assert_eq!(shader.render_bucket, 2);
        assert_eq!(shader.parameters[0].data_type, 1);
        assert_eq!(shader.parameters[0].name_hash, 0xAABB_CCDD);
        assert!(matches!(
            shader.parameters[0].value,
            ShaderParameterValue::Vectors(ref v)
                if v.len() == 1 && v[0] == Vec4::new(1.0, 2.0, 3.0, 4.0)
        ));

        assert_eq!(drawable.lods.len(), 1);
        let lod = &drawable.lods[0];
        assert_eq!(lod.level, LodLevel::High);
        assert_eq!(lod.models.len(), 1);
        assert_eq!(lod.models[0].geometries.len(), 1);
        assert_eq!(lod.models[0].render_mask_flags, 0x0102);
        assert_eq!(drawable.triangle_count(lod), 1);
        assert_eq!(drawable.best_lod().map(|l| l.level), Some(LodLevel::High));
        assert!(drawable.lod(LodLevel::Medium).is_none());

        let geometry = &lod.models[0].geometries[0];
        assert_eq!(geometry.shader_id, 0);
        assert_eq!(geometry.vertices_count, 3);
        assert_eq!(geometry.indices_count, 3);
        assert_eq!(geometry.triangles_count, 1);
        assert_eq!(geometry.vertex_stride, 12);

        let vertex_buffer = geometry.vertex_buffer.as_ref().expect("vertex buffer");
        assert_eq!(vertex_buffer.layout, VertexBufferLayout::Legacy);
        assert_eq!(vertex_buffer.data.len(), 36);
        assert_eq!(vertex_buffer.info_pointer, SYSTEM_BASE + 0x760);

        let declaration = vertex_buffer.declaration.as_ref().expect("declaration");
        assert_eq!(declaration.flags, 1);
        assert_eq!(declaration.stride, 12);
        assert_eq!(declaration.count, 1);
        assert_eq!(declaration.types, 0x7755_5555_5599_6996);
        assert_eq!(declaration.declaration_id(), 0x6);
        assert_eq!(declaration.components.len(), 1);
        assert_eq!(declaration.components[0].semantic, VertexSemantic::Position);
        assert_eq!(
            declaration.components[0].component_type,
            VertexComponentType::Float3
        );
        assert_eq!(declaration.components[0].offset, 0);
        assert_eq!(declaration.components[0].size, 12);
        assert_eq!(declaration.components[0].component_count, 3);

        let attributes = vertex_buffer.read_vertex_attributes(0).unwrap();
        assert_eq!(attributes.len(), 1);
        assert_eq!(
            attributes[0].value,
            VertexAttributeValue::Float3(Vec3::new(1.0, 2.0, 3.0))
        );

        assert_eq!(
            geometry.index_buffer.as_ref().unwrap().indices,
            vec![0u32, 1, 2]
        );

        let unified = vertex_buffer.to_unified_vertices().unwrap();
        assert_eq!(unified.len(), 3);
        assert_eq!(unified[0].position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(unified[2].position, Vec3::new(7.0, 8.0, 9.0));
        assert_eq!(unified[0].normal, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(unified[0].color0, [255, 255, 255, 255]);
    }

    #[test]
    fn parses_minimal_ydd() {
        let (system, graphics) = minimal_ydr_sections(true);
        let reader = sections_reader(&system, &graphics);
        let entries = parse_ydd_from_reader(&reader).expect("fixture should parse");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].hash, 0x1234_5678);
        assert_eq!(entries[0].name, "test_drawable");
        assert_eq!(entries[0].drawable.name, "test_drawable");
        assert_eq!(entries[0].drawable.name_hash, 0x1234_5678);
    }

    /// Every drawable in a real dictionary (e.g. a ped component .ydd) points
    /// at the resource's own name string, so all of them used to come back
    /// named after the file and callers writing "<name>.png" overwrote a
    /// single image. Shared names must fall back to the entry's hash.
    #[test]
    fn dictionary_entries_sharing_a_name_get_unique_hash_names() {
        let (mut system, graphics) = minimal_ydr_sections(true);

        // Two hashes and two pointers, both pointing at the one drawable, so
        // both entries parse with the same embedded name.
        write_u16(&mut system, 0x28, 2);
        write_u16(&mut system, 0x2A, 2);
        write_u32(&mut system, 0x44, 0xAABB_CCDD);
        write_u16(&mut system, 0x38, 2);
        write_u16(&mut system, 0x3A, 2);
        write_u64(&mut system, 0x58, SYSTEM_BASE + 0x100);

        let reader = sections_reader(&system, &graphics);
        let entries = parse_ydd_from_reader(&reader).expect("fixture should parse");

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].hash, 0x1234_5678);
        assert_eq!(entries[1].hash, 0xAABB_CCDD);
        assert_eq!(entries[0].name, "0x12345678");
        assert_eq!(entries[1].name, "0xAABBCCDD");
        // The underlying drawable still reports what the file says.
        assert_eq!(entries[0].drawable.name, "test_drawable");
    }

    #[test]
    fn unique_names_are_left_alone_and_hash_collisions_still_separate() {
        let mut names = vec!["alpha".to_string(), "beta".to_string()];
        make_names_unique(&mut names, &[1, 2]);
        assert_eq!(names, vec!["alpha", "beta"]);

        let mut names = vec!["same".to_string(), "same".to_string(), "other".to_string()];
        make_names_unique(&mut names, &[7, 7, 9]);
        assert_eq!(names, vec!["0x00000007", "0x00000007_1", "other"]);
    }

    /// A pointer into neither section: the fallbacks below have to trigger
    /// on a stream that fails to resolve, not only on a null pointer.
    const NOWHERE: u64 = SYSTEM_BASE + 0x10_0000;

    #[test]
    fn datapointer2_fallback() {
        let (mut system, graphics) = minimal_ydr_sections(false);
        // Point DataPointer1 off the end and move the stream to DataPointer2.
        write_u64(&mut system, 0x480 + 0x10, NOWHERE);
        write_u64(&mut system, 0x480 + 0x20, GRAPHICS_BASE);

        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();
        let buffer = drawable.lods[0].models[0].geometries[0]
            .vertex_buffer
            .as_ref()
            .expect("vertex buffer");

        assert_eq!(buffer.layout, VertexBufferLayout::LegacyData2);
        assert_eq!(buffer.data.len(), 36);
        assert!(buffer.declaration.is_some());
    }

    #[test]
    fn gen9_vertex_buffer_fallback() {
        let (mut system, graphics) = minimal_ydr_sections(false);
        // Break both legacy data pointers, then lay out the Gen9 fields.
        write_u64(&mut system, 0x480 + 0x10, NOWHERE);
        write_u64(&mut system, 0x480 + 0x20, NOWHERE);
        write_u32(&mut system, 0x480 + 0x08, 3); // vertex count
        write_u16(&mut system, 0x480 + 0x0C, 12); // stride
        write_u64(&mut system, 0x480 + 0x18, GRAPHICS_BASE); // data

        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();
        let buffer = drawable.lods[0].models[0].geometries[0]
            .vertex_buffer
            .as_ref()
            .expect("vertex buffer");

        assert_eq!(buffer.layout, VertexBufferLayout::Gen9);
        assert_eq!(buffer.vertex_count, 3);
        assert_eq!(buffer.vertex_stride, 12);
        assert_eq!(buffer.data.len(), 36);
        assert!(buffer.declaration.is_none());
        // Gen9 stores no declaration, so the legacy info slot is meaningless
        // and must not be reported as though it pointed at one.
        assert_eq!(buffer.info_pointer, 0);
    }

    /// Gen9 index buffers carry an explicit index size and keep the data one
    /// field further along; with a 4-byte size the indices are read as u32.
    #[test]
    fn gen9_index_buffer_reads_u32_indices() {
        let (mut system, mut graphics) = minimal_ydr_sections(false);
        write_u64(&mut system, 0x580 + 0x10, NOWHERE);
        write_u16(&mut system, 0x580 + 0x0C, 4);
        write_u64(&mut system, 0x580 + 0x18, GRAPHICS_BASE + 0x100);
        write_u32(&mut graphics, 0x100, 2);
        write_u32(&mut graphics, 0x104, 0x0001_0000);
        write_u32(&mut graphics, 0x108, 1);

        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();
        let indices = &drawable.lods[0].models[0].geometries[0].index_buffer.as_ref().unwrap().indices;

        assert_eq!(*indices, vec![2, 0x0001_0000, 1]);
    }

    /// Some resources leave the High LOD list null and point at the models
    /// through DrawableModelsPointer (0xA0) instead.
    #[test]
    fn high_lod_falls_back_to_the_models_pointer() {
        let (mut system, graphics) = minimal_ydr_sections(false);
        write_u64(&mut system, 0x50, 0);
        write_u64(&mut system, 0xA0, SYSTEM_BASE + 0x200);

        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();

        assert_eq!(drawable.lods.len(), 1);
        assert_eq!(drawable.lods[0].level, LodLevel::High);
        assert_eq!(drawable.lods[0].models.len(), 1);
    }

    #[test]
    fn geometry_inline_fallback() {
        let (mut system, graphics) = minimal_ydr_sections(false);
        // No usable vertex buffer struct: fall back to the inline geometry data.
        write_u64(&mut system, 0x300 + 0x18, NOWHERE);

        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();
        let buffer = drawable.lods[0].models[0].geometries[0]
            .vertex_buffer
            .as_ref()
            .expect("vertex buffer");

        assert_eq!(buffer.layout, VertexBufferLayout::GeometryInline);
        assert_eq!(buffer.vertex_count, 3);
        assert_eq!(buffer.vertex_stride, 12);
        assert_eq!(buffer.data.len(), 36);
        assert!(buffer.declaration.is_none());
    }

    #[test]
    fn missing_declaration_yields_positions() {
        let (mut system, graphics) = minimal_ydr_sections(false);
        write_u64(&mut system, 0x300 + 0x18, 0);

        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();
        let buffer = drawable.lods[0].models[0].geometries[0]
            .vertex_buffer
            .as_ref()
            .unwrap();
        assert!(buffer.declaration.is_none());

        let unified = buffer
            .to_unified_vertices()
            .expect("positions even without a declaration");
        assert_eq!(unified.len(), 3);
        assert_eq!(unified[0].position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(unified[1].position, Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(unified[2].position, Vec3::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn diffuse_sampler_lookup() {
        let (mut system, graphics) = minimal_ydr_sections(false);

        // Turn the single shader parameter into a texture parameter.
        write_u8(&mut system, 0x700, 0); // data_type 0 => texture
        write_u64(&mut system, 0x700 + 0x08, SYSTEM_BASE + 0x7B0);
        // Texture struct: name pointer at 0x28.
        write_u64(&mut system, 0x7B0 + 0x28, SYSTEM_BASE + 0x7A0);
        system[0x7A0..0x7A4].copy_from_slice(b"foo\0");
        // With an empty value block the hash array sits right after the parameters.
        write_u32(&mut system, 0x710, DIFFUSE_SAMPLER);

        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();

        assert_eq!(drawable.diffuse_texture_name(0), Some("foo"));
        assert_eq!(drawable.texture_parameter(0, DIFFUSE_SAMPLER), Some("foo"));
        assert_eq!(drawable.texture_parameter(0, BUMP_SAMPLER), None);
        assert_eq!(drawable.diffuse_texture_name(7), None);
    }

    #[test]
    fn bounds_recomputed_when_degenerate() {
        let (system, graphics) = minimal_ydr_sections(false);
        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();

        // The fixture leaves box_min == box_max == 0, so bounds must be rebuilt.
        let lod = drawable.best_lod().unwrap();
        let (bounds, computed) = drawable.bounds_or_computed(lod);
        assert!(computed);
        assert_eq!(bounds.box_min, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(bounds.box_max, Vec3::new(7.0, 8.0, 9.0));
        assert_eq!(bounds.center, Vec3::new(4.0, 5.0, 6.0));
        let expected_radius = (108.0f32).sqrt() / 2.0;
        assert!((bounds.sphere_radius - expected_radius).abs() < 1e-5);
    }

    /// The fixture's three vertices span (1,2,3)..(7,8,9). Writes a stored
    /// box/centre/radius into the drawable header, offset from that box by
    /// `slack` on every corner.
    fn ydr_with_stored_bounds(slack: f32) -> (Vec<u8>, Vec<u8>) {
        let (mut system, graphics) = minimal_ydr_sections(false);
        let (min, max) = (Vec3::new(1.0, 2.0, 3.0), Vec3::new(7.0, 8.0, 9.0));
        let center = (min + max) * 0.5;

        write_f32(&mut system, 0x20, center.x);
        write_f32(&mut system, 0x24, center.y);
        write_f32(&mut system, 0x28, center.z);
        write_f32(&mut system, 0x2C, (max - min).length() * 0.5);
        write_f32(&mut system, 0x30, min.x - slack);
        write_f32(&mut system, 0x34, min.y - slack);
        write_f32(&mut system, 0x38, min.z - slack);
        write_f32(&mut system, 0x40, max.x + slack);
        write_f32(&mut system, 0x44, max.y + slack);
        write_f32(&mut system, 0x48, max.z + slack);

        (system, graphics)
    }

    /// Stored bounds that do describe the vertices are handed back whole —
    /// centre and sphere radius included, since the camera uses both, and
    /// recomputing them moves the frame even when the box is right.
    #[test]
    fn stored_bounds_that_match_the_vertices_are_kept() {
        // Well inside the tolerance: the diagonal is sqrt(108) ~ 10.4, so a
        // thousandth of it is ~1e-2.
        let (system, graphics) = ydr_with_stored_bounds(0.001);
        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();

        let lod = drawable.best_lod().unwrap();
        let (bounds, computed) = drawable.bounds_or_computed(lod);
        assert!(!computed, "matching stored bounds should be kept");
        assert_eq!(bounds.box_min, drawable.bounds.box_min);
        assert_eq!(bounds.box_max, drawable.bounds.box_max);
        assert_eq!(bounds.center, drawable.bounds.center);
        assert_eq!(bounds.sphere_radius, drawable.bounds.sphere_radius);
    }

    /// Skinned drawables (ped components) store the *skeleton's* bounds, which
    /// are far larger than the part and centred elsewhere. Framing by them left
    /// the component a speck in the corner, or off-frame entirely, so the
    /// vertices must win even when the stored bounds look perfectly valid.
    #[test]
    fn oversized_stored_bounds_lose_to_the_vertices() {
        let (mut system, graphics) = minimal_ydr_sections(false);
        // A plausible but much too large stored box, centred away from the mesh.
        write_f32(&mut system, 0x20, 0.0);
        write_f32(&mut system, 0x24, 0.0);
        write_f32(&mut system, 0x28, 50.0);
        write_f32(&mut system, 0x2C, 100.0);
        write_f32(&mut system, 0x30, -100.0);
        write_f32(&mut system, 0x34, -100.0);
        write_f32(&mut system, 0x38, -50.0);
        write_f32(&mut system, 0x40, 100.0);
        write_f32(&mut system, 0x44, 100.0);
        write_f32(&mut system, 0x48, 150.0);

        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();
        assert_eq!(drawable.bounds.sphere_radius, 100.0);

        let lod = drawable.best_lod().unwrap();
        let (bounds, computed) = drawable.bounds_or_computed(lod);
        assert!(computed, "vertex bounds should win over the stored ones");
        assert_eq!(bounds.box_min, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(bounds.box_max, Vec3::new(7.0, 8.0, 9.0));
    }

    /// The line between the two: a box a whole unit out on every corner is a
    /// tenth of this mesh's diagonal, far past the thousandth allowed.
    #[test]
    fn stored_bounds_just_outside_the_tolerance_are_replaced() {
        let (system, graphics) = ydr_with_stored_bounds(1.0);
        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();

        let lod = drawable.best_lod().unwrap();
        let (bounds, computed) = drawable.bounds_or_computed(lod);
        assert!(computed);
        assert_eq!(bounds.box_min, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(bounds.box_max, Vec3::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn sampler_hashes_match_joaat() {
        assert_eq!(rage_joaat("diffusesampler"), DIFFUSE_SAMPLER);
        assert_eq!(rage_joaat("bumpsampler"), BUMP_SAMPLER);
        assert_eq!(rage_joaat("specsampler"), SPEC_SAMPLER);
        assert_eq!(rage_joaat("texturesampler"), TEXTURE_SAMPLER);
    }

    /// Drawables converted from the GTA IV / Max Payne pipelines bind their
    /// albedo as `TextureSampler` instead of `DiffuseSampler`; the lookup
    /// falls back to it so those do not render untextured.
    #[test]
    fn texture_sampler_is_a_fallback_for_the_diffuse_lookup() {
        let (mut system, graphics) = minimal_ydr_sections(false);
        write_u8(&mut system, 0x700, 0);
        write_u64(&mut system, 0x700 + 0x08, SYSTEM_BASE + 0x7B0);
        write_u64(&mut system, 0x7B0 + 0x28, SYSTEM_BASE + 0x7A0);
        system[0x7A0..0x7A4].copy_from_slice(b"foo\0");
        write_u32(&mut system, 0x710, TEXTURE_SAMPLER);

        let reader = sections_reader(&system, &graphics);
        let drawable = parse_drawable_at(&reader, SYSTEM_BASE, 0xA8, 0xD0, None).unwrap();

        assert_eq!(drawable.diffuse_texture_name(0), Some("foo"));
        assert_eq!(drawable.texture_parameter(0, DIFFUSE_SAMPLER), None);
    }

    #[test]
    fn drawable_kind_from_extension() {
        assert_eq!(DrawableKind::from_extension("ydr"), Some(DrawableKind::Ydr));
        assert_eq!(DrawableKind::from_extension(".YDR"), Some(DrawableKind::Ydr));
        assert_eq!(DrawableKind::from_extension("Ydd"), Some(DrawableKind::Ydd));
        assert_eq!(DrawableKind::from_extension(".yft"), Some(DrawableKind::Yft));
        assert_eq!(DrawableKind::from_extension("ytd"), None);
    }

    /// A geometry with a position-only vertex buffer at `positions`, no
    /// declaration (so `to_unified_vertices` falls back to raw Float3s),
    /// and one triangle if there are at least 3 vertices.
    fn position_geometry(shader_id: u16, positions: &[Vec3]) -> DrawableGeometry {
        let mut data = Vec::with_capacity(positions.len() * 12);
        for p in positions {
            data.extend_from_slice(&p.x.to_le_bytes());
            data.extend_from_slice(&p.y.to_le_bytes());
            data.extend_from_slice(&p.z.to_le_bytes());
        }
        let indices: Vec<u32> = if positions.len() >= 3 { vec![0, 1, 2] } else { Vec::new() };
        DrawableGeometry {
            shader_id,
            indices_count: indices.len() as u32,
            triangles_count: (indices.len() / 3) as u32,
            vertices_count: positions.len() as u16,
            vertex_stride: 12,
            vertex_buffer: Some(VertexBuffer {
                vertex_stride: 12,
                vertex_count: positions.len() as u32,
                data_pointer: 0,
                info_pointer: 0,
                declaration: None,
                data,
                layout: VertexBufferLayout::Legacy,
            }),
            index_buffer: Some(IndexBuffer {
                indices_count: indices.len() as u32,
                indices_pointer: 0,
                indices,
            }),
        }
    }

    /// Per-geometry bounds are reported separately even when the drawable's
    /// stored bounds fold them into one box — this is what lets rpf-cli#4's
    /// two-piece shape be measured directly.
    #[test]
    fn geometry_bounds_reports_each_geometry_separately() {
        let mut d = stub_drawable("two_geometries");
        d.lods.push(DrawableLod {
            level: LodLevel::High,
            models: vec![DrawableModel {
                skeleton_binding: 0,
                render_mask_flags: 0,
                shader_mapping: vec![0, 1],
                geometries: vec![
                    position_geometry(0, &[
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec3::new(1.0, 0.0, 0.0),
                        Vec3::new(0.0, 1.0, 0.0),
                    ]),
                    position_geometry(1, &[
                        Vec3::new(10.0, 10.0, 10.0),
                        Vec3::new(11.0, 10.0, 10.0),
                        Vec3::new(10.0, 11.0, 10.0),
                    ]),
                    // No vertex buffer: must be skipped, not panic.
                    DrawableGeometry {
                        shader_id: 0,
                        indices_count: 0,
                        triangles_count: 0,
                        vertices_count: 0,
                        vertex_stride: 0,
                        vertex_buffer: None,
                        index_buffer: None,
                    },
                ],
            }],
        });

        let lod = &d.lods[0];
        let geoms = d.geometry_bounds(lod);

        assert_eq!(geoms.len(), 2, "the geometry with no vertex buffer is skipped");

        assert_eq!(geoms[0].model, 0);
        assert_eq!(geoms[0].geometry, 0);
        assert_eq!(geoms[0].shader_id, 0);
        assert_eq!(geoms[0].vertices, 3);
        assert_eq!(geoms[0].triangles, 1);
        assert_eq!(geoms[0].min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(geoms[0].max, Vec3::new(1.0, 1.0, 0.0));
        let c0 = geoms[0].centroid;
        assert!((c0.x - 1.0 / 3.0).abs() < 1e-5 && (c0.y - 1.0 / 3.0).abs() < 1e-5 && c0.z == 0.0);

        assert_eq!(geoms[1].geometry, 1);
        assert_eq!(geoms[1].shader_id, 1);
        assert_eq!(geoms[1].min, Vec3::new(10.0, 10.0, 10.0));
        assert_eq!(geoms[1].max, Vec3::new(11.0, 11.0, 10.0));
    }

    #[test]
    fn lod_level_round_trips_through_strings() {
        for (text, level) in [
            ("high", LodLevel::High),
            ("Medium", LodLevel::Medium),
            ("LOW", LodLevel::Low),
            ("verylow", LodLevel::VeryLow),
        ] {
            assert_eq!(text.parse::<LodLevel>().unwrap(), level);
        }
        assert_eq!(LodLevel::VeryLow.to_string(), "verylow");
        assert!("nope".parse::<LodLevel>().is_err());
    }

    // ─── Fixture ─────────────────────────────────────────────────────────────

    /// A hand-built YDR (or YDD, when `as_dictionary`) resource: one drawable
    /// with one shader, one high LOD model and one three-vertex triangle.
    pub(crate) fn minimal_ydr_sections(as_dictionary: bool) -> (Vec<u8>, Vec<u8>) {
        let mut system = vec![0u8; 0x800];
        let mut graphics = vec![0u8; 0x200];
        let drawable_offset = if as_dictionary { 0x100 } else { 0x000 };

        if as_dictionary {
            write_u32(&mut system, 0x04, 1);
            write_u64(&mut system, 0x18, 1);
            write_u64(&mut system, 0x20, SYSTEM_BASE + 0x40);
            write_u16(&mut system, 0x28, 1);
            write_u16(&mut system, 0x2A, 1);
            write_u64(&mut system, 0x30, SYSTEM_BASE + 0x50);
            write_u16(&mut system, 0x38, 1);
            write_u16(&mut system, 0x3A, 1);
            write_u32(&mut system, 0x40, 0x1234_5678);
            write_u64(&mut system, 0x50, SYSTEM_BASE + drawable_offset as u64);
        }

        write_drawable(&mut system, drawable_offset);

        for index in 0..9usize {
            write_f32(&mut graphics, index * 4, index as f32 + 1.0);
        }
        write_u16(&mut graphics, 0x100, 0);
        write_u16(&mut graphics, 0x102, 1);
        write_u16(&mut graphics, 0x104, 2);

        (system, graphics)
    }

    fn write_drawable(system: &mut [u8], offset: usize) {
        write_u32(system, offset + 0x04, 1);
        write_u64(system, offset + 0x10, SYSTEM_BASE + 0x600);
        write_f32(system, offset + 0x2C, 10.0);
        write_u64(system, offset + 0x50, SYSTEM_BASE + 0x200);
        write_f32(system, offset + 0x70, 100.0);
        write_f32(system, offset + 0x74, 50.0);
        write_f32(system, offset + 0x78, 25.0);
        write_f32(system, offset + 0x7C, 12.5);
        write_u64(system, offset + 0xA8, SYSTEM_BASE + 0x400);
        system[0x400..0x40E].copy_from_slice(b"test_drawable\0");

        // High LOD pointer list -> one model.
        write_pointer_list_header(system, 0x200, 0x210, 1);
        write_u64(system, 0x210, SYSTEM_BASE + 0x240);

        // Model.
        write_u64(system, 0x240 + 0x08, SYSTEM_BASE + 0x280);
        write_u16(system, 0x240 + 0x10, 1);
        write_u16(system, 0x240 + 0x12, 1);
        write_u64(system, 0x240 + 0x20, SYSTEM_BASE + 0x290);
        write_u16(system, 0x240 + 0x2C, 0x0102);
        write_u16(system, 0x240 + 0x2E, 1);
        write_u64(system, 0x280, SYSTEM_BASE + 0x300);
        write_u16(system, 0x290, 0);

        // Geometry.
        write_u64(system, 0x300 + 0x18, SYSTEM_BASE + 0x480);
        write_u64(system, 0x300 + 0x38, SYSTEM_BASE + 0x580);
        write_u32(system, 0x300 + 0x58, 3);
        write_u32(system, 0x300 + 0x5C, 1);
        write_u16(system, 0x300 + 0x60, 3);
        write_u16(system, 0x300 + 0x62, 3);
        write_u16(system, 0x300 + 0x70, 12);
        write_u64(system, 0x300 + 0x78, GRAPHICS_BASE);

        // Vertex buffer and its declaration.
        write_u16(system, 0x480 + 0x08, 12);
        write_u64(system, 0x480 + 0x10, GRAPHICS_BASE);
        write_u32(system, 0x480 + 0x18, 3);
        write_u64(system, 0x480 + 0x30, SYSTEM_BASE + 0x760);
        write_u32(system, 0x760, 1);
        write_u16(system, 0x764, 12);
        write_u8(system, 0x766, 0);
        write_u8(system, 0x767, 1);
        write_u64(system, 0x768, 0x7755_5555_5599_6996);

        // Index buffer.
        write_u32(system, 0x580 + 0x08, 3);
        write_u64(system, 0x580 + 0x10, GRAPHICS_BASE + 0x100);

        // Shader group -> one shader.
        write_u64(system, 0x600 + 0x10, SYSTEM_BASE + 0x640);
        write_u16(system, 0x600 + 0x18, 1);
        write_u16(system, 0x600 + 0x1A, 1);
        write_u32(system, 0x600 + 0x30, 4);
        write_u64(system, 0x640, SYSTEM_BASE + 0x680);

        write_u64(system, 0x680, SYSTEM_BASE + 0x700);
        write_u32(system, 0x680 + 0x08, 0x1111_1111);
        write_u8(system, 0x680 + 0x10, 1);
        write_u8(system, 0x680 + 0x11, 2);
        write_u16(system, 0x680 + 0x14, 16);
        write_u16(system, 0x680 + 0x16, 48);
        write_u32(system, 0x680 + 0x18, 0x2222_2222);
        write_u32(system, 0x680 + 0x20, 0x0000_FF04);

        // One Vec4 parameter, with the name hash array after the value block.
        write_u8(system, 0x700, 1);
        write_u64(system, 0x700 + 0x08, SYSTEM_BASE + 0x710);
        write_f32(system, 0x710, 1.0);
        write_f32(system, 0x714, 2.0);
        write_f32(system, 0x718, 3.0);
        write_f32(system, 0x71C, 4.0);
        write_u32(system, 0x720, 0xAABB_CCDD);
    }

    fn write_pointer_list_header(system: &mut [u8], offset: usize, list_offset: usize, count: u16) {
        write_u64(system, offset, SYSTEM_BASE + list_offset as u64);
        write_u16(system, offset + 8, count);
        write_u16(system, offset + 10, count);
    }

    fn write_u8(data: &mut [u8], offset: usize, value: u8) {
        data[offset] = value;
    }

    fn write_u16(data: &mut [u8], offset: usize, value: u16) {
        data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u32(data: &mut [u8], offset: usize, value: u32) {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u64(data: &mut [u8], offset: usize, value: u64) {
        data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn write_f32(data: &mut [u8], offset: usize, value: f32) {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}
