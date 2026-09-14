//! Vertex buffers of a drawable geometry: the several on-disk layouts, the
//! vertex declaration that describes one, and the decoding of each
//! component type into renderer-ready [`UnifiedVertex`]es.
//!
//! [`parse_vertex_buffer_at`] is the entry point used by [`crate::ydd`];
//! everything else is the public vertex model re-exported from there.

use anyhow::{Context, Result};

use crate::math::{Vec2, Vec3, Vec4};
use crate::resource::{f32_le, u16_le, u32_le, u64_le, vec3_le, vec4_le, ResReader};

// ─── Vertex model ─────────────────────────────────────────────────────────────

/// Which of the several vertex buffer layouts a geometry's vertices came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexBufferLayout {
    /// Legacy PC layout, stream in `DataPointer1`.
    Legacy,
    /// Legacy layout whose usable stream lives in `DataPointer2` instead.
    LegacyData2,
    /// Gen9 layout — no vertex declaration is stored.
    Gen9,
    /// No vertex buffer struct: the geometry carries the stream itself.
    GeometryInline,
}

#[derive(Debug, Clone)]
pub struct VertexBuffer {
    pub vertex_stride: u16,
    pub vertex_count: u32,
    pub data_pointer: u64,
    pub info_pointer: u64,
    pub declaration: Option<VertexDeclaration>,
    pub data: Vec<u8>,
    pub layout: VertexBufferLayout,
}

#[derive(Debug, Clone)]
pub struct VertexDeclaration {
    pub flags: u32,
    pub stride: u16,
    pub unknown_6h: u8,
    pub count: u8,
    pub types: u64,
    pub components: Vec<VertexComponent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexComponent {
    pub semantic: VertexSemantic,
    pub semantic_index: u8,
    pub component_type: VertexComponentType,
    pub offset: u16,
    pub size: u8,
    pub component_count: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexSemantic {
    Position,
    BlendWeights,
    BlendIndices,
    Normal,
    Colour0,
    Colour1,
    TexCoord0,
    TexCoord1,
    TexCoord2,
    TexCoord3,
    TexCoord4,
    TexCoord5,
    TexCoord6,
    TexCoord7,
    Tangent,
    Binormal,
    Unknown(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexComponentType {
    Nothing,
    Half2,
    Float,
    Half4,
    FloatUnknown,
    Float2,
    Float3,
    Float4,
    UByte4,
    Colour,
    Rgba8Snorm,
    Unknown(u8),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexAttribute {
    pub component: VertexComponent,
    pub value: VertexAttributeValue,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VertexAttributeValue {
    Half2([f32; 2]),
    Float(f32),
    Half4([f32; 4]),
    Float2([f32; 2]),
    Float3(Vec3),
    Float4(Vec4),
    UByte4([u8; 4]),
    Colour([u8; 4]),
    Rgba8Snorm([f32; 4]),
    Unsupported,
}

/// One vertex with every semantic the renderer cares about resolved, whatever
/// the source declaration happened to store.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnifiedVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color0: [u8; 4],
    pub color1: [u8; 4],
    pub texcoord0: Vec2,
    pub texcoord1: Vec2,
    pub tangent: Vec4,
    pub blend_weights: Vec4,
    pub blend_indices: [u8; 4],
}

// ─── Vertex buffers ───────────────────────────────────────────────────────────

/// Tries each known vertex buffer layout in turn; the first whose stream
/// actually resolves wins. A geometry whose vertices cannot be found at all
/// yields `None` rather than failing the whole drawable.
pub(crate) fn parse_vertex_buffer_at(
    reader: &ResReader<'_>,
    va: u64,
    inline_data_pointer: u64,
    inline_vertex_count: u32,
    inline_vertex_stride: u16,
) -> Option<VertexBuffer> {
    if let Some(raw) = reader.resolve(va, 0x80) {
        let info_pointer = u64_le(raw, 0x30);
        let legacy_stride = u16_le(raw, 0x08);
        let legacy_count = u32_le(raw, 0x18);

        // 1. Legacy, stream in DataPointer1.
        if let Some(data) = read_vertex_data(reader, u64_le(raw, 0x10), legacy_count, legacy_stride)
        {
            return Some(VertexBuffer {
                vertex_stride: legacy_stride,
                vertex_count: legacy_count,
                data_pointer: u64_le(raw, 0x10),
                info_pointer,
                declaration: parse_vertex_declaration_at(reader, info_pointer),
                data,
                layout: VertexBufferLayout::Legacy,
            });
        }

        // 2. Legacy, but the usable stream sits in DataPointer2.
        if let Some(data) = read_vertex_data(reader, u64_le(raw, 0x20), legacy_count, legacy_stride)
        {
            return Some(VertexBuffer {
                vertex_stride: legacy_stride,
                vertex_count: legacy_count,
                data_pointer: u64_le(raw, 0x20),
                info_pointer,
                declaration: parse_vertex_declaration_at(reader, info_pointer),
                data,
                layout: VertexBufferLayout::LegacyData2,
            });
        }

        // 3. Gen9 — count and stride swap places and no declaration is stored.
        let gen9_count = u32_le(raw, 0x08);
        let gen9_stride = u16_le(raw, 0x0C);
        let gen9_pointer = u64_le(raw, 0x18);
        if let Some(data) = read_vertex_data(reader, gen9_pointer, gen9_count, gen9_stride) {
            // The legacy info slot holds something else in this layout, so
            // it is not reported as a declaration pointer.
            return Some(VertexBuffer {
                vertex_stride: gen9_stride,
                vertex_count: gen9_count,
                data_pointer: gen9_pointer,
                info_pointer: 0,
                declaration: None,
                data,
                layout: VertexBufferLayout::Gen9,
            });
        }
    }

    // 4. No usable vertex buffer struct: take the geometry's own stream.
    let data = read_vertex_data(
        reader,
        inline_data_pointer,
        inline_vertex_count,
        inline_vertex_stride,
    )?;

    Some(VertexBuffer {
        vertex_stride: inline_vertex_stride,
        vertex_count: inline_vertex_count,
        data_pointer: inline_data_pointer,
        info_pointer: 0,
        declaration: None,
        data,
        layout: VertexBufferLayout::GeometryInline,
    })
}

fn read_vertex_data(
    reader: &ResReader<'_>,
    data_pointer: u64,
    vertex_count: u32,
    vertex_stride: u16,
) -> Option<Vec<u8>> {
    if data_pointer == 0 || vertex_count == 0 || vertex_stride == 0 {
        return None;
    }

    let len = (vertex_count as usize).checked_mul(vertex_stride as usize)?;
    Some(reader.resolve(data_pointer, len)?.to_vec())
}

fn parse_vertex_declaration_at(reader: &ResReader<'_>, va: u64) -> Option<VertexDeclaration> {
    let raw = reader.resolve(va, 16)?;
    let flags = u32_le(raw, 0x00);
    let types = u64_le(raw, 0x08);

    let mut components = Vec::new();
    let mut offset = 0u16;
    for semantic_index in 0..16u8 {
        if ((flags >> semantic_index) & 1) != 1 {
            continue;
        }

        let component_type =
            VertexComponentType::from_nibble(((types >> (semantic_index * 4)) & 0xF) as u8);
        let size = component_type.size_in_bytes();
        components.push(VertexComponent {
            semantic: VertexSemantic::from_index(semantic_index),
            semantic_index,
            component_type,
            offset,
            size,
            component_count: component_type.component_count(),
        });
        offset = offset.checked_add(size as u16)?;
    }

    Some(VertexDeclaration {
        flags,
        stride: u16_le(raw, 0x04),
        unknown_6h: raw[0x06],
        count: raw[0x07],
        types,
        components,
    })
}

impl VertexDeclaration {
    /// The synthetic declaration used when a resource stores none: a single
    /// `Float3` position at offset 0, with the buffer's own stride.
    pub fn position_only(stride: u16) -> Self {
        Self {
            flags: 1,
            stride,
            unknown_6h: 0,
            count: 1,
            types: VertexComponentType::Float3.nibble() as u64,
            components: vec![VertexComponent {
                semantic: VertexSemantic::Position,
                semantic_index: 0,
                component_type: VertexComponentType::Float3,
                offset: 0,
                size: VertexComponentType::Float3.size_in_bytes(),
                component_count: VertexComponentType::Float3.component_count(),
            }],
        }
    }

    /// CodeWalker's "declaration id": the type nibbles of the used semantics.
    pub fn declaration_id(&self) -> u64 {
        let mut id = 0u64;
        for index in 0..16 {
            if ((self.flags >> index) & 1) == 1 {
                id |= self.types & (0xFu64 << (index * 4));
            }
        }
        id
    }
}

impl VertexSemantic {
    fn from_index(index: u8) -> Self {
        match index {
            0 => Self::Position,
            1 => Self::BlendWeights,
            2 => Self::BlendIndices,
            3 => Self::Normal,
            4 => Self::Colour0,
            5 => Self::Colour1,
            6 => Self::TexCoord0,
            7 => Self::TexCoord1,
            8 => Self::TexCoord2,
            9 => Self::TexCoord3,
            10 => Self::TexCoord4,
            11 => Self::TexCoord5,
            12 => Self::TexCoord6,
            13 => Self::TexCoord7,
            14 => Self::Tangent,
            15 => Self::Binormal,
            value => Self::Unknown(value),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Position => "Position",
            Self::BlendWeights => "BlendWeights",
            Self::BlendIndices => "BlendIndices",
            Self::Normal => "Normal",
            Self::Colour0 => "Colour0",
            Self::Colour1 => "Colour1",
            Self::TexCoord0 => "TexCoord0",
            Self::TexCoord1 => "TexCoord1",
            Self::TexCoord2 => "TexCoord2",
            Self::TexCoord3 => "TexCoord3",
            Self::TexCoord4 => "TexCoord4",
            Self::TexCoord5 => "TexCoord5",
            Self::TexCoord6 => "TexCoord6",
            Self::TexCoord7 => "TexCoord7",
            Self::Tangent => "Tangent",
            Self::Binormal => "Binormal",
            Self::Unknown(_) => "Unknown",
        }
    }
}

impl std::fmt::Display for VertexSemantic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(index) => write!(formatter, "Unknown{index}"),
            _ => formatter.write_str(self.as_str()),
        }
    }
}

impl VertexComponentType {
    fn from_nibble(value: u8) -> Self {
        match value {
            0 => Self::Nothing,
            1 => Self::Half2,
            2 => Self::Float,
            3 => Self::Half4,
            4 => Self::FloatUnknown,
            5 => Self::Float2,
            6 => Self::Float3,
            7 => Self::Float4,
            8 => Self::UByte4,
            9 => Self::Colour,
            10 => Self::Rgba8Snorm,
            value => Self::Unknown(value),
        }
    }

    fn nibble(self) -> u8 {
        match self {
            Self::Nothing => 0,
            Self::Half2 => 1,
            Self::Float => 2,
            Self::Half4 => 3,
            Self::FloatUnknown => 4,
            Self::Float2 => 5,
            Self::Float3 => 6,
            Self::Float4 => 7,
            Self::UByte4 => 8,
            Self::Colour => 9,
            Self::Rgba8Snorm => 10,
            Self::Unknown(value) => value,
        }
    }

    pub fn size_in_bytes(&self) -> u8 {
        match self {
            Self::Nothing | Self::FloatUnknown | Self::Unknown(_) => 0,
            Self::Half2 | Self::Float | Self::UByte4 | Self::Colour | Self::Rgba8Snorm => 4,
            Self::Half4 | Self::Float2 => 8,
            Self::Float3 => 12,
            Self::Float4 => 16,
        }
    }

    pub fn component_count(&self) -> u8 {
        match self {
            Self::Nothing | Self::FloatUnknown | Self::Unknown(_) => 0,
            Self::Float => 1,
            Self::Half2 | Self::Float2 => 2,
            Self::Float3 => 3,
            Self::Half4 | Self::Float4 | Self::UByte4 | Self::Colour | Self::Rgba8Snorm => 4,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Nothing => "Nothing",
            Self::Half2 => "Half2",
            Self::Float => "Float",
            Self::Half4 => "Half4",
            Self::FloatUnknown => "FloatUnk",
            Self::Float2 => "Float2",
            Self::Float3 => "Float3",
            Self::Float4 => "Float4",
            Self::UByte4 => "UByte4",
            Self::Colour => "Colour",
            Self::Rgba8Snorm => "RGBA8SNorm",
            Self::Unknown(_) => "Unknown",
        }
    }
}

impl std::fmt::Display for VertexComponentType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(value) => write!(formatter, "Unk{value}"),
            _ => formatter.write_str(self.as_str()),
        }
    }
}

impl VertexAttributeValue {
    pub fn as_vec3(&self) -> Vec3 {
        match self {
            Self::Float3(v) => *v,
            Self::Float4(v) => v.xyz(),
            Self::Half4(a) => Vec3::new(a[0], a[1], a[2]),
            Self::Float2(a) | Self::Half2(a) => Vec3::new(a[0], a[1], 0.0),
            Self::Rgba8Snorm(a) => Vec3::new(a[0], a[1], a[2]),
            Self::UByte4(a) | Self::Colour(a) => {
                Vec3::new(a[0] as f32 / 255.0, a[1] as f32 / 255.0, a[2] as f32 / 255.0)
            }
            Self::Float(f) => Vec3::new(*f, 0.0, 0.0),
            Self::Unsupported => Vec3::ZERO,
        }
    }

    pub fn as_vec4(&self) -> Vec4 {
        match self {
            Self::Float4(v) => *v,
            Self::Float3(v) => Vec4::new(v.x, v.y, v.z, 1.0),
            Self::Half4(a) => Vec4::new(a[0], a[1], a[2], a[3]),
            Self::Rgba8Snorm(a) => Vec4::new(a[0], a[1], a[2], a[3]),
            Self::UByte4(a) | Self::Colour(a) => Vec4::new(
                a[0] as f32 / 255.0,
                a[1] as f32 / 255.0,
                a[2] as f32 / 255.0,
                a[3] as f32 / 255.0,
            ),
            Self::Float2(a) | Self::Half2(a) => Vec4::new(a[0], a[1], 0.0, 1.0),
            Self::Float(f) => Vec4::new(*f, 0.0, 0.0, 1.0),
            Self::Unsupported => Vec4::new(0.0, 0.0, 0.0, 0.0),
        }
    }

    pub fn as_vec2(&self) -> Vec2 {
        match self {
            Self::Float2(a) | Self::Half2(a) => Vec2::new(a[0], a[1]),
            Self::Float3(v) => Vec2::new(v.x, v.y),
            Self::Float4(v) => Vec2::new(v.x, v.y),
            Self::Half4(a) => Vec2::new(a[0], a[1]),
            Self::Rgba8Snorm(a) => Vec2::new(a[0], a[1]),
            Self::UByte4(a) | Self::Colour(a) => {
                Vec2::new(a[0] as f32 / 255.0, a[1] as f32 / 255.0)
            }
            Self::Float(f) => Vec2::new(*f, 0.0),
            Self::Unsupported => Vec2::new(0.0, 0.0),
        }
    }

    pub fn as_rgba8(&self) -> [u8; 4] {
        match self {
            Self::UByte4(a) | Self::Colour(a) => *a,
            Self::Float4(v) => [
                (v.x.clamp(0.0, 1.0) * 255.0) as u8,
                (v.y.clamp(0.0, 1.0) * 255.0) as u8,
                (v.z.clamp(0.0, 1.0) * 255.0) as u8,
                (v.w.clamp(0.0, 1.0) * 255.0) as u8,
            ],
            Self::Half4(a) => [
                (a[0].clamp(0.0, 1.0) * 255.0) as u8,
                (a[1].clamp(0.0, 1.0) * 255.0) as u8,
                (a[2].clamp(0.0, 1.0) * 255.0) as u8,
                (a[3].clamp(0.0, 1.0) * 255.0) as u8,
            ],
            Self::Rgba8Snorm(a) => [
                ((a[0] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
                ((a[1] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
                ((a[2] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
                ((a[3] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
            ],
            _ => [255, 255, 255, 255],
        }
    }
}

impl VertexBuffer {
    /// Reads every declared attribute of one vertex. Buffers without a stored
    /// declaration are read through the synthetic position-only one.
    pub fn read_vertex_attributes(&self, vertex_index: usize) -> Result<Vec<VertexAttribute>> {
        let synthetic;
        let declaration = match &self.declaration {
            Some(declaration) => declaration,
            None => {
                synthetic = VertexDeclaration::position_only(self.vertex_stride);
                &synthetic
            }
        };

        self.read_vertex_attributes_with(declaration, vertex_index)
    }

    fn read_vertex_attributes_with(
        &self,
        declaration: &VertexDeclaration,
        vertex_index: usize,
    ) -> Result<Vec<VertexAttribute>> {
        if vertex_index >= self.vertex_count as usize {
            anyhow::bail!(
                "vertex index {vertex_index} is out of bounds for {} vertices",
                self.vertex_count
            );
        }

        let base = vertex_index
            .checked_mul(self.vertex_stride as usize)
            .context("vertex attribute offset overflowed")?;
        let mut attributes = Vec::with_capacity(declaration.components.len());

        for component in &declaration.components {
            let offset = base
                .checked_add(component.offset as usize)
                .context("vertex attribute offset overflowed")?;
            let value = read_vertex_attribute_value(&self.data, offset, component.component_type)?;
            attributes.push(VertexAttribute { component: *component, value });
        }

        Ok(attributes)
    }

    /// Decodes the whole buffer into renderer-ready vertices. A buffer with no
    /// declaration still yields positions, read as `Float3` at offset 0.
    pub fn to_unified_vertices(&self) -> Result<Vec<UnifiedVertex>> {
        let synthetic;
        let declaration = match &self.declaration {
            Some(declaration) => declaration,
            None => {
                synthetic = VertexDeclaration::position_only(self.vertex_stride);
                &synthetic
            }
        };

        let mut unified = Vec::with_capacity(self.vertex_count as usize);
        for index in 0..self.vertex_count as usize {
            let attributes = self.read_vertex_attributes_with(declaration, index)?;

            let mut vertex = UnifiedVertex {
                position: Vec3::ZERO,
                normal: Vec3::new(0.0, 0.0, 1.0),
                color0: [255, 255, 255, 255],
                color1: [255, 255, 255, 255],
                texcoord0: Vec2::new(0.0, 0.0),
                texcoord1: Vec2::new(0.0, 0.0),
                tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
                blend_weights: Vec4::new(0.0, 0.0, 0.0, 0.0),
                blend_indices: [0u8; 4],
            };

            for attr in attributes {
                match attr.component.semantic {
                    VertexSemantic::Position => vertex.position = attr.value.as_vec3(),
                    VertexSemantic::Normal => vertex.normal = attr.value.as_vec3(),
                    VertexSemantic::Colour0 => vertex.color0 = attr.value.as_rgba8(),
                    VertexSemantic::Colour1 => vertex.color1 = attr.value.as_rgba8(),
                    VertexSemantic::TexCoord0 => vertex.texcoord0 = attr.value.as_vec2(),
                    VertexSemantic::TexCoord1 => vertex.texcoord1 = attr.value.as_vec2(),
                    VertexSemantic::Tangent => vertex.tangent = attr.value.as_vec4(),
                    VertexSemantic::BlendWeights => vertex.blend_weights = attr.value.as_vec4(),
                    VertexSemantic::BlendIndices => {
                        if let VertexAttributeValue::UByte4(a) = attr.value {
                            vertex.blend_indices = a;
                        }
                    }
                    _ => {}
                }
            }

            unified.push(vertex);
        }

        Ok(unified)
    }
}

fn read_vertex_attribute_value(
    data: &[u8],
    offset: usize,
    component_type: VertexComponentType,
) -> Result<VertexAttributeValue> {
    let size = component_type.size_in_bytes() as usize;
    if size > 0 && data.len() < offset + size {
        anyhow::bail!("unexpected end of vertex data at 0x{offset:X}");
    }

    Ok(match component_type {
        VertexComponentType::Nothing
        | VertexComponentType::FloatUnknown
        | VertexComponentType::Unknown(_) => VertexAttributeValue::Unsupported,
        VertexComponentType::Half2 => VertexAttributeValue::Half2([
            read_f16(data, offset),
            read_f16(data, offset + 2),
        ]),
        VertexComponentType::Float => VertexAttributeValue::Float(f32_le(data, offset)),
        VertexComponentType::Half4 => VertexAttributeValue::Half4([
            read_f16(data, offset),
            read_f16(data, offset + 2),
            read_f16(data, offset + 4),
            read_f16(data, offset + 6),
        ]),
        VertexComponentType::Float2 => {
            VertexAttributeValue::Float2([f32_le(data, offset), f32_le(data, offset + 4)])
        }
        VertexComponentType::Float3 => VertexAttributeValue::Float3(vec3_le(data, offset)),
        VertexComponentType::Float4 => VertexAttributeValue::Float4(vec4_le(data, offset)),
        VertexComponentType::UByte4 => VertexAttributeValue::UByte4(read_fixed_4(data, offset)),
        VertexComponentType::Colour => VertexAttributeValue::Colour(read_fixed_4(data, offset)),
        VertexComponentType::Rgba8Snorm => {
            let bytes = read_fixed_4(data, offset);
            VertexAttributeValue::Rgba8Snorm([
                snorm8_to_f32(bytes[0]),
                snorm8_to_f32(bytes[1]),
                snorm8_to_f32(bytes[2]),
                snorm8_to_f32(bytes[3]),
            ])
        }
    })
}

fn read_fixed_4(data: &[u8], offset: usize) -> [u8; 4] {
    data.get(offset..offset + 4)
        .and_then(|bytes| bytes.try_into().ok())
        .unwrap_or([0u8; 4])
}

fn read_f16(data: &[u8], offset: usize) -> f32 {
    f16_to_f32(u16_le(data, offset))
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = (value >> 10) & 0x1F;
    let mantissa = value & 0x03FF;

    let bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut mantissa = mantissa as u32;
            let mut exponent = -14i32;
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x03FF;
            sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
        }
        0x1F => sign | 0x7F80_0000 | ((mantissa as u32) << 13),
        _ => sign | ((exponent as u32 + 112) << 23) | ((mantissa as u32) << 13),
    };

    f32::from_bits(bits)
}

fn snorm8_to_f32(value: u8) -> f32 {
    ((value as i8) as f32 / 127.0).max(-1.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn component(semantic: VertexSemantic, ty: VertexComponentType, offset: u16) -> VertexComponent {
        VertexComponent {
            semantic,
            semantic_index: 0,
            component_type: ty,
            offset,
            size: ty.size_in_bytes(),
            component_count: ty.component_count(),
        }
    }

    /// A one-vertex buffer laid out exactly as `components` says.
    fn buffer(components: Vec<VertexComponent>, data: Vec<u8>) -> VertexBuffer {
        let stride = data.len() as u16;
        VertexBuffer {
            vertex_stride: stride,
            vertex_count: 1,
            data_pointer: 0,
            info_pointer: 0,
            declaration: Some(VertexDeclaration {
                flags: 0,
                stride,
                unknown_6h: 0,
                count: components.len() as u8,
                types: 0,
                components,
            }),
            data,
            layout: VertexBufferLayout::Legacy,
        }
    }

    #[test]
    fn f16_covers_normals_subnormals_and_infinities() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24), "smallest subnormal");
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn snorm8_maps_the_byte_range_onto_minus_one_to_one() {
        assert_eq!(snorm8_to_f32(0x7F), 1.0);
        assert_eq!(snorm8_to_f32(0x00), 0.0);
        assert_eq!(snorm8_to_f32(0x81), -1.0);
        assert_eq!(snorm8_to_f32(0x80), -1.0, "-128 clamps rather than overshooting");
    }

    /// Every component type the game uses decodes to the value the bytes
    /// spell, and `Unsupported` for the ones that carry no data.
    #[test]
    fn each_component_type_decodes_its_bytes() {
        let decode = |ty, bytes: &[u8]| read_vertex_attribute_value(bytes, 0, ty).unwrap();

        assert_eq!(decode(VertexComponentType::Half2, &[0x00, 0x3C, 0x00, 0xC0]), VertexAttributeValue::Half2([1.0, -2.0]));
        assert_eq!(
            decode(VertexComponentType::Half4, &[0x00, 0x3C, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x38]),
            VertexAttributeValue::Half4([1.0, -2.0, 0.0, 0.5])
        );
        assert_eq!(decode(VertexComponentType::Float, &1.5f32.to_le_bytes()), VertexAttributeValue::Float(1.5));
        let mut two = 1.5f32.to_le_bytes().to_vec();
        two.extend_from_slice(&(-4.0f32).to_le_bytes());
        assert_eq!(decode(VertexComponentType::Float2, &two), VertexAttributeValue::Float2([1.5, -4.0]));
        assert_eq!(decode(VertexComponentType::UByte4, &[1, 2, 3, 4]), VertexAttributeValue::UByte4([1, 2, 3, 4]));
        assert_eq!(decode(VertexComponentType::Colour, &[10, 20, 30, 40]), VertexAttributeValue::Colour([10, 20, 30, 40]));
        assert_eq!(
            decode(VertexComponentType::Rgba8Snorm, &[0x7F, 0x81, 0x00, 0x80]),
            VertexAttributeValue::Rgba8Snorm([1.0, -1.0, 0.0, -1.0])
        );
        assert_eq!(decode(VertexComponentType::Nothing, &[]), VertexAttributeValue::Unsupported);
        assert_eq!(decode(VertexComponentType::Unknown(13), &[]), VertexAttributeValue::Unsupported);
    }

    #[test]
    fn a_short_buffer_is_an_error_not_a_panic() {
        assert!(read_vertex_attribute_value(&[0, 0], 0, VertexComponentType::Float).is_err());
        assert!(read_vertex_attribute_value(&[0; 4], 2, VertexComponentType::Float).is_err());
    }

    /// The compact layouts real drawables use: a half-float UV, a packed
    /// snorm normal, a byte colour and byte blend indices all reach the
    /// unified vertex as the renderer expects them.
    #[test]
    fn unified_vertex_is_built_from_packed_components() {
        let components = vec![
            component(VertexSemantic::Position, VertexComponentType::Float3, 0),
            component(VertexSemantic::Normal, VertexComponentType::Rgba8Snorm, 12),
            component(VertexSemantic::Colour0, VertexComponentType::Colour, 16),
            component(VertexSemantic::TexCoord0, VertexComponentType::Half2, 20),
            component(VertexSemantic::BlendIndices, VertexComponentType::UByte4, 24),
            component(VertexSemantic::Tangent, VertexComponentType::Half4, 28),
        ];
        let mut data = Vec::new();
        for value in [1.0f32, 2.0, 3.0] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        data.extend_from_slice(&[0x00, 0x00, 0x7F, 0x00]); // normal +Z
        data.extend_from_slice(&[10, 20, 30, 40]); // colour
        data.extend_from_slice(&[0x00, 0x38, 0x00, 0x3C]); // uv (0.5, 1.0)
        data.extend_from_slice(&[5, 6, 7, 8]); // blend indices
        data.extend_from_slice(&[0x00, 0x3C, 0x00, 0x00, 0x00, 0x00, 0x00, 0xBC]); // tangent (1,0,0,-1)

        let vertices = buffer(components, data).to_unified_vertices().unwrap();

        assert_eq!(vertices.len(), 1);
        let v = vertices[0];
        assert_eq!(v.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(v.normal, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(v.color0, [10, 20, 30, 40]);
        assert_eq!(v.texcoord0, Vec2::new(0.5, 1.0));
        assert_eq!(v.blend_indices, [5, 6, 7, 8]);
        assert_eq!(v.tangent, Vec4::new(1.0, 0.0, 0.0, -1.0));
    }

    /// Colours stored as floats or snorm bytes are requantised to 0..=255.
    #[test]
    fn colours_are_requantised_from_float_and_snorm() {
        let float = VertexAttributeValue::Float4(Vec4::new(1.0, 0.5, 0.0, 2.0));
        assert_eq!(float.as_rgba8(), [255, 127, 0, 255]);

        let snorm = VertexAttributeValue::Rgba8Snorm([1.0, -1.0, 0.0, 1.0]);
        assert_eq!(snorm.as_rgba8(), [255, 0, 127, 255]);

        assert_eq!(VertexAttributeValue::Unsupported.as_rgba8(), [255, 255, 255, 255]);
    }

    #[test]
    fn reading_past_the_vertex_count_is_an_error() {
        let one = buffer(vec![component(VertexSemantic::Position, VertexComponentType::Float, 0)], vec![0; 4]);
        assert!(one.read_vertex_attributes(0).is_ok());
        assert!(one.read_vertex_attributes(1).is_err());
    }
}
