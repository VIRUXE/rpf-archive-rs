use super::{
    VertexComponent, VertexComponentType, VertexDeclaration, VertexSemantic, G9_DECLARATION_SIZE,
    G9_FORMAT_COUNT,
};

const G9_SEMANTIC_ORDER: &[(u8, u8)] = &[
    (0, 0),
    (4, 3),
    (8, 14),
    (12, 15),
    (16, 1),
    (20, 2),
    (24, 4),
    (25, 5),
    (28, 6),
    (29, 7),
    (30, 8),
    (31, 9),
    (32, 10),
    (33, 11),
];

#[derive(Clone, Copy)]
struct G9Slot {
    g9_index: u8,
    semantic: VertexSemantic,
    semantic_index: u8,
    component_type: VertexComponentType,
    src_offset: u32,
    stream_stride: usize,
    elem_size: u8,
    g9_format: u8,
}

/// Parses a 320-byte Gen9 vertex declaration. When the layout is SoA, the
/// returned `Vec<u8>` is the interleaved AoS stream; otherwise it is `None`
/// and the original buffer bytes stay valid.
pub fn parse_gen9_declaration(
    raw: &[u8],
    vertex_stride: u16,
    vertex_count: u32,
    data: &[u8],
) -> Option<(VertexDeclaration, [u8; G9_FORMAT_COUNT], Option<Vec<u8>>)> {
    if raw.len() < G9_DECLARATION_SIZE {
        return None;
    }
    let packed = u64::from_le_bytes(raw[312..320].try_into().ok()?);
    let has_soa = (packed & 1) != 0;
    let vertex_size = {
        let size = ((packed >> 2) & 0xFF) as u16;
        if size == 0 {
            vertex_stride
        } else {
            size
        }
    };
    if vertex_size == 0 {
        return None;
    }
    let mut offsets = [0u32; G9_FORMAT_COUNT];
    let mut sizes = [0u8; G9_FORMAT_COUNT];
    let mut formats = [0u8; G9_FORMAT_COUNT];
    for index in 0..G9_FORMAT_COUNT {
        offsets[index] = u32::from_le_bytes(raw[index * 4..index * 4 + 4].try_into().ok()?);
        sizes[index] = raw[208 + index];
        formats[index] = raw[260 + index];
    }
    let mut slots = Vec::new();
    for index in 0..G9_FORMAT_COUNT as u8 {
        let format = formats[index as usize];
        if format == 0 {
            continue;
        }
        let elem_size = g9_element_size(index, &offsets, &sizes, &formats, vertex_size, has_soa);
        if elem_size == 0 {
            continue;
        }
        let component_type = type_from_g9_format(format, elem_size);
        let src_offset = offsets[index as usize];
        let stream_stride = g9_stream_stride(
            sizes[index as usize],
            elem_size,
            vertex_size,
            has_soa,
            src_offset,
            vertex_count,
            data.len(),
            next_g9_offset(index, &offsets, &formats),
        );
        let (semantic, semantic_index) = match g9_index_to_semantic(index) {
            Some(semantic) => (VertexSemantic::from_index(semantic), semantic),
            None => (VertexSemantic::Unknown(index), index),
        };
        slots.push(G9Slot {
            g9_index: index,
            semantic,
            semantic_index,
            component_type,
            src_offset,
            stream_stride,
            elem_size,
            g9_format: format,
        });
    }
    if slots.is_empty() {
        return None;
    }
    let needs_materialize = has_soa
        || slots.iter().any(|slot| {
            slot.src_offset >= vertex_size as u32
                || slot.src_offset > u16::MAX as u32
                || slot.src_offset as usize + slot.elem_size as usize > vertex_size as usize
        });
    let declaration = build_g9_declaration(&slots, vertex_size, needs_materialize);
    if declaration.components.is_empty() {
        return None;
    }
    let formats = formats_from_slots(&slots);
    if !needs_materialize || vertex_count == 0 || data.is_empty() {
        return Some((declaration, formats, None));
    }
    let aos = materialize_g9_aos(data, vertex_count, &slots, &declaration)?;
    Some((declaration, formats, Some(aos)))
}

pub(crate) fn g9_index_to_semantic(index: u8) -> Option<u8> {
    G9_SEMANTIC_ORDER
        .iter()
        .find(|(g9, _)| *g9 == index)
        .map(|(_, semantic)| *semantic)
}

fn formats_from_slots(slots: &[G9Slot]) -> [u8; G9_FORMAT_COUNT] {
    let mut formats = [0u8; G9_FORMAT_COUNT];
    for slot in slots {
        formats[slot.g9_index as usize] = slot.g9_format;
    }
    formats
}

fn g9_element_size(
    index: u8,
    offsets: &[u32; G9_FORMAT_COUNT],
    sizes: &[u8; G9_FORMAT_COUNT],
    formats: &[u8; G9_FORMAT_COUNT],
    vertex_size: u16,
    has_soa: bool,
) -> u8 {
    if let Some(size) = g9_format_bytes(formats[index as usize]) {
        return size;
    }
    let size_field = sizes[index as usize];
    if size_field > 0 && size_field <= 16 && (has_soa || size_field != vertex_size as u8) {
        return size_field;
    }
    let this_off = offsets[index as usize];
    if let Some(delta) = next_g9_offset(index, offsets, formats).map(|value| value.saturating_sub(this_off))
    {
        if delta > 0 && delta <= 16 {
            return delta as u8;
        }
    }
    if !has_soa && this_off < vertex_size as u32 {
        let rest = vertex_size as u32 - this_off;
        if rest > 0 && rest <= 16 {
            return rest as u8;
        }
    }
    16
}

fn next_g9_offset(
    index: u8,
    offsets: &[u32; G9_FORMAT_COUNT],
    formats: &[u8; G9_FORMAT_COUNT],
) -> Option<u32> {
    let this_off = offsets[index as usize];
    let mut next = None;
    for other in 0..G9_FORMAT_COUNT {
        if formats[other] == 0 {
            continue;
        }
        if offsets[other] > this_off {
            next = Some(next.map_or(offsets[other], |value: u32| value.min(offsets[other])));
        }
    }
    next
}

fn g9_stream_stride(
    size_field: u8,
    elem_size: u8,
    vertex_size: u16,
    has_soa: bool,
    src_offset: u32,
    vertex_count: u32,
    data_len: usize,
    next_offset: Option<u32>,
) -> usize {
    let soa = has_soa || src_offset >= vertex_size as u32;
    let preferred = if soa {
        if size_field > 0 {
            size_field as usize
        } else {
            elem_size as usize
        }
    } else if size_field as usize >= vertex_size as usize {
        size_field as usize
    } else {
        vertex_size as usize
    };
    let preferred = preferred.max(elem_size as usize);
    if !soa || vertex_count == 0 || data_len == 0 {
        return preferred;
    }
    if g9_stream_fits(
        src_offset,
        elem_size,
        vertex_count,
        data_len,
        preferred,
        next_offset,
        true,
    ) {
        return preferred;
    }
    let fallbacks = [elem_size as usize, vertex_size as usize, size_field as usize];
    for strict in [true, false] {
        for candidate in fallbacks {
            if candidate == 0 || candidate == preferred {
                continue;
            }
            if g9_stream_fits(
                src_offset,
                elem_size,
                vertex_count,
                data_len,
                candidate,
                next_offset,
                strict,
            ) {
                return candidate;
            }
        }
    }
    preferred
}

fn g9_stream_fits(
    offset: u32,
    elem_size: u8,
    vertex_count: u32,
    data_len: usize,
    stride: usize,
    next_offset: Option<u32>,
    check_next: bool,
) -> bool {
    if stride < elem_size as usize || stride == 0 {
        return false;
    }
    let count = vertex_count as usize;
    if count == 0 || data_len == 0 {
        return true;
    }
    let Some(last) = (count - 1)
        .checked_mul(stride)
        .and_then(|span| (offset as usize).checked_add(span))
        .and_then(|start| start.checked_add(elem_size as usize))
    else {
        return false;
    };
    if last > data_len {
        return false;
    }
    if check_next {
        if let Some(next) = next_offset {
            if next > offset && last > next as usize {
                return false;
            }
        }
    }
    true
}

fn build_g9_declaration(slots: &[G9Slot], vertex_size: u16, sequential: bool) -> VertexDeclaration {
    let mut components = Vec::new();
    let mut flags = 0u32;
    let mut types = 0u64;
    let mut offset = 0u16;
    for slot in slots {
        let component_offset = if sequential {
            offset
        } else {
            slot.src_offset as u16
        };
        if slot.semantic_index < 16 && !matches!(slot.semantic, VertexSemantic::Unknown(_)) {
            flags |= 1u32 << slot.semantic_index;
            types |= (slot.component_type.nibble() as u64) << (slot.semantic_index * 4);
        }
        components.push(VertexComponent {
            semantic: slot.semantic,
            semantic_index: slot.semantic_index,
            component_type: slot.component_type,
            offset: component_offset,
            size: slot.elem_size,
            component_count: slot.component_type.component_count(),
        });
        offset = offset.saturating_add(slot.elem_size as u16);
    }
    VertexDeclaration {
        flags,
        stride: if sequential { offset } else { vertex_size },
        unknown_6h: 0,
        count: components.len() as u8,
        types,
        components,
    }
}

fn materialize_g9_aos(
    data: &[u8],
    vertex_count: u32,
    slots: &[G9Slot],
    declaration: &VertexDeclaration,
) -> Option<Vec<u8>> {
    let count = vertex_count as usize;
    let dst_stride = declaration.stride as usize;
    if dst_stride == 0 {
        return None;
    }
    let mut out = vec![0u8; count * dst_stride];
    for slot in slots {
        let dst_off = declaration
            .components
            .iter()
            .find(|component| g9_component_matches(component, slot))?
            .offset as usize;
        let elem = slot.elem_size as usize;
        for vertex in 0..count {
            let src = slot.src_offset as usize + vertex * slot.stream_stride;
            let dst = vertex * dst_stride + dst_off;
            if src + elem > data.len() || dst + elem > out.len() {
                return None;
            }
            out[dst..dst + elem].copy_from_slice(&data[src..src + elem]);
        }
    }
    Some(out)
}

fn g9_component_matches(component: &VertexComponent, slot: &G9Slot) -> bool {
    match component.semantic {
        VertexSemantic::Unknown(index) => index == slot.g9_index,
        _ => component.semantic_index == slot.semantic_index && component.semantic == slot.semantic,
    }
}

fn g9_format_bytes(format: u8) -> Option<u8> {
    Some(match format {
        2 => 16,
        6 => 12,
        10 => 8,
        16 => 8,
        24 => 4,
        28 => 4,
        30 => 4,
        34 => 4,
        _ => return None,
    })
}

fn g9_format_to_type(format: u8) -> Option<VertexComponentType> {
    Some(match format {
        2 => VertexComponentType::Float4,
        6 => VertexComponentType::Float3,
        10 => VertexComponentType::Half4,
        16 => VertexComponentType::Float2,
        24 => VertexComponentType::Colour,
        28 => VertexComponentType::Colour,
        30 => VertexComponentType::UByte4,
        34 => VertexComponentType::Half2,
        _ => return None,
    })
}

fn type_from_g9_format(format: u8, size: u8) -> VertexComponentType {
    if let Some(component_type) = g9_format_to_type(format) {
        return component_type;
    }
    match size {
        16 => VertexComponentType::Float4,
        12 => VertexComponentType::Float3,
        8 => VertexComponentType::Float2,
        _ => VertexComponentType::Colour,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_format_is_kept_on_the_slot_map() {
        let mut raw = [0u8; G9_DECLARATION_SIZE];
        raw[0..4].copy_from_slice(&0u32.to_le_bytes());
        raw[4 * 4..4 * 4 + 4].copy_from_slice(&12u32.to_le_bytes());
        raw[28 * 4..28 * 4 + 4].copy_from_slice(&24u32.to_le_bytes());
        raw[208] = 36;
        raw[208 + 4] = 36;
        raw[208 + 28] = 36;
        raw[260] = 6;
        raw[260 + 4] = 99;
        raw[260 + 28] = 16;
        raw[312..320].copy_from_slice(&(36u64 << 2).to_le_bytes());
        let (declaration, formats, aos) = parse_gen9_declaration(&raw, 36, 0, &[]).unwrap();
        assert!(aos.is_none());
        assert_eq!(formats[4], 99);
        assert!(declaration
            .components
            .iter()
            .any(|component| component.semantic == VertexSemantic::Normal));
    }

    #[test]
    fn soa_deinterleaves_and_keeps_formats() {
        let mut raw = [0u8; G9_DECLARATION_SIZE];
        raw[0..4].copy_from_slice(&0u32.to_le_bytes());
        raw[28 * 4..28 * 4 + 4].copy_from_slice(&24u32.to_le_bytes());
        raw[208] = 12;
        raw[208 + 28] = 8;
        raw[260] = 6;
        raw[260 + 28] = 16;
        raw[312..320].copy_from_slice(&(1u64 | (20u64 << 2)).to_le_bytes());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&3.0f32.to_le_bytes());
        bytes.extend_from_slice(&4.0f32.to_le_bytes());
        bytes.extend_from_slice(&5.0f32.to_le_bytes());
        bytes.extend_from_slice(&6.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.25f32.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&0.75f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        let (declaration, formats, aos) = parse_gen9_declaration(&raw, 20, 2, &bytes).unwrap();
        let aos = aos.expect("SoA must rematerialize");
        assert_eq!(formats[0], 6);
        assert_eq!(formats[28], 16);
        assert_eq!(declaration.stride, 20);
        assert_eq!(&aos[12..16], &0.25f32.to_le_bytes());
    }
}
