//! YFT (fragment) parsing for GTA V.
//!
//! A fragment wraps one main drawable plus an optional array of extra
//! drawables (damaged variants, attached props and so on). The PhysicsLODGroup
//! children — wheels, breakable pieces, their bounds and articulation — are
//! deliberately out of scope here: this layer exists to hand the renderer
//! geometry, not physics.

use anyhow::{Context, Result};

use crate::math::Vec3;
use crate::resource::{f32_le, u32_le, u64_le, vec3_le, prepare_rsc7, ResReader, SYSTEM_BASE};
use crate::writer::rage_joaat;
use crate::ydd::{parse_drawable_at, Drawable, DrawableEntry};

/// A fragment's renderable content.
#[derive(Debug, Clone)]
pub struct Fragment {
    pub name: String,
    pub bound_center: Vec3,
    pub bound_radius: f32,
    pub drawable: Option<Drawable>,
    pub extra_drawables: Vec<DrawableEntry>,
}

/// A `fragDrawable` keeps the first 0xA8 bytes of a plain drawable but pushes
/// its name pointer past the extra fragment fields, to 0x130.
const FRAG_DRAWABLE_NAME_OFFSET: usize = 0x130;
const FRAG_DRAWABLE_STRUCT_LEN: usize = 0x150;

/// Parses a YFT resource.
pub fn parse_yft(data: &[u8]) -> Result<Fragment> {
    let (system, graphics) = prepare_rsc7(data)?;
    let reader = ResReader { system: &system, graphics: &graphics };
    parse_yft_from_reader(&reader)
}

pub(crate) fn parse_yft_from_reader(reader: &ResReader<'_>) -> Result<Fragment> {
    let raw = reader
        .resolve(SYSTEM_BASE, 0x60)
        .context("system section too small for a FragType")?;

    let bound_center = vec3_le(raw, 0x20);
    let bound_radius = f32_le(raw, 0x2C);
    let drawable_pointer = u64_le(raw, 0x30);
    let drawable_array_pointer = u64_le(raw, 0x38);
    let drawable_names_pointer = u64_le(raw, 0x40);
    let drawable_array_count = u32_le(raw, 0x48) as usize;
    let name = reader.string_at(u64_le(raw, 0x58)).unwrap_or_default();

    let drawable = if drawable_pointer == 0 {
        None
    } else {
        Some(parse_drawable_at(
            reader,
            drawable_pointer,
            FRAG_DRAWABLE_NAME_OFFSET,
            FRAG_DRAWABLE_STRUCT_LEN,
            None,
        )?)
    };

    let extra_drawables = parse_extra_drawables(
        reader,
        drawable_array_pointer,
        drawable_names_pointer,
        drawable_array_count,
    )?;

    Ok(Fragment {
        name,
        bound_center,
        bound_radius,
        drawable,
        extra_drawables,
    })
}

fn parse_extra_drawables(
    reader: &ResReader<'_>,
    array_pointer: u64,
    names_pointer: u64,
    count: usize,
) -> Result<Vec<DrawableEntry>> {
    if array_pointer == 0 || count == 0 {
        return Ok(Vec::new());
    }

    let pointers = reader
        .read_u64_list(array_pointer, count)
        .context("fragment drawable array out of bounds")?;
    // The names array is optional; each entry is a pointer to a C string.
    let name_pointers = reader.read_u64_list(names_pointer, count).unwrap_or_default();

    let mut entries = Vec::with_capacity(pointers.len());
    for (index, pointer) in pointers.iter().copied().enumerate() {
        if pointer == 0 {
            continue;
        }

        let drawable = parse_drawable_at(
            reader,
            pointer,
            FRAG_DRAWABLE_NAME_OFFSET,
            FRAG_DRAWABLE_STRUCT_LEN,
            None,
        )?;
        let name = name_pointers
            .get(index)
            .copied()
            .and_then(|va| reader.string_at(va))
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| drawable.name.clone());
        let hash = rage_joaat(&name.to_ascii_lowercase());

        entries.push(DrawableEntry { hash, name, drawable });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::GRAPHICS_BASE;

    #[test]
    fn parses_minimal_yft() {
        let mut system = vec![0u8; 0x1000];
        let graphics = vec![0u8; 0x10];

        // FragType header.
        write_f32(&mut system, 0x20, 1.0);
        write_f32(&mut system, 0x24, 2.0);
        write_f32(&mut system, 0x28, 3.0);
        write_f32(&mut system, 0x2C, 7.5);
        write_u64(&mut system, 0x30, SYSTEM_BASE + 0x200); // DrawablePointer
        write_u64(&mut system, 0x58, SYSTEM_BASE + 0x180); // NamePointer
        system[0x180..0x18A].copy_from_slice(b"test_frag\0");

        // FragDrawable at +0x200, with its name pointer at +0x130.
        write_u64(&mut system, 0x200 + 0x130, SYSTEM_BASE + 0x190);
        system[0x190..0x19E].copy_from_slice(b"frag_drawable\0");
        write_f32(&mut system, 0x200 + 0x2C, 4.0);
        write_u64(&mut system, 0x200 + 0x50, SYSTEM_BASE + 0x400); // High LOD list

        // One model with no geometries.
        write_u64(&mut system, 0x400, SYSTEM_BASE + 0x410);
        write_u16(&mut system, 0x408, 1);
        write_u16(&mut system, 0x40A, 1);
        write_u64(&mut system, 0x410, SYSTEM_BASE + 0x440);
        write_u16(&mut system, 0x440 + 0x10, 0);

        let reader = ResReader { system: &system, graphics: &graphics };
        let fragment = parse_yft_from_reader(&reader).expect("fixture should parse");

        assert_eq!(fragment.name, "test_frag");
        assert_eq!(fragment.bound_center, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(fragment.bound_radius, 7.5);
        assert!(fragment.extra_drawables.is_empty());

        let drawable = fragment.drawable.expect("fragment drawable");
        assert_eq!(drawable.name, "frag_drawable");
        assert_eq!(drawable.name_hash, rage_joaat("frag_drawable"));
        assert_eq!(drawable.bounds.sphere_radius, 4.0);
        assert_eq!(drawable.model_count(), 1);
        assert_eq!(drawable.geometry_count(), 0);

        // The graphics section is unused by this fixture but must still resolve.
        assert!(reader.resolve(GRAPHICS_BASE, 0x10).is_some());
    }

    fn write_u16(data: &mut [u8], offset: usize, value: u16) {
        data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u64(data: &mut [u8], offset: usize, value: u64) {
        data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn write_f32(data: &mut [u8], offset: usize, value: f32) {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}
