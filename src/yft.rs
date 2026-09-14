//! YFT (fragment) parsing for GTA V.
//!
//! A fragment wraps one main drawable plus an optional array of extra
//! drawables (damaged variants, attached props and so on), and a physics LOD
//! group whose children carry the parts that move or break off — a vehicle's
//! wheels, doors, bonnet and boot — each with its own drawable and a
//! transform placing it on the body. Only what the renderer needs is read:
//! the children's pristine drawables and transforms, and the default bone
//! pose for the main drawable's models. Bounds, articulation and masses are
//! out of scope.

use anyhow::{Context, Result};

use crate::math::{Mat4, Vec3};
use crate::resource::{f32_le, u16_le, u32_le, u64_le, vec3_le, vec4_le, prepare_rsc7, ResReader, SYSTEM_BASE};
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
    /// Default pose, one matrix per bone index, applied to the main
    /// drawable's models by their bone binding. Empty when the fragment
    /// stores none.
    pub bone_transforms: Vec<Mat4>,
    /// Physics children of the first physics LOD, in file order.
    pub children: Vec<FragmentChild>,
}

/// One physics child: a part that the game can detach or articulate.
#[derive(Debug, Clone)]
pub struct FragmentChild {
    pub group_index: u16,
    /// Tag of the bone the part hangs off; vehicle wheels have fixed tags
    /// (see [`wheel_slot`]).
    pub bone_tag: u16,
    /// The pristine drawable. Wheel slots often have none and borrow another
    /// wheel's mesh at render time.
    pub drawable: Option<Drawable>,
    /// Placement of the part in the fragment's space: the physics LOD's
    /// transform for this child, shifted by the LOD's position offset.
    pub transform: Mat4,
}

/// A drawable to render as part of a fragment, with its placement.
#[derive(Debug, Clone, Copy)]
pub struct FragmentPart<'a> {
    pub drawable: &'a Drawable,
    /// Applied to every model of the drawable.
    pub transform: Mat4,
    /// Per-bone pose applied before `transform`, indexed by each model's
    /// bone index; empty means every model stays where its vertices are.
    pub bone_transforms: &'a [Mat4],
}

/// Which wheel a physics child's bone tag names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WheelSlot {
    FrontLeft,
    FrontRight,
    RearLeft,
    RearRight,
}

impl WheelSlot {
    pub fn is_front(self) -> bool {
        matches!(self, WheelSlot::FrontLeft | WheelSlot::FrontRight)
    }

    pub fn is_right(self) -> bool {
        matches!(self, WheelSlot::FrontRight | WheelSlot::RearRight)
    }
}

/// Maps a vehicle wheel bone tag to its slot: wheel_lf/rf are the front
/// pair; wheel_lr/rr and the middle wheels wheel_lm1-3/rm1-3 count as rear,
/// the same grouping CodeWalker's renderer uses.
pub fn wheel_slot(bone_tag: u16) -> Option<WheelSlot> {
    match bone_tag {
        27922 => Some(WheelSlot::FrontLeft),
        26418 => Some(WheelSlot::FrontRight),
        27902 | 29921 | 29922 | 29923 => Some(WheelSlot::RearLeft),
        26398 | 5857 | 5858 | 5859 => Some(WheelSlot::RearRight),
        _ => None,
    }
}

impl Fragment {
    /// Everything to draw for a pristine preview of the fragment: the main
    /// drawable posed by the bone transforms, then every physics child that
    /// has a mesh, placed by its transform.
    ///
    /// Wheel slots without a mesh of their own borrow the front wheel's
    /// drawable (rear slots prefer a rear wheel), and right-hand wheels are
    /// mirrored in X and Z since the meshes are modelled for the left side.
    /// Children with no mesh and no wheel to borrow are skipped.
    pub fn render_parts(&self) -> Vec<FragmentPart<'_>> {
        let mut parts = Vec::new();
        if let Some(drawable) = &self.drawable {
            parts.push(FragmentPart {
                drawable,
                transform: Mat4::identity(),
                bone_transforms: &self.bone_transforms,
            });
        }

        let has_mesh = |child: &FragmentChild| {
            child.drawable.as_ref().is_some_and(|drawable| drawable.model_count() > 0)
        };
        let wheel_with_mesh = |front: bool| {
            self.children
                .iter()
                .find(|child| {
                    wheel_slot(child.bone_tag).is_some_and(|slot| slot.is_front() == front)
                        && has_mesh(child)
                })
                .and_then(|child| child.drawable.as_ref())
        };
        let front_wheel = wheel_with_mesh(true);
        let rear_wheel = wheel_with_mesh(false);

        for child in &self.children {
            let slot = wheel_slot(child.bone_tag);
            let drawable = if has_mesh(child) {
                child.drawable.as_ref()
            } else {
                match slot {
                    Some(slot) if slot.is_front() => front_wheel.or(rear_wheel),
                    Some(_) => rear_wheel.or(front_wheel),
                    None => None,
                }
            };
            let Some(drawable) = drawable else { continue };

            let transform = match slot {
                Some(slot) if slot.is_right() => mirror_wheel(&child.transform),
                _ => child.transform,
            };

            parts.push(FragmentPart { drawable, transform, bone_transforms: &[] });
        }

        parts
    }
}

/// A right-hand wheel's placement: the child's translation with the
/// rotation replaced by a flip of X and Z, as CodeWalker's renderer does.
fn mirror_wheel(transform: &Mat4) -> Mat4 {
    let mut flipped = Mat4::identity();
    flipped.0[0] = -1.0;
    flipped.0[10] = -1.0;
    flipped.with_translation(transform.translation())
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

    // The FragType is 0x130 bytes; the pose and physics pointers sit past
    // the 0x60 read above and are optional on short or older resources.
    let (bone_transforms_pointer, physics_group_pointer) = match reader.resolve(SYSTEM_BASE, 0xF8) {
        Some(raw) => (u64_le(raw, 0xA8), u64_le(raw, 0xF0)),
        None => (0, 0),
    };

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

    let bone_transforms = parse_bone_transforms(reader, bone_transforms_pointer);
    let children = parse_physics_children(reader, physics_group_pointer)?;

    Ok(Fragment {
        name,
        bound_center,
        bound_radius,
        drawable,
        extra_drawables,
        bone_transforms,
        children,
    })
}

/// `FragBoneTransforms`: a 0x20 header (item count at 0x10) followed by one
/// 3x4 matrix per bone — three rows of four floats, the fourth being the
/// translation. A missing or truncated block yields no pose.
fn parse_bone_transforms(reader: &ResReader<'_>, va: u64) -> Vec<Mat4> {
    const HEADER: usize = 0x20;
    const ITEM: usize = 48;

    let Some(header) = reader.resolve(va, HEADER) else { return Vec::new() };
    let count = header[0x10] as usize;
    let Some(items) = reader.resolve(va + HEADER as u64, count * ITEM) else { return Vec::new() };

    items
        .chunks_exact(ITEM)
        .map(|item| {
            let r1 = vec4_le(item, 0);
            let r2 = vec4_le(item, 16);
            let r3 = vec4_le(item, 32);
            // Row i holds the coefficients of output component i, so the
            // rows become the matrix's rows: m[col*4 + row].
            Mat4([
                r1.x, r2.x, r3.x, 0.0,
                r1.y, r2.y, r3.y, 0.0,
                r1.z, r2.z, r3.z, 0.0,
                r1.w, r2.w, r3.w, 1.0,
            ])
        })
        .collect()
}

/// Reads `FragPhysicsLODGroup -> PhysicsLOD1 -> Children`, placing each
/// child by the LOD's `FragTransforms` entry of the same index (shifted by
/// the LOD's position offset), or the identity when there is none.
fn parse_physics_children(reader: &ResReader<'_>, group_va: u64) -> Result<Vec<FragmentChild>> {
    let Some(group) = reader.resolve(group_va, 0x30) else { return Ok(Vec::new()) };
    let lod_pointer = u64_le(group, 0x10);
    let Some(lod) = reader.resolve(lod_pointer, 0x130) else { return Ok(Vec::new()) };

    let position_offset = vec3_le(lod, 0x30);
    let children_pointer = u64_le(lod, 0xD0);
    let transforms_pointer = u64_le(lod, 0x100);
    let children_count = lod[0x11D] as usize;

    let child_pointers = reader
        .read_u64_list(children_pointer, children_count)
        .context("physics children array out of bounds")?;
    let transforms = parse_physics_transforms(reader, transforms_pointer);

    let mut children = Vec::with_capacity(child_pointers.len());
    for (index, pointer) in child_pointers.iter().copied().enumerate() {
        let Some(raw) = reader.resolve(pointer, 0x100) else { continue };

        let group_index = u16_le(raw, 0x10);
        let bone_tag = u16_le(raw, 0x12);
        let drawable_pointer = u64_le(raw, 0xA0);

        let drawable = if drawable_pointer == 0 {
            None
        } else {
            Some(parse_drawable_at(
                reader,
                drawable_pointer,
                FRAG_DRAWABLE_NAME_OFFSET,
                FRAG_DRAWABLE_STRUCT_LEN,
                None,
            )
            .with_context(|| format!("physics child {index} drawable"))?)
        };

        let transform = match transforms.get(index) {
            Some(matrix) => matrix.with_translation(matrix.translation() + position_offset),
            None => Mat4::identity(),
        };

        children.push(FragmentChild { group_index, bone_tag, drawable, transform });
    }

    Ok(children)
}

/// `FragPhysTransforms`: a 0x20 header (count at 0x10) followed by one 4x4
/// D3D matrix (translation in the last row) per child.
fn parse_physics_transforms(reader: &ResReader<'_>, va: u64) -> Vec<Mat4> {
    const HEADER: usize = 0x20;
    const ITEM: usize = 64;

    let Some(header) = reader.resolve(va, HEADER) else { return Vec::new() };
    let count = u32_le(header, 0x10) as usize;
    let Some(items) = reader.resolve(va + HEADER as u64, count * ITEM) else { return Vec::new() };

    items
        .chunks_exact(ITEM)
        .map(|item| {
            let mut floats = [0.0f32; 16];
            for (i, value) in floats.iter_mut().enumerate() {
                *value = f32_le(item, i * 4);
            }
            Mat4::from_d3d(floats)
        })
        .collect()
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

    /// A fragment with a physics LOD holding two children — a front-left
    /// wheel with its own drawable and a front-right one without — plus a
    /// default bone pose. Each child is placed by its physics transform
    /// shifted by the LOD's position offset.
    #[test]
    fn parses_physics_children_and_bone_pose() {
        let mut system = vec![0u8; 0x2000];
        let graphics = vec![0u8; 0x10];
        let va = |offset: usize| SYSTEM_BASE + offset as u64;

        write_u64(&mut system, 0xA8, va(0x200)); // BoneTransformsPointer
        write_u64(&mut system, 0xF0, va(0x400)); // PhysicsLODGroupPointer

        // FragBoneTransforms: two Matrix3_s items after a 0x20 header.
        system[0x200 + 0x10] = 2;
        let rows = |data: &mut [u8], at: usize, r1: [f32; 4], r2: [f32; 4], r3: [f32; 4]| {
            for (i, v) in r1.iter().chain(&r2).chain(&r3).enumerate() {
                write_f32(data, at + i * 4, *v);
            }
        };
        rows(&mut system, 0x220, [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]);
        rows(&mut system, 0x250, [1.0, 0.0, 0.0, 7.0], [0.0, 1.0, 0.0, 8.0], [0.0, 0.0, 1.0, 9.0]);

        // FragPhysicsLODGroup -> PhysicsLOD1.
        write_u64(&mut system, 0x400 + 0x10, va(0x500));

        // FragPhysicsLOD: position offset, two children, transforms.
        write_f32(&mut system, 0x500 + 0x30, 0.5);
        write_u64(&mut system, 0x500 + 0xD0, va(0x700)); // ChildrenPointer
        write_u64(&mut system, 0x500 + 0x100, va(0x800)); // FragTransformsPointer
        system[0x500 + 0x11D] = 2; // ChildrenCount

        // Child pointer array.
        write_u64(&mut system, 0x700, va(0x900));
        write_u64(&mut system, 0x708, va(0xA00));

        // FragPhysTransforms: count 2, D3D matrices from +0x20.
        write_u32(&mut system, 0x800 + 0x10, 2);
        let identity_at = |data: &mut [u8], at: usize, t: [f32; 3]| {
            for i in 0..4 {
                write_f32(data, at + (i * 4 + i) * 4, 1.0);
            }
            for (i, v) in t.iter().enumerate() {
                write_f32(data, at + (12 + i) * 4, *v);
            }
        };
        identity_at(&mut system, 0x820, [1.0, 2.0, 3.0]);
        identity_at(&mut system, 0x860, [-1.0, 2.0, 3.0]);

        // Child 0: wheel_lf with a drawable; child 1: wheel_rf without.
        write_u16(&mut system, 0x900 + 0x10, 3);
        write_u16(&mut system, 0x900 + 0x12, 27922);
        write_u64(&mut system, 0x900 + 0xA0, va(0xB00));
        write_u16(&mut system, 0xA00 + 0x12, 26418);

        // The wheel FragDrawable: one model, no geometry, named.
        write_u64(&mut system, 0xB00 + 0x130, va(0xD00));
        system[0xD00..0xD09].copy_from_slice(b"wheel_lf\0");
        write_u64(&mut system, 0xB00 + 0x50, va(0xE00));
        write_u64(&mut system, 0xE00, va(0xE10));
        write_u16(&mut system, 0xE08, 1);
        write_u16(&mut system, 0xE0A, 1);
        write_u64(&mut system, 0xE10, va(0xE40));

        let reader = ResReader { system: &system, graphics: &graphics };
        let fragment = parse_yft_from_reader(&reader).expect("fixture should parse");

        assert_eq!(fragment.bone_transforms.len(), 2);
        assert!(fragment.bone_transforms[0].is_identity());
        assert_eq!(fragment.bone_transforms[1].translation(), Vec3::new(7.0, 8.0, 9.0));

        assert_eq!(fragment.children.len(), 2);
        let lf = &fragment.children[0];
        assert_eq!((lf.group_index, lf.bone_tag), (3, 27922));
        assert_eq!(lf.drawable.as_ref().map(|d| d.name.as_str()), Some("wheel_lf"));
        assert_eq!(lf.transform.translation(), Vec3::new(1.5, 2.0, 3.0));

        let rf = &fragment.children[1];
        assert_eq!(rf.bone_tag, 26418);
        assert!(rf.drawable.is_none());
        assert_eq!(rf.transform.translation(), Vec3::new(-0.5, 2.0, 3.0));
    }

    /// Wheel slots that have no mesh of their own borrow the front (or rear)
    /// wheel drawable, and right-hand wheels are mirrored, as CodeWalker does.
    #[test]
    fn render_parts_fill_empty_wheel_slots_and_mirror_the_right_side() {
        let wheel = |name: &str| Drawable {
            name: name.to_string(),
            name_hash: rage_joaat(name),
            bounds: crate::ydd::DrawableBounds {
                center: Vec3::ZERO,
                sphere_radius: 0.0,
                box_min: Vec3::ZERO,
                box_max: Vec3::ZERO,
            },
            lod_distances: [0.0; 4],
            render_masks: [0; 4],
            shader_group: None,
            lods: vec![crate::ydd::DrawableLod {
                level: crate::ydd::LodLevel::High,
                models: vec![crate::ydd::DrawableModel {
                    skeleton_binding: 0,
                    render_mask_flags: 0,
                    shader_mapping: Vec::new(),
                    geometries: Vec::new(),
                }],
            }],
        };
        let child = |bone_tag: u16, drawable: Option<Drawable>, x: f32| FragmentChild {
            group_index: 0,
            bone_tag,
            drawable,
            transform: Mat4::from_translation(Vec3::new(x, 0.0, 0.0)),
        };

        let fragment = Fragment {
            name: "car".to_string(),
            bound_center: Vec3::ZERO,
            bound_radius: 1.0,
            drawable: Some(wheel("chassis")),
            extra_drawables: Vec::new(),
            bone_transforms: vec![Mat4::identity()],
            children: vec![
                child(27922, Some(wheel("wheel_lf")), 1.0), // front left, has mesh
                child(26418, None, -1.0),                   // front right, empty slot
                child(27902, None, 1.0),                    // rear left, empty slot
                child(26398, None, -1.0),                   // rear right, empty slot
                child(1234, None, 5.0),                     // not a wheel: skipped
            ],
        };

        let parts = fragment.render_parts();
        let names: Vec<&str> = parts.iter().map(|part| part.drawable.name.as_str()).collect();
        assert_eq!(names, ["chassis", "wheel_lf", "wheel_lf", "wheel_lf", "wheel_lf"]);

        assert!(parts[0].transform.is_identity());
        assert_eq!(parts[0].bone_transforms.len(), 1, "the body is posed by the bone transforms");
        assert!(parts[1..].iter().all(|part| part.bone_transforms.is_empty()));

        // Left wheels keep their transform; right wheels are mirrored in X and Z.
        assert_eq!(parts[1].transform.transform_vector(Vec3::X), Vec3::X);
        assert_eq!(parts[2].transform.transform_vector(Vec3::X), -Vec3::X);
        assert_eq!(parts[2].transform.transform_vector(Vec3::Z), -Vec3::Z);
        assert_eq!(parts[2].transform.transform_vector(Vec3::Y), Vec3::Y);
        assert_eq!(parts[2].transform.translation(), Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(parts[4].transform.translation(), Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn wheel_slots_are_recognised_by_bone_tag() {
        assert_eq!(wheel_slot(27922), Some(WheelSlot::FrontLeft));
        assert_eq!(wheel_slot(26418), Some(WheelSlot::FrontRight));
        assert_eq!(wheel_slot(29922), Some(WheelSlot::RearLeft));
        assert_eq!(wheel_slot(5859), Some(WheelSlot::RearRight));
        assert_eq!(wheel_slot(26398), Some(WheelSlot::RearRight));
        assert_eq!(wheel_slot(0), None);
    }

    fn write_u32(data: &mut [u8], offset: usize, value: u32) {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
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
