use anyhow::{bail, Result};
use flate2::read::DeflateDecoder;
use std::io::Read;
use crate::archive::{resource_size_from_flags, RSC7_MAGIC};
use crate::math::{Vec3, Vec4};

pub const SYSTEM_BASE: u64 = 0x5000_0000;
pub const GRAPHICS_BASE: u64 = 0x6000_0000;

// ─── Internal virtual-memory reader ──────────────────────────────────────────

pub struct ResReader<'a> {
    pub system:   &'a [u8],
    pub graphics: &'a [u8],
}

/// Header of a `atArray`/pointer-list style structure: a pointer to the
/// backing array, followed by a `u16` count and a `u16` capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointerListHeader {
    pub pointer: u64,
    pub count: u16,
    pub capacity: u16,
}

impl<'a> ResReader<'a> {
    pub fn resolve(&self, va: u64, len: usize) -> Option<&'a [u8]> {
        if va == 0 { return None; }
        if (va & 0x50000000) == 0x50000000 && (va & 0x60000000) != 0x60000000 {
            let off = (va - 0x50000000) as usize;
            self.system.get(off..off.checked_add(len)?)
        } else if (va & 0x60000000) == 0x60000000 {
            let off = (va - 0x60000000) as usize;
            self.graphics.get(off..off.checked_add(len)?)
        } else {
            None
        }
    }

    /// Like [`Self::resolve`], but distinguishes a null pointer (`va == 0`,
    /// returns `Some(None)`) from an out-of-bounds pointer (`None`).
    pub fn resolve_optional(&self, va: u64, len: usize) -> Option<Option<&'a [u8]>> {
        if va == 0 {
            return Some(None);
        }
        self.resolve(va, len).map(Some)
    }

    pub fn read_u16_list(&self, va: u64, count: usize) -> Option<Vec<u16>> {
        if count == 0 || va == 0 {
            return Some(Vec::new());
        }
        let bytes = self.resolve(va, count.checked_mul(2)?)?;
        Some(bytes.chunks_exact(2).map(|c| u16_le(c, 0)).collect())
    }

    pub fn read_u32_list(&self, va: u64, count: usize) -> Option<Vec<u32>> {
        if count == 0 || va == 0 {
            return Some(Vec::new());
        }
        let bytes = self.resolve(va, count.checked_mul(4)?)?;
        Some(bytes.chunks_exact(4).map(|c| u32_le(c, 0)).collect())
    }

    pub fn read_u64_list(&self, va: u64, count: usize) -> Option<Vec<u64>> {
        if count == 0 || va == 0 {
            return Some(Vec::new());
        }
        let bytes = self.resolve(va, count.checked_mul(8)?)?;
        Some(bytes.chunks_exact(8).map(|c| u64_le(c, 0)).collect())
    }

    /// Reads a 16-byte pointer-list header: pointer@0, count@8, capacity@10.
    pub fn read_pointer_list_header(&self, va: u64) -> Option<PointerListHeader> {
        let bytes = self.resolve(va, 16)?;
        Some(PointerListHeader {
            pointer: u64_le(bytes, 0),
            count: u16_le(bytes, 8),
            capacity: u16_le(bytes, 10),
        })
    }

    /// Longest string this reads before giving up on finding a terminating
    /// NUL. Real names are short; a runaway scan across the rest of the
    /// section usually means the pointer is bogus, so it's treated as an
    /// unresolved string (`None`) rather than returned truncated.
    const MAX_STRING_LEN: usize = 256;

    pub fn string_at(&self, va: u64) -> Option<String> {
        if (va & 0x50000000) == 0x50000000 && (va & 0x60000000) != 0x60000000 {
            let off = (va - 0x50000000) as usize;
            let slice = self.system.get(off..)?;
            let scan_len = slice.len().min(Self::MAX_STRING_LEN);
            let end = slice[..scan_len].iter().position(|&b| b == 0)?;
            Some(String::from_utf8_lossy(&slice[..end]).into_owned())
        } else {
            None
        }
    }
}

pub fn u16_le(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap_or([0; 2]))
}
pub fn u32_le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap_or([0; 4]))
}
pub fn u64_le(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap_or([0; 8]))
}
pub fn f32_le(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(b[off..off + 4].try_into().unwrap_or([0; 4]))
}
pub fn vec3_le(b: &[u8], off: usize) -> Vec3 {
    Vec3::new(f32_le(b, off), f32_le(b, off + 4), f32_le(b, off + 8))
}
pub fn vec4_le(b: &[u8], off: usize) -> Vec4 {
    Vec4::new(f32_le(b, off), f32_le(b, off + 4), f32_le(b, off + 8), f32_le(b, off + 12))
}

/// Helper to decompress and prepare RSC7 resource sections.
pub fn prepare_rsc7(data: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    if data.len() < 16 {
        bail!("RSC7 data too short");
    }

    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != RSC7_MAGIC {
        bail!("Not an RSC7 file (magic = 0x{:08X})", magic);
    }

    let system_flags  = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let graphics_flags = u32::from_le_bytes(data[12..16].try_into().unwrap());

    let sys_size  = resource_size_from_flags(system_flags);
    let gfx_size  = resource_size_from_flags(graphics_flags);
    let body      = &data[16..];

    // Most resources are deflated, but a few are stored raw. Telling the two
    // apart by whether inflate succeeds is fine; what matters is not confusing
    // a *corrupt* stream for a stored one, because feeding the still-compressed
    // bytes on as though they were the resource produces wild pointers far
    // downstream instead of naming the real problem here.
    let decompressed = {
        let mut out = Vec::new();
        match DeflateDecoder::new(body).read_to_end(&mut out) {
            Ok(_) if !out.is_empty() => out,
            Ok(_) => body.to_vec(),
            // Never looked like deflate at all — treat it as stored.
            Err(_) if out.is_empty() => body.to_vec(),
            Err(_) => bail!(
                "corrupt deflate stream: inflated {} of an expected {} bytes before failing",
                out.len(),
                sys_size + gfx_size
            ),
        }
    };

    if decompressed.len() < sys_size {
        bail!(
            "Decompressed size {} < expected system size {}",
            decompressed.len(), sys_size
        );
    }

    let system = decompressed[..sys_size].to_vec();
    let graphics = if decompressed.len() >= sys_size + gfx_size {
        decompressed[sys_size..sys_size + gfx_size].to_vec()
    } else {
        decompressed[sys_size..].to_vec()
    };

    Ok((system, graphics))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a system buffer with a 16-byte pointer-list header at offset 0
    /// (pointer -> 0x100 within the system section, count=3, capacity=4)
    /// followed by three little-endian u32s at 0x50000100.
    fn build_system_buffer() -> Vec<u8> {
        let mut sys = vec![0u8; 0x200];

        // Pointer-list header at offset 0.
        let array_va = SYSTEM_BASE + 0x100;
        sys[0..8].copy_from_slice(&array_va.to_le_bytes());
        sys[8..10].copy_from_slice(&3u16.to_le_bytes());
        sys[10..12].copy_from_slice(&4u16.to_le_bytes());

        // Three u32s at 0x100.
        sys[0x100..0x104].copy_from_slice(&11u32.to_le_bytes());
        sys[0x104..0x108].copy_from_slice(&22u32.to_le_bytes());
        sys[0x108..0x10C].copy_from_slice(&33u32.to_le_bytes());

        sys
    }

    #[test]
    fn resolve_optional_null_out_of_bounds_and_valid() {
        let sys = build_system_buffer();
        let reader = ResReader { system: &sys, graphics: &[] };

        // va == 0 -> Some(None)
        assert_eq!(reader.resolve_optional(0, 4), Some(None));

        // Out of bounds -> None
        let far_va = SYSTEM_BASE + sys.len() as u64 + 0x1000;
        assert_eq!(reader.resolve_optional(far_va, 4), None);

        // Valid -> Some(Some(bytes))
        let array_va = SYSTEM_BASE + 0x100;
        let resolved = reader.resolve_optional(array_va, 4).expect("should resolve");
        let bytes = resolved.expect("should be Some(bytes)");
        assert_eq!(u32_le(bytes, 0), 11);
    }

    #[test]
    fn read_u32_list_reads_values() {
        let sys = build_system_buffer();
        let reader = ResReader { system: &sys, graphics: &[] };

        let array_va = SYSTEM_BASE + 0x100;
        let values = reader.read_u32_list(array_va, 3).expect("should read list");
        assert_eq!(values, vec![11, 22, 33]);

        // count == 0 -> empty vec, even for a null va.
        assert_eq!(reader.read_u32_list(0, 0), Some(Vec::new()));
    }

    #[test]
    fn read_lists_reject_counts_that_would_overflow_the_byte_length() {
        let sys = build_system_buffer();
        let reader = ResReader { system: &sys, graphics: &[] };
        let array_va = SYSTEM_BASE + 0x100;

        // On a 32-bit `usize` (wasm32), `count * N` wraps around instead of
        // overflowing, so pick a count that overflows even a 64-bit `usize`
        // multiplication to make the assertion hold on every target.
        let huge_count = usize::MAX / 4 + 1;

        assert_eq!(reader.read_u16_list(array_va, huge_count), None);
        assert_eq!(reader.read_u32_list(array_va, huge_count), None);
        assert_eq!(reader.read_u64_list(array_va, huge_count), None);
    }

    #[test]
    fn resolve_rejects_offset_length_overflow_instead_of_panicking() {
        let reader = ResReader { system: &[], graphics: &[] };

        // A `va` with bit 29 (system) or bit 30 (graphics) set near `u64::MAX`
        // plus a huge `len` used to overflow `off + len` in debug builds.
        assert_eq!(reader.resolve(u64::MAX, usize::MAX), None);
    }

    #[test]
    fn string_at_gives_up_past_the_scan_cap_instead_of_returning_untruncated() {
        // No NUL anywhere in a buffer larger than the scan cap -> None, not a
        // huge (or truncated) string.
        let sys = vec![b'A'; 512];
        let reader = ResReader { system: &sys, graphics: &[] };
        assert_eq!(reader.string_at(SYSTEM_BASE), None);

        // A NUL just past the cap still isn't found.
        let mut sys = vec![b'A'; 300];
        sys[300 - 1] = 0; // NUL at index 299, past the 256-byte scan window
        let reader = ResReader { system: &sys, graphics: &[] };
        assert_eq!(reader.string_at(SYSTEM_BASE), None);

        // A NUL within the cap still works normally.
        let mut sys = vec![b'A'; 10];
        sys[5] = 0;
        let reader = ResReader { system: &sys, graphics: &[] };
        assert_eq!(reader.string_at(SYSTEM_BASE), Some("AAAAA".to_string()));
    }

    #[test]
    fn read_pointer_list_header_reads_fields() {
        let sys = build_system_buffer();
        let reader = ResReader { system: &sys, graphics: &[] };

        let header = reader.read_pointer_list_header(SYSTEM_BASE).expect("should read header");
        assert_eq!(header.pointer, SYSTEM_BASE + 0x100);
        assert_eq!(header.count, 3);
        assert_eq!(header.capacity, 4);
    }
}
