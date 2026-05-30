use anyhow::{bail, Result, Context};
use flate2::read::DeflateDecoder;
use std::io::Read;
use crate::archive::{resource_size_from_flags, RSC7_MAGIC};

// ─── Internal virtual-memory reader ──────────────────────────────────────────

pub struct ResReader<'a> {
    pub system:   &'a [u8],
    pub graphics: &'a [u8],
}

impl<'a> ResReader<'a> {
    pub fn resolve(&self, va: u64, len: usize) -> Option<&'a [u8]> {
        if va == 0 { return None; }
        if (va & 0x50000000) == 0x50000000 && (va & 0x60000000) != 0x60000000 {
            let off = (va - 0x50000000) as usize;
            self.system.get(off..off + len)
        } else if (va & 0x60000000) == 0x60000000 {
            let off = (va - 0x60000000) as usize;
            self.graphics.get(off..off + len)
        } else {
            None
        }
    }

    pub fn string_at(&self, va: u64) -> Option<String> {
        if (va & 0x50000000) == 0x50000000 && (va & 0x60000000) != 0x60000000 {
            let off = (va - 0x50000000) as usize;
            let slice = self.system.get(off..)?;
            let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
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

    // Decompress
    let decompressed = {
        let mut out = Vec::new();
        if DeflateDecoder::new(body).read_to_end(&mut out).is_ok() && !out.is_empty() {
            out
        } else {
            // Try raw (uncompressed) body
            body.to_vec()
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
