/// YTD (Texture Dictionary) parser for GTA V (Gen8 / PC format).
///
/// Accepts the standalone RSC7 bytes as returned by `RpfArchive::extract_entry`.
use anyhow::{Result, Context};
use crate::resource::{ResReader, prepare_rsc7, u16_le, u32_le, u64_le, SYSTEM_BASE};

// ─── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TextureFormat {
    A8R8G8B8 = 21,
    X8R8G8B8 = 22,
    A1R5G5B5 = 25,
    A8       = 28,
    A8B8G8R8 = 32,
    L8       = 50,
    DXT1     = 0x31545844,
    DXT3     = 0x33545844,
    DXT5     = 0x35545844,
    ATI1     = 0x31495441,
    ATI2     = 0x32495441,
    BC7      = 0x20374342,
    Unknown  = 0,
}

impl TextureFormat {
    pub fn from_u32(v: u32) -> Self {
        match v {
            21          => Self::A8R8G8B8,
            22          => Self::X8R8G8B8,
            25          => Self::A1R5G5B5,
            28          => Self::A8,
            32          => Self::A8B8G8R8,
            50          => Self::L8,
            0x31545844  => Self::DXT1,
            0x33545844  => Self::DXT3,
            0x35545844  => Self::DXT5,
            0x31495441  => Self::ATI1,
            0x32495441  => Self::ATI2,
            0x20374342  => Self::BC7,
            _           => Self::Unknown,
        }
    }

    pub fn is_block_compressed(self) -> bool {
        matches!(self, Self::DXT1 | Self::DXT3 | Self::DXT5 | Self::ATI1 | Self::ATI2 | Self::BC7)
    }
}

impl std::fmt::Display for TextureFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::A8R8G8B8 => "A8R8G8B8",
            Self::X8R8G8B8 => "X8R8G8B8",
            Self::A1R5G5B5 => "A1R5G5B5",
            Self::A8       => "A8",
            Self::A8B8G8R8 => "A8B8G8R8",
            Self::L8       => "L8",
            Self::DXT1     => "DXT1",
            Self::DXT3     => "DXT3",
            Self::DXT5     => "DXT5",
            Self::ATI1     => "ATI1",
            Self::ATI2     => "ATI2",
            Self::BC7      => "BC7",
            Self::Unknown  => "Unknown",
        };
        f.write_str(s)
    }
}

/// One texture entry extracted from a YTD.
#[derive(Debug, Clone)]
pub struct YtdTexture {
    pub name: String,
    pub name_hash: u32,
    pub width: u16,
    pub height: u16,
    pub depth: u16,
    pub format: TextureFormat,
    pub levels: u8,
    pub stride: u16,
    pub pixel_data: Vec<u8>,
}

impl YtdTexture {
    /// Serialize this texture to a DDS file.
    pub fn to_dds(&self) -> Vec<u8> {
        let mut out = Vec::new();
        // DDS magic
        out.extend_from_slice(b"DDS ");

        // DDS_HEADER (124 bytes)
        let has_mips = self.levels > 1;
        let is_compressed = self.format.is_block_compressed();

        let mut flags: u32 = 0x1 | 0x2 | 0x4 | 0x1000; // CAPS | HEIGHT | WIDTH | PIXELFORMAT
        if has_mips { flags |= 0x20000; } // MIPMAPCOUNT
        if is_compressed { flags |= 0x80000; } else { flags |= 0x8; } // LINEARSIZE or PITCH

        let pitch_or_linear: u32 = self.stride as u32 * self.height as u32;

        out.extend_from_slice(&124u32.to_le_bytes());            // dwSize
        out.extend_from_slice(&flags.to_le_bytes());             // dwFlags
        out.extend_from_slice(&(self.height as u32).to_le_bytes()); // dwHeight
        out.extend_from_slice(&(self.width as u32).to_le_bytes());  // dwWidth
        out.extend_from_slice(&pitch_or_linear.to_le_bytes());   // dwPitchOrLinearSize
        out.extend_from_slice(&(self.depth as u32).to_le_bytes()); // dwDepth
        out.extend_from_slice(&(self.levels as u32).to_le_bytes()); // dwMipMapCount
        out.extend_from_slice(&[0u8; 44]);                       // dwReserved1[11]

        // DDS_PIXELFORMAT (32 bytes)
        self.write_pixelformat(&mut out);

        let mut caps: u32 = 0x1000; // DDSCAPS_TEXTURE
        if has_mips { caps |= 0x8 | 0x400000; } // COMPLEX | MIPMAP
        out.extend_from_slice(&caps.to_le_bytes());
        out.extend_from_slice(&[0u8; 16]); // Caps2/3/4 + Reserved2

        // Pixel data
        if self.format == TextureFormat::BC7 {
            write_dx10_header(&mut out);
        }
        out.extend_from_slice(&self.pixel_data);

        out
    }

    fn write_pixelformat(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&32u32.to_le_bytes()); // dwSize
        match self.format {
            TextureFormat::DXT1 | TextureFormat::DXT3 | TextureFormat::DXT5
            | TextureFormat::ATI1 | TextureFormat::ATI2 => {
                out.extend_from_slice(&0x4u32.to_le_bytes()); // DDPF_FOURCC
                out.extend_from_slice(&(self.format as u32).to_le_bytes()); // FourCC
                out.extend_from_slice(&[0u8; 20]); // RGB counts + masks
            }
            TextureFormat::BC7 => {
                out.extend_from_slice(&0x4u32.to_le_bytes()); // DDPF_FOURCC
                out.extend_from_slice(b"DX10");               // FourCC = DX10
                out.extend_from_slice(&[0u8; 20]);
            }
            TextureFormat::A8R8G8B8 => {
                out.extend_from_slice(&(0x1 | 0x40u32).to_le_bytes()); // ALPHAPIXELS | RGB
                out.extend_from_slice(&[0u8; 4]); // FourCC = 0
                out.extend_from_slice(&32u32.to_le_bytes()); // dwRGBBitCount
                out.extend_from_slice(&0x00FF0000u32.to_le_bytes()); // RMask
                out.extend_from_slice(&0x0000FF00u32.to_le_bytes()); // GMask
                out.extend_from_slice(&0x000000FFu32.to_le_bytes()); // BMask
                out.extend_from_slice(&0xFF000000u32.to_le_bytes()); // AMask
            }
            _ => {
                out.extend_from_slice(&[0u8; 28]);
            }
        }
    }
}

fn write_dx10_header(out: &mut Vec<u8>) {
    out.extend_from_slice(&98u32.to_le_bytes()); // DXGI_FORMAT_BC7_UNORM
    out.extend_from_slice(&3u32.to_le_bytes());  // D3D10_RESOURCE_DIMENSION_TEXTURE2D
    out.extend_from_slice(&0u32.to_le_bytes());  // miscFlag
    out.extend_from_slice(&1u32.to_le_bytes());  // arraySize
    out.extend_from_slice(&0u32.to_le_bytes());  // miscFlags2
}

// ─── Parser ───────────────────────────────────────────────────────────────────

pub fn parse_ytd(data: &[u8]) -> Result<Vec<YtdTexture>> {
    let (system, graphics) = prepare_rsc7(data)?;
    let reader = ResReader { system: &system, graphics: &graphics };
    parse_texture_dict_at(&reader, SYSTEM_BASE)
}

pub(crate) fn parse_texture_dict_at(reader: &ResReader<'_>, va: u64) -> Result<Vec<YtdTexture>> {
    let dict = reader.resolve(va, 0x40)
        .ok_or_else(|| anyhow::anyhow!("system section too small for TextureDictionary"))?;

    let hash_ptr   = u64_le(dict, 0x20);
    let hash_count = u16_le(dict, 0x28) as usize;
    let tex_ptr_array = u64_le(dict, 0x30);
    let tex_count     = u16_le(dict, 0x38) as usize;

    let hash_data = if hash_count > 0 {
        reader.resolve(hash_ptr, hash_count * 4)
    } else {
        None
    };

    let ptr_data = if tex_count > 0 {
        reader.resolve(tex_ptr_array, tex_count * 8)
            .with_context(|| format!("texture pointer array out of bounds (va=0x{:X})", tex_ptr_array))?
    } else {
        return Ok(vec![]);
    };

    let mut textures = Vec::with_capacity(tex_count);
    for i in 0..tex_count {
        let tex_va = u64_le(ptr_data, i * 8);
        if tex_va == 0 { continue; }

        let name_hash = hash_data
            .and_then(|h| h.get(i * 4..i * 4 + 4))
            .map(|b| u32_le(b, 0))
            .unwrap_or(0);

        match parse_texture(tex_va, name_hash, reader) {
            Ok(tex) => textures.push(tex),
            Err(e) => eprintln!("[YTD] Warning: texture {} parse error: {}", i, e),
        }
    }

    Ok(textures)
}

fn parse_texture(tex_va: u64, name_hash: u32, reader: &ResReader<'_>) -> Result<YtdTexture> {
    let raw = reader.resolve(tex_va, 0x90)
        .with_context(|| format!("texture struct out of bounds (va=0x{:X})", tex_va))?;

    let name_ptr = u64_le(raw, 0x28);
    let width  = u16_le(raw, 0x50);
    let height = u16_le(raw, 0x52);
    let depth  = u16_le(raw, 0x54);
    let stride = u16_le(raw, 0x56);
    let fmt    = TextureFormat::from_u32(u32_le(raw, 0x58));
    let levels = raw[0x5D];
    let data_ptr = u64_le(raw, 0x70);

    let name = reader.string_at(name_ptr).unwrap_or_default();
    let pixel_size = calc_pixel_data_size(stride, height, levels);

    let pixel_data = if pixel_size > 0 && data_ptr != 0 {
        reader.resolve(data_ptr, pixel_size)
            .with_context(|| format!("pixel data out of bounds (va=0x{:X}, size={})", data_ptr, pixel_size))?
            .to_vec()
    } else {
        vec![]
    };

    Ok(YtdTexture { name, name_hash, width, height, depth, format: fmt, levels, stride, pixel_data })
}

fn calc_pixel_data_size(stride: u16, height: u16, levels: u8) -> usize {
    let mut total = 0usize;
    let mut length = stride as usize * height as usize;
    for _ in 0..levels {
        total += length;
        length /= 4;
    }
    total
}
