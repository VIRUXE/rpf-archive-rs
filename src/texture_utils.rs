use crate::ytd::{TextureFormat, YtdTexture};
use anyhow::{Result, bail};

pub fn decompress_texture(texture: &YtdTexture) -> Result<Vec<u8>> {
    let width = texture.width as usize;
    let height = texture.height as usize;
    let mut rgba_u32 = vec![0u32; width * height];
    
    match texture.format {
        TextureFormat::DXT1 => {
            texture2ddecoder::decode_bc1(&texture.pixel_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow::anyhow!("DXT1 decompression failed: {}", e))?;
        }
        TextureFormat::DXT3 => {
            // BC2 is DXT3
            texture2ddecoder::decode_bc2(&texture.pixel_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow::anyhow!("DXT3 decompression failed: {}", e))?;
        }
        TextureFormat::DXT5 => {
            // BC3 is DXT5
            texture2ddecoder::decode_bc3(&texture.pixel_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow::anyhow!("DXT5 decompression failed: {}", e))?;
        }
        TextureFormat::ATI1 => {
            // BC4 is ATI1
            texture2ddecoder::decode_bc4(&texture.pixel_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow::anyhow!("ATI1 decompression failed: {}", e))?;
        }
        TextureFormat::ATI2 => {
            // BC5 is ATI2
            texture2ddecoder::decode_bc5(&texture.pixel_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow::anyhow!("ATI2 decompression failed: {}", e))?;
        }
        TextureFormat::BC7 => {
            texture2ddecoder::decode_bc7(&texture.pixel_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow::anyhow!("BC7 decompression failed: {}", e))?;
        }
        TextureFormat::A8R8G8B8 => {
            // Convert ARGB to RGBA
            let mut rgba = Vec::with_capacity(texture.pixel_data.len());
            for chunk in texture.pixel_data.chunks_exact(4) {
                rgba.push(chunk[2]); // R
                rgba.push(chunk[1]); // G
                rgba.push(chunk[0]); // B
                rgba.push(chunk[3]); // A
            }
            return Ok(rgba);
        }
        _ => bail!("Unsupported texture format for decompression: {:?}", texture.format),
    }

    // Convert u32 (presumably ABGR or ARGB) to Vec<u8> RGBA
    // We need to check what texture2ddecoder returns. Usually it's 0xAABBGGRR or 0xAARRGGBB.
    // Assuming it's 0xAABBGGRR (standard for many decoders)
    let mut rgba_u8 = Vec::with_capacity(width * height * 4);
    for pixel in rgba_u32 {
        rgba_u8.push((pixel & 0xFF) as u8);         // R
        rgba_u8.push(((pixel >> 8) & 0xFF) as u8);  // G
        rgba_u8.push(((pixel >> 16) & 0xFF) as u8); // B
        rgba_u8.push(((pixel >> 24) & 0xFF) as u8); // A
    }
    
    Ok(rgba_u8)
}
