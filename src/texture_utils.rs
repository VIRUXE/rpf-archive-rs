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

    // texture2ddecoder packs each pixel as 0xAARRGGBB, so red is the high
    // colour byte and blue the low one. Reading them the other way round
    // renders the game's reds as blues.
    let mut rgba_u8 = Vec::with_capacity(width * height * 4);
    for pixel in rgba_u32 {
        rgba_u8.push(((pixel >> 16) & 0xFF) as u8); // R
        rgba_u8.push(((pixel >> 8) & 0xFF) as u8);  // G
        rgba_u8.push((pixel & 0xFF) as u8);         // B
        rgba_u8.push(((pixel >> 24) & 0xFF) as u8); // A
    }
    
    Ok(rgba_u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(format: TextureFormat, pixel_data: Vec<u8>) -> YtdTexture {
        YtdTexture {
            name: "test".into(),
            name_hash: 0,
            width: 4,
            height: 4,
            depth: 1,
            format,
            levels: 1,
            stride: 0,
            pixel_data,
        }
    }

    /// A DXT1 block whose every texel is pure red must decode to pure red,
    /// not pure blue — the channel order is easy to get backwards.
    #[test]
    fn block_compressed_red_stays_red() {
        // color0 = 0xF800 (R=31, G=0, B=0), color1 = 0x0000, all indices 0.
        let block = vec![0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let rgba = decompress_texture(&texture(TextureFormat::DXT1, block)).unwrap();

        assert_eq!(rgba.len(), 4 * 4 * 4);
        for texel in rgba.chunks_exact(4) {
            assert_eq!(texel[0], 255, "red channel");
            assert_eq!(texel[1], 0, "green channel");
            assert_eq!(texel[2], 0, "blue channel");
            assert_eq!(texel[3], 255, "alpha channel");
        }
    }

    #[test]
    fn uncompressed_argb_is_reordered() {
        // One BGRA-ordered texel on disk: B=0x11, G=0x22, R=0x33, A=0x44.
        let rgba = decompress_texture(&texture(
            TextureFormat::A8R8G8B8,
            vec![0x11, 0x22, 0x33, 0x44],
        ))
        .unwrap();

        assert_eq!(rgba, vec![0x33, 0x22, 0x11, 0x44]);
    }
}
