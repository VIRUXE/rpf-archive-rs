use crate::ytd::{TextureFormat, YtdTexture};
use anyhow::{Result, bail};

/// Uncompressed pixel data carries the whole mip chain back to back, so only
/// the leading `width * height * bytes_per_pixel` bytes belong to the top
/// level. Block-compressed formats are handled by the decoder itself, which
/// reads exactly the blocks it needs.
fn top_level(data: &[u8], width: usize, height: usize, bytes_per_pixel: usize) -> &[u8] {
    let wanted = width.saturating_mul(height).saturating_mul(bytes_per_pixel);
    if data.len() > wanted { &data[..wanted] } else { data }
}

/// Decodes the top mip level of `texture` to RGBA bytes.
pub fn decompress_texture(texture: &YtdTexture) -> Result<Vec<u8>> {
    let rgba = decode_top_level(texture)?;

    // Callers hand these bytes straight to an image buffer, and
    // `RgbaImage::from_raw` accepts an over-long one, so a wrong length only
    // shows up much later as an assertion inside an encoder. Catching it here
    // keeps the complaint attached to the texture it came from, and makes
    // truncated input an error for every caller rather than just for
    // `to_rgba_image`.
    let wanted = texture.width as usize * texture.height as usize * 4;
    if rgba.len() != wanted {
        bail!(
            "decoded {} pixel bytes for a {}x{} {:?} texture, expected {}",
            rgba.len(), texture.width, texture.height, texture.format, wanted
        );
    }

    Ok(rgba)
}

fn decode_top_level(texture: &YtdTexture) -> Result<Vec<u8>> {
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
            let data = top_level(&texture.pixel_data, width, height, 4);
            let mut rgba = Vec::with_capacity(data.len());
            for chunk in data.chunks_exact(4) {
                rgba.push(chunk[2]); // R
                rgba.push(chunk[1]); // G
                rgba.push(chunk[0]); // B
                rgba.push(chunk[3]); // A
            }
            return Ok(rgba);
        }
        TextureFormat::X8R8G8B8 => {
            // BGRX bytes on disk -> RGBA with A=255.
            let data = top_level(&texture.pixel_data, width, height, 4);
            let mut rgba = Vec::with_capacity(data.len());
            for chunk in data.chunks_exact(4) {
                rgba.push(chunk[2]); // R
                rgba.push(chunk[1]); // G
                rgba.push(chunk[0]); // B
                rgba.push(255);      // A
            }
            return Ok(rgba);
        }
        TextureFormat::A8B8G8R8 => {
            // Already RGBA byte order on disk.
            return Ok(top_level(&texture.pixel_data, width, height, 4).to_vec());
        }
        TextureFormat::L8 => {
            // Grey -> R=G=B=L, A=255.
            let data = top_level(&texture.pixel_data, width, height, 1);
            let mut rgba = Vec::with_capacity(data.len() * 4);
            for &l in data {
                rgba.push(l);
                rgba.push(l);
                rgba.push(l);
                rgba.push(255);
            }
            return Ok(rgba);
        }
        TextureFormat::A8 => {
            // R=G=B=255, A=value.
            let data = top_level(&texture.pixel_data, width, height, 1);
            let mut rgba = Vec::with_capacity(data.len() * 4);
            for &a in data {
                rgba.push(255);
                rgba.push(255);
                rgba.push(255);
                rgba.push(a);
            }
            return Ok(rgba);
        }
        TextureFormat::A1R5G5B5 => {
            // 16-bit little-endian, 1 alpha bit, 5-bit channels expanded to 8-bit.
            let data = top_level(&texture.pixel_data, width, height, 2);
            let mut rgba = Vec::with_capacity(data.len() * 2);
            for chunk in data.chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                let a1 = (v >> 15) & 0x1;
                let r5 = (v >> 10) & 0x1F;
                let g5 = (v >> 5) & 0x1F;
                let b5 = v & 0x1F;
                // Expand 5-bit to 8-bit by replicating the top 3 bits.
                let r8 = ((r5 << 3) | (r5 >> 2)) as u8;
                let g8 = ((g5 << 3) | (g5 >> 2)) as u8;
                let b8 = ((b5 << 3) | (b5 >> 2)) as u8;
                let a8 = if a1 == 1 { 255 } else { 0 };
                rgba.push(r8);
                rgba.push(g8);
                rgba.push(b8);
                rgba.push(a8);
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

/// Decompresses the top mip level of `tex` and packs it into an `image::RgbaImage`.
pub fn to_rgba_image(tex: &YtdTexture) -> Result<image::RgbaImage> {
    let (width, height) = (tex.width as u32, tex.height as u32);
    let rgba = decompress_texture(tex)?;

    image::RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| anyhow::anyhow!("pixel buffer does not match texture dimensions"))
}

/// Resizes `img` so its longest edge does not exceed `max_edge`, preserving aspect
/// ratio. If the image already fits, it is returned unchanged.
pub fn fit_max_size(img: image::RgbaImage, max_edge: u32) -> image::RgbaImage {
    let (width, height) = img.dimensions();
    let longest = width.max(height);
    if longest <= max_edge || longest == 0 {
        return img;
    }

    let scale = max_edge as f64 / longest as f64;
    let new_width = ((width as f64 * scale).round() as u32).max(1);
    let new_height = ((height as f64 * scale).round() as u32).max(1);

    image::imageops::resize(&img, new_width, new_height, image::imageops::FilterType::Triangle)
}

/// Output image formats supported by [`encode_image`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    WebP,
}

impl std::str::FromStr for ImageFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "webp" => Ok(Self::WebP),
            other => bail!("Unknown image format '{}' (expected png, jpg, jpeg, or webp)", other),
        }
    }
}

impl ImageFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::WebP => "webp",
        }
    }
}

impl std::fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.extension())
    }
}

/// Encodes `img` in the given `format`. `quality` (0-100) is used for JPEG only;
/// WebP is always encoded losslessly, so `quality` is ignored for it, and PNG has
/// no quality setting either.
pub fn encode_image(img: &image::RgbaImage, format: ImageFormat, quality: u8) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let (width, height) = img.dimensions();

    match format {
        ImageFormat::Png => {
            let encoder = image::codecs::png::PngEncoder::new(&mut buf);
            image::ImageEncoder::write_image(
                encoder,
                img.as_raw(),
                width,
                height,
                image::ExtendedColorType::Rgba8,
            )?;
        }
        ImageFormat::Jpeg => {
            let rgb_img: image::RgbImage = image::DynamicImage::ImageRgba8(img.clone()).to_rgb8();
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
            image::ImageEncoder::write_image(
                encoder,
                rgb_img.as_raw(),
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )?;
        }
        ImageFormat::WebP => {
            let encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut buf);
            image::ImageEncoder::write_image(
                encoder,
                img.as_raw(),
                width,
                height,
                image::ExtendedColorType::Rgba8,
            )?;
        }
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

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

    /// Real textures store the whole mip chain in `pixel_data`. Uncompressed
    /// formats used to expand all of it, producing a buffer several times too
    /// long, which blew up as an assertion inside the PNG encoder.
    #[test]
    fn uncompressed_formats_use_only_the_top_mip_level() {
        // 4x4 + 2x2 + 1x1 single-byte levels.
        let mut data: Vec<u8> = vec![0x10; 16];
        data.extend_from_slice(&[0x20; 4]);
        data.push(0x30);

        let mut tex = texture(TextureFormat::A8, data.clone());
        tex.levels = 3;
        let image = to_rgba_image(&tex).expect("A8 with mips should decode");
        assert_eq!(image.dimensions(), (4, 4));
        assert_eq!(image.as_raw().len(), 4 * 4 * 4);
        assert!(image.pixels().all(|pixel| pixel.0 == [255, 255, 255, 0x10]));

        let mut tex = texture(TextureFormat::L8, data);
        tex.levels = 3;
        let image = to_rgba_image(&tex).expect("L8 with mips should decode");
        assert_eq!(image.as_raw().len(), 4 * 4 * 4);
        assert!(image.pixels().all(|pixel| pixel.0 == [0x10, 0x10, 0x10, 255]));
    }

    /// A buffer that cannot fill the top level is an error, not a panic.
    #[test]
    fn short_pixel_buffers_are_reported_as_errors() {
        let tex = texture(TextureFormat::A8, vec![0x10; 4]);
        assert!(to_rgba_image(&tex).is_err());
    }

    #[test]
    fn uncompressed_argb_is_reordered() {
        // One BGRA-ordered texel on disk: B=0x11, G=0x22, R=0x33, A=0x44.
        let rgba = decompress_texture(&texture_1x1(
            TextureFormat::A8R8G8B8,
            vec![0x11, 0x22, 0x33, 0x44],
        ))
        .unwrap();

        assert_eq!(rgba, vec![0x33, 0x22, 0x11, 0x44]);
    }

    /// The length check belongs to `decompress_texture`, so every caller sees
    /// truncated input as an error and not just `to_rgba_image`.
    #[test]
    fn decompress_texture_rejects_a_short_buffer() {
        let tex = texture(TextureFormat::A8, vec![0x10; 4]);
        assert!(decompress_texture(&tex).is_err(), "4 bytes cannot fill a 4x4 texture");
        assert!(to_rgba_image(&tex).is_err());
    }

    fn texture_1x1(format: TextureFormat, pixel_data: Vec<u8>) -> YtdTexture {
        YtdTexture {
            name: "test".into(),
            name_hash: 0,
            width: 1,
            height: 1,
            depth: 1,
            format,
            levels: 1,
            stride: 0,
            pixel_data,
        }
    }

    #[test]
    fn rgba_formats_decode() {
        // X8R8G8B8: BGRX bytes on disk -> RGBA with A=255.
        let rgba = decompress_texture(&texture_1x1(
            TextureFormat::X8R8G8B8,
            vec![0x11, 0x22, 0x33, 0xFF],
        ))
        .unwrap();
        assert_eq!(rgba, vec![0x33, 0x22, 0x11, 255]);

        // A8B8G8R8: already RGBA byte order, copied verbatim.
        let rgba = decompress_texture(&texture_1x1(
            TextureFormat::A8B8G8R8,
            vec![0x10, 0x20, 0x30, 0x40],
        ))
        .unwrap();
        assert_eq!(rgba, vec![0x10, 0x20, 0x30, 0x40]);

        // L8: grey -> R=G=B=L, A=255.
        let rgba = decompress_texture(&texture_1x1(TextureFormat::L8, vec![0x77])).unwrap();
        assert_eq!(rgba, vec![0x77, 0x77, 0x77, 255]);

        // A8: R=G=B=255, A=value.
        let rgba = decompress_texture(&texture_1x1(TextureFormat::A8, vec![0x99])).unwrap();
        assert_eq!(rgba, vec![255, 255, 255, 0x99]);

        // A1R5G5B5: 16-bit little-endian, 1 alpha bit, 5-bit channels expanded to 8-bit.
        // All bits set: A=1, R=G=B=0x1F -> RGBA (255,255,255,255).
        let rgba = decompress_texture(&texture_1x1(
            TextureFormat::A1R5G5B5,
            vec![0xFF, 0xFF],
        ))
        .unwrap();
        assert_eq!(rgba, vec![255, 255, 255, 255]);

        // Alpha bit clear -> A=0; everything else zero -> RGB all zero.
        let rgba = decompress_texture(&texture_1x1(
            TextureFormat::A1R5G5B5,
            vec![0x00, 0x00],
        ))
        .unwrap();
        assert_eq!(rgba, vec![0, 0, 0, 0]);

        // Red-only (r5=0x1F, g5=0, b5=0, alpha bit set): value = 0b1_11111_00000_00000 = 0xFC00.
        // Little-endian bytes: 0x00, 0xFC.
        let rgba = decompress_texture(&texture_1x1(
            TextureFormat::A1R5G5B5,
            vec![0x00, 0xFC],
        ))
        .unwrap();
        assert_eq!(rgba, vec![255, 0, 0, 255], "red-only A1R5G5B5 must not leak into blue");

        // Blue-only (r5=0, g5=0, b5=0x1F, alpha bit set): value = 0b1_00000_00000_11111 = 0x801F.
        // Little-endian bytes: 0x1F, 0x80.
        let rgba = decompress_texture(&texture_1x1(
            TextureFormat::A1R5G5B5,
            vec![0x1F, 0x80],
        ))
        .unwrap();
        assert_eq!(rgba, vec![0, 0, 255, 255], "blue-only A1R5G5B5 must not leak into red");
    }

    #[test]
    fn to_rgba_image_dimensions() {
        let tex = YtdTexture {
            name: "test".into(),
            name_hash: 0,
            width: 2,
            height: 3,
            depth: 1,
            format: TextureFormat::L8,
            levels: 1,
            stride: 0,
            pixel_data: vec![0u8; 2 * 3],
        };
        let img = to_rgba_image(&tex).unwrap();
        assert_eq!(img.dimensions(), (2, 3));
    }

    #[test]
    fn fit_max_size_keeps_aspect() {
        let img = image::RgbaImage::new(400, 200);
        let resized = fit_max_size(img, 100);
        assert_eq!(resized.dimensions(), (100, 50));

        let img = image::RgbaImage::new(50, 50);
        let resized = fit_max_size(img, 100);
        assert_eq!(resized.dimensions(), (50, 50));
    }

    #[test]
    fn encode_png_jpeg_webp_roundtrip() {
        let mut img = image::RgbaImage::new(4, 4);
        for (i, pixel) in img.pixels_mut().enumerate() {
            let v = (i * 16) as u8;
            *pixel = image::Rgba([v, 255 - v, v / 2, 255]);
        }

        for format in [ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::WebP] {
            let bytes = encode_image(&img, format, 90).unwrap();
            let decoded = image::load_from_memory(&bytes).unwrap();
            assert_eq!(decoded.dimensions(), (4, 4));

            if format == ImageFormat::Png {
                assert_eq!(decoded.to_rgba8(), img);
            }
        }
    }

    #[test]
    fn image_format_from_str() {
        assert_eq!("png".parse::<ImageFormat>().unwrap(), ImageFormat::Png);
        assert_eq!("PNG".parse::<ImageFormat>().unwrap(), ImageFormat::Png);
        assert_eq!("jpg".parse::<ImageFormat>().unwrap(), ImageFormat::Jpeg);
        assert_eq!("JPG".parse::<ImageFormat>().unwrap(), ImageFormat::Jpeg);
        assert_eq!("jpeg".parse::<ImageFormat>().unwrap(), ImageFormat::Jpeg);
        assert_eq!("webp".parse::<ImageFormat>().unwrap(), ImageFormat::WebP);
        assert_eq!("WEBP".parse::<ImageFormat>().unwrap(), ImageFormat::WebP);

        let err = "bmp".parse::<ImageFormat>().unwrap_err();
        assert!(err.to_string().contains("png"));
        assert!(err.to_string().contains("webp"));
    }
}
