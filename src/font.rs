//! Font rendering support — a tiny built-in 5x7 bitmap ASCII font, useful for
//! stamping labels onto rendered textures without an external font dependency.

use image::RgbaImage;

/// Glyph width in pixels (before scaling).
pub const GLYPH_W: u32 = 5;
/// Glyph height in pixels (before scaling).
pub const GLYPH_H: u32 = 7;

/// The classic public-domain 5x7 ASCII font (as used by Adafruit-GFX and many
/// LCD/OLED libraries), covering `0x20..=0x7E` plus a box glyph at `0x7F` used
/// as a fallback for unknown/non-ASCII characters.
///
/// Each glyph is 5 column bytes; within a column byte, bit 0 is the top row
/// and bit 6 is the bottom row (bits 7 is unused).
pub static FONT_5X7: [[u8; 5]; 96] = [
    [0x00, 0x00, 0x00, 0x00, 0x00], // 0x20 ' '
    [0x00, 0x00, 0x5F, 0x00, 0x00], // 0x21 '!'
    [0x00, 0x07, 0x00, 0x07, 0x00], // 0x22 '"'
    [0x14, 0x7F, 0x14, 0x7F, 0x14], // 0x23 '#'
    [0x24, 0x2A, 0x7F, 0x2A, 0x12], // 0x24 '$'
    [0x23, 0x13, 0x08, 0x64, 0x62], // 0x25 '%'
    [0x36, 0x49, 0x56, 0x20, 0x50], // 0x26 '&'
    [0x00, 0x08, 0x07, 0x03, 0x00], // 0x27 '''
    [0x00, 0x1C, 0x22, 0x41, 0x00], // 0x28 '('
    [0x00, 0x41, 0x22, 0x1C, 0x00], // 0x29 ')'
    [0x2A, 0x1C, 0x7F, 0x1C, 0x2A], // 0x2A '*'
    [0x08, 0x08, 0x3E, 0x08, 0x08], // 0x2B '+'
    [0x00, 0x80, 0x70, 0x30, 0x00], // 0x2C ','
    [0x08, 0x08, 0x08, 0x08, 0x08], // 0x2D '-'
    [0x00, 0x00, 0x60, 0x60, 0x00], // 0x2E '.'
    [0x20, 0x10, 0x08, 0x04, 0x02], // 0x2F '/'
    [0x3E, 0x51, 0x49, 0x45, 0x3E], // 0x30 '0'
    [0x00, 0x42, 0x7F, 0x40, 0x00], // 0x31 '1'
    [0x72, 0x49, 0x49, 0x49, 0x46], // 0x32 '2'
    [0x21, 0x41, 0x49, 0x4D, 0x33], // 0x33 '3'
    [0x18, 0x14, 0x12, 0x7F, 0x10], // 0x34 '4'
    [0x27, 0x45, 0x45, 0x45, 0x39], // 0x35 '5'
    [0x3C, 0x4A, 0x49, 0x49, 0x31], // 0x36 '6'
    [0x41, 0x21, 0x11, 0x09, 0x07], // 0x37 '7'
    [0x36, 0x49, 0x49, 0x49, 0x36], // 0x38 '8'
    [0x46, 0x49, 0x49, 0x29, 0x1E], // 0x39 '9'
    [0x00, 0x00, 0x14, 0x00, 0x00], // 0x3A ':'
    [0x00, 0x40, 0x34, 0x00, 0x00], // 0x3B ';'
    [0x00, 0x08, 0x14, 0x22, 0x41], // 0x3C '<'
    [0x14, 0x14, 0x14, 0x14, 0x14], // 0x3D '='
    [0x41, 0x22, 0x14, 0x08, 0x00], // 0x3E '>'
    [0x02, 0x01, 0x59, 0x09, 0x06], // 0x3F '?'
    [0x3E, 0x41, 0x5D, 0x59, 0x4E], // 0x40 '@'
    [0x7C, 0x12, 0x11, 0x12, 0x7C], // 0x41 'A'
    [0x7F, 0x49, 0x49, 0x49, 0x36], // 0x42 'B'
    [0x3E, 0x41, 0x41, 0x41, 0x22], // 0x43 'C'
    [0x7F, 0x41, 0x41, 0x41, 0x3E], // 0x44 'D'
    [0x7F, 0x49, 0x49, 0x49, 0x41], // 0x45 'E'
    [0x7F, 0x09, 0x09, 0x09, 0x01], // 0x46 'F'
    [0x3E, 0x41, 0x49, 0x49, 0x7A], // 0x47 'G'
    [0x7F, 0x08, 0x08, 0x08, 0x7F], // 0x48 'H'
    [0x00, 0x41, 0x7F, 0x41, 0x00], // 0x49 'I'
    [0x20, 0x40, 0x41, 0x3F, 0x01], // 0x4A 'J'
    [0x7F, 0x08, 0x14, 0x22, 0x41], // 0x4B 'K'
    [0x7F, 0x40, 0x40, 0x40, 0x40], // 0x4C 'L'
    [0x7F, 0x02, 0x1C, 0x02, 0x7F], // 0x4D 'M'
    [0x7F, 0x04, 0x08, 0x10, 0x7F], // 0x4E 'N'
    [0x3E, 0x41, 0x41, 0x41, 0x3E], // 0x4F 'O'
    [0x7F, 0x09, 0x09, 0x09, 0x06], // 0x50 'P'
    [0x3E, 0x41, 0x51, 0x21, 0x5E], // 0x51 'Q'
    [0x7F, 0x09, 0x19, 0x29, 0x46], // 0x52 'R'
    [0x26, 0x49, 0x49, 0x49, 0x32], // 0x53 'S'
    [0x01, 0x01, 0x7F, 0x01, 0x01], // 0x54 'T'
    [0x3F, 0x40, 0x40, 0x40, 0x3F], // 0x55 'U'
    [0x1F, 0x20, 0x40, 0x20, 0x1F], // 0x56 'V'
    [0x3F, 0x40, 0x38, 0x40, 0x3F], // 0x57 'W'
    [0x63, 0x14, 0x08, 0x14, 0x63], // 0x58 'X'
    [0x03, 0x04, 0x78, 0x04, 0x03], // 0x59 'Y'
    [0x61, 0x51, 0x49, 0x45, 0x43], // 0x5A 'Z'
    [0x00, 0x00, 0x7F, 0x41, 0x41], // 0x5B '['
    [0x02, 0x04, 0x08, 0x10, 0x20], // 0x5C '\'
    [0x41, 0x41, 0x7F, 0x00, 0x00], // 0x5D ']'
    [0x04, 0x02, 0x01, 0x02, 0x04], // 0x5E '^'
    [0x40, 0x40, 0x40, 0x40, 0x40], // 0x5F '_'
    [0x00, 0x01, 0x02, 0x04, 0x00], // 0x60 '`'
    [0x20, 0x54, 0x54, 0x54, 0x78], // 0x61 'a'
    [0x7F, 0x48, 0x44, 0x44, 0x38], // 0x62 'b'
    [0x38, 0x44, 0x44, 0x44, 0x20], // 0x63 'c'
    [0x38, 0x44, 0x44, 0x48, 0x7F], // 0x64 'd'
    [0x38, 0x54, 0x54, 0x54, 0x18], // 0x65 'e'
    [0x08, 0x7E, 0x09, 0x01, 0x02], // 0x66 'f'
    [0x0C, 0x52, 0x52, 0x52, 0x3E], // 0x67 'g'
    [0x7F, 0x08, 0x04, 0x04, 0x78], // 0x68 'h'
    [0x00, 0x44, 0x7D, 0x40, 0x00], // 0x69 'i'
    [0x20, 0x40, 0x44, 0x3D, 0x00], // 0x6A 'j'
    [0x7F, 0x10, 0x28, 0x44, 0x00], // 0x6B 'k'
    [0x00, 0x41, 0x7F, 0x40, 0x00], // 0x6C 'l'
    [0x7C, 0x04, 0x18, 0x04, 0x78], // 0x6D 'm'
    [0x7C, 0x08, 0x04, 0x04, 0x78], // 0x6E 'n'
    [0x38, 0x44, 0x44, 0x44, 0x38], // 0x6F 'o'
    [0x7C, 0x14, 0x14, 0x14, 0x08], // 0x70 'p'
    [0x08, 0x14, 0x14, 0x18, 0x7C], // 0x71 'q'
    [0x7C, 0x08, 0x04, 0x04, 0x08], // 0x72 'r'
    [0x48, 0x54, 0x54, 0x54, 0x20], // 0x73 's'
    [0x04, 0x3F, 0x44, 0x40, 0x20], // 0x74 't'
    [0x3C, 0x40, 0x40, 0x20, 0x7C], // 0x75 'u'
    [0x1C, 0x20, 0x40, 0x20, 0x1C], // 0x76 'v'
    [0x3C, 0x40, 0x30, 0x40, 0x3C], // 0x77 'w'
    [0x44, 0x28, 0x10, 0x28, 0x44], // 0x78 'x'
    [0x0C, 0x50, 0x50, 0x50, 0x3C], // 0x79 'y'
    [0x44, 0x64, 0x54, 0x4C, 0x44], // 0x7A 'z'
    [0x00, 0x08, 0x36, 0x41, 0x00], // 0x7B '{'
    [0x00, 0x00, 0x7F, 0x00, 0x00], // 0x7C '|'
    [0x00, 0x41, 0x36, 0x08, 0x00], // 0x7D '}'
    [0x08, 0x04, 0x08, 0x10, 0x08], // 0x7E '~'
    [0x7F, 0x41, 0x41, 0x41, 0x7F], // 0x7F fallback box (unknown / non-ASCII)
];

/// Index of the fallback box glyph, drawn for any character outside the
/// printable ASCII range covered by [`FONT_5X7`].
const FALLBACK_INDEX: usize = 0x7F - 0x20;

fn glyph_for(c: char) -> &'static [u8; 5] {
    let code = c as u32;
    if (0x20..=0x7E).contains(&code) {
        &FONT_5X7[(code - 0x20) as usize]
    } else {
        &FONT_5X7[FALLBACK_INDEX]
    }
}

/// Width, in pixels, that [`draw_text`] would occupy for `text` at the given
/// integer `scale`. Each glyph occupies `GLYPH_W` columns plus a 1-column
/// gap, except the gap trailing the final character is not counted. Empty
/// text has zero width.
pub fn text_width(text: &str, scale: u32) -> u32 {
    let count = text.chars().count() as u32;
    if count == 0 {
        return 0;
    }
    count * (GLYPH_W + 1) * scale - scale
}

/// Draws `text` onto `img` with its top-left corner at `(x, y)`, using
/// `scale`x nearest-neighbour magnification and a single-column gap between
/// glyphs. Characters outside the printable ASCII range are drawn as a
/// fallback box glyph. Drawing is clipped to the bounds of `img`; pixels that
/// would land outside the image (including at negative coordinates) are
/// silently skipped.
pub fn draw_text(img: &mut RgbaImage, x: i32, y: i32, text: &str, scale: u32, color: [u8; 4]) {
    if scale == 0 {
        return;
    }
    let (img_w, img_h) = (img.width() as i32, img.height() as i32);
    let advance = ((GLYPH_W + 1) * scale) as i32;

    for (i, c) in text.chars().enumerate() {
        let glyph = glyph_for(c);
        let base_x = x + i as i32 * advance;

        for col in 0..GLYPH_W {
            let column_bits = glyph[col as usize];
            for row in 0..GLYPH_H {
                if column_bits & (1 << row) == 0 {
                    continue;
                }
                let px0 = base_x + (col * scale) as i32;
                let py0 = y + (row * scale) as i32;
                for sy in 0..scale as i32 {
                    let py = py0 + sy;
                    if py < 0 || py >= img_h {
                        continue;
                    }
                    for sx in 0..scale as i32 {
                        let px = px0 + sx;
                        if px < 0 || px >= img_w {
                            continue;
                        }
                        img.put_pixel(px as u32, py as u32, image::Rgba(color));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_width_counts_gap() {
        assert_eq!(text_width("", 1), 0);
        assert_eq!(text_width("A", 1), GLYPH_W);
        assert_eq!(text_width("AB", 1), 2 * GLYPH_W + 1);
        assert_eq!(text_width("AB", 2), 2 * (2 * GLYPH_W + 1));
    }

    #[test]
    fn draw_text_sets_pixels_for_a() {
        let mut img = RgbaImage::from_pixel(16, 8, image::Rgba([0, 0, 0, 0]));
        let white = [255, 255, 255, 255];
        draw_text(&mut img, 0, 0, "A", 1, white);

        // 'A' glyph: apex at top-center, legs at bottom corners.
        assert_eq!(*img.get_pixel(2, 0), image::Rgba(white)); // apex
        assert_eq!(*img.get_pixel(0, 0), image::Rgba([0, 0, 0, 0])); // top-left empty
        assert_eq!(*img.get_pixel(0, 4), image::Rgba(white)); // left leg / crossbar row
        assert_eq!(*img.get_pixel(4, 4), image::Rgba(white)); // right leg / crossbar row
        assert_eq!(*img.get_pixel(0, 6), image::Rgba(white)); // bottom-left leg
        assert_eq!(*img.get_pixel(4, 6), image::Rgba(white)); // bottom-right leg

        // The gap column after the glyph (column index GLYPH_W == 5) stays empty.
        for row in 0..GLYPH_H {
            assert_eq!(*img.get_pixel(GLYPH_W, row), image::Rgba([0, 0, 0, 0]));
        }
    }

    #[test]
    fn draw_text_sets_known_pixels_for_more_glyphs() {
        let white = [255, 255, 255, 255];
        let empty = image::Rgba([0, 0, 0, 0]);

        // ' ' (space): entirely blank.
        let mut img = RgbaImage::from_pixel(GLYPH_W, GLYPH_H, image::Rgba([0, 0, 0, 0]));
        draw_text(&mut img, 0, 0, " ", 1, white);
        for row in 0..GLYPH_H {
            for col in 0..GLYPH_W {
                assert_eq!(*img.get_pixel(col, row), empty);
            }
        }

        // '-' (0x2D, columns all 0x08 -> bit 3 only): a single solid middle row.
        let mut img = RgbaImage::from_pixel(GLYPH_W, GLYPH_H, image::Rgba([0, 0, 0, 0]));
        draw_text(&mut img, 0, 0, "-", 1, white);
        for col in 0..GLYPH_W {
            assert_eq!(*img.get_pixel(col, 3), image::Rgba(white), "middle row set at col {col}");
            assert_eq!(*img.get_pixel(col, 0), empty, "top row empty at col {col}");
            assert_eq!(*img.get_pixel(col, 6), empty, "bottom row empty at col {col}");
        }

        // '_' (0x5F, columns all 0x40 -> bit 6 only): a single solid bottom row.
        let mut img = RgbaImage::from_pixel(GLYPH_W, GLYPH_H, image::Rgba([0, 0, 0, 0]));
        draw_text(&mut img, 0, 0, "_", 1, white);
        for col in 0..GLYPH_W {
            assert_eq!(*img.get_pixel(col, 6), image::Rgba(white), "bottom row set at col {col}");
            assert_eq!(*img.get_pixel(col, 0), empty, "top row empty at col {col}");
        }

        // '0' ([0x3E, 0x51, 0x49, 0x45, 0x3E]): left stroke spans rows 1..=5,
        // its top-left corner (row 0) is empty.
        let mut img = RgbaImage::from_pixel(GLYPH_W, GLYPH_H, image::Rgba([0, 0, 0, 0]));
        draw_text(&mut img, 0, 0, "0", 1, white);
        assert_eq!(*img.get_pixel(0, 0), empty); // top-left corner empty
        assert_eq!(*img.get_pixel(0, 3), image::Rgba(white)); // left stroke, middle row
        assert_eq!(*img.get_pixel(1, 0), image::Rgba(white)); // top edge starts at col 1

        // 'x' ([0x44, 0x28, 0x10, 0x28, 0x44]): the two diagonals cross at
        // the center pixel; the corners are empty.
        let mut img = RgbaImage::from_pixel(GLYPH_W, GLYPH_H, image::Rgba([0, 0, 0, 0]));
        draw_text(&mut img, 0, 0, "x", 1, white);
        assert_eq!(*img.get_pixel(2, 4), image::Rgba(white)); // crossing point
        assert_eq!(*img.get_pixel(0, 0), empty); // top-left corner empty
        assert_eq!(*img.get_pixel(4, 0), empty); // top-right corner empty
    }

    #[test]
    fn draw_text_clips_at_edges() {
        let mut img = RgbaImage::from_pixel(4, 4, image::Rgba([0, 0, 0, 0]));
        // Should not panic even though most of the glyph falls outside bounds.
        draw_text(&mut img, -3, -3, "Hello, world!", 3, [255, 0, 0, 255]);
        draw_text(&mut img, 100, 100, "off screen", 1, [0, 255, 0, 255]);
        draw_text(&mut img, 2, 2, "x", 0, [0, 0, 255, 255]); // scale 0 is a no-op
    }
}
