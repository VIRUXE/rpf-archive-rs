//! Texture sheet / atlas support — composes a grid of labelled thumbnails
//! (a "contact sheet") from a set of source images, useful for previewing a
//! batch of extracted textures at a glance.

use image::{imageops, Rgba, RgbaImage};

use crate::font::{draw_text, text_width, GLYPH_H};

/// Options controlling the layout of a composed contact sheet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SheetOptions {
    /// Size, in pixels, of the square area each item's image is fit into.
    pub cell: u32,
    /// Fixed number of columns; when `None`, computed as `ceil(sqrt(n))`.
    pub columns: Option<u32>,
    /// Padding, in pixels, around each cell.
    pub padding: u32,
    /// Integer scale used when drawing each item's label.
    pub label_scale: u32,
    /// Background colour (RGBA) filling the sheet outside of cells.
    pub background: [u8; 4],
}

impl Default for SheetOptions {
    fn default() -> Self {
        Self {
            cell: 256,
            columns: None,
            padding: 8,
            label_scale: 2,
            background: [40, 40, 40, 255],
        }
    }
}

/// One entry in a contact sheet: an image plus the label drawn beneath it.
pub struct SheetItem<'a> {
    pub label: String,
    pub image: &'a RgbaImage,
}

/// Computes the `(columns, rows)` grid needed to lay out `n` items, honouring
/// a fixed `columns` override when given. `n == 0` still yields a `(1, 1)`
/// grid (a single, empty cell).
pub fn sheet_layout(n: usize, columns: Option<u32>) -> (u32, u32) {
    if n == 0 {
        return (1, 1);
    }
    let cols = columns
        .unwrap_or_else(|| (n as f64).sqrt().ceil() as u32)
        .max(1);
    let rows = ((n as u32) + cols - 1) / cols;
    (cols, rows.max(1))
}

/// Draws an 8px checkerboard (light/dark grey) into the rectangle
/// `[x0, x0+w) x [y0, y0+h)` of `img`, clipped to the image bounds. Used as a
/// backdrop so that alpha-transparent regions of composited textures remain
/// visible.
fn draw_checkerboard(img: &mut RgbaImage, x0: u32, y0: u32, w: u32, h: u32) {
    const CHECKER: u32 = 8;
    const LIGHT: [u8; 4] = [200, 200, 200, 255];
    const DARK: [u8; 4] = [160, 160, 160, 255];

    let (img_w, img_h) = img.dimensions();
    let x1 = (x0 + w).min(img_w);
    let y1 = (y0 + h).min(img_h);
    for py in y0..y1 {
        for px in x0..x1 {
            let cx = (px - x0) / CHECKER;
            let cy = (py - y0) / CHECKER;
            let color = if (cx + cy) % 2 == 0 { LIGHT } else { DARK };
            img.put_pixel(px, py, Rgba(color));
        }
    }
}

/// Resizes `src` to fit within a `cell x cell` square while preserving
/// aspect ratio (nearest-neighbour for upscaling, triangle filtering for
/// downscaling), returning the resized image.
fn fit_resize(src: &RgbaImage, cell: u32) -> RgbaImage {
    let (w, h) = src.dimensions();
    if w == 0 || h == 0 || cell == 0 {
        return RgbaImage::new(0, 0);
    }
    let scale = (cell as f64 / w as f64).min(cell as f64 / h as f64);
    let new_w = ((w as f64 * scale).round() as u32).max(1);
    let new_h = ((h as f64 * scale).round() as u32).max(1);
    let filter = if scale > 1.0 {
        imageops::FilterType::Nearest
    } else {
        imageops::FilterType::Triangle
    };
    imageops::resize(src, new_w, new_h, filter)
}

/// Truncates `label` (appending a trailing `~`) so that its rendered width at
/// `label_scale` does not exceed `max_width`. Returns the label unchanged if
/// it already fits.
fn truncate_label(label: &str, label_scale: u32, max_width: u32) -> String {
    if text_width(label, label_scale) <= max_width {
        return label.to_string();
    }
    let mut chars: Vec<char> = label.chars().collect();
    while !chars.is_empty() {
        chars.pop();
        let candidate: String = chars.iter().collect::<String>() + "~";
        if text_width(&candidate, label_scale) <= max_width {
            return candidate;
        }
    }
    "~".to_string()
}

/// Composes a labelled contact sheet from `items`, laying them out in a grid
/// according to `opts`.
pub fn compose_sheet(items: &[SheetItem<'_>], opts: &SheetOptions) -> RgbaImage {
    let (cols, rows) = sheet_layout(items.len(), opts.columns);
    let label_h = GLYPH_H * opts.label_scale + 2 * opts.padding;
    let pitch_x = opts.cell + 2 * opts.padding;
    let pitch_y = opts.cell + label_h + 2 * opts.padding;

    let width = cols * pitch_x;
    let height = rows * pitch_y;
    let mut sheet = RgbaImage::from_pixel(width.max(1), height.max(1), Rgba(opts.background));

    for (i, item) in items.iter().enumerate() {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;

        let cell_x0 = col * pitch_x + opts.padding;
        let cell_y0 = row * pitch_y + opts.padding;

        draw_checkerboard(&mut sheet, cell_x0, cell_y0, opts.cell, opts.cell);

        let resized = fit_resize(item.image, opts.cell);
        let (rw, rh) = resized.dimensions();
        let dx = (opts.cell.saturating_sub(rw)) / 2;
        let dy = (opts.cell.saturating_sub(rh)) / 2;
        imageops::overlay(
            &mut sheet,
            &resized,
            (cell_x0 + dx) as i64,
            (cell_y0 + dy) as i64,
        );

        let label = truncate_label(&item.label, opts.label_scale, opts.cell);
        let text_y = (cell_y0 + opts.cell + opts.padding) as i32;
        draw_text(
            &mut sheet,
            cell_x0 as i32,
            text_y,
            &label,
            opts.label_scale,
            [255, 255, 255, 255],
        );
    }

    sheet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_5_items_is_3x2() {
        assert_eq!(sheet_layout(5, None), (3, 2));
    }

    #[test]
    fn layout_columns_override() {
        assert_eq!(sheet_layout(5, Some(2)), (2, 3));
        assert_eq!(sheet_layout(0, None), (1, 1));
        assert_eq!(sheet_layout(1, None), (1, 1));
    }

    #[test]
    fn sheet_dimensions() {
        let img = RgbaImage::from_pixel(4, 4, Rgba([255, 255, 255, 255]));
        let items = vec![
            SheetItem { label: "a".into(), image: &img },
            SheetItem { label: "b".into(), image: &img },
            SheetItem { label: "c".into(), image: &img },
        ];
        let opts = SheetOptions { cell: 64, ..Default::default() };
        let out = compose_sheet(&items, &opts);

        let (cols, rows) = sheet_layout(3, None); // (2, 2)
        let label_h = GLYPH_H * opts.label_scale + 2 * opts.padding;
        let pitch_x = opts.cell + 2 * opts.padding;
        let pitch_y = opts.cell + label_h + 2 * opts.padding;
        assert_eq!(out.width(), cols * pitch_x);
        assert_eq!(out.height(), rows * pitch_y);
    }

    #[test]
    fn sheet_draws_image_pixels() {
        let red = RgbaImage::from_pixel(8, 8, Rgba([255, 0, 0, 255]));
        let items = vec![SheetItem { label: "red".into(), image: &red }];
        let opts = SheetOptions { cell: 32, ..Default::default() };
        let out = compose_sheet(&items, &opts);

        let cx = opts.padding + opts.cell / 2;
        let cy = opts.padding + opts.cell / 2;
        assert_eq!(*out.get_pixel(cx, cy), Rgba([255, 0, 0, 255]));
    }

    #[test]
    fn alpha_shows_checkerboard() {
        let transparent = RgbaImage::from_pixel(8, 8, Rgba([0, 0, 0, 0]));
        let items = vec![SheetItem { label: "t".into(), image: &transparent }];
        let opts = SheetOptions { cell: 32, ..Default::default() };
        let out = compose_sheet(&items, &opts);

        let x0 = opts.padding + opts.cell / 2 - 4;
        let y = opts.padding + opts.cell / 2;
        let p1 = *out.get_pixel(x0, y);
        let p2 = *out.get_pixel(x0 + 8, y);
        assert_ne!(p1, p2, "checkerboard squares 8px apart should differ");
    }
}
