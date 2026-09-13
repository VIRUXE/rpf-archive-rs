//! Texture lookup for the renderer: decoded RGBA images keyed by name hash.

use std::collections::HashMap;

use image::RgbaImage;

use crate::texture_utils::to_rgba_image;
use crate::writer::rage_joaat;
use crate::ytd::YtdTexture;

/// Decoded textures, in priority layers: a name is resolved against layer 0
/// first, then layer 1, and so on. That lets an embedded texture dictionary
/// shadow a shared one without merging the two.
///
/// Keys are `rage_joaat` of the lowercased texture name.
#[derive(Debug, Default, Clone)]
pub struct TextureSet {
    layers: Vec<HashMap<u32, RgbaImage>>,
}

impl TextureSet {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Decodes `textures` into a new, lowest-priority layer. Returns the names
    /// of the textures that could not be decoded (they are simply absent).
    pub fn push_layer(&mut self, textures: &[YtdTexture]) -> Vec<String> {
        let mut failed = Vec::new();
        let mut layer = HashMap::with_capacity(textures.len());

        for texture in textures {
            match to_rgba_image(texture) {
                Ok(image) => {
                    layer.insert(rage_joaat(&texture.name.to_lowercase()), image);
                }
                Err(_) => failed.push(texture.name.clone()),
            }
        }

        self.layers.push(layer);
        failed
    }

    /// The image bound to `name`, searching layers in order.
    pub fn get(&self, name: &str) -> Option<&RgbaImage> {
        let hash = rage_joaat(&name.to_lowercase());
        self.layers.iter().find_map(|layer| layer.get(&hash))
    }

    /// True when no layer holds a decoded texture.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The total number of decoded textures across every layer.
    pub fn len(&self) -> usize {
        self.layers.iter().map(HashMap::len).sum()
    }
}
