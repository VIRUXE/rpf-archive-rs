//! Software renderer — a small CPU rasterizer that turns a parsed
//! [`Drawable`] into a preview image, with no GPU and no filesystem access so
//! it also runs under wasm.

mod camera;
mod mesh;
mod raster;
mod textures;

use anyhow::Result;
use image::RgbaImage;

use crate::ydd::{Drawable, LodLevel};
use raster::Framebuffer;

pub use textures::TextureSet;

/// One of the fixed camera angles a drawable can be previewed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Front,
    Back,
    Left,
    Right,
    Top,
    Iso,
}

impl View {
    pub const ALL: [View; 6] =
        [View::Front, View::Back, View::Left, View::Right, View::Top, View::Iso];

    pub fn label(self) -> &'static str {
        match self {
            View::Front => "front",
            View::Back => "back",
            View::Left => "left",
            View::Right => "right",
            View::Top => "top",
            View::Iso => "iso",
        }
    }
}

impl std::str::FromStr for View {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "front" => Ok(View::Front),
            "back" => Ok(View::Back),
            "left" => Ok(View::Left),
            "right" => Ok(View::Right),
            "top" => Ok(View::Top),
            "iso" => Ok(View::Iso),
            other => anyhow::bail!("unknown view '{other}'"),
        }
    }
}

impl std::fmt::Display for View {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.label())
    }
}

/// Everything the renderer needs beyond the model itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderOptions {
    pub width: u32,
    pub height: u32,
    pub view: View,
    pub background: [u8; 4],
    pub lod: LodLevel,
    pub backface_cull: bool,
    pub vertex_colors: bool,
    pub lighting: bool,
    pub fov_deg: f32,
    pub margin: f32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            view: View::Iso,
            background: [230, 230, 230, 255],
            lod: LodLevel::High,
            backface_cull: false,
            vertex_colors: false,
            lighting: true,
            fov_deg: 40.0,
            margin: 1.1,
        }
    }
}

/// What the renderer found while drawing — useful for diagnostics and for
/// telling the caller which textures it should have supplied.
#[derive(Debug, Default, Clone)]
pub struct RenderReport {
    /// Diffuse texture names the drawable referenced but the texture set did
    /// not hold, sorted and deduplicated.
    pub missing_textures: Vec<String>,
    pub triangles: usize,
    pub geometries: usize,
    /// Geometries drawn in flat grey — whether the name was missing or the
    /// shader had no diffuse parameter at all.
    pub untextured_geometries: usize,
    /// True when the drawable's stored bounds were degenerate and had to be
    /// rebuilt from the vertices.
    pub bounds_computed: bool,
    pub lod: Option<LodLevel>,
}

/// Renders `d` from `o.view`.
///
/// A drawable with no geometry is not an error: the result is a
/// background-only image and a report with zero triangles.
pub fn render_drawable(
    d: &Drawable,
    tex: &TextureSet,
    o: &RenderOptions,
) -> Result<(RgbaImage, RenderReport)> {
    let mut rendered = render_views(d, tex, o, &[o.view])?;
    let (_, image, report) = rendered.remove(0);
    Ok((image, report))
}

/// Renders `d` once per entry in `views`, preparing the mesh a single time.
pub fn render_views(
    d: &Drawable,
    tex: &TextureSet,
    o: &RenderOptions,
    views: &[View],
) -> Result<Vec<(View, RgbaImage, RenderReport)>> {
    let width = o.width.max(1);
    let height = o.height.max(1);

    // The requested LOD when it has models, otherwise whatever the drawable
    // actually carries.
    let lod = d
        .lod(o.lod)
        .filter(|lod| !lod.models.is_empty())
        .or_else(|| d.best_lod());

    let mut report = RenderReport::default();
    let mut geometries = Vec::new();
    let mut bounds = None;

    if let Some(lod) = lod {
        report.lod = Some(lod.level);
        let (computed_bounds, was_computed) = d.bounds_or_computed(lod);
        report.bounds_computed = was_computed;
        geometries = mesh::prepare(d, lod, tex, &mut report);
        bounds = Some(computed_bounds);
    }

    report.missing_textures.sort();
    report.missing_textures.dedup();

    let aspect = width as f32 / height as f32;
    let mut out = Vec::with_capacity(views.len());

    for &view in views {
        let mut framebuffer = Framebuffer::new(width, height, o.background);

        if let Some(bounds) = &bounds {
            if !geometries.is_empty() {
                let (view_proj, _eye, light) =
                    camera::camera_for(bounds, view, aspect, o.fov_deg, o.margin);

                // Solid geometry first, in model order, so the depth buffer
                // is complete before anything is blended over it.
                for geometry in geometries.iter().filter(|g| !g.blend.is_translucent()) {
                    raster::draw_geometry(&mut framebuffer, geometry, &view_proj, light, o);
                }

                // Then every translucent triangle, farthest first, so each
                // one composites over what lies behind it.
                let mut translucent: Vec<(f32, usize, &[u32])> = Vec::new();
                for (index, geometry) in geometries.iter().enumerate() {
                    if !geometry.blend.is_translucent() {
                        continue;
                    }
                    for triangle in geometry.indices.chunks_exact(3) {
                        if let Some(depth) =
                            raster::triangle_depth(&geometry.verts, triangle, &view_proj)
                        {
                            translucent.push((depth, index, triangle));
                        }
                    }
                }
                translucent.sort_by(|a, b| b.0.total_cmp(&a.0));
                for (_, index, triangle) in translucent {
                    raster::draw_triangle(
                        &mut framebuffer,
                        &geometries[index],
                        triangle,
                        &view_proj,
                        light,
                        o,
                    );
                }
            }
        }

        out.push((view, framebuffer.color, report.clone()));
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Vec2, Vec3};
    use crate::writer::rage_joaat;
    use crate::ydd::{
        Drawable, DrawableBounds, DrawableGeometry, DrawableLod, DrawableModel, IndexBuffer,
        LodLevel, ShaderFx, ShaderGroup, ShaderParameter, ShaderParameterValue, VertexBuffer,
        VertexBufferLayout, VertexComponent, VertexComponentType, VertexDeclaration,
        VertexSemantic, DIFFUSE_SAMPLER,
    };
    use std::str::FromStr;

    const STRIDE: u16 = 36;

    /// Position/Normal/TexCoord0/Colour0 — the layout the fixtures below write.
    fn declaration() -> VertexDeclaration {
        let component = |semantic, ty: VertexComponentType, offset| VertexComponent {
            semantic,
            semantic_index: 0,
            component_type: ty,
            offset,
            size: ty.size_in_bytes(),
            component_count: ty.component_count(),
        };
        VertexDeclaration {
            flags: 0,
            stride: STRIDE,
            unknown_6h: 0,
            count: 4,
            types: 0,
            components: vec![
                component(VertexSemantic::Position, VertexComponentType::Float3, 0),
                component(VertexSemantic::Normal, VertexComponentType::Float3, 12),
                component(VertexSemantic::TexCoord0, VertexComponentType::Float2, 24),
                component(VertexSemantic::Colour0, VertexComponentType::Colour, 32),
            ],
        }
    }

    type FixtureVertex = (Vec3, Vec3, Vec2, [u8; 4]);

    fn vertex_buffer(vertices: &[FixtureVertex], with_declaration: bool) -> VertexBuffer {
        let mut data = Vec::with_capacity(vertices.len() * STRIDE as usize);
        for (position, normal, uv, colour) in vertices {
            for value in [position.x, position.y, position.z, normal.x, normal.y, normal.z] {
                data.extend_from_slice(&value.to_le_bytes());
            }
            data.extend_from_slice(&uv.x.to_le_bytes());
            data.extend_from_slice(&uv.y.to_le_bytes());
            data.extend_from_slice(colour);
        }

        VertexBuffer {
            vertex_stride: STRIDE,
            vertex_count: vertices.len() as u32,
            data_pointer: 0,
            info_pointer: 0,
            declaration: with_declaration.then(declaration),
            data,
            layout: VertexBufferLayout::Legacy,
        }
    }

    fn geometry(vertices: &[FixtureVertex], indices: Vec<u32>, shader_id: u16) -> DrawableGeometry {
        DrawableGeometry {
            shader_id,
            indices_count: indices.len() as u32,
            triangles_count: (indices.len() / 3) as u32,
            vertices_count: vertices.len() as u16,
            vertex_stride: STRIDE,
            vertex_buffer: Some(vertex_buffer(vertices, true)),
            index_buffer: Some(IndexBuffer {
                indices_count: indices.len() as u32,
                indices_pointer: 0,
                indices,
            }),
        }
    }

    /// A shader in `render_bucket` whose diffuse sampler points at
    /// `texture_name` (or has no diffuse parameter at all when `None`).
    fn shader(texture_name: Option<&str>, render_bucket: u8) -> ShaderFx {
        let parameters = match texture_name {
            Some(name) => vec![ShaderParameter {
                name_hash: DIFFUSE_SAMPLER,
                data_type: 0,
                data_pointer: 0,
                value: ShaderParameterValue::Texture {
                    name: name.to_string(),
                    name_hash: rage_joaat(&name.to_lowercase()),
                },
            }],
            None => Vec::new(),
        };

        ShaderFx {
            name_hash: rage_joaat("default"),
            file_name_hash: 0,
            render_bucket,
            render_bucket_mask: (1 << render_bucket) | 0xFF00,
            parameter_count: parameters.len() as u8,
            texture_parameter_count: parameters.len() as u8,
            parameters,
        }
    }

    fn shader_group(texture_name: Option<&str>) -> ShaderGroup {
        ShaderGroup { textures: Vec::new(), shaders: vec![shader(texture_name, 0)] }
    }

    fn solid_texture(name: &str, rgba: [u8; 4]) -> crate::ytd::YtdTexture {
        crate::ytd::YtdTexture {
            name: name.to_string(),
            name_hash: rage_joaat(&name.to_lowercase()),
            width: 1,
            height: 1,
            depth: 1,
            format: crate::ytd::TextureFormat::A8B8G8R8,
            levels: 1,
            stride: 4,
            pixel_data: rgba.to_vec(),
        }
    }

    /// A quad in the XZ plane at `y`. The Front camera sits on +Y looking
    /// back at the origin, so larger `y` is nearer to it.
    fn quad_at_y(y: f32) -> Vec<FixtureVertex> {
        quad_vertices()
            .into_iter()
            .map(|(mut position, normal, uv, colour)| {
                position.y = y;
                (position, normal, uv, colour)
            })
            .collect()
    }

    /// Degenerate bounds, so `bounds_or_computed` rebuilds them from vertices.
    fn empty_bounds() -> DrawableBounds {
        DrawableBounds {
            center: Vec3::ZERO,
            sphere_radius: 0.0,
            box_min: Vec3::ZERO,
            box_max: Vec3::ZERO,
        }
    }

    fn drawable(geometries: Vec<DrawableGeometry>, shaders: ShaderGroup) -> Drawable {
        Drawable {
            name: "fixture".to_string(),
            name_hash: rage_joaat("fixture"),
            bounds: empty_bounds(),
            lod_distances: [0.0; 4],
            render_masks: [0; 4],
            shader_group: Some(shaders),
            lods: vec![DrawableLod {
                level: LodLevel::High,
                models: vec![DrawableModel {
                    skeleton_binding: 0,
                    render_mask_flags: 0,
                    shader_mapping: vec![0],
                    geometries,
                }],
            }],
        }
    }

    /// A quad in the XZ plane (facing -Y, toward the Front camera).
    fn quad_vertices() -> Vec<FixtureVertex> {
        let n = Vec3::new(0.0, -1.0, 0.0);
        vec![
            (Vec3::new(-1.0, 0.0, -1.0), n, Vec2::new(0.0, 1.0), [255, 255, 255, 255]),
            (Vec3::new(1.0, 0.0, -1.0), n, Vec2::new(1.0, 1.0), [255, 255, 255, 255]),
            (Vec3::new(1.0, 0.0, 1.0), n, Vec2::new(1.0, 0.0), [255, 255, 255, 255]),
            (Vec3::new(-1.0, 0.0, 1.0), n, Vec2::new(0.0, 0.0), [255, 255, 255, 255]),
        ]
    }

    fn quad_indices() -> Vec<u32> {
        vec![0, 1, 2, 0, 2, 3]
    }

    /// An axis-aligned box centred on the origin, with per-face outward
    /// normals. `half_extents` of 0.5 on every axis gives the unit cube.
    fn box_drawable(half_extents: Vec3) -> Drawable {
        let faces: [(Vec3, Vec3, Vec3); 6] = [
            // (normal, u axis, v axis)
            (Vec3::new(0.0, -1.0, 0.0), Vec3::X, Vec3::Z),
            (Vec3::new(0.0, 1.0, 0.0), -Vec3::X, Vec3::Z),
            (Vec3::new(-1.0, 0.0, 0.0), -Vec3::Y, Vec3::Z),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::Y, Vec3::Z),
            (Vec3::new(0.0, 0.0, -1.0), Vec3::X, Vec3::Y),
            (Vec3::new(0.0, 0.0, 1.0), Vec3::X, -Vec3::Y),
        ];

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (normal, u_axis, v_axis) in faces {
            let base = vertices.len() as u32;
            let centre = normal * 0.5;
            let corners = [
                centre - u_axis * 0.5 - v_axis * 0.5,
                centre + u_axis * 0.5 - v_axis * 0.5,
                centre + u_axis * 0.5 + v_axis * 0.5,
                centre - u_axis * 0.5 + v_axis * 0.5,
            ];
            let uvs = [
                Vec2::new(0.0, 1.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(0.0, 0.0),
            ];
            for (corner, uv) in corners.iter().zip(uvs) {
                let scaled = Vec3::new(
                    corner.x * half_extents.x * 2.0,
                    corner.y * half_extents.y * 2.0,
                    corner.z * half_extents.z * 2.0,
                );
                vertices.push((scaled, normal, uv, [200, 180, 160, 255]));
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        let geometries = vec![geometry(&vertices, indices, 0)];
        drawable(geometries, shader_group(None))
    }

    fn options(width: u32, height: u32) -> RenderOptions {
        RenderOptions { width, height, ..RenderOptions::default() }
    }

    fn background_count(image: &image::RgbaImage, background: [u8; 4]) -> usize {
        image.pixels().filter(|p| p.0 == background).count()
    }

    /// Width / height of the bounding box of the non-background pixels.
    fn silhouette_ratio(image: &image::RgbaImage, background: [u8; 4]) -> f32 {
        let mut min = (u32::MAX, u32::MAX);
        let mut max = (0u32, 0u32);
        for (x, y, pixel) in image.enumerate_pixels() {
            if pixel.0 == background {
                continue;
            }
            min = (min.0.min(x), min.1.min(y));
            max = (max.0.max(x), max.1.max(y));
        }
        assert!(min.0 <= max.0, "nothing was drawn");
        (max.0 - min.0 + 1) as f32 / (max.1 - min.1 + 1) as f32
    }

    // ─── Tests ──────────────────────────────────────────────────────────────

    #[test]
    fn view_from_str() {
        assert_eq!(View::from_str("front").unwrap(), View::Front);
        assert_eq!(View::from_str("BACK").unwrap(), View::Back);
        assert_eq!(View::from_str("Left").unwrap(), View::Left);
        assert_eq!(View::from_str("right").unwrap(), View::Right);
        assert_eq!(View::from_str("TOP").unwrap(), View::Top);
        assert_eq!(View::from_str("iso").unwrap(), View::Iso);
        assert!(View::from_str("sideways").is_err());

        for view in View::ALL {
            assert_eq!(View::from_str(view.label()).unwrap(), view);
            assert_eq!(view.to_string(), view.label());
        }
    }

    /// Render bucket 1 (alpha) blends, bucket 3 (cutout) discards below half
    /// alpha, and bucket 0 (opaque) ignores alpha entirely — the same texture
    /// with alpha 64 gives three different results.
    #[test]
    fn render_bucket_selects_how_alpha_is_applied() {
        let vertices = quad_vertices();
        let mut textures = TextureSet::new();
        textures.push_layer(&[solid_texture("faint_red", [255, 0, 0, 64])]);
        let background = [0, 0, 255, 255];
        let options = RenderOptions { view: View::Front, background, ..options(64, 64) };

        let render = |bucket: u8| {
            let drawable = drawable(
                vec![geometry(&vertices, quad_indices(), 0)],
                ShaderGroup { textures: Vec::new(), shaders: vec![shader(Some("faint_red"), bucket)] },
            );
            let no_light = RenderOptions { lighting: false, ..options };
            render_drawable(&drawable, &textures, &no_light).unwrap().0.get_pixel(32, 32).0
        };

        assert_eq!(render(0), [255, 0, 0, 255], "opaque ignores alpha");
        assert_eq!(render(3), background, "cutout discards alpha < 128");
        let blended = render(1);
        assert!((blended[0] as i32 - 64).abs() <= 2, "alpha blends red in: {blended:?}");
        assert!((blended[2] as i32 - 191).abs() <= 2, "alpha keeps most blue: {blended:?}");
    }

    /// Translucent geometry is drawn after the solid geometry it sits in
    /// front of, whatever order the model lists them in.
    #[test]
    fn translucent_geometry_is_drawn_after_opaque() {
        let near = quad_at_y(0.5);
        let far = quad_at_y(-0.5);
        let mut textures = TextureSet::new();
        textures.push_layer(&[
            solid_texture("half_red", [255, 0, 0, 128]),
            solid_texture("green", [0, 255, 0, 255]),
        ]);

        // The translucent quad is listed first and is nearer.
        let drawable = drawable(
            vec![geometry(&near, quad_indices(), 0), geometry(&far, quad_indices(), 1)],
            ShaderGroup {
                textures: Vec::new(),
                shaders: vec![shader(Some("half_red"), 1), shader(Some("green"), 0)],
            },
        );
        let options = RenderOptions { view: View::Front, lighting: false, ..options(64, 64) };
        let (image, _) = render_drawable(&drawable, &textures, &options).unwrap();

        let pixel = image.get_pixel(32, 32).0;
        assert!((pixel[0] as i32 - 128).abs() <= 2, "red over green: {pixel:?}");
        assert!((pixel[1] as i32 - 127).abs() <= 2, "green shows through: {pixel:?}");
    }

    /// Two translucent surfaces are composited back to front.
    #[test]
    fn translucent_geometry_is_sorted_back_to_front() {
        let near = quad_at_y(0.5);
        let far = quad_at_y(-0.5);
        let mut textures = TextureSet::new();
        textures.push_layer(&[
            solid_texture("half_red", [255, 0, 0, 128]),
            solid_texture("half_green", [0, 255, 0, 128]),
        ]);

        // Near listed first: drawn in list order it would be hidden by the
        // far one's blend; drawn back to front it ends up on top.
        let drawable = drawable(
            vec![geometry(&near, quad_indices(), 0), geometry(&far, quad_indices(), 1)],
            ShaderGroup {
                textures: Vec::new(),
                shaders: vec![shader(Some("half_red"), 1), shader(Some("half_green"), 1)],
            },
        );
        let background = [0, 0, 0, 255];
        let options =
            RenderOptions { view: View::Front, lighting: false, background, ..options(64, 64) };
        let (image, _) = render_drawable(&drawable, &textures, &options).unwrap();

        // far green over black = (0,128,0); near red over that = (128,64,0).
        let pixel = image.get_pixel(32, 32).0;
        assert!((pixel[0] as i32 - 128).abs() <= 2, "near red on top: {pixel:?}");
        assert!((pixel[1] as i32 - 64).abs() <= 2, "far green underneath: {pixel:?}");
    }

    #[test]
    fn missing_texture_reported_and_grey() {
        let vertices = quad_vertices();
        let drawable = drawable(
            vec![geometry(&vertices, quad_indices(), 0)],
            shader_group(Some("nope")),
        );

        let options = RenderOptions { view: View::Front, ..options(64, 64) };
        let (image, report) = render_drawable(&drawable, &TextureSet::new(), &options).unwrap();

        assert_eq!(report.missing_textures, vec!["nope".to_string()]);
        assert_eq!(report.triangles, 2);
        assert_eq!(report.geometries, 1);
        assert_eq!(report.untextured_geometries, 1);
        assert!(report.bounds_computed);
        assert_eq!(report.lod, Some(LodLevel::High));

        let centre = image.get_pixel(32, 32).0;
        assert_ne!(centre, options.background);
        assert_eq!(centre[0], centre[1]);
        assert_eq!(centre[1], centre[2]);
        assert_eq!(centre[3], 255);
    }

    /// A 1 x 2 x 3 box looks different from every side, so a swapped view
    /// direction or up vector shows up as the wrong silhouette shape.
    #[test]
    fn views_show_the_expected_silhouette_of_an_asymmetric_box() {
        let drawable = box_drawable(Vec3::new(0.5, 1.0, 1.5));
        let options = options(256, 256);
        let rendered =
            render_views(&drawable, &TextureSet::new(), &options, &View::ALL).unwrap();

        let ratio = |wanted: View| {
            let (_, image, _) = rendered
                .iter()
                .find(|(view, _, _)| *view == wanted)
                .expect("view rendered");
            silhouette_ratio(image, options.background)
        };

        // Front/Back look along Y: X (1) wide by Z (3) tall.
        // Left/Right look along X: Y (2) wide by Z (3) tall.
        // Top looks down Z with +Y up: X (1) wide by Y (2) tall.
        for (view, expected) in [
            (View::Front, 1.0 / 3.0),
            (View::Back, 1.0 / 3.0),
            (View::Left, 2.0 / 3.0),
            (View::Right, 2.0 / 3.0),
            (View::Top, 1.0 / 2.0),
        ] {
            let measured = ratio(view);
            assert!(
                (measured - expected).abs() < 0.05,
                "{view}: silhouette ratio {measured} != {expected}"
            );
        }

        assert!(ratio(View::Front) < ratio(View::Top));
        assert!(ratio(View::Top) < ratio(View::Left));
    }

    #[test]
    fn each_view_renders_nonempty() {
        let cube = box_drawable(Vec3::new(0.5, 0.5, 0.5));
        let options = options(64, 64);
        let rendered =
            render_views(&cube, &TextureSet::new(), &options, &View::ALL).unwrap();

        assert_eq!(rendered.len(), View::ALL.len());
        for (view, image, report) in rendered {
            assert_eq!(image.width(), 64);
            assert_eq!(image.height(), 64);
            assert_eq!(report.triangles, 12);
            let covered = 64 * 64 - background_count(&image, options.background);
            assert!(covered > 100, "{view} rendered only {covered} foreground pixels");
        }
    }

    #[test]
    fn drawable_without_geometry_renders_background_only() {
        let empty = Drawable {
            name: "empty".to_string(),
            name_hash: 0,
            bounds: empty_bounds(),
            lod_distances: [0.0; 4],
            render_masks: [0; 4],
            shader_group: None,
            lods: Vec::new(),
        };

        let options = options(16, 16);
        let (image, report) = render_drawable(&empty, &TextureSet::new(), &options).unwrap();

        assert_eq!(report.triangles, 0);
        assert_eq!(report.geometries, 0);
        assert_eq!(report.lod, None);
        assert_eq!(background_count(&image, options.background), 16 * 16);
    }

    #[test]
    fn texture_set_lookup_is_case_insensitive_and_layered() {
        use crate::ytd::{TextureFormat, YtdTexture};

        let texture = |name: &str, rgba: [u8; 4]| YtdTexture {
            name: name.to_string(),
            name_hash: rage_joaat(&name.to_lowercase()),
            width: 1,
            height: 1,
            depth: 1,
            format: TextureFormat::A8B8G8R8,
            levels: 1,
            stride: 4,
            pixel_data: rgba.to_vec(),
        };

        let mut set = TextureSet::new();
        assert!(set.is_empty());
        assert!(set.push_layer(&[texture("Skin", [255, 0, 0, 255])]).is_empty());
        assert!(set.push_layer(&[texture("skin", [0, 0, 255, 255])]).is_empty());

        assert!(!set.is_empty());
        assert_eq!(set.len(), 2);
        // Layer 0 wins.
        assert_eq!(set.get("SKIN").unwrap().get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert!(set.get("absent").is_none());
    }

    #[test]
    fn undecodable_texture_is_reported_by_push_layer() {
        use crate::ytd::{TextureFormat, YtdTexture};

        let broken = YtdTexture {
            name: "broken".to_string(),
            name_hash: rage_joaat("broken"),
            width: 4,
            height: 4,
            depth: 1,
            format: TextureFormat::Unknown,
            levels: 1,
            stride: 16,
            pixel_data: vec![0; 8],
        };

        let mut set = TextureSet::new();
        assert_eq!(set.push_layer(&[broken]), vec!["broken".to_string()]);
        assert!(set.is_empty());
    }
}
