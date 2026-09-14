# rpf-archive

Rust library for reading and writing Rockstar Games archive formats — RPF (Rockstar Package File) and IMG — across GTA III through Red Dead Redemption 2.

## Supported formats

| Version | Game(s) | Magic |
|---------|---------|-------|
| IMG1 | GTA III, Vice City | *(none — paired `.dir`+`.img`)* |
| IMG2 | GTA San Andreas | `VER2` |
| IMG3 | RAGE / modding tools | `0xA94E2A52` |
| RPF0 | Table Tennis | `RPF0` |
| RPF2 | GTA IV | `RPF2` |
| RPF3 | GTA IV Audio / MCLA | `RPF3` |
| RPF4 | Max Payne 3 | `RPF4` |
| RPF6 | Red Dead Redemption | `RPF6` |
| RPF7 | GTA V / FiveM | `RPF7` |
| RPF8 | Red Dead Redemption 2 | `RPF8` *(read-only)* |

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
rpf-archive = "0.8"
```

### Reading an RPF archive (RPF2–RPF7)

```rust
use rpf_archive::RpfFile;

let file = RpfFile::open("update.rpf".as_ref(), None)?;

file.walk(None, &mut |path, data| {
    println!("{} ({} bytes)", path, data.len());
})?;
```

### Reading a GTA V archive with encryption

```rust
use rpf_archive::{GtaKeys, RpfFile};

let keys = GtaKeys::load("gta5keys.bin")?;
let file = RpfFile::open("x64a.rpf".as_ref(), Some(&keys))?;
```

### Reading an IMG v2 archive (GTA San Andreas)

```rust
use rpf_archive::RpfFile;

let file = RpfFile::open("gta3.img".as_ref(), None)?;

file.walk(None, &mut |path, data| {
    println!("{}", path);
})?;
```

### Reading an IMG v1 archive (GTA III / Vice City)

```rust
use rpf_archive::RpfFile;

let file = RpfFile::open_img1("gta3.img".as_ref(), "gta3.dir".as_ref())?;

file.walk(None, &mut |path, data| {
    println!("{}", path);
})?;
```

### Building an RPF7 archive (GTA V)

```rust
use rpf_archive::{RpfBuilder, RpfEncryption};

let mut builder = RpfBuilder::new(RpfEncryption::Open);
builder.add_file("data/foo.ydr", my_bytes);
builder.add_file("data/bar.ytd", other_bytes);

let bytes = builder.build(None)?;
std::fs::write("output.rpf", bytes)?;
```

### Building an IMG v2 archive (GTA San Andreas)

```rust
use rpf_archive::{RpfBuilder, RpfEncryption, RpfVersion};

let mut builder = RpfBuilder::for_version(RpfVersion::Img2, RpfEncryption::None);
builder.add_file("vehicle.dff", dff_bytes);
builder.add_file("vehicle.txd", txd_bytes);

let bytes = builder.build(None)?;
std::fs::write("mod.img", bytes)?;
```

### Building an IMG v1 archive (GTA III / Vice City)

```rust
use rpf_archive::{RpfBuilder, RpfEncryption, RpfVersion};

let mut builder = RpfBuilder::for_version(RpfVersion::Img1, RpfEncryption::None);
builder.add_file("player.dff", dff_bytes);

let (dir_data, img_data) = builder.build_img1_pair()?;
std::fs::write("player.dir", dir_data)?;
std::fs::write("player.img", img_data)?;
```

### Directory tree

```rust
use rpf_archive::{RpfFile, build_directory_tree};

let file = RpfFile::open("update.rpf".as_ref(), None)?;
let tree = build_directory_tree(&file.archive);
```

### Drawables

`parse_drawables` reads any drawable-bearing resource — a lone `.ydr`, a
`.ydd` dictionary, or a `.yft` fragment — into one flat list of named
entries, letting the caller stay agnostic about which of the three it was
handed:

```rust
use rpf_archive::{parse_drawables, DrawableKind};

let entries = parse_drawables(&ydd_bytes, DrawableKind::Ydd)?;
for entry in &entries {
    println!("{} (0x{:08X}): {} triangle(s)", entry.name, entry.hash, entry.drawable.geometry_count());
}
```

Each `DrawableEntry { hash, name, drawable }` carries a `Drawable` — bounds,
LODs, shader group and geometry. Entries that would otherwise share a name
(dictionary members with no name of their own, or a fragment's extra
drawables with no names array) are automatically given unique `0x…` names
instead of overwriting each other.

### Textures to images

`.ytd` textures decode straight to an `image::RgbaImage`, ready to resize and
encode:

```rust
use rpf_archive::{parse_ytd, texture_utils::{to_rgba_image, fit_max_size, encode_image, ImageFormat}};

let textures = parse_ytd(&ytd_bytes)?;
for texture in &textures {
    let image = to_rgba_image(texture)?;
    let image = fit_max_size(image, 512);
    let png_bytes = encode_image(&image, ImageFormat::Png, 90)?;
    std::fs::write(format!("{}.png", texture.name), png_bytes)?;
}
```

### Rendering

A small CPU rasterizer (no GPU, runs under wasm) turns a parsed `Drawable`
into a preview image from one of six fixed camera angles:

```rust
use rpf_archive::{render_views, RenderOptions, TextureSet, View};

let mut textures = TextureSet::new();
textures.push_layer(&embedded_textures); // e.g. from the drawable's own shader group
textures.push_layer(&shared_textures);   // lower-priority fallback layer

// `paint` tints geometries drawn with a vehicle_paint*.sps shader: the YFT
// carries no body colour, the game applies it at runtime from carcols.
let options = RenderOptions { view: View::Iso, paint: Some([200, 30, 30]), ..Default::default() };
let rendered = render_views(&drawable, &textures, &options, &View::ALL)?;
for (view, image, report) in rendered {
    println!("{view}: {} triangle(s), {} missing texture(s)", report.triangles, report.missing_textures.len());
    image.save(format!("{view}.png"))?;
}
```

`render_drawable` is the single-view shorthand when only `options.view`
matters.

A fragment is more than its main drawable: wheels, doors and other physics
children are drawables of their own, each placed on the body by a transform.
`Fragment::render_parts` lists them (filling empty wheel slots from the
front/rear wheel mesh and mirroring right-hand wheels, as CodeWalker does)
and `render_parts` frames them together:

```rust
use rpf_archive::{parse_yft, render_parts, RenderOptions, RenderPart, TextureSet, View};

let fragment = parse_yft(&yft_bytes)?;
let parts: Vec<RenderPart<'_>> = fragment.render_parts().into_iter().map(RenderPart::from).collect();
let rendered = render_parts(&parts, &textures, &RenderOptions::default(), &[View::Iso])?;
```

A `RenderPart` carries the drawable, a transform applied to all of its models
and an optional per-bone pose selected by each model's bone index, so any
composite of drawables can be rendered the same way.

Each geometry is drawn the way its shader's RAGE render bucket says: bucket 0
is opaque and ignores alpha, bucket 3 (`cutout`, foliage, fences) is
alpha-tested at half, and buckets 1 and 2 (`*_alpha`, `decal`) are blended
over what is already drawn, after all solid geometry, back to front, without
writing depth. Blending over a transparent background keeps the coverage in
the output alpha.

One line each on two smaller pieces the renderer and CLI build on:
- **Contact sheet**: `compose_sheet`/`sheet_layout` lay out a grid of
  labelled thumbnails (e.g. one render per view, or a batch of extracted
  textures) into a single `RgbaImage`.
- **Bitmap font**: `draw_text`/`text_width` (backed by the `FONT_5X7` glyph
  table) draw simple pixel labels directly onto an `RgbaImage`, with no font
  file or text-shaping dependency.

## Changes in 0.9.1

- `Drawable::diffuse_texture_name` falls back to the `TextureSampler`
  parameter (`TEXTURE_SAMPLER`), so drawables converted from the GTA IV /
  Max Payne 3 pipelines no longer render untextured.
- `RenderReport::geometries_without_diffuse` counts the untextured
  geometries whose shader names no diffuse texture at all, apart from the
  ones listed in `missing_textures`.
- The vertex layer (`VertexBuffer`, `VertexDeclaration`, component types and
  decoding) lives in the new `vertex` module; every `ydd::` path still works.
  A Gen9 vertex buffer now reports `info_pointer` as 0 rather than the
  legacy slot's unrelated contents.
- Eighteen `FONT_5X7` glyphs that had drifted from Adafruit's `glcdfont.c`
  are back to the reference shapes.
- Back-face culling was checked against retail drawables: counter-clockwise
  front faces are the ones kept, on mirrored wheels too.

## Changes in 0.9.0

- `Fragment` now carries its physics children (`children`) and default bone
  pose (`bone_transforms`); `Fragment::render_parts` lists the body and every
  child with a mesh, wheel slots filled in and mirrored as needed.
- New `render_parts` / `RenderPart` render several placed drawables into one
  image; `render_views` is the single-part case. Models are posed by their
  bone index (`DrawableModel::bone_index`, `is_skinned`).
- `Mat4` gained `from_d3d`, `from_translation`, `translation`,
  `with_translation`, `is_identity` and `transform_vector`.
- Breaking: `Fragment` has two new public fields, so code constructing it by
  hand must set them.

## Changes in 0.8.1

- The renderer picks alpha handling from the shader's render bucket instead of
  alpha-testing any texture that has a translucent pixel. Translucent
  materials (ziplock bags, glass, decals) now blend instead of vanishing.

## Breaking changes in 0.8.0

- `parse_ydd` now returns `Vec<DrawableEntry>` instead of a bare list of
  `Drawable`s, so every entry carries its resolved hash and unique name
  alongside the drawable itself.
- `Drawable`'s shape changed to carry bounds, LODs, shader group and render
  masks needed by the new renderer.
- The wasm `convert_to_gltf` payload's envelope changed: it is now
  `{"format": "unified_v2", ...}`; the old `gta_geometry` envelope has been
  removed.
- New shader-parameter hash constants: `DIFFUSE_SAMPLER`, `BUMP_SAMPLER`,
  `SPEC_SAMPLER`.

## License

This project is released under the [Unlicense](LICENSE) — public domain.
