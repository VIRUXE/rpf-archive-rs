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
rpf-archive = "0.10"
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

## Changes in 0.10.0

The crate is archives only again. Everything that parsed or drew what is
*inside* an archive moved out:

- `rage-formats` — RSC7 resources: textures (`.ytd`), drawables (`.ydr`/
  `.ydd`), fragments (`.yft`), `.ymt`/`.ytyp` meta, `gtxd` relationships,
  `texture_utils`, `math`, `vertex`, `rage_joaat`.
- `rage-render` — the CPU rasteriser, contact sheets, bitmap font and the
  wasm glTF export.

`rpf-archive` keeps reading and writing RPF/IMG archives, the crypto, the
directory tree and the DLC list parsers. `RSC7_MAGIC`,
`resource_size_from_flags` and `resource_version_from_flags` are still
exported here (an archive needs them for entry sizes); `rage-formats`
exports the same values.

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
