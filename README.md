# rpf-archive

Rust library for reading and writing Rockstar Games archive formats, RPF
(Rockstar Package File) and IMG, across GTA III through Red Dead Redemption
2. Archives only: what is inside them is parsed by
[`rage-formats`](https://github.com/VIRUXE/rage-formats), drawn by
[`rage-render`](https://github.com/VIRUXE/rage-render), and driven from the
command line by [`rage-cli`](https://github.com/VIRUXE/rage-cli).

```toml
[dependencies]
rpf-archive = "0.10"
```

## Supported formats

| Version | Game(s) | Magic | Read | Write |
|---------|---------|-------|:-:|:-:|
| IMG1 | GTA III, Vice City | *(none; paired `.dir`+`.img`)* | yes | yes |
| IMG2 | GTA San Andreas | `VER2` | yes | yes |
| IMG3 | RAGE / modding tools | `0xA94E2A52` | yes | yes |
| RPF0 | Table Tennis | `RPF0` | yes | yes |
| RPF2 | GTA IV | `RPF2` | yes | yes |
| RPF3 | GTA IV Audio / MCLA | `RPF3` | yes | yes |
| RPF4 | Max Payne 3 | `RPF4` | yes | yes |
| RPF6 | Red Dead Redemption | `RPF6` | yes | yes |
| RPF7 | GTA V / FiveM | `RPF7` | yes | yes |
| RPF8 | Red Dead Redemption 2 | `RPF8` | yes | |

## What the crate covers

- **Reading**: the table of contents, directory tree, entry kinds (binary
  files, RSC7 resources with their page flags, nested archives) and
  extraction with decompression.
- **Encryption**: GTA V's AES and NG schemes. `GtaKeys` loads keys written
  out to disk; recovering them from `GTA5.exe` is `rage-cli`'s job (see its
  Keys section), this crate only consumes them. Open (unencrypted) archives,
  the kind FiveM resources use, need no keys.
- **Writing**: `RpfBuilder` assembles an archive of any supported version
  from paths and bytes, with the encryption you choose; RSC7 resources are
  detected by magic and recorded with the right flags.
- **Load order**: `parse_dlc_list` and `parse_dlc_setup_order` read
  `dlclist.xml` and each pack's `setup2.xml`, which is how a consumer ranks
  archives the way the game does (base, `update.rpf`, then DLC packs in
  order, later wins).
- **Resource sizes**: `resource_size_from_flags` and
  `resource_version_from_flags` decode an RSC7 entry's header words, since
  an archive listing needs them; `rage-formats` exports the same values for
  loose files.

## Usage

### Reading an RPF archive (RPF2 to RPF7)

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

## Where things live

```
src/
  archive.rs    parsing every archive version's table of contents; entry kinds; extraction
  crypto/       AES and NG ciphers, key loading
  writer.rs     RpfBuilder, and rage_joaat for entry name hashes
  tree.rs       directory tree over a parsed archive
  dlc.rs        dlclist.xml / setup2.xml load order
  tests.rs      write-then-read round trips for every version
```

## Changes in 0.10.0

The crate is archives only again. Everything that parsed or drew what is
*inside* an archive moved out, with its git history:

- `rage-formats`: RSC7 resources (textures, drawables, fragments, meta,
  gtxd), `texture_utils`, `math`, `vertex`, `rage_joaat`; since then also
  `ybn`, `ynv` (read and write), `ymap` and a full `ytyp` parser.
- `rage-render`: the CPU rasteriser, contact sheets, bitmap font and the
  wasm glTF export.

`rpf-archive` keeps reading and writing RPF/IMG archives, the crypto, the
directory tree and the DLC list parsers, and drops the `image`, `gltf`,
`texture2ddecoder`, `json` and `wasm-bindgen` dependencies. `RSC7_MAGIC`,
`resource_size_from_flags` and `resource_version_from_flags` are still
exported here (an archive needs them for entry sizes); `rage-formats`
exports the same values. Callers that imported parsers from this crate
import them from `rage-formats` now.

## Earlier changes

<details>
<summary>0.9.1</summary>

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

</details>

<details>
<summary>0.9.0</summary>

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

</details>

<details>
<summary>0.8.1</summary>

- The renderer picks alpha handling from the shader's render bucket instead of
  alpha-testing any texture that has a translucent pixel. Translucent
  materials (ziplock bags, glass, decals) now blend instead of vanishing.

</details>

<details>
<summary>0.8.0 (breaking)</summary>

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

</details>

## License

This project is released under the [Unlicense](LICENSE), public domain.
