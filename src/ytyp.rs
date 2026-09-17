// Parses just enough of a .ytyp (RSC7-wrapped "Meta" format) to answer one
// question: which texture dictionary does a given archetype (prop/model)
// reference? That is `CBaseArchetypeDef.textureDictionary`, keyed by the
// archetype's `name` (or `assetName` when `name` is 0), matching how the
// game's own renderer picks a texture dictionary for a drawable
// (CodeWalker `Archetype.cs` / `Renderer.cs TryGetRenderable`).
//
// A .ytyp's Meta container is a third format alongside the fixed-pointer-
// chasing RSC7 resources (Drawable, Texture, ...) and the PSO/RBF text
// format: a self-describing binary blob of typed "data blocks", each
// tagged with a structure-name hash. We only care about two structure
// types, `CMapTypes` (the root, holding the archetype list) and
// `CBaseArchetypeDef`/`CTimeArchetypeDef`/`CMloArchetypeDef` (one per
// archetype; all three share the same 144-byte layout for the fields we
// read, differing only in what follows).
//
// Byte layout ported from CodeWalker.Core (Meta.cs, MetaTypes.cs,
// YtypFile.cs) — see that project for the full, generic Meta/PSO reader
// this deliberately does not reimplement.

use anyhow::{bail, Context, Result};
use crate::resource::{prepare_rsc7, u16_le, u32_le, u64_le, ResReader, SYSTEM_BASE};

/// Structure-name hashes as CodeWalker's `MetaName` enum defines them: the
/// RAGE Jenkins hash of the exact-case structure name (unlike texture/file
/// names elsewhere in this crate, these are *not* lowercased first).
const HASH_CMAPTYPES: u32 = 3_649_811_809;
const HASH_CBASE_ARCHETYPE_DEF: u32 = 2_195_127_427;
const HASH_CTIME_ARCHETYPE_DEF: u32 = 1_991_296_364;
const HASH_CMLO_ARCHETYPE_DEF: u32 = 273_704_021;

/// One archetype's texture-dictionary binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchetypeTxd {
    /// `name` (or `assetName` when `name` is 0) — the archetype's own
    /// lowercase-JOAAT name hash, the same hash space as a drawable's file
    /// stem.
    pub name_hash: u32,
    /// `textureDictionary` — 0 when the archetype names none.
    pub texture_dict_hash: u32,
}

struct MetaBlock {
    name_hash: u32,
    data: Vec<u8>,
}

/// Decodes a packed Meta-format pointer (distinct from the resource VAs
/// used elsewhere: `block_id` in the low 12 bits, 1-based, an offset in the
/// next 20) into a zero-based block index and byte offset. `None` for a
/// null or zero-block pointer.
fn decode_meta_pointer(raw: u64) -> Option<(usize, usize)> {
    let block_id = (raw & 0xFFF) as usize;
    if block_id == 0 {
        return None;
    }
    let offset = ((raw >> 12) & 0xFFFFF) as usize;
    Some((block_id - 1, offset))
}

/// Reads every `MetaDataBlock` out of a .ytyp's `Meta` header at
/// `SYSTEM_BASE`. The header is 0x70 (112) bytes; the fields we need are
/// `DataBlocksPointer` at 0x30 and `DataBlocksCount` at 0x4C.
fn read_meta_blocks(reader: &ResReader<'_>) -> Result<Vec<MetaBlock>> {
    let header = reader.resolve(SYSTEM_BASE, 0x70).context("ytyp: Meta header out of bounds")?;

    let data_blocks_pointer = u64_le(header, 0x30);
    let data_blocks_count = u16_le(header, 0x4C) as usize;

    if data_blocks_pointer == 0 || data_blocks_count == 0 {
        bail!("ytyp: no data blocks");
    }

    let block_headers = reader
        .resolve(data_blocks_pointer, data_blocks_count.checked_mul(16).context("ytyp: data block count overflow")?)
        .context("ytyp: data block array out of bounds")?;

    let mut blocks = Vec::with_capacity(data_blocks_count);
    for i in 0..data_blocks_count {
        let off = i * 16;
        let name_hash = u32_le(block_headers, off);
        let length = u32_le(block_headers, off + 4) as usize;
        let data_ptr = u64_le(block_headers, off + 8);
        // A block whose data pointer doesn't resolve is skipped rather than
        // failing the whole file — other blocks may still be usable.
        let data = reader.resolve(data_ptr, length).map(|b| b.to_vec()).unwrap_or_default();
        blocks.push(MetaBlock { name_hash, data });
    }

    Ok(blocks)
}

/// Extracts every archetype's `(name hash, texture dictionary hash)` from a
/// .ytyp's raw bytes.
pub fn parse_archetype_txds(data: &[u8]) -> Result<Vec<ArchetypeTxd>> {
    let (system, graphics) = prepare_rsc7(data)?;
    let reader = ResReader { system: &system, graphics: &graphics };
    parse_archetype_txds_from_reader(&reader)
}

fn parse_archetype_txds_from_reader(reader: &ResReader<'_>) -> Result<Vec<ArchetypeTxd>> {
    let blocks = read_meta_blocks(reader)?;

    let cmaptypes = blocks
        .iter()
        .find(|b| b.name_hash == HASH_CMAPTYPES)
        .context("ytyp: CMapTypes block not found")?;

    // CMapTypes is 80 bytes; `archetypes` (an Array_StructurePointer: a
    // packed pointer + two u16 counts) sits at offset 24.
    const ARCHETYPES_FIELD_OFFSET: usize = 24;
    if cmaptypes.data.len() < ARCHETYPES_FIELD_OFFSET + 16 {
        bail!("ytyp: CMapTypes block too small");
    }

    let archetypes_pointer = u64_le(&cmaptypes.data, ARCHETYPES_FIELD_OFFSET);
    let archetypes_count = u16_le(&cmaptypes.data, ARCHETYPES_FIELD_OFFSET + 8) as usize;

    let Some((arr_block_idx, arr_offset)) = decode_meta_pointer(archetypes_pointer) else {
        return Ok(Vec::new());
    };
    let Some(arr_block) = blocks.get(arr_block_idx) else {
        bail!("ytyp: archetypes pointer block out of range");
    };

    let ptr_bytes_len = archetypes_count.checked_mul(8).context("ytyp: archetype count overflow")?;
    let Some(ptr_bytes) = arr_block.data.get(arr_offset..arr_offset + ptr_bytes_len) else {
        bail!("ytyp: archetypes pointer array out of bounds");
    };

    // The three archetype-def structure kinds all begin with the same
    // 144-byte `CBaseArchetypeDef` layout; only what follows differs, and
    // we don't read past byte 144.
    const BASE_ARCHETYPE_DEF_LEN: usize = 144;
    const NAME_OFFSET: usize = 88;
    const TEXTURE_DICT_OFFSET: usize = 92;
    const ASSET_NAME_OFFSET: usize = 112;

    let mut out = Vec::with_capacity(archetypes_count);
    for i in 0..archetypes_count {
        let ptr = u64_le(ptr_bytes, i * 8);
        let Some((a_block_idx, a_offset)) = decode_meta_pointer(ptr) else { continue };
        let Some(block) = blocks.get(a_block_idx) else { continue };
        if !matches!(
            block.name_hash,
            HASH_CBASE_ARCHETYPE_DEF | HASH_CTIME_ARCHETYPE_DEF | HASH_CMLO_ARCHETYPE_DEF
        ) {
            continue;
        }
        let Some(base) = block.data.get(a_offset..a_offset + BASE_ARCHETYPE_DEF_LEN) else { continue };

        let mut name_hash = u32_le(base, NAME_OFFSET);
        let texture_dict_hash = u32_le(base, TEXTURE_DICT_OFFSET);
        if name_hash == 0 {
            // CodeWalker Archetype.cs: `Hash = arch.assetName` when `name` is 0.
            name_hash = u32_le(base, ASSET_NAME_OFFSET);
        }

        out.push(ArchetypeTxd { name_hash, texture_dict_hash });
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::rage_joaat;

    #[test]
    fn structure_hashes_match_codewalkers_metaname_enum() {
        // MetaName hashes are the RAGE Jenkins hash of the exact-case
        // structure name (not lowercased) -- verified against CodeWalker's
        // MetaNames.cs constants.
        assert_eq!(rage_joaat_exact_case("CMapTypes"), HASH_CMAPTYPES);
        assert_eq!(rage_joaat_exact_case("CBaseArchetypeDef"), HASH_CBASE_ARCHETYPE_DEF);
        assert_eq!(rage_joaat_exact_case("CTimeArchetypeDef"), HASH_CTIME_ARCHETYPE_DEF);
        assert_eq!(rage_joaat_exact_case("CMloArchetypeDef"), HASH_CMLO_ARCHETYPE_DEF);
    }

    // `rage_joaat` documents its input as "should already be lowercase", but
    // the hash function itself is case-agnostic about that contract -- it
    // just hashes whatever bytes it's given. MetaName hashes are exact-case.
    fn rage_joaat_exact_case(s: &str) -> u32 {
        rage_joaat(s)
    }

    #[test]
    fn decode_meta_pointer_rejects_zero_block_id() {
        assert_eq!(decode_meta_pointer(0), None);
        // block_id occupies the low 12 bits; 0 there means null regardless
        // of a nonzero offset in the upper bits.
        assert_eq!(decode_meta_pointer(0x1000), None);
    }

    #[test]
    fn decode_meta_pointer_splits_block_id_and_offset() {
        // block_id = 1 (-> index 0), offset = 0x10
        let raw = 1u64 | (0x10u64 << 12);
        assert_eq!(decode_meta_pointer(raw), Some((0, 0x10)));

        // block_id = 3 (-> index 2), offset = 0
        assert_eq!(decode_meta_pointer(3), Some((2, 0)));
    }

    /// Builds a minimal .ytyp-shaped `Meta` system section: a `CMapTypes`
    /// block, a block holding the archetypes-pointer array (its own block,
    /// as the real format has it — the array is not inline in `CMapTypes`),
    /// and one `CBaseArchetypeDef` block.
    fn build_ytyp_system(name_hash: u32, texture_dict_hash: u32, asset_name_hash: u32) -> Vec<u8> {
        // Layout (all offsets are byte offsets within `system`; resource
        // VAs are SYSTEM_BASE + offset):
        //   0x000..0x070  Meta header
        //   0x070..0x0A0  DataBlocks array: 3 x MetaDataBlock (16 bytes each)
        //   0x0A0..0x0F0  block 0: CMapTypes data (80 bytes)
        //   0x0F0..0x0F8  block 1: archetypes pointer array (1 x u64)
        //   0x0F8..0x188  block 2: CBaseArchetypeDef data (144 bytes)
        let mut sys = vec![0u8; 0x188];

        // Meta header.
        sys[0x30..0x38].copy_from_slice(&(SYSTEM_BASE + 0x070).to_le_bytes()); // DataBlocksPointer
        sys[0x4C..0x4E].copy_from_slice(&3u16.to_le_bytes()); // DataBlocksCount

        // DataBlocks[0]: CMapTypes.
        sys[0x070..0x074].copy_from_slice(&HASH_CMAPTYPES.to_le_bytes());
        sys[0x074..0x078].copy_from_slice(&80u32.to_le_bytes());
        sys[0x078..0x080].copy_from_slice(&(SYSTEM_BASE + 0x0A0).to_le_bytes());

        // DataBlocks[1]: archetypes pointer array. Its structure-name hash
        // doesn't matter to the parser (only archetype-def blocks are
        // matched by hash), so it's left 0.
        sys[0x080..0x084].copy_from_slice(&0u32.to_le_bytes());
        sys[0x084..0x088].copy_from_slice(&8u32.to_le_bytes());
        sys[0x088..0x090].copy_from_slice(&(SYSTEM_BASE + 0x0F0).to_le_bytes());

        // DataBlocks[2]: CBaseArchetypeDef.
        sys[0x090..0x094].copy_from_slice(&HASH_CBASE_ARCHETYPE_DEF.to_le_bytes());
        sys[0x094..0x098].copy_from_slice(&144u32.to_le_bytes());
        sys[0x098..0x0A0].copy_from_slice(&(SYSTEM_BASE + 0x0F8).to_le_bytes());

        // CMapTypes.archetypes (offset 24 within the 80-byte struct):
        // packed pointer -> block index 1 (encoded as block_id=2), offset 0
        // within that block's data; count1 = 1.
        let archetypes_field = 0x0A0 + 24;
        let packed_ptr = 2u64; // block_id=2 -> index 1, offset=0
        sys[archetypes_field..archetypes_field + 8].copy_from_slice(&packed_ptr.to_le_bytes());
        sys[archetypes_field + 8..archetypes_field + 10].copy_from_slice(&1u16.to_le_bytes());

        // archetypes pointer array (block index 1): one packed pointer ->
        // block index 2 (block_id=3), offset 0.
        let packed_arch_ptr = 3u64; // block_id=3 -> index 2, offset=0
        sys[0x0F0..0x0F8].copy_from_slice(&packed_arch_ptr.to_le_bytes());

        // CBaseArchetypeDef fields (block index 2, base 0x0F8).
        sys[0x0F8 + 88..0x0F8 + 92].copy_from_slice(&name_hash.to_le_bytes());
        sys[0x0F8 + 92..0x0F8 + 96].copy_from_slice(&texture_dict_hash.to_le_bytes());
        sys[0x0F8 + 112..0x0F8 + 116].copy_from_slice(&asset_name_hash.to_le_bytes());

        sys
    }

    #[test]
    fn parses_one_archetypes_texture_dictionary() {
        let system = build_ytyp_system(rage_joaat("prop_table_02"), rage_joaat("prop_tableset_02"), 0);
        let reader = ResReader { system: &system, graphics: &[] };
        let archetypes = parse_archetype_txds_from_reader(&reader).expect("should parse");
        assert_eq!(archetypes.len(), 1);
        assert_eq!(archetypes[0].name_hash, rage_joaat("prop_table_02"));
        assert_eq!(archetypes[0].texture_dict_hash, rage_joaat("prop_tableset_02"));
    }

    #[test]
    fn falls_back_to_asset_name_when_name_is_zero() {
        let system = build_ytyp_system(0, rage_joaat("prop_tableset_02"), rage_joaat("prop_table_02"));
        let reader = ResReader { system: &system, graphics: &[] };
        let archetypes = parse_archetype_txds_from_reader(&reader).expect("should parse");
        assert_eq!(archetypes.len(), 1);
        assert_eq!(archetypes[0].name_hash, rage_joaat("prop_table_02"));
    }
}
