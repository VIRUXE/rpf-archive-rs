// Parses the texture-dictionary parent relationships the game declares in
// `gtxd.ymt`, `gtxd.meta`, `mph4_gtxd.ymt` and `vehicles.meta` — the "a txd
// can inherit textures from a parent txd" mechanism CodeWalker's
// `GameFileCache.InitGtxds` builds `textureParents` from.
//
// These files come in two on-disk shapes, both of which this module
// dispatches on by content rather than filename (CodeWalker switches on the
// extension, but it never has to handle the ambiguity described below):
//
//   - `gtxd.ymt` is normally RBF (see `rbf.rs`), rooted at a `CMapParentTxds`
//     structure. It may also arrive RSC7-wrapped: whether an RPF entry is a
//     "resource" or a plain "binary" file is a property of the archive (a
//     TOC flag bit), not of the extension, and a resource entry's extracted
//     bytes carry a synthesized RSC7 header with the body still deflated.
//   - `gtxd.meta` and `vehicles.meta` are plain XML, rooted at
//     `CMapParentTxds` and `CVehicleModelInfo__InitDataList` respectively,
//     otherwise identical in shape: a `txdRelationships` element holding
//     repeated `Item`/`item` elements, each with `parent`/`child` children
//     whose inner text is a texture dictionary name.
//
// Ported from CodeWalker.Core's GameFiles/FileTypes/GtxdFile.cs and
// VehiclesFile.cs.

use anyhow::{bail, Context, Result};
use crate::rbf;
use crate::resource::prepare_rsc7;

const RSC7_MAGIC: u32 = 0x37435352;

/// One texture dictionary's parent binding, as the file names it (no `.ytd`
/// extension, original case) — hashing and merge policy are left to the
/// caller, matching how CodeWalker keeps `GtxdFile.TxdRelationships` as
/// plain strings and only hashes them once, in `GameFileCache.InitGtxds`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TxdRelationship {
    pub child: String,
    pub parent: String,
}

/// Root element names accepted for the XML form of this file.
/// `CMapParentTxds` is `gtxd.meta`'s own root; `vehicles.meta` uses
/// `CVehicleModelInfo__InitDataList` for the same `txdRelationships` shape
/// (`VehiclesFile.cs`).
const XML_ROOTS: [&str; 2] = ["CMapParentTxds", "CVehicleModelInfo__InitDataList"];

/// Parses every `child -> parent` texture-dictionary relationship out of a
/// `gtxd.ymt`/`gtxd.meta`/`mph4_gtxd.ymt`/`vehicles.meta`'s raw bytes, in
/// file order. A well-formed file that simply declares none returns an
/// empty vec, not an error; an unrecognised root (RBF or XML) does too,
/// matching `GtxdFile.Load`'s silent skip.
pub fn parse_txd_relationships(data: &[u8]) -> Result<Vec<TxdRelationship>> {
    if rbf::is_rbf(data) {
        let root = rbf::parse(data).context("gtxd: malformed RBF stream")?;
        return Ok(from_rbf(&root));
    }

    if data.len() >= 4 && u32::from_le_bytes(data[0..4].try_into().unwrap()) == RSC7_MAGIC {
        let (system, _graphics) = prepare_rsc7(data).context("gtxd: RSC7 container did not unpack")?;
        if rbf::is_rbf(&system) {
            let root = rbf::parse(&system).context("gtxd: malformed RBF stream inside RSC7 container")?;
            return Ok(from_rbf(&root));
        }
        bail!("gtxd: RSC7-wrapped file is not RBF (a PSO/Meta .ymt is not supported)");
    }

    let text = std::str::from_utf8(strip_bom(data)).context("gtxd: not RBF, RSC7 or UTF-8 XML")?;
    from_xml(text)
}

fn strip_bom(data: &[u8]) -> &[u8] {
    data.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(data)
}

/// Transcribes `GtxdFile.LoadTxdRelationships(RbfStructure)`: root
/// `CMapParentTxds` -> `txdRelationships` -> children named `item`
/// (lowercase in RBF, unlike the XML form) -> child structures `parent`/
/// `child`, each holding its string as a raw-bytes first child.
fn from_rbf(root: &rbf::RbfStructure) -> Vec<TxdRelationship> {
    if root.name != "CMapParentTxds" {
        return Vec::new();
    }
    let Some(relationships) = root.child("txdRelationships") else { return Vec::new() };

    let mut out = Vec::new();
    for item in relationships.children_named("item") {
        let Some(parent) = item.child("parent").and_then(|s| s.first_bytes_as_string()) else { continue };
        let Some(child) = item.child("child").and_then(|s| s.first_bytes_as_string()) else { continue };
        if parent.is_empty() || child.is_empty() {
            continue;
        }
        out.push(TxdRelationship { child, parent });
    }
    out
}

/// Transcribes `GtxdFile`/`VehiclesFile`'s `LoadTxdRelationships(XmlDocument)`:
/// `<root>/<txdRelationships>/<Item|item>` with `<parent>`/`<child>` inner
/// text, for either accepted root element.
fn from_xml(text: &str) -> Result<Vec<TxdRelationship>> {
    let doc = roxmltree::Document::parse(text).context("gtxd: failed to parse XML")?;
    let root = doc.root_element();

    if !XML_ROOTS.contains(&root.tag_name().name()) {
        return Ok(Vec::new());
    }

    let Some(relationships) = root.children().find(|n| n.is_element() && n.tag_name().name() == "txdRelationships")
    else {
        return Ok(Vec::new());
    };

    let mut out = Vec::new();
    for item in relationships
        .children()
        .filter(|n| n.is_element() && (n.tag_name().name() == "Item" || n.tag_name().name() == "item"))
    {
        let parent = child_text(item, "parent");
        let child = child_text(item, "child");
        let (Some(parent), Some(child)) = (parent, child) else { continue };
        if parent.is_empty() || child.is_empty() {
            continue;
        }
        out.push(TxdRelationship { child, parent });
    }
    Ok(out)
}

fn child_text(node: roxmltree::Node<'_, '_>, name: &str) -> Option<String> {
    node.children()
        .find(|n| n.is_element() && n.tag_name().name() == name)
        .map(|n| n.text().unwrap_or("").trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── RBF fixtures (mirrors rbf.rs's own build_gtxd_rbf, kept local so
    // this module's tests exercise the real public entry point) ──────────

    fn open(idx: u8, name: Option<&str>, pending_attrs: i16) -> Vec<u8> {
        let mut v = vec![idx, 0x00];
        if let Some(n) = name {
            v.extend_from_slice(&(n.len() as i16).to_le_bytes());
            v.extend_from_slice(n.as_bytes());
        }
        v.extend_from_slice(&0i16.to_le_bytes());
        v.extend_from_slice(&0i16.to_le_bytes());
        v.extend_from_slice(&pending_attrs.to_le_bytes());
        v
    }

    fn bytes_node(s: &str) -> Vec<u8> {
        let mut payload = s.as_bytes().to_vec();
        payload.push(0);
        let mut v = vec![0xFD, 0xFF];
        v.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        v.extend_from_slice(&payload);
        v
    }

    fn close() -> Vec<u8> {
        vec![0xFF, 0xFF]
    }

    fn build_gtxd_rbf(root_name: &str, pairs: &[(&str, &str)]) -> Vec<u8> {
        let mut d = b"RBF0".to_vec();
        d.extend(open(0, Some(root_name), 0));
        d.extend(open(1, Some("txdRelationships"), 0));

        let (mut item_idx, mut parent_idx, mut child_idx): (Option<u8>, Option<u8>, Option<u8>) = (None, None, None);
        let mut next_free = 2u8;

        for (child, parent) in pairs {
            d.extend(open(item_idx.unwrap_or(next_free), item_idx.is_none().then_some("item"), 0));
            if item_idx.is_none() {
                item_idx = Some(next_free);
                next_free += 1;
            }
            d.extend(open(parent_idx.unwrap_or(next_free), parent_idx.is_none().then_some("parent"), 0));
            if parent_idx.is_none() {
                parent_idx = Some(next_free);
                next_free += 1;
            }
            d.extend(bytes_node(parent));
            d.extend(close());
            d.extend(open(child_idx.unwrap_or(next_free), child_idx.is_none().then_some("child"), 0));
            if child_idx.is_none() {
                child_idx = Some(next_free);
                next_free += 1;
            }
            d.extend(bytes_node(child));
            d.extend(close());
            d.extend(close());
        }
        d.extend(close());
        d.extend(close());
        d
    }

    #[test]
    fn parses_rbf_gtxd_relationships() {
        let data = build_gtxd_rbf("CMapParentTxds", &[("some_child_txd", "vehshare")]);
        let rels = parse_txd_relationships(&data).expect("should parse");
        assert_eq!(rels, vec![TxdRelationship { child: "some_child_txd".into(), parent: "vehshare".into() }]);
    }

    #[test]
    fn rbf_root_other_than_cmapparenttxds_yields_nothing() {
        let data = build_gtxd_rbf("SomethingElse", &[("a", "b")]);
        let rels = parse_txd_relationships(&data).expect("should parse");
        assert!(rels.is_empty());
    }

    #[test]
    fn rsc7_wrapped_rbf_is_unwrapped() {
        // prepare_rsc7 treats a body that never looks like a deflate stream
        // as "stored" (uncompressed), so an uncompressed RBF body round-trips.
        let rbf_bytes = build_gtxd_rbf("CMapParentTxds", &[("child_a", "parent_a")]);

        let mut data = Vec::new();
        data.extend_from_slice(&RSC7_MAGIC.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // version, unused by prepare_rsc7
        // system_flags / graphics_flags: resource_size_from_flags decodes a
        // packed size; a flags value of 0 with the "long size" bits unset
        // and a small enough size field yields the exact byte length below.
        // Simpler: pack size 0 for graphics (unused) and craft system_flags
        // for exactly rbf_bytes.len() using the same scheme prepare_rsc7's
        // sibling (resource_size_from_flags) expects. We defer to the
        // crate's own round-trip helper via ymt/ytyp fixtures elsewhere, but
        // gtxd's own test only needs *some* valid encoding, so we build one
        // with the smallest bucket count matching the data length.
        let (sys_flags, sys_padded_len) = encode_resource_flags(rbf_bytes.len());
        data.extend_from_slice(&sys_flags.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // graphics_flags: empty section
        let mut body = rbf_bytes.clone();
        body.resize(sys_padded_len, 0);
        data.extend_from_slice(&body);

        let rels = parse_txd_relationships(&data).expect("should unwrap RSC7 and parse RBF");
        assert_eq!(rels, vec![TxdRelationship { child: "child_a".into(), parent: "parent_a".into() }]);
    }

    /// Encodes a resource size using RAGE's flags scheme (`archive.rs`'s
    /// `resource_size_from_flags`) as a single "one base-size page" bucket:
    /// bit 27 set (contributing exactly one unit) and the base shift in the
    /// low 4 bits, so the decoded size is `0x200 << shift`, chosen as the
    /// smallest such size covering `len`.
    fn encode_resource_flags(len: usize) -> (u32, usize) {
        let mut shift = 0u32;
        while (0x200usize << shift) < len.max(1) {
            shift += 1;
        }
        let flags = (1u32 << 27) | shift;
        (flags, 0x200usize << shift)
    }

    // ─── XML fixtures ─────────────────────────────────────────────────────

    #[test]
    fn parses_xml_gtxd_meta() {
        let xml = "\u{FEFF}<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                   <CMapParentTxds>\n\
                   <!-- a comment -->\n\
                   <txdRelationships>\n\
                   <Item note=\"attr\">\n\
                   <parent>  vehshare  </parent>\n\
                   <child>some_child_txd</child>\n\
                   </Item>\n\
                   <item>\n\
                   <parent>other_parent</parent>\n\
                   <child>other_child</child>\n\
                   </item>\n\
                   </txdRelationships>\n\
                   </CMapParentTxds>";
        let rels = parse_txd_relationships(xml.as_bytes()).expect("should parse");
        assert_eq!(
            rels,
            vec![
                TxdRelationship { child: "some_child_txd".into(), parent: "vehshare".into() },
                TxdRelationship { child: "other_child".into(), parent: "other_parent".into() },
            ]
        );
    }

    #[test]
    fn parses_vehicles_meta_root_and_ignores_siblings() {
        let xml = "<CVehicleModelInfo__InitDataList>\
                   <InitDatas><Item><modelName>ignored</modelName></Item></InitDatas>\
                   <txdRelationships><Item><parent>vehshare</parent><child>some_veh_txd</child></Item></txdRelationships>\
                   </CVehicleModelInfo__InitDataList>";
        let rels = parse_txd_relationships(xml.as_bytes()).expect("should parse");
        assert_eq!(rels, vec![TxdRelationship { child: "some_veh_txd".into(), parent: "vehshare".into() }]);
    }

    #[test]
    fn xml_items_missing_parent_or_child_are_skipped() {
        let xml = "<CMapParentTxds><txdRelationships>\
                   <Item><parent>only_parent</parent></Item>\
                   <Item><child>only_child</child></Item>\
                   <Item><parent></parent><child>x</child></Item>\
                   </txdRelationships></CMapParentTxds>";
        let rels = parse_txd_relationships(xml.as_bytes()).expect("should parse");
        assert!(rels.is_empty());
    }

    #[test]
    fn xml_unknown_root_yields_nothing() {
        let xml = "<SomethingElse><txdRelationships><Item><parent>a</parent><child>b</child></Item></txdRelationships></SomethingElse>";
        let rels = parse_txd_relationships(xml.as_bytes()).expect("should parse");
        assert!(rels.is_empty());
    }

    #[test]
    fn xml_missing_txd_relationships_yields_nothing() {
        let xml = "<CMapParentTxds></CMapParentTxds>";
        let rels = parse_txd_relationships(xml.as_bytes()).expect("should parse");
        assert!(rels.is_empty());
    }

    #[test]
    fn rejects_data_that_is_neither_rbf_rsc7_nor_utf8() {
        let data = vec![0xFF, 0xFE, 0x00, 0x01, 0x02];
        assert!(parse_txd_relationships(&data).is_err());
    }
}
