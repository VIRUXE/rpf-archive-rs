// A minimal reader for RAGE's "RBF0" binary format — the format `gtxd.ymt`
// uses to describe texture-dictionary parent relationships. This is a
// separate, self-describing binary format from both the RSC7 fixed-pointer
// resources (Drawable, Texture, ...) and the Meta "data blocks" format
// `ytyp.rs` reads; ported from CodeWalker.Core's `MetaTypes/Rbf.cs`.
//
// The stream is a flat sequence of records with no length prefix at the
// record level — advancing past a record requires already knowing its
// payload type, which in turn requires the descriptor table built so far.
// That rules out a narrow scanner that only looks for the fields this crate
// currently cares about: skipping unrelated records still requires
// implementing the whole grammar, so this reader builds a full tree
// (structures with attributes and children) rather than trying to be
// narrower than that.
//
// Two deliberate departures from CodeWalker's reader:
//   - CodeWalker asserts the stream ends exactly at the root close tag
//     (`Rbf.cs` `Load`). Bytes handed to us here can arrive page-padded
//     from the RPF layer, so trailing bytes after the root closes are
//     ignored instead of rejected.
//   - Every out-of-bounds read is a `Result::Err`, never a panic.

use anyhow::{bail, Context, Result};

const RBF_MAGIC: [u8; 4] = *b"RBF0";

/// One value inside an [`RbfStructure`]. Bytes are the format's own
/// "raw blob" record (tag `0xFD`) — used by `gtxd.ymt` to store a
/// NUL-padded ASCII string one level down from a `<parent>`/`<child>`
/// structure, rather than the format's own `0x60` string type.
#[derive(Debug, Clone, PartialEq)]
pub enum RbfValue {
    Structure(RbfStructure),
    Bytes(Vec<u8>),
    Uint(u32),
    Bool(bool),
    Float(f32),
    Float3([f32; 3]),
    Str(String),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RbfStructure {
    pub name: String,
    pub attributes: Vec<(String, RbfValue)>,
    pub children: Vec<(String, RbfValue)>,
}

impl RbfStructure {
    /// The first child structure named `name`, if any.
    pub fn child(&self, name: &str) -> Option<&RbfStructure> {
        self.children.iter().find_map(|(n, v)| {
            if n == name {
                if let RbfValue::Structure(s) = v {
                    return Some(s);
                }
            }
            None
        })
    }

    /// Every child structure named `name`, in order.
    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a RbfStructure> {
        self.children.iter().filter_map(move |(n, v)| {
            if n == name {
                if let RbfValue::Structure(s) = v {
                    return Some(s);
                }
            }
            None
        })
    }

    /// This structure's first child, if it is a raw bytes blob, decoded as
    /// NUL-terminated/padded ASCII with the NULs stripped. This is the shape
    /// `gtxd.ymt`'s `<parent>`/`<child>` structures use for their string.
    pub fn first_bytes_as_string(&self) -> Option<String> {
        match self.children.first() {
            Some((_, RbfValue::Bytes(b))) => {
                let s = String::from_utf8_lossy(b);
                Some(s.trim_end_matches('\0').to_string())
            }
            _ => None,
        }
    }
}

struct Descriptor {
    name: String,
    /// The type the descriptor was first defined with. Kept for parity with
    /// CodeWalker's `RbfEntryDescription`; a back-reference's own type byte
    /// is what actually drives parsing (see the tolerated-mismatch case
    /// below), so this is otherwise unread.
    #[allow(dead_code)]
    kind: u8,
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("rbf: offset overflow")?;
        let bytes = self.data.get(self.pos..end).context("rbf: truncated")?;
        self.pos = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn ascii(&mut self, len: usize) -> Result<String> {
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }

    fn at_end(&self) -> bool {
        self.pos >= self.data.len()
    }
}

/// True when `data` starts with the RBF0 magic.
pub fn is_rbf(data: &[u8]) -> bool {
    data.starts_with(&RBF_MAGIC)
}

/// Parses an RBF0 byte stream into its root structure.
pub fn parse(data: &[u8]) -> Result<RbfStructure> {
    if !is_rbf(data) {
        bail!("rbf: not an RBF0 stream");
    }

    let mut cursor = Cursor { data, pos: 4 };
    let mut descriptors: Vec<Descriptor> = Vec::new();
    // The open-element stack: each entry is a structure still being filled
    // in, along with how many of its next values are attributes rather than
    // children (CodeWalker `RbfStructure.PendingAttributes`).
    let mut stack: Vec<(RbfStructure, i16)> = Vec::new();
    let mut root: Option<RbfStructure> = None;

    while !cursor.at_end() {
        let tag = cursor.u8()?;

        if tag == 0xFF {
            let b = cursor.u8()?;
            if b != 0xFF {
                bail!("rbf: malformed close tag");
            }
            let Some((finished, _)) = stack.pop() else {
                bail!("rbf: unmatched close tag");
            };
            match stack.last_mut() {
                Some((parent, pending)) => add_value(parent, pending, finished.name.clone(), RbfValue::Structure(finished)),
                None => {
                    root = Some(finished);
                    break; // Root closed; ignore any trailing padding.
                }
            }
            continue;
        }

        if tag == 0xFD {
            let b = cursor.u8()?;
            if b != 0xFF {
                bail!("rbf: malformed bytes tag");
            }
            let len = cursor.u32()? as usize;
            let bytes = cursor.take(len)?.to_vec();
            // A raw bytes value always lands in `children`, bypassing the
            // attribute split entirely (CodeWalker `Rbf.cs`: `Children.Add`,
            // not the shared `AddChild` helper that consumes
            // `PendingAttributes`).
            let (current, _) = stack.last_mut().context("rbf: bytes value with no open element")?;
            current.children.push((String::new(), RbfValue::Bytes(bytes)));
            continue;
        }

        let descriptor_index = tag as usize;
        let data_type = cursor.u8()?;

        let name = if descriptor_index == descriptors.len() {
            let name_len = cursor.i16()?.max(0) as usize;
            let name = cursor.ascii(name_len)?;
            descriptors.push(Descriptor { name: name.clone(), kind: data_type });
            name
        } else {
            let descriptor = descriptors.get(descriptor_index).context("rbf: descriptor index out of range")?;
            // A back-reference's own type byte drives how the payload is
            // read; a mismatch against the descriptor's recorded type is
            // tolerated (CodeWalker's equivalent check is commented out).
            descriptor.name.clone()
        };

        match data_type {
            0x00 => {
                let _x1 = cursor.i16()?;
                let _x2 = cursor.i16()?;
                let pending_attributes = cursor.i16()?;
                stack.push((RbfStructure { name, ..Default::default() }, pending_attributes));
            }
            0x10 => {
                let value = RbfValue::Uint(cursor.u32()?);
                let (current, pending) = stack.last_mut().context("rbf: value with no open element")?;
                add_value(current, pending, name, value);
            }
            0x20 => {
                let (current, pending) = stack.last_mut().context("rbf: value with no open element")?;
                add_value(current, pending, name, RbfValue::Bool(true));
            }
            0x30 => {
                let (current, pending) = stack.last_mut().context("rbf: value with no open element")?;
                add_value(current, pending, name, RbfValue::Bool(false));
            }
            0x40 => {
                let value = RbfValue::Float(cursor.f32()?);
                let (current, pending) = stack.last_mut().context("rbf: value with no open element")?;
                add_value(current, pending, name, value);
            }
            0x50 => {
                let value = RbfValue::Float3([cursor.f32()?, cursor.f32()?, cursor.f32()?]);
                let (current, pending) = stack.last_mut().context("rbf: value with no open element")?;
                add_value(current, pending, name, value);
            }
            0x60 => {
                let len = cursor.i16()?.max(0) as usize;
                let value = RbfValue::Str(cursor.ascii(len)?);
                let (current, pending) = stack.last_mut().context("rbf: value with no open element")?;
                add_value(current, pending, name, value);
            }
            other => bail!("rbf: unsupported data type 0x{other:02X}"),
        }
    }

    root.context("rbf: stream ended before the root element closed")
}

/// Adds `value` to `current`'s attributes or children, consuming one unit of
/// `pending` when it is positive (CodeWalker `RbfStructure.AddChild`).
fn add_value(current: &mut RbfStructure, pending: &mut i16, name: String, value: RbfValue) {
    if *pending > 0 {
        *pending -= 1;
        current.attributes.push((name, value));
    } else {
        current.children.push((name, value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(idx: u8, name: Option<&str>, pending_attrs: i16) -> Vec<u8> {
        let mut v = vec![idx, 0x00];
        if let Some(n) = name {
            v.extend_from_slice(&(n.len() as i16).to_le_bytes());
            v.extend_from_slice(n.as_bytes());
        }
        v.extend_from_slice(&0i16.to_le_bytes()); // x1, unused
        v.extend_from_slice(&0i16.to_le_bytes()); // x2, unused
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

    fn uint_node(idx: u8, name: Option<&str>, value: u32) -> Vec<u8> {
        let mut v = vec![idx, 0x10];
        if let Some(n) = name {
            v.extend_from_slice(&(n.len() as i16).to_le_bytes());
            v.extend_from_slice(n.as_bytes());
        }
        v.extend_from_slice(&value.to_le_bytes());
        v
    }

    fn float_node(idx: u8, name: Option<&str>, value: f32) -> Vec<u8> {
        let mut v = vec![idx, 0x40];
        if let Some(n) = name {
            v.extend_from_slice(&(n.len() as i16).to_le_bytes());
            v.extend_from_slice(n.as_bytes());
        }
        v.extend_from_slice(&value.to_le_bytes());
        v
    }

    fn float3_node(idx: u8, name: Option<&str>, value: [f32; 3]) -> Vec<u8> {
        let mut v = vec![idx, 0x50];
        if let Some(n) = name {
            v.extend_from_slice(&(n.len() as i16).to_le_bytes());
            v.extend_from_slice(n.as_bytes());
        }
        for f in value {
            v.extend_from_slice(&f.to_le_bytes());
        }
        v
    }

    fn bool_node(idx: u8, name: Option<&str>, value: bool) -> Vec<u8> {
        vec![idx, if value { 0x20 } else { 0x30 }]
            .into_iter()
            .chain(name.map(|n| {
                let mut v = (n.len() as i16).to_le_bytes().to_vec();
                v.extend_from_slice(n.as_bytes());
                v
            }).unwrap_or_default())
            .collect()
    }

    fn string_node(idx: u8, name: Option<&str>, value: &str) -> Vec<u8> {
        let mut v = vec![idx, 0x60];
        if let Some(n) = name {
            v.extend_from_slice(&(n.len() as i16).to_le_bytes());
            v.extend_from_slice(n.as_bytes());
        }
        v.extend_from_slice(&(value.len() as i16).to_le_bytes());
        v.extend_from_slice(value.as_bytes());
        v
    }

    /// The gtxd shape: `CMapParentTxds/txdRelationships/item/{parent,child}`,
    /// each `parent`/`child` a structure whose only child is a bytes node.
    /// Descriptors are assigned in first-use order across the whole stream:
    /// 0=CMapParentTxds, 1=txdRelationships, 2=item, 3=parent, 4=child — so
    /// every item after the first uses back-references.
    fn build_gtxd_rbf(pairs: &[(&str, &str)]) -> Vec<u8> {
        let mut d = RBF_MAGIC.to_vec();
        d.extend(open(0, Some("CMapParentTxds"), 0));
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

            d.extend(close()); // </item>
        }

        d.extend(close()); // </txdRelationships>
        d.extend(close()); // </CMapParentTxds>
        d
    }

    fn extract_pairs(root: &RbfStructure) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let Some(rels) = root.child("txdRelationships") else { return out };
        for item in rels.children_named("item") {
            let Some(parent) = item.child("parent").and_then(|s| s.first_bytes_as_string()) else { continue };
            let Some(child) = item.child("child").and_then(|s| s.first_bytes_as_string()) else { continue };
            out.push((child, parent));
        }
        out
    }

    #[test]
    fn parses_gtxd_relationships_including_back_references() {
        let data = build_gtxd_rbf(&[("child_a", "parent_a"), ("child_b", "parent_b")]);
        let root = parse(&data).expect("should parse");
        assert_eq!(root.name, "CMapParentTxds");
        assert_eq!(
            extract_pairs(&root),
            vec![("child_a".to_string(), "parent_a".to_string()), ("child_b".to_string(), "parent_b".to_string())]
        );
    }

    #[test]
    fn ignores_unrelated_value_nodes_spliced_into_an_item() {
        // Same shape as build_gtxd_rbf's first item, but with extra value
        // nodes of every other type inserted inside <item> before <parent>.
        let mut d = RBF_MAGIC.to_vec();
        d.extend(open(0, Some("CMapParentTxds"), 0));
        d.extend(open(1, Some("txdRelationships"), 0));
        d.extend(open(2, Some("item"), 0));
        d.extend(uint_node(3, Some("someUint"), 42));
        d.extend(float_node(4, Some("someFloat"), 1.5));
        d.extend(float3_node(5, Some("someFloat3"), [1.0, 2.0, 3.0]));
        d.extend(bool_node(6, Some("someBool"), true));
        d.extend(bool_node(7, Some("someOtherBool"), false));
        d.extend(string_node(8, Some("someString"), "hello"));
        d.extend(open(9, Some("parent"), 0));
        d.extend(bytes_node("vehshare"));
        d.extend(close());
        d.extend(open(10, Some("child"), 0));
        d.extend(bytes_node("some_child_txd"));
        d.extend(close());
        d.extend(close()); // </item>
        d.extend(close()); // </txdRelationships>
        d.extend(close()); // </CMapParentTxds>

        let root = parse(&d).expect("should parse despite unrelated nodes");
        assert_eq!(extract_pairs(&root), vec![("some_child_txd".to_string(), "vehshare".to_string())]);
    }

    #[test]
    fn pending_attributes_are_not_children() {
        // <parent pendingAttrs=1> — the first value added is an attribute,
        // so the bytes node (added second) must still land at children[0].
        let mut d = RBF_MAGIC.to_vec();
        d.extend(open(0, Some("CMapParentTxds"), 0));
        d.extend(open(1, Some("txdRelationships"), 0));
        d.extend(open(2, Some("item"), 0));
        d.extend(open(3, Some("parent"), 1)); // 1 pending attribute
        d.extend(uint_node(4, Some("attr"), 7)); // consumed as an attribute
        d.extend(bytes_node("vehshare")); // 0xFD always bypasses the split
        d.extend(close());
        d.extend(open(5, Some("child"), 0));
        d.extend(bytes_node("some_child_txd"));
        d.extend(close());
        d.extend(close());
        d.extend(close());
        d.extend(close());

        let root = parse(&d).expect("should parse");
        let parent_struct = root.child("txdRelationships").unwrap().child("item").unwrap().child("parent").unwrap();
        assert_eq!(parent_struct.attributes.len(), 1);
        assert_eq!(parent_struct.first_bytes_as_string(), Some("vehshare".to_string()));
    }

    #[test]
    fn back_reference_type_mismatch_is_tolerated() {
        // Descriptor 5 is first defined as a uint (0x10); a later record
        // reuses index 5 but claims type 0x40 (float). The record's own
        // type must win, matching CodeWalker's commented-out check.
        let mut d = RBF_MAGIC.to_vec();
        d.extend(open(0, Some("CMapParentTxds"), 0));
        d.extend(uint_node(1, Some("x"), 42));
        d.extend(float_node(1, None, 1.0)); // back-reference, mismatched type
        d.extend(close());

        let root = parse(&d).expect("should tolerate a mismatched back-reference type");
        assert_eq!(root.children.len(), 2);
        assert!(matches!(root.children[0].1, RbfValue::Uint(42)));
        assert!(matches!(root.children[1].1, RbfValue::Float(f) if f == 1.0));
    }

    #[test]
    fn root_other_than_cmapparenttxds_still_parses() {
        let mut d = RBF_MAGIC.to_vec();
        d.extend(open(0, Some("SomethingElse"), 0));
        d.extend(close());
        let root = parse(&d).expect("should parse");
        assert_eq!(root.name, "SomethingElse");
        // The gtxd-specific caller is responsible for checking the root
        // name and yielding no relationships; the reader itself is generic.
    }

    #[test]
    fn trailing_padding_after_root_close_is_ignored() {
        let mut d = RBF_MAGIC.to_vec();
        d.extend(open(0, Some("CMapParentTxds"), 0));
        d.extend(close());
        d.extend(vec![0u8; 37]); // arbitrary page padding
        let root = parse(&d).expect("trailing bytes must not be an error");
        assert_eq!(root.name, "CMapParentTxds");
    }

    #[test]
    fn truncated_input_never_panics() {
        let full = build_gtxd_rbf(&[("child_a", "parent_a"), ("child_b", "parent_b")]);
        for n in 0..full.len() {
            let _ = parse(&full[..n]); // must return Err, not panic
        }
    }

    #[test]
    fn rejects_missing_magic() {
        assert!(parse(b"not rbf data at all").is_err());
    }
}
