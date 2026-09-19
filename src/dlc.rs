// Parses the two files CodeWalker's `GameFileCache.InitDlcList` (in
// `GameFileCache.cs`) uses to compute the game's own DLC load order:
//
//   - `update/update.rpf`'s `common/data/dlclist.xml` — the ordered list of
//     DLC path entries the game itself mounts. Most are `dlcpacks:/<name>/`;
//     a handful of older ones are `platform:/dlcPacks/<Name>/`, which
//     CodeWalker remaps to a path under `x64...` rather than `dlcpacks` (see
//     `InitDlcList`'s `Replace("platform:", "x64")`) — those never resolve
//     to a `dlcpacks/<name>/dlc.rpf` on disk and are dropped here rather
//     than followed, since nothing in `rpf-cli`'s indexing walks `x64...`
//     DLC mounts separately from the base archives it already scans.
//   - each pack's own `setup2.xml`, whose `<order value="N"/>` is what
//     `DlcSetupFiles.OrderBy(o => o.order)` (a *stable* sort) ranks by.
//     `dlclist.xml` position is the tie-break for equal `order` values,
//     and also the only ordering available for a pack whose `setup2.xml`
//     is missing or malformed (`order` defaults to `-1`, matching the C#
//     default `int` and `GetIntAttribute`'s behaviour on a missing node).
//
// Unlike `gtxd.rs`'s two files, CodeWalker loads both of these purely as
// XML (`RpfMan.GetFileXml`) — no RBF/RSC7 form exists for either, so this
// module skips that dispatch entirely.

use anyhow::{Context, Result};

/// Every `dlcpacks:/<name>/` entry from `dlclist.xml`'s `<Paths>`, lowercased
/// and in file order — the order `DlcSetupFiles` falls back to (via a stable
/// sort) when two packs declare the same `setup2.xml` `order`. Entries under
/// `platform:/dlcPacks/...` are legacy packs baked into `update.rpf` itself
/// rather than a `dlcpacks/<name>/dlc.rpf` on disk, and are skipped.
pub fn parse_dlc_list(data: &[u8]) -> Result<Vec<String>> {
    let text = std::str::from_utf8(strip_bom(data)).context("dlclist: not UTF-8 XML")?;
    let doc = roxmltree::Document::parse(text).context("dlclist: failed to parse XML")?;
    let root = doc.root_element();

    if root.tag_name().name() != "SMandatoryPacksData" {
        return Ok(Vec::new());
    }
    let Some(paths) = root.children().find(|n| n.is_element() && n.tag_name().name() == "Paths") else {
        return Ok(Vec::new());
    };

    let mut out = Vec::new();
    for item in paths.children().filter(|n| n.is_element() && n.tag_name().name() == "Item") {
        let text = item.text().unwrap_or("").trim();
        let lower = text.to_lowercase();
        let Some(rest) = lower.strip_prefix("dlcpacks:/") else { continue };
        let name = rest.trim_matches('/');
        if !name.is_empty() {
            out.push(name.to_string());
        }
    }
    Ok(out)
}

/// A pack's `setup2.xml` `<order value="N"/>`, matching
/// `DlcSetupFile.order`/`Xml.GetIntAttribute`: missing element, missing
/// `value` attribute, or a value that doesn't parse as an integer all
/// default to `-1` rather than an error (a pack that ships a bad or
/// nonexistent `<order>` still needs *a* rank, and the game itself treats
/// an unset C# `int` the same way — its default is `0`, but the field is
/// only ever set here, never left at its bare default, so `-1` is used as
/// this crate's own explicit "unranked" sentinel instead).
pub fn parse_dlc_setup_order(data: &[u8]) -> Result<i32> {
    let text = std::str::from_utf8(strip_bom(data)).context("setup2: not UTF-8 XML")?;
    let doc = roxmltree::Document::parse(text).context("setup2: failed to parse XML")?;
    let root = doc.root_element();

    if root.tag_name().name() != "SSetupData" {
        return Ok(-1);
    }
    let order = root
        .children()
        .find(|n| n.is_element() && n.tag_name().name() == "order")
        .and_then(|n| n.attribute("value"))
        .and_then(|v| v.trim().parse::<i32>().ok())
        .unwrap_or(-1);
    Ok(order)
}

fn strip_bom(data: &[u8]) -> &[u8] {
    data.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dlclist_paths_in_order() {
        let xml = "\u{FEFF}<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                   <SMandatoryPacksData>\n\
                   <Paths>\n\
                   <Item>platform:/dlcPacks/mpBeach/</Item>\n\
                   <Item>dlcpacks:/mpHeist/</Item>\n\
                   <Item>dlcpacks:/patchday1ng/</Item>\n\
                   </Paths>\n\
                   </SMandatoryPacksData>";
        let names = parse_dlc_list(xml.as_bytes()).expect("should parse");
        assert_eq!(names, vec!["mpheist".to_string(), "patchday1ng".to_string()]);
    }

    #[test]
    fn dlclist_unknown_root_yields_nothing() {
        let xml = "<SomethingElse><Paths><Item>dlcpacks:/mpheist/</Item></Paths></SomethingElse>";
        assert!(parse_dlc_list(xml.as_bytes()).expect("should parse").is_empty());
    }

    #[test]
    fn dlclist_missing_paths_yields_nothing() {
        let xml = "<SMandatoryPacksData></SMandatoryPacksData>";
        assert!(parse_dlc_list(xml.as_bytes()).expect("should parse").is_empty());
    }

    #[test]
    fn parses_setup2_order() {
        let xml = "\u{FEFF}<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                   <SSetupData>\n\
                   <deviceName>dlc_mpHeist</deviceName>\n\
                   <order value=\"10\" />\n\
                   <minorOrder value=\"0\" />\n\
                   </SSetupData>";
        assert_eq!(parse_dlc_setup_order(xml.as_bytes()).expect("should parse"), 10);
    }

    #[test]
    fn setup2_negative_order_is_kept() {
        let xml = "<SSetupData><order value=\"-1\" /></SSetupData>";
        assert_eq!(parse_dlc_setup_order(xml.as_bytes()).expect("should parse"), -1);
    }

    #[test]
    fn setup2_missing_order_element_defaults_to_negative_one() {
        let xml = "<SSetupData><deviceName>x</deviceName></SSetupData>";
        assert_eq!(parse_dlc_setup_order(xml.as_bytes()).expect("should parse"), -1);
    }

    #[test]
    fn setup2_unparsable_order_value_defaults_to_negative_one() {
        let xml = "<SSetupData><order value=\"not-a-number\" /></SSetupData>";
        assert_eq!(parse_dlc_setup_order(xml.as_bytes()).expect("should parse"), -1);
    }

    #[test]
    fn setup2_unknown_root_defaults_to_negative_one() {
        let xml = "<SomethingElse><order value=\"5\" /></SomethingElse>";
        assert_eq!(parse_dlc_setup_order(xml.as_bytes()).expect("should parse"), -1);
    }

    #[test]
    fn rejects_data_that_is_not_utf8() {
        let data = vec![0xFF, 0xFE, 0x00, 0x01, 0x02];
        assert!(parse_dlc_list(&data).is_err());
        assert!(parse_dlc_setup_order(&data).is_err());
    }
}
