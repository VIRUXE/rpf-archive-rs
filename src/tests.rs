#[cfg(test)]
mod writer_tests {
    use crate::archive::{RpfArchive, RpfEncryption, RpfVersion};
    use crate::writer::RpfBuilder;

    // Flat files only (no subdirs) — IMG1/2 are flat formats
    const FLAT_FILES: &[(&str, &[u8])] = &[
        ("hello.txt",  b"Hello, world!"),
        ("data.bin",   &[0xDE, 0xAD, 0xBE, 0xEF]),
        ("deep.bin",   b"deep file content here"),
    ];

    const FILES: &[(&str, &[u8])] = &[
        ("hello.txt",              b"Hello, world!"),
        ("subdir/data.bin",        &[0xDE, 0xAD, 0xBE, 0xEF]),
        ("subdir/nested/deep.bin", b"deep file content here"),
    ];

    fn roundtrip_version(version: RpfVersion, archive_name: &str) {
        let mut builder = RpfBuilder::for_version(version, RpfEncryption::None);
        for (path, data) in FILES {
            builder.add_file(path, data.to_vec());
        }

        let bytes = builder.build(None).expect("build failed");
        let archive = RpfArchive::parse(&bytes, archive_name, None).expect("parse failed");

        assert_eq!(
            archive.entries.iter().filter(|e| e.is_file()).count(),
            FILES.len(),
            "{archive_name}: wrong file count"
        );

        // For V3 names are hashes — skip name checks, just verify count and extraction
        if version != RpfVersion::V3 {
            let names: Vec<&str> = archive.entries.iter().map(|e| e.name.as_str()).collect();
            assert!(names.contains(&"hello.txt"),  "{archive_name}: missing hello.txt");
            assert!(names.contains(&"data.bin"),   "{archive_name}: missing data.bin");
            assert!(names.contains(&"deep.bin"),   "{archive_name}: missing deep.bin");
        }

        // Verify extraction of every file
        for (path, expected) in FILES {
            let fname = std::path::Path::new(path).file_name().unwrap().to_str().unwrap();
            let entry = archive.entries.iter().find(|e| {
                if version == RpfVersion::V3 { e.is_file() && true } // just pick any file
                else { e.name == fname }
            });
            if version != RpfVersion::V3 {
                let entry = entry.expect(&format!("{archive_name}: entry {fname} not found"));
                let extracted = archive.extract_entry(&bytes, entry, None)
                    .expect(&format!("{archive_name}: extract {fname} failed"));
                assert_eq!(extracted.as_slice(), *expected,
                    "{archive_name}: content mismatch for {fname}");
            }
        }
    }

    fn roundtrip_v3_extraction(archive_name: &str) {
        let mut builder = RpfBuilder::for_version(RpfVersion::V3, RpfEncryption::None);
        for (path, data) in FILES {
            builder.add_file(path, data.to_vec());
        }
        let bytes = builder.build(None).expect("build failed");
        let archive = RpfArchive::parse(&bytes, archive_name, None).expect("parse failed");

        // Extract each file by position (names are hashes in V3)
        let file_entries: Vec<_> = archive.entries.iter().filter(|e| e.is_file()).collect();
        assert_eq!(file_entries.len(), FILES.len());
        for (i, (_, expected)) in FILES.iter().enumerate() {
            let extracted = archive.extract_entry(&bytes, file_entries[i], None)
                .expect("V3 extract failed");
            assert_eq!(extracted.as_slice(), *expected, "V3 content mismatch at index {i}");
        }
    }

    #[test]
    fn roundtrip_v0()   { roundtrip_version(RpfVersion::V0,   "test.rpf"); }

    #[test]
    fn roundtrip_v2()   { roundtrip_version(RpfVersion::V2,   "test.rpf"); }

    #[test]
    fn roundtrip_v3()   { roundtrip_v3_extraction("test.rpf"); }

    #[test]
    fn roundtrip_v4()   { roundtrip_version(RpfVersion::V4,   "test.rpf"); }

    #[test]
    fn roundtrip_v6()   { roundtrip_version(RpfVersion::V6,   "test.rpf"); }

    #[test]
    fn roundtrip_img3() { roundtrip_version(RpfVersion::Img3, "test.img"); }

    #[test]
    fn roundtrip_v7_open() {
        let mut builder = RpfBuilder::new(RpfEncryption::Open);
        for (path, data) in FILES {
            builder.add_file(path, data.to_vec());
        }
        let bytes = builder.build(None).expect("build failed");
        let archive = RpfArchive::parse(&bytes, "test.rpf", None).expect("parse failed");
        assert_eq!(archive.entries.iter().filter(|e| e.is_file()).count(), FILES.len());
        let entry = archive.entries.iter().find(|e| e.name == "hello.txt").unwrap();
        let extracted = archive.extract_entry(&bytes, entry, None).unwrap();
        assert_eq!(extracted.as_slice(), b"Hello, world!");
    }

    #[test]
    fn roundtrip_v7_none() {
        let mut builder = RpfBuilder::new(RpfEncryption::None);
        for (path, data) in FILES {
            builder.add_file(path, data.to_vec());
        }
        let bytes = builder.build(None).expect("build failed");
        let archive = RpfArchive::parse(&bytes, "test.rpf", None).expect("parse failed");
        assert_eq!(archive.entries.iter().filter(|e| e.is_file()).count(), FILES.len());
    }

    // Past 16 MiB the V7 entry's 24-bit size field can no longer hold the length.
    const OVER_24_BITS: usize = 0x0100_0000 + 4096;

    fn patterned(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn v7_stored_binary_has_zero_disk_size() {
        let mut builder = RpfBuilder::new(RpfEncryption::Open);
        builder.add_file("hello.txt", b"Hello, world!".to_vec());
        let bytes = builder.build(None).expect("build failed");
        let archive = RpfArchive::parse(&bytes, "test.rpf", None).expect("parse failed");
        let entry = archive.entries.iter().find(|e| e.name == "hello.txt").unwrap();
        match entry.kind {
            crate::archive::RpfEntryKind::BinaryFile { file_size, uncompressed_size, .. } => {
                assert_eq!(file_size, 0, "stored entry must have on-disk size 0 (non-zero means compressed)");
                assert_eq!(uncompressed_size, 13);
            }
            _ => panic!("expected a binary entry"),
        }
    }

    #[test]
    fn v7_binary_over_16mib_roundtrips() {
        let data = patterned(OVER_24_BITS);
        let mut builder = RpfBuilder::new(RpfEncryption::Open);
        builder.add_file("big/nested.rpf", data.clone());
        let bytes = builder.build(None).expect("build failed");
        let archive = RpfArchive::parse(&bytes, "test.rpf", None).expect("parse failed");
        let entry = archive.entries.iter().find(|e| e.name == "nested.rpf").unwrap();
        let extracted = archive.extract_entry(&bytes, entry, None).expect("extract failed");
        assert!(extracted == data, "content mismatch for a >16 MiB stored binary");
    }

    #[test]
    fn v7_resource_over_16mib_roundtrips() {
        let (sys, gfx) = (0x0000_0100u32, 0x0000_0200u32);
        let body = patterned(OVER_24_BITS);
        let mut data = Vec::with_capacity(16 + body.len());
        data.extend_from_slice(&crate::archive::RSC7_MAGIC.to_le_bytes());
        data.extend_from_slice(&13u32.to_le_bytes());
        data.extend_from_slice(&sys.to_le_bytes());
        data.extend_from_slice(&gfx.to_le_bytes());
        data.extend_from_slice(&body);

        let mut builder = RpfBuilder::new(RpfEncryption::Open);
        builder.add_file("x64/big.ytd", data);
        let bytes = builder.build(None).expect("build failed");
        let archive = RpfArchive::parse(&bytes, "test.rpf", None).expect("parse failed");
        let entry = archive.entries.iter().find(|e| e.name == "big.ytd").unwrap();
        let extracted = archive.extract_entry(&bytes, entry, None).expect("extract failed");
        assert_eq!(&extracted[8..16], &[sys.to_le_bytes(), gfx.to_le_bytes()].concat()[..]);
        assert!(extracted[16..] == body[..], "body mismatch for a >16 MiB resource");
    }

    #[test]
    fn empty_archive() {
        let builder = RpfBuilder::new(RpfEncryption::Open);
        let bytes = builder.build(None).expect("build failed");
        let archive = RpfArchive::parse(&bytes, "empty.rpf", None).expect("parse failed");
        assert_eq!(archive.entries.len(), 1); // root dir only
    }

    #[test]
    fn roundtrip_img2() {
        let mut builder = RpfBuilder::for_version(RpfVersion::Img2, RpfEncryption::None);
        for (path, data) in FLAT_FILES {
            builder.add_file(path, data.to_vec());
        }
        let bytes = builder.build(None).expect("build failed");
        let archive = RpfArchive::parse(&bytes, "test.img", None).expect("parse failed");

        assert_eq!(archive.entries.iter().filter(|e| e.is_file()).count(), FLAT_FILES.len());
        let names: Vec<&str> = archive.entries.iter().map(|e| e.name.as_str()).collect();
        for (fname, _) in FLAT_FILES {
            assert!(names.contains(fname), "img2: missing {fname}");
        }
        for (fname, expected) in FLAT_FILES {
            let entry = archive.entries.iter().find(|e| e.name.as_str() == *fname).unwrap();
            let extracted = archive.extract_entry(&bytes, entry, None)
                .expect(&format!("img2: extract {fname} failed"));
            // Extraction returns sector-padded data; check prefix matches
            assert!(extracted.starts_with(expected),
                "img2: content mismatch for {fname}");
        }
    }

    #[test]
    fn roundtrip_img1() {
        let mut builder = RpfBuilder::for_version(RpfVersion::Img1, RpfEncryption::None);
        for (path, data) in FLAT_FILES {
            builder.add_file(path, data.to_vec());
        }
        let (dir_data, img_data) = builder.build_img1_pair().expect("build_img1_pair failed");
        let archive = RpfArchive::parse_img1(&dir_data, "test.img").expect("parse_img1 failed");

        assert_eq!(archive.entries.len(), FLAT_FILES.len());
        let names: Vec<&str> = archive.entries.iter().map(|e| e.name.as_str()).collect();
        for (fname, _) in FLAT_FILES {
            assert!(names.contains(fname), "img1: missing {fname}");
        }
        for (fname, expected) in FLAT_FILES {
            let entry = archive.entries.iter().find(|e| e.name.as_str() == *fname).unwrap();
            let extracted = archive.extract_entry(&img_data, entry, None)
                .expect(&format!("img1: extract {fname} failed"));
            assert!(extracted.starts_with(expected),
                "img1: content mismatch for {fname}");
        }
    }
}
