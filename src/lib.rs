pub mod archive;
pub mod crypto;
pub mod tree;
pub mod writer;
pub mod dlc;
mod tests;

pub use archive::{RpfArchive, RpfEntry, RpfEntryKind, RpfEncryption, RpfFile, RpfVersion,
                  resource_size_from_flags, resource_version_from_flags,
                  RPF0_MAGIC, RPF2_MAGIC, RPF3_MAGIC, RPF4_MAGIC, RPF6_MAGIC,
                  RPF7_MAGIC, RPF8_MAGIC, RSC7_MAGIC, RSC8_MAGIC, IMG2_MAGIC, IMG3_MAGIC};
pub use crypto::keys::GtaKeys;
pub use tree::{DirNode, FileRef, build_directory_tree, list_all_files};
pub use writer::{RpfBuilder, rage_joaat};
pub use dlc::{parse_dlc_list, parse_dlc_setup_order};
