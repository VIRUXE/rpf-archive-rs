use anyhow::{Result, Context, bail};
use crate::resource::{prepare_rsc7, u16_le, u32_le, u64_le};

// ─── Meta Parser Structures ───────────────────────────────────────────────────

#[derive(Debug)]
pub struct MetaData {
    pub blocks: Vec<MetaBlock>,
    pub root_block_index: i32,
}

#[derive(Debug)]
pub struct MetaBlock {
    pub name_hash: u32,
    pub data_ptr: u64,
    pub length: u32,
    pub data: Vec<u8>,
}

#[derive(Debug, Copy, Clone)]
pub struct MetaPointer {
    pub block_id: usize,
    pub offset: usize,
}

impl MetaPointer {
    pub fn from_u64(val: u64) -> Self {
        Self {
            block_id: (val & 0xFFF) as usize,
            offset: ((val >> 12) & 0xFFFFF) as usize,
        }
    }
}

// ─── Structs ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct PedVariationInfo {
    pub has_tex_variations: bool,
    pub has_drawbl_variations: bool,
    pub has_low_lods: bool,
    pub is_super_lod: bool,
    pub avail_comp: [u8; 12],
    pub component_data: Vec<ComponentData>,
}

#[derive(Debug)]
pub struct ComponentData {
    pub num_avail_tex: u8,
    pub drawables: Vec<DrawableData>,
    pub block_id: usize,
    pub offset: usize,
}

#[derive(Debug)]
pub struct DrawableData {
    pub prop_mask: u8,
    pub num_alternatives: u8,
    pub textures: Vec<TextureData>,
    pub block_id: usize,
    pub offset: usize,
}

#[derive(Debug)]
pub struct TextureData {
    pub tex_id: u32,
    pub distribution: u8,
}

// ─── Implementation ───────────────────────────────────────────────────────────

pub fn parse_ymt(data: &[u8]) -> Result<(PedVariationInfo, Vec<u8>, Vec<u8>)> {
    let (system, graphics) = prepare_rsc7(data)?;
    
    // Meta block is 112 bytes long starting at 0
    if system.len() < 112 {
        bail!("System section too small for Meta header");
    }
    
    let root_block_index = i32::from_le_bytes(system[28..32].try_into().unwrap());
    let data_blocks_pointer = u64_le(&system, 48);
    let data_blocks_count = i16::from_le_bytes(system[76..78].try_into().unwrap()) as usize;
    
    let mut blocks = Vec::with_capacity(data_blocks_count);
    
    if data_blocks_count > 0 && data_blocks_pointer != 0 {
        let block_array_offset = (data_blocks_pointer & 0xFFFFFFF) as usize; 
        
        for i in 0..data_blocks_count {
            let offset = block_array_offset + (i * 16);
            let name_hash = u32_le(&system, offset);
            let length = u32_le(&system, offset + 4);
            let data_ptr = u64_le(&system, offset + 8);
            
            let data_offset = (data_ptr & 0xFFFFFFF) as usize;
            
            let mut block_data = Vec::new();
            if data_offset > 0 && data_offset + length as usize <= system.len() {
                block_data = system[data_offset..data_offset + length as usize].to_vec();
            }
            
            blocks.push(MetaBlock {
                name_hash,
                data_ptr,
                length,
                data: block_data
            });
        }
    } else {
        bail!("Invalid Meta header: no data blocks");
    }
    
    let meta = MetaData { blocks, root_block_index };
    
    // Find the CPedVariationInfo block (hash 0x16760659 = 376833625)
    let root_block_opt = meta.blocks.iter().find(|b| b.name_hash == 0x16760659);
    
    let root_block_container = root_block_opt.context("CPedVariationInfo block not found")?;
    let root_block = &root_block_container.data;
    
    if root_block.len() < 112 {
        bail!("Root block too small for CPedVariationInfo");
    }
    
    let has_tex_variations = root_block[0] != 0;
    let has_drawbl_variations = root_block[1] != 0;
    let has_low_lods = root_block[2] != 0;
    let is_super_lod = root_block[3] != 0;
    
    let mut avail_comp = [0u8; 12];
    avail_comp.copy_from_slice(&root_block[4..16]);
    
    let comp_data_ptr = MetaPointer::from_u64(u64_le(root_block, 16));
    let comp_data_count = u16_le(root_block, 24) as usize;
    
    let mut component_data = vec![];
    if comp_data_count > 0 && comp_data_ptr.block_id > 0 {
        let block_idx = comp_data_ptr.block_id - 1; // 1-based index
        let comp_block = meta.blocks.get(block_idx).context("Component block not found")?;
        
        for i in 0..comp_data_count {
            let offset = comp_data_ptr.offset + (i * 24);
            if offset + 24 > comp_block.data.len() { continue; }
            let num_avail_tex = comp_block.data[offset];
            
            let drawables_ptr = MetaPointer::from_u64(u64_le(&comp_block.data, offset + 8));
            let drawables_count = u16_le(&comp_block.data, offset + 16) as usize;
            
            let mut drawables = vec![];
            if drawables_count > 0 && drawables_ptr.block_id > 0 {
                let d_block_idx = drawables_ptr.block_id - 1;
                if let Some(draw_block) = meta.blocks.get(d_block_idx) {
                    for j in 0..drawables_count {
                        let d_offset = drawables_ptr.offset + (j * 48);
                        if d_offset + 48 > draw_block.data.len() { continue; }
                        let prop_mask = draw_block.data[d_offset];
                        let num_alternatives = draw_block.data[d_offset + 1];
                        
                        let tex_data_ptr = MetaPointer::from_u64(u64_le(&draw_block.data, d_offset + 8));
                        let tex_data_count = u16_le(&draw_block.data, d_offset + 16) as usize;
                        
                        let mut textures = vec![];
                        if tex_data_count > 0 && tex_data_ptr.block_id > 0 {
                            let t_block_idx = tex_data_ptr.block_id - 1;
                            if let Some(tex_block) = meta.blocks.get(t_block_idx) {
                                for k in 0..tex_data_count {
                                    let t_offset = tex_data_ptr.offset + (k * 8);
                                    if t_offset + 8 > tex_block.data.len() { continue; }
                                    let tex_id = u32_le(&tex_block.data, t_offset);
                                    let distribution = tex_block.data[t_offset + 4];
                                    textures.push(TextureData { tex_id, distribution });
                                }
                            }
                        }
                        
                        drawables.push(DrawableData { 
                            prop_mask, 
                            num_alternatives, 
                            textures,
                            block_id: d_block_idx,
                            offset: d_offset
                        });
                    }
                }
            }
            
            component_data.push(ComponentData { num_avail_tex, drawables, block_id: block_idx, offset });
        }
    }

    
    Ok((PedVariationInfo {
        has_tex_variations,
        has_drawbl_variations,
        has_low_lods,
        is_super_lod,
        avail_comp,
        component_data,
    }, system, graphics))
}
