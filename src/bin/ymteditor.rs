use std::env;
use std::fs;
use std::path::Path;
use rpf_archive::{parse_ymt, RSC7_MAGIC};
use anyhow::{Result, Context, bail};
use flate2::write::DeflateEncoder;
use flate2::Compression;
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: ymteditor <command> [options]");
        eprintln!("Commands:");
        eprintln!("  --dump <file.ymt>           Dump YMT content to JSON");
        eprintln!("  --remove <comp_idx> <draw_idx> <file> Remove entry (binary patch)");
        return Ok(());
    }

    let command = &args[1];
    match command.as_str() {
        "--dump" => {
            if args.len() < 3 {
                eprintln!("Usage: ymteditor --dump <file.ymt>");
                return Ok(());
            }
            dump_ymt(&args[2])?;
        }
        "--remove" => {
            if args.len() < 5 {
                eprintln!("Usage: ymteditor --remove <comp_idx> <draw_idx> <file.ymt>");
                return Ok(());
            }
            let comp_idx: usize = args[2].parse().context("Invalid component index")?;
            let draw_idx: usize = args[3].parse().context("Invalid drawable index")?;
            remove_entry(comp_idx, draw_idx, &args[4])?;
        }
        _ => {
            eprintln!("Unknown command: {}", command);
        }
    }

    Ok(())
}

fn dump_ymt(path: &str) -> Result<()> {
    let data = fs::read(path).with_context(|| format!("Failed to read {}", path))?;
    let (info, _, _) = parse_ymt(&data)?;
    
    let mut root = json::JsonValue::new_object();
    let mut components = json::JsonValue::new_array();
    for (i, comp) in info.component_data.iter().enumerate() {
        let mut c_obj = json::JsonValue::new_object();
        c_obj["component_index"] = i.into();
        let mut drawables = json::JsonValue::new_array();
        for (j, draw) in comp.drawables.iter().enumerate() {
            let mut d_obj = json::JsonValue::new_object();
            d_obj["index"] = j.into();
            d_obj["num_alternatives"] = draw.num_alternatives.into();
            d_obj["texture_count"] = draw.textures.len().into();
            drawables.push(d_obj).unwrap();
        }
        c_obj["drawables"] = drawables;
        components.push(c_obj).unwrap();
    }
    root["components"] = components;
    println!("{}", root.dump());
    Ok(())
}

fn remove_entry(comp_idx: usize, draw_idx: usize, path: &str) -> Result<()> {
    let data = fs::read(path).with_context(|| format!("Failed to read {}", path))?;
    
    // Copy original flags
    let sys_flags = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let gfx_flags = u32::from_le_bytes(data[12..16].try_into().unwrap());
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());

    let (info, mut system, mut graphics) = parse_ymt(&data)?;
    
    if comp_idx >= info.component_data.len() {
        bail!("Component index out of range");
    }
    let comp = &info.component_data[comp_idx];
    if draw_idx >= comp.drawables.len() {
        bail!("Drawable index out of range");
    }
    let drawable = &comp.drawables[draw_idx];
    
    let off = drawable.offset;
    if off + 48 > system.len() {
        bail!("Patch offset out of bounds");
    }
    
    // Patch: Zero out the drawable entry
    for i in 0..48 {
        system[off + i] = 0;
    }
    
    // Re-serialize back to RSC7
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&system)?;
    encoder.write_all(&graphics)?;
    let compressed = encoder.finish()?;
    
    let mut out = Vec::new();
    out.extend_from_slice(&RSC7_MAGIC.to_le_bytes());
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&sys_flags.to_le_bytes());
    out.extend_from_slice(&gfx_flags.to_le_bytes());
    out.extend_from_slice(&compressed);
    
    fs::write(path, out)?;
    println!("Successfully patched {} to remove component {} drawable {}", path, comp_idx, draw_idx);
    
    Ok(())
}
