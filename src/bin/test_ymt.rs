use std::env;
use std::fs;
use rpf_archive::{ymt::parse_ymt};

fn main() {
    let args: Vec<String> = env::args().collect();
    let data = fs::read(&args[1]).unwrap();
    match parse_ymt(&data) {
        Ok(_) => println!("Success"),
        Err(e) => println!("Error: {:?}", e),
    }
}
