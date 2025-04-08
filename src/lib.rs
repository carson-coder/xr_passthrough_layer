#![deny(rust_2018_idioms, rust_2024_compatibility, rust_2021_compatibility)]
pub mod api_layer;
pub mod camera;
pub mod config;
pub mod pipeline;
pub mod steam;
pub mod utils;

use anyhow::{Context, Result, anyhow};

/// Camera image will be (size * 2, size)
pub const CAMERA_SIZE: u32 = 960;
use glam::UVec2;
#[allow(unused_imports)]
use log::info;
pub struct FrameInfo {
    pub frame: Vec<u8>,
    pub frame_time: std::time::Instant,
    pub size: UVec2,
    pub needs_postprocess: bool,
}

pub fn find_index_camera() -> Result<std::path::PathBuf> {
    let mut it = udev::Enumerator::new()?;
    it.match_subsystem("video4linux")?;
    it.match_property("ID_VENDOR_ID", "28de")?;
    it.match_property("ID_MODEL_ID", "2400")?;

    let dev = it
        .scan_devices()?
        .next()
        .with_context(|| anyhow!("Index camera not found"))?;
    let devnode = dev
        .devnode()
        .with_context(|| anyhow!("Index camera cannot be accessed"))?;
    Ok(devnode.to_owned())
}
