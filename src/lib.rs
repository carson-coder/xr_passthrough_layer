#![deny(rust_2018_idioms)]
pub mod camera;
pub mod config;
pub mod pipeline;
pub mod steam;
pub mod utils;
pub mod xr;

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};

use vulkano::{
    format,
    image::{AllocateImageError, Image, ImageCreateInfo, ImageUsage},
    memory::allocator::MemoryTypeFilter,
    Validated,
};
/// Camera image will be (size * 2, size)
pub const CAMERA_SIZE: u32 = 960;
#[allow(unused_imports)]
use log::info;
pub struct FrameInfo {
    pub frame: Arc<vulkano::image::Image>,
    pub frame_time: std::time::Instant,
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

fn create_submittable_image(
    device: Arc<vulkano::device::Device>,
) -> Result<Arc<Image>, Validated<AllocateImageError>> {
    use crate::utils::DeviceExt;
    device.new_image(
        ImageCreateInfo {
            extent: [CAMERA_SIZE * 2, CAMERA_SIZE, 1],
            format: format::Format::R8G8B8A8_UNORM,
            usage: ImageUsage::TRANSFER_DST
                | ImageUsage::SAMPLED
                | ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::TRANSFER_SRC,
            mip_levels: 1,
            ..Default::default()
        },
        MemoryTypeFilter::PREFER_DEVICE,
    )
}
