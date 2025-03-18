#![deny(rust_2018_idioms)]
mod camera;
mod config;
mod pipeline;
mod steam;
mod utils;
mod xr;
use ash::khr::swapchain;
use smallvec::smallvec;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use std::{
    collections::HashSet,
    future,
    sync::{atomic::AtomicBool, Arc, Mutex},
};

use anyhow::{anyhow, Context, Result};

use glam::UVec2;
use pipeline::PostprocessPipeline as _;
use v4l::video::Capture;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{
            CommandBufferAllocator, StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo,
        },
        AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferBeginInfo,
        CommandBufferLevel, CommandBufferSubmitInfo, CommandBufferUsage, CopyBufferToImageInfo,
        CopyImageInfo, ImageBlit, PrimaryCommandBufferAbstract, RecordingCommandBuffer,
        SemaphoreSubmitInfo, SubmitInfo,
    },
    descriptor_set::allocator::{
        StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
    },
    device::Queue,
    format,
    image::{
        AllocateImageError, Image, ImageAspects, ImageCreateInfo, ImageLayout,
        ImageSubresourceRange, ImageUsage,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::cache::{PipelineCache, PipelineCacheCreateInfo},
    swapchain::{
        AcquireNextImageInfo, AcquiredImage, PresentInfo, SemaphorePresentInfo, Surface,
        SurfaceInfo, Swapchain, SwapchainPresentInfo,
    },
    sync::{
        fence::FenceCreateInfo,
        semaphore::{self, SemaphoreCreateInfo},
        AccessFlags, DependencyInfo, GpuFuture, ImageMemoryBarrier, PipelineStages,
    },
    Validated, VulkanObject,
};
use vulkano_shaders;
use xdg::BaseDirectories;
/// Camera image will be (size * 2, size)
const CAMERA_SIZE: u32 = 960;
#[allow(unused_imports)]
use log::info;

static APP_NAME: &str = "Camera\0";
static APP_VERSION: u32 = 0;

fn find_index_camera() -> Result<std::path::PathBuf> {
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

static SPLASH_IMAGE: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/splash.png"));

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

struct FrameInfo<I> {
    frame: I,
    frame_time: std::time::Instant,
}

fn load_splash(
    allocator: Arc<dyn MemoryAllocator>,
    cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    pp: &pipeline::Pipeline,
) -> Result<Arc<vulkano::image::Image>> {
    log::debug!("loading splash");
    let img = image::load_from_memory_with_format(SPLASH_IMAGE, image::ImageFormat::Png)?
        .into_rgba8()
        .into_raw();

    log::debug!("splash loaded");
    let vkimg = pp.allocate_image()?;
    let mut cmdbuf = AutoCommandBufferBuilder::primary(
        cmdbuf_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    let buffer = Buffer::new_unsized::<[u8]>(
        allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                | MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        img.len() as _,
    )?;
    buffer.write()?.copy_from_slice(&img);
    cmdbuf.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(buffer, vkimg.clone()))?;
    cmdbuf
        .build()?
        .execute(queue.clone())?
        .then_signal_fence()
        .wait(None)?;

    Ok(vkimg)
}

struct App {
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    format: format::Format,
    device: Arc<vulkano::device::Device>,
    surface: Arc<Surface>,
    cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    size: PhysicalSize<u32>,
    camera: camera::CameraThread<pipeline::Pipeline>,
    instance_fn: ash::InstanceFnV1_0,
}
impl App {
    fn redraw(&mut self) -> Result<()> {
        let semaphore = Arc::new(vulkano::sync::semaphore::Semaphore::new(
            self.device.clone(),
            SemaphoreCreateInfo::default(),
        )?);
        let image_index = loop {
            match unsafe {
                self.swapchain.acquire_next_image(&AcquireNextImageInfo {
                    semaphore: Some(semaphore.clone()),
                    ..Default::default()
                })
            } {
                Ok(AcquiredImage {
                    image_index,
                    is_suboptimal: false,
                }) => break image_index,
                Ok(AcquiredImage {
                    is_suboptimal: true,
                    ..
                })
                | Err(vulkano::Validated::Error(vulkano::VulkanError::OutOfDate)) => {
                    (self.swapchain, self.images) = vulkano::swapchain::Swapchain::new(
                        self.device.clone(),
                        self.surface.clone(),
                        vulkano::swapchain::SwapchainCreateInfo {
                            min_image_count: 2,
                            image_format: self.format,
                            image_extent: self.size.into(),
                            image_usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT,
                            composite_alpha: vulkano::swapchain::CompositeAlpha::Opaque,
                            present_mode: vulkano::swapchain::PresentMode::Fifo,
                            ..Default::default()
                        },
                    )?;
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
        };
        let mut cmdbuf = RecordingCommandBuffer::new(
            self.cmdbuf_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )?;
        let src_image = self.camera.with_frame(|f| f.frame.clone());
        let dst_image = self.images[image_index as usize].clone();
        let [w, h, _] = src_image.extent();
        let [dw, dh, _] = dst_image.extent();
        let aspect_ratio = w as f64 / h as f64;
        let (mut target_w, mut target_h) = (dh as f64 * aspect_ratio, dh as f64);
        if target_w > dw as _ {
            target_w = dw as _;
            target_h = target_w / aspect_ratio;
        }
        let crop_x = (dw as f64 - target_w).max(0.) / 2.0;
        let crop_y = (dh as f64 - target_h).max(0.) / 2.0;

        let cmdbuf = unsafe {
            cmdbuf
                .pipeline_barrier(&DependencyInfo {
                    image_memory_barriers: smallvec![ImageMemoryBarrier {
                        src_stages: PipelineStages::TOP_OF_PIPE,
                        dst_stages: PipelineStages::TOP_OF_PIPE,
                        old_layout: ImageLayout::Undefined,
                        new_layout: ImageLayout::TransferDstOptimal,
                        subresource_range: dst_image.subresource_range(),
                        ..ImageMemoryBarrier::image(dst_image.clone())
                    }],
                    ..Default::default()
                })?
                .clear_color_image(&ClearColorImageInfo::image(dst_image.clone()))?
                .blit_image(&BlitImageInfo {
                    src_image_layout: ImageLayout::TransferSrcOptimal,
                    dst_image_layout: ImageLayout::TransferDstOptimal,
                    filter: vulkano::image::sampler::Filter::Linear,
                    regions: smallvec![ImageBlit {
                        src_offsets: [[0, 0, 0], src_image.extent(),],
                        src_subresource: src_image.subresource_layers(),
                        dst_offsets: [
                            [crop_x as u32, crop_y as u32, 0],
                            [(crop_x + target_w) as u32, (crop_y + target_h) as u32, 1]
                        ],
                        dst_subresource: dst_image.subresource_layers(),
                        ..Default::default()
                    }],
                    ..BlitImageInfo::images(src_image.clone(), dst_image.clone())
                })?
                .pipeline_barrier(&DependencyInfo {
                    image_memory_barriers: smallvec![ImageMemoryBarrier {
                        src_stages: PipelineStages::BOTTOM_OF_PIPE,
                        dst_stages: PipelineStages::BOTTOM_OF_PIPE,
                        old_layout: ImageLayout::TransferDstOptimal,
                        new_layout: ImageLayout::PresentSrc,
                        ..ImageMemoryBarrier::image(dst_image.clone())
                    }],
                    ..Default::default()
                })?;
            cmdbuf.end()?
        };
        let semaphore = self.queue.with(|mut q| unsafe {
            let device = ash::Device::load(&self.instance_fn, self.device.handle());
            let semaphore2 = vulkano::sync::semaphore::Semaphore::new(
                self.device.clone(),
                SemaphoreCreateInfo::default(),
            )?;
            device.queue_submit2(
                self.queue.handle(),
                &[ash::vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&[
                        ash::vk::SemaphoreSubmitInfo::default().semaphore(semaphore.handle())
                    ])
                    .signal_semaphore_infos(&[
                        ash::vk::SemaphoreSubmitInfo::default().semaphore(semaphore2.handle())
                    ])
                    .command_buffer_infos(&[
                        ash::vk::CommandBufferSubmitInfo::default().command_buffer(cmdbuf.handle())
                    ])],
                ash::vk::Fence::null(),
            )?;
            Ok::<_, anyhow::Error>(q.present(&PresentInfo {
                wait_semaphores: vec![SemaphorePresentInfo::new(semaphore2.into())],
                swapchain_infos: vec![SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    image_index,
                )],
                ..Default::default()
            })?)
        })?;
        Ok(())
    }
}
impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {}
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.size = size,
            WindowEvent::RedrawRequested => {
                let Err(e) = self.redraw() else { return };
                log::warn!("Failed to redraw {e}");
                event_loop.exit();
            }
            _ => (),
        }
    }
    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, (): ()) {
        event_loop.exit();
    }
}
fn main() -> Result<()> {
    let env = env_logger::Env::default().default_filter_or("info");
    env_logger::Builder::from_env(env)
        .format_timestamp_millis()
        .init();
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .build(&event_loop)
        .expect("Failed to create window");
    let required_extensions = vulkano::swapchain::Surface::required_extensions(&event_loop)?;
    let xr = xr::OpenXr::new(required_extensions, UVec2::new(CAMERA_SIZE, CAMERA_SIZE))?;
    let instance = xr.vk_instance();
    let (device, queue) = xr.vk_device();
    let surface = vulkano::swapchain::Surface::from_window(instance.clone(), &window)?;
    let swapchain_format = device
        .physical_device()
        .surface_formats(
            &surface,
            SurfaceInfo {
                present_mode: Some(vulkano::swapchain::PresentMode::Fifo),
                ..Default::default()
            },
        )?
        .into_iter()
        .map(|(f, _)| f)
        .collect::<HashSet<_>>();
    const PREFERRED_FORMATS: &[format::Format] =
        &[format::Format::R8G8B8_UNORM, format::Format::B8G8R8_UNORM];
    let swapchain_format = PREFERRED_FORMATS
        .iter()
        .find(|f| swapchain_format.contains(f))
        .context("cannot find a suitable format for swapchain images")?;
    let (swapchain, images) = vulkano::swapchain::Swapchain::new(
        device.clone(),
        surface.clone(),
        vulkano::swapchain::SwapchainCreateInfo {
            min_image_count: 2,
            image_format: *swapchain_format,
            image_extent: window.inner_size().into(),
            image_usage: vulkano::image::ImageUsage::TRANSFER_DST,
            composite_alpha: vulkano::swapchain::CompositeAlpha::Opaque,
            present_mode: vulkano::swapchain::PresentMode::Fifo,
            ..Default::default()
        },
    )?;
    let allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let cmdbuf_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        StandardDescriptorSetAllocatorCreateInfo::default(),
    ));
    let camera =
        v4l::Device::with_path(find_index_camera()?).context("cannot open camera device")?;
    if !camera
        .query_caps()?
        .capabilities
        .contains(v4l::capability::Flags::VIDEO_CAPTURE)
    {
        return Err(anyhow!("Cannot capture from index camera"));
    }
    let format = camera.set_format(&v4l::Format::new(
        CAMERA_SIZE * 2,
        CAMERA_SIZE,
        v4l::FourCC::new(b"YUYV"),
    ))?;
    let cache_file = xdg::BaseDirectories::new()?
        .find_cache_file(std::path::Path::new("xr_passthrough").join("pipeline_cache"))
        .and_then(|f| std::fs::read(f).ok())
        .and_then(|data| {
            let buf = &data[..];
            let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(
                &buf[..ed25519_dalek::PUBLIC_KEY_LENGTH].try_into().unwrap(),
            )
            .ok()?;
            let buf = &buf[ed25519_dalek::PUBLIC_KEY_LENGTH..];
            let signature = &ed25519_dalek::Signature::from_bytes(
                buf[..ed25519_dalek::SIGNATURE_LENGTH].try_into().unwrap(),
            );
            let buf = &buf[ed25519_dalek::SIGNATURE_LENGTH..];

            verifying_key.verify_strict(buf, signature).ok()?;
            data.drain(..ed25519_dalek::PUBLIC_KEY_LENGTH + ed25519_dalek::SIGNATURE_LENGTH);
            Some(data)
        });
    // SAFETY: well we validated the on disk cache with a cryptographic signature.
    let pipeline_cache = unsafe {
        PipelineCache::new(
            device.clone(),
            PipelineCacheCreateInfo {
                initial_data: cache_file.unwrap_or_default(),
                ..Default::default()
            },
        )
    }?;
    let camera_config = steam::find_steam_config();
    log::info!("{}", format);
    let pp = pipeline::Pipeline::new(
        device.clone(),
        allocator.clone(),
        cmdbuf_allocator.clone(),
        queue.clone(),
        descriptor_set_allocator.clone(),
        true,
        camera_config,
        ImageLayout::TransferSrcOptimal,
        pipeline_cache,
        UVec2::new(CAMERA_SIZE, CAMERA_SIZE),
        UVec2::new(CAMERA_SIZE, CAMERA_SIZE),
    )?;
    log::debug!("pipeline: {pp:?}");
    camera.set_params(&v4l::video::capture::Parameters::with_fps(54))?;
    let splash = load_splash(
        allocator.clone(),
        cmdbuf_allocator.clone(),
        queue.clone(),
        &pp,
    )?;
    let camera = camera::CameraThread::new(camera, splash, pp);

    let proxy = event_loop.create_proxy();
    ctrlc::set_handler({
        move || {
            let _ = proxy.send_event(());
        }
    })
    .expect("Error setting Ctrl-C handler");
    // Event loop
    let app = App {
        swapchain,
        images,
        device: device.clone(),
        size: window.inner_size(),
        cmdbuf_allocator,
        queue,
        surface,
        format: *swapchain_format,
    };
    event_loop.run_app(&mut app);

    camera.exit();
    Ok(())
}
