#![deny(rust_2018_idioms)]
use smallvec::smallvec;
use winit::{event::WindowEvent, event_loop::EventLoop};

use std::{collections::HashSet, sync::Arc};

use anyhow::{Context, Result, anyhow};

use ::xr_passthrough_layer::{CAMERA_SIZE, camera, find_index_camera, pipeline, steam};
use glam::UVec2;
use v4l::video::Capture;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferUsage,
        ImageBlit,
        allocator::{
            CommandBufferAllocator, StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo,
        },
    },
    descriptor_set::allocator::{
        StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
    },
    device::{Device, Queue},
    format,
    image::{Image, ImageLayout, ImageUsage},
    memory::allocator::StandardMemoryAllocator,
    swapchain::{SurfaceInfo, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::GpuFuture,
};
static APP_NAME: &str = "Camera\0";
static APP_VERSION: u32 = 0;

static SPLASH_IMAGE: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../splash.png"));

#[derive(Clone)]
struct Window {
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    inner: Arc<winit::window::Window>,
}

struct App {
    pp: pipeline::Pipeline,
    device: Arc<Device>,
    cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    camera: camera::CameraThread,
    window: Option<Window>,
    instance: Arc<vulkano::instance::Instance>,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}
const PREFERRED_FORMATS: &[format::Format] = &[
    format::Format::B8G8R8_UNORM,
    format::Format::R8G8B8_UNORM,
    format::Format::B8G8R8A8_UNORM,
    format::Format::R8G8B8A8_UNORM,
];
impl App {
    fn setup_window(&mut self, window: Arc<winit::window::Window>) -> Result<()> {
        log::info!("setting up window");
        let surface =
            vulkano::swapchain::Surface::from_window(self.instance.clone(), window.clone())?;
        let surface_capabilities = self
            .device
            .physical_device()
            .surface_capabilities(&surface, SurfaceInfo::default())?;
        let swapchain_formats = self
            .device
            .physical_device()
            .surface_formats(&surface, SurfaceInfo::default())?
            .into_iter()
            .map(|(f, _)| f)
            .collect::<HashSet<_>>();
        log::info!("{swapchain_formats:?}");
        let swapchain_format = PREFERRED_FORMATS
            .iter()
            .find(|f| swapchain_formats.contains(f))
            .context("cannot find a suitable format for swapchain images")?;
        let (swapchain, images) = vulkano::swapchain::Swapchain::new(
            self.device.clone(),
            surface.clone(),
            vulkano::swapchain::SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format: *swapchain_format,
                image_extent: window.inner_size().into(),
                image_usage: vulkano::image::ImageUsage::TRANSFER_DST,
                composite_alpha: vulkano::swapchain::CompositeAlpha::Opaque,
                present_mode: vulkano::swapchain::PresentMode::Fifo,
                ..Default::default()
            },
        )?;
        self.window = Some(Window {
            swapchain,
            images,
            inner: window,
        });
        //self.previous_frame_end = Some(vulkano::sync::now(self.device.clone()).boxed());
        Ok(())
    }
    fn recreate_swapchain(&mut self) -> Result<Window> {
        log::info!("recreating swapchain");
        let Some(window) = &mut self.window else {
            panic!("recreate non-existent swapchain")
        };
        (window.swapchain, window.images) = window.swapchain.recreate(SwapchainCreateInfo {
            image_extent: window.inner.inner_size().into(),
            ..window.swapchain.create_info()
        })?;
        Ok(window.clone())
    }
    fn redraw(&mut self) -> Result<()> {
        let Some(mut window) = self.window.clone() else {
            return Ok(());
        };
        let (image_index, future) = loop {
            match vulkano::swapchain::acquire_next_image(
                window.swapchain.clone(),
                Some(std::time::Duration::from_secs(1)),
            ) {
                Ok((image_index, false, future)) => break (image_index, future),
                Ok((_, true, future)) => {
                    window = self.recreate_swapchain()?;
                    self.previous_frame_end = Some(future.boxed());
                    continue;
                }
                Err(vulkano::Validated::Error(vulkano::VulkanError::OutOfDate)) => {
                    window = self.recreate_swapchain()?;
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
        };
        log::trace!("acquired {image_index}");
        let mut cmdbuf = AutoCommandBufferBuilder::primary(
            self.cmdbuf_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        self.pp.maybe_postprocess(&self.camera.frame())?;
        let (src_image, pp_fut) = self.pp.image();
        // have submitted to cmdbuf, otherwise this
        // image could be reused by the camera thread.
        let dst_image = window.images[image_index as usize].clone();
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

        cmdbuf
            .clear_color_image(ClearColorImageInfo::new(dst_image.clone()))?
            .blit_image(BlitImageInfo {
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
                ..BlitImageInfo::new(src_image.clone(), dst_image.clone())
            })?;
        let mut previous_future = self.previous_frame_end.take().unwrap();
        previous_future.cleanup_finished();
        let future = future
            .join(previous_future)
            .join(pp_fut)
            .then_execute(self.queue.clone(), cmdbuf.build()?)?;
        window.inner.pre_present_notify();

        log::trace!("presenting");
        self.previous_frame_end = Some(
            future
                .then_swapchain_present(
                    self.queue.clone(),
                    SwapchainPresentInfo::new(window.swapchain.clone(), image_index),
                )
                .then_signal_fence_and_flush()?
                .boxed(),
        );
        Ok(())
    }
}
impl winit::application::ApplicationHandler for App {
    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.inner.request_redraw();
        }
    }
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        log::info!("resumed");
        let window = match event_loop.create_window(winit::window::Window::default_attributes()) {
            Ok(window) => Arc::new(window),
            Err(e) => {
                log::warn!("Failed to create window {e:#}");
                event_loop.exit();
                return;
            }
        };
        let Err(e) = self.setup_window(window.clone()) else {
            self.camera.resume().unwrap();
            window.request_redraw();
            return;
        };
        log::warn!("Failed to setup rendering surface {e:#}");
        event_loop.exit();
    }
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let Err(e) = self.redraw() else { return };
                log::warn!("Failed to redraw {e:#}");
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
    let required_extensions = vulkano::swapchain::Surface::required_extensions(&event_loop)?;
    let (xr, _frame_waiter, _frame_stream) = xr::OpenXr::new(
        required_extensions,
        &Default::default(),
        &[],
        APP_NAME,
        APP_VERSION,
    )?;
    let instance = xr.vk_instance();
    let (device, queue) = xr.vk_device();
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
    let pipeline_cache = xr_passthrough_layer::config::load_pipeline_cache(
        device.clone(),
        &xdg::BaseDirectories::new()?,
    )?;
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
        ImageUsage::TRANSFER_SRC,
        pipeline_cache,
        UVec2::new(CAMERA_SIZE, CAMERA_SIZE),
        UVec2::new(CAMERA_SIZE, CAMERA_SIZE),
    )?;
    log::info!("pipeline: {pp:?}");
    camera.set_params(&v4l::video::capture::Parameters::with_fps(54))?;
    let camera = camera::CameraThread::new(camera, SPLASH_IMAGE);

    let proxy = event_loop.create_proxy();
    ctrlc::set_handler({
        move || {
            let _ = proxy.send_event(());
        }
    })
    .expect("Error setting Ctrl-C handler");
    // Event loop
    let mut app = App {
        device: device.clone(),
        window: None,
        camera,
        cmdbuf_allocator,
        queue,
        pp,
        instance: instance.clone(),
        previous_frame_end: Some(vulkano::sync::now(device).boxed()),
    };
    log::info!("event loop start");
    event_loop.run_app(&mut app)?;
    log::info!("event loop exited");

    app.camera.exit()?;
    Ok(())
}
