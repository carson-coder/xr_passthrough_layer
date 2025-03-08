use glam::f64::{DVec2 as Vec2, DVec4 as Vec4};
use std::sync::Arc;

use crate::{
    steam::StereoCamera,
    utils::{Array, DeviceExt as _},
};
use anyhow::Result;
use log::{info, trace};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferBeginInfo,
        CommandBufferLevel, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract, RecordingCommandBuffer,
    },
    descriptor_set::allocator::DescriptorSetAllocator,
    device::{Device, DeviceOwned},
    format::{Format, FormatFeatures},
    image::{
        sampler::{
            ycbcr::{
                SamplerYcbcrConversion, SamplerYcbcrConversionCreateInfo,
                SamplerYcbcrModelConversion,
            },
            Filter, Sampler, SamplerCreateInfo,
        },
        Image as VkImage, ImageCreateInfo, ImageTiling, ImageUsage,
    },
    memory::allocator::{MemoryAllocator, MemoryTypeFilter},
    pipeline::{
        layout::PipelineDescriptorSetLayoutCreateInfo, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    shader::ShaderModule,
    sync::GpuFuture,
    Handle, VulkanObject,
};

/// Lens distortion correction parameters for a side-by-side stereo image
#[derive(Debug)]
pub struct StereoUndistortParams {
    /// field-of-view parameter, 0 = left eye, 1 = right eye
    fov: [Vec2; 2],
    scale: [Vec2; 2],
    focal: [Vec2; 2],
    center: [Vec2; 2],
    coeff: [Vec4; 2],
    size: f64,
}

impl StereoUndistortParams {
    pub fn fov(&self) -> [Vec2; 2] {
        self.fov
    }
    /// i.e. solving Undistort(src) = dst for the smallest non-zero root.
    fn undistort_inverse(coeff: &Vec4, dst: f64) -> Option<f64> {
        // solving: x * (1 + k1*x^2 + k2*x^4 + k3*x^6 + k4*x^8) - dst = 0
        let f = |x: f64| {
            let x2 = x * x;
            x * (1.0 + x2 * (coeff[0] + x2 * (coeff[1] + x2 * (coeff[2] + x2 * coeff[3])))) - dst
        };
        let fp = |x: f64| {
            let x2 = x * x;
            1.0 + x2
                * (3.0 * coeff[0]
                    + x2 * (5.0 * coeff[1] + x2 * (7.0 * coeff[2] + x2 * 9.0 * coeff[3])))
        };
        const MAX_ITER: u32 = 100;
        let mut x = 0.0;
        for _ in 0..MAX_ITER {
            if fp(x) == 0.0 {
                // Give up
                info!("Divided by zero");
                return None;
            }
            trace!("{} {} {}", x, f(x), fp(x));
            if f(x).abs() < 1e-6 {
                info!("Inverse is: {}, {} {}", x, f(x), dst);
                return Some(x);
            }
            x = x - f(x) / fp(x);
        }
        // Give up
        info!("Cannot find scale");
        None
    }
    // Find a scale that maps the middle point of 4 edges of the undistorted image to
    // the edge of the field of view of the distorted image.
    //
    // Returns the scales and the adjusted fovs
    fn find_scale(coeff: &Vec4, center: &Vec2, focal: &Vec2) -> (Vec2, Vec2) {
        let ret = [0, 1].map(|i| {
            let min_edge_dist = center[i].min(1.0 - center[i]) / focal[i];
            // Find the input theta angle where Undistort(theta) = min_edge_dist
            if let Some(theta) = Self::undistort_inverse(coeff, min_edge_dist) {
                if theta >= std::f64::consts::PI / 2.0 {
                    // infinity?
                    (1.0, focal[i])
                } else {
                    // Find the input coordinates that will give us that theta
                    let target_edge = theta.tan();
                    log::info!("{}", target_edge);
                    (target_edge / (0.5 / focal[i]), 1.0 / min_edge_dist / 2.0)
                }
            } else {
                // Cannot find scale so just don't scale
                (1.0, focal[i])
            }
        });
        (Vec2::new(ret[0].0, ret[1].0), Vec2::new(ret[0].1, ret[1].1))
    }
    /// Input size is (size * 2, size)
    /// returns also the adjusted FOV for left and right
    ///
    /// # Arguments
    ///
    /// - is_final: whether this is the final stage of the pipeline.
    ///             if true, the output image will be submitted to
    ///             the vr compositor.
    pub fn new(size: u32, camera_calib: &StereoCamera) -> Result<Self> {
        let size = size as f64;
        let center = [
            Vec2::new(
                camera_calib.left.intrinsics.center_x / size,
                camera_calib.left.intrinsics.center_y / size,
            ),
            Vec2::new(
                camera_calib.right.intrinsics.center_x / size,
                camera_calib.right.intrinsics.center_y / size,
            ),
        ];
        let focal = [
            Vec2::new(
                camera_calib.left.intrinsics.focal_x / size,
                camera_calib.left.intrinsics.focal_y / size,
            ),
            Vec2::new(
                camera_calib.right.intrinsics.focal_x / size,
                camera_calib.right.intrinsics.focal_y / size,
            ),
        ];
        let coeff: [Vec4; 2] = [
            camera_calib.left.intrinsics.distort.coeffs.into(),
            camera_calib.left.intrinsics.distort.coeffs.into(),
        ];
        let scale_fov = [0, 1].map(|i| Self::find_scale(&coeff[i], &center[i], &focal[i]));
        Ok(Self {
            fov: [scale_fov[0].1, scale_fov[1].1],
            scale: [scale_fov[0].0, scale_fov[1].0],
            focal,
            center,
            coeff,
            size,
        })
    }
}

pub(crate) struct Pipeline {
    correction: Option<StereoUndistortParams>,
    capture: bool,
    render_doc: Option<renderdoc::RenderDoc<renderdoc::V100>>,
    /// A cpu buffer for storing and uploading the input image.
    input_image_buffer: Arc<Buffer>,
    /// A texture on the GPU side for the input image.
    input_image_gpu: Arc<VkImage>,
    camera_config: Option<StereoCamera>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("correction", &self.correction)
            .field("capture", &self.capture)
            .field("render_doc", &self.render_doc)
            .field("input_texture", &self.input_image_gpu.handle().as_raw())
            .field("camera_config", &self.camera_config)
            .finish_non_exhaustive()
    }
}

use crate::CAMERA_SIZE;

impl Pipeline {
    pub(crate) fn submit_cpu_image(
        &self,
        img: &[u8],
        cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
        queue: &Arc<vulkano::device::Queue>,
        output: Arc<VkImage>,
    ) -> Result<impl GpuFuture> {
        let buffer = Subbuffer::new(self.input_image_buffer.clone()).slice(0..img.len() as u64);
        buffer.write()?.copy_from_slice(img);
        let mut cmdbuf = AutoCommandBufferBuilder::primary(
            cmdbuf_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        cmdbuf.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(buffer, output))?;
        Ok(cmdbuf.build()?.execute(queue.clone())?)
    }

    pub fn load_shader(
        device: &Arc<Device>,
        source_is_yuyv: bool,
        has_camera_config: bool,
        has_yuyv_sampler: bool,
    ) -> anyhow::Result<(Arc<ShaderModule>, Arc<ShaderModule>)> {
        let vs = vs::load(device.clone())?;
        let fs = match (source_is_yuyv && !has_yuyv_sampler, has_camera_config) {
            (true, true) => fs::yuyv_undistort::load(device.clone())?,
            (true, false) => fs::yuyv::load(device.clone())?,
            (false, true) => fs::undistort::load(device.clone())?,
            (false, false) => fs::unprocessed::load(device.clone())?,
        };
        Ok((vs, fs))
    }

    /// Create post-processing stages
    pub(crate) fn new(
        device: Arc<Device>,
        allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        source_is_yuyv: bool,
        camera_config: Option<StereoCamera>,
    ) -> Result<Self> {
        let render_doc = renderdoc::RenderDoc::new().ok();
        if render_doc.is_some() {
            log::info!("RenderDoc loaded");
        }
        let exts = device.enabled_extensions();
        let feats = device.enabled_features();
        let has_yuyv_sampler =
            if exts.khr_sampler_ycbcr_conversion && feats.sampler_ycbcr_conversion {
                let format_feats = device
                    .physical_device()
                    .format_properties(Format::G8B8G8R8_422_UNORM)?
                    .format_features(ImageTiling::Optimal, &[]);
                format_feats.contains(FormatFeatures::MIDPOINT_CHROMA_SAMPLES)
            } else {
                false
            };
        let format = if has_yuyv_sampler && source_is_yuyv {
            Format::G8B8G8R8_422_UNORM
        } else {
            Format::R8G8B8A8_UNORM
        };
        let (vs, fs) = Self::load_shader(
            &device,
            source_is_yuyv,
            camera_config.is_some(),
            has_yuyv_sampler,
        )?;
        // Allocate intermediate textures
        let input_texture = device.clone().new_image(
            ImageCreateInfo {
                extent: if source_is_yuyv && !has_yuyv_sampler {
                    [CAMERA_SIZE, CAMERA_SIZE, 1]
                } else {
                    [CAMERA_SIZE * 2, CAMERA_SIZE, 1]
                },
                format,
                usage: ImageUsage::TRANSFER_DST
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::SAMPLED
                    | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
            MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
        )?;
        let cpu_buffer = device.clone().new_buffer(
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                // This should be more than enough. Camera sources are YUV subsampled,
                // so it won't be 4 bytes per pixel.
                size: CAMERA_SIZE as u64 * CAMERA_SIZE as u64 * 2 * 4,
                ..Default::default()
            },
            MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
        )?;
        let correction = camera_config
            .map(|c| StereoUndistortParams::new(CAMERA_SIZE, &c))
            .transpose()?;
        log::debug!("correction fov: {:?}", correction.as_ref().map(|x| x.fov()));
        let fov = correction
            .as_ref()
            .map(|c| c.fov())
            .unwrap_or([Vec2::new(1.19, 1.19); 2]); // default to roughly 100 degrees fov, hopefully this is sensible
        let stages = [
            PipelineShaderStageCreateInfo::new(vs.entry_point("main").unwrap()),
            PipelineShaderStageCreateInfo::new(fs.entry_point("main").unwrap()),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())?,
        );
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                min_filter: Filter::Linear,
                mag_filter: Filter::Linear,
                sampler_ycbcr_conversion: has_yuyv_sampler
                    .then(|| {
                        SamplerYcbcrConversion::new(
                            device.clone(),
                            SamplerYcbcrConversionCreateInfo {
                                format,
                                ycbcr_model: SamplerYcbcrModelConversion::Ycbcr709,
                                ..Default::default()
                            },
                        )
                    })
                    .transpose()?,
                ..Default::default()
            },
        )?;
        log::info!("Adjusted FOV: {:?}", fov);
        Ok(Self {
            correction,
            capture: false,
            render_doc,
            camera_config,
            input_image_gpu: input_texture,
            input_image_buffer: cpu_buffer,
        })
    }
    pub fn fov(&self) -> [Vec2; 2] {
        self.correction
            .as_ref()
            .map(|c| c.fov())
            .unwrap_or([Vec2::new(1.19, 1.19); 2])
    }
    /// Run the pipeline
    ///
    /// # Arguments
    ///
    /// - time: Time offset into the past when the camera frame is captured
    pub(crate) fn run(
        &mut self,
        queue: &Arc<vulkano::device::Queue>,
        allocator: Arc<dyn MemoryAllocator>,
        cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
        input: &[u8],
        output: Arc<vulkano::image::Image>,
    ) -> Result<impl GpuFuture> {
        if self.capture {
            if let Some(rd) = self.render_doc.as_mut() {
                log::info!("Start Capture");
                rd.start_frame_capture(std::ptr::null(), std::ptr::null());
            }
        }

        // 1. submit image to GPU
        // 2. convert YUYV to RGB
        let future = self.submit_cpu_image(
            input,
            cmdbuf_allocator.clone(),
            queue,
            self.input_image_gpu.clone(),
        )?;
        // TODO: run shader
        future.flush()?;

        if self.capture {
            if let Some(rd) = self.render_doc.as_mut() {
                log::info!("End Capture");
                rd.end_frame_capture(std::ptr::null(), std::ptr::null());
            }
            self.capture = false;
        }
        Ok(future)
    }
    pub(crate) fn capture_next_frame(&mut self) {
        self.capture = true;
    }
}

mod fs {
    pub mod yuyv_undistort {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/combined.frag",
            define: [
                ("INPUT_IS_YUYV", "1"),
                ("UNDISTORT", "1"),
            ],
            custom_derives: [Copy, Clone, Debug],
        }
    }
    pub mod undistort {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/combined.frag",
            define: [
                ("UNDISTORT", "1"),
            ],
            custom_derives: [Copy, Clone, Debug],
        }
    }
    pub mod yuyv {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/combined.frag",
            define: [
                ("INPUT_IS_YUYV", "1"),
            ],
            custom_derives: [Copy, Clone, Debug],
        }
    }
    pub mod unprocessed {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/combined.frag",
            custom_derives: [Copy, Clone, Debug],
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/projection.vert",
        custom_derives: [Copy, Clone, Debug, Default],
    }
}
