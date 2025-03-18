use glam::{
    f64::{DVec2 as Vec2, DVec4 as Vec4},
    IVec2, UVec2,
};
use smallvec::smallvec;
use std::sync::Arc;

use crate::{steam::StereoCamera, utils::DeviceExt as _};
use anyhow::Result;
use log::{info, trace};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferExecFuture,
        CommandBufferUsage, CopyBufferToImageInfo, PrimaryCommandBufferAbstract,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    descriptor_set::{allocator::DescriptorSetAllocator, DescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    format::{Format, FormatFeatures},
    image::{
        sampler::{
            ycbcr::{
                SamplerYcbcrConversion, SamplerYcbcrConversionCreateInfo,
                SamplerYcbcrModelConversion,
            },
            Filter, Sampler, SamplerCreateInfo,
        },
        view::{ImageView, ImageViewCreateInfo},
        Image as VkImage, ImageCreateInfo, ImageLayout, ImageTiling, ImageUsage,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocatePreference, MemoryAllocator, MemoryTypeFilter,
    },
    padded::Padded,
    pipeline::{
        cache::PipelineCache,
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::{self, Vertex as _, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, Pipeline as _, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    shader::ShaderModule,
    sync::{future::NowFuture, GpuFuture},
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
    size: Vec2,
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
    pub fn new(size: UVec2, camera_calib: &StereoCamera) -> Result<Self> {
        let size = size.as_dvec2();
        let center = [
            Vec2::new(
                camera_calib.left.intrinsics.center_x / size.x,
                camera_calib.left.intrinsics.center_y / size.y,
            ),
            Vec2::new(
                camera_calib.right.intrinsics.center_x / size.x,
                camera_calib.right.intrinsics.center_y / size.y,
            ),
        ];
        let focal = [
            Vec2::new(
                camera_calib.left.intrinsics.focal_x / size.x,
                camera_calib.left.intrinsics.focal_y / size.y,
            ),
            Vec2::new(
                camera_calib.right.intrinsics.focal_x / size.x,
                camera_calib.right.intrinsics.focal_y / size.y,
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

#[derive(vertex_input::Vertex, Default, Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    in_center: [f32; 2],
    #[format(R32G32_SFLOAT)]
    in_tex_coord: [f32; 2],
}

pub(crate) struct Pipeline {
    correction: Option<StereoUndistortParams>,
    capture: bool,
    render_doc: Option<renderdoc::RenderDoc<renderdoc::V100>>,
    /// A cpu buffer for storing and uploading the input image.
    input_image_buffer: Arc<Buffer>,
    input_image_gpu: Arc<VkImage>,
    camera_config: Option<StereoCamera>,
    size: UVec2,
    allocator: Arc<dyn MemoryAllocator>,
    cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<vulkano::device::Queue>,

    vertices: Subbuffer<[Vertex]>,
    render_size: UVec2,

    // Vulkan states
    desc_set: Arc<DescriptorSet>,
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
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

pub trait PostprocessPipeline {
    type Image;
    type Future: GpuFuture;
    fn allocate_image(&self) -> Result<Self::Image>;
    /// Postprocess the camera image, taking input from a in memory buffer.
    fn postprocess(&self, input: &[u8], output: &Self::Image) -> Result<Self::Future>;
    // /// Postprocess the camera image, taking input from a dmabuf file descriptor.
    // fn postprocess_dmabuf(&self, input: OwnedFd, output: &Self::Image) -> Result<Self::Future>;
}

impl Pipeline {
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
    /// The camera image is two `size` images stitched together side-by-side.
    pub(crate) fn new(
        device: Arc<Device>,
        allocator: Arc<dyn MemoryAllocator>,
        cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
        queue: Arc<Queue>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        source_is_yuyv: bool,
        camera_config: Option<StereoCamera>,
        final_layout: ImageLayout,
        pipeline_cache: Arc<PipelineCache>,
        camera_size: UVec2,
        render_size: UVec2,
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
        let vs_main = vs.entry_point("main").unwrap();
        let fs_main = fs.entry_point("main").unwrap();

        // Allocate intermediate textures
        let input_texture = device.clone().new_image(
            ImageCreateInfo {
                extent: if source_is_yuyv && !has_yuyv_sampler {
                    // Source is raw, unconverted yuyv, therefore is downsampled 2x in the X
                    // direction.
                    [camera_size.x, camera_size.y, 1]
                } else {
                    [camera_size.x * 2, camera_size.y, 1]
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
                size: camera_size.x as u64
                    * camera_size.y as u64
                    * 2
                    * if source_is_yuyv { 2 } else { 4 },
                ..Default::default()
            },
            MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
        )?;
        let render_pass = vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                format: vulkano::format::Format::R8G8B8A8_UNORM,
                samples: 1,
                load_op: Load,
                store_op: Store,
                final_layout: final_layout,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {},
        })
        .unwrap();
        let correction = camera_config
            .map(|c| StereoUndistortParams::new(camera_size, &c))
            .transpose()?;
        log::debug!("correction fov: {:?}", correction.as_ref().map(|x| x.fov()));
        let fov = correction
            .as_ref()
            .map(|c| c.fov())
            .unwrap_or([Vec2::new(1.19, 1.19); 2]); // default to roughly 100 degrees fov, hopefully this is sensible
        let stages = smallvec![
            PipelineShaderStageCreateInfo::new(vs_main.clone()),
            PipelineShaderStageCreateInfo::new(fs_main),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())?,
        )?;
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
        let distortion_params = correction
            .as_ref()
            .map(|c| {
                Buffer::from_data(
                    allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                            | MemoryTypeFilter::PREFER_DEVICE,
                        allocate_preference: MemoryAllocatePreference::Unknown,
                        ..Default::default()
                    },
                    fs::yuyv_undistort::DistortionParameters {
                        center: c.center.map(|v| Padded(*v.as_vec2().as_ref())),
                        dcoef: c.coeff.map(|v| *v.as_vec4().as_ref()),
                        focal: c.focal.map(|v| Padded(*v.as_vec2().as_ref())),
                        scale: c.scale.map(|v| Padded(*v.as_vec2().as_ref())),
                    },
                )
                .map_err(anyhow::Error::from)
            })
            .transpose()?;
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            Some(pipeline_cache),
            GraphicsPipelineCreateInfo {
                vertex_input_state: Some(Vertex::per_vertex().definition(&vs_main)?),
                stages,
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleList,
                    ..Default::default()
                }),
                viewport_state: Some(ViewportState {
                    viewports: smallvec![Viewport {
                        offset: [0., 0.],
                        extent: *render_size.as_vec2().as_ref(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                subpass: Some(Subpass::from(render_pass.clone(), 0).unwrap().into()),
                multisample_state: Some(Default::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    1,
                    Default::default(),
                )),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?;
        let desc_set_writes = [WriteDescriptorSet::image_view_sampler(
            1,
            ImageView::new(
                input_texture.clone(),
                ImageViewCreateInfo::from_image(&input_texture),
            )?,
            sampler.clone(),
        )]
        .into_iter()
        .chain(
            distortion_params
                .clone()
                .map(|b| WriteDescriptorSet::buffer(2, b)),
        );
        let vertices = Buffer::from_iter::<Vertex, _>(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                allocate_preference: MemoryAllocatePreference::Unknown,
                ..Default::default()
            },
            [
                Vertex {
                    position: [-1.0, -1.0],
                    in_tex_coord: [-0.5, -0.5],
                    in_center: [-0.25, 0.],
                },
                Vertex {
                    position: [-1.0, 1.0],
                    in_tex_coord: [-0.5, 0.5],
                    in_center: [-0.25, 0.],
                },
                Vertex {
                    position: [0.0, -1.0],
                    in_tex_coord: [0., -0.5],
                    in_center: [-0.25, 0.],
                },
                Vertex {
                    position: [0.0, -1.0],
                    in_tex_coord: [0., -0.5],
                    in_center: [0.25, 0.],
                },
                Vertex {
                    position: [-1.0, 1.0],
                    in_tex_coord: [-0.5, 0.5],
                    in_center: [0.25, 0.],
                },
                Vertex {
                    position: [0.0, 1.0],
                    in_tex_coord: [0., 0.5],
                    in_center: [0.25, 0.],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap();
        let desc_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            pipeline.layout().set_layouts().first().unwrap().clone(),
            desc_set_writes,
            None,
        )?;

        log::info!("Adjusted FOV: {:?}", fov);
        Ok(Self {
            correction,
            capture: false,
            render_doc,
            camera_config,
            size: camera_size,
            vertices,
            input_image_buffer: cpu_buffer,
            input_image_gpu: input_texture,
            desc_set,
            pipeline,
            render_pass,
            allocator,
            cmdbuf_allocator,
            queue,
            render_size,
        })
    }
    pub fn fov(&self) -> [Vec2; 2] {
        self.correction
            .as_ref()
            .map(|c| c.fov())
            .unwrap_or([Vec2::new(1.19, 1.19); 2])
    }
}

impl PostprocessPipeline for Pipeline {
    type Image = Arc<VkImage>;
    type Future = CommandBufferExecFuture<NowFuture>;
    fn allocate_image(&self) -> Result<Self::Image> {
        Ok(VkImage::new(
            self.allocator.clone(),
            ImageCreateInfo {
                extent: [self.render_size.x as _, self.render_size.y as _, 1],
                format: vulkano::format::Format::R8G8B8A8_SNORM,
                usage: ImageUsage::TRANSFER_DST
                    | ImageUsage::SAMPLED
                    | ImageUsage::COLOR_ATTACHMENT,
                mip_levels: 1,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )?)
    }
    /// Run the pipeline
    ///
    /// # Arguments
    ///
    /// - time: Time offset into the past when the camera frame is captured
    fn postprocess(&self, input: &[u8], output: &Self::Image) -> Result<Self::Future> {
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            vulkano::render_pass::FramebufferCreateInfo {
                attachments: vec![ImageView::new(
                    output.clone(),
                    ImageViewCreateInfo::from_image(output),
                )?],
                ..Default::default()
            },
        )?;
        let mut cmdbuf = AutoCommandBufferBuilder::primary(
            self.cmdbuf_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        // 1. submit image to GPU
        // 2. convert YUYV to RGB
        let buffer = Subbuffer::new(self.input_image_buffer.clone()).slice(0..input.len() as u64);
        buffer.write()?.copy_from_slice(input);
        cmdbuf.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            buffer,
            self.input_image_gpu.clone(),
        ))?;
        cmdbuf
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.desc_set.clone(),
            )?
            .bind_vertex_buffers(0, self.vertices.clone())?;
        unsafe { cmdbuf.draw(self.vertices.len() as u32, 1, 0, 0)? }
            .end_render_pass(SubpassEndInfo::default())?;

        Ok(cmdbuf.build()?.execute(self.queue.clone())?)
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
        src: "#version 450
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 in_center;
layout(location = 2) in vec2 in_texCoord;
layout(location = 0) out flat vec2 center;
layout(location = 1) out vec2 texCoord;

void main() {
    gl_Position = vec4(position, 0, 1);
    center = in_center;
    texCoord = in_texCoord;
}"
    }
}
