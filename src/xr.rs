#![allow(dead_code, unused_imports)] // remove after i clean this up
use anyhow::{anyhow, Context, Result};
use glam::UVec2;
use itertools::Itertools;
use nalgebra::{Affine3, Matrix3, UnitQuaternion, Vector3};
use openxr::{
    ApplicationInfo, Extent2Df, Extent2Di, EyeVisibility, Offset2Di, OverlaySessionCreateFlagsEXTX,
    Rect2Di, ReferenceSpaceType, SwapchainSubImage,
};
use std::sync::{Arc, OnceLock};
use vulkano::{
    command_buffer::allocator::CommandBufferAllocator,
    descriptor_set::allocator::DescriptorSetAllocator,
    device::{Device, Queue, QueueFlags},
    image::Image,
    instance::{Instance, InstanceExtensions as VkInstanceExtensions},
    memory::allocator::MemoryAllocator,
    Handle, VulkanObject,
};

use vulkano::{
    device::QueueCreateInfo,
    image::{ImageCreateInfo, ImageUsage},
};

static VULKAN_LIBRARY: OnceLock<Arc<vulkano::VulkanLibrary>> = OnceLock::new();

fn get_vulkan_library() -> &'static Arc<vulkano::VulkanLibrary> {
    VULKAN_LIBRARY.get_or_init(|| vulkano::VulkanLibrary::new().unwrap())
}

pub struct OpenXr {
    instance: openxr::Instance,

    session_state: openxr::SessionState,
    session: openxr::Session<openxr::Vulkan>,
    frame_waiter: openxr::FrameWaiter,
    frame_stream: openxr::FrameStream<openxr::Vulkan>,
    swapchain: openxr::Swapchain<openxr::Vulkan>,
    swapchain_images: Vec<Arc<Image>>,
    frame_state: Option<openxr::FrameState>,
    space: openxr::Space,
    saved_poses: [(UnitQuaternion<f32>, Vector3<f32>); 2],
    saved_overlay_pose: Option<openxr::Posef>,

    device: Arc<Device>,
    queue: Arc<Queue>,
    vk_instance: Arc<Instance>,

    render_texture: Option<Arc<Image>>,
}
fn affine_to_posef(t: Affine3<f32>) -> openxr::Posef {
    let m = t.to_homogeneous();
    let r: Matrix3<f32> = m.fixed_columns::<3>(0).fixed_rows::<3>(0).into();
    let rotation = nalgebra::geometry::Rotation3::from_matrix(&r);
    let quaternion = UnitQuaternion::from_rotation_matrix(&rotation);
    let quaternion = &quaternion.as_ref().coords;
    let translation: nalgebra::Vector3<f32> =
        [m.data.0[3][0], m.data.0[3][1], m.data.0[3][2]].into();
    openxr::Posef {
        orientation: openxr::Quaternionf {
            x: quaternion.x,
            y: quaternion.y,
            z: quaternion.z,
            w: quaternion.w,
        },
        position: openxr::Vector3f {
            x: translation.x,
            y: translation.y,
            z: translation.z,
        },
    }
}

fn posef_to_nalgebra(posef: openxr::Posef) -> (UnitQuaternion<f32>, nalgebra::Vector3<f32>) {
    let quaternion = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        posef.orientation.w,
        posef.orientation.x,
        posef.orientation.y,
        posef.orientation.z,
    ));
    let translation: nalgebra::Vector3<f32> =
        [posef.position.x, posef.position.y, posef.position.z].into();
    (quaternion, translation)
}

impl OpenXr {
    fn create_vk_device(
        xr_instance: &openxr::Instance,
        xr_system: openxr::SystemId,
        instance: &Arc<Instance>,
    ) -> Result<(Arc<Device>, Arc<Queue>)> {
        let vk_requirements = xr_instance.graphics_requirements::<openxr::Vulkan>(xr_system)?;
        let physical_device = unsafe {
            let physical_device =
                xr_instance.vulkan_graphics_device(xr_system, instance.handle().as_raw() as _)?;
            vulkano::device::physical::PhysicalDevice::from_handle(
                instance.clone(),
                ash::vk::PhysicalDevice::from_raw(physical_device as _),
            )
        }?;
        let min_version = vulkano::Version::major_minor(
            vk_requirements.min_api_version_supported.major() as u32,
            vk_requirements.min_api_version_supported.minor() as u32,
        );
        if physical_device.api_version() < min_version {
            return Err(anyhow!(
                "Vulkan API version not supported {}",
                physical_device.api_version(),
            ));
        }
        let extensions = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            ..Default::default()
        };
        let raw_extensions = extensions
            .into_iter()
            .filter(|&(_, enabled)| enabled)
            .map(|(name, _)| std::ffi::CString::new(name).unwrap())
            .collect::<Vec<_>>();
        let raw_extensions = raw_extensions
            .iter()
            .map(|s| s.as_ptr())
            .collect::<Vec<_>>();
        let queue_family = physical_device
            .queue_family_properties()
            .iter()
            .position(|qf| qf.queue_flags.contains(QueueFlags::GRAPHICS))
            .context("No graphics queue found")?;
        log::debug!("queue family: {queue_family}");
        let queue_create_info = ash::vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family as u32)
            .queue_priorities(std::slice::from_ref(&1.0));
        let create_info = ash::vk::DeviceCreateInfo::default()
            .enabled_extension_names(&raw_extensions)
            .queue_create_infos(std::slice::from_ref(&queue_create_info));
        let vulkano_create_info = vulkano::device::DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: queue_family as u32,
                queues: vec![1.0],
                ..Default::default()
            }],
            enabled_extensions: extensions,
            physical_devices: [physical_device.clone()].into_iter().collect(),
            ..Default::default()
        };
        let (device, mut queues) = unsafe {
            vulkano::device::Device::from_handle(
                physical_device.clone(),
                ash::vk::Device::from_raw(
                    xr_instance
                        .create_vulkan_device(
                            xr_system,
                            get_instance_proc_addr,
                            physical_device.handle().as_raw() as _,
                            (&create_info) as *const _ as _,
                        )?
                        .map_err(ash::vk::Result::from_raw)? as _,
                ),
                vulkano_create_info,
            )
        };
        Ok((device, queues.next().unwrap()))
    }

    fn create_vk_instance(
        vk_instance_extensions: VkInstanceExtensions,
        xr_instance: &openxr::Instance,
        xr_system: openxr::SystemId,
    ) -> Result<Arc<Instance>> {
        let vk_requirements = xr_instance.graphics_requirements::<openxr::Vulkan>(xr_system)?;
        let extensions = *get_vulkan_library().supported_extensions();
        if let Some(unsupported) = vk_instance_extensions
            .difference(&extensions)
            .into_iter()
            .find_map(|(name, enabled)| enabled.then_some(name))
        {
            return Err(anyhow!(
                "Required instance extension {unsupported} not supported"
            ));
        }
        let vk_version = vulkano::Version::major_minor(
            vk_requirements.max_api_version_supported.major() as u32,
            vk_requirements.max_api_version_supported.minor() as u32,
        );
        let vk_version = vk_version.min(get_vulkan_library().api_version());

        let vulkano_create_info = vulkano::instance::InstanceCreateInfo {
            max_api_version: Some(vk_version),
            enabled_extensions: vk_instance_extensions,
            enabled_layers: vec![
                "VK_LAYER_KHRONOS_validation".to_owned(),
                //                "VK_LAYER_LUNARG_gfxreconstruct".to_owned(),
            ],
            ..Default::default()
        };
        let extensions = vk_instance_extensions
            .into_iter()
            .filter(|(_, enabled)| *enabled)
            .map(|(ext, _)| std::ffi::CString::new(ext).unwrap())
            .collect::<Vec<_>>();
        let extensions = extensions
            .iter()
            .map(|s| s.as_c_str().as_ptr())
            .collect::<Vec<_>>();
        let application_info =
            ash::vk::ApplicationInfo::default().api_version(vk_version.try_into().unwrap());
        let instance = unsafe {
            xr_instance.create_vulkan_instance(
                xr_system,
                get_instance_proc_addr,
                (&ash::vk::InstanceCreateInfo::default()
                    .enabled_extension_names(&extensions)
                    .application_info(&application_info)
                    .enabled_layer_names(&[
                        c"VK_LAYER_KHRONOS_validation".as_ptr(),
                        //                        c"VK_LAYER_LUNARG_gfxreconstruct".as_ptr(),
                    ])) as *const _ as _,
            )?
        }
        .map_err(ash::vk::Result::from_raw)?;
        let instance = ash::vk::Instance::from_raw(instance as _);
        Ok(unsafe {
            Instance::from_handle(get_vulkan_library().clone(), instance, vulkano_create_info)
        })
    }

    fn composition_layers<'a>(
        saved_overlay_pose: &'a Option<openxr::Posef>,
        swapchain: &'a openxr::Swapchain<openxr::Vulkan>,
        space: &'a openxr::Space,
        is_stereo: bool,
    ) -> Option<(
        openxr::CompositionLayerQuad<'a, openxr::Vulkan>,
        openxr::CompositionLayerQuad<'a, openxr::Vulkan>,
    )> {
        saved_overlay_pose.map(|overlay_posef| {
            let left = openxr::CompositionLayerQuad::<openxr::Vulkan>::new()
                .eye_visibility(EyeVisibility::LEFT)
                .pose(overlay_posef)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_rect(Rect2Di {
                            offset: Offset2Di { x: 0, y: 0 },
                            extent: Extent2Di {
                                width: crate::CAMERA_SIZE as i32,
                                height: crate::CAMERA_SIZE as i32,
                            },
                        }),
                )
                .space(space)
                .size(Extent2Df {
                    width: 1.0,
                    height: 1.0,
                });
            let right = openxr::CompositionLayerQuad::<openxr::Vulkan>::new()
                .eye_visibility(EyeVisibility::RIGHT)
                .pose(overlay_posef)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_rect(Rect2Di {
                            offset: Offset2Di {
                                x: if is_stereo {
                                    crate::CAMERA_SIZE as i32
                                } else {
                                    0
                                },
                                y: 0,
                            },
                            extent: Extent2Di {
                                width: crate::CAMERA_SIZE as i32,
                                height: crate::CAMERA_SIZE as i32,
                            },
                        }),
                )
                .space(space)
                .size(Extent2Df {
                    width: 1.0,
                    height: 1.0,
                });
            (left, right)
        })
    }

    /// render_size: Resolution of the swapchain image for a *single* eye.
    pub fn new(
        vk_instance_extensions: VkInstanceExtensions,
        render_size: UVec2,
        app_name: &str,
        app_version: u32,
    ) -> Result<Self> {
        let entry = unsafe { openxr::Entry::load()? };
        let mut extension = openxr::ExtensionSet::default();
        extension.extx_overlay = true;
        extension.khr_vulkan_enable2 = true;
        extension.khr_convert_timespec_time = true;
        let instance = entry.create_instance(
            &ApplicationInfo {
                application_name: app_name,
                application_version: app_version,
                api_version: openxr::Version::new(1, 1, 0),
                engine_name: "engine",
                engine_version: 0,
            },
            &extension,
            &[],
        )?;
        let system = instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
        let blend_modes = instance.enumerate_environment_blend_modes(
            system,
            openxr::ViewConfigurationType::PRIMARY_STEREO,
        )?;
        if !blend_modes.contains(&openxr::EnvironmentBlendMode::OPAQUE) {
            return Err(anyhow!("OpenXR runtime doesn't support opaque blend mode"));
        }
        let vk_instance = Self::create_vk_instance(vk_instance_extensions, &instance, system)?;
        let (device, queue) = Self::create_vk_device(&instance, system, &vk_instance)?;
        let binding = openxr::sys::GraphicsBindingVulkanKHR {
            ty: openxr::sys::GraphicsBindingVulkanKHR::TYPE,
            next: std::ptr::null(),
            instance: vk_instance.handle().as_raw() as _,
            physical_device: device.physical_device().handle().as_raw() as _,
            device: device.handle().as_raw() as _,
            queue_family_index: queue.queue_family_index(),
            queue_index: queue.queue_index(),
        };
        let info = openxr::sys::SessionCreateInfo {
            ty: openxr::sys::SessionCreateInfo::TYPE,
            next: &binding as *const _ as *const _,
            create_flags: Default::default(),
            system_id: system,
        };
        let mut out = openxr::sys::Session::NULL;
        let ret = unsafe { (instance.fp().create_session)(instance.as_raw(), &info, &mut out) };
        if ret.into_raw() < 0 {
            return Err(ret.into());
        }
        let (session, frame_waiter, frame_stream) = unsafe {
            openxr::Session::<openxr::Vulkan>::from_raw(instance.clone(), out, Box::new(()))
        };
        let formats = session.enumerate_swapchain_formats()?;
        if !formats
            .iter()
            .contains(&(vulkano::format::Format::R8G8B8A8_UNORM as u32))
        {
            return Err(anyhow!("No suitable format found for swapchain"));
        }
        let swapchain = session.create_swapchain(&openxr::SwapchainCreateInfo {
            array_size: 1,
            face_count: 1,
            create_flags: Default::default(),
            usage_flags: openxr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | openxr::SwapchainUsageFlags::TRANSFER_DST,
            format: vulkano::format::Format::R8G8B8A8_UNORM as u32,
            sample_count: 1,
            width: render_size.x * 2,
            height: render_size.y,
            mip_count: 1,
        })?;
        log::debug!("created swapchain");
        let swapchain_images = swapchain
            .enumerate_images()?
            .into_iter()
            .map(|handle| {
                let handle = ash::vk::Image::from_raw(handle);
                let raw_image = unsafe {
                    vulkano::image::sys::RawImage::from_handle_borrowed(
                        device.clone(),
                        handle,
                        ImageCreateInfo {
                            format: vulkano::format::Format::R8G8B8A8_UNORM,
                            extent: [render_size.x * 2, render_size.y, 1],
                            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                            ..Default::default()
                        },
                    )?
                };
                // SAFETY: OpenXR guarantees that the image is a swapchain image, thus has memory backing it.
                let image = unsafe { raw_image.assume_bound() };
                Ok::<_, anyhow::Error>(Arc::new(image))
            })
            .try_collect()?;
        log::debug!("got swapchain images");
        let space =
            session.create_reference_space(ReferenceSpaceType::STAGE, openxr::Posef::IDENTITY)?;
        log::debug!("created actions");

        Ok(Self {
            instance,
            session_state: openxr::SessionState::IDLE,
            session,
            frame_waiter,
            frame_stream,
            swapchain,
            swapchain_images,
            frame_state: None,
            space,
            saved_poses: [Default::default(); 2],
            saved_overlay_pose: None,

            vk_instance,
            device,
            queue,

            render_texture: None,
        })
    }
    pub fn vk_device(&self) -> (Arc<Device>, Arc<Queue>) {
        (self.device.clone(), self.queue.clone())
    }
    pub fn vk_instance(&self) -> Arc<Instance> {
        self.vk_instance.clone()
    }
}

unsafe extern "system" fn get_instance_proc_addr(
    instance: openxr::sys::platform::VkInstance,
    name: *const std::ffi::c_char,
) -> Option<unsafe extern "system" fn()> {
    let instance = ash::vk::Instance::from_raw(instance as _);
    let library = get_vulkan_library();
    library.get_instance_proc_addr(instance, name)
}

#[cfg(asdfasdf)]
impl Vr for OpenXr {
    fn acknowledge_quit(&mut self) {
        // intentionally left blank
    }

    type Error = OpenXrError;

    fn load_camera_paramter(&mut self) -> Option<StereoCamera> {
        self.camera_config
    }

    fn set_fallback_camera_config(&mut self, cfg: StereoCamera) {
        self.camera_config = Some(cfg);
    }

    fn submit_texture(
        &mut self,
        elapsed: Duration,
        fov: &[[f32; 2]; 2],
    ) -> Result<(), Self::Error> {
        log::trace!("submit texture");
        let frame_state = self.frame_state.as_ref().unwrap();
        let now = self.instance.now()?;
        let time_at_capture =
            openxr::Time::from_nanos((now.as_nanos() as u128 - elapsed.as_nanos()) as i64);
        let (view_state_flags, views) = self.session.locate_views(
            ViewConfigurationType::PRIMARY_STEREO,
            time_at_capture,
            &self.space,
        )?;
        let view_poses = if !view_state_flags.contains(ViewStateFlags::ORIENTATION_VALID)
            || !view_state_flags.contains(ViewStateFlags::POSITION_VALID)
        {
            log::trace!("view_state_flags: {:?}", view_state_flags);
            self.saved_poses
        } else {
            log::trace!("update pose");
            let poses = [0, 1].map(|id| posef_to_nalgebra(views[id].pose));
            self.saved_poses = poses;
            poses
        };
        let rotation_center = UnitQuaternion::from_quaternion(
            (view_poses[0].0.as_ref() + view_poses[1].0.as_ref()) / 2.0,
        );
        let center: Translation3<f32> = ((view_poses[0].1 + view_poses[1].1) / 2.0).into();
        let hmd_transform = center.to_homogeneous() * rotation_center.to_homogeneous();
        if self.reposition {
            self.position_mode.reposition(hmd_transform);
            self.reposition = false;
        }
        let transform = self.position_mode.transform(hmd_transform);
        let overlay_posef = affine_to_posef(transform);
        self.saved_overlay_pose = Some(overlay_posef);
        if self.display_mode.projection_mode().is_some() {
            // Apply projection
            let image = self.swapchain.acquire_image()? as usize;
            let view_transforms = [
                Translation3::from(view_poses[0].1).to_homogeneous()
                    * view_poses[0].0.to_homogeneous(),
                Translation3::from(view_poses[1].1).to_homogeneous()
                    * view_poses[1].0.to_homogeneous(),
            ];
            let ipd = view_poses[1].1.x - view_poses[0].1.x;
            self.swapchain.wait_image(openxr::Duration::INFINITE)?;
            let output = self.swapchain_images[image].clone();
            let projector = self.projector.as_mut().unwrap();
            projector.update_mvps(transform.matrix(), fov, &view_transforms, &hmd_transform)?;
            projector.set_ipd(ipd);
            let future = projector.project(
                self.allocator.clone(),
                self.cmdbuf_allocator.clone(),
                vulkano::sync::future::now(self.device.clone()),
                &self.queue,
                output,
            )?;
            future.flush()?;
            future.then_signal_fence().wait(None)?;
        } else {
            self.render_texture.take();
        }
        self.swapchain.release_image()?;
        let (left, right) = Self::composition_layers(
            &self.saved_overlay_pose,
            &self.swapchain,
            &self.space,
            self.display_mode.is_stereo(),
        )
        .unwrap();
        self.frame_stream.end(
            frame_state.predicted_display_time,
            EnvironmentBlendMode::OPAQUE,
            &[&left, &right],
        )?;
        Ok(())
    }

    fn is_synchronized(&self) -> bool {
        true
    }

    fn refresh(&mut self) -> Result<(), Self::Error> {
        log::trace!("refresh");
        if !self.overlay_visible {
            std::thread::sleep(std::time::Duration::from_millis(100));
            return Ok(());
        }
        let frame_state = self.frame_state.insert(self.frame_waiter.wait()?);
        self.frame_stream.begin()?;
        if let Some((left, right)) = Self::composition_layers(
            &self.saved_overlay_pose,
            &self.swapchain,
            &self.space,
            self.display_mode.is_stereo(),
        ) {
            log::trace!("reuse last image {:?}", frame_state.predicted_display_time);
            self.frame_stream.end(
                frame_state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                &[&left, &right],
            )?;
        } else {
            log::trace!("no saved overlay pose");
            self.frame_stream.end(
                frame_state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                &[],
            )?;
        }
        Ok(())
    }

    fn get_render_texture(&mut self) -> Result<Option<Arc<Image>>, Self::Error> {
        if (self.session_state != openxr::SessionState::FOCUSED
            && self.session_state != openxr::SessionState::VISIBLE
            && self.session_state != openxr::SessionState::SYNCHRONIZED
            && self.session_state != openxr::SessionState::READY)
            || !self.overlay_visible
        {
            log::debug!("VR runtime not ready");
            return Ok(None);
        }
        let frame_state = self.frame_state.insert(self.frame_waiter.wait()?);
        self.frame_stream.begin()?;
        if !frame_state.should_render {
            self.frame_stream.end(
                frame_state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                &[],
            )?;
            return Ok(None);
        }
        if self.display_mode.projection_mode().is_some() {
            log::trace!("render to intermediate texture");
            assert!(self.render_texture.is_some());
            return Ok(self.render_texture.clone());
        }
        log::trace!("render to swapchain image");
        let image = self.swapchain.acquire_image()? as usize;
        self.render_texture = Some(self.swapchain_images[image].clone());
        self.swapchain.wait_image(openxr::Duration::INFINITE)?;
        Ok(self.render_texture.clone())
    }

    fn set_display_mode(&mut self, mode: DisplayMode) -> Result<(), Self::Error> {
        self.display_mode = mode;
        if let Some(projection_mode) = self.display_mode.projection_mode() {
            let camera_calib = self.load_camera_paramter();
            if self.projector.is_none() {
                self.render_texture = Some(crate::create_submittable_image(self.device.clone())?);
                let mut projector = crate::projection::Projection::new(
                    self.device.clone(),
                    self.allocator.clone(),
                    self.descriptor_set_allocator.clone(),
                    self.render_texture.as_ref().unwrap(),
                    1.0,
                    &camera_calib,
                    ImageLayout::ColorAttachmentOptimal,
                )?;
                projector.set_mode(projection_mode);
                self.projector = Some(projector);
            }
        } else {
            self.render_texture = None;
            self.projector = None;
        }
        Ok(())
    }

    fn show_overlay(&mut self) -> Result<(), Self::Error> {
        if !self.overlay_visible {
            log::debug!("show overlay, {:?}", self.session_state);
            self.overlay_visible = true;
        }
        Ok(())
    }

    fn hide_overlay(&mut self) -> Result<(), Self::Error> {
        if self.overlay_visible {
            // HACK!: show a zero sized quad to hide the overlay. It's a bit ugly we
            // blocks the mainloop here to wait for a frame
            let frame_state = self.frame_waiter.wait()?;
            self.frame_stream.begin()?;
            let empty = openxr::CompositionLayerQuad::<openxr::Vulkan>::new()
                .eye_visibility(EyeVisibility::BOTH)
                .pose(openxr::Posef::IDENTITY)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(&self.swapchain)
                        .image_rect(Rect2Di {
                            offset: Offset2Di { x: 0, y: 0 },
                            extent: Extent2Di {
                                width: 1,
                                height: 1,
                            },
                        }),
                )
                .space(&self.space)
                .size(Extent2Df {
                    width: 0.0,
                    height: 0.0,
                });
            self.frame_stream.end(
                frame_state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                &[&empty],
            )?;
            self.overlay_visible = false;
        }
        Ok(())
    }

    fn set_position_mode(&mut self, mode: PositionMode) -> Result<(), Self::Error> {
        self.position_mode = mode;
        if matches!(mode, PositionMode::Sticky { .. }) {
            self.reposition = true;
        }
        Ok(())
    }

    fn poll_next_event(&mut self) -> Result<Option<Event>, Self::Error> {
        let mut event = EventDataBuffer::default();
        let ret = loop {
            let event = self.instance.poll_event(&mut event)?;
            let Some(event) = event else { break None };
            use openxr::Event as XrEvent;
            match event {
                XrEvent::InstanceLossPending(_) => break Some(Event::RequestExit),
                XrEvent::SessionStateChanged(ssc) => {
                    use openxr::SessionState;
                    log::debug!(
                        "session state changed: {:?}, visible: {}",
                        ssc.state(),
                        self.overlay_visible
                    );
                    self.session_state = ssc.state();
                    match self.session_state {
                        SessionState::EXITING | SessionState::LOSS_PENDING => {
                            self.session.end()?;
                            break Some(Event::RequestExit);
                        }
                        SessionState::READY => {
                            log::debug!("begin session");
                            self.session
                                .begin(openxr::ViewConfigurationType::PRIMARY_STEREO)?;
                        }
                        SessionState::STOPPING => {
                            if self.overlay_visible {
                                self.session.end()?;
                            }
                        }
                        _ => (),
                    }
                }
                XrEvent::EventsLost(_) => (), // ? should we do something?
                _ => (),
            }
        };
        Ok(ret)
    }

    fn update_action_state(&mut self) -> Result<(), Self::Error> {
        self.session
            .sync_actions(&[openxr::ActiveActionSet::new(&self.action_set)])?;
        Ok(())
    }

    fn get_action_state(&self, action: Action) -> Result<bool, Self::Error> {
        Ok(match action {
            Action::Button1 => self
                .action_button1
                .state(&self.session, openxr::Path::NULL)?,
            Action::Button2 => self
                .action_button2
                .state(&self.session, openxr::Path::NULL)?,
            Action::Debug => self.action_debug.state(&self.session, openxr::Path::NULL)?,
            Action::Reposition => self
                .action_reposition
                .state(&self.session, openxr::Path::NULL)?,
        }
        .current_state)
    }

    fn wait_for_ready(&mut self) -> Result<(), Self::Error> {
        log::debug!("current state {:?}", self.session_state);
        while self.session_state != openxr::SessionState::FOCUSED
            && self.session_state != openxr::SessionState::VISIBLE
            && self.session_state != openxr::SessionState::SYNCHRONIZED
            && self.session_state != openxr::SessionState::READY
        {
            log::debug!("VR runtime not ready");
            self.poll_next_event()?;
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        while self.session_state != openxr::SessionState::SYNCHRONIZED
            && self.session_state != openxr::SessionState::FOCUSED
            && self.session_state != openxr::SessionState::VISIBLE
        {
            let frame_state = self.frame_state.insert(self.frame_waiter.wait()?);
            self.frame_stream.begin()?;
            self.frame_stream.end(
                frame_state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                &[],
            )?;
            self.poll_next_event()?;
        }
        Ok(())
    }
}
