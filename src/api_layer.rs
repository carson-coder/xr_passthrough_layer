use ash::vk::Handle as _;
use glam::{UVec2, Vec3};
use log::{debug, error, warn};
use openxr::{
    AsHandle, CompositionLayerFlags, Extent2Di, Offset2Di, Rect2Di, ReferenceSpaceType,
    SwapchainCreateFlags, SwapchainCreateInfo, SwapchainSubImage, SwapchainUsageFlags,
    ViewStateFlags, sys::Handle,
};
use quark::{Hooked as _, Low as _, prelude::*, try_xr, types::AnySession};
use smallvec::smallvec;
use std::{
    collections::{HashMap, HashSet},
    ffi::{CStr, c_char},
    hint::unreachable_unchecked,
    mem::MaybeUninit,
    sync::{Arc, LazyLock, OnceLock},
};
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, ImageBlit,
        allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    },
    descriptor_set::allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator},
    device::QueueCreateInfo,
    image::{ImageCreateInfo, ImageLayout, ImageUsage},
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    sync::GpuFuture,
};
struct CameraResources {
    /// Extra swapchain for our own rendering needs
    swapchain: openxr::Swapchain<openxr::Vulkan>,
    size: UVec2,
    images: Vec<Arc<vulkano::image::Image>>,
    camera: crate::camera::CameraThread,
    pp: crate::pipeline::Pipeline,
}
enum SessionState {
    Running {
        view_type: openxr::sys::ViewConfigurationType,
    },
    Idle,
    RunningWithPassthrough {
        view_type: openxr::sys::ViewConfigurationType,
        camera: CameraResources,
        passthrough: Arc<PassthroughInner>,
        image_index: Option<u32>,
        should_render: Option<bool>,
    },
    IdleWithPassthrough {
        passthrough: Arc<PassthroughInner>,
    },
}
impl SessionState {
    fn view_type(&self) -> Option<openxr::sys::ViewConfigurationType> {
        match self {
            &Self::Running { view_type } | &Self::RunningWithPassthrough { view_type, .. } => {
                Some(view_type)
            }
            _ => None,
        }
    }
    fn get_or_add_passthrough(
        &mut self,
        f: impl FnOnce() -> PassthroughInner,
        create_camera_resources: impl FnOnce() -> Result<CameraResources, XrErr>,
    ) -> Result<&mut Arc<PassthroughInner>, XrErr> {
        match self {
            &mut Self::Running { view_type } => {
                *self = Self::RunningWithPassthrough {
                    view_type,
                    camera: create_camera_resources()?,
                    passthrough: f().into(),
                    image_index: None,
                    should_render: None,
                };
            }
            Self::Idle => {
                *self = Self::IdleWithPassthrough {
                    passthrough: f().into(),
                };
            }
            _ => (),
        }
        match self {
            Self::RunningWithPassthrough { passthrough, .. }
            | Self::IdleWithPassthrough { passthrough } => Ok(passthrough),
            _ => unsafe { unreachable_unchecked() },
        }
    }
}
fn xrcvt(e: XrErr) -> Result<(), XrErr> {
    if e == XrErr::SUCCESS { Ok(()) } else { Err(e) }
}
struct SessionDataInner {
    device: Arc<vulkano::device::Device>,
    queue: Arc<vulkano::device::Queue>,
    state: SessionState,
    system_id: openxr::SystemId,
    allocator: Arc<dyn MemoryAllocator>,
    cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
    descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
    space: openxr::Space,
}
#[derive(Default)]
pub struct SessionData {
    inner: Option<SessionDataInner>,
}
impl SessionData {
    unsafe fn end_frame(
        &mut self,
        session: &quark::types::AnySession,
        instance: &openxr::Instance,
        info: &openxr::sys::FrameEndInfo,
    ) -> Result<(), XrErr> {
        let Some(data) = &mut self.inner else {
            // We are not wrapping this session, passed it through.
            debug!("Unhandled session, like passthrough extension wasn't enabled");
            return xrcvt(unsafe { (instance.fp().end_frame)(session.as_handle(), info) });
        };
        let quark::types::AnySession::Vulkan(xr_vk_session) = session else {
            unreachable!()
        };
        let SessionState::RunningWithPassthrough {
            camera,
            image_index,
            view_type,
            ..
        } = &mut data.state
        else {
            if matches!(data.state, SessionState::Running { .. }) {
                return Ok(());
            } else {
                return Err(XrErr::ERROR_SESSION_NOT_RUNNING);
            }
        };
        let layers = unsafe {
            std::slice::from_raw_parts(
                // Safety: Option<&T> and *const T are bitwise identical.
                info.layers as *const Option<&openxr::sys::CompositionLayerBaseHeader>,
                info.layer_count as _,
            )
        };
        let has_passthrough = layers
            .iter()
            .filter(|l| {
                l.is_some_and(|l| l.ty == openxr::sys::CompositionLayerPassthroughHTC::TYPE)
            })
            .count();
        if has_passthrough > 1 {
            warn!("More than one passthrough layer, not supported");
            return Err(XrErr::ERROR_VALIDATION_FAILURE);
        }
        if has_passthrough == 0 || image_index.is_none() {
            // No passthrough layer, we can just pass the frame to openxr.
            return xrcvt(unsafe { (instance.fp().end_frame)(session.as_handle(), info) });
        }
        let (view_state_flags, view_locations) =
            xr_vk_session.locate_views(*view_type, info.display_time, &data.space)?;
        if !view_state_flags
            .contains(ViewStateFlags::POSITION_VALID | ViewStateFlags::ORIENTATION_VALID)
        {
            warn!("Pose or orientation invalid {view_state_flags:?}, skip passthrough layer");
            let layers = layers
                .iter()
                .filter(|l| {
                    l.is_none_or(|l| l.ty != openxr::sys::CompositionLayerPassthroughHTC::TYPE)
                })
                .collect::<Vec<_>>();
            let mut info = *info;
            info.layer_count = layers.len() as _;
            info.layers = layers.as_ptr() as *const _;
            return xrcvt(unsafe { (instance.fp().end_frame)(session.as_handle(), &info) });
        }
        let image_index = image_index.take().unwrap();
        // Copy camera image to swapchain
        let mut cmdbuf = AutoCommandBufferBuilder::primary(
            data.cmdbuf_allocator.clone(),
            data.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| {
            warn!("Failed to create command buffer {e:#}");
            XrErr::ERROR_RUNTIME_FAILURE
        })?;
        let camera_frame = camera.camera.frame();
        camera.pp.maybe_postprocess(&camera_frame).map_err(|e| {
            warn!("Failed to postprocess camera frame {e:#}");
            XrErr::ERROR_RUNTIME_FAILURE
        })?;
        let (camera_image, fut) = camera.pp.image();
        let camera_extent = camera.pp.image_extent();
        cmdbuf
            .blit_image(BlitImageInfo {
                src_image: camera_image.clone(),
                dst_image: camera.images[image_index as usize].clone(),
                regions: smallvec![ImageBlit {
                    src_subresource: camera_image.subresource_layers(),
                    src_offsets: [[0, 0, 0], [camera_extent[0], camera_extent[1], 1],],
                    dst_subresource: camera.images[image_index as usize].subresource_layers(),
                    dst_offsets: [[0, 0, 0], [camera.size.x * 2, camera.size.y, 1],],
                    ..Default::default()
                }],
                ..BlitImageInfo::new(
                    camera_image.clone(),
                    camera.images[image_index as usize].clone(),
                )
            })
            .map_err(|e| {
                warn!("Failed to blit image {e:#}");
                XrErr::ERROR_RUNTIME_FAILURE
            })?;
        let cmdbuf = cmdbuf.build().map_err(|e| {
            warn!("Failed to build command buffer {e:#}");
            XrErr::ERROR_RUNTIME_FAILURE
        })?;
        fut.then_execute(data.queue.clone(), cmdbuf)
            .map_err(|e| {
                warn!("Failed to execute command buffer {e:#}");
                XrErr::ERROR_RUNTIME_FAILURE
            })?
            .then_signal_fence_and_flush()
            .map_err(|e| {
                warn!("Failed to flush command buffer {e:#}");
                XrErr::ERROR_RUNTIME_FAILURE
            })?
            .wait(None)
            .map_err(|e| {
                warn!("Failed to wait for fence {e:#}");
                XrErr::ERROR_RUNTIME_FAILURE
            })?;
        assert_eq!(view_locations.len(), 2);
        camera.swapchain.release_image()?;
        log::trace!("{:?}", view_locations[0].fov);
        log::trace!("{:?}", view_locations[1].fov);
        log::trace!("{:?}", view_locations[0].pose);
        log::trace!("{:?}", view_locations[1].pose);
        let views = [
            openxr::CompositionLayerProjectionView::new()
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(&camera.swapchain)
                        .image_rect(Rect2Di {
                            offset: Offset2Di { x: 0, y: 0 },
                            extent: Extent2Di {
                                width: camera.size.x as _,
                                height: camera.size.y as _,
                            },
                        }),
                )
                .pose(view_locations[0].pose)
                .fov(view_locations[0].fov),
            openxr::CompositionLayerProjectionView::new()
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(&camera.swapchain)
                        .image_rect(Rect2Di {
                            offset: Offset2Di {
                                x: camera.size.x as _,
                                y: 0,
                            },
                            extent: Extent2Di {
                                width: camera.size.x as _,
                                height: camera.size.y as _,
                            },
                        }),
                )
                .pose(view_locations[1].pose)
                .fov(view_locations[1].fov),
        ];

        let replaced_passthrough_layer = openxr::CompositionLayerProjection::new()
            .space(&data.space)
            .layer_flags(CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA)
            .views(&views);
        let pos = layers
            .iter()
            .position(|l| {
                l.is_some_and(|l| l.ty == openxr::sys::CompositionLayerPassthroughHTC::TYPE)
            })
            .unwrap();
        let mut layers = layers.to_vec();
        layers[pos] = unsafe {
            std::mem::transmute::<
                Option<&openxr::sys::CompositionLayerProjection>,
                Option<&openxr::sys::CompositionLayerBaseHeader>,
            >(Some(replaced_passthrough_layer.as_raw()))
        };
        let mut info = *info;
        info.layers = layers.as_ptr() as *const _;
        xrcvt(unsafe { (instance.fp().end_frame)(session.as_handle(), &info) })
    }
}

// Define your instance data
pub struct InstanceData {
    is_passthrough_enabled: bool,
    /// Mapping raw vulkan VkInstance handles to the api version it was created with.
    instance_api_version: HashMap<u64, vulkano::Version>,
}

struct PassthroughMesh<'a> {
    vertices: &'a [openxr::Vector3f],
    indices: &'a [u32],
    base_space: openxr::sys::Space,
    time: openxr::sys::Time,
    pose: openxr::Posef,
    scale: Vec3,
}

// SAFETY: `fp` must a function that takes arguments like this: `fp($args..., capacity, count,
// out_array)`. It must return the number of elements it has written to `out_array` via `count`,
// and it must not right more than `capacity` elements into `out_array`.
#[allow(edition_2024_expr_fragment_specifier)]
macro_rules! call_enumerate {
    ($f:expr => [$size:literal]; $($args:expr),*) => {
        {
            let fp = $f;
            let mut buf =  smallvec::SmallVec::<[_; $size]>::new();
            let mut size_out = 0u32;
            let ret = fp($($args),* , ($size) as u32, &mut size_out, buf.as_mut_ptr());
            if ret != XrErr::SUCCESS && ret != XrErr::ERROR_SIZE_INSUFFICIENT {
                Err(ret)
            } else if ret == XrErr::SUCCESS {
                assert!((size_out as usize) < $size, "{} written more than capacity", stringify!($fp));
                buf.set_len(size_out as usize);
                Ok(buf)
            } else {
                buf.reserve(size_out as usize);
                let ret = fp($($args),* , size_out, &mut size_out, buf.as_mut_ptr());
                if ret != XrErr::SUCCESS {
                    Err(ret)
                } else {
                    assert!(size_out as usize <= buf.capacity(), "{} written more than capacity", stringify!($f));
                    buf.set_len(size_out as usize);
                    Ok(buf)
                }
            }
        }
    };
}

impl<'a> PassthroughMesh<'a> {
    unsafe fn find_mesh(layer: &'a openxr::sys::CompositionLayerPassthroughHTC) -> Option<Self> {
        let mut curr = unsafe { &*(layer.next as *const openxr::sys::CompositionLayerBaseHeader) };
        while curr.ty != openxr::sys::PassthroughMeshTransformInfoHTC::TYPE {
            if curr.next.is_null() {
                return None;
            }
            curr = unsafe { &*(curr.next as *const _) };
        }
        let curr =
            unsafe { &*(curr as *const _ as *const openxr::sys::PassthroughMeshTransformInfoHTC) };
        Some(Self {
            vertices: unsafe { std::slice::from_raw_parts(curr.vertices, curr.vertex_count as _) },
            indices: unsafe { std::slice::from_raw_parts(curr.indices, curr.index_count as _) },
            base_space: curr.base_space,
            time: curr.time,
            scale: Vec3::new(curr.scale.x, curr.scale.y, curr.scale.z),
            pose: curr.pose,
        })
    }
}

pub struct PassthroughInner {}

impl Drop for PassthroughInner {
    fn drop(&mut self) {
        debug!("Passthrough destroyed")
    }
}

pub struct PassthroughData {
    _inner: Arc<PassthroughInner>,
    form: openxr::sys::PassthroughFormHTC,

    /// Allocate 1 byte whose address is used as an unique id for the passthrough
    /// object.
    unique: Box<MaybeUninit<u8>>,
}

pub struct PassthroughFactory;

unsafe impl quark::Factory<PassthroughData> for PassthroughFactory {
    unsafe fn create(
        args: quark::CreateArgs<openxr::sys::PassthroughHTC>,
    ) -> Result<(quark::Facade<openxr::sys::PassthroughHTC>, PassthroughData), XrErr> {
        let mut session = args.0.registered_with_hook_mut()?;
        let (session_data, session) = session.both();
        let quark::types::AnySession::Vulkan(session) = session else {
            warn!("Creating passthrough out of a non-vulkan session is not supported");
            return Err(XrErr::ERROR_FEATURE_UNSUPPORTED);
        };
        debug!("Creating passthrough {:#x}", session.as_handle().into_raw());
        let inner = session_data
            .inner
            .as_mut()
            .ok_or(XrErr::ERROR_FEATURE_UNSUPPORTED)?;
        let form = unsafe { *args.1 }.form;

        let view_type = inner.state.view_type();
        let passthrough = inner
            .state
            .get_or_add_passthrough(
                || PassthroughInner {},
                || {
                    create_camera_resources(
                        session.instance(),
                        session,
                        inner.system_id,
                        view_type.unwrap(),
                        &inner.device,
                        &inner.queue,
                        &inner.allocator,
                        &inner.cmdbuf_allocator,
                        &inner.descriptor_set_allocator,
                    )
                },
            )?
            .clone();

        let ret = PassthroughData {
            _inner: passthrough,
            form,
            unique: Box::new(MaybeUninit::uninit()),
        };
        let handle =
            openxr::sys::PassthroughHTC::from_raw(&*ret.unique as *const _ as usize as u64);
        unsafe { *args.2 = handle };
        let (high, _) = unsafe { handle.into_high(args) }?;
        Ok((high, ret))
    }
}

impl quark::Hook for PassthroughData {
    type Target = openxr::sys::PassthroughHTC;
    type Factory = PassthroughFactory;
}

const REQUIRED_VK_INSTANCE_EXTENSIONS: &[&CStr] = &[
    ash::vk::KHR_EXTERNAL_MEMORY_CAPABILITIES_NAME,
    ash::vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME,
    ash::vk::KHR_XCB_SURFACE_NAME,
];

const REQUIRED_VK_DEVICE_EXTENSIONS: &[&CStr] = &[ash::vk::KHR_COPY_COMMANDS2_NAME];

static VULKAN_LIBRARY: LazyLock<Arc<vulkano::library::VulkanLibrary>> =
    LazyLock::new(|| vulkano::library::VulkanLibrary::new().unwrap());

unsafe fn get_vulkan_extensions_override(
    instance: openxr::sys::Instance,
    system_id: openxr::sys::SystemId,
    cap: u32,
    count: *mut u32,
    buffer: *mut c_char,
    required_extensions: &[&CStr],
    original_fn: unsafe extern "system" fn(
        openxr::sys::Instance,
        openxr::sys::SystemId,
        u32,
        *mut u32,
        *mut c_char,
    ) -> XrErr,
) -> XrErr {
    if cap != 0 && buffer.is_null() {
        return XrErr::ERROR_VALIDATION_FAILURE;
    }

    let mut len = 0;
    let ret = unsafe { (original_fn)(instance, system_id, 0, &mut len, std::ptr::null_mut()) };
    if ret != XrErr::SUCCESS {
        return ret;
    }
    let mut buf: Vec<u8> = Vec::with_capacity(len as _);
    let ret = unsafe {
        (original_fn)(
            instance,
            system_id,
            len,
            &mut len,
            buf.spare_capacity_mut().as_mut_ptr() as *mut _,
        )
    };
    if ret != XrErr::SUCCESS {
        return ret;
    }
    // SAFETY: `original_fn` should've written `len` bytes into `buf`.
    unsafe {
        buf.set_len((len - 1) as _); // ignore the nul
    }
    let extensions = buf.split(|b| *b == b' ').collect::<HashSet<_>>();
    let extra_extensions = required_extensions
        .iter()
        .map(|e| e.to_bytes())
        .filter(|e| !extensions.contains(e));
    // First, check if there is enough space.
    let extra_len = extra_extensions.clone().fold(0, |a, b| a + b.len() + 1);
    let total_len = buf.len() + extra_len
        - if buf.is_empty() && extra_len > 0 {
            // If buf is empty string, then we don't need to put a space between it and
            // extra_extensions
            1
        } else {
            0
        };
    let total_len: u32 = total_len.try_into().unwrap();

    unsafe { *count = total_len + 1 }; // add the nul
    if total_len > cap {
        return XrErr::SUCCESS;
    }

    // Can't create &mut [] from `buffer` because it might be uninitialized.
    unsafe {
        buffer.copy_from_nonoverlapping(buf.as_ptr() as *const _, buf.len());
        let mut pos = buffer.add(buf.len());
        for e in extra_extensions {
            if !std::ptr::eq(pos, buffer) {
                pos.write(b' ' as _);
                pos = pos.add(1);
            }
            pos.copy_from_nonoverlapping((*e).as_ptr() as *const _, e.len());
            pos = pos.add(e.len());
        }
        pos.write(0);
        assert_eq!(pos.offset_from(buffer), total_len as isize);
    }

    XrErr::SUCCESS
}

unsafe extern "system" fn get_vulkan_instance_extensions(
    instance: openxr::sys::Instance,
    system_id: openxr::sys::SystemId,
    cap: u32,
    count: *mut u32,
    buffer: *mut c_char,
) -> XrErr {
    let wrapped_instance = try_xr!(instance.registered_with_hook());
    let Some(vulkan_enable) = wrapped_instance.get().exts().khr_vulkan_enable else {
        return XrErr::ERROR_VALIDATION_FAILURE;
    };
    unsafe {
        get_vulkan_extensions_override(
            instance,
            system_id,
            cap,
            count,
            buffer,
            REQUIRED_VK_INSTANCE_EXTENSIONS,
            vulkan_enable.get_vulkan_instance_extensions,
        )
    }
}

unsafe extern "system" fn get_vulkan_device_extensions(
    instance: openxr::sys::Instance,
    system_id: openxr::sys::SystemId,
    cap: u32,
    count: *mut u32,
    buffer: *mut c_char,
) -> XrErr {
    let wrapped_instance = try_xr!(instance.registered_with_hook());
    let Some(vulkan_enable) = wrapped_instance.get().exts().khr_vulkan_enable else {
        return XrErr::ERROR_VALIDATION_FAILURE;
    };
    unsafe {
        get_vulkan_extensions_override(
            instance,
            system_id,
            cap,
            count,
            buffer,
            REQUIRED_VK_DEVICE_EXTENSIONS,
            vulkan_enable.get_vulkan_device_extensions,
        )
    }
}

unsafe extern "system" fn create_vulkan_instance(
    instance: openxr::sys::Instance,
    create_info: *const openxr::sys::VulkanInstanceCreateInfoKHR,
    out_vk_instance: *mut ash::vk::Instance,
    result: *mut ash::vk::Result,
) -> XrErr {
    debug!("Creating Vulkan instance");
    let mut wrapped_instance = try_xr!(instance.registered_with_hook_mut());
    let Some(vulkan_enable2) = wrapped_instance.get().exts().khr_vulkan_enable2 else {
        warn!("Calling create_vulkan_instance without khr_vulkan_enable2");
        return XrErr::ERROR_VALIDATION_FAILURE;
    };
    let mut vk_create_info =
        unsafe { *((*create_info).vulkan_create_info as *const ash::vk::InstanceCreateInfo<'_>) };
    let requested_extensions = if vk_create_info.enabled_extension_count > 0 {
        let extensions = unsafe {
            std::slice::from_raw_parts(
                vk_create_info.pp_enabled_extension_names,
                vk_create_info.enabled_extension_count as _,
            )
        };
        extensions
            .iter()
            .map(|&e| unsafe { CStr::from_ptr(e) })
            .collect::<HashSet<_>>()
    } else {
        HashSet::new()
    };
    let extra_extensions = REQUIRED_VK_INSTANCE_EXTENSIONS
        .iter()
        .filter(|&&e| !requested_extensions.contains(e));
    debug!("Extensions: {:?}", requested_extensions);
    let mut new_create_info = unsafe { *create_info };
    let mut new_extensions = Vec::new();
    if extra_extensions.clone().count() > 0 {
        new_extensions.extend(extra_extensions.map(|e| e.as_ptr()));
        new_extensions.extend(requested_extensions.into_iter().map(|e| e.as_ptr()));
        vk_create_info.enabled_extension_count = new_extensions.len() as _;
        vk_create_info.pp_enabled_extension_names = new_extensions.as_ptr();
        new_create_info.vulkan_create_info = &vk_create_info as *const _ as *const _;
    }
    let ret = unsafe {
        (vulkan_enable2.create_vulkan_instance)(
            instance,
            &new_create_info,
            out_vk_instance as *mut _,
            result as *mut _,
        )
    };
    if ret != XrErr::SUCCESS {
        return ret;
    }

    wrapped_instance.hook().instance_api_version.insert(
        unsafe { *out_vk_instance }.as_raw(),
        unsafe { *vk_create_info.p_application_info }
            .api_version
            .into(),
    );
    ret
}

unsafe extern "system" fn create_vulkan_device(
    instance: openxr::sys::Instance,
    create_info: *const openxr::sys::VulkanDeviceCreateInfoKHR,
    device: *mut ash::vk::Device,
    result: *mut ash::vk::Result,
) -> XrErr {
    debug!("Creating Vulkan device");
    let wrapped_instance = try_xr!(instance.registered_with_hook());
    let Some(vulkan_enable2) = wrapped_instance.get().exts().khr_vulkan_enable2 else {
        return XrErr::ERROR_VALIDATION_FAILURE;
    };
    let mut vk_create_info =
        unsafe { *((*create_info).vulkan_create_info as *const ash::vk::DeviceCreateInfo<'_>) };
    let requested_extensions = if vk_create_info.enabled_extension_count > 0 {
        let extensions = unsafe {
            std::slice::from_raw_parts(
                vk_create_info.pp_enabled_extension_names,
                vk_create_info.enabled_extension_count as _,
            )
        };
        extensions
            .iter()
            .map(|&e| unsafe { CStr::from_ptr(e) })
            .collect::<HashSet<_>>()
    } else {
        HashSet::new()
    };
    let extra_extensions = REQUIRED_VK_DEVICE_EXTENSIONS
        .iter()
        .filter(|&&e| !requested_extensions.contains(e));
    debug!("Extensions: {:?}", requested_extensions);
    debug!(
        "Extra Extensions: {:?}",
        extra_extensions.clone().collect::<Vec<_>>()
    );
    let mut new_create_info = unsafe { *create_info };
    let mut new_extensions = Vec::new();
    if extra_extensions.clone().count() > 0 {
        new_extensions.extend(extra_extensions.map(|e| e.as_ptr()));
        new_extensions.extend(requested_extensions.into_iter().map(|e| e.as_ptr()));
        vk_create_info.enabled_extension_count = new_extensions.len() as _;
        vk_create_info.pp_enabled_extension_names = new_extensions.as_ptr();
        new_create_info.vulkan_create_info = &vk_create_info as *const _ as *const _;
    }

    unsafe {
        (vulkan_enable2.create_vulkan_device)(
            instance,
            &new_create_info,
            device as *mut _,
            result as *mut _,
        )
    }
}

static LOG_INIT: OnceLock<()> = OnceLock::new();

pub struct InstanceFactory;
unsafe impl quark::Factory<InstanceData> for InstanceFactory {
    // SAFETY: `args` must be valid
    unsafe fn create(
        args: quark::CreateArgs<openxr::sys::Instance>,
    ) -> Result<(openxr::Instance, InstanceData), XrErr> {
        LOG_INIT.get_or_init(env_logger::init);
        debug!("Creating OpenXR instance");
        let (info, api_layer_info, instance) = args;
        let mut instance_info = unsafe { *info };
        let mut new_exts = Vec::new();
        let enabled = if instance_info.enabled_extension_count != 0 {
            let exts = unsafe {
                std::slice::from_raw_parts(
                    instance_info.enabled_extension_names,
                    instance_info.enabled_extension_count as _,
                )
            };
            let it = exts.iter().map(|e| unsafe { CStr::from_ptr(*e) });
            let dbge = it.clone().collect::<Vec<_>>();
            for e in dbge {
                debug!("Extension: {:?}", e.to_str());
            }
            let enabled = it.clone().any(|e| e == c"XR_HTC_passthrough");
            if enabled {
                debug!("Passthrough extension enabled");
                new_exts.reserve(exts.len());
                new_exts.extend(
                    it.filter(|e| *e != c"XR_HTC_passthrough")
                        .map(|e| e.as_ptr()),
                );
                instance_info.enabled_extension_names = new_exts.as_ptr();
                instance_info.enabled_extension_count = new_exts.len() as _;
            }
            enabled
        } else {
            false
        };
        let layer_info = unsafe { &*api_layer_info };
        let r = unsafe {
            ((*layer_info.next_info).next_create_api_layer_instance)(
                &instance_info,
                api_layer_info,
                instance,
            )
        };
        if r != XrErr::SUCCESS {
            return Err(r);
        }
        let (high, _create_info) = unsafe { (*instance).into_high(args) }?;
        let this = InstanceData {
            is_passthrough_enabled: enabled,
            instance_api_version: HashMap::new(),
        };
        Ok((high, this))
    }
}
impl quark::Hook for InstanceData {
    type Target = openxr::sys::Instance;
    type Factory = InstanceFactory;
}

impl quark::Hook for SessionData {
    type Target = openxr::sys::Session;
    type Factory = quark::FactoryOf<Self>;
    fn on_create(
        session: &AnySession,
        create_info: quark::types::SessionCreateInfo,
    ) -> XrResult<Self> {
        debug!("on_create(): OpenXR session");
        // Do we have vulkan?
        let AnySession::Vulkan(xr_vk_session) = session else {
            warn!("Not a vulkan session, don't know how to handle it");
            return Ok(Self::default());
        };
        let gb = create_info.graphics_binding.unwrap();
        let quark::types::GraphicsBinding::Vulkan(gb) = gb else {
            unreachable!()
        };
        let instance = xr_vk_session
            .instance()
            .as_handle()
            .registered_with_hook()?;
        if instance.hook().is_passthrough_enabled {
            let api_version = instance
                .hook()
                .instance_api_version
                .get(&(gb.instance as u64));
            let api_version = if let Some(v) = api_version {
                *v
            } else {
                let req = xr_vk_session
                    .instance()
                    .graphics_requirements::<openxr::Vulkan>(create_info.system_id)?;
                // This vulkan instance wasn't created via `xrCreateVulkanInstanceKHR`, we have to
                // assume minimum supported api version.
                (req.min_api_version_supported.into_raw() as u32).into()
            };
            log::info!("Vulkan API version: {api_version}");
            let enabled_vk_instance_extensions = REQUIRED_VK_INSTANCE_EXTENSIONS
                .iter()
                .map(|&e| e.to_str().unwrap())
                .collect();
            let vk_create_info = vulkano::instance::InstanceCreateInfo {
                enabled_extensions: &enabled_vk_instance_extensions,
                max_api_version: Some(api_version),
                ..Default::default()
            };
            let vk_instance = unsafe {
                vulkano::instance::Instance::from_handle_borrowed(
                    &VULKAN_LIBRARY,
                    ash::vk::Handle::from_raw(gb.instance as usize as u64),
                    &vk_create_info,
                )
            };
            let physical_device = match unsafe {
                vulkano::device::physical::PhysicalDevice::from_handle(
                    &vk_instance,
                    ash::vk::Handle::from_raw(gb.physical_device as usize as u64),
                )
            } {
                Ok(pd) => pd,
                Err(e) => {
                    warn!("Failed to wrap vulkan physical device {e}");
                    return Ok(Self::default());
                }
            };
            let queues = vec![0.0; gb.queue_index as usize + 1];
            let enabled_vk_device_extensions = REQUIRED_VK_DEVICE_EXTENSIONS
                .iter()
                .map(|&e| e.to_str().unwrap())
                .collect();
            let vk_create_info = vulkano::device::DeviceCreateInfo {
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index: gb.queue_family_index,
                    queues: &queues,
                    ..Default::default()
                }],
                enabled_extensions: &enabled_vk_device_extensions,
                ..Default::default()
            };
            let (device, mut queue) = unsafe {
                vulkano::device::Device::from_handle_borrowed(
                    &physical_device,
                    ash::vk::Handle::from_raw(gb.device as usize as u64),
                    &vk_create_info,
                )
            };
            let Some(queue) = queue.nth(gb.queue_index as _) else {
                warn!("Failed to get requested Vulkan queue");
                return Ok(Self::default());
            };

            let allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
            let cmdbuf_allocator = Arc::new(StandardCommandBufferAllocator::new(
                &device,
                &Default::default(),
            ));
            let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
                &device,
                &Default::default(),
            ));
            let space = xr_vk_session
                .create_reference_space(ReferenceSpaceType::STAGE, openxr::Posef::IDENTITY)?;

            Ok(Self {
                inner: Some(SessionDataInner {
                    device,
                    queue,
                    state: SessionState::Idle,
                    space,
                    system_id: create_info.system_id,
                    allocator,
                    cmdbuf_allocator,
                    descriptor_set_allocator,
                }),
            })
        } else {
            debug!("passthrough extension not enabled, not hooking session");
            Ok(Self::default())
        }
    }
}

static CAMERA_CONFIG: LazyLock<Option<crate::steam::StereoCamera>> =
    LazyLock::new(crate::steam::find_steam_config);
static SPLASH_IMAGE: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/splash.png"));

#[allow(clippy::too_many_arguments)]
fn create_camera_resources(
    instance: &openxr::Instance,
    session: &openxr::Session<openxr::Vulkan>,
    system_id: openxr::sys::SystemId,
    view_type: openxr::sys::ViewConfigurationType,
    device: &Arc<vulkano::device::Device>,
    queue: &Arc<vulkano::device::Queue>,
    allocator: &Arc<dyn MemoryAllocator>,
    cmdbuf_allocator: &Arc<dyn CommandBufferAllocator>,
    descriptor_set_allocator: &Arc<dyn DescriptorSetAllocator>,
) -> Result<CameraResources, XrErr> {
    let cfgs = instance.enumerate_view_configuration_views(system_id, view_type)?;
    if cfgs.len() != 1 && cfgs.len() != 2 {
        error!("unsupported view count? {}", cfgs.len());
    }
    let xdg = xdg::BaseDirectories::new().map_err(|e| {
        warn!("xdg: {e:#}");
        XrErr::ERROR_RUNTIME_FAILURE
    })?;
    let pipeline_cache = crate::config::load_pipeline_cache(device, &xdg).map_err(|e| {
        warn!("Failed to load pipeline cache {e:#}");
        XrErr::ERROR_RUNTIME_FAILURE
    })?;
    let width = cfgs[0]
        .recommended_image_rect_width
        .max(cfgs[1].recommended_image_rect_width);
    let height = cfgs[0]
        .recommended_image_rect_height
        .max(cfgs[1].recommended_image_rect_height);

    let pp = crate::pipeline::Pipeline::new(
        device,
        allocator,
        cmdbuf_allocator,
        queue,
        descriptor_set_allocator,
        true,
        *CAMERA_CONFIG,
        ImageLayout::ShaderReadOnlyOptimal,
        ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC,
        pipeline_cache,
        UVec2::new(crate::CAMERA_SIZE, crate::CAMERA_SIZE),
        UVec2::new(width, height),
    )
    .map_err(|e| {
        warn!("Failed to create pipeline {e:#}");
        XrErr::ERROR_RUNTIME_FAILURE
    })?;
    let camera = crate::find_index_camera().map_err(|e| {
        warn!("Cannot find camera {e:#}");
        XrErr::ERROR_RUNTIME_FAILURE
    })?;
    let camera = v4l::Device::with_path(camera).map_err(|e| {
        warn!("Failed to open camera {e:#}");
        XrErr::ERROR_RUNTIME_FAILURE
    })?;
    let formats = session
        .enumerate_swapchain_formats()?
        .into_iter()
        .map(|f| vulkano::format::Format::try_from(ash::vk::Format::from_raw(f as i32)))
        .collect::<Result<HashSet<_>, _>>()
        .map_err(|()| {
            warn!("Invalid swapchain formats");
            XrErr::ERROR_RUNTIME_FAILURE
        })?;
    let camera = crate::camera::CameraThread::new(camera, SPLASH_IMAGE);
    const PREFERRED_FORMATS: [vulkano::format::Format; 4] = [
        vulkano::format::Format::R8G8B8A8_UNORM,
        vulkano::format::Format::B8G8R8A8_UNORM,
        vulkano::format::Format::R8G8B8A8_SRGB,
        vulkano::format::Format::B8G8R8A8_SRGB,
    ];

    let Some(format) = PREFERRED_FORMATS
        .iter()
        .find(|f| formats.contains(f))
        .copied()
    else {
        warn!("No suitable format found for swapchain");
        return Err(XrErr::ERROR_RUNTIME_FAILURE);
    };
    let swapchain = session.create_swapchain(&SwapchainCreateInfo {
        array_size: 1,
        face_count: 1,
        format: format as u32,
        mip_count: 1,
        sample_count: cfgs[0].recommended_swapchain_sample_count,
        usage_flags: SwapchainUsageFlags::COLOR_ATTACHMENT | SwapchainUsageFlags::TRANSFER_DST,
        create_flags: SwapchainCreateFlags::EMPTY,
        width: width * 2,
        height,
    })?;
    let images = swapchain
        .enumerate_images()?
        .into_iter()
        .map(|raw_img| unsafe {
            Ok::<_, XrErr>(Arc::new(
                vulkano::image::sys::RawImage::from_handle_borrowed(
                    device,
                    ash::vk::Image::from_raw(raw_img),
                    &ImageCreateInfo {
                        format,
                        extent: [width * 2, height, 1],
                        array_layers: 1,
                        mip_levels: 1,
                        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                )
                .map_err(|e| {
                    warn!("Failed to wrap image {e:#}");
                    XrErr::ERROR_RUNTIME_FAILURE
                })?
                .assume_bound(),
            ))
        })
        .collect::<Result<_, _>>()?;
    camera.resume().unwrap();
    Ok(CameraResources {
        swapchain,
        camera,
        pp,
        images,
        size: UVec2::new(width, height),
    })
}

unsafe extern "system" fn begin_session(
    session: openxr::sys::Session,
    info: *const openxr::sys::SessionBeginInfo,
) -> XrErr {
    debug!("begin session {:#x}", session.into_raw());
    let mut wrapped_session = try_xr!(session.registered_with_hook_mut());
    let instance = try_xr!(quark::find_instance(session));
    let info = &unsafe { *info };
    let wrapped_instance = try_xr!(instance.registered_with_hook());
    let (data, wrapped_session) = wrapped_session.both();
    let Some(data) = &mut data.inner else {
        // We are not wrapping this session, passed it through.
        debug!("Unhandled session, like passthrough extension wasn't enabled");
        return unsafe { (wrapped_instance.get().fp().begin_session)(session, info) };
    };
    let quark::types::AnySession::Vulkan(xr_vk_session) = wrapped_session else {
        unreachable!()
    };
    // Transition first, if `session.begin` failed then the session will still not be running, and
    // we don't need to do anything.
    try_xr!(xr_vk_session.begin(info.primary_view_configuration_type));
    data.state = match &data.state {
        SessionState::Running { .. } | SessionState::RunningWithPassthrough { .. } => {
            return XrErr::ERROR_SESSION_RUNNING;
        }
        SessionState::Idle => SessionState::Running {
            view_type: info.primary_view_configuration_type,
        },
        SessionState::IdleWithPassthrough { passthrough } => {
            let camera = try_xr!(create_camera_resources(
                wrapped_instance.get(),
                xr_vk_session,
                data.system_id,
                info.primary_view_configuration_type,
                &data.device,
                &data.queue,
                &data.allocator,
                &data.cmdbuf_allocator,
                &data.descriptor_set_allocator
            ));
            SessionState::RunningWithPassthrough {
                camera,
                passthrough: passthrough.clone(),
                view_type: info.primary_view_configuration_type,
                should_render: None,
                image_index: None,
            }
        }
    };
    XrErr::SUCCESS
}

unsafe extern "system" fn end_session(raw_session: openxr::sys::Session) -> XrErr {
    debug!("end session {:#x}", raw_session.into_raw());
    let instance = try_xr!(quark::find_instance(raw_session));
    let mut session = try_xr!(raw_session.registered_with_hook_mut());
    let (data, session) = session.both();
    let Some(data) = &mut data.inner else {
        // We are not wrapping this session, passed it through.
        debug!("Unhandled session, like passthrough extension wasn't enabled");
        return unsafe { (try_xr!(instance.registered()).fp().end_session)(raw_session) };
    };
    let quark::types::AnySession::Vulkan(xr_vk_session) = session else {
        unreachable!()
    };
    data.state = match &data.state {
        SessionState::Idle | SessionState::IdleWithPassthrough { .. } => {
            return XrErr::ERROR_SESSION_NOT_RUNNING;
        }
        SessionState::Running { .. } => SessionState::Idle,
        SessionState::RunningWithPassthrough { passthrough, .. } => {
            SessionState::IdleWithPassthrough {
                passthrough: passthrough.clone(),
            }
        }
    };
    try_xr!(xr_vk_session.end());
    XrErr::SUCCESS
}

unsafe extern "system" fn wait_frame(
    raw_session: openxr::sys::Session,
    wait_info: *const openxr::sys::FrameWaitInfo,
    frame_state: *mut openxr::sys::FrameState,
) -> XrErr {
    debug!("wait frame {:#x}, before", raw_session.into_raw());
    let fp = {
        // Can't keep the `registered` high-level instance object, because it
        // locks the object registry. but `xrWaitFrame` must be callable from any thread, while
        // `Begin/EndFrame` might be called concurrently from other threads, which needs this
        // registry entry lock too, and `xrWaitFrame` might enter into wait.
        let instance = try_xr!(quark::find_instance(raw_session));
        try_xr!(instance.registered()).fp().wait_frame
    };

    // We didn't keep the frame waiter given by openxr crate, just call the raw function.
    let ret = unsafe { (fp)(raw_session, wait_info, frame_state) };
    if ret != XrErr::SUCCESS {
        return ret;
    }
    debug!("wait frame {:#x}, after", raw_session.into_raw());

    let mut session = try_xr!(raw_session.registered_with_hook_mut());
    let data = session.hook();
    let Some(data) = &mut data.inner else {
        // We are not wrapping this session, passed it through.
        debug!("Unhandled session, like passthrough extension wasn't enabled");
        return XrErr::SUCCESS;
    };
    let SessionState::RunningWithPassthrough { should_render, .. } = &mut data.state else {
        if matches!(data.state, SessionState::Running { .. }) {
            return XrErr::SUCCESS;
        } else {
            return XrErr::ERROR_SESSION_NOT_RUNNING;
        }
    };
    assert!(should_render.is_none());
    *should_render = Some(bool::from(unsafe { *frame_state }.should_render));

    XrErr::SUCCESS
}

unsafe extern "system" fn begin_frame(
    raw_session: openxr::sys::Session,
    info: *mut openxr::sys::FrameBeginInfo,
) -> XrErr {
    debug!("begin frame {:#x} before", raw_session.into_raw());
    let instance = try_xr!(quark::find_instance(raw_session));
    // Lock the registry entry for the xrSession. this is because xrBeginFrame might unblock
    // a currently blocked xrWaitFrame. Our override for xrWaitFrame will acquire the xrSession
    // from the registry to update `should_render`. We want to make sure it will only get the lock
    // after we have set `should_render` to None.
    let mut session = try_xr!(raw_session.registered_with_hook_mut());
    let ret = unsafe { (try_xr!(instance.registered()).fp().begin_frame)(raw_session, info) };
    if ret != XrErr::SUCCESS {
        return ret;
    }
    debug!("begin frame {:#x} after", raw_session.into_raw());
    let data = session.hook();
    let Some(data) = &mut data.inner else {
        // We are not wrapping this session, passed it through.
        debug!("Unhandled session, like passthrough extension wasn't enabled");
        return XrErr::SUCCESS;
    };
    let SessionState::RunningWithPassthrough {
        image_index,
        should_render,
        camera,
        ..
    } = &mut data.state
    else {
        if matches!(data.state, SessionState::Running { .. }) {
            return XrErr::SUCCESS;
        } else {
            return XrErr::ERROR_SESSION_NOT_RUNNING;
        }
    };
    let Some(should_render) = should_render.take() else {
        return XrErr::ERROR_CALL_ORDER_INVALID;
    };
    if !should_render {
        return XrErr::SUCCESS;
    }
    if image_index.is_some() {
        // App might called 2 begin frame in a row, we can use the image we alredy have.
        return XrErr::SUCCESS;
    }
    *image_index = Some(try_xr!(camera.swapchain.acquire_image()));
    try_xr!(camera.swapchain.wait_image(openxr::Duration::INFINITE));
    XrErr::SUCCESS
}

unsafe extern "system" fn end_frame(
    raw_session: openxr::sys::Session,
    info: *const openxr::sys::FrameEndInfo,
) -> XrErr {
    debug!("end frame {:#x}", raw_session.into_raw());
    let instance = try_xr!(quark::find_instance(raw_session));
    let instance = try_xr!(instance.registered());
    let mut session = try_xr!(raw_session.registered_with_hook_mut());
    let info = unsafe { *info };
    let (data, session) = session.both();
    try_xr!(unsafe { data.end_frame(session, &instance, &info) });
    XrErr::SUCCESS
}

#[allow(dead_code)]
unsafe extern "system" fn enumerate_environment_blend_modes(
    raw_instance: openxr::sys::Instance,
    system_id: openxr::sys::SystemId,
    view_config_type: openxr::sys::ViewConfigurationType,
    capacity: u32,
    count: *mut u32,
    modes: *mut openxr::sys::EnvironmentBlendMode,
) -> XrErr {
    let instance = try_xr!(raw_instance.registered());
    let original_modes = try_xr!(unsafe {
        call_enumerate!(instance.fp().enumerate_environment_blend_modes => [3];
        raw_instance, system_id, view_config_type)
    });
    if original_modes.contains(&openxr::sys::EnvironmentBlendMode::ALPHA_BLEND) {
        // Already has ALPHA_BLEND, just forward the call
        unsafe {
            (instance.fp().enumerate_environment_blend_modes)(
                raw_instance,
                system_id,
                view_config_type,
                capacity,
                count,
                modes,
            )
        }
    } else {
        debug!("Insert ALPHA_BLEND into supported environment blend modes");
        if (capacity as usize) < original_modes.len() + 1 {
            unsafe { *count = (original_modes.len() + 1) as _ };
            if capacity == 0 {
                XrErr::SUCCESS
            } else {
                XrErr::ERROR_SIZE_INSUFFICIENT
            }
        } else {
            unsafe {
                modes.copy_from_nonoverlapping(original_modes.as_ptr(), original_modes.len());
                modes
                    .add(original_modes.len())
                    .write(openxr::sys::EnvironmentBlendMode::ALPHA_BLEND);
                *count = (original_modes.len() + 1) as _;
            }
            XrErr::SUCCESS
        }
    }
}

quark::api_layer! {
    hooks: {
        Session: SessionData,
        Instance: InstanceData,
        PassthroughHTC: PassthroughData,
    },
    override_fns: {
        xrGetVulkanInstanceExtensionsKHR: get_vulkan_instance_extensions,
        xrGetVulkanDeviceExtensionsKHR: get_vulkan_device_extensions,
        xrCreateVulkanInstanceKHR: create_vulkan_instance,
        xrCreateVulkanDeviceKHR: create_vulkan_device,
        xrBeginSession: begin_session,
        xrEndSession: end_session,
        xrWaitFrame: wait_frame,
        xrBeginFrame: begin_frame,
        xrEndFrame: end_frame,
        //xrEnumerateEnvironmentBlendModes: enumerate_environment_blend_modes,
    },
}
