use glam::Vec3;
use log::debug;
use openxr::{
    AsHandle,
    sys::{Handle, Result as XrErr},
};
use quark::{Hooked as _, Low as _};
use std::{
    ffi::CStr,
    mem::MaybeUninit,
    sync::{Arc, LazyLock, OnceLock},
};

mod instance;
mod session;
use instance::{
    InstanceData, create_vulkan_device, create_vulkan_instance, enumerate_environment_blend_modes,
    get_vulkan_device_extensions, get_vulkan_instance_extensions,
};
use session::{SessionData, begin_frame, begin_session, end_frame, end_session, wait_frame};

const REQUIRED_VK_INSTANCE_EXTENSIONS: &[&CStr] = &[
    ash::vk::KHR_EXTERNAL_MEMORY_CAPABILITIES_NAME,
    ash::vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME,
    ash::vk::KHR_XCB_SURFACE_NAME,
];

const REQUIRED_VK_DEVICE_EXTENSIONS: &[&CStr] = &[ash::vk::KHR_COPY_COMMANDS2_NAME];

fn xrcvt(e: XrErr) -> Result<(), XrErr> {
    if e == XrErr::SUCCESS { Ok(()) } else { Err(e) }
}

struct CameraResources {
    camera: crate::camera::CameraThread,
    pp: crate::pipeline::Pipeline,
}

struct PassthroughMesh<'a> {
    vertices: &'a [openxr::Vector3f],
    indices: &'a [u32],
    base_space: openxr::sys::Space,
    time: openxr::sys::Time,
    pose: openxr::Posef,
    scale: Vec3,
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

#[derive(Default)]
pub struct PassthroughInner {}

impl PassthroughInner {
    fn camera_cfg() -> Option<&'static crate::steam::StereoCamera> {
        CAMERA_CONFIG.as_ref()
    }
    fn splash() -> &'static [u8] {
        SPLASH_IMAGE
    }
}

impl Drop for PassthroughInner {
    fn drop(&mut self) {
        debug!("Passthrough destroyed")
    }
}

pub struct PassthroughData {
    inner: Arc<PassthroughInner>,
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
        debug!("Creating passthrough {:#x}", session.as_handle().into_raw());
        let form = unsafe { *args.1 }.form;
        let passthrough = session_data.maybe_get_or_add_passthrough()?;
        let passthrough = passthrough.ok_or(XrErr::ERROR_FEATURE_UNSUPPORTED)?;

        let ret = PassthroughData {
            inner: passthrough,
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

static VULKAN_LIBRARY: LazyLock<Arc<vulkano::library::VulkanLibrary>> =
    LazyLock::new(|| unsafe { vulkano::library::VulkanLibrary::new().unwrap() });

static LOG_INIT: OnceLock<()> = OnceLock::new();

static CAMERA_CONFIG: LazyLock<Option<crate::steam::StereoCamera>> =
    LazyLock::new(crate::steam::find_steam_config);
static SPLASH_IMAGE: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/splash.png"));

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
        xrEnumerateEnvironmentBlendModes: enumerate_environment_blend_modes,
    },
}
