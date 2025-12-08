use crate::{find_instance, Low, XrErr};
use log::warn;
use openxr::sys::Handle;
use std::{
    ffi::{c_void, CStr},
    os::raw::c_char,
};
pub struct ApplicationInfo {
    pub application_name: std::ffi::CString,
    pub application_version: u32,
    pub engine_name: std::ffi::CString,
    pub engine_version: u32,
    pub api_version: openxr::Version,
}

fn c_char_to_u8<const N: usize>(input: &[c_char; N]) -> &[u8; N] {
    // SAFETY: i8 -> u8 is legal.
    unsafe { &*(input as *const _ as *const [u8; N]) }
}

impl ApplicationInfo {
    pub fn from_raw(raw: &openxr::sys::ApplicationInfo) -> Result<Self, XrErr> {
        Ok(Self {
            api_version: raw.api_version,
            engine_version: raw.engine_version,
            application_version: raw.application_version,
            // SAFETY: application name is `MAX_APPLICATION_NAME_SIZE` long, and cast from i8 -> u8
            // is legal
            application_name: std::ffi::CStr::from_bytes_until_nul(c_char_to_u8(
                &raw.application_name,
            ))
            .map_err(|_| XrErr::ERROR_VALIDATION_FAILURE)?
            .to_owned(),
            // SAFETY: ditto
            engine_name: std::ffi::CStr::from_bytes_until_nul(c_char_to_u8(&raw.engine_name))
                .map_err(|_| XrErr::ERROR_VALIDATION_FAILURE)?
                .to_owned(),
        })
    }
}

pub struct InstanceCreateInfo {
    pub application_info: ApplicationInfo,
    pub required_extensions: openxr::ExtensionSet,
    pub layers: Vec<std::ffi::CString>,
}

impl InstanceCreateInfo {
    /// # Safety
    ///
    /// `raw` must be valid
    pub unsafe fn from_raw(raw: *const openxr::sys::InstanceCreateInfo) -> Result<Self, XrErr> {
        let app_info = unsafe { ApplicationInfo::from_raw(&(*raw).application_info) }?;
        let extensions = if (*raw).enabled_extension_count > 0 {
            unsafe {
                std::slice::from_raw_parts(
                    (*raw).enabled_extension_names,
                    (*raw).enabled_extension_count as _,
                )
                .iter()
                .map(|&e| std::ffi::CStr::from_ptr(e).to_bytes())
            }
            .collect()
        } else {
            openxr::ExtensionSet::default()
        };
        let layers = if (*raw).enabled_api_layer_count > 0 {
            unsafe {
                std::slice::from_raw_parts(
                    (*raw).enabled_api_layer_names,
                    (*raw).enabled_api_layer_count as _,
                )
                .iter()
                .map(|&e| std::ffi::CStr::from_ptr(e).to_owned())
                .collect()
            }
        } else {
            Vec::new()
        };
        Ok(Self {
            application_info: app_info,
            layers,
            required_extensions: extensions,
        })
    }
}

#[derive(Copy, Clone)]
pub enum GraphicsBinding {
    Vulkan(openxr::vulkan::SessionCreateInfo),
    GlXlib {
        glx_fb_config: *mut c_void,
        glx_drawable: u64,
        glx_context: *mut c_void,
        x_display: *mut c_void,
        visualid: u32,
    },
    GlXcb {
        connection: *mut c_void,
        screen_number: u32,
        fbconfigid: u32,
        visualid: u32,
        glx_drawable: u32,
        glx_context: u32,
    },
    GlWayland {
        display: *mut c_void,
    },
}

pub struct SessionCreateInfo {
    pub system_id: openxr::SystemId,
    // If None, this session will be wrapped as `Session::Unhandled`.
    pub graphics_binding: Option<GraphicsBinding>,
}

impl GraphicsBinding {
    pub fn from_raw_vulkan(raw: &openxr::sys::GraphicsBindingVulkanKHR) -> Self {
        Self::Vulkan(openxr::vulkan::SessionCreateInfo {
            device: raw.device,
            instance: raw.instance,
            physical_device: raw.physical_device,
            queue_family_index: raw.queue_family_index,
            queue_index: raw.queue_index,
        })
    }
    pub fn from_raw_gl_xcb(raw: &openxr::sys::GraphicsBindingOpenGLXcbKHR) -> Self {
        Self::GlXcb {
            connection: raw.connection,
            screen_number: raw.screen_number,
            fbconfigid: raw.fbconfigid,
            visualid: raw.visualid,
            glx_drawable: raw.glx_drawable,
            glx_context: raw.glx_context,
        }
    }
    pub fn from_raw_gl_xlib(raw: &openxr::sys::GraphicsBindingOpenGLXlibKHR) -> Self {
        Self::GlXlib {
            glx_fb_config: raw.glx_fb_config,
            x_display: raw.x_display,
            visualid: raw.visualid,
            glx_drawable: raw.glx_drawable,
            glx_context: raw.glx_context,
        }
    }
    pub fn from_raw_gl_wayland(raw: &openxr::sys::GraphicsBindingOpenGLWaylandKHR) -> Self {
        Self::GlWayland {
            display: raw.display,
        }
    }
}
impl SessionCreateInfo {
    /// # Safety
    ///
    /// `raw` must be valid
    pub unsafe fn from_raw(raw: *const openxr::sys::SessionCreateInfo) -> Self {
        let gb = {
            // Do we have vulkan?
            let mut curr = raw as *const openxr::sys::BaseInStructure;
            loop {
                match (*curr).ty {
                    openxr::sys::GraphicsBindingVulkanKHR::TYPE => {
                        break Some(GraphicsBinding::from_raw_vulkan(&*(curr as *const _)))
                    }
                    openxr::sys::GraphicsBindingOpenGLXlibKHR::TYPE => {
                        break Some(GraphicsBinding::from_raw_gl_xlib(&*(curr as *const _)))
                    }
                    openxr::sys::GraphicsBindingOpenGLXcbKHR::TYPE => {
                        break Some(GraphicsBinding::from_raw_gl_xcb(&*(curr as *const _)))
                    }
                    openxr::sys::GraphicsBindingOpenGLWaylandKHR::TYPE => {
                        break Some(GraphicsBinding::from_raw_gl_wayland(&*(curr as *const _)))
                    }
                    _ => (),
                }

                if (*curr).next.is_null() {
                    break None;
                }
                curr = (*curr).next;
            }
        };
        Self {
            system_id: (*raw).system_id,
            graphics_binding: gb,
        }
    }
}

pub enum AnySession {
    Vulkan(openxr::Session<openxr::Vulkan>),
    Gl(openxr::Session<openxr::OpenGL>),
    Unhandled(openxr::sys::Session),
}

impl AnySession {
    pub fn end(&self) -> openxr::Result<openxr::sys::Result> {
        match self {
            Self::Vulkan(s) => s.end(),
            Self::Gl(s) => s.end(),
            Self::Unhandled(_) => unimplemented!("unsupported session type"),
        }
    }
}

impl openxr::AsHandle for AnySession {
    type Handle = openxr::sys::Session;
    fn as_handle(&self) -> Self::Handle {
        match self {
            Self::Vulkan(v) => v.as_handle(),
            Self::Gl(g) => g.as_handle(),
            &Self::Unhandled(u) => u,
        }
    }
}

impl Drop for AnySession {
    fn drop(&mut self) {
        let Self::Unhandled(raw) = self else { return };
        let Ok(instance) = find_instance(*raw) else {
            warn!("Session handle not found {}", raw.into_raw());
            return;
        };
        let Ok(instance) = instance.registered() else {
            warn!("No OpenXR instance found?");
            return;
        };
        unsafe { (instance.fp().destroy_session)(*raw) };
    }
}

pub enum AnySwapchain {
    Vulkan(openxr::Swapchain<openxr::Vulkan>),
    Gl(openxr::Swapchain<openxr::OpenGL>),
    Unhandled(openxr::sys::Swapchain),
}

impl openxr::AsHandle for AnySwapchain {
    type Handle = openxr::sys::Swapchain;
    fn as_handle(&self) -> Self::Handle {
        match self {
            Self::Vulkan(v) => v.as_handle(),
            Self::Gl(g) => g.as_handle(),
            &Self::Unhandled(u) => u,
        }
    }
}

impl Drop for AnySwapchain {
    fn drop(&mut self) {
        let Self::Unhandled(raw) = self else { return };
        let Ok(instance) = find_instance(*raw) else {
            warn!("Swapchain object not found {}", raw.into_raw());
            return;
        };
        let Ok(instance) = instance.registered() else {
            warn!("No instance found?");
            return;
        };
        unsafe { (instance.fp().destroy_swapchain)(*raw) };
    }
}

pub struct ActionSetCreateInfo {
    pub action_set_name: String,
    pub localized_action_set_name: String,
    pub priority: u32,
}

impl ActionSetCreateInfo {
    /// # Safety
    ///
    /// `raw` must point to valid memory
    pub unsafe fn from_raw(raw: *const openxr::sys::ActionSetCreateInfo) -> Result<Self, XrErr> {
        Ok(Self {
            action_set_name: CStr::from_bytes_until_nul(c_char_to_u8(&(*raw).action_set_name))
                .map_err(|_| XrErr::ERROR_VALIDATION_FAILURE)
                .and_then(|s| s.to_str().map_err(|_| XrErr::ERROR_VALIDATION_FAILURE))?
                .to_owned(),
            localized_action_set_name: CStr::from_bytes_until_nul(c_char_to_u8(
                &(*raw).localized_action_set_name,
            ))
            .map_err(|_| XrErr::ERROR_VALIDATION_FAILURE)
            .and_then(|s| s.to_str().map_err(|_| XrErr::ERROR_VALIDATION_FAILURE))?
            .to_owned(),
            priority: (*raw).priority,
        })
    }
}
