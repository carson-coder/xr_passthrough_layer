use crate::api_layer::{LOG_INIT, REQUIRED_VK_DEVICE_EXTENSIONS, REQUIRED_VK_INSTANCE_EXTENSIONS};
use log::{debug, warn};
use openxr::sys::Result as XrErr;
use quark::{Hooked as _, Low as _, try_xr};
use std::{
    collections::{HashMap, HashSet},
    ffi::{CStr, c_char},
};
use vulkano::Handle as _;

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

pub(super) unsafe extern "system" fn get_vulkan_instance_extensions(
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

pub(super) unsafe extern "system" fn get_vulkan_device_extensions(
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

pub(super) unsafe extern "system" fn create_vulkan_instance(
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

pub(super) unsafe extern "system" fn create_vulkan_device(
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

// Define your instance data
pub struct InstanceData {
    /// Whether the XR_HTC_passthrough extension is enabled. Note passthrough resources are created
    /// regardless to support the ALPHA_BLEND environment blend mode, which is not behind an
    /// extension.
    _is_passthrough_enabled: bool,
    /// Mapping raw vulkan VkInstance handles to the api version it was created with.
    instance_api_version: HashMap<u64, vulkano::Version>,
}

impl InstanceData {
    pub(super) fn instance_api_version(&self) -> &HashMap<u64, vulkano::Version> {
        &self.instance_api_version
    }
}

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
            _is_passthrough_enabled: enabled,
            instance_api_version: HashMap::new(),
        };
        Ok((high, this))
    }
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

pub(super) unsafe extern "system" fn enumerate_environment_blend_modes(
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

impl quark::Hook for InstanceData {
    type Target = openxr::sys::Instance;
    type Factory = InstanceFactory;
}
