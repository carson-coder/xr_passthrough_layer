pub mod util;
pub mod prelude {
    pub use openxr::sys::Result as XrErr;
    pub use proc_macros::*;
    pub type XrResult<T> = Result<T, XrErr>;
    pub use crate::util::*;
}
pub mod types;
pub use log::info as debug;
use log::{error, trace};

use std::{any::Any, collections::HashSet, marker::PhantomData, ops::Deref, sync::LazyLock};
use types::*;

use dashmap::DashMap;
pub use openxr;
use openxr::sys::Handle;
use prelude::*;
pub use proc_macros::*;

type BoxedDataPair = (Box<dyn Any + Send + Sync>, Box<dyn Any + Send + Sync>);
static DATA_REGISTRY: LazyLock<DashMap<u64, BoxedDataPair>> = LazyLock::new(DashMap::default);

/// Implemented on hooked openxr handles, parameterized by the associated data for the hook.
///
/// # Safety
///
/// This trait is NEVER SAFE to implement manually. This should only be implemented by the
/// `api_layer` macro.
pub unsafe trait Hooked<By>: Low
where
    By: Hook<Target = Self>,
{
    fn registered_with_hook(self) -> Result<WithHook<By>, RegisterationError> {
        let r = DATA_REGISTRY
            .get(&self.into_raw())
            .ok_or(RegisterationError::NotFound(self.into_raw()))?;
        if r.0.is::<By>() {
            Ok(WithHook(r, PhantomData))
        } else {
            Err(RegisterationError::TypeMismatch(
                self.into_raw(),
                std::any::type_name::<Self>(),
            ))
        }
    }
    fn registered_with_hook_mut(self) -> Result<WithHookMut<By>, RegisterationError> {
        let r = DATA_REGISTRY
            .get_mut(&self.into_raw())
            .ok_or(RegisterationError::NotFound(self.into_raw()))?;
        if r.0.is::<By>() {
            Ok(WithHookMut(r, PhantomData))
        } else {
            Err(RegisterationError::TypeMismatch(
                self.into_raw(),
                std::any::type_name::<Self>(),
            ))
        }
    }
    // fn handle(&self) -> Self::Handle;

    // /// Retrieve the associated hook object
    // fn hook(&self) -> impl Deref<Target = Self::Hook> + 'static {
    //     // SAFETY: Because of the guarantees we uphold, there is no need to type check.
    //     unsafe {
    //         DATA_REGISTRY
    //             .get(&self.handle().into_raw())
    //             .unwrap_unchecked()
    //             .map(|(h, _)| &*(&**h as &dyn Any as *const _ as *const Self::Hook))
    //     }
    // }

    // /// Retrieve the associated hook object
    // fn hook_mut(&self) -> impl DerefMut<Target = Self::Hook> + 'static {
    //     // SAFETY: Because of the guarantees we uphold, there is no need to type check.
    //     unsafe {
    //         DATA_REGISTRY
    //             .get_mut(&self.handle().into_raw())
    //             .unwrap_unchecked()
    //             .map(|(h, _)| &mut *(&mut **h as &mut dyn Any as *mut _ as *mut Self::Hook))
    //     }
    // }
}

#[derive(thiserror::Error, Debug)]
pub enum RegisterationError {
    #[error("handle {0:#x} not found in the registery")]
    NotFound(u64),
    #[error("handle {0:#x} is not a {1}")]
    TypeMismatch(u64, &'static str),
}

impl From<RegisterationError> for XrErr {
    fn from(_value: RegisterationError) -> Self {
        XrErr::ERROR_HANDLE_INVALID
    }
}

pub struct WithHook<H: Hook>(
    dashmap::mapref::one::Ref<'static, u64, BoxedDataPair>,
    PhantomData<H>,
);

/// A pair of hook data and the high-level object it's hooking.
impl<H: Hook> WithHook<H> {
    pub fn get(&self) -> &<H::Target as Low>::High {
        self.both().1
    }
    pub fn hook(&self) -> &H {
        self.both().0
    }
    pub fn both(&self) -> (&H, &<H::Target as Low>::High) {
        // SAFETY: type check is performed in `Low::registered_with_hook`, and `dashmap`
        // guarantees the ref to this map entry will be valid as long as `self` is valid.
        let (data, high) = &*self.0;
        unsafe {
            (
                &*(&**data as &dyn Any as *const dyn Any as *const _),
                &*(&**high as &dyn Any as *const dyn Any as *const _),
            )
        }
    }
}

pub struct WithHookMut<H: Hook>(
    dashmap::mapref::one::RefMut<'static, u64, BoxedDataPair>,
    PhantomData<H>,
);

/// A pair of hook data and the high-level object it's hooking.
impl<H: Hook> WithHookMut<H> {
    pub fn get(&mut self) -> &mut <H::Target as Low>::High {
        self.both().1
    }
    pub fn hook(&mut self) -> &mut H {
        self.both().0
    }
    pub fn both(&mut self) -> (&mut H, &mut <H::Target as Low>::High) {
        // SAFETY: type check is performed in `Low::registered_with_hook`, and `dashmap`
        // guarantees the ref to this map entry will be valid as long as `self` is valid.
        let (data, high) = &mut *self.0;
        unsafe {
            (
                &mut *(&mut **data as &mut dyn Any as *mut dyn Any as *mut _),
                &mut *(&mut **high as &mut dyn Any as *mut dyn Any as *mut _),
            )
        }
    }
}

/// A low-level openxr handle, this is a sealed trait and can't be implemented by the user.
///
/// # Safety
///
/// Must create the correct object.
pub unsafe trait Low: sealed::Create {
    /// The corresponding high-level object encompassing the low-level handle.
    type High: openxr::AsHandle<Handle = Self> + Send + Sync + 'static;
    type HighCreateInfo;

    /// # Safety
    ///
    /// `args` must all point to valid memory, and must be what was used to create this handle.
    /// This handle should not be destroyed after calling this function.
    unsafe fn into_high(
        self,
        args: <Self as sealed::Create>::Args,
    ) -> Result<(Self::High, Self::HighCreateInfo), openxr::sys::Result>;

    /// Get the high-level object for this handle.
    fn registered(self) -> Result<impl Deref<Target = Self::High>, RegisterationError> {
        DATA_REGISTRY
            .get(&self.into_raw())
            .ok_or(RegisterationError::NotFound(self.into_raw()))?
            .try_map(|(_, high)| (&**high as &dyn Any).downcast_ref())
            .map_err(|_| {
                RegisterationError::TypeMismatch(self.into_raw(), std::any::type_name::<Self>())
            })
    }
}

/// An adapter implementing openxr::AsHandle for handles that don't have a high-level wrapper.
/// This does not manage the lifetime of the wrapped handle, it's user's responsibility to destroy
/// it when appropriate.
pub struct Facade<Low: sealed::Create>(Low);

impl<Low: sealed::Create> openxr::AsHandle for Facade<Low> {
    type Handle = Low;
    fn as_handle(&self) -> Self::Handle {
        self.0
    }
}

pub(crate) mod sealed {
    /// A marker trait for openxr handle that can be created. This trait is auto generated.
    pub trait Create: openxr::sys::Handle {
        type Args: Copy + Clone;
    }

    /// A marker trait for referencing to the factory for a given handle type.
    pub trait HasFactory: super::Low {
        type Factory;
    }
}

pub type CreateArgs<H> = <H as sealed::Create>::Args;
pub type FactoryOf<H> = <<H as Hook>::Target as sealed::HasFactory>::Factory;

/// A trait for creating any given openxr handle and its associated hook.
///
/// # Safety
///
/// `create` must create the correct openxr handle.
pub unsafe trait Factory<H: Hook> {
    /// Override this to customize how objects are created.
    ///
    /// # Returns
    ///
    /// A `Result` of a tuple, consisting of the high-level wrap object for the openxr handle, and
    /// the hook associated data object.
    ///
    /// # Safety
    ///
    /// `args` must be valid.
    unsafe fn create(
        args: <H::Target as sealed::Create>::Args,
    ) -> Result<(<H::Target as Low>::High, H), XrErr>;
}

/// A hook that's attached to an openxr handle. This is the trait you need to implement if you want
/// to add hooks to openxr handles. Generally if you want to override functions that operates on
/// some handle, you should define a hook for that handle. Due to some technicalities, instead of
/// implementing `Hook<Target = OpenXrHandleType>` directly, you need to implement
/// `Hook<Target = Wrap<OpenXrHandleType>>` instead, `Wrap` here is a generic type generated by the
/// `api_layer` macro. This is to ensure type safety, that only one type of hook can be attached to
/// any openxr handle.
///
/// # Safety
///
/// `create` must correctly create an object of the desired type. Don't override the default impl
/// and you will be fine. Just say "`create` is not overridden" in you safety comment.
pub trait Hook: Any + Send + Sync + Sized {
    type Target: Hooked<Self>;
    /// A factory type that's capable of creating the hooked handle. Use
    /// `quark::FactoryOf<Self>` for the default factory, which will call `Hook::on_create` after
    /// having created the handle. Or you can implement a factory yourself, in which case
    /// `on_create` can be ignored.
    type Factory: Factory<Self>;
    /// Called by the default factory after the openxr handle has been created, should return custom data for the probe.
    /// If you override the factory, you don't have to call, or implement this function at all.
    fn on_create(
        _handle: &<Self::Target as Low>::High,
        _info: <Self::Target as Low>::HighCreateInfo,
    ) -> XrResult<Self> {
        unimplemented!()
    }
}
pub(crate) fn remove_data(obj: u64) {
    DATA_REGISTRY.remove(&obj);
}

/// # Safety
///
/// This trait should NEVER be implemented manually. Use `api_layer!` instead.
pub unsafe trait ApiLayerOverride {
    fn overrides() -> &'static phf::Map<&'static str, openxr::sys::pfn::VoidFunction>;
}

/// Map from an openxr object to the instance it is created from
static OBJECT_OWNER: LazyLock<DashMap<u64, u64>> = LazyLock::new(DashMap::default);
static OBJECT_OWNED: LazyLock<DashMap<u64, HashSet<u64>>> = LazyLock::new(DashMap::default);

#[macro_export]
macro_rules! try_xr {
    ($e:expr) => {
        match $e {
            Ok(ok) => ok,
            Err(e) => {
                $crate::debug!("{} failed with {e}", stringify!($e));
                return e.into();
            }
        }
    };
}

pub mod wrappers {
    use crate::{
        sealed::Create,
        types::{InstanceCreateInfo, SessionCreateInfo},
        Low, XrErr,
    };
    use openxr::AsHandle;

    /// # Safety
    /// you are gay
    unsafe impl Low for openxr::sys::Instance {
        type High = openxr::Instance;
        type HighCreateInfo = crate::InstanceCreateInfo;
        unsafe fn into_high(
            self,
            args: <Self as Create>::Args,
        ) -> Result<(Self::High, Self::HighCreateInfo), openxr::sys::Result> {
            let (info, api_layer_info, instance) = args;
            if info.is_null() || api_layer_info.is_null() || instance.is_null() {
                return Err(XrErr::ERROR_VALIDATION_FAILURE);
            }
            let layer_info = &*api_layer_info;
            let entry = unsafe {
                openxr::Entry::from_get_instance_proc_addr(
                    (*layer_info.next_info).next_get_instance_proc_addr,
                )
            }?;
            let supported_extensions = entry.enumerate_extensions()?;
            let extensions: openxr::ExtensionSet = if (*info).enabled_extension_count > 0 {
                log::info!("extension count: {}", (*info).enabled_extension_count);
                unsafe {
                    let extensions = std::slice::from_raw_parts(
                        (*info).enabled_extension_names,
                        (*info).enabled_extension_count as _,
                    );
                    extensions
                        .iter()
                        .copied()
                        .map(|e| std::ffi::CStr::from_ptr(e).to_bytes_with_nul())
                }
                .collect()
            } else {
                Default::default()
            };
            // Since instance creation succeeded, if there's any extension in the required list that's
            // not supported, they are probably handled by the API layer, so we filter them out.
            let extensions = extensions.intersection(&supported_extensions);
            let extensions = openxr::InstanceExtensions::load(&entry, *instance, &extensions)?;
            Ok((
                openxr::Instance::from_raw(entry, *instance, extensions)?,
                InstanceCreateInfo::from_raw(info)?,
            ))
        }
    }

    unsafe impl Low for openxr::sys::Session {
        type High = crate::AnySession;
        type HighCreateInfo = crate::SessionCreateInfo;
        unsafe fn into_high(
            self,
            args: <Self as Create>::Args,
        ) -> Result<(Self::High, Self::HighCreateInfo), openxr::sys::Result> {
            let (instance, info, _) = args;
            let create_info = SessionCreateInfo::from_raw(info);
            let wrapped_instance = instance.registered()?;
            let session = match &create_info.graphics_binding {
                Some(crate::GraphicsBinding::Vulkan(_)) => crate::AnySession::Vulkan(
                    openxr::Session::<openxr::Vulkan>::from_raw(
                        wrapped_instance.clone(),
                        self,
                        Box::new(()),
                    )
                    .0,
                ),
                Some(crate::GraphicsBinding::GlXcb { .. })
                | Some(crate::GraphicsBinding::GlXlib { .. })
                | Some(crate::GraphicsBinding::GlWayland { .. }) => crate::AnySession::Gl(
                    openxr::Session::<openxr::OpenGL>::from_raw(
                        wrapped_instance.clone(),
                        self,
                        Box::new(()),
                    )
                    .0,
                ),
                None => crate::AnySession::Unhandled(self),
            };
            Ok((session, create_info))
        }
    }

    unsafe impl Low for openxr::sys::Swapchain {
        type High = crate::AnySwapchain;
        type HighCreateInfo = openxr::sys::SwapchainCreateInfo;
        unsafe fn into_high(
            self,
            args: <Self as crate::sealed::Create>::Args,
        ) -> Result<(Self::High, Self::HighCreateInfo), openxr::sys::Result> {
            let session = (args.0).registered()?;
            let sc = match &*session {
                crate::AnySession::Vulkan(v) => {
                    crate::AnySwapchain::Vulkan(openxr::Swapchain::from_raw(v.clone(), self))
                }
                crate::AnySession::Gl(v) => {
                    crate::AnySwapchain::Gl(openxr::Swapchain::from_raw(v.clone(), self))
                }
                crate::AnySession::Unhandled(_) => crate::AnySwapchain::Unhandled(self),
            };
            Ok((sc, *args.1))
        }
    }

    unsafe impl Low for openxr::sys::ActionSet {
        type High = openxr::ActionSet;
        type HighCreateInfo = crate::ActionSetCreateInfo;
        unsafe fn into_high(
            self,
            args: <Self as crate::sealed::Create>::Args,
        ) -> Result<(Self::High, Self::HighCreateInfo), openxr::sys::Result> {
            let instance = (args.0).registered()?;
            Ok((
                openxr::ActionSet::from_raw(instance.clone(), self),
                crate::ActionSetCreateInfo::from_raw(args.1)?,
            ))
        }
    }

    fn invoke_on_create<Hook: super::Hook>(
        handle: Hook::Target,
        args: <Hook::Target as super::sealed::Create>::Args,
    ) -> Result<(<Hook::Target as super::Low>::High, Hook), XrErr> {
        let (high, create_info) = unsafe { handle.into_high(args) }?;
        let hook = Hook::on_create(&high, create_info)?;
        Ok((high, hook))
    }

    macro_rules! gen_wrappers {
        (fn create) => {
            proc_macros::gen_create_wrapper! { $crate }
        };
        (impl Create) => {
            proc_macros::impl_create! { $crate }
        };
        (Facade; manual_impls: [ $($manual:ident),* ]) => {
            proc_macros::gen_facades! { $crate $($manual)* }
        };
    }

    gen_wrappers! {fn create}
    gen_wrappers! {
        Facade;
        manual_impls: [Session, Instance, ActionSet, Swapchain]
    }

    mod private {
        use super::invoke_on_create;
        use crate::Low;
        gen_wrappers! {impl Create}
        pub struct InstanceFactory;
        impl crate::sealed::HasFactory for openxr::sys::Instance {
            type Factory = InstanceFactory;
        }
        unsafe impl<Hook: crate::Hook<Target = openxr::sys::Instance>> crate::Factory<Hook>
            for InstanceFactory
        {
            unsafe fn create(
                args: <Hook::Target as crate::sealed::Create>::Args,
            ) -> Result<(openxr::Instance, Hook), crate::XrErr> {
                let layer_info = *args.1;
                let r = ((*layer_info.next_info).next_create_api_layer_instance)(
                    args.0, args.1, args.2,
                );
                if r != crate::XrErr::SUCCESS {
                    return Err(r);
                }
                invoke_on_create(*args.2, args)
            }
        }
    }

    impl Create for openxr::sys::Instance {
        type Args = (
            *const openxr::sys::InstanceCreateInfo,
            *const openxr::sys::loader::ApiLayerCreateInfo,
            *mut openxr::sys::Instance,
        );
    }

    /// # Safety
    /// ¯\_(ツ)_/¯
    pub unsafe extern "system" fn destroy_object<O1: openxr::sys::Handle>(obj: O1) -> XrErr {
        crate::unregister_object(obj.into_raw());
        XrErr::SUCCESS
    }
}

/// Register a object with its parent, along with its associated data. Do not use this directly,
/// use `api_layer!` to define probes for openxr handles instead.
///
/// # Safety
///
/// `parent` must correctly identify the parent object of `obj`. `Data` must be the only type you
/// will ever associate with this handle for the entirity of its lifetime.
unsafe fn register_object_boxed<
    O1: openxr::AsHandle + Send + Sync + 'static,
    O2: openxr::sys::Handle,
    Data: Any + Send + Sync,
>(
    obj: O1,
    parent: O2,
    probe: Box<Data>,
) {
    debug!(
        "reg {} {:#x}",
        std::any::type_name::<Data>(),
        obj.as_handle().into_raw()
    );
    OBJECT_OWNER.insert(obj.as_handle().into_raw(), parent.into_raw());
    OBJECT_OWNED
        .entry(parent.into_raw())
        .or_default()
        .insert(obj.as_handle().into_raw());
    DATA_REGISTRY.insert(obj.as_handle().into_raw(), (probe, Box::new(obj)));
}

/// Register a object with its parent, along with its associated data. Do not use this directly,
/// use `api_layer!` to define probes for openxr handles instead.
///
/// # Safety
///
/// `parent` must correctly identify the parent object of `obj`. `Data` must be the only type you
/// will ever associate with this handle for the entirity of its lifetime.
unsafe fn register_object<
    O1: openxr::AsHandle + Send + Sync + 'static,
    O2: openxr::sys::Handle,
    Data: Any + Send + Sync,
>(
    obj: O1,
    parent: O2,
    probe: Data,
) {
    register_object_boxed(obj, parent, Box::new(probe))
}

/// Unregister an object previously registered with `register_object`. Do not call this directly.
///
/// # Panic
///
/// If `obj` is not a valid handle to an object.
///
/// # Safety
///
/// can only be called after the application has called the destroy function of the given object.
unsafe fn unregister_object(obj: u64) {
    debug!("unreg {obj:#x}");
    remove_data(obj);
    let (_, owner) = OBJECT_OWNER.remove(&obj).unwrap();
    OBJECT_OWNED.get_mut(&owner).unwrap().remove(&obj);
    let children = OBJECT_OWNED
        .get(&obj)
        .map(|s| s.iter().copied().collect::<Vec<_>>())
        .unwrap_or_default();
    for child in children {
        unregister_object(child);
    }
    assert!(OBJECT_OWNED
        .remove(&obj)
        .map(|(_, children)| children.is_empty())
        .unwrap_or(true));
}

pub fn find_instance<O: openxr::sys::Handle>(obj: O) -> Result<openxr::sys::Instance, XrErr> {
    let mut curr = obj.into_raw();
    loop {
        let parent = *OBJECT_OWNER.get(&curr).ok_or(XrErr::ERROR_HANDLE_INVALID)?;
        if parent == curr {
            break Ok(openxr::sys::Instance::from_raw(curr));
        }
        curr = parent;
    }
}

/// # Docs
/// [xrGetInstanceProcAddr](https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#xrGetInstanceProcAddr)
/// # Safety
/// you are gay
#[allow(non_snake_case, unreachable_code)]
unsafe extern "system" fn get_instance_proc_addr<
    I: Hook<Target = openxr::sys::Instance> + ApiLayerOverride,
>(
    instance: openxr::sys::Instance,
    name: *const i8,
    function: *mut Option<openxr::sys::pfn::VoidFunction>,
) -> openxr::sys::Result {
    let wrapped_instance = try_xr!(instance.registered());
    let Ok(rusty_name) = <*const i8 as crate::util::Rustify>::to_rust_string(&name) else {
        return openxr::sys::Result::ERROR_VALIDATION_FAILURE;
    };
    if let Some(&pfn) = <I as ApiLayerOverride>::overrides().get(rusty_name) {
        debug!("{rusty_name} overridden");
        *function = Some(pfn);
        XrErr::SUCCESS
    } else {
        trace!("{rusty_name} unchanged");
        (wrapped_instance.fp().get_instance_proc_addr)(instance, name, function)
    }
}

/// # Safety
/// don't be stupid
pub unsafe fn negotiate_loader_api_layer_interface_impl<
    I: Hook<Target = openxr::sys::Instance> + ApiLayerOverride,
>(
    loader_info: &openxr::sys::loader::XrNegotiateLoaderInfo,
    _api_layer_name: *const u8,
    api_layer_request: &mut openxr::sys::loader::XrNegotiateApiLayerRequest,
) -> XrResult<()> {
    use openxr::sys::loader::*;
    use openxr::sys::CURRENT_API_VERSION;
    if loader_info.ty != XrNegotiateLoaderInfo::TYPE
        || loader_info.struct_version != XrNegotiateLoaderInfo::VERSION
        || loader_info.struct_size != size_of::<XrNegotiateLoaderInfo>()
    {
        return Err(XrErr::ERROR_INITIALIZATION_FAILED);
    }
    if api_layer_request.ty != XrNegotiateApiLayerRequest::TYPE
        || api_layer_request.struct_version != XrNegotiateApiLayerRequest::VERSION
        || api_layer_request.struct_size != size_of::<XrNegotiateApiLayerRequest>()
    {
        return Err(XrErr::ERROR_INITIALIZATION_FAILED);
    }
    if CURRENT_API_VERSION > loader_info.max_api_version
        || CURRENT_API_VERSION < loader_info.min_api_version
    {
        error!(
            "OpenXR API Layer doesn't support major version {} < {} < {}",
            loader_info.max_api_version, CURRENT_API_VERSION, loader_info.min_api_version
        );
        return Err(XrErr::ERROR_INITIALIZATION_FAILED);
    }
    api_layer_request.layer_interface_version = CURRENT_LOADER_API_LAYER_VERSION;
    api_layer_request.layer_api_version = CURRENT_API_VERSION;
    api_layer_request.get_instance_proc_addr = Some(get_instance_proc_addr::<I>);
    api_layer_request.create_api_layer_instance = Some(wrappers::create_api_layer_instance::<I>);
    Ok(())
}

#[macro_export]
macro_rules! api_layer {
    (
        hooks: {
            $($handle:ident: $data:ident),* $(,)?
        },
        override_fns: {
            $($fn_name:ident: $override_fn:ident),* $(,)?
        } $(,)?
    ) => {
        $crate::gen_override_table! {
            $crate;
            hooks: {
                $($handle: $data),*
            },
            override_fns: {
                $($fn_name: $override_fn),*
            }
        }

        // Generate the `Hooked` marker trait to allow `impl Hook`. This trait should NEVER be
        // impl'd by the user.
        $(unsafe impl $crate::Hooked<$data> for openxr::sys::$handle { })*
    };
}
