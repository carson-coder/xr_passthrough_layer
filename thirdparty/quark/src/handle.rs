use crate::prelude::*;
use dashmap::{mapref::one::MappedRefMut, DashMap};
use openxr::sys::Handle as _;
use std::{
    any::{Any, TypeId},
    sync::LazyLock,
};

// pub trait HandleData: Sized + Send + Sync + 'static {
//     fn registry<'a>() -> &'a DashMap<u64, Self, BuildHasherDefault<FxHasher>>;

//     fn store_in_new_handle(self) -> u64 {

//         let id = Self::counter()
//             .fetch_update(
//                 std::sync::atomic::Ordering::SeqCst,
//                 std::sync::atomic::Ordering::SeqCst,
//                 |x| Some(x + 1),
//             )
//             .unwrap();
//         Self::registry().insert(id, self);
//         id
//     }
//     fn store_in_existing_handle(self, handle: u64) {
//         Self::registry().insert(handle, self);
//     }

//     fn destroy(handle: u64) -> XrResult {
//         match Self::registry().remove(&handle) {
//             Some(_) => Ok(()),
//             None => Err(XrErr::ERROR_HANDLE_INVALID),
//         }
//     }

//     fn borrow_raw<'a>(handle: u64) -> Result<RefMut<'a, u64, Self>, XrErr> {
//         Self::registry()
//             .get_mut(&handle)
//             .ok_or(XrErr::ERROR_HANDLE_INVALID)
//     }
// }
