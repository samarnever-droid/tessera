//! Native TPU handshake via the OpenXLA PJRT C API — the stable ABI that
//! JAX, TensorFlow and PyTorch/XLA all sit on. Loaded directly with dlopen:
//! **no Python, no PyTorch, no crates**.
//!
//! Scope (v1): plugin discovery, `PJRT_Plugin_Initialize`, `PJRT_Client_Create`,
//! platform name/version, process index and addressable-device enumeration —
//! a full native "hello TPU". HLO/MLIR execution (BufferFromHostBuffer +
//! Execute) is the phase-2 milestone; struct layouts here are transcribed
//! from xla/pjrt/c/pjrt_c_api.h (PJRT_API_MINOR 115), and every call is
//! guarded so older/newer plugins degrade to a clear message, never a crash.

use std::ffi::{c_char, c_int, c_void, CStr, CString};

// ─── opaque handles ─────────────────────────────────────────────────────────
#[repr(C)]
struct ExtBase {
    _x: [u8; 0],
}
#[repr(C)]
struct PjrtError {
    _x: [u8; 0],
}
#[repr(C)]
struct Client {
    _x: [u8; 0],
}
#[repr(C)]
struct Device {
    _x: [u8; 0],
}
#[repr(C)]
struct NamedValue {
    _x: [u8; 0],
}

type ErrPtr = *mut PjrtError;

// ─── args structs (field order transcribed from pjrt_c_api.h) ───────────────
#[repr(C)]
struct ApiVersion {
    struct_size: usize,
    extension_start: *mut ExtBase,
    major: c_int,
    minor: c_int,
}

#[repr(C)]
struct ErrorDestroyArgs {
    struct_size: usize,
    extension_start: *mut ExtBase,
    error: *mut PjrtError,
}
#[repr(C)]
struct ErrorMessageArgs {
    struct_size: usize,
    extension_start: *mut ExtBase,
    error: *const PjrtError,
    message: *const c_char,
    message_size: usize,
}
#[repr(C)]
struct ErrorCodeArgs {
    struct_size: usize,
    extension_start: *mut ExtBase,
    error: *const PjrtError,
    code: c_int,
}
#[repr(C)]
struct PluginInitializeArgs {
    struct_size: usize,
    extension_start: *mut ExtBase,
}
#[repr(C)]
struct ClientCreateArgs {
    struct_size: usize,
    extension_start: *mut ExtBase,
    create_options: *const NamedValue,
    num_options: usize,
    kv_get: *mut c_void,
    kv_get_user_arg: *mut c_void,
    kv_put: *mut c_void,
    kv_put_user_arg: *mut c_void,
    client: *mut Client,
    kv_try_get: *mut c_void,
    kv_try_get_user_arg: *mut c_void,
}
#[repr(C)]
struct ClientDestroyArgs {
    struct_size: usize,
    extension_start: *mut ExtBase,
    client: *mut Client,
}
/// shared shape of Client_PlatformName / Client_PlatformVersion
#[repr(C)]
struct ClientStrArgs {
    struct_size: usize,
    extension_start: *mut ExtBase,
    client: *mut Client,
    s: *const c_char,
    s_len: usize,
}
#[repr(C)]
struct ProcessIndexArgs {
    struct_size: usize,
    extension_start: *mut ExtBase,
    client: *mut Client,
    process_index: c_int,
}
#[repr(C)]
struct AddressableDevicesArgs {
    struct_size: usize,
    extension_start: *mut ExtBase,
    client: *mut Client,
    devices: *mut *mut Device,
    num_devices: usize,
}

// ─── the API struct (prefix — we never read past addressable_devices) ───────
#[repr(C)]
struct Api {
    struct_size: usize,
    extension_start: *mut ExtBase,
    version: ApiVersion,
    // 40..64: error functions
    error_destroy: unsafe extern "C" fn(*mut ErrorDestroyArgs),
    error_message: unsafe extern "C" fn(*mut ErrorMessageArgs),
    error_get_code: unsafe extern "C" fn(*mut ErrorCodeArgs),
    // 64..80: plugin
    plugin_initialize: Option<unsafe extern "C" fn(*mut PluginInitializeArgs) -> ErrPtr>,
    plugin_attributes: usize,
    // 80..120: event slots (unused in v1, kept for offset fidelity)
    event_destroy: usize,
    event_is_ready: usize,
    event_error: usize,
    event_await: usize,
    event_on_ready: usize,
    // 120..168: client functions
    client_create: Option<unsafe extern "C" fn(*mut ClientCreateArgs) -> ErrPtr>,
    client_destroy: Option<unsafe extern "C" fn(*mut ClientDestroyArgs) -> ErrPtr>,
    client_platform_name: Option<unsafe extern "C" fn(*mut ClientStrArgs) -> ErrPtr>,
    client_process_index: usize,
    client_platform_version: Option<unsafe extern "C" fn(*mut ClientStrArgs) -> ErrPtr>,
    client_devices: usize,
    client_addressable_devices: Option<unsafe extern "C" fn(*mut AddressableDevicesArgs) -> ErrPtr>,
}

unsafe fn err_details(api: &Api, e: ErrPtr) -> (i32, String) {
    let mut margs = ErrorMessageArgs {
        struct_size: std::mem::size_of::<ErrorMessageArgs>(),
        extension_start: std::ptr::null_mut(),
        error: e,
        message: std::ptr::null(),
        message_size: 0,
    };
    (api.error_message)(&mut margs);
    let msg = if margs.message.is_null() {
        String::new()
    } else {
        CStr::from_ptr(margs.message).to_string_lossy().into_owned()
    };
    let mut cargs = ErrorCodeArgs {
        struct_size: std::mem::size_of::<ErrorCodeArgs>(),
        extension_start: std::ptr::null_mut(),
        error: e,
        code: -1,
    };
    (api.error_get_code)(&mut cargs);
    let mut dargs = ErrorDestroyArgs {
        struct_size: std::mem::size_of::<ErrorDestroyArgs>(),
        extension_start: std::ptr::null_mut(),
        error: e,
    };
    (api.error_destroy)(&mut dargs);
    (cargs.code, msg)
}

// ─── dlopen/dlsym (same zero-dep FFI as the CUDA path) ──────────────────────
mod ffi {
    use std::ffi::{c_char, c_int, c_void};
    #[link(name = "dl")]
    extern "C" {
        pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
}

/// What went wrong while probing: `Skip` = this box has no TPU runtime
/// (never a failure), `Fail` = a TPU runtime is present but misbehaving.
pub enum Probe {
    Skip(String),
    Fail(String),
}

use Probe::Fail;

pub struct TpuInfo {
    pub lib: String,
    pub api_version: (i32, i32),
    pub platform: String,
    pub platform_version: String,
    pub process_index: i32,
    pub devices: usize,
}

fn candidate_libs() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("TESSERA_TPU_LIB") {
        if !p.is_empty() {
            v.push(p);
        }
    }
    for n in ["libtpu.so", "pjrt_c_api_tpu_plugin.so", "libpjrt_c_api_tpu_plugin.so"] {
        v.push(n.to_string());
    }
    for p in [
        "/lib/libtpu.so",
        "/usr/lib/libtpu.so",
        "/usr/local/lib/libtpu.so",
        "/usr/local/lib/x86_64-linux-gnu/libtpu.so",
    ] {
        v.push(p.to_string());
    }
    // filesystem scan of the usual python-package homes (files only — we never
    // execute python to find the runtime)
    fn scan(base: &std::path::Path, depth: usize, v: &mut Vec<String>) {
        if depth > 3 {
            return;
        }
        if let Ok(rd) = std::fs::read_dir(base) {
            for e in rd.flatten() {
                let p = e.path();
                let name = e.file_name().to_string_lossy().into_owned();
                if name == "libtpu.so" || (name.contains("pjrt") && name.contains("tpu") && name.ends_with(".so")) {
                    v.push(p.to_string_lossy().into_owned());
                } else if p.is_dir()
                    && (name.starts_with("python3")
                        || name == "dist-packages"
                        || name == "site-packages"
                        || name.starts_with("libtpu")
                        || name == "lib")
                {
                    scan(&p, depth + 1, v);
                }
            }
        }
    }
    if let Ok(rd) = std::fs::read_dir("/usr/lib") {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if n.starts_with("python3") || n.starts_with("libtpu") {
                let p = e.path();
                if p.is_file() && (n == "libtpu.so" || (n.contains("pjrt") && n.contains("tpu"))) {
                    v.push(p.to_string_lossy().into_owned());
                } else if p.is_dir() {
                    scan(&p, 1, &mut v);
                }
            }
        }
    }
    v
}

/// Full native handshake: load plugin -> version -> initialize -> client ->
/// platform strings -> device enumeration.
pub fn probe() -> Result<TpuInfo, Probe> {
    let mut last_missing: Option<String> = None;
    for path in candidate_libs() {
        let Ok(cname) = CString::new(path.as_str()) else { continue };
        let h = unsafe { ffi::dlopen(cname.as_ptr(), 2) }; // RTLD_NOW
        if h.is_null() {
            last_missing = Some(path.clone());
            continue;
        }
        return handshake(h, &path);
    }
    Err(Probe::Skip(format!(
        "no TPU runtime plugin on this box (looked for libtpu.so / pjrt_c_api_tpu_plugin.so{})",
        last_missing.map(|p| format!(", last tried {p}")).unwrap_or_default()
    )))
}

fn handshake(h: *mut c_void, lib: &str) -> Result<TpuInfo, Probe> {
    unsafe {
        // the TPU plugin exposes the API through one of two symbols
        let mut api_ptr: *const Api = std::ptr::null();
        for (sym, is_data) in [("TPU_PJRT_C_API", true), ("GetPjrtApi", false)] {
            let cs = CString::new(sym).unwrap();
            let p = ffi::dlsym(h, cs.as_ptr());
            if p.is_null() {
                continue;
            }
            api_ptr = if is_data {
                *(p as *const *const Api)
            } else {
                let f: unsafe extern "C" fn() -> *const Api = std::mem::transmute(p);
                f()
            };
            break;
        }
        if api_ptr.is_null() {
            return Err(Fail(format!(
                "'{lib}' loaded but exports neither TPU_PJRT_C_API nor GetPjrtApi — not a PJRT TPU plugin?"
            )));
        }
        let api = &*api_ptr;
        if api.struct_size < 512 {
            return Err(Fail(format!(
                "'{lib}': PJRT_Api.struct_size = {} — implausible header, refusing",
                api.struct_size
            )));
        }
        let (vmaj, vmin) = (api.version.major, api.version.minor);

        // plugin init
        let Some(plugin_initialize) = api.plugin_initialize else {
            return Err(Fail(format!("'{lib}': PJRT_Plugin_Initialize missing (API v{vmaj}.{vmin})")));
        };
        let mut iargs = PluginInitializeArgs {
            struct_size: std::mem::size_of::<PluginInitializeArgs>(),
            extension_start: std::ptr::null_mut(),
        };
        let e = plugin_initialize(&mut iargs);
        if !e.is_null() {
            let (code, msg) = err_details(api, e);
            return Err(Fail(format!("PJRT_Plugin_Initialize: code {code}: {msg}")));
        }

        // client
        let Some(client_create) = api.client_create else {
            return Err(Fail(format!("'{lib}': PJRT_Client_Create missing")));
        };
        let mut cargs = ClientCreateArgs {
            struct_size: std::mem::size_of::<ClientCreateArgs>(),
            extension_start: std::ptr::null_mut(),
            create_options: std::ptr::null(),
            num_options: 0,
            kv_get: std::ptr::null_mut(),
            kv_get_user_arg: std::ptr::null_mut(),
            kv_put: std::ptr::null_mut(),
            kv_put_user_arg: std::ptr::null_mut(),
            client: std::ptr::null_mut(),
            kv_try_get: std::ptr::null_mut(),
            kv_try_get_user_arg: std::ptr::null_mut(),
        };
        let e = client_create(&mut cargs);
        if !e.is_null() {
            let (code, msg) = err_details(api, e);
            return Err(Fail(format!("PJRT_Client_Create: code {code}: {msg}")));
        }
        let client = cargs.client;

        // strings
        let get_str = |f: Option<unsafe extern "C" fn(*mut ClientStrArgs) -> ErrPtr>| -> Result<String, String> {
            let Some(f) = f else { return Err("accessor missing".into()) };
            let mut a = ClientStrArgs {
                struct_size: std::mem::size_of::<ClientStrArgs>(),
                extension_start: std::ptr::null_mut(),
                client,
                s: std::ptr::null(),
                s_len: 0,
            };
            let e = f(&mut a);
            if !e.is_null() {
                let (code, msg) = err_details(api, e);
                return Err(format!("code {code}: {msg}"));
            }
            let s = if a.s.is_null() {
                String::new()
            } else {
                std::slice::from_raw_parts(a.s as *const u8, a.s_len).iter().map(|&b| b as char).collect::<String>()
            };
            Ok(s)
        };
        let platform = match get_str(api.client_platform_name) {
            Ok(s) => s,
            Err(e) => return Err(Fail(format!("PJRT_Client_PlatformName: {e}"))),
        };
        let platform_version = get_str(api.client_platform_version).unwrap_or_default();

        // process index + device enumeration (best-effort, non-fatal)
        let mut process_index: i32 = -1;
        let mut devices: usize = 0;
        {
            let mut pargs = ProcessIndexArgs {
                struct_size: std::mem::size_of::<ProcessIndexArgs>(),
                extension_start: std::ptr::null_mut(),
                client,
                process_index: -1,
            };
            // the slot is usize in our prefix struct; only call if non-null
            if api.client_process_index != 0 {
                let f: unsafe extern "C" fn(*mut ProcessIndexArgs) -> ErrPtr =
                    std::mem::transmute(api.client_process_index);
                let e = f(&mut pargs);
                if !e.is_null() {
                    let _ = err_details(api, e);
                } else {
                    process_index = pargs.process_index;
                }
            }
        }
        if let Some(f) = api.client_addressable_devices {
            let mut list: Vec<*mut Device> = vec![std::ptr::null_mut(); 64];
            let mut dargs = AddressableDevicesArgs {
                struct_size: std::mem::size_of::<AddressableDevicesArgs>(),
                extension_start: std::ptr::null_mut(),
                client,
                devices: list.as_mut_ptr(),
                num_devices: 0,
            };
            let e = f(&mut dargs);
            if !e.is_null() {
                let _ = err_details(api, e);
            } else {
                devices = dargs.num_devices;
            }
        }

        // teardown (best-effort)
        if let Some(client_destroy) = api.client_destroy {
            let mut dargs = ClientDestroyArgs {
                struct_size: std::mem::size_of::<ClientDestroyArgs>(),
                extension_start: std::ptr::null_mut(),
                client,
            };
            let e = client_destroy(&mut dargs);
            if !e.is_null() {
                let _ = err_details(api, e);
            }
        }

        Ok(TpuInfo { lib: lib.to_string(), api_version: (vmaj, vmin), platform, platform_version, process_index, devices })
    }
}
