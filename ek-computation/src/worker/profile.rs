use std::ffi::{CString, c_char};

unsafe extern "C" {
    fn ek_nvtx_range_push(message: *const c_char);
    fn ek_nvtx_range_pop();
    fn ek_nvtx_range_start(message: *const c_char) -> u64;
    fn ek_nvtx_range_end(id: u64);
}

pub struct NvtxRange {
    enabled: bool,
}

impl NvtxRange {
    pub fn new(label: impl Into<String>) -> Self {
        if std::env::var("EK_NSYS_PROFILE").as_deref() != Ok("1") {
            return Self { enabled: false };
        }
        let Ok(label) = CString::new(label.into()) else {
            return Self { enabled: false };
        };
        unsafe { ek_nvtx_range_push(label.as_ptr()) };
        Self { enabled: true }
    }

    pub fn expert(request_id: &str, microbatch_id: u32, layer_id: u32, expert_id: &str) -> Self {
        Self::new(format!(
            "EK:E:req={request_id}:u={microbatch_id}:l={layer_id}:expert={expert_id}"
        ))
    }

    pub fn phase(phase: &str, request_id: &str, microbatch_id: u32, layer_id: u32) -> Self {
        Self::new(format!(
            "EKR:{phase}:req={request_id}:u={microbatch_id}:l={layer_id}"
        ))
    }
}

pub struct NvtxAsyncRange {
    id: Option<u64>,
}

impl NvtxAsyncRange {
    pub fn phase(phase: &str, request_id: &str, microbatch_id: u32, layer_id: u32) -> Self {
        if std::env::var("EK_NSYS_PROFILE").as_deref() != Ok("1") {
            return Self { id: None };
        }
        let label = format!("EKR:{phase}:req={request_id}:u={microbatch_id}:l={layer_id}");
        let Ok(label) = CString::new(label) else {
            return Self { id: None };
        };
        let id = unsafe { ek_nvtx_range_start(label.as_ptr()) };
        Self { id: Some(id) }
    }
}

impl Drop for NvtxAsyncRange {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            unsafe { ek_nvtx_range_end(id) };
        }
    }
}

impl Drop for NvtxRange {
    fn drop(&mut self) {
        if self.enabled {
            // SAFETY: every enabled guard owns exactly one unmatched range push.
            unsafe { ek_nvtx_range_pop() };
        }
    }
}
