//! Native Rust port of OSOR-SDXL Phase II.
//!
//! Current status: repository and reference-contract scaffold. Neural inference and the
//! numerical parity harness are pending; this crate does not yet remove objects.
#![forbid(unsafe_code)]

/// Immutable reference used by the initial SDXL Phase II port.
pub const UPSTREAM_REVISION: &str = "d5b68e822996d61aae6ba612e5e9b5a14f576879";
