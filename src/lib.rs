//! Native Rust port of OSOR-SDXL Phase II.
//!
//! CPU is the default backend; Metal, CUDA and WGPU are opt-in Cargo features.
//! Model artifacts are supplied locally by the application. Nothing is downloaded.
#![forbid(unsafe_code)]

pub mod artifact;
pub mod conversion;
pub mod native;
pub mod parity;
pub mod pipeline;
pub mod preprocess;
pub mod runtime;
pub mod scheduler;
pub mod tensor_io;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use pipeline::{InferenceInput, InferenceLimits, InferenceResult};
pub use runtime::{Backend, LoadOptions, Osor};

pub use candle_core::{DType, Device, Tensor};

/// Errors returned by validation, offline artifact tools, and inference.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid model artifact: {0}")]
    InvalidArtifact(String),
    #[error("unavailable: {0}")]
    Unavailable(String),
    #[error("inference cancelled")]
    Cancelled,
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Immutable reference used by the initial SDXL Phase II port.
pub const UPSTREAM_REVISION: &str = "d5b68e822996d61aae6ba612e5e9b5a14f576879";
