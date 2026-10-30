//! Native Rust port of OSOR-SDXL Phase II.
//!
//! CPU is the default backend; Metal, CUDA and WGPU are opt-in Cargo features.
//! Model artifacts are supplied locally by the application. Nothing is downloaded.
//!
//! # Quick start
//!
//! Load a [prepared bundle](https://github.com/doublesharp/osor-rs/blob/main/docs/model-preparation.md),
//! generate reproducible noise, and run inference with normalized RGB and mask tensors:
//!
//! ```no_run
//! use osor::{InferenceInput, InferenceResult, Osor, Result, Tensor};
//!
//! fn remove_object(bundle: &str, image: &Tensor, mask: &Tensor) -> Result<InferenceResult> {
//!     let model = Osor::load(bundle)?;
//!     let image = image.to_device(model.device())?;
//!     let mask = mask.to_device(model.device())?;
//!     let (batch, _, height, width) = image.dims4()?;
//!     let noise = model.seeded_noise(42, batch, height, width)?;
//!     model.infer(InferenceInput {
//!         image: &image,
//!         mask: &mask,
//!         posterior_noise: &noise.posterior,
//!         diffusion_noise: &noise.diffusion,
//!     })
//! }
//! ```
//!
//! `image` is F32 RGB `[B, 3, H, W]` in `[0, 1]`; `mask` is F32
//! `[B, 1, H, W]` in `[0, 1]`, with white marking removal. Height and width are
//! multiples of eight in `32..=1024`. The result contains decoded RGB,
//! full-resolution predicted edit alpha, and latent alpha. Retain the loaded
//! [`Osor`] across jobs to reuse its weights.
//!
//! # Backends and image bytes
//!
//! CPU is the default. Enable `metal`, `cuda`, or `wgpu` and select a [`Backend`]
//! through [`Osor::builder`] for GPU execution. [`preprocess`] provides RGB8/mask
//! resizing, tensor padding, output quantization and optional alpha compositing.
//! See the [usage guide](https://github.com/doublesharp/osor-rs#use-from-rust)
//! for backend setup and input/output details.
#![forbid(unsafe_code)]

pub mod artifact;
pub mod conversion;
pub mod native;
pub mod noise;
pub mod parity;
pub mod pipeline;
pub mod preprocess;
pub mod runtime;
pub mod scheduler;
pub mod tensor_io;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use pipeline::{CancellationToken, InferenceInput, InferenceLimits, InferenceResult, Stage};
pub use runtime::{Backend, LoadOptions, Osor, OsorBuilder};

pub use candle_core::{DType, Device, Tensor};

/// Errors returned by validation, offline artifact tools, and inference.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("invalid input: {0}")]
    /// A caller-supplied tensor, option or file failed validation.
    InvalidInput(String),
    #[error("invalid model artifact: {0}")]
    /// A model bundle, manifest or conversion input is malformed or incompatible.
    InvalidArtifact(String),
    #[error("unavailable: {0}")]
    /// The requested backend, device or feature is not available in this build or host.
    Unavailable(String),
    #[error("inference cancelled")]
    /// An observer or conversion checkpoint returned this to stop the job.
    Cancelled,
    #[error(transparent)]
    /// A tensor operation failed in Candle.
    Tensor(#[from] candle_core::Error),
    #[error(transparent)]
    /// A filesystem or stream operation failed.
    Io(#[from] std::io::Error),
    #[error(transparent)]
    /// A manifest, tolerance or conversion plan could not be (de)serialized.
    Json(#[from] serde_json::Error),
}

/// Result type used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Immutable reference used by the initial SDXL Phase II port.
pub const UPSTREAM_REVISION: &str = "d5b68e822996d61aae6ba612e5e9b5a14f576879";
