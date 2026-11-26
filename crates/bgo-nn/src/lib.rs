//! Hand-written conditional flow engine for amortised neural posteriors.
//!
//! Boundaries: this crate stays free of I/O and browser APIs so it can be
//! unit-tested on the host and linked into `bgo-wasm`. The exported model is a
//! two-parameter M0 conditional rational-quadratic spline flow trained offline
//! under `train/`.

#![forbid(unsafe_code)]

pub mod flow;
pub mod manifest;
pub mod spline;

pub use flow::{Flow, FlowError};
pub use manifest::FlowManifest;
pub use spline::Spline;

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
