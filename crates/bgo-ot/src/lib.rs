//! Optimal transport distances and transport maps for posterior comparison.
//!
//! Modules:
//! - `sliced`: sliced Wasserstein-2 with a bootstrap projection-noise floor
//! - `sinkhorn`: planned log-domain stabilised entropic transport
//! - `interpolate`: planned displacement interpolation between particle sets
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

pub mod sliced;

pub use sliced::{sliced_wasserstein_2, SlicedError, SlicedW2};

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
