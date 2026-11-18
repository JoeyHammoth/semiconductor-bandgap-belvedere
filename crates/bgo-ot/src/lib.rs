//! Optimal transport distances and transport maps for posterior comparison.
//!
//! Planned modules:
//! - `sinkhorn`: log-domain stabilised entropic transport
//! - `sliced`: sliced Wasserstein-2 with bootstrap noise floor
//! - `interpolate`: displacement interpolation between particle sets
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
