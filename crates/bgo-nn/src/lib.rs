//! Hand-written conditional flow engine for amortised neural posteriors.
//!
//! Planned modules:
//! - `manifest`: JSON architecture description and weight layout
//! - `layers`: dense layers and activations
//! - `spline`: rational-quadratic spline transforms
//! - `flow`: sampling and log-probability evaluation
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
