//! Samplers, evidence estimation and convergence diagnostics.
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

pub mod diagnostics;
pub mod mclmc;
pub mod model;

pub use diagnostics::{bulk_ess, split_rhat, Diagnostics};
pub use mclmc::{run_mclmc, MclmcConfig, MclmcError, MclmcOutput};
pub use model::{Case, Model, ModelError};

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
