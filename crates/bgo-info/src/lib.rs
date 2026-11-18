//! Fisher information, sloppiness, and Bayesian experiment design.
//!
//! Planned modules:
//! - `fisher`: Fisher information from the model Jacobian
//! - `sloppy`: eigenvalue spectrum and eigenvector labelling
//! - `design`: D-optimal candidate selection
//! - `eig`: nested-Monte-Carlo expected information gain
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
