//! Dual and hyper-dual numbers for exact first and second derivatives.
//!
//! Planned modules:
//! - `dual`: forward-mode first derivatives
//! - `hyperdual`: second derivatives without finite differences
//! - `scalar`: trait shared by model code so it can be evaluated generically
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
