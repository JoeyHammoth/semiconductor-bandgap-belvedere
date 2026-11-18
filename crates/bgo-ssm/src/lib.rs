//! State-space thermal model, Kalman filtering and Koopman operators.
//!
//! Planned modules:
//! - `thermal`: plate/junction/sensor model and exact discretisation
//! - `ukf`: unscented Kalman filter and RTS smoother
//! - `matern`: Matern-3/2 Gaussian process as a linear SDE state
//! - `dmd`: Hankel-DMD with control and small SVD
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
