//! State-space thermal model, Kalman filtering and Koopman operators.
//!
//! Implemented modules:
//! - `thermal`: plate/junction/sensor first-order twin and lag recovery
//!
//! Planned modules:
//! - `ukf`: unscented Kalman filter and RTS smoother
//! - `matern`: Matern-3/2 Gaussian process as a linear SDE state
//! - `dmd`: Hankel-DMD with control and small SVD
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

pub mod thermal;

pub use thermal::{simulate_thermal, ThermalConfig, ThermalRun};

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
