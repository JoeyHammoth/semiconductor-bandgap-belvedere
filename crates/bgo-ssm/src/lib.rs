//! State-space thermal model, Kalman filtering and Koopman operators.
//!
//! Implemented modules:
//! - `thermal`: plate/junction/sensor first-order twin and lag recovery
//! - `ukf`: unscented Kalman filter, RTS smoother, GP residual discrepancy and
//!   the Kalman marginal likelihood
//! - `matern`: Matern-3/2 Gaussian process as a linear SDE state
//! - `dmd`: Hankel-DMDc time constants from the delay-embedded trace
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

pub mod dmd;
pub mod matern;
pub mod thermal;
pub mod ukf;

pub use dmd::{estimate_thermal_modes, ThermalModes};
pub use matern::{matern_discrepancy, MaternDiscrepancy};
pub use thermal::{heater_target_k, simulate_thermal, HeatPulse, ThermalConfig, ThermalRun};
pub use ukf::{estimate_thermal, smooth_thermal, ThermalEstimate};

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
