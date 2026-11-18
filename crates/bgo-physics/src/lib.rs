//! Diode physics, band-gap models, Lambert W, colour science and units.
//!
//! The crate is pure: no I/O, no browser APIs and no clock. It is unit-tested on
//! the host and linked into `bgo-wasm`.
//!
//! Modules:
//! - [`units`]: SI constants and conversions
//! - [`fit`]: ordinary least squares for the lab's M0 straight line
//! - [`diode`]: full Shockley model, ideal limit and two-current identity
//! - [`gap`]: Varshni and corrected Bose–Einstein gap models
//! - [`lambert_w`]: real branches of `W`
//! - [`colour`]: Planck spectrum, CIE 1931 fit and sRGB

#![forbid(unsafe_code)]

pub mod colour;
pub mod diode;
pub mod fit;
pub mod gap;
pub mod lambert_w;
pub mod units;

pub use colour::{
    blackbody_srgb, blackbody_xyz, chromaticity, cie_xyz, encode_srgb, linear_srgb_from_xyz,
    planck_spectral_radiance, Chromaticity, LinearSrgb, Srgb, Xyz,
};
pub use diode::{
    diode_current, diode_voltage, ideal_diode_voltage, ln_saturation_current, saturation_current,
    two_current_voltage_difference, DiodeError, DiodeParameters,
};
pub use fit::{fit_m0, FitError, LinearFit, Observation};
pub use gap::{bose_einstein_gap, varshni_gap, BoseEinsteinGap, VarshniGap};
pub use lambert_w::{lambert_w, lambert_w0, lambert_w_minus_one, LambertWBranch, LambertWError};
pub use units::{
    celsius_to_kelvin, energy_to_wavelength_nm, ev_to_joules, joules_to_ev, kelvin_to_celsius,
    thermal_voltage, wavelength_nm_to_energy_ev, BOLTZMANN, ELEMENTARY_CHARGE, PLANCK,
    SPEED_OF_LIGHT, ZERO_CELSIUS,
};

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
