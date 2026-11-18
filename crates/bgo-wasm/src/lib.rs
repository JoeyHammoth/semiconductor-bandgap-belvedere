//! WebAssembly surface for the Browser core.
//!
//! The JavaScript side speaks typed arrays; configuration crosses as small
//! structs. Each worker owns its own instance, so nothing here may hold global
//! mutable state.

#![forbid(unsafe_code)]

use wasm_bindgen::prelude::*;

/// Core version, rendered in the status bar.
#[wasm_bindgen]
pub fn core_version() -> String {
    format!("bgo-wasm {}", env!("CARGO_PKG_VERSION"))
}

/// Ideal constant-current diode voltage from the model specification.
///
/// `eg_ev` is the gap in electronvolts; numerically it equals volts because
/// `E_g[J] / q = E_g[eV]`.
#[wasm_bindgen]
pub fn ideal_diode_voltage(current: f64, temperature: f64, ln_b: f64, eta: f64, eg_ev: f64) -> f64 {
    bgo_physics::ideal_diode_voltage(current, temperature, ln_b, eta, eg_ev)
}

/// Varshni gap at a temperature, in eV.
#[wasm_bindgen]
pub fn varshni_gap(eg0: f64, alpha: f64, beta: f64, temperature: f64) -> f64 {
    bgo_physics::varshni_gap(eg0, alpha, beta, temperature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_reported() {
        assert!(core_version().starts_with("bgo-wasm"));
    }
}
