//! Diode physics, band-gap models, Lambert W, colour science and units.
//!
//! Planned modules:
//! - `units`: SI constants and conversions (eV <-> J, Celsius <-> kelvin)
//! - `diode`: Shockley model, ideal constant-current voltage, series resistance
//! - `gap`: Varshni and Bose-Einstein gap models
//! - `lambert_w`: closed-form I(V) with series resistance
//! - `colour`: Planck spectrum, CIE 1931 analytic fit, sRGB conversion
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Elementary charge, coulomb (exact, SI 2019).
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
/// Boltzmann constant, joule per kelvin (exact, SI 2019).
pub const BOLTZMANN: f64 = 1.380_649e-23;
/// Planck constant, joule second (exact, SI 2019).
pub const PLANCK: f64 = 6.626_070_15e-34;
/// Speed of light in vacuum, metre per second (exact).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// 0 degrees Celsius in kelvin (exact).
pub const ZERO_CELSIUS: f64 = 273.15;

/// Thermal voltage `kT/q` in volts.
pub fn thermal_voltage(temperature_k: f64) -> f64 {
    BOLTZMANN * temperature_k / ELEMENTARY_CHARGE
}

/// Convert a temperature in degrees Celsius to kelvin.
pub fn celsius_to_kelvin(celsius: f64) -> f64 {
    celsius + ZERO_CELSIUS
}

/// Convert an energy in electronvolts to joules.
pub fn ev_to_joules(ev: f64) -> f64 {
    ev * ELEMENTARY_CHARGE
}

/// Convert a photon energy in electronvolts to a wavelength in nanometres.
pub fn energy_to_wavelength_nm(ev: f64) -> f64 {
    1_239.841_984 / ev
}

/// Varshni gap `E_g(T) = E_g(0) - alpha*T^2/(T+beta)` in eV.
///
/// `eg0` is in eV, `alpha` in eV/K, `beta` in K, `temperature_k` in K.
pub fn varshni_gap(eg0: f64, alpha: f64, beta: f64, temperature_k: f64) -> f64 {
    eg0 - alpha * temperature_k * temperature_k / (temperature_k + beta)
}

/// Ideal constant-current diode voltage from the model specification, in volts.
///
/// `V(T) = (kT/q)[ln(I/B) - eta*ln(T)] + E_g/q`, with `B` passed as `ln B`,
/// `E_g` in eV (numerically equal to volts after dividing by `q`), and `I` in
/// amperes. Series resistance and finite ideality are deliberately absent; the
/// full model lives in the `diode` module once it lands.
pub fn ideal_diode_voltage(
    current_a: f64,
    temperature_k: f64,
    ln_b: f64,
    eta: f64,
    eg_ev: f64,
) -> f64 {
    let vt = thermal_voltage(temperature_k);
    vt * ((current_a.ln() - ln_b) - eta * temperature_k.ln()) + eg_ev
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_voltage_at_room_temperature() {
        assert!((thermal_voltage(300.0) - 0.025_852).abs() < 1e-6);
    }

    #[test]
    fn varshni_reduces_to_zero_kelvin_gap() {
        assert_eq!(varshni_gap(1.1557, 7.021e-4, 1108.0, 0.0), 1.1557);
    }

    #[test]
    fn gap_shrinks_with_temperature() {
        let cold = varshni_gap(1.1557, 7.021e-4, 1108.0, 200.0);
        let warm = varshni_gap(1.1557, 7.021e-4, 1108.0, 350.0);
        assert!(warm < cold);
    }

    #[test]
    fn wavelength_of_a_red_photon() {
        assert!((energy_to_wavelength_nm(2.030) - 610.76).abs() < 0.02);
    }

    #[test]
    fn ideal_voltage_moves_negatively_with_temperature() {
        let cold = ideal_diode_voltage(1e-3, 300.0, 20.0, 0.0, 1.12);
        let warm = ideal_diode_voltage(1e-3, 320.0, 20.0, 0.0, 1.12);
        assert!(warm < cold);
    }
}
