//! SI constants and the conversions used at the model boundaries.
//!
//! The model itself is SI and kelvin-only. Electronvolts and nanometres appear
//! only where the interface or a published value requires them.

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

/// Convert a temperature in kelvin to degrees Celsius.
pub fn kelvin_to_celsius(kelvin: f64) -> f64 {
    kelvin - ZERO_CELSIUS
}

/// Convert an energy in electronvolts to joules.
pub fn ev_to_joules(ev: f64) -> f64 {
    ev * ELEMENTARY_CHARGE
}

/// Convert an energy in joules to electronvolts.
pub fn joules_to_ev(joules: f64) -> f64 {
    joules / ELEMENTARY_CHARGE
}

/// Convert a photon energy in electronvolts to a wavelength in nanometres.
pub fn energy_to_wavelength_nm(ev: f64) -> f64 {
    1_239.841_984 / ev
}

/// Convert a wavelength in nanometres to a photon energy in electronvolts.
pub fn wavelength_nm_to_energy_ev(wavelength_nm: f64) -> f64 {
    1_239.841_984 / wavelength_nm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_voltage_at_room_temperature() {
        assert!((thermal_voltage(300.0) - 0.025_852).abs() < 1e-6);
    }

    #[test]
    fn temperature_conversions_round_trip() {
        for celsius in [-40.0, 0.0, 25.0, 100.0] {
            let kelvin = celsius_to_kelvin(celsius);
            assert!((kelvin_to_celsius(kelvin) - celsius).abs() < 1e-12);
        }
    }

    #[test]
    fn energy_and_wavelength_round_trip() {
        for ev in [0.7, 1.12, 2.03, 2.796, 3.1] {
            let nm = energy_to_wavelength_nm(ev);
            assert!((wavelength_nm_to_energy_ev(nm) - ev).abs() < 1e-12);
        }
    }

    #[test]
    fn electronvolt_is_the_charge_in_joules() {
        assert!((joules_to_ev(ev_to_joules(1.0)) - 1.0).abs() < 1e-15);
    }
}
