//! Shockley diode model in the constant-current and constant-voltage directions.
//!
//! The saturation current is
//! `I_s(T) = B T^eta exp(-E_g(T)/(n_a k T))`, with `B` supplied as `ln B` so
//! that small prefactors do not underflow. The terminal relation is
//! `V = I R_s + n V_T ln(I/I_s + 1)`.
//!
//! The lab's M0 equation is the `n = n_a = 1`, `R_s = 0`, `I >> I_s` limit.
//! It is provided separately as [`ideal_diode_voltage`] so that the
//! approximation is never silently substituted for the full model.

use crate::lambert_w::{lambert_w0, LambertWError};
use crate::units::thermal_voltage;

/// Parameters of a single diode at a fixed temperature.
///
/// `activation_exponent` is the model's `n_a`; `ideality` is the terminal
/// ideality factor `n`. They are separate because the saturation-current
/// temperature dependence and the exponential slope are different effects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiodeParameters {
    /// `ln B`, with `B` in `A K^-eta`.
    pub ln_b: f64,
    /// Prefactor temperature exponent `eta`.
    pub eta: f64,
    /// Band gap at the junction temperature, in eV.
    pub eg_ev: f64,
    /// Terminal ideality factor `n`.
    pub ideality: f64,
    /// Saturation-current activation exponent `n_a`.
    pub activation_exponent: f64,
    /// Series resistance in ohms.
    pub series_resistance_ohm: f64,
}

impl DiodeParameters {
    pub const fn new(
        ln_b: f64,
        eta: f64,
        eg_ev: f64,
        ideality: f64,
        activation_exponent: f64,
        series_resistance_ohm: f64,
    ) -> Self {
        Self {
            ln_b,
            eta,
            eg_ev,
            ideality,
            activation_exponent,
            series_resistance_ohm,
        }
    }

    /// The ideal constant-current model: `n = n_a = 1`, no series resistance.
    pub const fn ideal(ln_b: f64, eta: f64, eg_ev: f64) -> Self {
        Self::new(ln_b, eta, eg_ev, 1.0, 1.0, 0.0)
    }
}

/// Why a diode evaluation was rejected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiodeError {
    NonFinite,
    NonPositiveTemperature,
    NonPositiveCurrent,
    NonPositiveIdeality,
    NonPositiveActivationExponent,
    NegativeSeriesResistance,
    LambertW(LambertWError),
}

impl From<LambertWError> for DiodeError {
    fn from(error: LambertWError) -> Self {
        Self::LambertW(error)
    }
}

impl std::fmt::Display for DiodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "diode parameters must be finite"),
            Self::NonPositiveTemperature => write!(f, "temperature must be positive"),
            Self::NonPositiveCurrent => write!(f, "forward current must be positive"),
            Self::NonPositiveIdeality => write!(f, "ideality factor must be positive"),
            Self::NonPositiveActivationExponent => {
                write!(f, "activation exponent must be positive")
            }
            Self::NegativeSeriesResistance => write!(f, "series resistance cannot be negative"),
            Self::LambertW(error) => write!(f, "Lambert W failed: {error}"),
        }
    }
}

impl std::error::Error for DiodeError {}

/// Log saturation current, `ln I_s`, avoiding underflow of `I_s` itself.
pub fn ln_saturation_current(
    parameters: &DiodeParameters,
    temperature_k: f64,
) -> Result<f64, DiodeError> {
    validate(parameters, temperature_k)?;
    let vt = thermal_voltage(temperature_k);
    Ok(parameters.ln_b + parameters.eta * temperature_k.ln()
        - parameters.eg_ev / (parameters.activation_exponent * vt))
}

/// Saturation current `I_s(T)` in amperes.
pub fn saturation_current(
    parameters: &DiodeParameters,
    temperature_k: f64,
) -> Result<f64, DiodeError> {
    Ok(ln_saturation_current(parameters, temperature_k)?.exp())
}

/// Terminal voltage at a forced forward current, from the full model.
pub fn diode_voltage(
    parameters: &DiodeParameters,
    current_a: f64,
    temperature_k: f64,
) -> Result<f64, DiodeError> {
    if !current_a.is_finite() || current_a <= 0.0 {
        return Err(DiodeError::NonPositiveCurrent);
    }
    let ln_is = ln_saturation_current(parameters, temperature_k)?;
    let ln_ratio = current_a.ln() - ln_is;
    // ln(I/I_s + 1); the direct exp overflows for large forward currents.
    let log_term = if ln_ratio > 50.0 {
        ln_ratio
    } else {
        ln_ratio.exp().ln_1p()
    };
    let vt = thermal_voltage(temperature_k);
    Ok(current_a * parameters.series_resistance_ohm + parameters.ideality * vt * log_term)
}

/// Terminal current at a forced forward voltage, from the full model.
///
/// With series resistance this is the closed form using `W_0`; without it the
/// result reduces to `I_s (exp(V/(n V_T)) - 1)`.
pub fn diode_current(
    parameters: &DiodeParameters,
    voltage_v: f64,
    temperature_k: f64,
) -> Result<f64, DiodeError> {
    if !voltage_v.is_finite() {
        return Err(DiodeError::NonFinite);
    }
    let ln_is = ln_saturation_current(parameters, temperature_k)?;
    let is = ln_is.exp();
    let nvt = parameters.ideality * thermal_voltage(temperature_k);
    if parameters.series_resistance_ohm == 0.0 {
        return Ok(is * (voltage_v / nvt).exp_m1());
    }

    let rs = parameters.series_resistance_ohm;
    let log_argument = (rs * is / nvt).ln() + (voltage_v + is * rs) / nvt;
    if log_argument > 700.0 {
        // exp(log_argument) would overflow. The asymptotic W ~ log(a)-log(log(a))
        // keeps the forward current finite and avoids returning NaN.
        let w = log_argument - log_argument.ln();
        return Ok(w * nvt / rs - is);
    }
    let w = lambert_w0(log_argument.exp())?;
    Ok(w * nvt / rs - is)
}

/// The lab's ideal constant-current equation, in volts.
///
/// `V(T) = (kT/q)[ln(I/B) - eta*ln(T)] + E_g/q`. This is the `n = n_a = 1`,
/// `R_s = 0`, `I >> I_s` limit, not the full terminal relation.
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

/// Two-current voltage difference, `Delta V = n V_T ln(I1/I2) + (I1-I2) R_s`.
///
/// This identity is independent of `E_g`, `B` and the activation exponent.
pub fn two_current_voltage_difference(
    parameters: &DiodeParameters,
    current_1_a: f64,
    current_2_a: f64,
    temperature_k: f64,
) -> Result<f64, DiodeError> {
    if !current_1_a.is_finite() || current_1_a <= 0.0 {
        return Err(DiodeError::NonPositiveCurrent);
    }
    if !current_2_a.is_finite() || current_2_a <= 0.0 {
        return Err(DiodeError::NonPositiveCurrent);
    }
    validate(parameters, temperature_k)?;
    let vt = thermal_voltage(temperature_k);
    Ok(parameters.ideality * vt * (current_1_a / current_2_a).ln()
        + (current_1_a - current_2_a) * parameters.series_resistance_ohm)
}

fn validate(parameters: &DiodeParameters, temperature_k: f64) -> Result<(), DiodeError> {
    if !parameters.ln_b.is_finite()
        || !parameters.eta.is_finite()
        || !parameters.eg_ev.is_finite()
        || !parameters.ideality.is_finite()
        || !parameters.activation_exponent.is_finite()
        || !parameters.series_resistance_ohm.is_finite()
    {
        return Err(DiodeError::NonFinite);
    }
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return Err(DiodeError::NonPositiveTemperature);
    }
    if parameters.ideality <= 0.0 {
        return Err(DiodeError::NonPositiveIdeality);
    }
    if parameters.activation_exponent <= 0.0 {
        return Err(DiodeError::NonPositiveActivationExponent);
    }
    if parameters.series_resistance_ohm < 0.0 {
        return Err(DiodeError::NegativeSeriesResistance);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn ideal_limit_reduces_to_the_lab_equation() {
        let parameters = DiodeParameters::ideal(10.0, 0.0, 1.12);
        let current = 1e-3;
        let temperature = 300.0;
        let full = diode_voltage(&parameters, current, temperature).unwrap();
        let lab = ideal_diode_voltage(current, temperature, 10.0, 0.0, 1.12);
        close(full, lab, 1e-12);
    }

    #[test]
    fn lambert_form_reproduces_the_implicit_relation() {
        let parameters = DiodeParameters::new(-10.0, 3.0, 1.12, 1.2, 1.0, 10.0);
        for voltage in [0.2, 0.4, 0.6, 0.7] {
            let current = diode_current(&parameters, voltage, 300.0).unwrap();
            let recovered = diode_voltage(&parameters, current, 300.0).unwrap();
            assert!(
                (recovered - voltage).abs() < 1e-12,
                "V={voltage}, recovered={recovered}, I={current}"
            );
        }
    }

    #[test]
    fn zero_voltage_gives_zero_current() {
        let parameters = DiodeParameters::new(-10.0, 3.0, 1.12, 1.2, 1.0, 10.0);
        let current = diode_current(&parameters, 0.0, 300.0).unwrap();
        assert!(current.abs() < 1e-18, "I(0)={current}");
    }

    #[test]
    fn current_increases_with_forward_voltage() {
        let parameters = DiodeParameters::new(-10.0, 3.0, 1.12, 1.2, 1.0, 10.0);
        let low = diode_current(&parameters, 0.4, 300.0).unwrap();
        let high = diode_current(&parameters, 0.6, 300.0).unwrap();
        assert!(high > low);
    }

    #[test]
    fn reverse_bias_approaches_minus_saturation_current() {
        let parameters = DiodeParameters::new(-10.0, 3.0, 1.12, 1.2, 1.0, 10.0);
        let is = saturation_current(&parameters, 300.0).unwrap();
        let current = diode_current(&parameters, -1.0, 300.0).unwrap();
        assert!((current + is).abs() < 1e-18);
    }

    #[test]
    fn two_current_identity_holds_on_a_noiseless_fixture() {
        let parameters = DiodeParameters::new(-10.0, 3.0, 1.12, 1.2, 1.0, 10.0);
        let i1 = 1e-3;
        let i2 = 1e-4;
        let v1 = diode_voltage(&parameters, i1, 300.0).unwrap();
        let v2 = diode_voltage(&parameters, i2, 300.0).unwrap();
        let identity = two_current_voltage_difference(&parameters, i1, i2, 300.0).unwrap();
        close(v1 - v2, identity, 1e-10);
    }

    #[test]
    fn rejected_inputs_are_explicit() {
        let parameters = DiodeParameters::ideal(10.0, 0.0, 1.12);
        assert_eq!(
            diode_voltage(&parameters, 0.0, 300.0),
            Err(DiodeError::NonPositiveCurrent)
        );
        assert_eq!(
            diode_voltage(&parameters, 1e-3, 0.0),
            Err(DiodeError::NonPositiveTemperature)
        );
        assert_eq!(
            diode_current(
                &DiodeParameters::new(10.0, 0.0, 1.12, 0.0, 1.0, 0.0),
                0.5,
                300.0
            ),
            Err(DiodeError::NonPositiveIdeality)
        );
    }
}
