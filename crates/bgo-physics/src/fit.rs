//! Least-squares regression for the lab's M0 model.
//!
//! M0 fits a straight line `V(T) = a + b*T` to the recorded forward voltage.
//! The fitted intercept is **not** `E_g(0)` and must never be labelled as such;
//! see `docs/physics-audit.md` sections 3 and 4. This module only reports what
//! the line estimates. It deliberately does not know about currents, ideality
//! or the source dataset.

use std::fmt;

/// One recorded point in the M0 regression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    /// Junction temperature in kelvin.
    pub temperature_k: f64,
    /// Forward voltage in volts.
    pub voltage_v: f64,
}

impl Observation {
    pub fn new(temperature_k: f64, voltage_v: f64) -> Self {
        Self {
            temperature_k,
            voltage_v,
        }
    }
}

/// Ordinary least-squares fit of `V` against `T`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearFit {
    /// Fitted intercept in volts. Numerically equal to the eV value only when
    /// quoted as `intercept_v * q`; it is not a band gap at any temperature.
    pub intercept_v: f64,
    /// Fitted slope in volts per kelvin.
    pub slope_v_per_k: f64,
    /// Standard error of the intercept from the residual variance.
    pub intercept_standard_error_v: f64,
    /// Standard error of the slope from the residual variance.
    pub slope_standard_error_v: f64,
    /// Residual standard error in volts (`n - 2` degrees of freedom).
    pub residual_standard_error_v: f64,
    /// Coefficient of determination.
    pub r_squared: f64,
    /// Number of observations used.
    pub n: usize,
}

/// Why a regression could not be performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitError {
    /// Fewer than three points: the residual variance would be meaningless.
    TooFewPoints,
    /// A temperature or voltage was NaN or infinite.
    NonFinite,
    /// Every temperature is the same, so the slope is undefined.
    ZeroTemperatureSpread,
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewPoints => write!(f, "need at least three observations"),
            Self::NonFinite => write!(f, "observations must be finite"),
            Self::ZeroTemperatureSpread => write!(f, "temperatures have no spread"),
        }
    }
}

impl std::error::Error for FitError {}

/// Fit M0 by ordinary least squares.
///
/// The standard errors are the usual `s^2 (X'X)^-1` values and assume
/// independent, constant-variance residuals. They are not the report's
/// max–min trend-line spread; that requires per-point uncertainties which the
/// recordings do not yet carry.
pub fn fit_m0(observations: &[Observation]) -> Result<LinearFit, FitError> {
    if observations.len() < 3 {
        return Err(FitError::TooFewPoints);
    }
    if observations
        .iter()
        .any(|point| !point.temperature_k.is_finite() || !point.voltage_v.is_finite())
    {
        return Err(FitError::NonFinite);
    }

    let n = observations.len();
    let count = n as f64;
    let mean_temperature = observations
        .iter()
        .map(|point| point.temperature_k)
        .sum::<f64>()
        / count;
    let mean_voltage = observations
        .iter()
        .map(|point| point.voltage_v)
        .sum::<f64>()
        / count;

    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut sst = 0.0;
    for point in observations {
        let dt = point.temperature_k - mean_temperature;
        let dv = point.voltage_v - mean_voltage;
        sxx += dt * dt;
        sxy += dt * dv;
        sst += dv * dv;
    }

    if sxx <= f64::EPSILON {
        return Err(FitError::ZeroTemperatureSpread);
    }

    let slope = sxy / sxx;
    let intercept = mean_voltage - slope * mean_temperature;

    let sse = observations
        .iter()
        .map(|point| {
            let residual = point.voltage_v - (intercept + slope * point.temperature_k);
            residual * residual
        })
        .sum::<f64>();

    let residual_standard_error = (sse / (count - 2.0)).sqrt();
    let intercept_standard_error =
        residual_standard_error * (1.0 / count + mean_temperature.powi(2) / sxx).sqrt();
    let slope_standard_error = residual_standard_error / sxx.sqrt();
    let r_squared = if sst <= f64::EPSILON {
        1.0
    } else {
        1.0 - sse / sst
    };

    Ok(LinearFit {
        intercept_v: intercept,
        slope_v_per_k: slope,
        intercept_standard_error_v: intercept_standard_error,
        slope_standard_error_v: slope_standard_error,
        residual_standard_error_v: residual_standard_error,
        r_squared,
        n,
    })
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
    fn recovers_an_exact_line() {
        let points = [
            Observation::new(280.0, 0.7 - 0.002 * 280.0),
            Observation::new(320.0, 0.7 - 0.002 * 320.0),
            Observation::new(360.0, 0.7 - 0.002 * 360.0),
        ];
        let fit = fit_m0(&points).unwrap();
        close(fit.intercept_v, 0.7, 1e-12);
        close(fit.slope_v_per_k, -0.002, 1e-12);
        close(fit.intercept_standard_error_v, 0.0, 1e-12);
        close(fit.r_squared, 1.0, 1e-12);
        assert_eq!(fit.n, 3);
    }

    #[test]
    fn matches_a_hand_computed_regression() {
        // t = 1,2,3; v = 1,3,2 gives means 2,2, slope 0.5, intercept 1.
        let points = [
            Observation::new(1.0, 1.0),
            Observation::new(2.0, 3.0),
            Observation::new(3.0, 2.0),
        ];
        let fit = fit_m0(&points).unwrap();
        close(fit.intercept_v, 1.0, 1e-12);
        close(fit.slope_v_per_k, 0.5, 1e-12);
        close(fit.residual_standard_error_v, 1.224_744_871_391_589, 1e-12);
        close(
            fit.intercept_standard_error_v,
            1.870_828_693_386_970_7,
            1e-12,
        );
        close(fit.slope_standard_error_v, 0.866_025_403_784_438_6, 1e-12);
        close(fit.r_squared, 0.25, 1e-12);
    }

    #[test]
    fn rejects_unusable_inputs() {
        assert_eq!(
            fit_m0(&[Observation::new(300.0, 0.6), Observation::new(310.0, 0.59)]),
            Err(FitError::TooFewPoints)
        );
        assert_eq!(
            fit_m0(&[
                Observation::new(300.0, 0.6),
                Observation::new(310.0, f64::NAN),
                Observation::new(320.0, 0.58),
            ]),
            Err(FitError::NonFinite)
        );
        assert_eq!(
            fit_m0(&[
                Observation::new(300.0, 0.6),
                Observation::new(300.0, 0.59),
                Observation::new(300.0, 0.58),
            ]),
            Err(FitError::ZeroTemperatureSpread)
        );
    }
}
