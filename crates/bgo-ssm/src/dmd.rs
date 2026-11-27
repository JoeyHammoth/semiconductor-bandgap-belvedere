//! Hankel-DMDc time constants for the synthetic thermal chain.
//!
//! Dynamic mode decomposition with control fits `x_{k+1} = A x_k + B u_k` to a
//! delay-embedded (Hankel) observable. With a scalar observable the Hankel state
//! is `[y_k, y_{k-1}, ..., y_{k-p+1}]`, so `A` is the companion matrix of the
//! autoregressive part and its eigenvalues are the roots of that polynomial.
//! This module fits the regression with the known heater input, then turns the
//! discrete eigenvalues into damping time constants and angular frequencies.
//!
//! The estimate is a diagnostic, not a tuned model: it uses the noisy synthetic
//! observation, a fixed embedding depth of four (matching the three-state
//! chain) and no model-selection search. The identification trace is driven by
//! the configured multi-pulse heater schedule, whose narrow first pulse gives
//! the fast sensor mode its own step edge.

use crate::thermal::{heater_target_k, ThermalConfig};

/// States in the forward chain; that many modes are reported.
const STATES: usize = 3;
/// Embedding depth. The three-state chain needs one companion mode to
/// represent its direct feedthrough, so the Hankel order is `STATES + 1`.
const ORDER: usize = STATES + 1;
/// Modes closer than this in time constant are treated as the same pole.
const DUPLICATE_TOLERANCE: f64 = 0.05;
/// Lags of the heater input kept as control columns.
const CONTROL_LAGS: usize = 3;
const MAX_ITERATIONS: usize = 500;
const CONVERGED: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalModes {
    /// Damping time constants in seconds, slowest first.
    pub time_constants_s: Vec<f64>,
    /// Angular frequency of each mode in rad/s (zero for a real eigenvalue).
    pub frequencies_rad_s: Vec<f64>,
    /// Relative one-step prediction residual of the DMDc fit.
    pub relative_residual: f64,
    /// The chain's own time constants, slowest first.
    pub true_time_constants_s: Vec<f64>,
    /// Largest relative error between a true constant and its nearest fit.
    pub max_relative_error: f64,
    /// Embedding depth that was used.
    pub order: usize,
}

/// Fit a Hankel-DMDc model to the observed sensor series and report its modes.
pub fn estimate_thermal_modes(
    config: &ThermalConfig,
    time_s: &[f64],
    series: &[f64],
) -> Result<ThermalModes, String> {
    fit_modes(config, time_s, series, ORDER, CONTROL_LAGS)
}

fn fit_modes(
    config: &ThermalConfig,
    time_s: &[f64],
    series: &[f64],
    order: usize,
    control_lags: usize,
) -> Result<ThermalModes, String> {
    let count = time_s.len();
    if order == 0 || count <= order.max(control_lags) || series.len() != count {
        return Err("thermal run is too short for a Hankel embedding".into());
    }
    let dt = config.sample_s;
    if !dt.is_finite() || dt <= 0.0 {
        return Err("sample interval must be positive".into());
    }

    let columns = order + control_lags + 1;
    let mut design = Vec::with_capacity(count);
    let mut targets = Vec::with_capacity(count);

    // Work on deviations from the mean: the absolute series sits near ambient,
    // which makes the offset column nearly collinear with the lag columns.
    let offset = series.iter().sum::<f64>() / count as f64;
    let centred = series
        .iter()
        .map(|value| value - offset)
        .collect::<Vec<_>>();

    let start = order.max(control_lags) - 1;
    for index in start..(count - 1) {
        let mut row = Vec::with_capacity(columns);
        for lag in 0..order {
            row.push(centred[index - lag]);
        }
        for lag in 0..control_lags {
            row.push(heater_target(config, index - lag));
        }
        row.push(1.0);
        design.push(row);
        targets.push(centred[index + 1]);
    }

    let solution = solve_least_squares(design, targets)?;
    let roots = polynomial_roots(&solution[..order])?;

    let mut modes = roots
        .iter()
        .filter_map(|root| {
            let magnitude = root.abs();
            if !magnitude.is_finite() || magnitude <= 1.0e-12 || magnitude >= 1.0 {
                return None;
            }
            let tau = -dt / magnitude.ln();
            if !tau.is_finite() || tau <= 0.0 {
                return None;
            }
            Some((tau, root.arg() / dt))
        })
        .collect::<Vec<_>>();
    if modes.is_empty() {
        return Err("no stable DMDc modes were recovered".into());
    }
    modes.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut distinct = Vec::with_capacity(STATES);
    for (tau, frequency) in modes {
        let duplicate = distinct.iter().any(|(existing, _): &(f64, f64)| {
            (tau - existing).abs() / existing < DUPLICATE_TOLERANCE
        });
        if !duplicate && distinct.len() < STATES {
            distinct.push((tau, frequency));
        }
    }

    let time_constants_s = distinct.iter().map(|(tau, _)| *tau).collect::<Vec<_>>();
    let frequencies_rad_s = distinct.iter().map(|(_, frequency)| *frequency).collect();
    let true_time_constants_s = {
        let mut values = vec![
            config.tau_plate_s,
            config.tau_junction_s,
            config.tau_sensor_s,
        ];
        values.sort_by(|left, right| right.total_cmp(left));
        values
    };
    let max_relative_error = true_time_constants_s
        .iter()
        .map(|truth| {
            time_constants_s
                .iter()
                .map(|fit| (fit - truth).abs() / truth)
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0_f64, f64::max);

    Ok(ThermalModes {
        relative_residual: relative_residual(config, order, control_lags, &solution, &centred),
        time_constants_s,
        frequencies_rad_s,
        true_time_constants_s,
        max_relative_error,
        order,
    })
}

/// Heater target at a sample index, in kelvin above ambient.
fn heater_target(config: &ThermalConfig, index: usize) -> f64 {
    heater_target_k(config, index as f64 * config.sample_s)
}

/// One-step prediction residual of the fitted model, relative to the signal.
/// The control terms belong to the fitted model, so they are included here;
/// omitting them would report the residual of the autoregression alone.
fn relative_residual(
    config: &ThermalConfig,
    order: usize,
    control_lags: usize,
    solution: &[f64],
    observed: &[f64],
) -> f64 {
    let mut residual_energy = 0.0;
    let mut signal_energy = 0.0;
    let start = order.max(control_lags) - 1;
    for index in start..(observed.len() - 1) {
        let mut prediction = solution[order + control_lags];
        for lag in 0..order {
            prediction += solution[lag] * observed[index - lag];
        }
        for lag in 0..control_lags {
            prediction += solution[order + lag] * heater_target(config, index - lag);
        }
        let target = observed[index + 1];
        residual_energy += (target - prediction).powi(2);
        signal_energy += target * target;
    }
    if signal_energy <= 0.0 {
        return f64::NAN;
    }
    (residual_energy / signal_energy).sqrt()
}

/// Least squares by Householder QR. The delay embedding is strongly collinear,
/// so normal equations would square an already large condition number.
fn solve_least_squares(
    mut design: Vec<Vec<f64>>,
    mut targets: Vec<f64>,
) -> Result<Vec<f64>, String> {
    let rows = design.len();
    let columns = design.first().map_or(0, Vec::len);
    if columns == 0 || rows < columns || targets.len() != rows {
        return Err("DMDc regression system is empty or underdetermined".into());
    }
    let mut reflectors = Vec::with_capacity(columns);
    for column in 0..columns {
        let norm = (column..rows)
            .map(|row| design[row][column] * design[row][column])
            .sum::<f64>()
            .sqrt();
        if norm < 1.0e-14 {
            return Err("DMDc regression is singular".into());
        }
        let alpha = if design[column][column] > 0.0 {
            -norm
        } else {
            norm
        };
        let mut reflector = vec![0.0; rows];
        reflector[column] = design[column][column] - alpha;
        for (row, value) in reflector.iter_mut().enumerate().skip(column + 1) {
            *value = design[row][column];
        }
        let squared: f64 = reflector[column..].iter().map(|value| value * value).sum();
        if squared < 1.0e-30 {
            return Err("DMDc regression is singular".into());
        }
        for entry in column..columns {
            let dot = (column..rows)
                .map(|row| reflector[row] * design[row][entry])
                .sum::<f64>();
            let factor = 2.0 * dot / squared;
            for row in column..rows {
                design[row][entry] -= factor * reflector[row];
            }
        }
        let dot = (column..rows)
            .map(|row| reflector[row] * targets[row])
            .sum::<f64>();
        let factor = 2.0 * dot / squared;
        for row in column..rows {
            targets[row] -= factor * reflector[row];
        }
        design[column][column] = alpha;
        reflectors.push(reflector);
    }

    let mut solution = vec![0.0; columns];
    for row in (0..columns).rev() {
        let mut value = targets[row];
        for column in (row + 1)..columns {
            value -= design[row][column] * solution[column];
        }
        solution[row] = value / design[row][row];
    }
    Ok(solution)
}

/// Roots of `z^n - ar[0] z^(n-1) - ... - ar[n-1]` by Durand-Kerner iteration.
fn polynomial_roots(ar: &[f64]) -> Result<Vec<Complex>, String> {
    let degree = ar.len();
    if degree == 0 {
        return Err("DMDc companion polynomial is empty".into());
    }
    let mut coefficients = Vec::with_capacity(degree + 1);
    coefficients.push(Complex::new(1.0, 0.0));
    coefficients.extend(ar.iter().map(|value| Complex::new(-value, 0.0)));

    let seed = Complex::new(0.4, 0.9);
    let mut roots = (0..degree)
        .map(|index| seed.powi(index as u32))
        .collect::<Vec<_>>();

    for _ in 0..MAX_ITERATIONS {
        let mut largest_step: f64 = 0.0;
        for index in 0..degree {
            let current = roots[index];
            let mut denominator = Complex::new(1.0, 0.0);
            for (other, value) in roots.iter().enumerate() {
                if other != index {
                    denominator = denominator.mul(current.sub(*value));
                }
            }
            if denominator.abs() < 1.0e-15 {
                continue;
            }
            let step = evaluate(&coefficients, current).div(denominator);
            roots[index] = current.sub(step);
            largest_step = largest_step.max(step.abs());
        }
        if largest_step < CONVERGED {
            break;
        }
    }
    Ok(roots)
}

fn evaluate(coefficients: &[Complex], z: Complex) -> Complex {
    let mut value = Complex::new(0.0, 0.0);
    for coefficient in coefficients {
        value = value.mul(z).add(*coefficient);
    }
    value
}

#[derive(Debug, Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    fn div(self, other: Self) -> Self {
        let denominator = other.re * other.re + other.im * other.im;
        if denominator < 1.0e-30 {
            return Self::new(0.0, 0.0);
        }
        Self::new(
            (self.re * other.re + self.im * other.im) / denominator,
            (self.im * other.re - self.re * other.im) / denominator,
        )
    }

    fn powi(self, exponent: u32) -> Self {
        let mut value = Self::new(1.0, 0.0);
        for _ in 0..exponent {
            value = value.mul(self);
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thermal::{simulate_thermal, ThermalRun};

    #[test]
    fn recovers_the_chain_time_constants() {
        let config = ThermalConfig::default();
        let run = simulate_thermal(&config).expect("thermal trace");
        let modes =
            estimate_thermal_modes(&config, &run.time_s, &run.observed_k).expect("thermal modes");
        assert_eq!(modes.order, ORDER);
        assert_eq!(modes.time_constants_s.len(), STATES);
        assert!(modes
            .time_constants_s
            .iter()
            .all(|tau| tau.is_finite() && *tau > 0.0));
        // The three-pulse schedule conditions the slow and middle poles to a
        // few percent; measurement noise still whitens the fastest pole, so the
        // estimate stays a diagnostic rather than an identification claim.
        assert!(
            modes.max_relative_error < 0.18,
            "modes {:?} vs true {:?}",
            modes.time_constants_s,
            modes.true_time_constants_s
        );
        assert!(
            modes.relative_residual < 0.05,
            "relative residual {}",
            modes.relative_residual
        );
    }

    #[test]
    fn rejects_a_short_run() {
        let config = ThermalConfig::default();
        let run = simulate_thermal(&config).expect("thermal trace");
        let short = ThermalRun {
            time_s: run.time_s[..2].to_vec(),
            plate_k: run.plate_k[..2].to_vec(),
            junction_k: run.junction_k[..2].to_vec(),
            sensor_k: run.sensor_k[..2].to_vec(),
            observed_k: run.observed_k[..2].to_vec(),
            ..run
        };
        assert!(estimate_thermal_modes(&config, &short.time_s, &short.observed_k).is_err());
    }
}
