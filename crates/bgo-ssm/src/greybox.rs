//! Grey-box (output-error) identification of the thermal chain.
//!
//! The Hankel-DMDc estimator in [`crate::dmd`] is blind: it fits an ARX model to
//! the delay-embedded observation and never assumes the physical structure. That
//! makes it a fair diagnostic, and it also means the measurement noise enters both
//! sides of its regression and biases the fastest pole.
//!
//! This module fits the same chain as a grey-box model instead. The three
//! first-order time constants are the only free parameters, the heater schedule and
//! ambient temperature are known inputs, and the residual is the difference between
//! the observed series and the sensor output simulated from the initial observation.
//! Simulating the model for every candidate avoids the errors-in-variables bias, at
//! the cost of assuming the chain structure is correct — which is why the two
//! estimates are reported side by side rather than one replacing the other.
//!
//! The fit is Levenberg–Marquardt on the log time constants, with a numerical
//! Jacobian. A log parameterisation keeps every candidate positive and makes the
//! three parameters comparable in scale.

use crate::thermal::{heater_target_k, ThermalConfig};

/// Free parameters: the three time constants.
const PARAMETERS: usize = 3;
/// Iteration cap. The fit converges in a handful of steps from the fixed start.
const MAX_ITERATIONS: usize = 60;
/// Step in log-tau used for the numerical Jacobian.
const JACOBIAN_STEP: f64 = 1.0e-4;
/// Stop when the log-tau step falls below this.
const STEP_TOLERANCE: f64 = 1.0e-9;
/// Log-tau bounds, wide enough to contain any physical chain and no wider.
const LOG_TAU_MIN: f64 = -1.0;
const LOG_TAU_MAX: f64 = 10.0;

#[derive(Debug, Clone, PartialEq)]
pub struct GreyBoxFit {
    /// Fitted damping time constants in seconds, slowest first.
    pub time_constants_s: Vec<f64>,
    /// The chain's own time constants, slowest first.
    pub true_time_constants_s: Vec<f64>,
    /// Largest relative error between a true constant and its nearest fit.
    pub max_relative_error: f64,
    /// RMS of the output error over the trace, in kelvin.
    pub output_rms_k: f64,
    /// Levenberg–Marquardt iterations used.
    pub iterations: usize,
}

/// Fit the three time constants by output error on the observed series.
pub fn fit_output_error(
    config: &ThermalConfig,
    time_s: &[f64],
    series: &[f64],
) -> Result<GreyBoxFit, String> {
    if time_s.len() != series.len() || time_s.len() < PARAMETERS + 2 {
        return Err("thermal run is inconsistent or too short for a grey-box fit".into());
    }
    if !config.sample_s.is_finite() || config.sample_s <= 0.0 {
        return Err("sample interval must be positive".into());
    }
    if !series.iter().all(|value| value.is_finite()) {
        return Err("observed series must be finite".into());
    }

    // Start independently of the configured constants, so the fit is an estimate
    // rather than a refinement of the answer it is compared against.
    let mut theta = [300.0_f64.ln(), 100.0_f64.ln(), 50.0_f64.ln()];
    let initial_k = series[0];
    let mut residuals = residuals_at(config, time_s, series, initial_k, theta);
    let mut cost = squared_sum(&residuals);
    let mut damping = 1.0e-2;
    let mut iterations = 0;

    while iterations < MAX_ITERATIONS {
        iterations += 1;
        let jacobian = jacobian_at(config, time_s, series, initial_k, theta);
        let normal = normal_equations(&jacobian, &residuals);
        let Some(step) = solve_damped(&normal, damping) else {
            break;
        };
        let candidate = [
            (theta[0] + step[0]).clamp(LOG_TAU_MIN, LOG_TAU_MAX),
            (theta[1] + step[1]).clamp(LOG_TAU_MIN, LOG_TAU_MAX),
            (theta[2] + step[2]).clamp(LOG_TAU_MIN, LOG_TAU_MAX),
        ];
        let candidate_residuals = residuals_at(config, time_s, series, initial_k, candidate);
        let candidate_cost = squared_sum(&candidate_residuals);
        if candidate_cost.is_finite() && candidate_cost < cost {
            theta = candidate;
            cost = candidate_cost;
            residuals = candidate_residuals;
            damping = (damping / 3.0).max(1.0e-12);
            if step.iter().map(|value| value * value).sum::<f64>().sqrt() < STEP_TOLERANCE {
                break;
            }
        } else {
            damping *= 5.0;
            if damping > 1.0e12 {
                break;
            }
        }
    }

    let mut time_constants_s = theta
        .iter()
        .map(|log_tau| log_tau.exp())
        .collect::<Vec<_>>();
    time_constants_s.sort_by(|left, right| right.total_cmp(left));
    let mut true_time_constants_s = vec![
        config.tau_plate_s,
        config.tau_junction_s,
        config.tau_sensor_s,
    ];
    true_time_constants_s.sort_by(|left, right| right.total_cmp(left));
    let max_relative_error = true_time_constants_s
        .iter()
        .map(|truth| {
            time_constants_s
                .iter()
                .map(|fit| (fit - truth).abs() / truth)
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0_f64, f64::max);
    let output_rms_k = (cost / residuals.len() as f64).sqrt();

    Ok(GreyBoxFit {
        time_constants_s,
        true_time_constants_s,
        max_relative_error,
        output_rms_k,
        iterations,
    })
}

/// Residual of the observed series against the simulated sensor output.
fn residuals_at(
    config: &ThermalConfig,
    time_s: &[f64],
    series: &[f64],
    initial_k: f64,
    theta: [f64; 3],
) -> Vec<f64> {
    let simulated = simulate_output(config, time_s, initial_k, theta);
    series
        .iter()
        .zip(&simulated)
        .map(|(observed, predicted)| observed - predicted)
        .collect()
}

/// Deterministic chain response to the known heater schedule.
fn simulate_output(
    config: &ThermalConfig,
    time_s: &[f64],
    initial_k: f64,
    theta: [f64; 3],
) -> Vec<f64> {
    let dt = config.sample_s;
    let alpha = theta.map(|log_tau| 1.0 - (-dt / log_tau.exp()).exp());
    let mut plate = initial_k;
    let mut junction = initial_k;
    let mut sensor = initial_k;
    let mut output = Vec::with_capacity(time_s.len());
    for (index, time) in time_s.iter().enumerate() {
        if index > 0 {
            let target = config.ambient_k + heater_target_k(config, *time);
            plate += (target - plate) * alpha[0];
            junction += (plate - junction) * alpha[1];
            sensor += (junction - sensor) * alpha[2];
        }
        output.push(sensor);
    }
    output
}

/// Residual sensitivity to each log time constant, by central differences.
fn jacobian_at(
    config: &ThermalConfig,
    time_s: &[f64],
    series: &[f64],
    initial_k: f64,
    theta: [f64; 3],
) -> Vec<[f64; PARAMETERS]> {
    let mut rows = Vec::with_capacity(series.len());
    let mut columns = vec![vec![0.0; series.len()]; PARAMETERS];
    for (parameter, column) in columns.iter_mut().enumerate() {
        let mut forward = theta;
        forward[parameter] += JACOBIAN_STEP;
        let mut backward = theta;
        backward[parameter] -= JACOBIAN_STEP;
        let plus = simulate_output(config, time_s, initial_k, forward);
        let minus = simulate_output(config, time_s, initial_k, backward);
        for row in 0..series.len() {
            column[row] = -(plus[row] - minus[row]) / (2.0 * JACOBIAN_STEP);
        }
    }
    for row in 0..series.len() {
        rows.push([columns[0][row], columns[1][row], columns[2][row]]);
    }
    rows
}

fn normal_equations(jacobian: &[[f64; PARAMETERS]], residuals: &[f64]) -> [[f64; 4]; PARAMETERS] {
    let mut augmented = [[0.0; 4]; PARAMETERS];
    for (row, jacobian_row) in jacobian.iter().enumerate() {
        for left in 0..PARAMETERS {
            augmented[left][PARAMETERS] += jacobian_row[left] * residuals[row];
            for right in 0..PARAMETERS {
                augmented[left][right] += jacobian_row[left] * jacobian_row[right];
            }
        }
    }
    augmented
}

/// Solve `(A + λ·diag(A)) δ = -g` by Gaussian elimination with partial pivoting.
fn solve_damped(normal: &[[f64; 4]; PARAMETERS], damping: f64) -> Option<[f64; PARAMETERS]> {
    let mut matrix = [[0.0; 4]; PARAMETERS];
    for row in 0..PARAMETERS {
        for column in 0..PARAMETERS {
            matrix[row][column] = normal[row][column];
        }
        matrix[row][row] += damping * normal[row][row].max(1.0e-12);
        matrix[row][PARAMETERS] = -normal[row][PARAMETERS];
    }
    for column in 0..PARAMETERS {
        let pivot = (column..PARAMETERS).max_by(|left, right| {
            matrix[*left][column]
                .abs()
                .total_cmp(&matrix[*right][column].abs())
        })?;
        if matrix[pivot][column].abs() < 1.0e-14 {
            return None;
        }
        matrix.swap(column, pivot);
        let divisor = matrix[column][column];
        for entry in matrix[column].iter_mut().skip(column) {
            *entry /= divisor;
        }
        for row in 0..PARAMETERS {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for entry in column..=PARAMETERS {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
        }
    }
    Some([
        matrix[0][PARAMETERS],
        matrix[1][PARAMETERS],
        matrix[2][PARAMETERS],
    ])
}

fn squared_sum(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dmd::estimate_thermal_modes;
    use crate::thermal::simulate_thermal;

    #[test]
    fn recovers_the_chain_time_constants() {
        let config = ThermalConfig::default();
        let run = simulate_thermal(&config).expect("thermal trace");
        let fit = fit_output_error(&config, &run.time_s, &run.observed_k).expect("grey-box fit");
        println!(
            "grey-box taus {:?} max error {:.4} rms {:.5} iterations {}",
            fit.time_constants_s, fit.max_relative_error, fit.output_rms_k, fit.iterations
        );
        assert!(fit
            .time_constants_s
            .iter()
            .all(|tau| tau.is_finite() && *tau > 0.0));
        assert!(
            fit.max_relative_error < 0.05,
            "taus {:?} vs true {:?}",
            fit.time_constants_s,
            fit.true_time_constants_s
        );
        assert!(fit.output_rms_k < config.sensor_noise_k * 2.0);
    }

    #[test]
    fn fastest_pole_beats_the_blind_estimate_across_seeds() {
        let mut greybox = Vec::new();
        let mut blind = Vec::new();
        for seed in [1_u64, 7, 42, 0x5eed_7e11, 12_345] {
            let config = ThermalConfig {
                seed,
                ..ThermalConfig::default()
            };
            let run = simulate_thermal(&config).expect("thermal trace");
            let fit = fit_output_error(&config, &run.time_s, &run.observed_k).expect("grey-box");
            let modes = estimate_thermal_modes(&config, &run.time_s, &run.observed_k)
                .expect("thermal modes");
            let greybox_error = fastest_error(&fit.time_constants_s, config.tau_sensor_s);
            let blind_error = fastest_error(&modes.time_constants_s, config.tau_sensor_s);
            println!(
                "seed {seed}: grey-box {:.2} s ({:.1}%) blind {:.2} s ({:.1}%)",
                fit.time_constants_s.last().copied().unwrap_or(f64::NAN),
                greybox_error * 100.0,
                modes.time_constants_s.last().copied().unwrap_or(f64::NAN),
                blind_error * 100.0,
            );
            greybox.push(greybox_error);
            blind.push(blind_error);
        }
        let greybox_worst = greybox.iter().fold(0.0_f64, |a, b| a.max(*b));
        let blind_worst = blind.iter().fold(0.0_f64, |a, b| a.max(*b));
        println!("worst grey-box {greybox_worst:.3} blind {blind_worst:.3}");
        assert!(
            greybox_worst < 0.05,
            "grey-box fastest-pole errors {greybox:?}"
        );
        assert!(
            blind_worst > greybox_worst,
            "blind {blind_worst} vs grey-box {greybox_worst}"
        );
    }

    #[test]
    fn rejects_an_inconsistent_run() {
        let config = ThermalConfig::default();
        let run = simulate_thermal(&config).expect("thermal trace");
        assert!(fit_output_error(&config, &run.time_s, &run.observed_k[..2]).is_err());
    }

    fn fastest_error(fitted: &[f64], truth: f64) -> f64 {
        fitted
            .iter()
            .map(|tau| (tau - truth).abs() / truth)
            .fold(f64::INFINITY, f64::min)
    }
}
