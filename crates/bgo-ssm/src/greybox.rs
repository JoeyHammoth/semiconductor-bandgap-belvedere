//! Grey-box (output-error) identification of the thermal chain.
//!
//! The Hankel-DMDc estimator in [`crate::dmd`] is blind: it fits an ARX model to
//! the delay-embedded observation and never assumes the physical structure. That
//! makes it a fair diagnostic, and it also means the measurement noise enters both
//! sides of its regression and biases the fastest pole.
//!
//! This module fits the same chain as a grey-box model instead. The three
//! first-order time constants and the initial temperature are the free parameters,
//! the heater schedule and ambient temperature are known inputs, and the residual is
//! the difference between the observed series and the sensor output simulated from
//! those parameters. Simulating the model for every candidate avoids the
//! errors-in-variables bias, at the cost of assuming the chain structure is correct
//! — which is why the two estimates are reported side by side rather than one
//! replacing the other.
//!
//! The fit is Levenberg–Marquardt on the log time constants and the initial
//! temperature, with a numerical Jacobian. A log parameterisation keeps every
//! constant positive and makes them comparable in scale. Each constant is reported
//! with a 5–95% band from a residual bootstrap: the fitted output-error residuals
//! are resampled with replacement, added back to the fitted response, and the fit is
//! repeated. That band is conditional on the chain structure, like the estimate
//! itself. The initial temperature is estimated rather than pinned to the first
//! sample, because a noisy first sample otherwise biases the slowest pole by about
//! the width of its own band.

use crate::rng::Rng;
use crate::thermal::{heater_target_k, ThermalConfig};

/// Free parameters: three time constants and the initial temperature.
const PARAMETERS: usize = 4;
/// Iteration cap. The fit converges in a handful of steps from the fixed start.
const MAX_ITERATIONS: usize = 60;
/// Step in log-tau used for the numerical Jacobian.
const JACOBIAN_STEP: f64 = 1.0e-4;
/// Step in kelvin used for the initial-temperature derivative.
const TEMPERATURE_STEP: f64 = 1.0e-3;
/// Stop when the parameter step falls below this.
const STEP_TOLERANCE: f64 = 1.0e-9;
/// Log-tau bounds, wide enough to contain any physical chain and no wider.
const LOG_TAU_MIN: f64 = -1.0;
const LOG_TAU_MAX: f64 = 10.0;
/// The initial temperature stays within this many kelvin of ambient.
const INITIAL_BOUND_K: f64 = 2.0;
/// Residual-bootstrap replicates.
const BOOTSTRAP_REPLICATES: usize = 64;

/// Start point for the time constants, chosen independently of the configured
/// constants so the fit is an estimate rather than a refinement of the answer it is
/// compared against. The initial temperature starts at the first observation.
fn start(initial_k: f64) -> [f64; PARAMETERS] {
    [300.0_f64.ln(), 100.0_f64.ln(), 50.0_f64.ln(), initial_k]
}

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
    /// Fitted initial temperature, in kelvin.
    pub initial_k: f64,
    /// Levenberg–Marquardt iterations used.
    pub iterations: usize,
    /// 5th and 95th percentile of a residual bootstrap, per reported mode. `NaN`
    /// when too few replicates converged.
    pub time_constant_low_s: Vec<f64>,
    pub time_constant_high_s: Vec<f64>,
    /// Replicates that converged.
    pub bootstrap_replicates: usize,
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

    let core = fit_from(config, time_s, series, start(series[0]))?;
    let (time_constant_low_s, time_constant_high_s, bootstrap_replicates) =
        bootstrap_bands(config, time_s, &core);

    let mut time_constants_s = core.theta[..3]
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
    let output_rms_k = (core.cost / core.residuals.len() as f64).sqrt();

    Ok(GreyBoxFit {
        time_constants_s,
        true_time_constants_s,
        max_relative_error,
        output_rms_k,
        initial_k: core.theta[3],
        iterations: core.iterations,
        time_constant_low_s,
        time_constant_high_s,
        bootstrap_replicates,
    })
}

/// A converged output-error fit and the pieces the bootstrap needs.
struct FitCore {
    theta: [f64; PARAMETERS],
    residuals: Vec<f64>,
    cost: f64,
    iterations: usize,
}

/// Levenberg–Marquardt from `start`.
fn fit_from(
    config: &ThermalConfig,
    time_s: &[f64],
    series: &[f64],
    start: [f64; PARAMETERS],
) -> Result<FitCore, String> {
    let mut theta = start;
    let mut residuals = residuals_at(config, time_s, series, theta);
    let mut cost = squared_sum(&residuals);
    let mut damping = 1.0e-2;
    let mut iterations = 0;

    while iterations < MAX_ITERATIONS {
        iterations += 1;
        let jacobian = jacobian_at(config, time_s, series, theta);
        let normal = normal_equations(&jacobian, &residuals);
        let Some(step) = solve_damped(&normal, damping) else {
            break;
        };
        let candidate = clamp_parameters(
            [
                theta[0] + step[0],
                theta[1] + step[1],
                theta[2] + step[2],
                theta[3] + step[3],
            ],
            config.ambient_k,
        );
        let candidate_residuals = residuals_at(config, time_s, series, candidate);
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

    if !cost.is_finite() {
        return Err("grey-box fit did not produce a finite output error".into());
    }
    Ok(FitCore {
        theta,
        residuals,
        cost,
        iterations,
    })
}

fn clamp_parameters(theta: [f64; PARAMETERS], ambient_k: f64) -> [f64; PARAMETERS] {
    [
        theta[0].clamp(LOG_TAU_MIN, LOG_TAU_MAX),
        theta[1].clamp(LOG_TAU_MIN, LOG_TAU_MAX),
        theta[2].clamp(LOG_TAU_MIN, LOG_TAU_MAX),
        theta[3].clamp(ambient_k - INITIAL_BOUND_K, ambient_k + INITIAL_BOUND_K),
    ]
}

/// Residual bootstrap of the output-error fit. The residuals are resampled with
/// replacement and added to the fitted response, which keeps their empirical shape
/// without assuming a parametric noise model; each replicate is refitted from the
/// point estimate.
fn bootstrap_bands(
    config: &ThermalConfig,
    time_s: &[f64],
    core: &FitCore,
) -> (Vec<f64>, Vec<f64>, usize) {
    let mut point_sorted = core.theta[..3]
        .iter()
        .map(|log_tau| log_tau.exp())
        .collect::<Vec<_>>();
    point_sorted.sort_by(|left, right| right.total_cmp(left));
    if core.residuals.len() < 8 {
        let missing = vec![f64::NAN; 3];
        return (missing.clone(), missing, 0);
    }

    let simulated = simulate_output(config, time_s, core.theta);
    let mut rng = Rng::new(config.seed ^ 0x6f00_d1e5);
    let mut draws = (0..3)
        .map(|_| Vec::with_capacity(BOOTSTRAP_REPLICATES))
        .collect::<Vec<_>>();
    let mut successful = 0;
    for _ in 0..BOOTSTRAP_REPLICATES {
        let replicate = simulated
            .iter()
            .map(|value| {
                let draw = ((rng.uniform() * core.residuals.len() as f64) as usize)
                    .min(core.residuals.len() - 1);
                value + core.residuals[draw]
            })
            .collect::<Vec<_>>();
        let Ok(refit) = fit_from(config, time_s, &replicate, core.theta) else {
            continue;
        };
        successful += 1;
        let mut modes = refit.theta[..3]
            .iter()
            .map(|log_tau| log_tau.exp())
            .collect::<Vec<_>>();
        modes.sort_by(|left, right| right.total_cmp(left));
        for (target, values) in point_sorted.iter().zip(draws.iter_mut()) {
            if let Some(nearest) = modes
                .iter()
                .copied()
                .min_by(|left, right| (left - target).abs().total_cmp(&(right - target).abs()))
            {
                values.push(nearest);
            }
        }
    }

    let low = draws
        .iter()
        .map(|values| percentile(values, 0.05))
        .collect();
    let high = draws
        .iter()
        .map(|values| percentile(values, 0.95))
        .collect();
    (low, high, successful)
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    if values.len() < 8 {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

/// Residual of the observed series against the simulated sensor output.
fn residuals_at(
    config: &ThermalConfig,
    time_s: &[f64],
    series: &[f64],
    theta: [f64; PARAMETERS],
) -> Vec<f64> {
    let simulated = simulate_output(config, time_s, theta);
    series
        .iter()
        .zip(&simulated)
        .map(|(observed, predicted)| observed - predicted)
        .collect()
}

/// Deterministic chain response to the known heater schedule.
fn simulate_output(config: &ThermalConfig, time_s: &[f64], theta: [f64; PARAMETERS]) -> Vec<f64> {
    let dt = config.sample_s;
    let alpha = [
        1.0 - (-dt / theta[0].exp()).exp(),
        1.0 - (-dt / theta[1].exp()).exp(),
        1.0 - (-dt / theta[2].exp()).exp(),
    ];
    let initial = theta[3];
    let mut plate = initial;
    let mut junction = initial;
    let mut sensor = initial;
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

/// Residual sensitivity to each parameter, by central differences.
fn jacobian_at(
    config: &ThermalConfig,
    time_s: &[f64],
    series: &[f64],
    theta: [f64; PARAMETERS],
) -> Vec<[f64; PARAMETERS]> {
    let mut columns = vec![vec![0.0; series.len()]; PARAMETERS];
    for (parameter, column) in columns.iter_mut().enumerate() {
        let step = if parameter < 3 {
            JACOBIAN_STEP
        } else {
            TEMPERATURE_STEP
        };
        let mut forward = theta;
        forward[parameter] += step;
        let mut backward = theta;
        backward[parameter] -= step;
        let plus = simulate_output(config, time_s, forward);
        let minus = simulate_output(config, time_s, backward);
        for ((value, plus_value), minus_value) in column.iter_mut().zip(&plus).zip(&minus) {
            *value = -(plus_value - minus_value) / (2.0 * step);
        }
    }
    (0..series.len())
        .map(|row| std::array::from_fn(|parameter| columns[parameter][row]))
        .collect()
}

fn normal_equations(
    jacobian: &[[f64; PARAMETERS]],
    residuals: &[f64],
) -> [[f64; PARAMETERS + 1]; PARAMETERS] {
    let mut augmented = [[0.0; PARAMETERS + 1]; PARAMETERS];
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
fn solve_damped(
    normal: &[[f64; PARAMETERS + 1]; PARAMETERS],
    damping: f64,
) -> Option<[f64; PARAMETERS]> {
    let mut matrix = [[0.0; PARAMETERS + 1]; PARAMETERS];
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
    let mut step = [0.0; PARAMETERS];
    for (row, value) in step.iter_mut().enumerate() {
        *value = matrix[row][PARAMETERS];
    }
    Some(step)
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
            "grey-box taus {:?} max error {:.4} rms {:.5} initial {:.4} iterations {}",
            fit.time_constants_s,
            fit.max_relative_error,
            fit.output_rms_k,
            fit.initial_k,
            fit.iterations
        );
        assert!(fit
            .time_constants_s
            .iter()
            .all(|tau| tau.is_finite() && *tau > 0.0));
        assert!(
            fit.max_relative_error < 0.02,
            "taus {:?} vs true {:?}",
            fit.time_constants_s,
            fit.true_time_constants_s
        );
        assert!(fit.output_rms_k < config.sensor_noise_k * 2.0);
        assert!(
            (fit.initial_k - config.initial_k).abs() < 0.2,
            "initial {:.3} K",
            fit.initial_k
        );
    }

    #[test]
    fn bootstrap_band_brackets_the_point_estimate_and_the_truth() {
        let config = ThermalConfig::default();
        let run = simulate_thermal(&config).expect("thermal trace");
        let fit = fit_output_error(&config, &run.time_s, &run.observed_k).expect("grey-box fit");
        assert!(
            fit.bootstrap_replicates >= BOOTSTRAP_REPLICATES / 2,
            "only {} replicates converged",
            fit.bootstrap_replicates
        );
        for (index, ((low, high), tau)) in fit
            .time_constant_low_s
            .iter()
            .zip(&fit.time_constant_high_s)
            .zip(&fit.time_constants_s)
            .enumerate()
        {
            println!(
                "mode {index}: fitted {tau:.1} s band {low:.1}–{high:.1} s truth {:.1}",
                fit.true_time_constants_s[index]
            );
            assert!(low.is_finite() && high.is_finite());
            assert!(
                low <= tau && tau <= high,
                "interval {low}..{high} excludes {tau}"
            );
            assert!(
                *low <= fit.true_time_constants_s[index]
                    && fit.true_time_constants_s[index] <= *high,
                "interval {low}..{high} excludes the configured {}",
                fit.true_time_constants_s[index]
            );
            assert!(high / low < 1.5, "interval too wide: {low}..{high}");
        }
    }

    #[test]
    fn bands_cover_the_configured_constants_across_seeds() {
        let mut covered = 0;
        let mut total = 0;
        let mut ratios = Vec::new();
        for seed in [1_u64, 7, 42, 0x5eed_7e11, 12_345] {
            let config = ThermalConfig {
                seed,
                ..ThermalConfig::default()
            };
            let run = simulate_thermal(&config).expect("thermal trace");
            let fit = fit_output_error(&config, &run.time_s, &run.observed_k).expect("grey-box");
            for (index, (low, high)) in fit
                .time_constant_low_s
                .iter()
                .zip(&fit.time_constant_high_s)
                .enumerate()
            {
                let truth = fit.true_time_constants_s[index];
                total += 1;
                if *low <= truth && truth <= *high {
                    covered += 1;
                } else {
                    println!(
                        "seed {seed} mode {index}: truth {truth:.1} s outside {low:.1}–{high:.1} s"
                    );
                }
                // Width relative to the point estimate: a runaway band would show
                // up here before it reached the page.
                ratios.push((high - low) / fit.time_constants_s[index]);
            }
        }
        let widest = ratios.iter().fold(0.0_f64, |a, b| a.max(*b));
        println!("covered {covered}/{total}, widest relative band {widest:.3}");
        // A 90% interval on 15 cases can legitimately miss one or two; the
        // threshold rejects a band that is systematically too tight.
        assert!(
            covered * 10 >= total * 8,
            "bands covered {covered} of {total} configured constants"
        );
        assert!(widest < 0.6, "widest relative band {widest}");
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
