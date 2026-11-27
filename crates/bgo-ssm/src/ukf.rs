//! Unscented Kalman filter, Rauch-Tung-Striebel smoother and a small GP
//! discrepancy fit for the synthetic thermal chain.
//!
//! The state is `[plate, junction, sensor]` in kelvin. The transition is the
//! exact first-order chain used by the forward twin, with the known heater
//! control. The measurement is the noisy sensor temperature. The smoother is
//! causal in the filter pass and uses all observations only in the RTS pass.
//!
//! The GP is deliberately a diagnostic, not a latent state: it is fitted to
//! the post-smoother sensor residual with fixed RBF hyperparameters. That
//! keeps the experiment transparent and avoids pretending the residual has a
//! calibrated physical prior.

use crate::matern::matern_discrepancy;
use crate::thermal::{heater_target_k, simulate_thermal, ThermalConfig, ThermalRun};

type Vec3 = [f64; 3];
type Mat3 = [[f64; 3]; 3];

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalEstimate {
    pub filtered_junction_k: Vec<f64>,
    pub smoothed_junction_k: Vec<f64>,
    pub smoothed_junction_sd_k: Vec<f64>,
    pub smoothed_sensor_k: Vec<f64>,
    pub gp_mean_k: Vec<f64>,
    pub gp_sd_k: Vec<f64>,
    pub junction_rmse_k: f64,
    pub sensor_rmse_k: f64,
    pub gp_rms_k: f64,
    /// Post-smoother sensor residual under a Matérn-3/2 latent state: the
    /// smoothed discrepancy mean and its standard deviation.
    pub matern_mean_k: Vec<f64>,
    pub matern_sd_k: Vec<f64>,
    pub matern_rms_k: f64,
    /// Gaussian log-likelihood of the observations under the filter's own
    /// predictive distribution (the Kalman prediction-error decomposition).
    pub log_likelihood: f64,
}

#[derive(Debug, Clone, Copy)]
struct Dynamics {
    config: ThermalConfig,
    plate_alpha: f64,
    junction_alpha: f64,
    sensor_alpha: f64,
    sample_s: f64,
}

impl Dynamics {
    fn new(config: &ThermalConfig) -> Self {
        let dt = config.sample_s;
        Self {
            config: *config,
            plate_alpha: 1.0 - (-dt / config.tau_plate_s).exp(),
            junction_alpha: 1.0 - (-dt / config.tau_junction_s).exp(),
            sensor_alpha: 1.0 - (-dt / config.tau_sensor_s).exp(),
            sample_s: dt,
        }
    }

    fn step(&self, state: Vec3, index: usize) -> Vec3 {
        let time = index as f64 * self.sample_s;
        let target = self.config.ambient_k + heater_target_k(&self.config, time);
        let plate = state[0] + (target - state[0]) * self.plate_alpha;
        let junction = state[1] + (plate - state[1]) * self.junction_alpha;
        let sensor = state[2] + (junction - state[2]) * self.sensor_alpha;
        [plate, junction, sensor]
    }
}

/// Filter and smooth a synthetic thermal run. The run is supplied separately
/// so callers that already generated the trace do not pay for it twice.
pub fn smooth_thermal(config: &ThermalConfig, run: &ThermalRun) -> Result<ThermalEstimate, String> {
    let count = run.time_s.len();
    if count == 0 || run.observed_k.len() != count {
        return Err("thermal run is empty or inconsistent".into());
    }
    let dynamics = Dynamics::new(config);
    let observed = &run.observed_k;
    let measurement_variance = config.sensor_noise_k.max(1.0e-3).powi(2);
    let process_variance = 1.0e-4;
    let initial_covariance = mat_diag([1.0e-4, 1.0e-4, 1.0e-4]);

    let mut filtered = Vec::with_capacity(count);
    let mut filtered_covariance = Vec::with_capacity(count);
    let mut predicted = Vec::with_capacity(count);
    let mut predicted_covariance = Vec::with_capacity(count);
    let mut cross_covariance = vec![mat_zero(); count];
    let mut marginal_log_likelihood = 0.0;

    for index in 0..count {
        let (mean, covariance) = if index == 0 {
            (
                [config.initial_k, config.initial_k, config.initial_k],
                initial_covariance,
            )
        } else {
            let (mean, covariance, cross) = predict(
                filtered[index - 1],
                filtered_covariance[index - 1],
                index,
                &dynamics,
                process_variance,
            )?;
            cross_covariance[index - 1] = cross;
            (mean, covariance)
        };
        let (updated, updated_covariance, log_likelihood) =
            update(mean, covariance, observed[index], measurement_variance)?;
        marginal_log_likelihood += log_likelihood;
        filtered.push(updated);
        filtered_covariance.push(updated_covariance);
        predicted.push(mean);
        predicted_covariance.push(covariance);
    }

    let mut smoothed = filtered.clone();
    let mut smoothed_covariance = filtered_covariance.clone();
    for index in (0..count - 1).rev() {
        let gain = mat_mul(
            cross_covariance[index],
            mat_inverse(predicted_covariance[index + 1])?,
        );
        let difference = sub(smoothed[index + 1], predicted[index + 1]);
        smoothed[index] = add(filtered[index], mat_vec(gain, difference));
        let covariance_difference = mat_sub(
            smoothed_covariance[index + 1],
            predicted_covariance[index + 1],
        );
        smoothed_covariance[index] = mat_add(
            filtered_covariance[index],
            mat_mul(mat_mul(gain, covariance_difference), transpose(gain)),
        );
        symmetrise(&mut smoothed_covariance[index]);
    }

    let filtered_junction_k = filtered.iter().map(|state| state[1]).collect::<Vec<_>>();
    let smoothed_junction_k = smoothed.iter().map(|state| state[1]).collect::<Vec<_>>();
    let smoothed_junction_sd_k = smoothed_covariance
        .iter()
        .map(|covariance| covariance[1][1].max(0.0).sqrt())
        .collect::<Vec<_>>();

    let smoothed_sensor_k = smoothed.iter().map(|state| state[2]).collect::<Vec<_>>();
    let residual = observed
        .iter()
        .zip(&smoothed_sensor_k)
        .map(|(measurement, estimate)| measurement - estimate)
        .collect::<Vec<_>>();
    let (gp_mean_k, gp_sd_k) = gp_residual(
        &run.time_s,
        &residual,
        180.0,
        0.20,
        config.sensor_noise_k.max(1.0e-3),
    )?;

    // The same residual, now modelled as a latent Matérn-3/2 state instead of
    // an RBF kernel: fixed length scale and signal scale, filtered and smoothed
    // exactly, so the discrepancy is a state of the model rather than a fit.
    let matern = matern_discrepancy(
        &run.time_s,
        &residual,
        180.0,
        0.20,
        config.sensor_noise_k.max(1.0e-3),
    )?;

    let junction_rmse_k = rmse(&smoothed_junction_k, &run.junction_k);
    let sensor_rmse_k = rmse(&smoothed_sensor_k, observed);
    let gp_rms_k =
        (gp_mean_k.iter().map(|value| value * value).sum::<f64>() / gp_mean_k.len() as f64).sqrt();
    let matern_rms_k = (matern.mean_k.iter().map(|value| value * value).sum::<f64>()
        / matern.mean_k.len() as f64)
        .sqrt();

    Ok(ThermalEstimate {
        filtered_junction_k,
        smoothed_junction_k,
        smoothed_junction_sd_k,
        smoothed_sensor_k,
        gp_mean_k,
        gp_sd_k,
        junction_rmse_k,
        sensor_rmse_k,
        gp_rms_k,
        matern_mean_k: matern.mean_k,
        matern_sd_k: matern.sd_k,
        matern_rms_k,
        log_likelihood: marginal_log_likelihood,
    })
}

/// Convenience wrapper that generates the synthetic run and smooths it.
pub fn estimate_thermal(config: &ThermalConfig) -> Result<ThermalEstimate, String> {
    let run = simulate_thermal(config)?;
    smooth_thermal(config, &run)
}

fn predict(
    mean: Vec3,
    covariance: Mat3,
    index: usize,
    dynamics: &Dynamics,
    process_variance: f64,
) -> Result<(Vec3, Mat3, Mat3), String> {
    let points = sigma_points(mean, covariance)?;
    let propagated = points
        .iter()
        .map(|point| dynamics.step(*point, index))
        .collect::<Vec<_>>();
    let weights = sigma_weights();
    let predicted_mean = weighted_mean(&propagated, &weights.0);
    let mut predicted_covariance = mat_scale(mat_identity(), process_variance);
    let mut cross = mat_zero();
    for (index, (point, propagated_point)) in points.iter().zip(&propagated).enumerate() {
        let delta = sub(*point, mean);
        let propagated_delta = sub(*propagated_point, predicted_mean);
        predicted_covariance = mat_add(
            predicted_covariance,
            mat_scale(mat_outer(propagated_delta), weights.1[index]),
        );
        cross = mat_add(
            cross,
            mat_scale(mat_outer_pair(delta, propagated_delta), weights.1[index]),
        );
    }
    Ok((predicted_mean, predicted_covariance, cross))
}

/// One unscented update. Returns the posterior mean and covariance together with
/// the log density of the observation under the predicted measurement, which is
/// summed into the Kalman marginal likelihood.
fn update(
    mean: Vec3,
    covariance: Mat3,
    observation: f64,
    measurement_variance: f64,
) -> Result<(Vec3, Mat3, f64), String> {
    let points = sigma_points(mean, covariance)?;
    let weights = sigma_weights();
    let measurements = points.iter().map(|point| point[2]).collect::<Vec<_>>();
    let measurement_mean = weights
        .0
        .iter()
        .zip(&measurements)
        .map(|(weight, value)| weight * value)
        .sum::<f64>();
    let mut measurement_covariance = measurement_variance;
    let mut cross = [0.0; 3];
    for (index, point) in points.iter().enumerate() {
        let measurement_delta = measurements[index] - measurement_mean;
        measurement_covariance += weights.1[index] * measurement_delta * measurement_delta;
        let state_delta = sub(*point, mean);
        for row in 0..3 {
            cross[row] += weights.1[index] * state_delta[row] * measurement_delta;
        }
    }
    if measurement_covariance <= 1.0e-15 {
        return Err("UKF measurement covariance is not positive".into());
    }
    let gain = cross.map(|value| value / measurement_covariance);
    let innovation = observation - measurement_mean;
    let log_likelihood = -0.5
        * ((std::f64::consts::TAU * measurement_covariance).ln()
            + innovation * innovation / measurement_covariance);
    let updated = [
        mean[0] + gain[0] * innovation,
        mean[1] + gain[1] * innovation,
        mean[2] + gain[2] * innovation,
    ];
    let mut updated_covariance = covariance;
    for row in 0..3 {
        for column in 0..3 {
            updated_covariance[row][column] -= gain[row] * cross[column];
        }
    }
    symmetrise(&mut updated_covariance);
    Ok((updated, updated_covariance, log_likelihood))
}

fn sigma_weights() -> ([f64; 7], [f64; 7]) {
    let (alpha, beta, kappa) = (0.5, 2.0, 0.0);
    let n = 3.0;
    let lambda = alpha * alpha * (n + kappa) - n;
    let denominator = 2.0 * (n + lambda);
    let mut mean = [1.0 / denominator; 7];
    let mut covariance = [1.0 / denominator; 7];
    mean[0] = lambda / (n + lambda);
    covariance[0] = mean[0] + (1.0 - alpha * alpha + beta);
    (mean, covariance)
}

fn sigma_points(mean: Vec3, covariance: Mat3) -> Result<[Vec3; 7], String> {
    let lower = cholesky(covariance)?;
    let scale = 0.75_f64.sqrt();
    let mut points = [mean; 7];
    for column in 0..3 {
        for row in 0..3 {
            points[1 + column][row] = mean[row] + scale * lower[row][column];
            points[4 + column][row] = mean[row] - scale * lower[row][column];
        }
    }
    Ok(points)
}

fn weighted_mean(points: &[Vec3], weights: &[f64; 7]) -> Vec3 {
    let mut mean = [0.0; 3];
    for (point, weight) in points.iter().zip(weights) {
        for row in 0..3 {
            mean[row] += weight * point[row];
        }
    }
    mean
}

fn gp_residual(
    time: &[f64],
    residual: &[f64],
    length_scale_s: f64,
    signal_variance: f64,
    noise_k: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let count = time.len();
    if count == 0 || residual.len() != count {
        return Err("GP residual is empty or inconsistent".into());
    }
    let signal_variance = signal_variance * signal_variance;
    let noise_variance = noise_k * noise_k;
    let mut kernel = vec![vec![0.0; count]; count];
    for row in 0..count {
        for column in 0..count {
            let distance = (time[row] - time[column]) / length_scale_s;
            kernel[row][column] = signal_variance * (-0.5 * distance * distance).exp();
        }
        kernel[row][row] += noise_variance;
    }
    let lower = cholesky_dense(&kernel)?;
    let alpha = solve_dense(&lower, residual);
    let mut mean = Vec::with_capacity(count);
    let mut variance = Vec::with_capacity(count);
    for row in 0..count {
        let mut kernel_row = vec![0.0; count];
        for column in 0..count {
            let distance = (time[row] - time[column]) / length_scale_s;
            kernel_row[column] = signal_variance * (-0.5 * distance * distance).exp();
        }
        let solved = solve_dense(&lower, &kernel_row);
        mean.push(kernel_row.iter().zip(&alpha).map(|(a, b)| a * b).sum());
        let explained = kernel_row
            .iter()
            .zip(&solved)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        variance.push((signal_variance - explained).max(0.0));
    }
    Ok((mean, variance.into_iter().map(f64::sqrt).collect()))
}

fn rmse(values: &[f64], truth: &[f64]) -> f64 {
    if values.is_empty() || values.len() != truth.len() {
        return f64::NAN;
    }
    (values
        .iter()
        .zip(truth)
        .map(|(value, expected)| (value - expected).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn mat_zero() -> Mat3 {
    [[0.0; 3]; 3]
}

fn mat_identity() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn mat_diag(diagonal: Vec3) -> Mat3 {
    let mut value = mat_zero();
    for index in 0..3 {
        value[index][index] = diagonal[index];
    }
    value
}

fn mat_scale(value: Mat3, scale: f64) -> Mat3 {
    value.map(|row| row.map(|entry| entry * scale))
}

fn mat_add(left: Mat3, right: Mat3) -> Mat3 {
    let mut value = mat_zero();
    for row in 0..3 {
        for column in 0..3 {
            value[row][column] = left[row][column] + right[row][column];
        }
    }
    value
}

fn mat_sub(left: Mat3, right: Mat3) -> Mat3 {
    let mut value = mat_zero();
    for row in 0..3 {
        for column in 0..3 {
            value[row][column] = left[row][column] - right[row][column];
        }
    }
    value
}

fn sub(left: Vec3, right: Vec3) -> Vec3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn add(left: Vec3, right: Vec3) -> Vec3 {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn mat_vec(matrix: Mat3, vector: Vec3) -> Vec3 {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn mat_outer(vector: Vec3) -> Mat3 {
    mat_outer_pair(vector, vector)
}

fn mat_outer_pair(left: Vec3, right: Vec3) -> Mat3 {
    let mut value = mat_zero();
    for row in 0..3 {
        for column in 0..3 {
            value[row][column] = left[row] * right[column];
        }
    }
    value
}

fn mat_mul(left: Mat3, right: Mat3) -> Mat3 {
    let mut value = mat_zero();
    for row in 0..3 {
        for column in 0..3 {
            value[row][column] = (0..3)
                .map(|index| left[row][index] * right[index][column])
                .sum();
        }
    }
    value
}

fn transpose(value: Mat3) -> Mat3 {
    let mut result = mat_zero();
    for row in 0..3 {
        for column in 0..3 {
            result[row][column] = value[column][row];
        }
    }
    result
}

fn symmetrise(value: &mut Mat3) {
    for row in 0..3 {
        for column in row + 1..3 {
            let average = 0.5 * (value[row][column] + value[column][row]);
            value[row][column] = average;
            value[column][row] = average;
        }
    }
}

fn mat_inverse(value: Mat3) -> Result<Mat3, String> {
    let determinant = value[0][0] * (value[1][1] * value[2][2] - value[1][2] * value[2][1])
        - value[0][1] * (value[1][0] * value[2][2] - value[1][2] * value[2][0])
        + value[0][2] * (value[1][0] * value[2][1] - value[1][1] * value[2][0]);
    if determinant.abs() < 1.0e-15 {
        return Err("UKF covariance is singular".into());
    }
    let mut inverse = mat_zero();
    inverse[0][0] = (value[1][1] * value[2][2] - value[1][2] * value[2][1]) / determinant;
    inverse[0][1] = (value[0][2] * value[2][1] - value[0][1] * value[2][2]) / determinant;
    inverse[0][2] = (value[0][1] * value[1][2] - value[0][2] * value[1][1]) / determinant;
    inverse[1][0] = (value[1][2] * value[2][0] - value[1][0] * value[2][2]) / determinant;
    inverse[1][1] = (value[0][0] * value[2][2] - value[0][2] * value[2][0]) / determinant;
    inverse[1][2] = (value[0][2] * value[1][0] - value[0][0] * value[1][2]) / determinant;
    inverse[2][0] = (value[1][0] * value[2][1] - value[1][1] * value[2][0]) / determinant;
    inverse[2][1] = (value[0][1] * value[2][0] - value[0][0] * value[2][1]) / determinant;
    inverse[2][2] = (value[0][0] * value[1][1] - value[0][1] * value[1][0]) / determinant;
    Ok(inverse)
}

fn cholesky(value: Mat3) -> Result<Mat3, String> {
    let mut lower = mat_zero();
    for row in 0..3 {
        for column in 0..=row {
            let mut sum = value[row][column];
            for index in 0..column {
                sum -= lower[row][index] * lower[column][index];
            }
            if row == column {
                if sum <= 1.0e-15 {
                    return Err("UKF covariance is not positive definite".into());
                }
                lower[row][column] = sum.sqrt();
            } else {
                lower[row][column] = sum / lower[column][column];
            }
        }
    }
    Ok(lower)
}

fn cholesky_dense(value: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let count = value.len();
    let mut lower = vec![vec![0.0; count]; count];
    for row in 0..count {
        for column in 0..=row {
            let mut sum = value[row][column];
            for index in 0..column {
                sum -= lower[row][index] * lower[column][index];
            }
            if row == column {
                if sum <= 1.0e-15 {
                    return Err("GP kernel is not positive definite".into());
                }
                lower[row][column] = sum.sqrt();
            } else {
                lower[row][column] = sum / lower[column][column];
            }
        }
    }
    Ok(lower)
}

fn solve_dense(lower: &[Vec<f64>], right: &[f64]) -> Vec<f64> {
    let count = lower.len();
    let mut intermediate = vec![0.0; count];
    for row in 0..count {
        let mut value = right[row];
        for column in 0..row {
            value -= lower[row][column] * intermediate[column];
        }
        intermediate[row] = value / lower[row][row];
    }
    let mut result = vec![0.0; count];
    for row in (0..count).rev() {
        let mut value = intermediate[row];
        for column in row + 1..count {
            value -= lower[column][row] * result[column];
        }
        result[row] = value / lower[row][row];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoother_recovers_the_latent_junction() {
        let config = ThermalConfig::default();
        let estimate = estimate_thermal(&config).expect("thermal estimate");
        assert_eq!(estimate.smoothed_junction_k.len(), 181);
        assert!(
            estimate.junction_rmse_k < 0.1,
            "junction RMSE {} K",
            estimate.junction_rmse_k
        );
        assert!(
            estimate.sensor_rmse_k < config.sensor_noise_k * 3.0,
            "sensor RMSE {} K",
            estimate.sensor_rmse_k
        );
        assert!(estimate
            .smoothed_junction_sd_k
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn marginal_likelihood_penalises_a_mismatched_noise_level() {
        let config = ThermalConfig::default();
        let run = simulate_thermal(&config).expect("thermal trace");
        let matched = smooth_thermal(&config, &run).expect("matched likelihood");
        assert!(matched.log_likelihood.is_finite());

        // The same observations judged by a filter that assumes ten times less
        // sensor noise must score the data as far less likely.
        let tighter = ThermalConfig {
            sensor_noise_k: config.sensor_noise_k * 0.1,
            ..config
        };
        let mismatched = smooth_thermal(&tighter, &run).expect("mismatched likelihood");
        assert!(
            mismatched.log_likelihood < matched.log_likelihood - 10.0,
            "matched {} vs mismatched {}",
            matched.log_likelihood,
            mismatched.log_likelihood
        );
    }

    #[test]
    fn matern_state_is_finite_and_bounded_by_the_signal_scale() {
        let config = ThermalConfig::default();
        let estimate = estimate_thermal(&config).expect("thermal estimate");
        assert_eq!(
            estimate.matern_mean_k.len(),
            estimate.smoothed_sensor_k.len()
        );
        assert!(estimate.matern_rms_k.is_finite());
        assert!(estimate
            .matern_mean_k
            .iter()
            .all(|value| value.is_finite() && value.abs() <= 0.2 * 3.0));
        assert!(estimate
            .matern_sd_k
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn gp_discrepancy_is_finite() {
        let config = ThermalConfig::default();
        let estimate = estimate_thermal(&config).expect("thermal estimate");
        assert_eq!(estimate.gp_mean_k.len(), 181);
        assert_eq!(estimate.gp_sd_k.len(), 181);
        assert!(estimate.gp_rms_k.is_finite());
        assert!(estimate.gp_mean_k.iter().all(|value| value.is_finite()));
        assert!(estimate
            .gp_sd_k
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }
}
