//! Matérn-3/2 discrepancy as a two-state linear SDE.
//!
//! A Matérn-3/2 Gaussian process with length scale `ℓ` and stationary variance
//! `σ²` is exactly the stationary distribution of
//!
//! ```text
//! d/dt [f, f'] = [[0, 1], [-λ², -2λ]] [f, f'] + [0, 1]ᵀ w,    λ = √3 / ℓ
//! ```
//!
//! with white noise of spectral density `q = 4 λ³ σ²`, so the discrepancy can be
//! carried as a latent state instead of a kernel matrix over the whole trace.
//! Every step is `O(1)`: the transition is the exact matrix exponential, the
//! process noise is that transition integrated over the step, and the filter is
//! a two-state Kalman filter with an RTS smoother.
//!
//! The hyperparameters are fixed, like the RBF diagnostic in `ukf`: this is a
//! stated prior on the discrepancy, not a learned or calibrated one.

type Vec2 = [f64; 2];
type Mat2 = [[f64; 2]; 2];

/// Eight-point Gauss–Legendre nodes and weights on `[-1, 1]`.
const GAUSS_NODES: [f64; 8] = [
    -0.960_289_856_497_536_3,
    -0.796_666_477_413_626_7,
    -0.525_532_409_916_329,
    -0.183_434_642_495_649_8,
    0.183_434_642_495_649_8,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];

const GAUSS_WEIGHTS: [f64; 8] = [
    0.101_228_536_290_376_3,
    0.222_381_034_453_374_5,
    0.313_706_645_877_887_3,
    0.362_683_783_378_362,
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_5,
    0.101_228_536_290_376_3,
];

#[derive(Debug, Clone, PartialEq)]
pub struct MaternDiscrepancy {
    /// Smoothed discrepancy mean in kelvin.
    pub mean_k: Vec<f64>,
    /// Standard deviation of the smoothed discrepancy in kelvin.
    pub sd_k: Vec<f64>,
    /// Root mean square of the smoothed discrepancy in kelvin.
    pub rms_k: f64,
    /// Predictive log-likelihood of the residual series under the SDE.
    pub log_likelihood: f64,
    pub length_scale_s: f64,
    pub signal_sd_k: f64,
}

/// Filter and smooth a residual series with a Matérn-3/2 latent state.
pub fn matern_discrepancy(
    time_s: &[f64],
    residual_k: &[f64],
    length_scale_s: f64,
    signal_sd_k: f64,
    noise_k: f64,
) -> Result<MaternDiscrepancy, String> {
    let count = time_s.len();
    if count == 0 || residual_k.len() != count {
        return Err("Matérn discrepancy is empty or inconsistent".into());
    }
    if !length_scale_s.is_finite()
        || length_scale_s <= 0.0
        || !signal_sd_k.is_finite()
        || signal_sd_k < 0.0
        || !noise_k.is_finite()
        || noise_k < 0.0
    {
        return Err("Matérn hyperparameters are outside their physical domain".into());
    }
    let step = if count > 1 {
        time_s[1] - time_s[0]
    } else {
        0.0
    };
    if count > 1 && (!step.is_finite() || step <= 0.0) {
        return Err("Matérn discrepancy needs a positive sample interval".into());
    }

    let lambda = 3.0_f64.sqrt() / length_scale_s;
    let transition = transition_matrix(lambda, step);
    let process = process_covariance(lambda, signal_sd_k, step);
    let measurement_variance = noise_k * noise_k;

    let mut prior_mean = Vec::with_capacity(count);
    let mut prior_covariance = Vec::with_capacity(count);
    let mut filtered_mean = Vec::with_capacity(count);
    let mut filtered_covariance = Vec::with_capacity(count);
    let mut log_likelihood = 0.0;

    let mut mean: Vec2 = [0.0, 0.0];
    let mut covariance = stationary_covariance(lambda, signal_sd_k);
    for (index, measurement) in residual_k.iter().enumerate() {
        if index > 0 {
            mean = mat_vec(transition, mean);
            covariance = mat_add(
                mat_mul(mat_mul(transition, covariance), transpose(transition)),
                process,
            );
        }
        prior_mean.push(mean);
        prior_covariance.push(covariance);

        let innovation_variance = covariance[0][0] + measurement_variance;
        if !innovation_variance.is_finite() || innovation_variance <= 0.0 {
            return Err("Matérn innovation variance is not positive".into());
        }
        let gain: Vec2 = [
            covariance[0][0] / innovation_variance,
            covariance[1][0] / innovation_variance,
        ];
        let innovation = measurement - mean[0];
        log_likelihood += -0.5
            * ((std::f64::consts::TAU * innovation_variance).ln()
                + innovation * innovation / innovation_variance);
        mean = [
            mean[0] + gain[0] * innovation,
            mean[1] + gain[1] * innovation,
        ];
        // P − K H P, with H P the first row of P.
        let first_row = covariance[0];
        for row in 0..2 {
            for column in 0..2 {
                covariance[row][column] -= gain[row] * first_row[column];
            }
        }
        covariance = symmetrise(covariance);
        filtered_mean.push(mean);
        filtered_covariance.push(covariance);
    }

    let mut smoothed_mean = filtered_mean.clone();
    let mut smoothed_covariance = filtered_covariance.clone();
    for index in (0..count - 1).rev() {
        let gain = mat_mul(
            mat_mul(filtered_covariance[index], transpose(transition)),
            inverse(prior_covariance[index + 1]),
        );
        let difference = [
            smoothed_mean[index + 1][0] - prior_mean[index + 1][0],
            smoothed_mean[index + 1][1] - prior_mean[index + 1][1],
        ];
        smoothed_mean[index] = [
            filtered_mean[index][0] + gain[0][0] * difference[0] + gain[0][1] * difference[1],
            filtered_mean[index][1] + gain[1][0] * difference[0] + gain[1][1] * difference[1],
        ];
        let covariance_difference =
            mat_sub(smoothed_covariance[index + 1], prior_covariance[index + 1]);
        smoothed_covariance[index] = mat_add(
            filtered_covariance[index],
            mat_mul(mat_mul(gain, covariance_difference), transpose(gain)),
        );
        smoothed_covariance[index] = symmetrise(smoothed_covariance[index]);
    }

    let mean_k = smoothed_mean
        .iter()
        .map(|state| state[0])
        .collect::<Vec<_>>();
    let sd_k = smoothed_covariance
        .iter()
        .map(|covariance| covariance[0][0].max(0.0).sqrt())
        .collect::<Vec<_>>();
    let rms_k = (mean_k.iter().map(|value| value * value).sum::<f64>() / count as f64).sqrt();

    Ok(MaternDiscrepancy {
        mean_k,
        sd_k,
        rms_k,
        log_likelihood,
        length_scale_s,
        signal_sd_k,
    })
}

/// Stationary covariance of the SDE, `diag(σ², λ²σ²)`.
fn stationary_covariance(lambda: f64, signal_sd_k: f64) -> Mat2 {
    let variance = signal_sd_k * signal_sd_k;
    [[variance, 0.0], [0.0, lambda * lambda * variance]]
}

/// Exact matrix exponential of the Matérn-3/2 generator.
fn transition_matrix(lambda: f64, step: f64) -> Mat2 {
    let decay = (-lambda * step).exp();
    [
        [decay * (1.0 + lambda * step), decay * step],
        [
            -decay * lambda * lambda * step,
            decay * (1.0 - lambda * step),
        ],
    ]
}

/// Discretised process noise, `∫₀^Δ Φ(s) G q Gᵀ Φ(s)ᵀ ds`.
///
/// The second column of `Φ(s)` carries the noise, so each quadrature node
/// contributes `q·w·v vᵀ` with `v = e^{−λs}[s, 1 − λs]`.
fn process_covariance(lambda: f64, signal_sd_k: f64, step: f64) -> Mat2 {
    if step <= 0.0 {
        return [[0.0; 2]; 2];
    }
    let variance = signal_sd_k * signal_sd_k;
    let spectral_density = 4.0 * lambda * lambda * lambda * variance;
    let mut value = [[0.0; 2]; 2];
    for (node, weight) in GAUSS_NODES.iter().zip(&GAUSS_WEIGHTS) {
        let time = 0.5 * step * (node + 1.0);
        let scaled_weight = 0.5 * step * weight * (-2.0 * lambda * time).exp();
        let first = time;
        let second = 1.0 - lambda * time;
        value[0][0] += scaled_weight * first * first;
        value[0][1] += scaled_weight * first * second;
        value[1][1] += scaled_weight * second * second;
    }
    value[1][0] = value[0][1];
    mat_scale(value, spectral_density)
}

fn mat_vec(matrix: Mat2, vector: Vec2) -> Vec2 {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

fn mat_mul(left: Mat2, right: Mat2) -> Mat2 {
    let mut value = [[0.0; 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            value[row][column] = left[row][0] * right[0][column] + left[row][1] * right[1][column];
        }
    }
    value
}

fn mat_add(left: Mat2, right: Mat2) -> Mat2 {
    [
        [left[0][0] + right[0][0], left[0][1] + right[0][1]],
        [left[1][0] + right[1][0], left[1][1] + right[1][1]],
    ]
}

fn mat_sub(left: Mat2, right: Mat2) -> Mat2 {
    [
        [left[0][0] - right[0][0], left[0][1] - right[0][1]],
        [left[1][0] - right[1][0], left[1][1] - right[1][1]],
    ]
}

fn mat_scale(value: Mat2, scale: f64) -> Mat2 {
    [
        [value[0][0] * scale, value[0][1] * scale],
        [value[1][0] * scale, value[1][1] * scale],
    ]
}

fn transpose(value: Mat2) -> Mat2 {
    [[value[0][0], value[1][0]], [value[0][1], value[1][1]]]
}

fn symmetrise(value: Mat2) -> Mat2 {
    let average = 0.5 * (value[0][1] + value[1][0]);
    [[value[0][0], average], [average, value[1][1]]]
}

fn inverse(value: Mat2) -> Mat2 {
    let determinant = value[0][0] * value[1][1] - value[0][1] * value[1][0];
    if !determinant.is_finite() || determinant.abs() < 1.0e-18 {
        // A singular predictive covariance means the step carried no
        // uncertainty; the smoother gain is then zero, not infinite.
        return [[0.0; 2]; 2];
    }
    let scale = 1.0 / determinant;
    [
        [value[1][1] * scale, -value[0][1] * scale],
        [-value[1][0] * scale, value[0][0] * scale],
    ]
}

/// Noise-free check helper: the Matérn-3/2 autocovariance at lag `τ`.
pub fn autocovariance(tau_s: f64, length_scale_s: f64, signal_sd_k: f64) -> f64 {
    let scaled = 3.0_f64.sqrt() * tau_s.abs() / length_scale_s;
    signal_sd_k * signal_sd_k * (1.0 + scaled) * (-scaled).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn discretisation_keeps_the_stationary_covariance() {
        let lambda = 3.0_f64.sqrt() / 180.0;
        let stationarity = stationary_covariance(lambda, 0.4);
        let transition = transition_matrix(lambda, 10.0);
        let process = process_covariance(lambda, 0.4, 10.0);
        let propagated = mat_add(
            mat_mul(mat_mul(transition, stationarity), transpose(transition)),
            process,
        );
        assert!(
            (propagated[0][0] - stationarity[0][0]).abs() < 1.0e-12,
            "{propagated:?} vs {stationarity:?}"
        );
        assert!((propagated[1][1] - stationarity[1][1]).abs() < 1.0e-12);
        assert!(propagated[0][1].abs() < 1.0e-12);
        // Zero-lag autocovariance is the stationary variance.
        assert!((autocovariance(0.0, 180.0, 0.4) - 0.16).abs() < 1.0e-12);
    }

    #[test]
    fn smoother_recovers_a_smooth_residual() {
        let count = 181;
        let time = (0..count)
            .map(|index| index as f64 * 10.0)
            .collect::<Vec<_>>();
        let truth = time
            .iter()
            .map(|value| 0.6 * (2.0 * PI * value / 600.0).sin())
            .collect::<Vec<_>>();
        // Deterministic pseudo-noise so the test does not depend on an RNG.
        let observed = truth
            .iter()
            .enumerate()
            .map(|(index, value)| value + 0.02 * ((index as f64) * 2.399_963).sin())
            .collect::<Vec<_>>();
        let estimate =
            matern_discrepancy(&time, &observed, 180.0, 0.6, 0.02).expect("Matérn discrepancy");
        assert_eq!(estimate.mean_k.len(), count);
        let rmse = (estimate
            .mean_k
            .iter()
            .zip(&truth)
            .map(|(value, expected)| (value - expected).powi(2))
            .sum::<f64>()
            / count as f64)
            .sqrt();
        assert!(rmse < 0.06, "smoothed RMSE {rmse} K");
        assert!(estimate.log_likelihood.is_finite());
        assert!(estimate
            .sd_k
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn rejects_empty_input() {
        assert!(matern_discrepancy(&[], &[], 180.0, 0.2, 0.04).is_err());
        let time = [0.0, 10.0];
        let residual = [0.0, 0.1];
        assert!(matern_discrepancy(&time, &residual, 0.0, 0.2, 0.04).is_err());
        assert!(matern_discrepancy(&time, &[0.0], 180.0, 0.2, 0.04).is_err());
    }
}
