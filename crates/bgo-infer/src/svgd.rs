//! Deterministic Stein variational gradient descent (SVGD).
//!
//! The update is
//! `x_i <- x_i + eps / n * sum_j [k(x_i, x_j) grad log p(x_j) + grad_{x_j} k(x_i, x_j)]`
//! with an isotropic RBF kernel. The repulsion term preserves diversity, but
//! SVGD is still a deterministic approximation: its variance can be smaller
//! than the target variance, especially in more than one dimension. Callers
//! must report that bias rather than treating the particle cloud as a sampler.

use crate::model::Case;

/// A log density with its gradient. Implementations must return a gradient
/// with the same length as `theta`.
pub trait SvgdTarget {
    fn dimension(&self) -> usize;
    fn logp_and_grad(&self, theta: &[f64]) -> (f64, Vec<f64>);
}

/// Isotropic Gaussian target used by the validation fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianTarget {
    mean: Vec<f64>,
    sigma: Vec<f64>,
}

impl GaussianTarget {
    pub fn new(mean: Vec<f64>, sigma: Vec<f64>) -> Result<Self, SvgdError> {
        if mean.is_empty() || mean.len() != sigma.len() {
            return Err(SvgdError::DimensionMismatch);
        }
        if mean.iter().chain(&sigma).any(|value| !value.is_finite())
            || sigma.iter().any(|value| *value <= 0.0)
        {
            return Err(SvgdError::InvalidConfig);
        }
        Ok(Self { mean, sigma })
    }
}

impl SvgdTarget for GaussianTarget {
    fn dimension(&self) -> usize {
        self.mean.len()
    }

    fn logp_and_grad(&self, theta: &[f64]) -> (f64, Vec<f64>) {
        if theta.len() != self.mean.len() {
            return (f64::NEG_INFINITY, vec![0.0; self.mean.len()]);
        }
        let mut logp = 0.0;
        let mut gradient = vec![0.0; theta.len()];
        for (index, value) in theta.iter().enumerate() {
            let standardized = (value - self.mean[index]) / self.sigma[index];
            logp -= 0.5 * standardized * standardized + self.sigma[index].ln();
            gradient[index] = -standardized / self.sigma[index];
        }
        (logp, gradient)
    }
}

/// A [`Case`] used as an SVGD target. The case owns the bounds and transforms.
pub struct CaseTarget<'a> {
    case: &'a Case,
}

impl<'a> CaseTarget<'a> {
    pub fn new(case: &'a Case) -> Self {
        Self { case }
    }
}

impl SvgdTarget for CaseTarget<'_> {
    fn dimension(&self) -> usize {
        self.case.latent_len()
    }

    fn logp_and_grad(&self, theta: &[f64]) -> (f64, Vec<f64>) {
        self.case.logp_and_grad(theta)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvgdConfig {
    pub iterations: usize,
    pub step_size: f64,
    pub bandwidth: f64,
    /// Capture the particle cloud every `trace_every` updates; the final cloud
    /// is always captured.
    pub trace_every: usize,
}

impl Default for SvgdConfig {
    fn default() -> Self {
        Self {
            iterations: 240,
            step_size: 0.08,
            bandwidth: 0.5,
            trace_every: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SvgdOutput {
    pub initial_particles: Vec<Vec<f64>>,
    pub particles: Vec<Vec<f64>>,
    /// Frame-major, particle-major, coordinate-major snapshots.
    pub trajectory: Vec<Vec<Vec<f64>>>,
    pub initial_mean: Vec<f64>,
    pub mean: Vec<f64>,
    /// Population variance of the final particle cloud.
    pub variance: Vec<f64>,
    /// Mean final log density, ignoring non-finite values.
    pub final_logp_mean: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SvgdError {
    TooFewParticles,
    DimensionMismatch,
    InvalidConfig,
    NonFinite,
    EmptyTarget,
}

impl std::fmt::Display for SvgdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewParticles => write!(f, "SVGD needs at least two particles"),
            Self::DimensionMismatch => write!(f, "particle and target dimensions differ"),
            Self::InvalidConfig => write!(f, "invalid SVGD step, bandwidth or iteration count"),
            Self::NonFinite => write!(f, "SVGD produced a non-finite particle or gradient"),
            Self::EmptyTarget => write!(f, "the SVGD target has zero dimensions"),
        }
    }
}

impl std::error::Error for SvgdError {}

/// Evolve the supplied particles under the target.
///
/// The caller owns initialization. This keeps the prior or proposal explicit:
/// the returned trajectory starts at exactly the supplied cloud, and no
/// posterior draws are used internally.
pub fn run_svgd<T: SvgdTarget>(
    target: &T,
    initial: &[Vec<f64>],
    config: &SvgdConfig,
) -> Result<SvgdOutput, SvgdError> {
    let dimension = target.dimension();
    if dimension == 0 {
        return Err(SvgdError::EmptyTarget);
    }
    if initial.len() < 2 {
        return Err(SvgdError::TooFewParticles);
    }
    if initial.iter().any(|particle| {
        particle.len() != dimension || particle.iter().any(|value| !value.is_finite())
    }) {
        return Err(SvgdError::DimensionMismatch);
    }
    if config.iterations == 0
        || config.trace_every == 0
        || !config.step_size.is_finite()
        || config.step_size <= 0.0
        || !config.bandwidth.is_finite()
        || config.bandwidth <= 0.0
    {
        return Err(SvgdError::InvalidConfig);
    }

    let count = initial.len();
    let count_f64 = count as f64;
    let bandwidth_squared = config.bandwidth * config.bandwidth;
    let mut particles = initial.to_vec();
    let mut trajectory = vec![particles.clone()];

    for iteration in 0..config.iterations {
        let mut scores = Vec::with_capacity(count);
        for particle in &particles {
            let (_logp, gradient) = target.logp_and_grad(particle);
            if gradient.len() != dimension || gradient.iter().any(|value| !value.is_finite()) {
                return Err(SvgdError::NonFinite);
            }
            scores.push(gradient);
        }

        let mut updates = vec![vec![0.0; dimension]; count];
        for (i, particle_i) in particles.iter().enumerate() {
            let update = &mut updates[i];
            for (j, particle_j) in particles.iter().enumerate() {
                let mut squared_distance = 0.0;
                for (left, right) in particle_i.iter().zip(particle_j) {
                    let difference = left - right;
                    squared_distance += difference * difference;
                }
                let kernel = (-squared_distance / (2.0 * bandwidth_squared)).exp();
                for coordinate in 0..dimension {
                    let difference = particle_i[coordinate] - particle_j[coordinate];
                    update[coordinate] +=
                        kernel * scores[j][coordinate] + kernel * difference / bandwidth_squared;
                }
            }
            for value in update.iter_mut() {
                *value /= count_f64;
            }
        }

        for (particle, update) in particles.iter_mut().zip(updates) {
            for (coordinate, delta) in particle.iter_mut().zip(update) {
                *coordinate += config.step_size * delta;
                if !coordinate.is_finite() {
                    return Err(SvgdError::NonFinite);
                }
            }
        }

        if (iteration + 1) % config.trace_every == 0 || iteration + 1 == config.iterations {
            trajectory.push(particles.clone());
        }
    }

    let mut final_logp_sum = 0.0;
    let mut final_logp_count = 0usize;
    for particle in &particles {
        let (logp, _gradient) = target.logp_and_grad(particle);
        if logp.is_finite() {
            final_logp_sum += logp;
            final_logp_count += 1;
        }
    }
    let final_logp_mean = if final_logp_count == 0 {
        f64::NEG_INFINITY
    } else {
        final_logp_sum / final_logp_count as f64
    };

    Ok(SvgdOutput {
        initial_particles: initial.to_vec(),
        initial_mean: column_mean(initial, dimension),
        mean: column_mean(&particles, dimension),
        variance: column_variance(&particles, dimension),
        particles,
        trajectory,
        final_logp_mean,
    })
}

fn column_mean(particles: &[Vec<f64>], dimension: usize) -> Vec<f64> {
    (0..dimension)
        .map(|coordinate| {
            particles
                .iter()
                .map(|particle| particle[coordinate])
                .sum::<f64>()
                / particles.len() as f64
        })
        .collect()
}

fn column_variance(particles: &[Vec<f64>], dimension: usize) -> Vec<f64> {
    let mean = column_mean(particles, dimension);
    (0..dimension)
        .map(|coordinate| {
            particles
                .iter()
                .map(|particle| {
                    let difference = particle[coordinate] - mean[coordinate];
                    difference * difference
                })
                .sum::<f64>()
                / particles.len() as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Case, Model};

    #[test]
    fn gaussian_mean_converges_and_variance_bias_is_exposed() {
        let target = GaussianTarget::new(vec![0.0, 0.0], vec![1.0, 1.0]).expect("target");
        let mut initial = Vec::new();
        for row in 0..16 {
            for column in 0..16 {
                initial.push(vec![
                    1.2 + (row as f64 - 7.5) * 0.08,
                    -0.8 + (column as f64 - 7.5) * 0.08,
                ]);
            }
        }
        let output = run_svgd(
            &target,
            &initial,
            &SvgdConfig {
                iterations: 500,
                step_size: 0.08,
                bandwidth: 0.5,
                trace_every: 50,
            },
        )
        .expect("SVGD");
        assert!(output.mean[0].abs() < 0.05, "mean[0] = {}", output.mean[0]);
        assert!(output.mean[1].abs() < 0.05, "mean[1] = {}", output.mean[1]);
        // The RBF kernel is expected to contract relative to the target. The
        // exact amount is reported by the caller; this test only guards the
        // direction and a useful lower bound.
        assert!(
            output.variance[0] < 1.05,
            "variance = {}",
            output.variance[0]
        );
        assert!(
            output.variance[0] > 0.35,
            "variance = {}",
            output.variance[0]
        );
        assert_eq!(output.trajectory.len(), 11);
        assert!(output.final_logp_mean.is_finite());
    }

    #[test]
    fn case_target_returns_finite_scores() {
        let case = Case::new(
            Model::M0,
            vec![293.15, 320.0, 350.0, 383.15],
            vec![0.65775, 0.65025, 0.6409, 0.6298],
            3.0e-4,
            1.253_079_539_906_918e-4,
        )
        .expect("case");
        let target = CaseTarget::new(&case);
        let initial = vec![
            vec![1.00, 5.50],
            vec![1.01, 5.52],
            vec![0.99, 5.48],
            vec![1.02, 5.54],
        ];
        let output = run_svgd(
            &target,
            &initial,
            &SvgdConfig {
                iterations: 8,
                step_size: 0.005,
                bandwidth: 1.0,
                trace_every: 2,
            },
        )
        .expect("model SVGD");
        assert!(output.mean.iter().all(|value| value.is_finite()));
        assert!(output.variance.iter().all(|value| value.is_finite()));
        assert_eq!(output.trajectory.len(), 5);
    }
}
