//! Nested sampling on a unit-cube prior.
//!
//! The target owns the dimension and evaluates `ln L + ln pi` for a point in
//! `[0, 1]^d`. The sampler replaces the lowest live point with a constrained
//! slice sample, accumulates the one-dimensional evidence integral, and emits
//! a typed event for each replacement so the S2 visualization can remain a
//! projection of the actual computation.

/// A target with a uniform prior on the unit cube.
pub trait LogLikePrior {
    fn dimension(&self) -> usize;

    /// `ln L(theta) + ln pi(theta)` for `theta` in `[0, 1]^d`.
    fn log_like_prior(&self, theta: &[f64]) -> f64;
}

#[derive(Debug, Clone, PartialEq)]
pub struct NestedConfig {
    pub live_points: usize,
    pub max_iterations: usize,
    /// Stop when the remaining evidence bound is below this fraction of `Z`.
    pub remaining_evidence: f64,
    pub seed: u64,
    /// Full coordinate sweeps per replacement.
    pub slice_steps: usize,
    pub slice_width: f64,
    pub max_step_out: usize,
    pub max_shrink: usize,
}

impl Default for NestedConfig {
    fn default() -> Self {
        Self {
            live_points: 400,
            max_iterations: 20_000,
            remaining_evidence: 1.0e-3,
            seed: 0x6e65_7374_6564,
            slice_steps: 20,
            slice_width: 0.05,
            max_step_out: 50,
            max_shrink: 1_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NestedEvent {
    pub iteration: usize,
    pub live_points: usize,
    pub log_likelihood_threshold: f64,
    pub log_evidence: f64,
    pub log_prior_volume: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NestedPoint {
    pub position: Vec<f64>,
    pub log_likelihood: f64,
    pub log_weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NestedOutput {
    pub log_evidence: f64,
    pub log_evidence_error: f64,
    pub information: f64,
    pub iterations: usize,
    pub converged: bool,
    pub live_points: Vec<NestedPoint>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NestedError {
    InvalidConfig,
    NonFiniteTarget,
    SamplingFailed,
}

impl std::fmt::Display for NestedError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig => write!(formatter, "invalid nested-sampling configuration"),
            Self::NonFiniteTarget => write!(formatter, "the target returned a non-finite value"),
            Self::SamplingFailed => write!(formatter, "constrained slice sampling failed"),
        }
    }
}

impl std::error::Error for NestedError {}

/// Run nested sampling and call `on_event` after each replacement.
pub fn run_nested<T, F>(
    target: &T,
    config: &NestedConfig,
    mut on_event: F,
) -> Result<NestedOutput, NestedError>
where
    T: LogLikePrior + ?Sized,
    F: FnMut(NestedEvent),
{
    validate_config(target.dimension(), config)?;
    let dimension = target.dimension();
    let live_count = config.live_points;
    let mut rng = SplitMix64::new(config.seed);
    let mut live = initialize_live(target, dimension, live_count, &mut rng)?;

    let mut log_prior_volume = 0.0;
    let mut log_evidence = f64::NEG_INFINITY;
    let mut log_weights: Vec<(f64, f64)> = Vec::new();
    let mut iterations = 0;
    let mut converged = false;
    let log_shell = (1.0 - (-1.0 / live_count as f64).exp()).ln();

    while iterations < config.max_iterations {
        let max_live = live
            .iter()
            .map(|point| point.1)
            .fold(f64::NEG_INFINITY, f64::max);
        if log_evidence.is_finite()
            && max_live + log_prior_volume - log_evidence < config.remaining_evidence.ln()
        {
            converged = true;
            break;
        }

        let minimum = live
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.1.total_cmp(&right.1))
            .map(|(index, _)| index)
            .ok_or(NestedError::SamplingFailed)?;
        let removed = live.swap_remove(minimum);
        let next_log_volume = log_prior_volume - 1.0 / live_count as f64;
        let log_weight = log_prior_volume + log_shell;
        log_evidence = logsumexp(log_evidence, log_weight + removed.1);
        log_weights.push((log_weight, removed.1));
        log_prior_volume = next_log_volume;
        iterations += 1;

        let replacement = replace_live_point(target, &live, removed.1, config, &mut rng)?;
        live.push(replacement.clone());
        on_event(NestedEvent {
            iteration: iterations,
            live_points: live.len(),
            log_likelihood_threshold: removed.1,
            log_evidence,
            log_prior_volume,
        });
    }

    let final_log_weight = log_prior_volume - (live_count as f64).ln();
    for (position, log_likelihood) in &live {
        log_weights.push((final_log_weight, *log_likelihood));
        log_evidence = logsumexp(log_evidence, final_log_weight + log_likelihood);
        debug_assert_eq!(position.len(), dimension);
    }

    let total_evidence = log_weights
        .iter()
        .map(|(log_weight, log_likelihood)| (log_weight + log_likelihood).exp())
        .sum::<f64>();
    let weighted_log_likelihood = log_weights
        .iter()
        .map(|(log_weight, log_likelihood)| (log_weight + log_likelihood).exp() * log_likelihood)
        .sum::<f64>();
    let information = if total_evidence > 0.0 {
        weighted_log_likelihood / total_evidence - log_evidence
    } else {
        f64::NAN
    };
    if !log_evidence.is_finite() || !information.is_finite() {
        return Err(NestedError::NonFiniteTarget);
    }

    Ok(NestedOutput {
        log_evidence,
        log_evidence_error: (information.max(0.0) / live_count as f64).sqrt(),
        information,
        iterations,
        converged,
        live_points: live
            .into_iter()
            .map(|(position, log_likelihood)| NestedPoint {
                position,
                log_likelihood,
                log_weight: final_log_weight,
            })
            .collect(),
    })
}

fn validate_config(dimension: usize, config: &NestedConfig) -> Result<(), NestedError> {
    if dimension == 0
        || config.live_points < 2
        || config.max_iterations == 0
        || !config.remaining_evidence.is_finite()
        || !(0.0..1.0).contains(&config.remaining_evidence)
        || config.slice_steps == 0
        || !config.slice_width.is_finite()
        || config.slice_width <= 0.0
        || config.max_step_out == 0
        || config.max_shrink == 0
    {
        return Err(NestedError::InvalidConfig);
    }
    Ok(())
}

fn initialize_live<T>(
    target: &T,
    dimension: usize,
    live_count: usize,
    rng: &mut SplitMix64,
) -> Result<Vec<(Vec<f64>, f64)>, NestedError>
where
    T: LogLikePrior + ?Sized,
{
    let mut live = Vec::with_capacity(live_count);
    for _ in 0..live_count {
        let position = (0..dimension).map(|_| rng.uniform()).collect::<Vec<_>>();
        let log_likelihood = target.log_like_prior(&position);
        if !log_likelihood.is_finite() {
            return Err(NestedError::NonFiniteTarget);
        }
        live.push((position, log_likelihood));
    }
    Ok(live)
}

fn replace_live_point<T>(
    target: &T,
    live: &[(Vec<f64>, f64)],
    threshold: f64,
    config: &NestedConfig,
    rng: &mut SplitMix64,
) -> Result<(Vec<f64>, f64), NestedError>
where
    T: LogLikePrior + ?Sized,
{
    for _ in 0..8 {
        let start = &live[rng.index(live.len())];
        if let Some(point) = slice_sample(target, start, threshold, config, rng) {
            return Ok(point);
        }
    }
    Err(NestedError::SamplingFailed)
}

fn slice_sample<T>(
    target: &T,
    start: &(Vec<f64>, f64),
    threshold: f64,
    config: &NestedConfig,
    rng: &mut SplitMix64,
) -> Option<(Vec<f64>, f64)>
where
    T: LogLikePrior + ?Sized,
{
    let mut position = start.0.clone();
    let mut log_likelihood = start.1;
    for _ in 0..config.slice_steps {
        for coordinate in 0..position.len() {
            let current = position[coordinate];
            let log_slice = slice_height(log_likelihood, threshold, rng)?;
            let width = config.slice_width;
            let left_offset = rng.uniform() * width;
            let mut left = (current - left_offset).max(0.0);
            let mut right = (left + width).min(1.0);

            for _ in 0..config.max_step_out {
                if left <= 0.0 {
                    break;
                }
                let edge = (left - width).max(0.0);
                left = edge;
                if log_at(target, &position, coordinate, edge) < log_slice {
                    break;
                }
            }
            for _ in 0..config.max_step_out {
                if right >= 1.0 {
                    break;
                }
                let edge = (right + width).min(1.0);
                right = edge;
                if log_at(target, &position, coordinate, edge) < log_slice {
                    break;
                }
            }

            let mut accepted = false;
            for _ in 0..config.max_shrink {
                let candidate = left + rng.uniform() * (right - left);
                let value = log_at(target, &position, coordinate, candidate);
                if value >= log_slice {
                    position[coordinate] = candidate;
                    log_likelihood = value;
                    accepted = true;
                    break;
                }
                if candidate < current {
                    left = candidate;
                } else {
                    right = candidate;
                }
            }
            if !accepted {
                return None;
            }
        }
    }

    if log_likelihood >= threshold
        && position
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    {
        Some((position, log_likelihood))
    } else {
        None
    }
}

fn slice_height(current: f64, threshold: f64, rng: &mut SplitMix64) -> Option<f64> {
    // The auxiliary is uniform on (0, L(current)]. Values below the
    // truncation all induce the same (full constrained-support) slice, so
    // clamping them to the threshold preserves the x transition while keeping
    // the slice above the support boundary.
    let candidate = (current + rng.uniform().ln()).max(threshold);
    if candidate.is_finite() {
        Some(candidate)
    } else {
        None
    }
}

fn log_at<T>(target: &T, position: &[f64], coordinate: usize, value: f64) -> f64
where
    T: LogLikePrior + ?Sized,
{
    let mut candidate = position.to_vec();
    candidate[coordinate] = value;
    target.log_like_prior(&candidate)
}

fn logsumexp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn index(&mut self, length: usize) -> usize {
        (self.next_u64() % length as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Gaussian {
        center: [f64; 2],
        sigma: f64,
    }

    impl LogLikePrior for Gaussian {
        fn dimension(&self) -> usize {
            2
        }

        fn log_like_prior(&self, theta: &[f64]) -> f64 {
            let first = (theta[0] - self.center[0]) / self.sigma;
            let second = (theta[1] - self.center[1]) / self.sigma;
            -0.5 * (first * first + second * second)
        }
    }

    struct Constant {
        dimension: usize,
    }

    impl LogLikePrior for Constant {
        fn dimension(&self) -> usize {
            self.dimension
        }

        fn log_like_prior(&self, _theta: &[f64]) -> f64 {
            0.0
        }
    }

    #[test]
    fn constant_target_has_unit_evidence() {
        let config = NestedConfig {
            live_points: 100,
            seed: 0xc0_1157,
            ..NestedConfig::default()
        };
        let output = run_nested(&Constant { dimension: 1 }, &config, |_| {})
            .expect("constant target should converge");
        assert!(
            output.log_evidence.abs() < 1.0e-12,
            "ln Z = {} expected 0",
            output.log_evidence
        );
    }

    struct Gaussian1 {
        center: f64,
        sigma: f64,
    }

    impl LogLikePrior for Gaussian1 {
        fn dimension(&self) -> usize {
            1
        }

        fn log_like_prior(&self, theta: &[f64]) -> f64 {
            let standardized = (theta[0] - self.center) / self.sigma;
            -0.5 * standardized * standardized
        }
    }

    struct Linear;

    impl LogLikePrior for Linear {
        fn dimension(&self) -> usize {
            1
        }

        fn log_like_prior(&self, theta: &[f64]) -> f64 {
            theta[0].ln()
        }
    }

    #[test]
    fn constrained_slice_preserves_a_truncated_linear_mean() {
        let target = Linear;
        let threshold = 0.5_f64.ln();
        let config = NestedConfig::default();
        let mut rng = SplitMix64::new(0x11ce);
        let mut point = (vec![0.6], target.log_like_prior(&[0.6]));
        let mut total = 0.0;
        let count = 10_000;
        for _ in 0..count {
            point = slice_sample(&target, &point, threshold, &config, &mut rng)
                .expect("slice sample should stay in the constrained support");
            total += point.0[0];
        }
        let mean = total / count as f64;
        assert!(
            (mean - 7.0 / 9.0).abs() <= 0.01,
            "truncated mean {mean} should be 7/9"
        );
    }

    #[test]
    fn constrained_slice_preserves_a_truncated_gaussian_mean() {
        let target = Gaussian1 {
            center: 0.5,
            sigma: 0.1,
        };
        let threshold = target.log_like_prior(&[0.55]);
        let config = NestedConfig::default();
        let mut rng = SplitMix64::new(0x51ce);
        let mut point = (vec![0.5], target.log_like_prior(&[0.5]));
        let mut total = 0.0;
        let count = 10_000;
        for _ in 0..count {
            point = slice_sample(&target, &point, threshold, &config, &mut rng)
                .expect("slice sample should stay in the constrained support");
            total += point.0[0];
        }
        let mean = total / count as f64;
        assert!(
            (mean - 0.5).abs() <= 0.005,
            "truncated mean {mean} should be 0.5"
        );
    }

    #[test]
    fn gaussian_fixture_emits_a_finite_ordered_event_stream() {
        let target = Gaussian {
            center: [0.5, 0.5],
            sigma: 0.1,
        };
        let config = NestedConfig {
            seed: 0x5eed,
            ..NestedConfig::default()
        };
        let mut events = 0usize;
        let output = run_nested(&target, &config, |event| {
            assert_eq!(event.iteration, events + 1);
            assert_eq!(event.live_points, config.live_points);
            assert!(event.log_evidence.is_finite());
            assert!(event.log_likelihood_threshold.is_finite());
            events += 1;
        })
        .expect("nested sampling should run");

        assert_eq!(output.live_points[0].position.len(), 2);
        assert!(output.log_evidence.is_finite());
        assert!(output.iterations > 0);
        assert_eq!(events, output.iterations);
        assert!(output.log_evidence_error.is_finite());
        assert_eq!(output.live_points.len(), config.live_points);
    }
}
