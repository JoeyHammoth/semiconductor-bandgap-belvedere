//! Sliced Wasserstein-2 between empirical particle clouds.
//!
//! A random unit projection reduces the multivariate distance to a sorted 1D
//! problem. The returned distance is `sqrt(mean(W2_projection^2))`. The
//! bootstrap resamples the projection costs to expose the projection noise
//! floor; it does not estimate sampling error in the underlying posteriors.

#[derive(Debug, Clone, PartialEq)]
pub struct SlicedW2 {
    pub distance: f64,
    pub bootstrap_standard_error: f64,
    pub projections: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlicedError {
    Empty,
    DimensionMismatch,
    InvalidConfig,
    NonFinite,
}

impl std::fmt::Display for SlicedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "both particle clouds must be non-empty"),
            Self::DimensionMismatch => write!(f, "particle clouds have different dimensions"),
            Self::InvalidConfig => write!(f, "invalid projection or bootstrap count"),
            Self::NonFinite => write!(f, "particle coordinates must be finite"),
        }
    }
}

impl std::error::Error for SlicedError {}

/// Compute sliced Wasserstein-2 between `a` and `b`.
///
/// `projections` random directions are used; `bootstrap_samples` resamples
/// their squared costs. The clouds may have different sizes: each projected
/// distribution is compared through a common quantile grid.
pub fn sliced_wasserstein_2(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    projections: usize,
    bootstrap_samples: usize,
    seed: u64,
) -> Result<SlicedW2, SlicedError> {
    if a.is_empty() || b.is_empty() {
        return Err(SlicedError::Empty);
    }
    let dimension = a[0].len();
    if dimension == 0
        || b[0].len() != dimension
        || a.iter().any(|point| point.len() != dimension)
        || b.iter().any(|point| point.len() != dimension)
    {
        return Err(SlicedError::DimensionMismatch);
    }
    if a.iter()
        .chain(b)
        .flat_map(|point| point.iter())
        .any(|value| !value.is_finite())
    {
        return Err(SlicedError::NonFinite);
    }
    if projections == 0 || bootstrap_samples < 2 {
        return Err(SlicedError::InvalidConfig);
    }

    let mut rng = SplitMix64::new(seed);
    let mut squared_costs = Vec::with_capacity(projections);
    for _ in 0..projections {
        let direction = rng.unit_vector(dimension);
        let mut first = a
            .iter()
            .map(|point| dot(point, &direction))
            .collect::<Vec<_>>();
        let mut second = b
            .iter()
            .map(|point| dot(point, &direction))
            .collect::<Vec<_>>();
        first.sort_by(f64::total_cmp);
        second.sort_by(f64::total_cmp);
        squared_costs.push(one_dimensional_squared_w2(&first, &second));
    }

    let mean_squared = squared_costs.iter().sum::<f64>() / projections as f64;
    let distance = mean_squared.max(0.0).sqrt();
    let mut bootstrap = Vec::with_capacity(bootstrap_samples);
    for _ in 0..bootstrap_samples {
        let mut sum = 0.0;
        for _ in 0..projections {
            let index = (rng.next_u64() % projections as u64) as usize;
            sum += squared_costs[index];
        }
        bootstrap.push((sum / projections as f64).max(0.0).sqrt());
    }
    let bootstrap_mean = bootstrap.iter().sum::<f64>() / bootstrap.len() as f64;
    let bootstrap_variance = bootstrap
        .iter()
        .map(|value| {
            let difference = value - bootstrap_mean;
            difference * difference
        })
        .sum::<f64>()
        / (bootstrap.len() - 1) as f64;

    Ok(SlicedW2 {
        distance,
        bootstrap_standard_error: bootstrap_variance.max(0.0).sqrt(),
        projections,
    })
}

fn dot(point: &[f64], direction: &[f64]) -> f64 {
    point
        .iter()
        .zip(direction)
        .map(|(value, weight)| value * weight)
        .sum()
}

fn one_dimensional_squared_w2(first: &[f64], second: &[f64]) -> f64 {
    let count = first.len().max(second.len());
    let mut sum = 0.0;
    for index in 0..count {
        let probability = (index as f64 + 0.5) / count as f64;
        let left = quantile(first, probability);
        let right = quantile(second, probability);
        let difference = left - right;
        sum += difference * difference;
    }
    sum / count as f64
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let first = self.uniform();
        let second = self.uniform();
        (-2.0 * first.ln()).sqrt() * (2.0 * std::f64::consts::PI * second).cos()
    }

    fn unit_vector(&mut self, dimension: usize) -> Vec<f64> {
        loop {
            let mut vector = (0..dimension).map(|_| self.normal()).collect::<Vec<_>>();
            let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
            if norm.is_finite() && norm > 1.0e-12 {
                for value in &mut vector {
                    *value /= norm;
                }
                return vector;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_translation_has_the_expected_one_dimensional_distance() {
        let first = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let second = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let result = sliced_wasserstein_2(&first, &second, 32, 64, 7).expect("W2");
        assert!((result.distance - 1.0).abs() < 1.0e-12);
        assert!(result.bootstrap_standard_error.is_finite());
        assert!(result.bootstrap_standard_error < 1.0e-12);
    }

    #[test]
    fn identical_clouds_have_zero_distance_and_noise() {
        let cloud = vec![vec![0.0, 1.0], vec![1.0, -1.0], vec![0.5, 0.25]];
        let result = sliced_wasserstein_2(&cloud, &cloud, 64, 64, 0x5eed).expect("W2");
        assert!(result.distance < 1.0e-12);
        assert!(result.bootstrap_standard_error < 1.0e-12);
    }

    #[test]
    fn bootstrap_noise_is_finite_for_different_clouds() {
        let first = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let second = vec![
            vec![0.2, 0.1],
            vec![1.2, 0.1],
            vec![0.2, 1.1],
            vec![1.2, 1.1],
        ];
        let result = sliced_wasserstein_2(&first, &second, 64, 128, 11).expect("W2");
        assert!(result.distance > 0.0);
        assert!(result.bootstrap_standard_error.is_finite());
        assert!(result.bootstrap_standard_error >= 0.0);
    }

    #[test]
    fn dimensions_must_match() {
        let first = vec![vec![0.0, 1.0]];
        let second = vec![vec![0.0]];
        assert_eq!(
            sliced_wasserstein_2(&first, &second, 8, 8, 1),
            Err(SlicedError::DimensionMismatch)
        );
    }
}
