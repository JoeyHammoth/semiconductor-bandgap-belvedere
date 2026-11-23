//! Convergence diagnostics for multi-chain samplers.
//!
//! The gate is deliberately conservative: rank-normalized split R-hat, bulk
//! ESS, discretization energy error and divergence count must all pass before
//! the UI is allowed to render a converged state.

/// The diagnostics that gate the converged badge.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    pub rhat: Vec<f64>,
    pub ess_bulk: Vec<f64>,
    pub rhat_max: f64,
    pub ess_bulk_min: f64,
    pub energy_error_mean: f64,
    pub divergences: usize,
    pub passes: bool,
}

impl Diagnostics {
    pub const RHAT_LIMIT: f64 = 1.01;
    pub const ESS_LIMIT: f64 = 400.0;
    pub const ENERGY_ERROR_LIMIT: f64 = 0.1;

    pub fn new(
        rhat_max: f64,
        ess_bulk_min: f64,
        energy_error_mean: f64,
        divergences: usize,
    ) -> Self {
        let passes = rhat_max.is_finite()
            && rhat_max < Self::RHAT_LIMIT
            && ess_bulk_min.is_finite()
            && ess_bulk_min > Self::ESS_LIMIT
            && energy_error_mean.is_finite()
            && energy_error_mean < Self::ENERGY_ERROR_LIMIT
            && divergences == 0;
        Self {
            rhat: Vec::new(),
            ess_bulk: Vec::new(),
            rhat_max,
            ess_bulk_min,
            energy_error_mean,
            divergences,
            passes,
        }
    }

    pub fn from_parameters(
        rhat: Vec<f64>,
        ess_bulk: Vec<f64>,
        energy_error_mean: f64,
        divergences: usize,
    ) -> Self {
        let rhat_max = rhat.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ess_bulk_min = ess_bulk.iter().copied().fold(f64::INFINITY, f64::min);
        let mut diagnostics = Self::new(rhat_max, ess_bulk_min, energy_error_mean, divergences);
        diagnostics.rhat = rhat;
        diagnostics.ess_bulk = ess_bulk;
        diagnostics
    }
}

/// Rank-normalized split R-hat.
pub fn split_rhat(chains: &[Vec<f64>]) -> f64 {
    let split = split_chains(chains);
    if split.len() < 2 {
        return f64::INFINITY;
    }
    let normal = rank_normalize(&split);
    rhat_of(&normal)
}

/// Rank-normalized bulk effective sample size.
pub fn bulk_ess(chains: &[Vec<f64>]) -> f64 {
    let split = split_chains(chains);
    if split.len() < 2 {
        return 0.0;
    }
    let normal = rank_normalize(&split);
    ess_of(&normal)
}

fn split_chains(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut split = Vec::with_capacity(chains.len() * 2);
    for chain in chains {
        let half = chain.len() / 2;
        if half < 2 {
            continue;
        }
        split.push(chain[..half].to_vec());
        split.push(chain[half..half * 2].to_vec());
    }
    split
}

fn rank_normalize(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let total: usize = chains.iter().map(Vec::len).sum();
    if total == 0 {
        return chains.iter().map(|_| Vec::new()).collect();
    }
    let mut indexed = Vec::with_capacity(total);
    for (chain, values) in chains.iter().enumerate() {
        for (draw, value) in values.iter().enumerate() {
            indexed.push((*value, chain, draw));
        }
    }
    indexed.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut ranks = vec![0.0; total];
    let mut start = 0;
    while start < indexed.len() {
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].0 == indexed[start].0 {
            end += 1;
        }
        let average_rank = ((start + 1) as f64 + end as f64) / 2.0;
        for entry in &indexed[start..end] {
            ranks[entry.1 * chains[entry.1].len() + entry.2] = average_rank;
        }
        start = end;
    }

    let mut normal = Vec::with_capacity(chains.len());
    let denominator = total as f64 + 0.25;
    let mut offset = 0;
    for values in chains {
        let mut chain = Vec::with_capacity(values.len());
        for _ in 0..values.len() {
            let rank = ranks[offset];
            chain.push(inverse_normal_cdf((rank - 0.375) / denominator));
            offset += 1;
        }
        normal.push(chain);
    }
    normal
}

fn rhat_of(chains: &[Vec<f64>]) -> f64 {
    let n = chains[0].len() as f64;
    if n < 2.0 {
        return f64::INFINITY;
    }
    let means: Vec<f64> = chains.iter().map(|chain| mean(chain)).collect();
    let variances: Vec<f64> = chains
        .iter()
        .zip(&means)
        .map(|(chain, mean)| sample_variance(chain, *mean))
        .collect();
    let within = mean(&variances);
    let between = n * sample_variance(&means, mean(&means));
    if !within.is_finite() || within <= 0.0 {
        return if between == 0.0 { 1.0 } else { f64::INFINITY };
    }
    let variance_hat = ((n - 1.0) / n) * within + between / n;
    (variance_hat / within).sqrt()
}

fn ess_of(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len() as f64;
    let n = chains[0].len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let means: Vec<f64> = chains.iter().map(|chain| mean(chain)).collect();
    let variances: Vec<f64> = chains
        .iter()
        .zip(&means)
        .map(|(chain, mean)| sample_variance(chain, *mean))
        .collect();
    let within = mean(&variances);
    let between = n * sample_variance(&means, mean(&means));
    let variance_plus = ((n - 1.0) / n) * within + between / n;
    if !variance_plus.is_finite() || variance_plus <= 0.0 {
        return 0.0;
    }

    let max_lag = chains[0].len().saturating_sub(2);
    let mut rho = Vec::with_capacity(max_lag + 1);
    rho.push(1.0);
    for lag in 1..=max_lag {
        let autocovariance = chains
            .iter()
            .zip(&means)
            .map(|(chain, mean)| autocovariance(chain, *mean, lag))
            .sum::<f64>()
            / m;
        rho.push(1.0 - (within - autocovariance) / variance_plus);
    }

    let mut sum = 0.0;
    let mut lag = 1;
    while lag + 1 < rho.len() {
        let pair = rho[lag] + rho[lag + 1];
        if !pair.is_finite() || pair <= 0.0 {
            break;
        }
        sum += pair;
        lag += 2;
    }
    let ess = m * n / (1.0 + 2.0 * sum);
    if ess.is_finite() {
        ess.max(1.0).min(m * n)
    } else {
        0.0
    }
}

fn autocovariance(values: &[f64], mean: f64, lag: usize) -> f64 {
    let n = values.len();
    if lag >= n {
        return 0.0;
    }
    values[..n - lag]
        .iter()
        .zip(&values[lag..])
        .map(|(left, right)| (left - mean) * (right - mean))
        .sum::<f64>()
        / n as f64
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_variance(values: &[f64], mean: f64) -> f64 {
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() as f64 - 1.0)
}

/// Acklam's inverse normal CDF approximation, accurate to about 1e-9.
fn inverse_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let tail = |q: f64| {
        let r = (-2.0 * q.ln()).sqrt();
        let num = ((((C[0] * r + C[1]) * r + C[2]) * r + C[3]) * r + C[4]) * r + C[5];
        let den = (((D[0] * r + D[1]) * r + D[2]) * r + D[3]) * r + 1.0;
        num / den
    };
    let central = |q: f64| {
        let r = q * q;
        let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num / den
    };
    if p < 0.024_25 {
        tail(p)
    } else if p <= 1.0 - 0.024_25 {
        central(p - 0.5)
    } else {
        -tail(1.0 - p)
    }
}
