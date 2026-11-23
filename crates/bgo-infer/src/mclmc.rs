//! Metropolis-adjusted isokinetic MCLMC.
//!
//! The proposal integrates the unit-velocity microcanonical dynamics with the
//! five-stage McLachlan splitting. The accumulated energy error is the
//! Metropolis-Hastings statistic from the MAMS construction, so the chain is
//! asymptotically unbiased rather than merely low-bias.

use bgo_ad::{Dual, Scalar};

use crate::diagnostics::Diagnostics;
use crate::model::{Case, Model, ModelError};

const B1: f64 = 0.193_183_327_503_783_6;
const COEFFICIENTS: [f64; 5] = [B1, 0.5, 1.0 - 2.0 * B1, 0.5, B1];

#[derive(Debug, Clone, PartialEq)]
pub struct MclmcConfig {
    pub chains: usize,
    pub warmup: usize,
    pub draws: usize,
    pub seed: u64,
    pub target_accept: f64,
    pub initial_step: f64,
    pub mean_trajectory_steps: f64,
    pub energy_error_limit: f64,
}

impl Default for MclmcConfig {
    fn default() -> Self {
        Self {
            chains: 4,
            warmup: 5_000,
            draws: 4_000,
            seed: 0x6d63_6c6d,
            target_accept: 0.995,
            initial_step: 0.4,
            mean_trajectory_steps: 16.0,
            energy_error_limit: Diagnostics::ENERGY_ERROR_LIMIT,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MclmcOutput {
    pub parameter_names: Vec<&'static str>,
    /// chain -> draw -> reported parameters.
    pub chains: Vec<Vec<Vec<f64>>>,
    pub diagnostics: Diagnostics,
    pub acceptance_rate: f64,
    pub energy_error_mean: f64,
    pub step_size: f64,
}

impl MclmcOutput {
    /// Parameter-major draws, matching the reference summary layout.
    pub fn by_parameter(&self) -> Vec<Vec<f64>> {
        let parameters = self.parameter_names.len();
        let mut values = vec![Vec::new(); parameters];
        for chain in &self.chains {
            for draw in chain {
                for (index, value) in draw.iter().enumerate() {
                    values[index].push(*value);
                }
            }
        }
        values
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MclmcError {
    Model(ModelError),
    NoChains,
    TooFewDraws,
    InvalidConfig,
}

impl std::fmt::Display for MclmcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(error) => error.fmt(f),
            Self::NoChains => write!(f, "at least one chain is required"),
            Self::TooFewDraws => write!(f, "at least two draws per chain are required"),
            Self::InvalidConfig => write!(f, "invalid sampler configuration"),
        }
    }
}

impl std::error::Error for MclmcError {}

impl From<ModelError> for MclmcError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

pub fn run_mclmc(case: &Case, config: &MclmcConfig) -> Result<MclmcOutput, MclmcError> {
    if config.chains == 0 {
        return Err(MclmcError::NoChains);
    }
    if config.draws < 2 || config.warmup == 0 {
        return Err(MclmcError::TooFewDraws);
    }
    if !config.target_accept.is_finite()
        || !(0.0..1.0).contains(&config.target_accept)
        || !config.initial_step.is_finite()
        || config.initial_step <= 0.0
        || !config.mean_trajectory_steps.is_finite()
        || config.mean_trajectory_steps < 1.0
        || !config.energy_error_limit.is_finite()
        || config.energy_error_limit <= 0.0
    {
        return Err(MclmcError::InvalidConfig);
    }

    let mut preconditioner = Preconditioner::build(case);
    if matches!(case.model, Model::M2 | Model::M3 | Model::M4) {
        let pilot_config = MclmcConfig {
            warmup: (config.warmup / 2).max(1),
            draws: (config.draws / 2).max(2),
            ..config.clone()
        };
        let mut pilot_rng = SplitMix64::new(config.seed ^ 0x9e37_79b9_7f4a_7c15);
        let mut pilot = ChainSampler::new(case, &preconditioner, &pilot_config, &mut pilot_rng);
        let (_, latent, _, _, _, _) = pilot.run();
        if latent.len() >= 10 {
            preconditioner = Preconditioner::from_samples(&latent);
        }
    }
    let mut chains = Vec::with_capacity(config.chains);
    let mut all_acceptance = Vec::new();
    let mut all_energy = Vec::new();
    let mut divergences = 0usize;
    let mut final_step = config.initial_step;

    for chain in 0..config.chains {
        let mut rng = SplitMix64::new(config.seed.wrapping_add(chain as u64 * 0x9e37_79b9));
        let mut sampler = ChainSampler::new(case, &preconditioner, config, &mut rng);
        let (draws, _, acceptance, energy, chain_divergences, step_size) = sampler.run();
        chains.push(draws);
        all_acceptance.push(acceptance);
        all_energy.push(energy);
        divergences += chain_divergences;
        final_step = step_size;
    }

    let parameter_count = case.parameter_names().len();
    let mut rhat = Vec::with_capacity(parameter_count);
    let mut ess = Vec::with_capacity(parameter_count);
    for parameter in 0..parameter_count {
        let values = chains
            .iter()
            .map(|chain| chain.iter().map(|draw| draw[parameter]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        rhat.push(crate::diagnostics::split_rhat(&values));
        ess.push(crate::diagnostics::bulk_ess(&values));
    }
    let energy_error_mean = all_energy.iter().sum::<f64>() / all_energy.len() as f64;
    let acceptance_rate = all_acceptance.iter().sum::<f64>() / all_acceptance.len() as f64;

    let mut diagnostics = Diagnostics::from_parameters(rhat, ess, energy_error_mean, divergences);
    if energy_error_mean >= config.energy_error_limit {
        diagnostics.passes = false;
    }

    Ok(MclmcOutput {
        parameter_names: case.parameter_names().to_vec(),
        chains,
        diagnostics,
        acceptance_rate,
        energy_error_mean,
        step_size: final_step,
    })
}

type ChainRun = (Vec<Vec<f64>>, Vec<Vec<f64>>, f64, f64, usize, f64);

struct ChainSampler<'a> {
    case: &'a Case,
    preconditioner: &'a Preconditioner,
    config: &'a MclmcConfig,
    rng: &'a mut SplitMix64,
    z: Vec<f64>,
    logp: f64,
    gradient: Vec<f64>,
    momentum: Vec<f64>,
    normalized: Vec<f64>,
    raw: Vec<f64>,
    step_size: f64,
    dual_averaging: DualAveraging,
    acceptance_sum: f64,
    acceptance_count: usize,
    energy_sum: f64,
    divergences: usize,
}

impl<'a> ChainSampler<'a> {
    fn new(
        case: &'a Case,
        preconditioner: &'a Preconditioner,
        config: &'a MclmcConfig,
        rng: &'a mut SplitMix64,
    ) -> Self {
        let dimension = case.latent_len();
        let z = vec![0.0; dimension];
        let (logp, gradient) = preconditioner.logp_and_grad(case, &z);
        let momentum = rng.unit_vector(dimension);
        Self {
            case,
            preconditioner,
            config,
            rng,
            z,
            logp,
            gradient,
            momentum,
            normalized: vec![0.0; dimension],
            raw: vec![0.0; dimension],
            step_size: config.initial_step,
            dual_averaging: DualAveraging::new(config.initial_step, config.target_accept),
            acceptance_sum: 0.0,
            acceptance_count: 0,
            energy_sum: 0.0,
            divergences: 0,
        }
    }

    fn run(&mut self) -> ChainRun {
        for _ in 0..self.config.warmup {
            self.proposal(true);
        }
        self.step_size = self.dual_averaging.final_step();
        self.acceptance_sum = 0.0;
        self.acceptance_count = 0;
        self.energy_sum = 0.0;
        self.divergences = 0;
        let mut draws = Vec::with_capacity(self.config.draws);
        let mut latent_draws = Vec::with_capacity(self.config.draws);
        for _ in 0..self.config.draws {
            self.proposal(false);
            let theta = self.theta();
            latent_draws.push(theta.clone());
            draws.push(self.case.reported(&theta));
        }
        let acceptance = self.acceptance_sum / self.acceptance_count.max(1) as f64;
        let energy = self.energy_sum / self.acceptance_count.max(1) as f64;
        (
            draws,
            latent_draws,
            acceptance,
            energy,
            self.divergences,
            self.step_size,
        )
    }

    fn proposal(&mut self, warmup: bool) {
        let steps =
            1 + (2.0 * self.rng.uniform() * self.config.mean_trajectory_steps).floor() as usize;
        let mut z = self.z.clone();
        let mut logp = self.logp;
        let mut gradient = self.gradient.clone();
        let mut momentum = self.momentum.clone();
        let mut energy_error = 0.0;

        for _ in 0..steps {
            for (index, coefficient) in COEFFICIENTS.iter().enumerate() {
                if index % 2 == 0 {
                    energy_error += momentum_update(
                        &mut momentum,
                        &gradient,
                        self.step_size,
                        *coefficient,
                        &mut self.normalized,
                        &mut self.raw,
                    );
                } else {
                    for (value, velocity) in z.iter_mut().zip(&momentum) {
                        *value += self.step_size * coefficient * velocity;
                    }
                    let (next_logp, next_gradient) =
                        self.preconditioner.logp_and_grad(self.case, &z);
                    logp = next_logp;
                    gradient = next_gradient;
                }
            }
        }

        let delta = energy_error - (logp - self.logp);
        let valid = delta.is_finite() && logp.is_finite() && self.logp.is_finite();
        let accepted = if !valid || delta > 100.0 {
            self.divergences += 1;
            false
        } else if delta <= -700.0 {
            true
        } else {
            self.rng.uniform() < (-delta).exp().min(1.0)
        };

        let acceptance = if accepted { 1.0 } else { 0.0 };
        self.acceptance_sum += acceptance;
        self.acceptance_count += 1;
        self.energy_sum += delta.abs();

        if accepted {
            self.z = z;
            self.logp = logp;
            self.gradient = gradient;
        }
        self.momentum = self.rng.unit_vector(self.case.latent_len());
        if warmup {
            self.step_size = self.dual_averaging.update(acceptance).clamp(1e-10, 10.0);
        }
    }

    fn theta(&self) -> Vec<f64> {
        self.preconditioner.theta(&self.z)
    }
}

struct DualAveraging {
    target: f64,
    mu: f64,
    log_step: f64,
    log_step_bar: f64,
    hbar: f64,
    t: f64,
    gamma: f64,
    t0: f64,
    kappa: f64,
}

impl DualAveraging {
    fn new(initial_step: f64, target: f64) -> Self {
        let log_step = initial_step.ln();
        Self {
            target,
            mu: (10.0 * initial_step).ln(),
            log_step,
            log_step_bar: log_step,
            hbar: 0.0,
            t: 0.0,
            gamma: 0.05,
            t0: 10.0,
            kappa: 0.75,
        }
    }

    fn update(&mut self, acceptance: f64) -> f64 {
        self.t += 1.0;
        let weight = 1.0 / self.t;
        self.hbar = (1.0 - weight) * self.hbar + weight * (self.target - acceptance);
        self.log_step = self.mu - (self.t.sqrt() / self.gamma) * self.hbar;
        let eta = 1.0 / (self.t0 + self.t).powf(self.kappa);
        self.log_step_bar = eta * self.log_step + (1.0 - eta) * self.log_step_bar;
        self.log_step.exp()
    }

    fn final_step(&self) -> f64 {
        self.log_step_bar.exp()
    }
}

struct Preconditioner {
    center: Vec<f64>,
    transform: Vec<Vec<f64>>,
}

fn least_squares_start(case: &Case) -> Vec<f64> {
    let (mut theta, include_priors, fixed_rs) = if case.model == Model::M4 {
        (case.prior_center(), false, Some(8.0))
    } else {
        (case.prior_center(), true, None)
    };
    let bounds = latent_bounds(case);
    let scales = latent_scales(case);
    let mut lambda = 1e-3;
    let mut best_cost = residual_cost(case, &theta, include_priors, fixed_rs);

    for _ in 0..80 {
        let residual = residual_vector(case, &theta, include_priors, fixed_rs);
        if residual.iter().any(|value| !value.is_finite()) {
            break;
        }
        let jacobian = residual_jacobian(case, &theta, &scales, include_priors, fixed_rs);
        let mut normal = vec![vec![0.0; theta.len()]; theta.len()];
        let mut rhs = vec![0.0; theta.len()];
        for row in 0..theta.len() {
            for column in 0..theta.len() {
                normal[row][column] = (0..residual.len())
                    .map(|point| jacobian[point][row] * jacobian[point][column])
                    .sum();
            }
            rhs[row] = -(0..residual.len())
                .map(|point| jacobian[point][row] * residual[point])
                .sum::<f64>();
            normal[row][row] += lambda * normal[row][row].abs().max(1e-8);
        }
        let Some(delta) = solve_linear(&normal, &rhs) else {
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
            continue;
        };
        let mut candidate = theta.clone();
        for index in 0..candidate.len() {
            candidate[index] =
                (candidate[index] + delta[index]).clamp(bounds[index].0, bounds[index].1);
        }
        let candidate_cost = residual_cost(case, &candidate, include_priors, fixed_rs);
        if candidate_cost.is_finite() && candidate_cost < best_cost {
            theta = candidate;
            best_cost = candidate_cost;
            lambda = (lambda * 0.5).max(1e-12);
        } else {
            lambda *= 4.0;
            if lambda > 1e12 {
                break;
            }
        }
    }
    theta
}

fn residual_vector(
    case: &Case,
    theta: &[f64],
    include_priors: bool,
    fixed_rs: Option<f64>,
) -> Vec<f64> {
    let mut residual = case
        .predicted_voltage(theta)
        .iter()
        .zip(&case.voltage_v)
        .map(|(predicted, observed)| (predicted - observed) / case.sigma_v)
        .collect::<Vec<_>>();
    if include_priors {
        residual.extend(case.prior_residual(theta));
    }
    if let Some(rs) = fixed_rs {
        residual.push((case.reported(theta)[5] - rs) / 1.0e-3);
    }
    residual
}

fn residual_cost(case: &Case, theta: &[f64], include_priors: bool, fixed_rs: Option<f64>) -> f64 {
    residual_vector(case, theta, include_priors, fixed_rs)
        .iter()
        .map(|residual| 0.5 * residual * residual)
        .sum()
}

fn residual_jacobian(
    case: &Case,
    theta: &[f64],
    scales: &[f64],
    include_priors: bool,
    fixed_rs: Option<f64>,
) -> Vec<Vec<f64>> {
    let mut jacobian =
        vec![vec![0.0; theta.len()]; residual_vector(case, theta, include_priors, fixed_rs).len()];
    for parameter in 0..theta.len() {
        let step = 1e-4 * scales[parameter].max(1.0e-8);
        let mut plus = theta.to_vec();
        let mut minus = theta.to_vec();
        plus[parameter] += step;
        minus[parameter] -= step;
        let plus_residual = residual_vector(case, &plus, include_priors, fixed_rs);
        let minus_residual = residual_vector(case, &minus, include_priors, fixed_rs);
        for point in 0..jacobian.len() {
            jacobian[point][parameter] =
                (plus_residual[point] - minus_residual[point]) / (2.0 * step);
        }
    }
    jacobian
}

fn latent_bounds(case: &Case) -> Vec<(f64, f64)> {
    match case.model {
        crate::model::Model::M0 => vec![(0.05, 3.5), (-20.0, 40.0)],
        crate::model::Model::M1 => vec![(0.05, 3.5), (-20.0, 40.0), (1.5, 5.0)],
        crate::model::Model::M2 => vec![
            (-1.0, 4.0),
            (-1.0, 4.0),
            (-20.0, 40.0),
            (100.0, 3000.0),
            (1.5, 5.0),
        ],
        crate::model::Model::M3 => vec![
            (-1.0, 4.0),
            (-1.0, 4.0),
            (-20.0, 40.0),
            (100.0, 3000.0),
            (0.7, 1.6),
        ],
        crate::model::Model::M4 => vec![
            (-1.0, 4.0),
            (-1.0, 4.0),
            (-20.0, 40.0),
            (100.0, 3000.0),
            (0.7, 1.6),
            (-6.0, 6.0),
        ],
    }
}

fn latent_scales(case: &Case) -> Vec<f64> {
    match case.model {
        crate::model::Model::M0 => vec![0.35, 8.0],
        crate::model::Model::M1 => vec![0.35, 8.0, 1.0],
        crate::model::Model::M2 => vec![0.01, 0.01, 8.0, 150.0, 1.0],
        crate::model::Model::M3 => vec![0.01, 0.01, 8.0, 150.0, 0.15],
        crate::model::Model::M4 => vec![0.01, 0.01, 8.0, 150.0, 0.15, 1.0],
    }
}

fn solve_linear(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    let mut augmented = vec![vec![0.0; n + 1]; n];
    for row in 0..n {
        augmented[row][..n].copy_from_slice(&matrix[row]);
        augmented[row][n] = rhs[row];
    }
    for column in 0..n {
        let pivot = (column..n).max_by(|left, right| {
            augmented[*left][column]
                .abs()
                .total_cmp(&augmented[*right][column].abs())
        })?;
        if augmented[pivot][column].abs() < 1e-14 {
            return None;
        }
        augmented.swap(column, pivot);
        let divisor = augmented[column][column];
        for value in &mut augmented[column][column..] {
            *value /= divisor;
        }
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = augmented[row][column];
            for index in column..=n {
                augmented[row][index] -= factor * augmented[column][index];
            }
        }
    }
    Some((0..n).map(|row| augmented[row][n]).collect())
}

impl Preconditioner {
    fn build(case: &Case) -> Self {
        let center = least_squares_start(case);
        let n = center.len();
        let jacobian = residual_jacobian(case, &center, &latent_scales(case), true, None);
        let mut precision = vec![vec![0.0; n]; n];
        for row in 0..n {
            for column in 0..n {
                precision[row][column] = jacobian
                    .iter()
                    .map(|point| point[row] * point[column])
                    .sum();
            }
        }
        let scale = precision
            .iter()
            .enumerate()
            .map(|(index, row)| row[index].abs())
            .fold(1.0_f64, f64::max);
        let mut jitter = 1e-9 * scale;

        for _ in 0..8 {
            if let Some(cholesky) = cholesky(&precision) {
                let inverse = invert_lower(&cholesky);
                let mut transform = vec![vec![0.0; n]; n];
                for row in 0..n {
                    for column in 0..n {
                        transform[row][column] = inverse[column][row];
                    }
                }
                return Self { center, transform };
            }
            for (index, row) in precision.iter_mut().enumerate().take(n) {
                row[index] += jitter;
            }
            jitter *= 10.0;
        }

        let mut transform = vec![vec![0.0; n]; n];
        for index in 0..n {
            let diagonal = precision[index][index].abs().max(1e-12);
            transform[index][index] = 1.0 / diagonal.sqrt();
        }
        Self { center, transform }
    }

    fn from_samples(samples: &[Vec<f64>]) -> Self {
        let n = samples[0].len();
        let count = samples.len() as f64;
        let center = (0..n)
            .map(|index| samples.iter().map(|sample| sample[index]).sum::<f64>() / count)
            .collect::<Vec<_>>();
        let mut covariance = vec![vec![0.0; n]; n];
        for row in 0..n {
            for column in 0..n {
                covariance[row][column] = samples
                    .iter()
                    .map(|sample| (sample[row] - center[row]) * (sample[column] - center[column]))
                    .sum::<f64>()
                    / (count - 1.0).max(1.0);
            }
        }
        let scale = covariance
            .iter()
            .enumerate()
            .map(|(index, row)| row[index].abs())
            .fold(1.0_f64, f64::max);
        let mut jitter = 1e-9 * scale;
        for _ in 0..8 {
            if let Some(lower) = cholesky(&covariance) {
                return Self {
                    center,
                    transform: lower,
                };
            }
            for (index, row) in covariance.iter_mut().enumerate().take(n) {
                row[index] += jitter;
            }
            jitter *= 10.0;
        }
        let mut transform = vec![vec![0.0; n]; n];
        for index in 0..n {
            transform[index][index] = covariance[index][index].abs().sqrt().max(1e-12);
        }
        Self { center, transform }
    }

    fn theta(&self, z: &[f64]) -> Vec<f64> {
        self.center
            .iter()
            .enumerate()
            .map(|(row, center)| {
                center
                    + self.transform[row]
                        .iter()
                        .zip(z)
                        .map(|(coefficient, value)| coefficient * value)
                        .sum::<f64>()
            })
            .collect()
    }

    fn logp_and_grad(&self, case: &Case, z: &[f64]) -> (f64, Vec<f64>) {
        let mut value = f64::NAN;
        let mut gradient = vec![0.0; z.len()];
        for index in 0..z.len() {
            let mut dual = z.iter().copied().map(Dual::constant).collect::<Vec<_>>();
            dual[index] = Dual::variable(z[index]);
            let evaluation = self.logp(case, &dual);
            if index == 0 {
                value = evaluation.value();
            }
            gradient[index] = evaluation.du;
        }
        (value, gradient)
    }

    fn logp<T: Scalar>(&self, case: &Case, z: &[T]) -> T {
        let theta = (0..self.center.len())
            .map(|row| {
                let mut value = T::from_f64(self.center[row]);
                for (coefficient, coordinate) in self.transform[row].iter().zip(z) {
                    value = value + T::from_f64(*coefficient) * *coordinate;
                }
                value
            })
            .collect::<Vec<_>>();
        case.logp(&theta)
    }
}

fn momentum_update(
    momentum: &mut [f64],
    gradient: &[f64],
    step_size: f64,
    coefficient: f64,
    normalized: &mut [f64],
    raw: &mut [f64],
) -> f64 {
    let dimension = momentum.len();
    let norm = gradient
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm < 1e-300 {
        return 0.0;
    }
    for (unit, value) in normalized.iter_mut().zip(gradient) {
        *unit = value / norm;
    }
    let projection = momentum
        .iter()
        .zip(normalized.iter())
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let delta = step_size * coefficient * norm / (dimension - 1) as f64;
    let zeta = (-delta).exp();
    for index in 0..dimension {
        raw[index] = normalized[index] * (1.0 - zeta) * (1.0 + zeta + projection * (1.0 - zeta))
            + 2.0 * zeta * momentum[index];
    }
    let raw_norm = raw.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !raw_norm.is_finite() || raw_norm < 1e-300 {
        return 0.0;
    }
    let s_plus = 0.5
        * momentum
            .iter()
            .zip(normalized.iter())
            .map(|(left, right)| (left + right) * (left + right))
            .sum::<f64>();
    let s_minus = 0.5
        * momentum
            .iter()
            .zip(normalized.iter())
            .map(|(left, right)| (left - right) * (left - right))
            .sum::<f64>();
    for (value, raw) in momentum.iter_mut().zip(raw.iter()) {
        *value = raw / raw_norm;
    }
    let log_a = if zeta * zeta <= 1.0 {
        (s_plus * (1.0 - zeta * zeta) + 2.0 * zeta * zeta)
            .max(1e-300)
            .ln()
    } else {
        (s_minus * (zeta * zeta - 1.0) + 2.0).max(1e-300).ln()
    };
    (delta - std::f64::consts::LN_2 + log_a) * (dimension - 1) as f64
}

fn cholesky(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut lower = vec![vec![0.0; n]; n];
    for row in 0..n {
        for column in 0..=row {
            let mut sum = matrix[row][column];
            for k in 0..column {
                sum -= lower[row][k] * lower[column][k];
            }
            if row == column {
                if !sum.is_finite() || sum <= 1e-18 {
                    return None;
                }
                lower[row][column] = sum.sqrt();
            } else {
                lower[row][column] = sum / lower[column][column];
            }
        }
    }
    Some(lower)
}

fn invert_lower(lower: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = lower.len();
    let mut inverse = vec![vec![0.0; n]; n];
    for row in 0..n {
        inverse[row][row] = 1.0 / lower[row][row];
        for column in 0..row {
            let sum = (column..row)
                .map(|k| lower[row][k] * inverse[k][column])
                .sum::<f64>();
            inverse[row][column] = -sum / lower[row][row];
        }
    }
    inverse
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn unit_vector(&mut self, dimension: usize) -> Vec<f64> {
        let mut vector = (0..dimension).map(|_| self.normal()).collect::<Vec<_>>();
        let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        for value in &mut vector {
            *value /= norm;
        }
        vector
    }
}
