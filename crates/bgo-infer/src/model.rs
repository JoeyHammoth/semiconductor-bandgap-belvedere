//! The M0-M4 synthetic model contract used by the browser sampler.
//!
//! The equations and priors mirror `train/src/bgo_train/reference.py`; the
//! observations come from the committed synthetic references. No measured data
//! enters this module.

use bgo_ad::{Dual, HyperDual, Scalar};

const BOLTZMANN: f64 = 1.380_649e-23;
const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
const REFERENCE_TEMPERATURE_K: f64 = 300.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    M0,
    M1,
    M2,
    M3,
    M4,
}

impl Model {
    pub fn parameter_names(self) -> &'static [&'static str] {
        match self {
            Self::M0 => &["eg_ev", "ln_b"],
            Self::M1 => &["eg_ev", "ln_b", "eta"],
            Self::M2 => &["eg0_ev", "ln_b", "eta", "alpha_ev_per_k", "beta_k"],
            Self::M3 => &["eg0_ev", "ln_b", "alpha_ev_per_k", "beta_k", "ideality"],
            Self::M4 => &[
                "eg0_ev",
                "ln_b",
                "alpha_ev_per_k",
                "beta_k",
                "ideality",
                "series_resistance_ohm",
            ],
        }
    }

    pub fn latent_len(self) -> usize {
        self.parameter_names().len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    pub model: Model,
    pub temperature_k: Vec<f64>,
    pub voltage_v: Vec<f64>,
    pub current_a: f64,
    pub sigma_v: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    EmptyData,
    LengthMismatch,
    NonFinite,
    NonPositiveCurrent,
    NonPositiveSigma,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "at least two observations are required"),
            Self::LengthMismatch => write!(f, "temperature and voltage lengths differ"),
            Self::NonFinite => write!(f, "observations must be finite"),
            Self::NonPositiveCurrent => write!(f, "current must be positive"),
            Self::NonPositiveSigma => write!(f, "likelihood sigma must be positive"),
        }
    }
}

impl std::error::Error for ModelError {}

impl Case {
    pub fn new(
        model: Model,
        temperature_k: Vec<f64>,
        voltage_v: Vec<f64>,
        current_a: f64,
        sigma_v: f64,
    ) -> Result<Self, ModelError> {
        if temperature_k.len() < 2 {
            return Err(ModelError::EmptyData);
        }
        if temperature_k.len() != voltage_v.len() {
            return Err(ModelError::LengthMismatch);
        }
        if !temperature_k
            .iter()
            .chain(&voltage_v)
            .all(|value| value.is_finite())
        {
            return Err(ModelError::NonFinite);
        }
        if !current_a.is_finite() || current_a <= 0.0 {
            return Err(ModelError::NonPositiveCurrent);
        }
        if !sigma_v.is_finite() || sigma_v <= 0.0 {
            return Err(ModelError::NonPositiveSigma);
        }
        Ok(Self {
            model,
            temperature_k,
            voltage_v,
            current_a,
            sigma_v,
        })
    }

    pub fn parameter_names(&self) -> &'static [&'static str] {
        self.model.parameter_names()
    }

    pub fn latent_len(&self) -> usize {
        self.model.latent_len()
    }

    /// Prior-centred starting point. It deliberately does not use the truth.
    pub fn prior_center(&self) -> Vec<f64> {
        match self.model {
            Model::M0 => vec![0.9, 8.0],
            Model::M1 => vec![0.9, 8.0, 3.0],
            Model::M2 => {
                let physical = [0.9, 8.0, 7.021e-4, 1108.0, 3.0];
                let (low, high) = self.gap_anchors();
                let low_gap = low * low / (low + physical[3]);
                let high_gap = high * high / (high + physical[3]);
                vec![
                    physical[0] - physical[2] * low_gap,
                    physical[0] - physical[2] * high_gap,
                    physical[1],
                    physical[3],
                    physical[4],
                ]
            }
            Model::M3 => {
                let physical = [0.9, 8.0, 7.021e-4, 1108.0, 1.0];
                let (low, high) = self.gap_anchors();
                let low_gap = low * low / (low + physical[3]);
                let high_gap = high * high / (high + physical[3]);
                vec![
                    physical[0] - physical[2] * low_gap,
                    physical[0] - physical[2] * high_gap,
                    physical[1],
                    physical[3],
                    physical[4],
                ]
            }
            Model::M4 => {
                let physical = [0.9, 8.0, 7.021e-4, 1108.0, 1.0, 8.0];
                let (low, high) = self.gap_anchors();
                let low_gap = low * low / (low + physical[3]);
                let high_gap = high * high / (high + physical[3]);
                // M4 samples eg0 + I*Rs instead of eg0. The two are nearly
                // degenerate because I*Rs is a constant voltage offset;
                // this parameterization makes the ridge axis-aligned.
                let effective_gap = physical[0] + self.current_a * physical[5];
                vec![
                    effective_gap - physical[2] * low_gap,
                    effective_gap - physical[2] * high_gap,
                    physical[1],
                    physical[3],
                    physical[4],
                    normal_from_half_normal(physical[5]),
                ]
            }
        }
    }

    fn gap_anchors(&self) -> (f64, f64) {
        let first = self.temperature_k.first().copied().unwrap_or(220.0);
        let last = self.temperature_k.last().copied().unwrap_or(383.15);
        (first.min(last), first.max(last))
    }

    fn gap_jacobian<T: Scalar>(&self, beta: T) -> T {
        let (low, high) = self.gap_anchors();
        let low_gap = T::from_f64(low * low) / (T::from_f64(low) + beta);
        let high_gap = T::from_f64(high * high) / (T::from_f64(high) + beta);
        high_gap - low_gap
    }

    /// The Varshni models' latent coordinates are the gap at the two data
    /// endpoints plus `beta`, rather than `(eg0, alpha, beta)`. The likelihood
    /// is nearly linear in those endpoint gaps, which removes the curved MAP
    /// ridge. M3/M4 also sample `n * gap` and `n * ln_b_ref`, because ideality
    /// multiplies both terms and otherwise leaves a curved multiplicative
    /// ridge. M4 samples `ln Rs` so proposals do not collapse at the half-normal
    /// boundary.
    fn physical_theta<T: Scalar>(&self, theta: &[T]) -> Vec<T> {
        if !matches!(self.model, Model::M2 | Model::M3 | Model::M4) {
            return theta.to_vec();
        }
        let (low, high) = self.gap_anchors();
        let beta = theta[3];
        let low_gap = T::from_f64(low * low) / (T::from_f64(low) + beta);
        let high_gap = T::from_f64(high * high) / (T::from_f64(high) + beta);
        let denominator = high_gap - low_gap;
        let denominator_value = denominator.value();
        if !denominator_value.is_finite() || denominator_value <= 0.0 {
            return vec![T::from_f64(f64::NAN); theta.len()];
        }

        if matches!(self.model, Model::M3 | Model::M4) {
            let ideality = theta[4];
            if !ideality.value().is_finite() || ideality.value() <= 0.0 {
                return vec![T::from_f64(f64::NAN); theta.len()];
            }
            let alpha = (theta[0] - theta[1]) / (ideality * denominator);
            let effective_gap = theta[0] / ideality + alpha * low_gap;
            let ln_b_ref = theta[2] / ideality;
            let mut physical = vec![effective_gap, ln_b_ref, alpha, beta, ideality];
            if self.model == Model::M4 {
                let series_resistance = half_normal_from_normal(theta[5]);
                physical[0] = effective_gap - T::from_f64(self.current_a) * series_resistance;
                physical.push(series_resistance);
            }
            return physical;
        }

        let alpha = (theta[0] - theta[1]) / denominator;
        let eg0 = theta[0] + alpha * low_gap;
        let mut physical = vec![eg0, theta[2], alpha, beta];
        physical.extend_from_slice(&theta[4..]);
        physical
    }

    pub(crate) fn prior_residual(&self, theta: &[f64]) -> Vec<f64> {
        let physical = self.physical_theta(theta);
        match self.model {
            Model::M0 => vec![(physical[0] - 0.9) / 0.35, (physical[1] - 8.0) / 8.0],
            Model::M1 => vec![
                (physical[0] - 0.9) / 0.35,
                (physical[1] - 8.0) / 8.0,
                (physical[2] - 3.0) / 1.0,
            ],
            Model::M2 => vec![
                (physical[0] - 0.9) / 0.35,
                (physical[1] - 8.0) / 8.0,
                (physical[2] - 7.021e-4) / 1.0e-4,
                (physical[3] - 1108.0) / 150.0,
                (physical[4] - 3.0) / 1.0,
            ],
            Model::M3 => vec![
                (physical[0] - 0.9) / 0.35,
                (physical[1] - 8.0) / 8.0,
                (physical[2] - 7.021e-4) / 1.0e-4,
                (physical[3] - 1108.0) / 150.0,
                (physical[4] - 1.0) / 0.15,
            ],
            Model::M4 => vec![
                (physical[0] - 0.9) / 0.35,
                (physical[1] - 8.0) / 8.0,
                (physical[2] - 7.021e-4) / 1.0e-4,
                (physical[3] - 1108.0) / 150.0,
                (physical[4] - 1.0) / 0.15,
                theta[5],
            ],
        }
    }

    /// Convert reported parameters back to the latent coordinates used by the
    /// sampler. This is the inverse of [`Case::reported`] inside the model's
    /// bounds; it exists so round trips and fixed-point target checks do not
    /// duplicate the transform in tests.
    pub fn latent_from_reported(&self, reported: &[f64]) -> Vec<f64> {
        if reported.len() != self.latent_len() {
            return vec![f64::NAN; self.latent_len()];
        }
        let (low, high) = self.gap_anchors();
        match self.model {
            Model::M0 => vec![reported[0], reported[1]],
            Model::M1 => vec![
                reported[0],
                reported[1] + reported[2] * REFERENCE_TEMPERATURE_K.ln(),
                reported[2],
            ],
            Model::M2 => {
                let beta = reported[4];
                let low_gap = low * low / (low + beta);
                let high_gap = high * high / (high + beta);
                let alpha = reported[3];
                vec![
                    reported[0] - alpha * low_gap,
                    reported[0] - alpha * high_gap,
                    reported[1] + reported[2] * REFERENCE_TEMPERATURE_K.ln(),
                    beta,
                    reported[2],
                ]
            }
            Model::M3 | Model::M4 => {
                let ideality = reported[4];
                let beta = reported[3];
                let low_gap = low * low / (low + beta);
                let high_gap = high * high / (high + beta);
                let alpha = reported[2];
                let effective_gap = if self.model == Model::M4 {
                    reported[0] + self.current_a * reported[5]
                } else {
                    reported[0]
                };
                let mut theta = vec![
                    ideality * (effective_gap - alpha * low_gap),
                    ideality * (effective_gap - alpha * high_gap),
                    ideality * (reported[1] + 3.0 * REFERENCE_TEMPERATURE_K.ln()),
                    beta,
                    ideality,
                ];
                if self.model == Model::M4 {
                    theta.push(normal_from_half_normal(reported[5]));
                }
                theta
            }
        }
    }

    /// Convert latent coordinates to the parameter names used by the reference.
    pub fn reported(&self, theta: &[f64]) -> Vec<f64> {
        let physical = self.physical_theta(theta);
        match self.model {
            Model::M0 => vec![physical[0], physical[1]],
            Model::M1 => {
                let eta = physical[2];
                vec![
                    physical[0],
                    physical[1] - eta * REFERENCE_TEMPERATURE_K.ln(),
                    eta,
                ]
            }
            Model::M2 => {
                let eta = physical[4];
                vec![
                    physical[0],
                    physical[1] - eta * REFERENCE_TEMPERATURE_K.ln(),
                    eta,
                    physical[2],
                    physical[3],
                ]
            }
            Model::M3 => vec![
                physical[0],
                physical[1] - 3.0 * REFERENCE_TEMPERATURE_K.ln(),
                physical[2],
                physical[3],
                physical[4],
            ],
            Model::M4 => vec![
                physical[0],
                physical[1] - 3.0 * REFERENCE_TEMPERATURE_K.ln(),
                physical[2],
                physical[3],
                physical[4],
                physical[5],
            ],
        }
    }

    pub fn logp_and_grad(&self, theta: &[f64]) -> (f64, Vec<f64>) {
        let value = self.logp(theta);
        let mut gradient = vec![0.0; theta.len()];
        for index in 0..theta.len() {
            let mut dual = theta
                .iter()
                .copied()
                .map(Dual::constant)
                .collect::<Vec<_>>();
            dual[index] = Dual::variable(theta[index]);
            gradient[index] = self.logp(&dual).du;
        }
        (value, gradient)
    }

    /// The Hessian at a point, used only as a preconditioner.
    pub fn hessian(&self, center: &[f64]) -> Vec<Vec<f64>> {
        let n = center.len();
        let mut hessian = vec![vec![0.0; n]; n];
        for row in 0..n {
            let mut diagonal = center
                .iter()
                .copied()
                .map(HyperDual::constant)
                .collect::<Vec<_>>();
            diagonal[row] = HyperDual::variable(center[row]);
            hessian[row][row] = self.logp(&diagonal).d12;
            for column in 0..row {
                let mut mixed = center
                    .iter()
                    .copied()
                    .map(HyperDual::constant)
                    .collect::<Vec<_>>();
                mixed[row] = HyperDual::seed_first(center[row]);
                mixed[column] = HyperDual::seed_second(center[column]);
                let value = self.logp(&mixed).d12;
                hessian[row][column] = value;
                hessian[column][row] = value;
            }
        }
        hessian
    }

    pub fn logp_value(&self, theta: &[f64]) -> f64 {
        self.logp(theta)
    }

    pub fn predicted_voltage(&self, theta: &[f64]) -> Vec<f64> {
        let physical = self.physical_theta(theta);
        self.temperature_k
            .iter()
            .map(|temperature| self.forward(&physical, *temperature))
            .collect()
    }

    fn forward<T: Scalar>(&self, theta: &[T], temperature_k: f64) -> T {
        let (eg0, ln_b_ref, alpha, beta, eta, ideality, series_resistance) = match self.model {
            Model::M0 => (
                theta[0],
                theta[1],
                T::from_f64(0.0),
                T::from_f64(0.0),
                T::from_f64(0.0),
                T::from_f64(1.0),
                T::from_f64(0.0),
            ),
            Model::M1 => (
                theta[0],
                theta[1],
                T::from_f64(0.0),
                T::from_f64(0.0),
                theta[2],
                T::from_f64(1.0),
                T::from_f64(0.0),
            ),
            Model::M2 => (
                theta[0],
                theta[1],
                theta[2],
                theta[3],
                theta[4],
                T::from_f64(1.0),
                T::from_f64(0.0),
            ),
            Model::M3 => (
                theta[0],
                theta[1],
                theta[2],
                theta[3],
                T::from_f64(3.0),
                theta[4],
                T::from_f64(0.0),
            ),
            Model::M4 => (
                theta[0],
                theta[1],
                theta[2],
                theta[3],
                T::from_f64(3.0),
                theta[4],
                theta[5],
            ),
        };
        let ln_b = match self.model {
            Model::M0 => ln_b_ref,
            _ => ln_b_ref - eta * T::from_f64(REFERENCE_TEMPERATURE_K.ln()),
        };
        let temperature = T::from_f64(temperature_k);
        let vt = T::from_f64(BOLTZMANN / ELEMENTARY_CHARGE) * temperature;
        let gap = match self.model {
            Model::M0 | Model::M1 => eg0,
            _ => eg0 - alpha * temperature * temperature / (temperature + beta),
        };
        let ln_is = ln_b + eta * temperature.ln() - gap / vt;
        let log_argument = T::from_f64(self.current_a.ln()) - ln_is;
        let log_term = if log_argument.value() > 50.0 {
            log_argument
        } else {
            log_argument.exp().ln_1p()
        };
        T::from_f64(self.current_a) * series_resistance + ideality * vt * log_term
    }

    pub(crate) fn logp<T: Scalar>(&self, theta: &[T]) -> T {
        if theta.len() != self.latent_len() {
            return T::from_f64(f64::NEG_INFINITY);
        }

        let latent = theta;
        let physical;
        let theta = if matches!(self.model, Model::M2 | Model::M3 | Model::M4) {
            physical = self.physical_theta(theta);
            &physical
        } else {
            theta
        };
        let mut logp = T::from_f64(0.0);
        let _ = match self.model {
            Model::M0 => {
                let eg = theta[0];
                let ln_b = theta[1];
                if !in_bounds(eg, 0.05, 3.5) {
                    return T::from_f64(f64::NEG_INFINITY);
                }
                logp = logp + normal_prior(eg, 0.9, 0.35) + normal_prior(ln_b, 8.0, 8.0);
                (
                    eg,
                    ln_b,
                    T::from_f64(0.0),
                    T::from_f64(0.0),
                    T::from_f64(0.0),
                    T::from_f64(1.0),
                    T::from_f64(0.0),
                )
            }
            Model::M1 => {
                let eg = theta[0];
                let ln_b_ref = theta[1];
                let eta = theta[2];
                if !in_bounds(eg, 0.05, 3.5) || !in_bounds(eta, 1.5, 5.0) {
                    return T::from_f64(f64::NEG_INFINITY);
                }
                logp = logp
                    + normal_prior(eg, 0.9, 0.35)
                    + normal_prior(ln_b_ref, 8.0, 8.0)
                    + normal_prior(eta, 3.0, 1.0);
                (
                    eg,
                    ln_b_ref,
                    T::from_f64(0.0),
                    T::from_f64(0.0),
                    eta,
                    T::from_f64(1.0),
                    T::from_f64(0.0),
                )
            }
            Model::M2 => {
                let eg = theta[0];
                let ln_b_ref = theta[1];
                let alpha = theta[2];
                let beta = theta[3];
                let eta = theta[4];
                if !in_bounds(eg, 0.05, 3.5)
                    || !in_bounds(alpha, 0.0, 2.0e-3)
                    || !in_bounds(beta, 100.0, 3000.0)
                    || !in_bounds(eta, 1.5, 5.0)
                {
                    return T::from_f64(f64::NEG_INFINITY);
                }
                logp = logp
                    + normal_prior(eg, 0.9, 0.35)
                    + normal_prior(ln_b_ref, 8.0, 8.0)
                    + normal_prior(alpha, 7.021e-4, 1.0e-4)
                    + normal_prior(beta, 1108.0, 150.0)
                    + normal_prior(eta, 3.0, 1.0);
                (
                    eg,
                    ln_b_ref,
                    alpha,
                    beta,
                    eta,
                    T::from_f64(1.0),
                    T::from_f64(0.0),
                )
            }
            Model::M3 => {
                let eg = theta[0];
                let ln_b_ref = theta[1];
                let alpha = theta[2];
                let beta = theta[3];
                let ideality = theta[4];
                if !in_bounds(eg, 0.05, 3.5)
                    || !in_bounds(alpha, 0.0, 2.0e-3)
                    || !in_bounds(beta, 100.0, 3000.0)
                    || !in_bounds(ideality, 0.7, 1.6)
                {
                    return T::from_f64(f64::NEG_INFINITY);
                }
                logp = logp
                    + normal_prior(eg, 0.9, 0.35)
                    + normal_prior(ln_b_ref, 8.0, 8.0)
                    + normal_prior(alpha, 7.021e-4, 1.0e-4)
                    + normal_prior(beta, 1108.0, 150.0)
                    + normal_prior(ideality, 1.0, 0.15);
                (
                    eg,
                    ln_b_ref,
                    alpha,
                    beta,
                    T::from_f64(3.0),
                    ideality,
                    T::from_f64(0.0),
                )
            }
            Model::M4 => {
                let eg = theta[0];
                let ln_b_ref = theta[1];
                let alpha = theta[2];
                let beta = theta[3];
                let ideality = theta[4];
                let series_resistance = theta[5];
                let latent_series = latent[5];
                if !in_bounds(eg, 0.05, 3.5)
                    || !in_bounds(alpha, 0.0, 2.0e-3)
                    || !in_bounds(beta, 100.0, 3000.0)
                    || !in_bounds(ideality, 0.7, 1.6)
                    || !in_bounds(latent_series, -6.0, 6.0)
                    || series_resistance.value() <= 0.0
                {
                    return T::from_f64(f64::NEG_INFINITY);
                }
                logp = logp
                    + normal_prior(eg, 0.9, 0.35)
                    + normal_prior(ln_b_ref, 8.0, 8.0)
                    + normal_prior(alpha, 7.021e-4, 1.0e-4)
                    + normal_prior(beta, 1108.0, 150.0)
                    + normal_prior(ideality, 1.0, 0.15)
                    + normal_prior(latent_series, 0.0, 1.0);
                (
                    eg,
                    ln_b_ref,
                    alpha,
                    beta,
                    T::from_f64(3.0),
                    ideality,
                    series_resistance,
                )
            }
        };

        if matches!(self.model, Model::M2 | Model::M3 | Model::M4) {
            let jacobian = self.gap_jacobian(theta[3]);
            let jacobian_value = jacobian.value();
            if !jacobian_value.is_finite() || jacobian_value <= 0.0 {
                return T::from_f64(f64::NEG_INFINITY);
            }
            logp = logp - jacobian.ln();
            if matches!(self.model, Model::M3 | Model::M4) {
                let ideality = theta[4];
                if !ideality.value().is_finite() || ideality.value() <= 0.0 {
                    return T::from_f64(f64::NEG_INFINITY);
                }
                logp = logp - T::from_f64(3.0) * ideality.ln();
            }
        }

        let sigma = T::from_f64(self.sigma_v);
        for (temperature, voltage) in self.temperature_k.iter().zip(&self.voltage_v) {
            let predicted = self.forward(theta, *temperature);
            let residual = (predicted - T::from_f64(*voltage)) / sigma;
            logp = logp + T::from_f64(-0.5) * residual * residual;
        }
        logp
    }
}

fn in_bounds<T: Scalar>(value: T, low: f64, high: f64) -> bool {
    value.value() >= low && value.value() <= high
}

fn normal_prior<T: Scalar>(value: T, location: f64, scale: f64) -> T {
    let standardized = (value - T::from_f64(location)) / T::from_f64(scale);
    T::from_f64(-0.5) * standardized * standardized - T::from_f64(scale.ln())
}

fn normal_cdf<T: Scalar>(value: T) -> T {
    T::from_f64(0.5) * (T::from_f64(1.0) + (value / T::from_f64(std::f64::consts::SQRT_2)).erf())
}

/// Map a standard-normal coordinate to a half-normal(10) draw.
fn half_normal_from_normal<T: Scalar>(value: T) -> T {
    let probability = T::from_f64(0.5) * (T::from_f64(1.0) + normal_cdf(value));
    T::from_f64(10.0) * inverse_normal_cdf_scalar(probability)
}

/// Inverse of `half_normal_from_normal`, used for the prior centre.
fn normal_from_half_normal<T: Scalar>(value: T) -> T {
    let probability = T::from_f64(2.0) * normal_cdf(value / T::from_f64(10.0)) - T::from_f64(1.0);
    inverse_normal_cdf_scalar(probability)
}

fn inverse_normal_cdf_scalar<T: Scalar>(probability: T) -> T {
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
    let tail = |q: T| {
        let r = (T::from_f64(-2.0) * q.ln()).sqrt();
        let num = ((((T::from_f64(C[0]) * r + T::from_f64(C[1])) * r + T::from_f64(C[2])) * r
            + T::from_f64(C[3]))
            * r
            + T::from_f64(C[4]))
            * r
            + T::from_f64(C[5]);
        let den = (((T::from_f64(D[0]) * r + T::from_f64(D[1])) * r + T::from_f64(D[2])) * r
            + T::from_f64(D[3]))
            * r
            + T::from_f64(1.0);
        num / den
    };
    let central = |q: T| {
        let r = q * q;
        let num = (((((T::from_f64(A[0]) * r + T::from_f64(A[1])) * r + T::from_f64(A[2])) * r
            + T::from_f64(A[3]))
            * r
            + T::from_f64(A[4]))
            * r
            + T::from_f64(A[5]))
            * q;
        let den = ((((T::from_f64(B[0]) * r + T::from_f64(B[1])) * r + T::from_f64(B[2])) * r
            + T::from_f64(B[3]))
            * r
            + T::from_f64(B[4]))
            * r
            + T::from_f64(1.0);
        num / den
    };
    let p = probability.value();
    if p < 0.024_25 {
        tail(probability)
    } else if p <= 1.0 - 0.024_25 {
        central(probability - T::from_f64(0.5))
    } else {
        -tail(T::from_f64(1.0) - probability)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reported_round_trips_through_the_latent_transform() {
        let temperatures = vec![220.0, 300.0, 383.15];
        let voltages = vec![0.8, 0.7, 0.6];
        let points = [
            (Model::M0, vec![1.002, 5.504]),
            (Model::M1, vec![1.002, 5.504 - 3.0 * 300.0_f64.ln(), 3.0]),
            (Model::M2, vec![1.1557, -7.397705993, 3.0, 7.021e-4, 1108.0]),
            (
                Model::M3,
                vec![1.1557, -6.200414441, 7.021e-4, 1108.0, 1.05],
            ),
            (
                Model::M4,
                vec![1.1557, -6.200414441, 7.021e-4, 1108.0, 1.05, 8.0],
            ),
        ];
        for (model, reported) in points {
            let case = Case::new(
                model,
                temperatures.clone(),
                voltages.clone(),
                3.0e-4,
                1.253e-4,
            )
            .expect("valid case");
            let latent = case.latent_from_reported(&reported);
            let round_trip = case.reported(&latent);
            for (expected, observed) in reported.iter().zip(round_trip) {
                assert!(
                    (expected - observed).abs() <= 1e-7 * expected.abs().max(1.0),
                    "{model:?}: {expected} != {observed}"
                );
            }
        }
    }
}
