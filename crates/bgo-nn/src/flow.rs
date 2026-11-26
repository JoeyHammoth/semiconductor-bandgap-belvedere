//! Conditional rational-quadratic spline flow.
//!
//! The flow transforms a two-dimensional standard-normal base into the
//! standardised parameter space. Layer 0 transforms coordinate 1 conditioned on
//! coordinate 0; layer 1 transforms coordinate 0 conditioned on the updated
//! coordinate 1. The exported architecture is deliberately fixed: dense SiLU
//! conditioner, one spline per layer, identity outside the spline domain.

use crate::manifest::FlowManifest;
use crate::spline::Spline;
use std::fmt;

const SPLINE_MIN: f64 = 0.05;

#[derive(Debug, Clone)]
pub struct FlowError(String);

impl FlowError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for FlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FlowError {}

type Result<T> = std::result::Result<T, FlowError>;

#[derive(Debug, Clone)]
struct Layer {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    w3: Vec<f64>,
    b3: Vec<f64>,
    hidden: usize,
    input_dim: usize,
    output_dim: usize,
}

impl Layer {
    fn load(
        weights: &[f64],
        offset: &mut usize,
        input_dim: usize,
        hidden: usize,
        output_dim: usize,
    ) -> Self {
        let w1 = take(weights, offset, input_dim * hidden);
        let b1 = take(weights, offset, hidden);
        let w2 = take(weights, offset, hidden * hidden);
        let b2 = take(weights, offset, hidden);
        let w3 = take(weights, offset, hidden * output_dim);
        let b3 = take(weights, offset, output_dim);
        Self {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            hidden,
            input_dim,
            output_dim,
        }
    }

    fn apply(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.input_dim {
            return Err(FlowError::new(format!(
                "conditioner input has {} values, expected {}",
                input.len(),
                self.input_dim
            )));
        }
        let mut first = vec![0.0; self.hidden];
        for row in 0..self.hidden {
            let mut value = self.b1[row];
            for column in 0..self.input_dim {
                value += self.w1[column * self.hidden + row] * input[column];
            }
            first[row] = silu(value);
        }
        let mut second = vec![0.0; self.hidden];
        for row in 0..self.hidden {
            let mut value = self.b2[row];
            for column in 0..self.hidden {
                value += self.w2[column * self.hidden + row] * first[column];
            }
            second[row] = silu(value);
        }
        let mut output = vec![0.0; self.output_dim];
        for row in 0..self.output_dim {
            let mut value = self.b3[row];
            for column in 0..self.hidden {
                value += self.w3[column * self.output_dim + row] * second[column];
            }
            output[row] = value;
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(FlowError::new("conditioner produced a non-finite value"));
        }
        Ok(output)
    }
}

fn take(values: &[f64], offset: &mut usize, count: usize) -> Vec<f64> {
    let end = *offset + count;
    let slice = values[*offset..end].to_vec();
    *offset = end;
    slice
}

fn silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

#[derive(Debug, Clone)]
pub struct Flow {
    manifest: FlowManifest,
    layers: Vec<Layer>,
    spline: Option<Spline>,
}

impl Flow {
    /// Parse a manifest and little-endian float64 weight blob.
    pub fn from_json(manifest_json: &str, weights_bytes: &[u8]) -> Result<Self> {
        let manifest: FlowManifest = serde_json::from_str(manifest_json)
            .map_err(|error| FlowError::new(format!("invalid flow manifest: {error}")))?;
        Self::validate_manifest(&manifest)?;
        if weights_bytes.len() % 8 != 0 {
            return Err(FlowError::new("weight blob is not a multiple of 8 bytes"));
        }
        let weights = weights_bytes
            .chunks_exact(8)
            .map(|bytes| {
                let mut array = [0u8; 8];
                array.copy_from_slice(bytes);
                f64::from_le_bytes(array)
            })
            .collect::<Vec<_>>();
        if weights.len() != manifest.weight_count {
            return Err(FlowError::new(format!(
                "weight count {} does not match manifest {}",
                weights.len(),
                manifest.weight_count
            )));
        }
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(FlowError::new("weights contain a non-finite value"));
        }
        let expected = manifest.expected_weight_count();
        if weights.len() != expected {
            return Err(FlowError::new(format!(
                "weight count {} does not match architecture {}",
                weights.len(),
                expected
            )));
        }
        let output_dim = if manifest.is_affine() {
            2
        } else {
            3 * manifest.bins - 1
        };
        let input_dim = manifest.condition_dim + 1;
        let mut layers = Vec::with_capacity(manifest.layers);
        let mut offset = 0usize;
        for _ in 0..manifest.layers {
            layers.push(Layer::load(
                &weights,
                &mut offset,
                input_dim,
                manifest.hidden_dim,
                output_dim,
            ));
        }
        if offset != weights.len() {
            return Err(FlowError::new("weight layout did not consume the blob"));
        }
        let spline = if manifest.is_affine() {
            None
        } else {
            Some(
                Spline::new(manifest.bins, manifest.bound)
                    .map_err(|error| FlowError::new(error.to_string()))?,
            )
        };
        Ok(Self {
            manifest,
            layers,
            spline,
        })
    }

    fn validate_manifest(manifest: &FlowManifest) -> Result<()> {
        if manifest.schema_version != 1
            || !matches!(
                manifest.kind.as_str(),
                "conditional-affine-flow" | "conditional-rational-quadratic-spline-flow"
            )
            || !manifest.synthetic
            || manifest.training_framework != "jax"
        {
            return Err(FlowError::new(
                "manifest must be a synthetic schema-1 JAX flow",
            ));
        }
        if manifest.model != "M0"
            || manifest.parameter_dim != 2
            || manifest.parameter_names.len() != 2
            || manifest.parameter_names[0] != "eg_ev"
            || manifest.parameter_names[1] != "ln_b"
        {
            return Err(FlowError::new(
                "manifest must describe the M0 two-parameter target",
            ));
        }
        if manifest.condition_dim == 0
            || manifest.hidden_dim == 0
            || manifest.layers == 0
            || manifest.bins < 2
            || !manifest.bound.is_finite()
            || manifest.bound <= 0.0
        {
            return Err(FlowError::new(
                "manifest architecture dimensions are invalid",
            ));
        }
        if manifest.layer_split.len() != manifest.layers {
            return Err(FlowError::new("manifest layer_split length is invalid"));
        }
        for (index, split) in manifest.layer_split.iter().enumerate() {
            if *split != index % 2 {
                return Err(FlowError::new(
                    "manifest layer_split does not match the engine",
                ));
            }
        }
        let expected = [
            manifest.parameter_center.len(),
            manifest.parameter_scale.len(),
            manifest.condition_mean.len(),
            manifest.condition_scale.len(),
        ];
        if expected
            .iter()
            .any(|length| *length != manifest.parameter_dim && *length != manifest.condition_dim)
        {
            return Err(FlowError::new(
                "manifest standardisation vectors have invalid lengths",
            ));
        }
        if manifest.condition_mean.len() != manifest.condition_dim
            || manifest.condition_scale.len() != manifest.condition_dim
            || manifest.parameter_center.len() != manifest.parameter_dim
            || manifest.parameter_scale.len() != manifest.parameter_dim
        {
            return Err(FlowError::new(
                "manifest standardisation vectors have invalid lengths",
            ));
        }
        if manifest.ood_summary_mean.len() != 5
            || manifest.ood_summary_scale.len() != 5
            || manifest.ood_mean.len() != 5
            || manifest.ood_inverse_covariance.len() != 5
            || manifest
                .ood_inverse_covariance
                .iter()
                .any(|row| row.len() != 5)
        {
            return Err(FlowError::new(
                "manifest OOD summary dimensions are invalid",
            ));
        }
        if !manifest.ood_threshold.is_finite() || manifest.ood_threshold <= 0.0 {
            return Err(FlowError::new("manifest OOD threshold is invalid"));
        }
        Ok(())
    }

    pub fn manifest(&self) -> &FlowManifest {
        &self.manifest
    }

    pub fn parameter_names(&self) -> &[String] {
        &self.manifest.parameter_names
    }

    pub fn condition_dim(&self) -> usize {
        self.manifest.condition_dim
    }

    pub fn parameter_dim(&self) -> usize {
        self.manifest.parameter_dim
    }

    pub fn weights_sha256(&self) -> &str {
        &self.manifest.weights_sha256
    }

    pub fn standardise_condition(&self, condition: &[f64]) -> Result<Vec<f64>> {
        if condition.len() != self.manifest.condition_dim {
            return Err(FlowError::new(format!(
                "condition has {} values, expected {}",
                condition.len(),
                self.manifest.condition_dim
            )));
        }
        if condition.iter().any(|value| !value.is_finite()) {
            return Err(FlowError::new("condition must be finite"));
        }
        Ok(condition
            .iter()
            .zip(&self.manifest.condition_mean)
            .zip(&self.manifest.condition_scale)
            .map(|((value, mean), scale)| (value - mean) / scale)
            .collect())
    }

    fn standardise_parameter(&self, parameter: &[f64]) -> Result<Vec<f64>> {
        if parameter.len() != self.manifest.parameter_dim {
            return Err(FlowError::new("parameter vector has the wrong length"));
        }
        if parameter.iter().any(|value| !value.is_finite()) {
            return Err(FlowError::new("parameter vector must be finite"));
        }
        Ok(parameter
            .iter()
            .zip(&self.manifest.parameter_center)
            .zip(&self.manifest.parameter_scale)
            .map(|((value, center), scale)| (value - center) / scale)
            .collect())
    }

    fn raw_conditioner_input(
        &self,
        x: &[f64],
        condition: &[f64],
        index: usize,
    ) -> Result<Vec<f64>> {
        let transformed = index % 2;
        let context = 1 - transformed;
        let mut input = Vec::with_capacity(self.manifest.condition_dim + 1);
        input.push(x[context]);
        input.extend_from_slice(condition);
        Ok(input)
    }

    pub fn log_prob_standardised(&self, condition: &[f64], x: &[f64]) -> Result<f64> {
        if condition.len() != self.manifest.condition_dim || x.len() != self.manifest.parameter_dim
        {
            return Err(FlowError::new(
                "standardised log-prob dimensions are invalid",
            ));
        }
        let mut z = x.to_vec();
        let mut log_det = 0.0;
        for index in 0..self.layers.len() {
            let input = self.raw_conditioner_input(&z, condition, index)?;
            let raw = self.layers[index].apply(&input)?;
            let transformed = index % 2;
            if self.manifest.is_affine() {
                let log_scale = raw[0].clamp(-6.0, 6.0);
                z[transformed] = log_scale.exp() * z[transformed] + raw[1];
                log_det += log_scale;
            } else {
                let spline = self
                    .spline
                    .as_ref()
                    .ok_or_else(|| FlowError::new("spline engine is missing"))?;
                let (value, term) = spline
                    .forward(z[transformed], &raw)
                    .map_err(|error| FlowError::new(error.to_string()))?;
                z[transformed] = value;
                log_det += term;
            }
        }
        if z.iter().any(|value| !value.is_finite()) || !log_det.is_finite() {
            return Err(FlowError::new("flow produced a non-finite log-probability"));
        }
        let base = -0.5 * z.iter().map(|value| value * value).sum::<f64>()
            - 0.5 * self.manifest.parameter_dim as f64 * (2.0 * std::f64::consts::PI).ln();
        Ok(base + log_det)
    }

    pub fn log_prob(&self, condition: &[f64], parameter: &[f64]) -> Result<f64> {
        let condition = self.standardise_condition(condition)?;
        let parameter = self.standardise_parameter(parameter)?;
        self.log_prob_standardised(&condition, &parameter)
    }

    pub fn inverse_from_z_standardised(&self, condition: &[f64], z: &[f64]) -> Result<Vec<f64>> {
        if condition.len() != self.manifest.condition_dim || z.len() != self.manifest.parameter_dim
        {
            return Err(FlowError::new(
                "standardised inverse dimensions are invalid",
            ));
        }
        let mut x = z.to_vec();
        for index in (0..self.layers.len()).rev() {
            let input = self.raw_conditioner_input(&x, condition, index)?;
            let raw = self.layers[index].apply(&input)?;
            let transformed = index % 2;
            if self.manifest.is_affine() {
                let log_scale = raw[0].clamp(-6.0, 6.0);
                x[transformed] = (x[transformed] - raw[1]) * (-log_scale).exp();
            } else {
                let spline = self
                    .spline
                    .as_ref()
                    .ok_or_else(|| FlowError::new("spline engine is missing"))?;
                x[transformed] = spline
                    .inverse(x[transformed], &raw)
                    .map_err(|error| FlowError::new(error.to_string()))?;
            }
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(FlowError::new("inverse flow produced a non-finite value"));
        }
        Ok(x)
    }

    /// Draw samples from the amortised posterior. The output is sample-major:
    /// `count * parameter_dim`, in reported parameter units.
    pub fn sample(&self, condition: &[f64], count: usize, seed: u64) -> Result<Vec<f64>> {
        if count == 0 || count > 1_000_000 {
            return Err(FlowError::new(
                "sample count must be between 1 and 1,000,000",
            ));
        }
        let standardised_condition = self.standardise_condition(condition)?;
        let mut rng = Rng::new(seed);
        let mut output = Vec::with_capacity(count * self.manifest.parameter_dim);
        for _ in 0..count {
            let z = (0..self.manifest.parameter_dim)
                .map(|_| rng.normal())
                .collect::<Vec<_>>();
            let x = self.inverse_from_z_standardised(&standardised_condition, &z)?;
            for (value, center) in x.iter().zip(&self.manifest.parameter_center) {
                let index = output.len() % self.manifest.parameter_dim;
                let scale = self.manifest.parameter_scale[index];
                output.push(center + scale * value);
            }
        }
        Ok(output)
    }

    /// Mahalanobis distance of a five-number sweep summary, in the units fitted
    /// during training. The flow manifest carries the mean, scale, inverse
    /// covariance and 99th-percentile threshold.
    pub fn ood_distance(&self, condition: &[f64]) -> Result<f64> {
        if condition.len() != self.manifest.condition_dim {
            return Err(FlowError::new("OOD condition has the wrong length"));
        }
        if condition.iter().any(|value| !value.is_finite()) {
            return Err(FlowError::new("OOD condition must be finite"));
        }
        let standardised = condition
            .iter()
            .zip(&self.manifest.ood_summary_mean)
            .zip(&self.manifest.ood_summary_scale)
            .map(|((value, mean), scale)| (value - mean) / scale)
            .collect::<Vec<_>>();
        let mut distance = 0.0;
        for row in 0..5 {
            for column in 0..5 {
                distance += (standardised[row] - self.manifest.ood_mean[row])
                    * self.manifest.ood_inverse_covariance[row][column]
                    * (standardised[column] - self.manifest.ood_mean[column]);
            }
        }
        if !distance.is_finite() || distance < 0.0 {
            return Err(FlowError::new("OOD distance is not finite"));
        }
        Ok(distance)
    }

    pub fn ood_threshold(&self) -> f64 {
        self.manifest.ood_threshold
    }
}

#[allow(dead_code)]
fn sweep_summaries(condition: &[f64]) -> Result<[f64; 5]> {
    if condition.len() < 4 {
        return Err(FlowError::new("OOD summary needs at least four points"));
    }
    let n = condition.len() as f64;
    let step = 2.0 / (condition.len() - 1) as f64;
    let mut s = [0.0; 5];
    let mut y = [0.0; 3];
    for (index, value) in condition.iter().enumerate() {
        let t = -1.0 + step * index as f64;
        let powers = [1.0, t, t * t, t * t * t, t * t * t * t];
        for power in 0..5 {
            s[power] += powers[power];
        }
        for power in 0..3 {
            y[power] += powers[power] * value;
        }
    }
    let matrix = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
    let coefficients = solve3(matrix, y)?;
    let mut residual_sum = 0.0;
    for (index, value) in condition.iter().enumerate() {
        let t = -1.0 + step * index as f64;
        let fitted = coefficients[0] + coefficients[1] * t + coefficients[2] * t * t;
        residual_sum += (value - fitted) * (value - fitted);
    }
    Ok([
        condition.iter().sum::<f64>() / n,
        coefficients[1],
        coefficients[2],
        (residual_sum / n).sqrt(),
        condition[condition.len() - 1] - condition[0],
    ])
}

#[allow(dead_code)]
fn solve3(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Result<[f64; 3]> {
    let mut augmented = [[0.0; 4]; 3];
    for row in 0..3 {
        for column in 0..3 {
            augmented[row][column] = matrix[row][column];
        }
        augmented[row][3] = rhs[row];
    }
    for pivot in 0..3 {
        let mut best = pivot;
        for row in pivot + 1..3 {
            if augmented[row][pivot].abs() > augmented[best][pivot].abs() {
                best = row;
            }
        }
        if augmented[best][pivot].abs() < 1.0e-14 {
            return Err(FlowError::new("OOD summary fit is singular"));
        }
        augmented.swap(pivot, best);
        let divisor = augmented[pivot][pivot];
        for column in pivot..4 {
            augmented[pivot][column] /= divisor;
        }
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            for column in pivot..4 {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }
    Ok([augmented[0][3], augmented[1][3], augmented[2][3]])
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
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

    fn normal(&mut self) -> f64 {
        let first = self.uniform().max(SPLINE_MIN);
        let second = self.uniform();
        (-2.0 * first.ln()).sqrt() * (2.0 * std::f64::consts::PI * second).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Golden {
        conditions: Vec<Vec<f64>>,
        x: Vec<Vec<f64>>,
        log_prob: Vec<f64>,
        sample_z: Vec<Vec<f64>>,
        sample_x: Vec<Vec<f64>>,
    }

    fn artifacts() -> (Flow, Golden) {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../");
        let manifest = std::fs::read_to_string(format!("{root}web/static/weights/m0-flow.json"))
            .expect("manifest");
        let weights =
            std::fs::read(format!("{root}web/static/weights/m0-flow.bin")).expect("weights");
        let golden: Golden = serde_json::from_str(
            &std::fs::read_to_string(format!(
                "{root}crates/bgo-nn/tests/references/m0-flow-golden.json"
            ))
            .expect("golden"),
        )
        .expect("golden json");
        (Flow::from_json(&manifest, &weights).expect("flow"), golden)
    }

    #[test]
    fn matches_golden_vectors() {
        let (flow, golden) = artifacts();
        assert_eq!(golden.conditions.len(), golden.x.len());
        for index in 0..golden.conditions.len() {
            let expected = golden.log_prob[index];
            let actual = flow
                .log_prob_standardised(&golden.conditions[index], &golden.x[index])
                .expect("log prob");
            assert!(
                (actual - expected).abs() < 1.0e-9,
                "log prob {index}: {actual} vs {expected}"
            );
            let inverse = flow
                .inverse_from_z_standardised(&golden.conditions[index], &golden.sample_z[index])
                .expect("inverse");
            for (actual, expected) in inverse.iter().zip(&golden.sample_x[index]) {
                assert!(
                    (actual - expected).abs() < 1.0e-9,
                    "sample {index}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn draws_are_finite_and_have_the_right_shape() {
        let (flow, golden) = artifacts();
        let samples = flow.sample(&golden.conditions[0], 128, 7).expect("samples");
        assert_eq!(samples.len(), 256);
        assert!(samples.iter().all(|value| value.is_finite()));
        assert!(flow.ood_threshold().is_finite());
    }
}
