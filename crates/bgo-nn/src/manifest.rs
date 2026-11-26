//! JSON architecture manifest for an exported conditional spline flow.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct FlowManifest {
    pub schema_version: u32,
    pub kind: String,
    pub flow_kind: Option<String>,
    pub synthetic: bool,
    pub training_framework: String,
    pub model: String,
    pub parameter_names: Vec<String>,
    pub condition_dim: usize,
    pub parameter_dim: usize,
    pub hidden_dim: usize,
    pub layers: usize,
    pub bins: usize,
    pub bound: f64,
    pub layer_split: Vec<usize>,
    pub parameter_center: Vec<f64>,
    pub parameter_scale: Vec<f64>,
    pub condition_mean: Vec<f64>,
    pub condition_scale: Vec<f64>,
    pub ood_summary_mean: Vec<f64>,
    pub ood_summary_scale: Vec<f64>,
    pub ood_mean: Vec<f64>,
    pub ood_inverse_covariance: Vec<Vec<f64>>,
    pub ood_threshold: f64,
    pub weight_count: usize,
    pub weights_sha256: String,
}

impl FlowManifest {
    pub fn expected_weight_count(&self) -> usize {
        let input = self.condition_dim + 1;
        let output = if self.kind == "conditional-affine-flow" {
            2
        } else {
            3 * self.bins - 1
        };
        self.layers
            * (input * self.hidden_dim
                + self.hidden_dim
                + self.hidden_dim * self.hidden_dim
                + self.hidden_dim
                + self.hidden_dim * output
                + output)
    }

    pub fn is_affine(&self) -> bool {
        self.kind == "conditional-affine-flow"
    }
}
