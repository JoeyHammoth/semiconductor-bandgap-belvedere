//! WebAssembly surface for the Browser core.
//!
//! The JavaScript side speaks typed arrays; configuration crosses as small
//! structs. Each worker owns its own instance, so nothing here may hold global
//! mutable state.

#![forbid(unsafe_code)]

use bgo_infer::{
    bulk_ess, run_nested, run_svgd, split_rhat, Case, Diagnostics, LogLikePrior, MclmcConfig,
    MclmcOutput, Model, NestedConfig, SvgdConfig, SvgdTarget,
};
use bgo_info::{analyze_fisher, covariance_ellipse, fisher_from_jacobian};
use bgo_ot::sliced_wasserstein_2;
use wasm_bindgen::prelude::*;

/// Core version, rendered in the status bar.
#[wasm_bindgen]
pub fn core_version() -> String {
    format!("bgo-wasm {}", env!("CARGO_PKG_VERSION"))
}

/// Ideal constant-current diode voltage from the model specification.
///
/// `eg_ev` is the gap in electronvolts; numerically it equals volts because
/// `E_g[J] / q = E_g[eV]`.
#[wasm_bindgen]
pub fn ideal_diode_voltage(current: f64, temperature: f64, ln_b: f64, eta: f64, eg_ev: f64) -> f64 {
    bgo_physics::ideal_diode_voltage(current, temperature, ln_b, eta, eg_ev)
}

/// Varshni gap at a temperature, in eV.
#[wasm_bindgen]
pub fn varshni_gap(eg0: f64, alpha: f64, beta: f64, temperature: f64) -> f64 {
    bgo_physics::varshni_gap(eg0, alpha, beta, temperature)
}

/// A completed sampler run. Arrays are copied out of WASM once, then the
/// JavaScript side owns them; no live view into WASM memory crosses the
/// boundary.
#[wasm_bindgen]
pub struct SamplerOutput {
    output: MclmcOutput,
    draws: Vec<f64>,
}

impl SamplerOutput {
    fn new(output: MclmcOutput) -> Self {
        let draws = output.by_parameter().concat();
        Self { output, draws }
    }
}

#[wasm_bindgen]
impl SamplerOutput {
    /// Comma-separated reported parameter names, in draw-array order.
    #[wasm_bindgen(getter)]
    pub fn parameter_names(&self) -> String {
        self.output.parameter_names.join(",")
    }

    /// Parameter-major draws: `parameter_count * draws_per_parameter` values.
    #[wasm_bindgen(getter)]
    pub fn draws(&self) -> Vec<f64> {
        self.draws.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn parameter_count(&self) -> u32 {
        self.output.parameter_names.len() as u32
    }

    #[wasm_bindgen(getter)]
    pub fn draws_per_parameter(&self) -> u32 {
        if self.output.parameter_names.is_empty() {
            0
        } else {
            (self.draws.len() / self.output.parameter_names.len()) as u32
        }
    }

    #[wasm_bindgen(getter)]
    pub fn chain_count(&self) -> u32 {
        self.output.chains.len() as u32
    }

    #[wasm_bindgen(getter)]
    pub fn draws_per_chain(&self) -> u32 {
        self.draws_per_parameter()
            .checked_div(self.chain_count())
            .unwrap_or(0)
    }

    /// Chain-major absolute energy errors, one value per post-warmup draw.
    #[wasm_bindgen(getter)]
    pub fn energy_error_trace(&self) -> Vec<f64> {
        self.output
            .energy_error_trace
            .iter()
            .flatten()
            .copied()
            .collect()
    }

    #[wasm_bindgen(getter)]
    pub fn rhat(&self) -> Vec<f64> {
        self.output.diagnostics.rhat.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn ess_bulk(&self) -> Vec<f64> {
        self.output.diagnostics.ess_bulk.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn rhat_max(&self) -> f64 {
        self.output.diagnostics.rhat_max
    }

    #[wasm_bindgen(getter)]
    pub fn ess_bulk_min(&self) -> f64 {
        self.output.diagnostics.ess_bulk_min
    }

    #[wasm_bindgen(getter)]
    pub fn energy_error_mean(&self) -> f64 {
        self.output.energy_error_mean
    }

    #[wasm_bindgen(getter)]
    pub fn divergences(&self) -> u32 {
        self.output.diagnostics.divergences as u32
    }

    #[wasm_bindgen(getter)]
    pub fn passes(&self) -> bool {
        self.output.diagnostics.passes
    }

    #[wasm_bindgen(getter)]
    pub fn acceptance_rate(&self) -> f64 {
        self.output.acceptance_rate
    }

    #[wasm_bindgen(getter)]
    pub fn step_size(&self) -> f64 {
        self.output.step_size
    }
}

/// Diagnostics for chains sampled independently in different workers.
///
/// `draws` is chain-major and, within each chain, parameter-major:
/// `[chain][parameter][draw]`. R-hat and bulk ESS are computed in Rust so the
/// browser cannot substitute its own convergence verdict.
#[wasm_bindgen]
pub struct DiagnosticsOutput {
    diagnostics: Diagnostics,
}

#[wasm_bindgen]
impl DiagnosticsOutput {
    #[wasm_bindgen(getter)]
    pub fn rhat(&self) -> Vec<f64> {
        self.diagnostics.rhat.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn ess_bulk(&self) -> Vec<f64> {
        self.diagnostics.ess_bulk.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn rhat_max(&self) -> f64 {
        self.diagnostics.rhat_max
    }

    #[wasm_bindgen(getter)]
    pub fn ess_bulk_min(&self) -> f64 {
        self.diagnostics.ess_bulk_min
    }

    #[wasm_bindgen(getter)]
    pub fn energy_error_mean(&self) -> f64 {
        self.diagnostics.energy_error_mean
    }

    #[wasm_bindgen(getter)]
    pub fn divergences(&self) -> u32 {
        self.diagnostics.divergences as u32
    }

    #[wasm_bindgen(getter)]
    pub fn passes(&self) -> bool {
        self.diagnostics.passes
    }
}

#[wasm_bindgen]
pub fn diagnose_chains(
    chains: u32,
    parameter_count: u32,
    draws_per_chain: u32,
    draws: &[f64],
    energy_error_mean: f64,
    divergences: u32,
) -> Result<DiagnosticsOutput, JsValue> {
    let chains = chains as usize;
    let parameter_count = parameter_count as usize;
    let draws_per_chain = draws_per_chain as usize;
    if chains == 0 || parameter_count == 0 || draws_per_chain < 2 {
        return Err(JsValue::from_str("invalid chain dimensions"));
    }
    let expected = chains
        .checked_mul(parameter_count)
        .and_then(|value| value.checked_mul(draws_per_chain))
        .ok_or_else(|| JsValue::from_str("chain dimensions overflow"))?;
    if draws.len() != expected {
        return Err(JsValue::from_str(
            "chain draw count does not match dimensions",
        ));
    }
    let mut parameters = vec![vec![Vec::with_capacity(draws_per_chain); chains]; parameter_count];
    let chain_stride = parameter_count * draws_per_chain;
    for (chain, chain_draws) in draws.chunks_exact(chain_stride).enumerate() {
        for (parameter, values) in parameters.iter_mut().enumerate() {
            let start = parameter * draws_per_chain;
            values[chain].extend_from_slice(&chain_draws[start..start + draws_per_chain]);
        }
    }

    let rhat = parameters.iter().map(|values| split_rhat(values)).collect();
    let ess_bulk = parameters.iter().map(|values| bulk_ess(values)).collect();
    Ok(DiagnosticsOutput {
        diagnostics: Diagnostics::from_parameters(
            rhat,
            ess_bulk,
            energy_error_mean,
            divergences as usize,
        ),
    })
}

/// Fisher information and its sloppy spectrum at one reported parameter point.
///
/// The model, observations and Jacobian stay in Rust; the browser receives the
/// covariance, eigenvectors and two-dimensional covariance ellipses.
#[wasm_bindgen]
pub struct FisherOutput {
    parameter_names: String,
    parameter_count: u32,
    covariance: Vec<f64>,
    eigenvalues: Vec<f64>,
    eigenvectors: Vec<f64>,
    condition_number: f64,
    stiffest_index: u32,
    sloppiest_index: u32,
    ellipse_1: Vec<f64>,
    ellipse_2: Vec<f64>,
}

#[wasm_bindgen]
impl FisherOutput {
    #[wasm_bindgen(getter)]
    pub fn parameter_names(&self) -> String {
        self.parameter_names.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn parameter_count(&self) -> u32 {
        self.parameter_count
    }

    /// Row-major covariance in reported-parameter coordinates.
    #[wasm_bindgen(getter)]
    pub fn covariance(&self) -> Vec<f64> {
        self.covariance.clone()
    }

    /// Descending Fisher eigenvalues: stiffest first, sloppiest last.
    #[wasm_bindgen(getter)]
    pub fn eigenvalues(&self) -> Vec<f64> {
        self.eigenvalues.clone()
    }

    /// Eigenvectors in the same order, flattened vector-major.
    #[wasm_bindgen(getter)]
    pub fn eigenvectors(&self) -> Vec<f64> {
        self.eigenvectors.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn condition_number(&self) -> f64 {
        self.condition_number
    }

    #[wasm_bindgen(getter)]
    pub fn stiffest_index(&self) -> u32 {
        self.stiffest_index
    }

    #[wasm_bindgen(getter)]
    pub fn sloppiest_index(&self) -> u32 {
        self.sloppiest_index
    }

    /// `[semi-major, semi-minor, angle]` for the 1-sigma ellipse in the first
    /// two reported parameters.
    #[wasm_bindgen(getter)]
    pub fn ellipse_1(&self) -> Vec<f64> {
        self.ellipse_1.clone()
    }

    /// `[semi-major, semi-minor, angle]` for the 2-sigma ellipse.
    #[wasm_bindgen(getter)]
    pub fn ellipse_2(&self) -> Vec<f64> {
        self.ellipse_2.clone()
    }
}

#[wasm_bindgen]
pub fn fisher_report(
    model: &str,
    temperature_k: &[f64],
    current_a: f64,
    sigma_v: f64,
    reported: &[f64],
) -> Result<FisherOutput, JsValue> {
    if temperature_k.len() < 2 {
        return Err(JsValue::from_str("at least two temperatures are required"));
    }
    let case = Case::new(
        model_from_name(model).map_err(|error| JsValue::from_str(&error))?,
        temperature_k.to_vec(),
        vec![0.0; temperature_k.len()],
        current_a,
        sigma_v,
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let jacobian = case
        .observation_jacobian_reported(reported)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let fisher = fisher_from_jacobian(&jacobian, sigma_v)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let report = analyze_fisher(&fisher).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let ellipse_1 = covariance_ellipse(&report.covariance, 0, 1, 1.0)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let ellipse_2 = covariance_ellipse(&report.covariance, 0, 1, 2.0)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    Ok(FisherOutput {
        parameter_names: case.parameter_names().join(","),
        parameter_count: case.parameter_names().len() as u32,
        covariance: report.covariance.data().to_vec(),
        eigenvalues: report.spectrum.eigenvalues.clone(),
        eigenvectors: report
            .spectrum
            .eigenvectors
            .iter()
            .flatten()
            .copied()
            .collect(),
        condition_number: report.condition_number,
        stiffest_index: report.stiffest_index as u32,
        sloppiest_index: report.sloppiest_index as u32,
        ellipse_1: vec![
            ellipse_1.semi_major,
            ellipse_1.semi_minor,
            ellipse_1.angle_rad,
        ],
        ellipse_2: vec![
            ellipse_2.semi_major,
            ellipse_2.semi_minor,
            ellipse_2.angle_rad,
        ],
    })
}

/// Synthetic 2D Gaussian calibration fixture for S2. The unit-cube prior is
/// mapped to a labelled `(E_g, ln B)` window; the evidence is compared with
/// the closed-form value from the same mapping.
const EVIDENCE_LOW: [f64; 2] = [0.8, 5.0];
const EVIDENCE_HIGH: [f64; 2] = [1.2, 6.0];
const EVIDENCE_MEAN: [f64; 2] = [1.002, 5.504_239_846];
const EVIDENCE_SIGMA: [f64; 2] = [0.003, 0.02];

struct EvidenceGaussian;

impl LogLikePrior for EvidenceGaussian {
    fn dimension(&self) -> usize {
        2
    }

    fn log_like_prior(&self, theta: &[f64]) -> f64 {
        if theta.len() != 2 {
            return f64::NEG_INFINITY;
        }
        let eg = EVIDENCE_LOW[0] + theta[0] * (EVIDENCE_HIGH[0] - EVIDENCE_LOW[0]);
        let ln_b = EVIDENCE_LOW[1] + theta[1] * (EVIDENCE_HIGH[1] - EVIDENCE_LOW[1]);
        let first = (eg - EVIDENCE_MEAN[0]) / EVIDENCE_SIGMA[0];
        let second = (ln_b - EVIDENCE_MEAN[1]) / EVIDENCE_SIGMA[1];
        -0.5 * (first * first + second * second)
    }
}

/// Nested-sampling event trace for S2, including live-point snapshots.
#[wasm_bindgen]
pub struct NestedTraceOutput {
    log_evidence: f64,
    log_evidence_error: f64,
    expected_log_evidence: f64,
    information: f64,
    iterations: u32,
    converged: bool,
    live_points: u32,
    frame_count: u32,
    parameter_names: String,
    event_iterations: Vec<u32>,
    event_log_evidence: Vec<f64>,
    event_thresholds: Vec<f64>,
    event_log_prior_volume: Vec<f64>,
    live_positions: Vec<f64>,
    prior_low: Vec<f64>,
    prior_high: Vec<f64>,
}

#[wasm_bindgen]
impl NestedTraceOutput {
    #[wasm_bindgen(getter)]
    pub fn log_evidence(&self) -> f64 {
        self.log_evidence
    }

    #[wasm_bindgen(getter)]
    pub fn log_evidence_error(&self) -> f64 {
        self.log_evidence_error
    }

    #[wasm_bindgen(getter)]
    pub fn expected_log_evidence(&self) -> f64 {
        self.expected_log_evidence
    }

    #[wasm_bindgen(getter)]
    pub fn information(&self) -> f64 {
        self.information
    }

    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.converged
    }

    #[wasm_bindgen(getter)]
    pub fn live_points(&self) -> u32 {
        self.live_points
    }

    #[wasm_bindgen(getter)]
    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    #[wasm_bindgen(getter)]
    pub fn parameter_names(&self) -> String {
        self.parameter_names.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn event_iterations(&self) -> Vec<u32> {
        self.event_iterations.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn event_log_evidence(&self) -> Vec<f64> {
        self.event_log_evidence.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn event_thresholds(&self) -> Vec<f64> {
        self.event_thresholds.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn event_log_prior_volume(&self) -> Vec<f64> {
        self.event_log_prior_volume.clone()
    }

    /// Frame-major, then point-major, then two physical coordinates.
    #[wasm_bindgen(getter)]
    pub fn live_positions(&self) -> Vec<f64> {
        self.live_positions.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn prior_low(&self) -> Vec<f64> {
        self.prior_low.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn prior_high(&self) -> Vec<f64> {
        self.prior_high.clone()
    }
}

fn run_nested_gaussian_impl(live_points: u32, seed: u32) -> Result<NestedTraceOutput, String> {
    let live_points = (live_points.max(32) as usize).min(400);
    let config = NestedConfig {
        live_points,
        max_iterations: 4_000,
        remaining_evidence: 1.0e-3,
        seed: u64::from(seed),
        slice_steps: 30,
        trace_stride: 1,
        ..NestedConfig::default()
    };
    let mut events = Vec::new();
    let output = run_nested(&EvidenceGaussian, &config, |event| events.push(event))
        .map_err(|error| error.to_string())?;
    if events.is_empty() || events.iter().any(|event| event.live_positions.is_empty()) {
        return Err("nested sampling emitted no live-point trace".into());
    }

    let mut live_positions = Vec::with_capacity(events.len() * live_points * 2);
    for event in &events {
        for point in event.live_positions.chunks_exact(2) {
            live_positions.push(EVIDENCE_LOW[0] + point[0] * (EVIDENCE_HIGH[0] - EVIDENCE_LOW[0]));
            live_positions.push(EVIDENCE_LOW[1] + point[1] * (EVIDENCE_HIGH[1] - EVIDENCE_LOW[1]));
        }
    }
    let expected_log_evidence =
        (2.0 * std::f64::consts::PI * EVIDENCE_SIGMA[0] * EVIDENCE_SIGMA[1]
            / ((EVIDENCE_HIGH[0] - EVIDENCE_LOW[0]) * (EVIDENCE_HIGH[1] - EVIDENCE_LOW[1])))
            .ln();

    Ok(NestedTraceOutput {
        log_evidence: output.log_evidence,
        log_evidence_error: output.log_evidence_error,
        expected_log_evidence,
        information: output.information,
        iterations: output.iterations as u32,
        converged: output.converged,
        live_points: live_points as u32,
        frame_count: events.len() as u32,
        parameter_names: "eg_ev,ln_b".into(),
        event_iterations: events.iter().map(|event| event.iteration as u32).collect(),
        event_log_evidence: events.iter().map(|event| event.log_evidence).collect(),
        event_thresholds: events
            .iter()
            .map(|event| event.log_likelihood_threshold)
            .collect(),
        event_log_prior_volume: events.iter().map(|event| event.log_prior_volume).collect(),
        live_positions,
        prior_low: EVIDENCE_LOW.to_vec(),
        prior_high: EVIDENCE_HIGH.to_vec(),
    })
}

/// Run the S2 calibration fixture and return the validated event stream.
#[wasm_bindgen]
pub fn run_nested_gaussian(live_points: u32, seed: u32) -> Result<NestedTraceOutput, JsValue> {
    run_nested_gaussian_impl(live_points, seed).map_err(|error| JsValue::from_str(&error))
}

fn model_from_name(name: &str) -> Result<Model, String> {
    match name.to_ascii_uppercase().as_str() {
        "M0" => Ok(Model::M0),
        "M1" => Ok(Model::M1),
        "M2" => Ok(Model::M2),
        "M3" => Ok(Model::M3),
        "M4" => Ok(Model::M4),
        other => Err(format!("unknown model {other:?}")),
    }
}

/// Posterior predictive summaries for the M0-M4 model family.
///
/// The input draws are parameter-major, matching `SamplerOutput.draws`. Each
/// draw is converted back to the model's latent coordinates before prediction.
/// The band includes the likelihood noise, while `lines` are noiseless curves
/// sampled from the same posterior for the S1 scene.
#[wasm_bindgen]
pub struct PredictiveOutput {
    mean: Vec<f64>,
    q05: Vec<f64>,
    q50: Vec<f64>,
    q95: Vec<f64>,
    lines: Vec<f64>,
    line_count: u32,
    point_count: u32,
}

#[wasm_bindgen]
impl PredictiveOutput {
    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> Vec<f64> {
        self.mean.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn q05(&self) -> Vec<f64> {
        self.q05.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn q50(&self) -> Vec<f64> {
        self.q50.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn q95(&self) -> Vec<f64> {
        self.q95.clone()
    }

    /// Line-major noiseless posterior curves: `line_count * point_count`.
    #[wasm_bindgen(getter)]
    pub fn lines(&self) -> Vec<f64> {
        self.lines.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn line_count(&self) -> u32 {
        self.line_count
    }

    #[wasm_bindgen(getter)]
    pub fn point_count(&self) -> u32 {
        self.point_count
    }
}

struct PredictiveRng {
    state: u64,
}

impl PredictiveRng {
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
        ((self.next_u64() >> 11) as f64) / 9_007_199_254_740_992.0
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

#[allow(clippy::too_many_arguments)]
fn posterior_predictive_impl(
    model: &str,
    temperature_k: &[f64],
    current_a: f64,
    sigma_v: f64,
    draws: &[f64],
    parameter_count: u32,
    draws_per_parameter: u32,
    seed: u32,
    line_count: u32,
) -> Result<PredictiveOutput, String> {
    if temperature_k.is_empty() {
        return Err("at least one temperature is required".into());
    }
    if parameter_count == 0 || draws_per_parameter < 1 {
        return Err("invalid predictive dimensions".into());
    }
    if !current_a.is_finite() || current_a <= 0.0 {
        return Err("current must be positive".into());
    }
    if !sigma_v.is_finite() || sigma_v <= 0.0 {
        return Err("likelihood sigma must be positive".into());
    }
    let parameter_count = parameter_count as usize;
    let draws_per_parameter = draws_per_parameter as usize;
    let expected = parameter_count
        .checked_mul(draws_per_parameter)
        .ok_or_else(|| "predictive dimensions overflow".to_string())?;
    if draws.len() != expected {
        return Err("predictive draw count does not match dimensions".into());
    }

    let model = model_from_name(model)?;
    let case = Case::new(
        model,
        temperature_k.to_vec(),
        vec![0.0; temperature_k.len()],
        current_a,
        sigma_v,
    )
    .map_err(|error| error.to_string())?;
    if parameter_count != case.parameter_names().len() {
        return Err("predictive parameter count does not match model".into());
    }

    let line_count = (line_count.max(1) as usize)
        .min(draws_per_parameter)
        .min(1_000);
    let point_count = temperature_k.len();
    let mut samples = vec![Vec::with_capacity(draws_per_parameter); point_count];
    let mut lines = vec![Vec::with_capacity(point_count); line_count];
    let mut line_cursor = 0usize;
    let mut rng = PredictiveRng::new(u64::from(seed));

    for draw_index in 0..draws_per_parameter {
        let mut reported = Vec::with_capacity(parameter_count);
        for parameter in 0..parameter_count {
            reported.push(draws[parameter * draws_per_parameter + draw_index]);
        }
        let latent = case.latent_from_reported(&reported);
        if latent.iter().any(|value| !value.is_finite()) {
            return Err("posterior draw could not be mapped to latent coordinates".into());
        }
        let curve = case.predicted_voltage(&latent);
        let is_line = line_cursor < line_count
            && draw_index == line_cursor * draws_per_parameter / line_count;
        for (point, base) in curve.iter().enumerate() {
            if !base.is_finite() {
                return Err("posterior predictive voltage is not finite".into());
            }
            samples[point].push(*base + sigma_v * rng.normal());
            if is_line {
                lines[line_cursor].push(*base);
            }
        }
        if is_line {
            line_cursor += 1;
        }
    }

    let mut mean = Vec::with_capacity(point_count);
    let mut q05 = Vec::with_capacity(point_count);
    let mut q50 = Vec::with_capacity(point_count);
    let mut q95 = Vec::with_capacity(point_count);
    for values in &mut samples {
        values.sort_by(f64::total_cmp);
        mean.push(values.iter().sum::<f64>() / values.len() as f64);
        q05.push(quantile(values, 0.05));
        q50.push(quantile(values, 0.5));
        q95.push(quantile(values, 0.95));
    }

    Ok(PredictiveOutput {
        mean,
        q05,
        q50,
        q95,
        lines: lines.into_iter().flatten().collect(),
        line_count: line_count as u32,
        point_count: point_count as u32,
    })
}

/// Generate predictive intervals and S1 posterior lines from a completed run.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn posterior_predictive(
    model: &str,
    temperature_k: &[f64],
    current_a: f64,
    sigma_v: f64,
    draws: &[f64],
    parameter_count: u32,
    draws_per_parameter: u32,
    seed: u32,
    line_count: u32,
) -> Result<PredictiveOutput, JsValue> {
    posterior_predictive_impl(
        model,
        temperature_k,
        current_a,
        sigma_v,
        draws,
        parameter_count,
        draws_per_parameter,
        seed,
        line_count,
    )
    .map_err(|error| JsValue::from_str(&error))
}

#[allow(clippy::too_many_arguments)]
fn run_sampler_impl(
    model: &str,
    temperature_k: &[f64],
    voltage_v: &[f64],
    current_a: f64,
    sigma_v: f64,
    chains: u32,
    warmup: u32,
    draws: u32,
    seed: u32,
    pilot_seed: u32,
    target_accept: f64,
    initial_step: f64,
    mean_trajectory_steps: f64,
    energy_error_limit: f64,
) -> Result<SamplerOutput, String> {
    let case = Case::new(
        model_from_name(model)?,
        temperature_k.to_vec(),
        voltage_v.to_vec(),
        current_a,
        sigma_v,
    )
    .map_err(|error| error.to_string())?;
    let config = MclmcConfig {
        chains: chains as usize,
        warmup: warmup as usize,
        draws: draws as usize,
        seed: u64::from(seed),
        pilot_seed: u64::from(pilot_seed),
        target_accept,
        initial_step,
        mean_trajectory_steps,
        energy_error_limit,
    };
    bgo_infer::run_mclmc(&case, &config)
        .map(SamplerOutput::new)
        .map_err(|error| error.to_string())
}

/// Run the browser sampler on a synthetic case.
///
/// `passes` is the fail-closed diagnostic gate. Callers must not render a
/// converged state when it is false.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn run_sampler(
    model: &str,
    temperature_k: &[f64],
    voltage_v: &[f64],
    current_a: f64,
    sigma_v: f64,
    chains: u32,
    warmup: u32,
    draws: u32,
    seed: u32,
    pilot_seed: u32,
    target_accept: f64,
    initial_step: f64,
    mean_trajectory_steps: f64,
    energy_error_limit: f64,
) -> Result<SamplerOutput, JsValue> {
    run_sampler_impl(
        model,
        temperature_k,
        voltage_v,
        current_a,
        sigma_v,
        chains,
        warmup,
        draws,
        seed,
        pilot_seed,
        target_accept,
        initial_step,
        mean_trajectory_steps,
        energy_error_limit,
    )
    .map_err(|error| JsValue::from_str(&error))
}

/// Standardise a model target so one isotropic RBF bandwidth is meaningful
/// across parameters with very different prior scales.
struct StandardizedCaseTarget<'a> {
    case: &'a Case,
    center: Vec<f64>,
    scale: Vec<f64>,
}

impl SvgdTarget for StandardizedCaseTarget<'_> {
    fn dimension(&self) -> usize {
        self.case.latent_len()
    }

    fn logp_and_grad(&self, standardised: &[f64]) -> (f64, Vec<f64>) {
        if standardised.len() != self.center.len() {
            return (f64::NEG_INFINITY, vec![0.0; self.center.len()]);
        }
        let theta = self
            .center
            .iter()
            .zip(&self.scale)
            .zip(standardised)
            .map(|((center, scale), value)| center + scale * value)
            .collect::<Vec<_>>();
        let (logp, gradient) = self.case.logp_and_grad(&theta);
        let gradient = gradient
            .iter()
            .zip(&self.scale)
            .map(|(value, scale)| value * scale)
            .collect();
        (logp, gradient)
    }
}

struct SvgdRng(u64);

impl SvgdRng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
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
}

/// SVGD particle trajectory and the MCLMC/W2 comparison for S3.
#[wasm_bindgen]
pub struct SvgdOutput {
    parameter_names: String,
    parameter_count: u32,
    particle_count: u32,
    frame_count: u32,
    iterations: u32,
    trajectory: Vec<f64>,
    particles: Vec<f64>,
    svgd_mean: Vec<f64>,
    svgd_variance: Vec<f64>,
    mclmc_mean: Vec<f64>,
    mclmc_variance: Vec<f64>,
    mclmc_particles: Vec<f64>,
    mclmc_particle_count: u32,
    w2_distance: f64,
    w2_bootstrap_se: f64,
    bandwidth: f64,
    step_size: f64,
    final_logp_mean: f64,
}

#[wasm_bindgen]
impl SvgdOutput {
    #[wasm_bindgen(getter)]
    pub fn parameter_names(&self) -> String {
        self.parameter_names.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn parameter_count(&self) -> u32 {
        self.parameter_count
    }

    #[wasm_bindgen(getter)]
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }

    #[wasm_bindgen(getter)]
    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    /// Frame-major, particle-major, parameter-major reported coordinates.
    #[wasm_bindgen(getter)]
    pub fn trajectory(&self) -> Vec<f64> {
        self.trajectory.clone()
    }

    /// Final particles, particle-major, reported coordinates.
    #[wasm_bindgen(getter)]
    pub fn particles(&self) -> Vec<f64> {
        self.particles.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn svgd_mean(&self) -> Vec<f64> {
        self.svgd_mean.clone()
    }

    /// Population variance of the SVGD cloud, in reported units.
    #[wasm_bindgen(getter)]
    pub fn svgd_variance(&self) -> Vec<f64> {
        self.svgd_variance.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn mclmc_mean(&self) -> Vec<f64> {
        self.mclmc_mean.clone()
    }

    /// Population variance of the MCLMC draws, in reported units.
    #[wasm_bindgen(getter)]
    pub fn mclmc_variance(&self) -> Vec<f64> {
        self.mclmc_variance.clone()
    }

    /// Evenly subsampled MCLMC draws used for the visual overlay and W2.
    #[wasm_bindgen(getter)]
    pub fn mclmc_particles(&self) -> Vec<f64> {
        self.mclmc_particles.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn mclmc_particle_count(&self) -> u32 {
        self.mclmc_particle_count
    }

    /// Sliced-W2 in prior-standardised units.
    #[wasm_bindgen(getter)]
    pub fn w2_distance(&self) -> f64 {
        self.w2_distance
    }

    #[wasm_bindgen(getter)]
    pub fn w2_bootstrap_se(&self) -> f64 {
        self.w2_bootstrap_se
    }

    #[wasm_bindgen(getter)]
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    #[wasm_bindgen(getter)]
    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    #[wasm_bindgen(getter)]
    pub fn final_logp_mean(&self) -> f64 {
        self.final_logp_mean
    }
}

#[allow(clippy::too_many_arguments)]
fn run_svgd_comparison_impl(
    model: &str,
    temperature_k: &[f64],
    voltage_v: &[f64],
    current_a: f64,
    sigma_v: f64,
    mclmc_draws: &[f64],
    parameter_count: u32,
    draws_per_parameter: u32,
    particles: u32,
    iterations: u32,
    step_size: f64,
    bandwidth: f64,
    seed: u32,
    trace_frames: u32,
) -> Result<SvgdOutput, String> {
    let case = Case::new(
        model_from_name(model)?,
        temperature_k.to_vec(),
        voltage_v.to_vec(),
        current_a,
        sigma_v,
    )
    .map_err(|error| error.to_string())?;
    if case.model != Model::M0 {
        return Err("S3 currently uses the two-parameter M0 target".into());
    }
    let dimension = case.latent_len();
    let parameter_count = parameter_count as usize;
    let draws_per_parameter = draws_per_parameter as usize;
    if parameter_count != dimension {
        return Err("MCLMC parameter count does not match the S3 target".into());
    }
    if draws_per_parameter < 2
        || mclmc_draws.len() != parameter_count.saturating_mul(draws_per_parameter)
    {
        return Err("MCLMC draw dimensions do not match the S3 target".into());
    }
    if !(8..=512).contains(&particles) {
        return Err("S3 particle count must be between 8 and 512".into());
    }
    if !(1..=2_000).contains(&iterations) {
        return Err("S3 iterations must be between 1 and 2000".into());
    }
    if !(2..=iterations).contains(&trace_frames) {
        return Err("S3 trace frames must be between 2 and the iteration count".into());
    }
    if !step_size.is_finite() || step_size <= 0.0 || !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err("S3 step size and bandwidth must be positive and finite".into());
    }

    let center = case.prior_center();
    let scale = case.prior_scales();
    let target = StandardizedCaseTarget {
        case: &case,
        center: center.clone(),
        scale: scale.clone(),
    };
    let mut rng = SvgdRng::new(u64::from(seed));
    let mut initial = Vec::with_capacity(particles as usize);
    for _ in 0..particles {
        let mut particle = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            particle.push(rng.normal().clamp(-2.5, 2.5));
        }
        initial.push(particle);
    }
    let trace_every = (iterations as usize / trace_frames as usize).max(1);
    let svgd = run_svgd(
        &target,
        &initial,
        &SvgdConfig {
            iterations: iterations as usize,
            step_size,
            bandwidth,
            trace_every,
        },
    )
    .map_err(|error| error.to_string())?;

    let mut trajectory = Vec::with_capacity(svgd.trajectory.len() * particles as usize * dimension);
    for frame in &svgd.trajectory {
        for particle in frame {
            for (coordinate, value) in particle.iter().enumerate() {
                trajectory.push(center[coordinate] + scale[coordinate] * value);
            }
        }
    }
    let particles_flat = svgd
        .particles
        .iter()
        .flat_map(|particle| {
            particle
                .iter()
                .enumerate()
                .map(|(coordinate, value)| center[coordinate] + scale[coordinate] * value)
        })
        .collect::<Vec<_>>();
    let svgd_mean = center
        .iter()
        .zip(&scale)
        .zip(&svgd.mean)
        .map(|((center, scale), value)| center + scale * value)
        .collect::<Vec<_>>();
    let svgd_variance = scale
        .iter()
        .zip(&svgd.variance)
        .map(|(scale, value)| scale * scale * value)
        .collect::<Vec<_>>();

    let mut mclmc_mean = vec![0.0; parameter_count];
    let mut mclmc_variance = vec![0.0; parameter_count];
    for parameter in 0..parameter_count {
        let values = (0..draws_per_parameter)
            .map(|draw| mclmc_draws[parameter * draws_per_parameter + draw])
            .collect::<Vec<_>>();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| {
                let difference = value - mean;
                difference * difference
            })
            .sum::<f64>()
            / values.len() as f64;
        mclmc_mean[parameter] = mean;
        mclmc_variance[parameter] = variance;
    }

    let comparison_count = (particles as usize).min(draws_per_parameter);
    let mut mclmc_particles = Vec::with_capacity(comparison_count * parameter_count);
    let mut mclmc_standardised = Vec::with_capacity(comparison_count);
    for index in 0..comparison_count {
        let draw = index * draws_per_parameter / comparison_count;
        let mut point = Vec::with_capacity(parameter_count);
        let mut standardised = Vec::with_capacity(parameter_count);
        for parameter in 0..parameter_count {
            let value = mclmc_draws[parameter * draws_per_parameter + draw];
            point.push(value);
            standardised.push((value - center[parameter]) / scale[parameter]);
        }
        mclmc_particles.extend(point);
        mclmc_standardised.push(standardised);
    }

    let w2 = sliced_wasserstein_2(
        &svgd.particles,
        &mclmc_standardised,
        96,
        128,
        u64::from(seed) ^ 0x9e37_79b9_7f4a_7c15,
    )
    .map_err(|error| error.to_string())?;

    Ok(SvgdOutput {
        parameter_names: case.parameter_names().join(","),
        parameter_count: parameter_count as u32,
        particle_count: particles,
        frame_count: svgd.trajectory.len() as u32,
        iterations,
        trajectory,
        particles: particles_flat,
        svgd_mean,
        svgd_variance,
        mclmc_mean,
        mclmc_variance,
        mclmc_particles,
        mclmc_particle_count: comparison_count as u32,
        w2_distance: w2.distance,
        w2_bootstrap_se: w2.bootstrap_standard_error,
        bandwidth,
        step_size,
        final_logp_mean: svgd.final_logp_mean,
    })
}

/// Run SVGD on the M0 target and compare its final cloud with MCLMC draws.
///
/// The W2 distance is computed in prior-standardised coordinates so the
/// `E_g` and `ln B` scales cannot dominate the comparison arbitrarily.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn run_svgd_comparison(
    model: &str,
    temperature_k: &[f64],
    voltage_v: &[f64],
    current_a: f64,
    sigma_v: f64,
    mclmc_draws: &[f64],
    parameter_count: u32,
    draws_per_parameter: u32,
    particles: u32,
    iterations: u32,
    step_size: f64,
    bandwidth: f64,
    seed: u32,
    trace_frames: u32,
) -> Result<SvgdOutput, JsValue> {
    run_svgd_comparison_impl(
        model,
        temperature_k,
        voltage_v,
        current_a,
        sigma_v,
        mclmc_draws,
        parameter_count,
        draws_per_parameter,
        particles,
        iterations,
        step_size,
        bandwidth,
        seed,
        trace_frames,
    )
    .map_err(|error| JsValue::from_str(&error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_reported() {
        assert!(core_version().starts_with("bgo-wasm"));
    }

    #[test]
    fn diagnostics_accept_chain_major_parameter_major_draws() {
        // Two chains, one parameter, four draws per chain. The exact pass/fail
        // value is not the point here; the layout and lengths are.
        let draws = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let output = diagnose_chains(2, 1, 4, &draws, 0.01, 0).expect("diagnostics");
        assert_eq!(output.rhat().len(), 1);
        assert_eq!(output.ess_bulk().len(), 1);
        assert_eq!(output.energy_error_mean(), 0.01);
        assert_eq!(output.divergences(), 0);

        let broken = diagnose_chains(2, 1, 4, &draws, 0.01, 1).expect("diagnostics");
        assert!(!broken.passes());
    }

    #[test]
    fn sampler_returns_flat_parameter_major_draws() {
        let temperature = [300.0, 320.0, 340.0];
        let voltage = [0.650, 0.630, 0.610];
        let output = run_sampler_impl(
            "M0",
            &temperature,
            &voltage,
            3.0e-4,
            1.2e-4,
            2,
            20,
            20,
            7,
            7,
            0.995,
            0.4,
            16.0,
            0.1,
        )
        .expect("sampler");
        assert_eq!(output.parameter_names(), "eg_ev,ln_b");
        assert_eq!(output.parameter_count(), 2);
        assert_eq!(output.draws_per_parameter(), 40);
        assert_eq!(output.draws().len(), 80);
        assert_eq!(output.chain_count(), 2);
        assert_eq!(output.draws_per_chain(), 20);
        assert_eq!(output.energy_error_trace().len(), 40);
        assert!(output.draws().iter().all(|value| value.is_finite()));
        assert!(output
            .energy_error_trace()
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(output.rhat().len(), 2);
        assert_eq!(output.ess_bulk().len(), 2);
    }

    #[test]
    fn fisher_report_recovers_the_m0_ridge() {
        let output = fisher_report(
            "M0",
            &[293.15, 320.0, 350.0, 383.15],
            3.0e-4,
            1.253e-4,
            &[1.002, 5.504],
        )
        .expect("fisher report");
        assert_eq!(output.parameter_count(), 2);
        assert_eq!(output.eigenvalues().len(), 2);
        assert!(output.eigenvalues()[0] > output.eigenvalues()[1]);
        assert!(output.condition_number() > 1.0);
        assert!(output.ellipse_1()[0] >= output.ellipse_1()[1]);
    }

    #[test]
    fn nested_trace_has_one_frame_per_iteration() {
        let output = run_nested_gaussian_impl(64, 0x5eed).expect("nested trace");
        assert_eq!(
            output.event_iterations().len(),
            output.iterations() as usize
        );
        assert_eq!(
            output.event_log_evidence().len(),
            output.iterations() as usize
        );
        assert_eq!(
            output.live_positions().len(),
            output.frame_count() as usize * output.live_points() as usize * 2
        );
        assert!(output.log_evidence().is_finite());
        assert!((output.log_evidence() - output.expected_log_evidence()).abs() < 0.5);
    }

    #[test]
    fn svgd_comparison_returns_a_finite_w2() {
        let temperature = [293.15, 320.0, 350.0, 383.15];
        let voltage = [0.65775, 0.65025, 0.6409, 0.6298];
        let draws_per_parameter = 64usize;
        let mut draws = Vec::with_capacity(2 * draws_per_parameter);
        for parameter in 0..2 {
            for draw in 0..draws_per_parameter {
                let offset = (draw as f64 - 32.0) * 0.0005;
                draws.push(if parameter == 0 {
                    1.002 + offset
                } else {
                    5.504 + offset * 4.0
                });
            }
        }
        let output = run_svgd_comparison_impl(
            "M0",
            &temperature,
            &voltage,
            3.0e-4,
            1.253_079_539_906_918e-4,
            &draws,
            2,
            draws_per_parameter as u32,
            64,
            120,
            0.08,
            0.5,
            7,
            24,
        )
        .expect("SVGD comparison");
        assert_eq!(output.parameter_count(), 2);
        assert_eq!(output.particle_count(), 64);
        assert_eq!(output.mclmc_particle_count(), 64);
        assert_eq!(output.frame_count(), 25);
        assert!(output.trajectory().iter().all(|value| value.is_finite()));
        assert!(output.svgd_mean().iter().all(|value| value.is_finite()));
        assert!(output.svgd_variance().iter().all(|value| value.is_finite()));
        assert!(output.mclmc_mean().iter().all(|value| value.is_finite()));
        assert!(output
            .mclmc_variance()
            .iter()
            .all(|value| value.is_finite()));
        assert!(output.w2_distance().is_finite());
        assert!(output.w2_bootstrap_se().is_finite());
    }

    #[test]
    fn posterior_predictive_returns_intervals_and_lines() {
        let draws = [
            1.00, 1.01, 0.99, 1.02, // eg_ev, parameter-major
            5.00, 5.01, 4.99, 5.02, // ln_b
        ];
        let output =
            posterior_predictive_impl("M0", &[300.0, 340.0], 3.0e-4, 1.253e-4, &draws, 2, 4, 7, 2)
                .expect("predictive");
        assert_eq!(output.point_count(), 2);
        assert_eq!(output.line_count(), 2);
        assert_eq!(output.lines().len(), 4);
        assert!(output
            .mean()
            .iter()
            .chain(output.q05().iter())
            .chain(output.q50().iter())
            .chain(output.q95().iter())
            .chain(output.lines().iter())
            .all(|value| value.is_finite()));
        for index in 0..output.point_count() as usize {
            assert!(output.q05()[index] <= output.q50()[index]);
            assert!(output.q50()[index] <= output.q95()[index]);
        }
    }
}
