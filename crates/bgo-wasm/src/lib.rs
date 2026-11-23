//! WebAssembly surface for the Browser core.
//!
//! The JavaScript side speaks typed arrays; configuration crosses as small
//! structs. Each worker owns its own instance, so nothing here may hold global
//! mutable state.

#![forbid(unsafe_code)]

use bgo_infer::{bulk_ess, split_rhat, Case, Diagnostics, MclmcConfig, MclmcOutput, Model};
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
        target_accept,
        initial_step,
        mean_trajectory_steps,
        energy_error_limit,
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
