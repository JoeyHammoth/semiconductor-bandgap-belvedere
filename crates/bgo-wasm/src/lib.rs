//! WebAssembly surface for the Browser core.
//!
//! The JavaScript side speaks typed arrays; configuration crosses as small
//! structs. Each worker owns its own instance, so nothing here may hold global
//! mutable state.

#![forbid(unsafe_code)]

use bgo_infer::{Case, MclmcConfig, MclmcOutput, Model};
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
        assert!(output.draws().iter().all(|value| value.is_finite()));
        assert_eq!(output.rhat().len(), 2);
        assert_eq!(output.ess_bulk().len(), 2);
    }
}
