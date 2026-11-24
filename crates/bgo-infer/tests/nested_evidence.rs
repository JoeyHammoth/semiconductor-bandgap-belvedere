use bgo_infer::{run_nested, LogLikePrior, NestedConfig};
use serde_json::Value;
use std::f64::consts::PI;
use std::fs;

struct Gaussian {
    sigma: f64,
}

impl LogLikePrior for Gaussian {
    fn dimension(&self) -> usize {
        2
    }

    fn log_like_prior(&self, theta: &[f64]) -> f64 {
        let first = (theta[0] - 0.5) / self.sigma;
        let second = (theta[1] - 0.5) / self.sigma;
        -0.5 * (first * first + second * second)
    }
}

fn gaussian_config(seed: u64) -> NestedConfig {
    NestedConfig {
        seed,
        slice_steps: 200,
        ..NestedConfig::default()
    }
}

fn run_gaussian(seed: u64) -> (f64, f64) {
    let target = Gaussian { sigma: 0.1 };
    let output =
        run_nested(&target, &gaussian_config(seed), |_| {}).expect("nested sampling should run");
    (output.log_evidence, output.log_evidence_error)
}

#[test]
fn closed_form_gaussian_evidence_is_recovered() {
    let target = Gaussian { sigma: 0.1 };
    let config = gaussian_config(0x5eed);
    let output = run_nested(&target, &config, |_| {}).expect("nested sampling should run");
    let expected = (2.0 * PI * target.sigma * target.sigma).ln();
    assert!(
        (output.log_evidence - expected).abs() <= 0.2,
        "ln Z = {} expected {expected}; iterations={} converged={} error={}",
        output.log_evidence,
        output.iterations,
        output.converged,
        output.log_evidence_error
    );
}

#[test]
fn committed_dynesty_reference_agrees() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/references/nested_gaussian_dynesty.json"
    );
    let document: Value = serde_json::from_str(
        &fs::read_to_string(path).expect("the committed dynesty reference should be readable"),
    )
    .expect("the dynesty reference should be valid JSON");
    assert_eq!(document["synthetic"].as_bool(), Some(true));
    let reference = document["log_evidence"]
        .as_f64()
        .expect("the dynesty reference should contain log_evidence");
    let (actual, error) = run_gaussian(0x5eed);
    assert!(
        (actual - reference).abs() <= 0.3,
        "ln Z = {actual} vs dynesty {reference}; own error={error}"
    );
}

#[test]
fn replicate_scatter_matches_information_bound() {
    let seeds = [0x5eed, 0x5eee, 0x5eef, 0x5ef0, 0x5ef1, 0x5ef2];
    let runs = seeds.map(run_gaussian);
    let values: Vec<f64> = runs.iter().map(|(log_evidence, _)| *log_evidence).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let observed_sd = variance.sqrt();
    let theoretical_sd = runs.iter().map(|(_, error)| error).sum::<f64>() / runs.len() as f64;
    assert!(
        observed_sd >= 0.5 * theoretical_sd && observed_sd <= 2.0 * theoretical_sd,
        "observed sd {observed_sd} outside [0.5, 2]x theoretical {theoretical_sd}; values={values:?}"
    );
}
