use std::fs;
use std::path::PathBuf;

use bgo_infer::model::{Case, Model};
use bgo_infer::{run_mclmc, MclmcConfig};

const CASES: &[&str] = &["m0-report-si", "m0-report-ge", "m1", "m2", "m3", "m4"];

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../train/references")
        .canonicalize()
        .expect("reference directory")
}

fn case_from_json(name: &str) -> (Case, serde_json::Value) {
    let path = reference_dir().join(format!("{name}.json"));
    let text = fs::read_to_string(&path).expect("read reference");
    let document: serde_json::Value = serde_json::from_str(&text).expect("parse reference");
    let case = &document["case"];
    let observations = &document["observations"];

    let model = match case["model"].as_str().expect("model") {
        "M0" => Model::M0,
        "M1" => Model::M1,
        "M2" => Model::M2,
        "M3" => Model::M3,
        "M4" => Model::M4,
        other => panic!("unknown model {other}"),
    };
    let temperatures = observations["temperature_k"]
        .as_array()
        .expect("temperatures")
        .iter()
        .map(|value| value.as_f64().expect("temperature"))
        .collect::<Vec<_>>();
    let voltages = observations["voltage_v"]
        .as_array()
        .expect("voltages")
        .iter()
        .map(|value| value.as_f64().expect("voltage"))
        .collect::<Vec<_>>();
    let current = observations["current_a"].as_f64().expect("current");
    let sigma = observations["noise"]["likelihood_sigma_v"]
        .as_f64()
        .expect("likelihood sigma");

    (
        Case::new(model, temperatures, voltages, current, sigma).expect("valid case"),
        document,
    )
}

#[test]
fn mclmc_matches_the_committed_numpyro_posteriors() {
    let filter = std::env::var("BGO_CASE").ok();
    for name in CASES {
        if filter.as_deref().is_some_and(|value| value != *name) {
            continue;
        }
        let config = MclmcConfig::default();
        let (case, document) = case_from_json(name);
        let reference = &document["reference"];
        let names = reference["parameters"]
            .as_array()
            .expect("parameter names")
            .iter()
            .map(|value| value.as_str().expect("name"))
            .collect::<Vec<_>>();

        let output = run_mclmc(&case, &config)
            .unwrap_or_else(|error| panic!("{name}: sampler failed: {error}"));
        println!(
            "{name}: acceptance={:.3} energy_error={:.4} rhat_max={:.4} ess_min={:.1}",
            output.acceptance_rate,
            output.energy_error_mean,
            output.diagnostics.rhat_max,
            output.diagnostics.ess_bulk_min,
        );
        assert!(
            output.diagnostics.passes,
            "{name}: diagnostics failed: {:?}",
            output.diagnostics
        );
        assert_eq!(output.parameter_names, names);

        let draws = output.by_parameter();
        for (index, parameter) in names.iter().enumerate() {
            let expected_mean = reference["mean"][*parameter]
                .as_f64()
                .expect("reference mean");
            let expected_sd = reference["sd"][*parameter].as_f64().expect("reference sd");
            let observed = &draws[index];
            let mean = mean(observed);
            let sd = standard_deviation(observed);
            println!(
                "{name}.{parameter}: mean={mean:.10} vs {expected_mean:.10} (delta={:.10}, tol={:.10}), sd={sd:.10} vs {expected_sd:.10}",
                mean - expected_mean,
                0.05 * expected_sd
            );
            assert!(
                (mean - expected_mean).abs() <= 0.05 * expected_sd,
                "{name}.{parameter}: mean {mean} vs {expected_mean}, tolerance {}",
                0.05 * expected_sd
            );
            assert!(
                (sd / expected_sd - 1.0).abs() <= 0.10,
                "{name}.{parameter}: sd {sd} vs {expected_sd}"
            );
        }

        let observed_correlation = correlation_matrix(&draws);
        let expected_correlation = reference["correlation"]["matrix"]
            .as_array()
            .expect("correlation matrix");
        let mut max_difference = 0.0_f64;
        for (row, expected_row) in expected_correlation.iter().enumerate() {
            let expected_row = expected_row.as_array().expect("correlation row");
            for (column, expected) in expected_row.iter().enumerate() {
                let expected = expected.as_f64().expect("correlation");
                max_difference =
                    max_difference.max((observed_correlation[row][column] - expected).abs());
            }
        }
        assert!(
            max_difference <= 0.10,
            "{name}: max correlation difference {max_difference}"
        );

        println!(
            "{name}: acceptance={:.3} energy_error={:.4} rhat_max={:.4} ess_min={:.1} corr_diff={max_difference:.4}",
            output.acceptance_rate,
            output.energy_error_mean,
            output.diagnostics.rhat_max,
            output.diagnostics.ess_bulk_min,
        );
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn standard_deviation(values: &[f64]) -> f64 {
    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    variance.sqrt()
}

fn correlation_matrix(parameters: &[Vec<f64>]) -> Vec<Vec<f64>> {
    parameters
        .iter()
        .map(|left| {
            parameters
                .iter()
                .map(|right| correlation(left, right))
                .collect()
        })
        .collect()
}

fn correlation(left: &[f64], right: &[f64]) -> f64 {
    let left_mean = mean(left);
    let right_mean = mean(right);
    let numerator = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - left_mean) * (right - right_mean))
        .sum::<f64>();
    let left_norm = left
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>()
        .sqrt();
    numerator / (left_norm * right_norm)
}
