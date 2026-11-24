use bgo_infer::{Case, Model};
use bgo_info::{analyze_fisher, fisher_from_jacobian};
use serde_json::{json, Value};
use std::fs;

fn main() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../train/references/m0-report-si.json"
    );
    let document: Value = serde_json::from_str(
        &fs::read_to_string(path).expect("the committed M0 reference should exist"),
    )
    .expect("valid reference JSON");
    let observations = &document["observations"];
    let temperature = observations["temperature_k"]
        .as_array()
        .expect("temperatures")
        .iter()
        .map(|value| value.as_f64().expect("numeric temperature"))
        .collect::<Vec<_>>();
    let voltage = observations["voltage_v"]
        .as_array()
        .expect("voltages")
        .iter()
        .map(|value| value.as_f64().expect("numeric voltage"))
        .collect::<Vec<_>>();
    let case = Case::new(
        Model::M0,
        temperature,
        voltage,
        observations["current_a"].as_f64().expect("current"),
        observations["noise"]["likelihood_sigma_v"]
            .as_f64()
            .expect("sigma"),
    )
    .expect("valid M0 case");
    let mean = &document["reference"]["mean"];
    let reported = vec![
        mean["eg_ev"].as_f64().expect("mean eg"),
        mean["ln_b"].as_f64().expect("mean ln b"),
    ];
    let jacobian = case
        .observation_jacobian_reported(&reported)
        .expect("reported Jacobian");
    let report =
        analyze_fisher(&fisher_from_jacobian(&jacobian, case.sigma_v).expect("Fisher matrix"))
            .expect("Fisher analysis");
    let names = ["eg_ev", "ln_b"];
    let fisher_sd = names
        .iter()
        .enumerate()
        .map(|(index, _)| {
            report
                .covariance
                .get(index, index)
                .expect("diagonal")
                .sqrt()
        })
        .collect::<Vec<_>>();
    let posterior_sd = names
        .iter()
        .map(|name| document["reference"]["sd"][*name].as_f64().expect("sd"))
        .collect::<Vec<_>>();
    let relative_error = fisher_sd
        .iter()
        .zip(&posterior_sd)
        .map(|(fisher, posterior)| (fisher - posterior).abs() / posterior)
        .collect::<Vec<_>>();
    let fisher_correlation = report.covariance.get(0, 1).expect("off-diagonal")
        / (report.covariance.get(0, 0).expect("diagonal")
            * report.covariance.get(1, 1).expect("diagonal"))
        .sqrt();
    let posterior_correlation = document["reference"]["correlation"]["matrix"][0][1]
        .as_f64()
        .expect("correlation");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "synthetic": true,
            "case_id": "m0-report-si",
            "source_reference": "train/references/m0-report-si.json",
            "parameters": names,
            "fisher_sd": fisher_sd,
            "posterior_sd": posterior_sd,
            "relative_sd_error": relative_error,
            "fisher_correlation": fisher_correlation,
            "posterior_correlation": posterior_correlation,
            "condition_number": report.condition_number,
        }))
        .expect("serialize comparison")
    );
}
