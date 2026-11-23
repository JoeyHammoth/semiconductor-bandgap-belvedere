use std::fs;
use std::path::PathBuf;

use bgo_infer::model::{Case, Model};

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
fn target_gradient_matches_finite_differences_at_the_reference_truth() {
    for name in ["m0-report-si", "m0-report-ge", "m1", "m2", "m3", "m4"] {
        let (case, document) = case_from_json(name);
        let truth = &document["truth"];
        let reported = case
            .parameter_names()
            .iter()
            .map(|parameter| truth[*parameter].as_f64().expect("truth value"))
            .collect::<Vec<_>>();
        let latent = case.latent_from_reported(&reported);

        let (value, gradient) = case.logp_and_grad(&latent);
        assert!(value.is_finite(), "{name}: log density is not finite");
        assert_eq!(gradient.len(), latent.len());

        for (index, analytic) in gradient.iter().enumerate() {
            let step = 1e-6 * latent[index].abs().max(1.0);
            let mut plus = latent.clone();
            let mut minus = latent.clone();
            plus[index] += step;
            minus[index] -= step;
            let finite_difference =
                (case.logp_value(&plus) - case.logp_value(&minus)) / (2.0 * step);
            assert!(
                (analytic - finite_difference).abs() <= 1e-5 * finite_difference.abs().max(1.0),
                "{name}[{index}]: analytic {analytic} vs finite difference {finite_difference}"
            );
        }
    }
}

#[test]
fn m2_fixed_point_log_density_difference_matches_numpyro() {
    let (case, document) = case_from_json("m2");
    let reference = &document["reference"];
    let names = reference["parameters"]
        .as_array()
        .expect("parameter names")
        .iter()
        .map(|value| value.as_str().expect("name"))
        .collect::<Vec<_>>();
    let mean = names
        .iter()
        .map(|name| reference["mean"][*name].as_f64().expect("mean"))
        .collect::<Vec<_>>();
    let sd = names
        .iter()
        .map(|name| reference["sd"][*name].as_f64().expect("sd"))
        .collect::<Vec<_>>();
    let ln_b_index = names.iter().position(|name| *name == "ln_b").expect("ln_b");
    let mut shifted = mean.clone();
    shifted[ln_b_index] += 0.01 * sd[ln_b_index];

    let mean_logp = case.logp_value(&case.latent_from_reported(&mean));
    let shifted_logp = case.logp_value(&case.latent_from_reported(&shifted));
    // Hold beta fixed so the endpoint-gap Jacobian cancels. NumPyro evaluates
    // these physical points 256.2554176464834 apart in log density. Only the
    // difference is compared because the target constants (truncated-prior and
    // likelihood normalisation) do not enter the sampler.
    let expected = -256.255_417_646_5;
    let observed = mean_logp - shifted_logp;
    assert!(
        (observed - expected).abs() <= 1e-5,
        "M2 log-density difference {observed} vs NumPyro {expected}"
    );
}
