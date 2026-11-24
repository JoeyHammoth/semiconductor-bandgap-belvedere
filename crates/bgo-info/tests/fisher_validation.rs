use bgo_infer::{Case, Model};
use bgo_info::{analyze_fisher, covariance_ellipse, fisher_from_jacobian};
use serde_json::Value;
use std::f64::consts::TAU;
use std::fs;

const BOLTZMANN: f64 = 1.380_649e-23;
const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;

fn reference() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../train/references/m0-report-si.json"
    );
    serde_json::from_str(
        &fs::read_to_string(path).expect("the committed M0 reference should exist"),
    )
    .expect("the committed M0 reference should be valid JSON")
}

fn temperature(case: &Value) -> Vec<f64> {
    case["observations"]["temperature_k"]
        .as_array()
        .expect("temperatures")
        .iter()
        .map(|value| value.as_f64().expect("numeric temperature"))
        .collect()
}

fn voltage(case: &Value) -> Vec<f64> {
    case["observations"]["voltage_v"]
        .as_array()
        .expect("voltages")
        .iter()
        .map(|value| value.as_f64().expect("numeric voltage"))
        .collect()
}

fn case_from(document: &Value) -> Case {
    let observations = &document["observations"];
    Case::new(
        Model::M0,
        temperature(document),
        voltage(document),
        observations["current_a"].as_f64().expect("current"),
        observations["noise"]["likelihood_sigma_v"]
            .as_f64()
            .expect("sigma"),
    )
    .expect("valid M0 case")
}

#[test]
fn fisher_inverse_matches_the_committed_m0_posterior() {
    let document = reference();
    let case = case_from(&document);
    let reference_mean = &document["reference"]["mean"];
    let reported = vec![
        reference_mean["eg_ev"].as_f64().expect("mean eg"),
        reference_mean["ln_b"].as_f64().expect("mean ln b"),
    ];
    let jacobian = case
        .observation_jacobian_reported(&reported)
        .expect("reported Jacobian");
    let sigma = case.sigma_v;
    let fisher = fisher_from_jacobian(&jacobian, sigma).expect("Fisher matrix");
    let report = analyze_fisher(&fisher).expect("Fisher analysis");

    let names = ["eg_ev", "ln_b"];
    for (index, name) in names.iter().enumerate() {
        let fisher_sd = report
            .covariance
            .get(index, index)
            .expect("diagonal")
            .sqrt();
        let posterior_sd = document["reference"]["sd"][*name]
            .as_f64()
            .expect("reference sd");
        let relative = (fisher_sd - posterior_sd).abs() / posterior_sd;
        assert!(
            relative <= 0.10,
            "{name}: Fisher sd {fisher_sd} vs posterior {posterior_sd} ({:.2}%)",
            relative * 100.0
        );
    }

    let fisher_correlation = report.covariance.get(0, 1).expect("off-diagonal")
        / (report.covariance.get(0, 0).expect("diagonal")
            * report.covariance.get(1, 1).expect("diagonal"))
        .sqrt();
    let posterior_correlation = document["reference"]["correlation"]["matrix"][0][1]
        .as_f64()
        .expect("posterior correlation");
    assert!(
        (fisher_correlation - posterior_correlation).abs() <= 0.1,
        "Fisher correlation {fisher_correlation} vs posterior {posterior_correlation}"
    );

    let ellipse = covariance_ellipse(&report.covariance, 0, 1, 1.0).expect("ellipse");
    assert!(ellipse.semi_major >= ellipse.semi_minor);
}

#[test]
fn committed_comparison_table_records_the_m0_check() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/references/fisher_m0_comparison.json"
    );
    let table: Value = serde_json::from_str(&fs::read_to_string(path).expect("comparison table"))
        .expect("valid comparison JSON");
    assert_eq!(table["synthetic"].as_bool(), Some(true));
    let errors = table["relative_sd_error"]
        .as_array()
        .expect("relative errors");
    assert!(errors.iter().all(|value| value.as_f64().unwrap() <= 0.10));
    let fisher_correlation = table["fisher_correlation"].as_f64().expect("correlation");
    let posterior_correlation = table["posterior_correlation"]
        .as_f64()
        .expect("posterior correlation");
    assert!((fisher_correlation - posterior_correlation).abs() <= 0.1);
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let first = self.uniform().max(f64::MIN_POSITIVE);
        let second = self.uniform();
        (-2.0 * first.ln()).sqrt() * (TAU * second).cos()
    }
}

#[test]
fn repeated_m0_fits_respect_the_cramer_rao_bound() {
    let document = reference();
    let case = case_from(&document);
    let truth = vec![
        document["truth"]["eg_ev"].as_f64().expect("truth eg"),
        document["truth"]["ln_b"].as_f64().expect("truth ln b"),
    ];
    let predicted = case.predicted_voltage(&truth);
    let temperatures = temperature(&document);
    let sigma = case.sigma_v;
    let jacobian = case
        .observation_jacobian_reported(&truth)
        .expect("reported Jacobian");
    let report =
        analyze_fisher(&fisher_from_jacobian(&jacobian, sigma).expect("Fisher")).expect("analysis");

    let count = 400usize;
    let mut rng = Lcg::new(0x4352_414d);
    let mut sum_eg = 0.0;
    let mut sum_ln_b = 0.0;
    let mut sum_eg_squared = 0.0;
    let mut sum_ln_b_squared = 0.0;
    let mut sum_cross = 0.0;
    let ln_current = case.current_a.ln();
    for _ in 0..count {
        let observations = predicted
            .iter()
            .map(|value| value + sigma * rng.normal())
            .collect::<Vec<_>>();
        let n = temperatures.len() as f64;
        let mean_temperature = temperatures.iter().sum::<f64>() / n;
        let mean_voltage = observations.iter().sum::<f64>() / n;
        let mut covariance = 0.0;
        let mut variance = 0.0;
        for (temperature, observation) in temperatures.iter().zip(&observations) {
            covariance += (temperature - mean_temperature) * (observation - mean_voltage);
            variance += (temperature - mean_temperature).powi(2);
        }
        let slope = covariance / variance;
        let intercept = mean_voltage - slope * mean_temperature;
        // M0 is V = E_g - V_T ln B + V_T ln I, so the ordinate intercept is E_g.
        let eg = intercept;
        let ln_b = ln_current - slope * (ELEMENTARY_CHARGE / BOLTZMANN);
        sum_eg += eg;
        sum_ln_b += ln_b;
        sum_eg_squared += eg * eg;
        sum_ln_b_squared += ln_b * ln_b;
        sum_cross += eg * ln_b;
    }

    let divisor = count as f64;
    let mean_eg = sum_eg / divisor;
    let mean_ln_b = sum_ln_b / divisor;
    let variance_eg = (sum_eg_squared - divisor * mean_eg * mean_eg) / (divisor - 1.0);
    let variance_ln_b = (sum_ln_b_squared - divisor * mean_ln_b * mean_ln_b) / (divisor - 1.0);
    let covariance = (sum_cross - divisor * mean_eg * mean_ln_b) / (divisor - 1.0);
    let fisher_sd_eg = report.covariance.get(0, 0).expect("diagonal").sqrt();
    let fisher_sd_ln_b = report.covariance.get(1, 1).expect("diagonal").sqrt();
    let empirical_sd_eg = variance_eg.sqrt();
    let empirical_sd_ln_b = variance_ln_b.sqrt();
    let empirical_correlation = covariance / (empirical_sd_eg * empirical_sd_ln_b);

    assert!(
        empirical_sd_eg / fisher_sd_eg >= 0.9 && empirical_sd_eg / fisher_sd_eg <= 1.1,
        "empirical eg sd {empirical_sd_eg} vs Fisher {fisher_sd_eg}"
    );
    assert!(
        empirical_sd_ln_b / fisher_sd_ln_b >= 0.9 && empirical_sd_ln_b / fisher_sd_ln_b <= 1.1,
        "empirical ln B sd {empirical_sd_ln_b} vs Fisher {fisher_sd_ln_b}"
    );
    assert!(
        (empirical_correlation - 0.9969).abs() < 0.01,
        "empirical correlation {empirical_correlation} should be near the M0 ridge"
    );
}
