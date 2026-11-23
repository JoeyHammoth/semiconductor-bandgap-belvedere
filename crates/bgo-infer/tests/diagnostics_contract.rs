use bgo_infer::diagnostics::{bulk_ess, split_rhat, Diagnostics};

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((self.0 >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    }
}

#[test]
fn healthy_iid_chains_pass_the_diagnostic_gate() {
    let mut rng = Lcg::new(0x5eed);
    let chains: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..2_000).map(|_| rng.normal()).collect())
        .collect();

    let rhat = split_rhat(&chains);
    let ess = bulk_ess(&chains);
    let diagnostics = Diagnostics::new(rhat, ess, 0.002, 0);

    assert!(rhat < 1.01, "rhat={rhat}");
    assert!(ess > 400.0, "ess={ess}");
    assert!(
        diagnostics.passes,
        "healthy chains must pass: {diagnostics:?}"
    );
}

#[test]
fn a_trending_chain_fails_rhat_and_ess() {
    let mut chains: Vec<Vec<f64>> = (0..4)
        .map(|chain| {
            (0..2_000)
                .map(|draw| (draw as f64) / 200.0 + chain as f64)
                .collect()
        })
        .collect();
    chains[0].iter_mut().for_each(|value| *value += 10.0);

    let rhat = split_rhat(&chains);
    let ess = bulk_ess(&chains);
    let diagnostics = Diagnostics::new(rhat, ess, 0.002, 0);

    assert!(rhat > 1.01, "trending chains should fail rhat: {rhat}");
    assert!(!diagnostics.passes, "broken chains must not pass");
}

#[test]
fn energy_error_and_divergences_fail_closed() {
    let healthy = Diagnostics::new(1.001, 1_000.0, 0.002, 0);
    assert!(healthy.passes);

    let high_energy = Diagnostics::new(1.001, 1_000.0, 0.5, 0);
    assert!(!high_energy.passes);

    let divergent = Diagnostics::new(1.001, 1_000.0, 0.002, 1);
    assert!(!divergent.passes);
}
