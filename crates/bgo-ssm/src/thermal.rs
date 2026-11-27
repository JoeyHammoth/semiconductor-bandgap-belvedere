//! First-order plate → junction → sensor thermal twin.
//!
//! This is the first slice of the M5 latent-temperature model. It is a
//! deterministic synthetic chain, not a UKF/RTS or GP implementation. The
//! sensor lag is recovered from the synthetic junction/sensor series by the
//! exact first-order relation `dT_sensor/dt = (T_junction - T_sensor) / tau`.
//!
//! Every input is synthetic: the original lab recordings do not exist
//! (ADR 0006). The model exists so the browser scene can show the difference
//! between the plate, the junction and the lagging sensor without inventing a
//! measured trace.

use std::f64::consts::PI;

/// Heater-on intervals in the identification schedule.
pub const HEAT_PULSES: usize = 3;

/// One heater-on interval, in seconds from the start of the run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatPulse {
    pub start_s: f64,
    pub stop_s: f64,
}

impl HeatPulse {
    const fn new(start_s: f64, stop_s: f64) -> Self {
        Self { start_s, stop_s }
    }

    /// The stop edge is exclusive, so adjacent pulses do not double up.
    fn is_on(&self, time_s: f64) -> bool {
        time_s >= self.start_s && time_s < self.stop_s
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalConfig {
    pub ambient_k: f64,
    pub initial_k: f64,
    pub heater_k: f64,
    /// Heater-on intervals, in seconds. Three pulses of increasing width give
    /// the fast, middle and slow chain modes their own step edges, which is
    /// what conditions the Hankel-DMDc regression.
    pub heat_pulses: [HeatPulse; HEAT_PULSES],
    /// Total simulated time. The final sample is at or just before this value.
    pub duration_s: f64,
    pub sample_s: f64,
    pub tau_plate_s: f64,
    pub tau_junction_s: f64,
    pub tau_sensor_s: f64,
    pub sensor_noise_k: f64,
    pub seed: u64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            ambient_k: 293.15,
            initial_k: 293.15,
            heater_k: 25.0,
            heat_pulses: [
                HeatPulse::new(60.0, 150.0),
                HeatPulse::new(300.0, 600.0),
                HeatPulse::new(780.0, 1_500.0),
            ],
            duration_s: 1_800.0,
            sample_s: 10.0,
            tau_plate_s: 420.0,
            tau_junction_s: 75.0,
            tau_sensor_s: 25.0,
            sensor_noise_k: 0.04,
            seed: 0x5eed_7e11,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalRun {
    pub time_s: Vec<f64>,
    pub plate_k: Vec<f64>,
    pub junction_k: Vec<f64>,
    pub sensor_k: Vec<f64>,
    pub observed_k: Vec<f64>,
    /// Latent-series estimate of the sensor time constant.
    pub lag_tau_s: f64,
    pub true_tau_s: f64,
    pub max_lag_k: f64,
    pub ambient_k: f64,
    pub heater_k: f64,
    /// Heater-on intervals used by the run, in seconds.
    pub heat_pulses: [HeatPulse; HEAT_PULSES],
}

/// Heater target at `time_s`, in kelvin above ambient.
pub fn heater_target_k(config: &ThermalConfig, time_s: f64) -> f64 {
    if config.heat_pulses.iter().any(|pulse| pulse.is_on(time_s)) {
        config.heater_k
    } else {
        0.0
    }
}

pub fn simulate_thermal(config: &ThermalConfig) -> Result<ThermalRun, String> {
    validate(config)?;
    let steps = ((config.duration_s / config.sample_s).round() as usize)
        .saturating_add(1)
        .max(2);
    let dt = config.sample_s;
    let plate_alpha = 1.0 - (-dt / config.tau_plate_s).exp();
    let junction_alpha = 1.0 - (-dt / config.tau_junction_s).exp();
    let sensor_alpha = 1.0 - (-dt / config.tau_sensor_s).exp();

    let mut time_s = Vec::with_capacity(steps);
    let mut plate_k = Vec::with_capacity(steps);
    let mut junction_k = Vec::with_capacity(steps);
    let mut sensor_k = Vec::with_capacity(steps);
    let mut observed_k = Vec::with_capacity(steps);
    let mut rng = Rng::new(config.seed);

    let mut plate = config.initial_k;
    let mut junction = config.initial_k;
    let mut sensor = config.initial_k;
    for index in 0..steps {
        let time = index as f64 * dt;
        if index > 0 {
            let target = config.ambient_k + heater_target_k(config, time);
            plate += (target - plate) * plate_alpha;
            junction += (plate - junction) * junction_alpha;
            sensor += (junction - sensor) * sensor_alpha;
        }
        let observed = sensor + config.sensor_noise_k * rng.normal();
        time_s.push(time);
        plate_k.push(plate);
        junction_k.push(junction);
        sensor_k.push(sensor);
        observed_k.push(observed);
    }

    let lag_tau_s = estimate_lag(&junction_k, &sensor_k, dt);
    let max_lag_k = junction_k
        .iter()
        .zip(&sensor_k)
        .map(|(junction, sensor)| junction - sensor)
        .fold(0.0_f64, f64::max);

    Ok(ThermalRun {
        time_s,
        plate_k,
        junction_k,
        sensor_k,
        observed_k,
        lag_tau_s,
        true_tau_s: config.tau_sensor_s,
        max_lag_k,
        ambient_k: config.ambient_k,
        heater_k: config.heater_k,
        heat_pulses: config.heat_pulses,
    })
}

fn validate(config: &ThermalConfig) -> Result<(), String> {
    let values = [
        ("ambient", config.ambient_k),
        ("initial", config.initial_k),
        ("heater", config.heater_k),
        ("duration", config.duration_s),
        ("sample", config.sample_s),
        ("plate tau", config.tau_plate_s),
        ("junction tau", config.tau_junction_s),
        ("sensor tau", config.tau_sensor_s),
        ("sensor noise", config.sensor_noise_k),
    ];
    if let Some((name, _)) = values.iter().find(|(_, value)| !value.is_finite()) {
        return Err(format!("{name} must be finite"));
    }
    if config.ambient_k <= 0.0
        || config.initial_k <= 0.0
        || config.sample_s <= 0.0
        || config.duration_s < config.sample_s
        || config.tau_plate_s <= 0.0
        || config.tau_junction_s <= 0.0
        || config.tau_sensor_s <= 0.0
        || config.sensor_noise_k < 0.0
    {
        return Err("thermal config is outside its physical domain".into());
    }
    for (index, pulse) in config.heat_pulses.iter().enumerate() {
        if !pulse.start_s.is_finite() || !pulse.stop_s.is_finite() {
            return Err(format!("heat pulse {index} must be finite"));
        }
        if pulse.start_s < 0.0 || pulse.stop_s < pulse.start_s {
            return Err(format!("heat pulse {index} is outside its physical domain"));
        }
        if index > 0 && pulse.start_s < config.heat_pulses[index - 1].stop_s {
            return Err("heat pulses must be ordered and non-overlapping".into());
        }
    }
    Ok(())
}

/// Exact lag estimate from the latent series. The discrete update is
/// `ΔT_sensor = alpha * (T_junction_new - T_sensor_old)`, with
/// `alpha = 1 - exp(-dt / tau)`. Least squares gives `alpha` first, then the
/// time constant; using the midpoint forcing here would bias the estimate.
fn estimate_lag(junction: &[f64], sensor: &[f64], dt: f64) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for index in 1..junction.len().min(sensor.len()) {
        let forcing = junction[index] - sensor[index - 1];
        let change = sensor[index] - sensor[index - 1];
        numerator += forcing * change;
        denominator += forcing * forcing;
    }
    if denominator.abs() < 1.0e-12 {
        return f64::NAN;
    }
    let alpha = (numerator / denominator).clamp(1.0e-12, 1.0 - 1.0e-12);
    -dt / (1.0 - alpha).ln()
}

struct Rng {
    state: u64,
}

impl Rng {
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
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let first = self.uniform().max(f64::MIN_POSITIVE);
        let second = self.uniform();
        (-2.0 * first.ln()).sqrt() * (2.0 * PI * second).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_is_finite_and_causal() {
        let config = ThermalConfig::default();
        let run = simulate_thermal(&config).expect("thermal trace");
        assert_eq!(run.time_s.len(), 181);
        assert_eq!(run.sensor_k.len(), run.time_s.len());
        assert!(run
            .time_s
            .iter()
            .chain(&run.plate_k)
            .chain(&run.junction_k)
            .chain(&run.sensor_k)
            .chain(&run.observed_k)
            .all(|value| value.is_finite()));
        assert!(run.max_lag_k > 0.0);
        assert!(run
            .junction_k
            .iter()
            .all(|value| *value >= config.ambient_k - 1.0e-9));
        assert!(run
            .sensor_k
            .iter()
            .all(|value| *value >= config.ambient_k - 1.0e-9));
    }

    #[test]
    fn recovers_the_sensor_time_constant() {
        let config = ThermalConfig::default();
        let run = simulate_thermal(&config).expect("thermal trace");
        let relative = (run.lag_tau_s - config.tau_sensor_s).abs() / config.tau_sensor_s;
        assert!(
            relative < 0.02,
            "estimated {} s vs true {} s",
            run.lag_tau_s,
            config.tau_sensor_s
        );
    }

    #[test]
    fn rejects_an_invalid_time_constant() {
        let config = ThermalConfig {
            tau_sensor_s: 0.0,
            ..ThermalConfig::default()
        };
        assert!(simulate_thermal(&config).is_err());
    }
}
