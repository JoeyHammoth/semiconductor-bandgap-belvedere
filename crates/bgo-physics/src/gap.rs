//! Temperature-dependent band-gap models.
//!
//! Both models return `E_g(0)` at 0 K. The Bose–Einstein form is the standard
//! single-oscillator expression from O'Donnell and Chen (1991). The design
//! document's `E_B` is related by `E_B = E_g(0) + a_B`; it is not the 0 K gap.

/// Varshni parameters, with energies in eV and `beta` in kelvin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarshniGap {
    pub eg0_ev: f64,
    pub alpha_ev_per_k: f64,
    pub beta_k: f64,
}

impl VarshniGap {
    pub const fn new(eg0_ev: f64, alpha_ev_per_k: f64, beta_k: f64) -> Self {
        Self {
            eg0_ev,
            alpha_ev_per_k,
            beta_k,
        }
    }

    /// `E_g(T) = E_g(0) - alpha*T^2/(T+beta)`, in eV.
    pub fn at(&self, temperature_k: f64) -> f64 {
        varshni_gap(self.eg0_ev, self.alpha_ev_per_k, self.beta_k, temperature_k)
    }
}

/// Varshni gap `E_g(T) = E_g(0) - alpha*T^2/(T+beta)` in eV.
///
/// `eg0` is in eV, `alpha` in eV/K, `beta` in K, `temperature_k` in K.
pub fn varshni_gap(eg0: f64, alpha: f64, beta: f64, temperature_k: f64) -> f64 {
    eg0 - alpha * temperature_k * temperature_k / (temperature_k + beta)
}

/// Single-effective-phonon Bose–Einstein gap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoseEinsteinGap {
    /// The 0 K gap, in eV.
    pub eg0_ev: f64,
    /// `a_B = S<ħω>`, in eV.
    pub a_b_ev: f64,
    /// Effective phonon temperature `<ħω>/k_B`, in kelvin.
    pub theta_b_k: f64,
}

impl BoseEinsteinGap {
    pub const fn new(eg0_ev: f64, a_b_ev: f64, theta_b_k: f64) -> Self {
        Self {
            eg0_ev,
            a_b_ev,
            theta_b_k,
        }
    }

    /// Build from the design's `E_B` parameterisation.
    ///
    /// The design wrote `E_B - a_B[1 + 2/(e^{Theta/T}-1)]`, which is the same
    /// curve as the standard form only when `E_B = E_g(0) + a_B`.
    pub fn from_e_b(e_b_ev: f64, a_b_ev: f64, theta_b_k: f64) -> Self {
        Self::new(e_b_ev - a_b_ev, a_b_ev, theta_b_k)
    }

    /// The design's `E_B`, in eV. This is not the 0 K gap.
    pub fn e_b_ev(&self) -> f64 {
        self.eg0_ev + self.a_b_ev
    }

    /// `E_g(T) = E_g(0) - 2a_B/(exp(Theta/T)-1)`, in eV.
    pub fn at(&self, temperature_k: f64) -> f64 {
        bose_einstein_gap(self.eg0_ev, self.a_b_ev, self.theta_b_k, temperature_k)
    }
}

/// Bose–Einstein gap `E_g(T) = E_g(0) - 2a_B/(exp(Theta/T)-1)` in eV.
///
/// Returns `eg0` exactly at 0 K and NaN for negative temperatures.
pub fn bose_einstein_gap(eg0: f64, a_b: f64, theta_b: f64, temperature_k: f64) -> f64 {
    if temperature_k == 0.0 {
        return eg0;
    }
    if temperature_k < 0.0 {
        return f64::NAN;
    }
    eg0 - 2.0 * a_b / (theta_b / temperature_k).exp_m1()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SI_VARSHNI: VarshniGap = VarshniGap::new(1.1557, 7.021e-4, 1108.0);
    const GE_VARSHNI: VarshniGap = VarshniGap::new(0.7412, 4.56e-4, 210.0);

    // O'Donnell and Chen (1991), Si fit: E_g(0)=1.170 eV, S=1.49,
    // <hw>=25.5 meV, hence a_B=37.995 meV and Theta=25.5 meV/k_B.
    const SI_BOSE: BoseEinsteinGap = BoseEinsteinGap::new(1.170, 0.037_995, 295.916_16);

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn varshni_returns_the_zero_kelvin_gap_at_zero() {
        assert_eq!(SI_VARSHNI.at(0.0), 1.1557);
        assert_eq!(GE_VARSHNI.at(0.0), 0.7412);
    }

    #[test]
    fn varshni_matches_the_audit_values_at_300_kelvin() {
        close(SI_VARSHNI.at(300.0), 1.1108, 5e-5);
        close(GE_VARSHNI.at(300.0), 0.6607, 5e-5);
    }

    #[test]
    fn bose_einstein_returns_the_zero_kelvin_gap_at_zero() {
        assert_eq!(SI_BOSE.at(0.0), 1.170);
    }

    #[test]
    fn bose_einstein_matches_the_oc91_fit_at_300_kelvin() {
        // 1.170 - 2*0.037995/(exp(295.91616/300)-1)
        close(SI_BOSE.at(300.0), 1.124_773, 5e-5);
    }

    #[test]
    fn both_gaps_shrink_with_temperature() {
        let cold = SI_VARSHNI.at(200.0);
        let warm = SI_VARSHNI.at(350.0);
        assert!(warm < cold);
        let cold = SI_BOSE.at(200.0);
        let warm = SI_BOSE.at(350.0);
        assert!(warm < cold);
    }

    #[test]
    fn design_e_b_is_not_the_zero_kelvin_gap() {
        let gap = BoseEinsteinGap::from_e_b(1.208, 0.038, 300.0);
        close(gap.eg0_ev, 1.170, 1e-15);
        close(gap.e_b_ev(), 1.208, 1e-15);
        close(gap.at(0.0), 1.170, 1e-15);
    }

    #[test]
    fn negative_temperature_is_not_a_physical_gap() {
        assert!(bose_einstein_gap(1.17, 0.038, 300.0, -1.0).is_nan());
    }
}
