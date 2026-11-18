//! Real branches of the Lambert W function.
//!
//! `W(z)` is defined by `W(z) * exp(W(z)) = z`. The principal branch `W_0`
//! covers `z >= -1/e`; the lower branch `W_-1` covers `-1/e <= z < 0`.
//! The implementation uses Halley iteration with branch-specific seeds and
//! stops at a relative fixed-point residual well below 1e-12.

/// Which real branch of `W` to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LambertWBranch {
    /// `W_0`, the branch through `W(0) = 0`.
    Principal,
    /// `W_-1`, defined on `[-1/e, 0)`.
    MinusOne,
}

/// Why a Lambert W evaluation was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LambertWError {
    /// The argument is NaN, infinite, or outside the branch's domain.
    OutOfDomain,
    /// The iteration failed to converge to the requested relative tolerance.
    DidNotConverge,
}

impl std::fmt::Display for LambertWError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfDomain => write!(f, "argument outside the Lambert W branch domain"),
            Self::DidNotConverge => write!(f, "Lambert W iteration did not converge"),
        }
    }
}

impl std::error::Error for LambertWError {}

const INV_E: f64 = 0.367_879_441_171_442_33;
const BRANCH_POINT: f64 = -INV_E;
const TOLERANCE: f64 = 1e-15;
const MAX_ITERATIONS: usize = 50;

/// Evaluate the requested real branch.
pub fn lambert_w(z: f64, branch: LambertWBranch) -> Result<f64, LambertWError> {
    match branch {
        LambertWBranch::Principal => lambert_w0(z),
        LambertWBranch::MinusOne => lambert_w_minus_one(z),
    }
}

/// Principal branch `W_0(z)`.
pub fn lambert_w0(z: f64) -> Result<f64, LambertWError> {
    if z.is_nan() || z < BRANCH_POINT {
        return Err(LambertWError::OutOfDomain);
    }
    if z == f64::INFINITY {
        return Ok(f64::INFINITY);
    }
    if z == 0.0 {
        return Ok(0.0);
    }
    if (z - BRANCH_POINT).abs() <= f64::EPSILON {
        return Ok(-1.0);
    }

    let mut w = initial_w0(z);
    halley(z, &mut w)
}

/// Lower branch `W_-1(z)`.
pub fn lambert_w_minus_one(z: f64) -> Result<f64, LambertWError> {
    if !z.is_finite() || z < BRANCH_POINT || z >= 0.0 {
        return Err(LambertWError::OutOfDomain);
    }
    if (z - BRANCH_POINT).abs() <= f64::EPSILON {
        return Ok(-1.0);
    }

    let mut w = initial_w_minus_one(z);
    halley(z, &mut w)
}

fn initial_w0(z: f64) -> f64 {
    if z <= 1.0 {
        if z > -0.3 {
            // The first terms of the series at zero.
            z * (1.0 - z + 1.5 * z * z)
        } else {
            // Near the branch point W_0 = -1 + sqrt(2(e z + 1)) + ...
            let delta = (2.0 * (std::f64::consts::E * z + 1.0)).sqrt();
            -1.0 + delta
        }
    } else if z < std::f64::consts::E {
        z.ln() - (z.ln()).ln()
    } else {
        let l = z.ln();
        l - l.ln()
    }
}

fn initial_w_minus_one(z: f64) -> f64 {
    if z > -0.25 {
        // W_-1 -> -infinity as z -> 0-, with W ~ log(-z) - log(-log(-z)).
        let l1 = (-z).ln();
        l1 - (-l1).ln()
    } else {
        // Near the branch point both branches meet at -1.
        let delta = (2.0 * (std::f64::consts::E * z + 1.0)).sqrt();
        -1.0 - delta
    }
}

fn halley(z: f64, w: &mut f64) -> Result<f64, LambertWError> {
    for _ in 0..MAX_ITERATIONS {
        let ew = w.exp();
        let residual = *w * ew - z;
        let wp1 = *w + 1.0;
        let denominator = ew * wp1 - (*w + 2.0) * residual / (2.0 * wp1);
        if !denominator.is_finite() || denominator == 0.0 {
            return Err(LambertWError::DidNotConverge);
        }
        let step = residual / denominator;
        *w -= step;
        if !w.is_finite() {
            return Err(LambertWError::DidNotConverge);
        }
        if step.abs() <= TOLERANCE * (1.0 + w.abs()) {
            return Ok(*w);
        }
    }
    Err(LambertWError::DidNotConverge)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_residual(z: f64, w: f64) {
        let residual = (w * w.exp() - z).abs();
        let scale = z.abs().max(f64::MIN_POSITIVE);
        assert!(
            residual / scale <= 1e-12,
            "z={z:e}, w={w:e}, relative residual={:e}",
            residual / scale
        );
    }

    #[test]
    fn principal_branch_known_values() {
        assert_eq!(lambert_w0(0.0).unwrap(), 0.0);
        assert!((lambert_w0(1.0).unwrap() - 0.567_143_290_409_783_8).abs() < 1e-14);
        assert!((lambert_w0(std::f64::consts::E).unwrap() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn both_branches_satisfy_the_defining_equation() {
        let principal = [
            -0.36, -0.3, -0.1, -1e-6, -1e-12, 0.0, 1e-12, 1e-6, 0.1, 0.5, 1.0, 2.0, 10.0, 1e3,
        ];
        for z in principal {
            assert_residual(z, lambert_w0(z).unwrap());
        }

        let lower = [-0.36, -0.3, -0.1, -1e-3, -1e-6, -1e-12, -1e-100];
        for z in lower {
            assert_residual(z, lambert_w_minus_one(z).unwrap());
        }
    }

    #[test]
    fn branch_point_is_shared() {
        let z = -1.0 / std::f64::consts::E;
        assert!((lambert_w0(z).unwrap() + 1.0).abs() < 1e-15);
        assert!((lambert_w_minus_one(z).unwrap() + 1.0).abs() < 1e-15);
    }

    #[test]
    fn out_of_domain_is_rejected() {
        assert_eq!(lambert_w0(-0.4), Err(LambertWError::OutOfDomain));
        assert_eq!(lambert_w0(f64::NAN), Err(LambertWError::OutOfDomain));
        assert_eq!(lambert_w_minus_one(0.0), Err(LambertWError::OutOfDomain));
        assert_eq!(lambert_w_minus_one(1.0), Err(LambertWError::OutOfDomain));
    }

    #[test]
    fn large_arguments_converge() {
        for z in [1e6, 1e12, 1e30, 1e100] {
            assert_residual(z, lambert_w0(z).unwrap());
        }
    }
}
