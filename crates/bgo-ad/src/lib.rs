//! Dual and hyper-dual numbers for exact first and second derivatives.
//!
//! The crate is pure arithmetic: no I/O, no browser APIs and no dependencies.
//! Model code can be written once against [`Scalar`] and evaluated at `f64`,
//! [`Dual`] or [`HyperDual`] precision.

#![forbid(unsafe_code)]

pub mod dual;
pub mod hyperdual;
pub mod scalar;

pub use dual::{derivative, Dual};
pub use hyperdual::{mixed_partial, second_derivative, HyperDual};
pub use scalar::Scalar;

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, relative: f64, absolute: f64) {
        let error = (actual - expected).abs();
        assert!(
            error <= relative * (1.0 + expected.abs()) + absolute,
            "expected {expected:.16e}, got {actual:.16e}, error {error:.3e}"
        );
    }

    /// A composed function that exercises chain, product, quotient and the
    /// elementary operations.
    fn model<T: Scalar>(x: T) -> T {
        let one = T::from_f64(1.0);
        let two = T::from_f64(2.0);
        (x.exp() * x.sin() + (one + x * x).ln() + x.powi(3) / (two + x)).tanh()
    }

    fn first_difference(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)
    }

    fn second_difference(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (-f(x + 2.0 * h) + 16.0 * f(x + h) - 30.0 * f(x) + 16.0 * f(x - h) - f(x - 2.0 * h))
            / (12.0 * h * h)
    }

    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / (1u64 << 53) as f64
        }

        fn uniform(&mut self, low: f64, high: f64) -> f64 {
            low + (high - low) * self.next_f64()
        }
    }

    #[test]
    fn first_derivatives_match_five_point_differences() {
        let h = 1e-3;
        let mut rng = Lcg::new(0x5eed_1234);
        for _ in 0..64 {
            let x = rng.uniform(-1.8, 1.8);
            let exact = derivative(x, model);
            let numeric = first_difference(model, x, h);
            close(exact, numeric, 1e-8, 1e-10);
        }
    }

    #[test]
    fn second_derivatives_match_five_point_differences() {
        let h = 1e-3;
        let mut rng = Lcg::new(0xabcd_9876);
        for _ in 0..64 {
            let x = rng.uniform(-1.8, 1.8);
            let exact = second_derivative(x, model);
            let numeric = second_difference(model, x, h);
            close(exact, numeric, 1e-6, 1e-8);
        }
    }

    #[test]
    fn product_quotient_and_chain_rules_are_independent() {
        let x = 1.234_5;
        let product = |t: f64| t.exp() * t.sin();
        let product_d = |t: Dual| t.exp() * t.sin();
        close(
            derivative(x, product_d),
            first_difference(product, x, 1e-4),
            1e-9,
            1e-10,
        );

        let quotient = |t: f64| (t * t + 1.0) / (t.sin() + 2.0);
        let quotient_d = |t: Dual| (t * t + Dual::constant(1.0)) / (t.sin() + Dual::constant(2.0));
        close(
            derivative(x, quotient_d),
            first_difference(quotient, x, 1e-4),
            1e-9,
            1e-10,
        );

        let chain = |t: f64| (t.powi(3) + 1.0).ln().tanh();
        let chain_d = |t: Dual| (t.powi(3) + Dual::constant(1.0)).ln().tanh();
        close(
            derivative(x, chain_d),
            first_difference(chain, x, 1e-4),
            1e-9,
            1e-10,
        );
    }

    #[test]
    fn mixed_partials_match_cross_differences() {
        let f = |x: f64, y: f64| x * x * y + (x * y).sin() + x.exp() / (1.0 + y * y);
        let h = 1e-3;
        let mut rng = Lcg::new(0x1357_9bdf);
        for _ in 0..32 {
            let x = rng.uniform(-1.5, 1.5);
            let y = rng.uniform(-1.5, 1.5);
            let exact = mixed_partial(x, y, |a, b| {
                a * a * b + (a * b).sin() + a.exp() / (HyperDual::from_f64(1.0) + b * b)
            });
            let numeric = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h))
                / (4.0 * h * h);
            close(exact, numeric, 1e-5, 1e-6);
        }
    }

    #[test]
    fn near_zero_and_large_inputs_remain_accurate() {
        let tiny = derivative(1e-12, |x| x.ln());
        close(tiny, 1e12, 1e-12, 0.0);

        let large = derivative(20.0, |x| x.exp());
        close(large, 20.0_f64.exp(), 1e-12, 0.0);

        let second = second_derivative(10.0, |x| x.exp());
        close(second, 10.0_f64.exp(), 1e-12, 0.0);

        let reciprocal_second = second_derivative(2.0, |x| x.powi(-2));
        close(reciprocal_second, 0.375, 1e-12, 0.0);
    }

    #[test]
    fn powf_uses_the_general_chain_rule() {
        let value = derivative(2.0, |x| x.powf(Dual::constant(2.5)));
        close(value, 2.5 * 2.0_f64.powf(1.5), 1e-12, 0.0);
    }

    #[test]
    fn constants_have_zero_derivatives() {
        assert_eq!(derivative(3.0, |_| Dual::constant(7.0)), 0.0);
        assert_eq!(second_derivative(3.0, |_| HyperDual::constant(7.0)), 0.0);
    }
}
