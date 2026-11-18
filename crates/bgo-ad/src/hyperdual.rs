//! Hyper-dual numbers for exact second and mixed partial derivatives.
//!
//! A hyper-dual has a value, two directional seeds and their mixed coefficient:
//! `(f, D1 f, D2 f, D1 D2 f)`. Seeding both directions with 1 gives `(f, f',
//! f', f'')`; seeding one variable in each direction gives the mixed partial.

use crate::scalar::Scalar;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A value, two first directional derivatives and the mixed second derivative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperDual {
    pub re: f64,
    pub d1: f64,
    pub d2: f64,
    pub d12: f64,
}

impl HyperDual {
    pub const fn constant(value: f64) -> Self {
        Self {
            re: value,
            d1: 0.0,
            d2: 0.0,
            d12: 0.0,
        }
    }

    /// Seed both directions, so `d12` becomes the ordinary second derivative.
    pub const fn variable(value: f64) -> Self {
        Self {
            re: value,
            d1: 1.0,
            d2: 1.0,
            d12: 0.0,
        }
    }

    /// Seed only the first direction.
    pub const fn seed_first(value: f64) -> Self {
        Self {
            re: value,
            d1: 1.0,
            d2: 0.0,
            d12: 0.0,
        }
    }

    /// Seed only the second direction.
    pub const fn seed_second(value: f64) -> Self {
        Self {
            re: value,
            d1: 0.0,
            d2: 1.0,
            d12: 0.0,
        }
    }

    pub fn value(self) -> f64 {
        self.re
    }

    pub fn first_derivative(self) -> f64 {
        self.d1
    }

    pub fn second_derivative(self) -> f64 {
        self.d12
    }

    fn unary(
        self,
        f: impl Fn(f64) -> f64,
        df: impl Fn(f64) -> f64,
        ddf: impl Fn(f64) -> f64,
    ) -> Self {
        let first = df(self.re);
        Self {
            re: f(self.re),
            d1: first * self.d1,
            d2: first * self.d2,
            d12: ddf(self.re) * self.d1 * self.d2 + first * self.d12,
        }
    }
}

impl Add for HyperDual {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            d1: self.d1 + other.d1,
            d2: self.d2 + other.d2,
            d12: self.d12 + other.d12,
        }
    }
}

impl Sub for HyperDual {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            d1: self.d1 - other.d1,
            d2: self.d2 - other.d2,
            d12: self.d12 - other.d12,
        }
    }
}

impl Mul for HyperDual {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re,
            d1: self.d1 * other.re + self.re * other.d1,
            d2: self.d2 * other.re + self.re * other.d2,
            d12: self.d12 * other.re
                + self.d1 * other.d2
                + self.d2 * other.d1
                + self.re * other.d12,
        }
    }
}

impl Div for HyperDual {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let inverse = other.recip();
        Self {
            re: self.re * inverse.re,
            d1: self.d1 * inverse.re + self.re * inverse.d1,
            d2: self.d2 * inverse.re + self.re * inverse.d2,
            d12: self.d12 * inverse.re
                + self.d1 * inverse.d2
                + self.d2 * inverse.d1
                + self.re * inverse.d12,
        }
    }
}

impl Neg for HyperDual {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            re: -self.re,
            d1: -self.d1,
            d2: -self.d2,
            d12: -self.d12,
        }
    }
}

impl HyperDual {
    fn recip(self) -> Self {
        self.unary(|x| 1.0 / x, |x| -1.0 / (x * x), |x| 2.0 / (x * x * x))
    }
}

impl Scalar for HyperDual {
    fn from_f64(value: f64) -> Self {
        Self::constant(value)
    }

    fn value(self) -> f64 {
        self.re
    }

    fn exp(self) -> Self {
        self.unary(f64::exp, f64::exp, f64::exp)
    }

    fn ln(self) -> Self {
        self.unary(f64::ln, |x| 1.0 / x, |x| -1.0 / (x * x))
    }

    fn ln_1p(self) -> Self {
        self.unary(
            f64::ln_1p,
            |x| 1.0 / (1.0 + x),
            |x| -1.0 / ((1.0 + x) * (1.0 + x)),
        )
    }

    fn sqrt(self) -> Self {
        self.unary(
            f64::sqrt,
            |x| 1.0 / (2.0 * x.sqrt()),
            |x| -1.0 / (4.0 * x.powf(1.5)),
        )
    }

    fn sin(self) -> Self {
        self.unary(f64::sin, f64::cos, |x| -x.sin())
    }

    fn cos(self) -> Self {
        self.unary(f64::cos, |x| -x.sin(), |x| -x.cos())
    }

    fn tanh(self) -> Self {
        self.unary(
            f64::tanh,
            |x| 1.0 - x.tanh().powi(2),
            |x| -2.0 * x.tanh() * (1.0 - x.tanh().powi(2)),
        )
    }

    fn powi(self, exponent: i32) -> Self {
        if exponent == 0 {
            return Self::constant(1.0);
        }
        if exponent == 1 {
            return self;
        }
        self.unary(
            |x| x.powi(exponent),
            |x| exponent as f64 * x.powi(exponent - 1),
            |x| exponent as f64 * (exponent as f64 - 1.0) * x.powi(exponent - 2),
        )
    }

    fn powf(self, exponent: Self) -> Self {
        (exponent * self.ln()).exp()
    }

    fn abs(self) -> Self {
        self.unary(f64::abs, f64::signum, |_| 0.0)
    }
}

/// Evaluate a function and its second derivative at `x`.
pub fn second_derivative<F>(x: f64, function: F) -> f64
where
    F: Fn(HyperDual) -> HyperDual,
{
    function(HyperDual::variable(x)).d12
}

/// Evaluate the mixed partial `d^2 f / dx dy` at `(x, y)`.
pub fn mixed_partial<F>(x: f64, y: f64, function: F) -> f64
where
    F: Fn(HyperDual, HyperDual) -> HyperDual,
{
    function(HyperDual::seed_first(x), HyperDual::seed_second(y)).d12
}
