//! Forward-mode dual numbers.
//!
//! A dual number `(x, x')` carries a value and one directional derivative.
//! The arithmetic below implements the chain, product and quotient rules
//! exactly, without finite differences.

use crate::scalar::Scalar;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A value and its first derivative with respect to one seed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// Function value.
    pub re: f64,
    /// Directional derivative.
    pub du: f64,
}

impl Dual {
    pub const fn constant(value: f64) -> Self {
        Self { re: value, du: 0.0 }
    }

    pub const fn variable(value: f64) -> Self {
        Self { re: value, du: 1.0 }
    }

    pub fn value(self) -> f64 {
        self.re
    }

    pub fn derivative(self) -> f64 {
        self.du
    }
}

impl Add for Dual {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            du: self.du + other.du,
        }
    }
}

impl Sub for Dual {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            du: self.du - other.du,
        }
    }
}

impl Mul for Dual {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re,
            du: self.du * other.re + self.re * other.du,
        }
    }
}

impl Div for Dual {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let denominator = other.re * other.re;
        Self {
            re: self.re / other.re,
            du: (self.du * other.re - self.re * other.du) / denominator,
        }
    }
}

impl Neg for Dual {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            re: -self.re,
            du: -self.du,
        }
    }
}

impl Scalar for Dual {
    fn from_f64(value: f64) -> Self {
        Self::constant(value)
    }

    fn value(self) -> f64 {
        self.re
    }

    fn exp(self) -> Self {
        let value = self.re.exp();
        Self {
            re: value,
            du: value * self.du,
        }
    }

    fn ln(self) -> Self {
        Self {
            re: self.re.ln(),
            du: self.du / self.re,
        }
    }

    fn ln_1p(self) -> Self {
        let shifted = 1.0 + self.re;
        Self {
            re: shifted.ln(),
            du: self.du / shifted,
        }
    }

    fn sqrt(self) -> Self {
        let value = self.re.sqrt();
        Self {
            re: value,
            du: self.du / (2.0 * value),
        }
    }

    fn sin(self) -> Self {
        Self {
            re: self.re.sin(),
            du: self.re.cos() * self.du,
        }
    }

    fn cos(self) -> Self {
        Self {
            re: self.re.cos(),
            du: -self.re.sin() * self.du,
        }
    }

    fn tanh(self) -> Self {
        let value = self.re.tanh();
        Self {
            re: value,
            du: (1.0 - value * value) * self.du,
        }
    }

    fn erf(self) -> Self {
        let derivative = 2.0 / std::f64::consts::PI.sqrt() * (-self.re * self.re).exp();
        Self {
            re: libm::erf(self.re),
            du: derivative * self.du,
        }
    }

    fn powi(self, exponent: i32) -> Self {
        if exponent == 0 {
            return Self::constant(1.0);
        }
        if exponent == 1 {
            return self;
        }
        let value = self.re.powi(exponent);
        Self {
            re: value,
            du: exponent as f64 * self.re.powi(exponent - 1) * self.du,
        }
    }

    fn powf(self, exponent: Self) -> Self {
        let value = self.re.powf(exponent.re);
        Self {
            re: value,
            du: value * (exponent.du * self.re.ln() + exponent.re * self.du / self.re),
        }
    }

    fn abs(self) -> Self {
        Self {
            re: self.re.abs(),
            du: self.re.signum() * self.du,
        }
    }
}

/// Evaluate a function and its derivative at `x`.
pub fn derivative<F>(x: f64, function: F) -> f64
where
    F: Fn(Dual) -> Dual,
{
    function(Dual::variable(x)).du
}
