//! The scalar operations model code may use without knowing the number type.
//!
//! A generic function written against [`Scalar`] can be evaluated on `f64` for
//! values, on [`crate::Dual`] for first derivatives, and on
//! [`crate::HyperDual`] for second and mixed derivatives.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A real scalar with the elementary operations used by the physics models.
pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + PartialEq
    + Sized
{
    fn from_f64(value: f64) -> Self;
    fn value(self) -> f64;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn ln_1p(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
    fn erf(self) -> Self;
    fn powi(self, exponent: i32) -> Self;
    fn powf(self, exponent: Self) -> Self;
    fn abs(self) -> Self;
}

impl Scalar for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }

    fn value(self) -> f64 {
        self
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn ln_1p(self) -> Self {
        f64::ln_1p(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn sin(self) -> Self {
        f64::sin(self)
    }

    fn cos(self) -> Self {
        f64::cos(self)
    }

    fn tanh(self) -> Self {
        f64::tanh(self)
    }

    fn erf(self) -> Self {
        libm::erf(self)
    }

    fn powi(self, exponent: i32) -> Self {
        f64::powi(self, exponent)
    }

    fn powf(self, exponent: Self) -> Self {
        f64::powf(self, exponent)
    }

    fn abs(self) -> Self {
        f64::abs(self)
    }
}
