//! Samplers, evidence estimation and convergence diagnostics.
//!
//! Planned modules:
//! - `logdensity`: unconstrained parameter transforms and Jacobians
//! - `mclmc`: microcanonical Langevin/HMC with adaptation
//! - `nested`: Skilling nested sampling with slice sampling
//! - `svgd`: Stein variational gradient descent (CPU path)
//! - `laplace`: Gaussian approximation from the Hessian
//! - `diagnostics`: split R-hat, bulk ESS, energy error
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
