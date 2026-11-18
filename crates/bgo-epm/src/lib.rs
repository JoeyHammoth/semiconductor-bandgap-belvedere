//! Empirical pseudopotential band structure, DOS and Kernel Polynomial Method.
//!
//! Planned modules:
//! - `lattice`: diamond-structure geometry and reciprocal vectors
//! - `hamiltonian`: real-symmetric EPM Hamiltonian
//! - `eigen`: Householder tridiagonalisation, implicit QL, Lanczos
//! - `dos`: k-point sampling, histogram and tetrahedron DOS
//! - `kpm`: Chebyshev moments with stochastic trace (stretch)
//!
//! Boundaries: this crate must stay free of I/O and of browser APIs so that it
//! can be unit-tested on the host and linked into `bgo-wasm`.

#![forbid(unsafe_code)]

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
