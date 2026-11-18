# 0004 — Band structure on the CPU, WebGPU for visuals only

- Status: accepted
- Date: 2026-11-18

## Context

The empirical pseudopotential Hamiltonian for Si and Ge is real symmetric at about 113–137 plane waves per k-point. A few hundred k-points means a few hundred small eigenproblems — fast on a CPU, awkward on a GPU without specialised libraries. WebGPU availability also varies across devices.

## Decision

Compute band structure, density of states and (later) the Kernel Polynomial Method in a Web Worker in Rust/WASM. Reserve WebGPU for rendering and particle inference (SVGD kernel sums, hero particles).

## Consequences

- Band structure works on every device that runs WASM, independent of GPU support.
- The team owns tridiagonalisation/QL or Lanczos and a small SVD implementation rather than depending on a large linear-algebra crate.
- The CPU worker can be time-sliced and cancelled by terminating the worker; results are cached by content hash.
- WebGPU scenes must still degrade to WebGL2 and to static diagrams.
