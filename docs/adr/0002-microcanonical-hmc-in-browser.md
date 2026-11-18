# 0002 — Microcanonical Langevin/HMC as the in-browser sampler

- Status: accepted
- Date: 2026-11-18

## Context

The posterior is low-dimensional, smooth and strongly correlated. NUTS is the reference standard but needs tree building, careful adaptation and a large, subtle implementation in Rust/WASM. Microcanonical Langevin/HMC (MCLMC) keeps dynamics on a constant-energy surface with unit-norm velocity, gives a built-in energy diagnostic and has simpler adaptation.

## Decision

Implement MCLMC in `bgo-infer` for the browser. Keep NumPyro NUTS as the offline ground truth in `train/` and compare posteriors in CI. Ship a dual-averaging HMC fallback behind a flag if MCLMC proves fragile.

## Consequences

- Fewer lines of sampler code to audit, and a diagnostic (energy error) that doubles as a UI element.
- The team owns the adaptation heuristics and must verify them against the papers and against NUTS.
- No gradient caching across leapfrog steps is assumed; `bgo-ad` must be fast enough.
- Results are gated on split R-hat, bulk ESS and energy error; a failed diagnostic hides the "converged" badge.
