# 0006 — No recorded dataset; audit the report summary instead

- Status: accepted
- Date: 2026-11-18

## Context

The five original PHYC30021 diode-thermometry runs no longer exist. The surviving material is the lab report, its Table 1, its figures and the handout. The Phase 0 plan assumed the runs could be recovered, hash-pinned and refitted; that assumption is no longer true.

A figure can be digitized and plausible values can be invented, but neither is the recording. Treating either as raw data would make every downstream "measurement" claim unverifiable while looking rigorous.

## Decision

Do not reconstruct, digitize or publish a substitute dataset. Keep `data/raw/` and `data/clean/` empty, retain the schema, validator and hash tooling for any future authentic runs, and run the site in labelled simulator mode.

Audit only what survives:

- the report's Table 1 values and their percentage arithmetic;
- wavelength conversions from the reported gaps, including the red 613 nm discrepancy;
- the M0 fitter as a tested numerical component, with no Table 1 reproduction claim.

Re-scope #10 and #15 around this limitation and close Phase 0 on documented evidence, not on an invented regression.

## Consequences

- No raw-data claim is made anywhere: no per-run `R_set`/current, no thermocouple calibration, no 1N4007 ideality, no per-point uncertainty.
- The Table 1 intercepts are quoted as report summary values and cannot be refitted by the site.
- Phase 1 and later use labelled simulated data and report-anchored examples. Simulated output must never be presented as the lost measurement.
- Any future authentic logs require provenance, a regenerated manifest and a passing `--require-complete` validation before they can be used.
- The site's provenance rule becomes "code, a cited source or a hash-pinned dataset, and a commit".
- Rejected alternatives: digitizing `fig_3a` (not raw data, no provenance), fabricating runs to satisfy the old tests, and quietly dropping the verification claim without recording why.
