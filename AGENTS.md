# Agent guidance

This repository is a static, browser-native physics instrument. Read `README.md`, `docs/model.md` and `docs/architecture.md` before changing behaviour.

## Commands

```bash
# Rust (requires rustup + wasm32-unknown-unknown target)
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build crates/bgo-wasm --release --target web --out-dir ../../web/static/wasm

# Web
cd web
npm ci
npm run check
npm run build
npm run test          # unit tests
npm run e2e           # Playwright, includes axe-core
```

## Layout

Physics and inference live in `crates/bgo-*`. UI and scenes live in `web/src/`. Offline training and reference generation live in `train/`. There is no recorded dataset; any future authentic data would be hash-pinned in `data/manifest.json` (see ADR 0006). Documents are in `docs/`; decisions in `docs/adr/`.

## Invariants

- No server, database, paid API or account system. Ever.
- No `SharedArrayBuffer` or dependency on COOP/COEP headers.
- Scene code never computes physics; it asks the engine for numbers.
- Every published number traces to code, a cited source or a hash-pinned dataset, and a commit.
- The lost recordings must not be reconstructed, digitized or presented as measurements.
- Datasets and weights are plain files under 10 MB; never Git LFS.
- A failed sampler diagnostic must not render a "converged" badge.
- New browser dependencies require a row in `docs/licences.md` in the same PR.

## Working agreements

- Branch from `main`; use `phase-N/<topic>` or `<type>/<topic>`.
- Conventional-ish commit subjects, imperative mood, scoped (`physics: add Varshni inverse`).
- PRs state acceptance criteria and link the issue they serve. Do not close a phase epic from a partial PR.
- Update `docs/validation.md` when a claim becomes measurable; mark it **verified** only with committed evidence.
- Rust is formatted with `cargo fmt`; web with the repo Prettier config.

## Agent skills

### Issue tracker

Issues and epics live in GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` plus `docs/adr/`. See `docs/agents/domain.md`.
