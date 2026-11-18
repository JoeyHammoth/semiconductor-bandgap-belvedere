# Contributing

This is a personal portfolio project, but the conventions are those of a small scientific codebase: claims must be traceable, budgets must hold, and nothing ships unverified.

## Before you start

- Read [`README.md`](README.md), [`docs/model.md`](docs/model.md) and [`docs/architecture.md`](docs/architecture.md).
- Check the roadmap and the open Phase 0 tasks in the project board.
- For anything beyond a typo, open an issue first (or pick an existing one) so the scope is agreed.

## Setup

```bash
# Web
cd web && npm ci

# Rust (wasm target needed for bgo-wasm)
rustup target add wasm32-unknown-unknown
cargo test --workspace

# Python references (optional, offline)
cd train && uv sync
```

Node 24 and the Rust toolchain pinned in `rust-toolchain.toml` are expected. CI is the source of truth for Rust builds if your machine lacks the toolchain.

## Workflow

1. Branch from `main`: `phase-N/<topic>` for phase work, `<type>/<topic>` otherwise.
2. Keep commits scoped and imperative: `physics: add Varshni inverse`, `ci: enforce gzip budgets`.
3. Open a PR describing the change, its acceptance criteria, the evidence, and the blast radius.
4. Merge only with green CI, including budgets and accessibility checks.

## Engineering rules

- Physics goes in `crates/`, never in a scene component. Scenes ask the engine for numbers.
- No server, database, paid API or account system.
- No `SharedArrayBuffer` or COOP/COEP dependency.
- New browser dependencies need a row in [`docs/licences.md`](docs/licences.md) in the same PR.
- Datasets and weights are plain files under 10 MB. Never Git LFS.
- Update [`docs/validation.md`](docs/validation.md) when a claim becomes measurable. Mark it **verified** only with committed evidence.
- If you relax a tolerance, explain why and link the failing evidence.
- Add a term to [`GLOSSARY.md`](GLOSSARY.md) before it appears in public copy.

## Data and figures

Do not commit new experimental data without recording consent and attribution in the manifest. The public site must run in simulator mode until that is resolved.

## Reporting problems

Use the issue templates. For security concerns, see [`SECURITY.md`](SECURITY.md) instead of opening a public issue.
