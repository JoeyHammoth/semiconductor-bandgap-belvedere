# Contributing

This is a personal portfolio project, but the conventions are those of a small scientific codebase: claims must be traceable, budgets must hold, and nothing ships unverified. Contributions of all sizes are welcome.

## Ways to contribute

- Pick up a [`good first issue`](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues?q=is%3Aopen+label%3A%22good+first+issue%22). These are small, specified, and need no physics background.
- Take a [`help wanted`](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues?q=is%3Aopen+label%3A%22help+wanted%22) item. These are larger and may need a judgement call.
- Report a bug or request a feature with the issue templates.
- Improve docs, examples or the accessible equivalent of a scene.
- Answer [questions](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues?q=is%3Aopen+label%3Aquestion).

Issues stay unassigned until someone says they are taking one, so a comment first avoids duplicate work. If you claim an issue and cannot finish it, say so and it goes back in the pool.

## Before you start

- Read [`README.md`](README.md), [`docs/model.md`](docs/model.md) and [`docs/architecture.md`](docs/architecture.md).
- For anything beyond a typo, open or claim an issue first so the scope is agreed.
- Changes to physics, a tolerance or a public claim need the maintainer: [`docs/validation.md`](docs/validation.md) says what evidence each claim requires.

## Setup

One command, from the repository root:

```bash
node scripts/setup.mjs          # check the toolchains, add the wasm target, npm ci
node scripts/setup.mjs --check  # report only, change nothing
```

It never installs Node, Rust or wasm-pack; it prints the exact command when one is missing.

Or open the repository in a Codespace, or in a local container with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) pins Node 24, Rust and `wasm-pack` 0.15.1, and runs the setup script when the container is created. Python is optional there and is not installed.

The same steps by hand:

```bash
# Web
cd web && npm ci

# Rust (the wasm target is needed by bgo-wasm)
rustup target add wasm32-unknown-unknown
cargo test --workspace

# Build the WASM core the scenes load
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build crates/bgo-wasm --release --target web --out-dir ../../web/static/wasm

# Python references (optional, offline)
cd train && uv sync --locked --extra dev
```

Requirements: Node 24 ([`.nvmrc`](.nvmrc)), the Rust toolchain pinned in [`rust-toolchain.toml`](rust-toolchain.toml) (which already lists `wasm32-unknown-unknown`), `wasm-pack` 0.15.1, and optionally Python 3.13 with `uv`.

## Checks

| What | Command |
|---|---|
| Rust format | `cargo fmt --all -- --check` |
| Rust lint | `cargo clippy --workspace --all-targets -- -D warnings` |
| Rust tests | `cargo test --workspace` |
| Web formatting | `cd web && npm run format:check` (fix with `npm run format`) |
| Web types | `cd web && npm run check` |
| Web unit tests | `cd web && npm run test` |
| Browser and axe tests | `cd web && npm run e2e` |
| Size budgets | `cd web && npm run build && cd .. && node scripts/size-budget.mjs` |
| Contrast | `node scripts/contrast.mjs` |
| Contrast self-test | `node --test scripts/contrast.test.mjs` |

While iterating, one test is usually enough:

```bash
cargo test -p bgo-ssm greybox::                        # one module in one crate
cargo test -p bgo-physics --test report_arithmetic     # one integration test file
cd web && npx vitest run src/lib/infer/case.test.ts    # one unit test file
cd web && npx playwright test tests/thermal.spec.ts    # one browser spec
```

CI runs all of the above plus the accessibility, Lighthouse and reference checks. Playwright needs the production build and the WASM core, so run it after `npm run build`.

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
- Format Rust with `cargo fmt` and web sources with the repo Prettier config; do not reformat generated data or weights.
- Add a term to [`GLOSSARY.md`](GLOSSARY.md) before it appears in public copy.

## Data and figures

Do not commit new experimental data without recording consent and attribution in the manifest. The public site must run in simulator mode until that is resolved.

## What to expect

This is a solo project. There is no response-time guarantee: review happens in batches, and an issue may sit unlabelled for a while. A polite ping is welcome. An out-of-scope change is closed with the reason written down, not left silent.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Use [SUPPORT.md](SUPPORT.md) for usage questions and [SECURITY.md](SECURITY.md) for anything sensitive, never a public issue.
