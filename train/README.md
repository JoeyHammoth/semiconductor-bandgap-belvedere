# Offline simulation and references

Nothing here ships as measured data. This directory generates the committed,
labelled synthetic cases and their NumPyro reference summaries for the Rust
sampler comparison in #26.

## Current contents

```text
train/
├─ pyproject.toml            # pinned through uv.lock
├─ uv.lock
├─ .python-version           # 3.13
├─ configs/                  # M0-M4 case contracts and seeds
├─ references/               # synthetic observations + NUTS summaries
│  ├─ manifest.json          # config/seed hashes; no data hash
│  └─ *.json
├─ src/bgo_train/
│  ├─ config.py              # validated TOML contract
│  ├─ simulation.py          # M0-M4 forward models and seeded noise
│  └─ reference.py           # NumPyro NUTS + CLI
└─ tests/
```

The future NPE, dynesty, export and training modules remain planned; they are
not part of the M0-M4 reference pipeline.

## Reproduce everything

```bash
cd train
uv sync --locked --extra dev
uv run python -m bgo_train.reference --all
```

`--all` is the single offline command that regenerates every golden artifact.
It writes observations generated from each config plus seed, NumPyro posterior
summaries, correlations and convergence diagnostics. The observations are
synthetic and are stored under `train/references/`, never under `data/`.

Validate the committed manifest without rerunning NUTS:

```bash
uv run python -m bgo_train.reference --check
```

Run the Python tests and lint:

```bash
uv run pytest
uv run ruff check src tests
```

## Case rules

- Every case has `synthetic = true` and an explicit anchor label.
- `m0-report-si` and `m0-report-ge` use the report's quoted Table 1 intercepts
  as **synthetic anchors**, not as recovered measurements. They do not reproduce
  Table 1 or imply the lost runs exist.
- M1-M4 use the audited Si Varshni parameters where applicable; `eta` remains
  illustrative, and M4's series resistance is a simulation fixture. They span
  an extended synthetic 220-383 K window so slope and curvature are
  identifiable; this is not a claim about the lost lab runs.
- M3/M4 hold `eta` at 3.0. A single constant-current sweep cannot separate
  `eta` from the terminal ideality factor, so those cases infer `n` and `R_s`
  instead; M1/M2 infer `eta`.
- The manifest records config and seed hashes only. There is no data hash and
  no measured-data claim.
- Goldens are written only when NUTS has zero divergences, split R-hat below
  1.01 and bulk ESS above 400 for every reported parameter.

## Notes on the reference

The reference uses NumPyro NUTS with four chains, 3,000 warmup and 2,000 retained
draws per chain by default, target acceptance 0.99 and a dense mass matrix.
`ln B` is parameterized at 300 K and transformed back to the physical `ln B`
for reporting; this avoids the `ln B`-`eta` ridge that made the raw
parameterization mix poorly. A bounded least-squares fit to the synthetic
observations supplies the initial point (it never uses the known truth); if
that fit hits a prior bound, warmup starts at the prior centre instead. The
committed summaries are what `bgo-infer` must match in #26.
