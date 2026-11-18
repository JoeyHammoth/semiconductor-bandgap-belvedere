# Offline training and references

Nothing here ships to the browser. This directory produces the small artifacts the site consumes: golden posteriors, quantised neural-posterior weights, and reference numbers for `docs/validation.md`.

## Planned contents

```text
train/
├─ pyproject.toml
├─ src/bgo_train/
│  ├─ simulator.py        # JAX forward model and noise
│  ├─ train_npe.py        # conditional spline flow training
│  ├─ reference.py        # NumPyro NUTS + dynesty + ArviZ references
│  ├─ sr.py               # PySR / KAN comparator (stretch)
│  └─ export.py           # quantised weights + JSON manifest + golden vectors
├─ configs/               # model and training configurations
└─ notebooks/             # Colab/Kaggle entry points, "Run all" reproducible
```

## Rules

- Training is offline and one-off; free GPU only (Colab/Kaggle) or local CPU.
- Seeds and configs are committed; artifacts are reproducible from them.
- Exported weights stay well under 10 MB per file and are committed, not stored in LFS.
- Reference results are compared against the Rust implementations in CI, with tolerances from `docs/validation.md`.
- Python dependencies are pinned with `uv.lock` once the first pipeline lands.
