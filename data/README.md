# Data

**There is no recorded dataset.** The five original diode-thermometry runs no longer exist.

`raw/` and `clean/` are reserved for authentic runs that might surface later. `schema.json` is the contract those runs would have to satisfy, and `manifest.json` is intentionally empty. Do not add digitized figures, interpolated rows or reconstructed data here; they would look like measurements without being measurements.

The validator and hash tooling remain on `main` and are tested in CI, so a future recovery has a gate to pass rather than a blank cheque.

Rules:

- Never edit `source/`; it is immutable input material.
- Every change under `raw/` or `clean/` requires a regenerated manifest in the same commit.
- Temperatures may be stored in Celsius but must say so; the model converts on load.
- No file may exceed 10 MB; no Git LFS.
- `--require-complete` is only for authentic recordings.

See [`docs/data.md`](../docs/data.md) and [`docs/adr/0006-no-recorded-dataset.md`](../docs/adr/0006-no-recorded-dataset.md).
