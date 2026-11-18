# Data

Hash-pinned diode-thermometry runs. `raw/` holds what was recorded, `clean/` holds validated SI-unit data derived from it, `schema.json` is the contract and `manifest.json` is the integrity record.

Nothing here yet: the five runs (Ge, Si, red, green, blue) are recovered and validated under the Phase 0 data audit. The site must stay in simulator mode until consent and attribution are recorded.

Rules:

- Never edit `source/`; it is immutable input material.
- Every change under `raw/` or `clean/` requires a regenerated manifest in the same commit.
- Temperatures may be stored in Celsius but must say so; the model converts on load.
- No file may exceed 10 MB; no Git LFS.

See [`docs/data.md`](../docs/data.md) for the full schema, provenance and known issues.
