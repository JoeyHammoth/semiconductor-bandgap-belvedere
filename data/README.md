# Data

Hash-pinned diode-thermometry runs. `raw/` holds what was recorded, `clean/` holds validated SI-unit data derived from it, `schema.json` is the contract and `manifest.json` is the integrity record.

No run files are committed yet: the five recordings (Ge, Si, red, green, blue) are still being recovered. Consent for publication with attribution has been granted, but the site stays in simulator mode until the runs are validated and hash-pinned under the Phase 0 data audit.

Rules:

- Never edit `source/`; it is immutable input material.
- Every change under `raw/` or `clean/` requires a regenerated manifest in the same commit.
- Temperatures may be stored in Celsius but must say so; the model converts on load.
- No file may exceed 10 MB; no Git LFS.

See [`docs/data.md`](../docs/data.md) for the full schema, provenance and known issues.
