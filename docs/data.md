# Data and provenance

The project's credibility rests on being able to say exactly which numbers produced which claim. Every dataset is a plain JSON file in git, hash-pinned in a manifest, and validated against a schema in CI.

## 1. Sources

| Source | Path | Notes |
|---|---|---|
| PHYC30021 lab report | `source/Journal_Report/main.md` | narrative, Table 1, instrument description |
| Report figures | `source/Journal_Report/images/` | figure provenance only; not shipped to the site |
| Lab handout | `source/Semiconductors.pdf` | method and parameter targets |
| Design document | `source/bandgap-observatory-design.md` | specification, not data |

`source/` is treated as immutable input material. Cleaned, machine-readable runs live in `data/`.

## 2. Layout

```text
data/
├─ raw/<run-id>.json          # as recorded, plus instrument configuration
├─ raw/<run-id>.md            # human notes: what happened, known issues
├─ clean/<run-id>.json        # validated, SI units, derived quantities
├─ schema.json                # JSON Schema for raw and clean runs
└─ manifest.json              # SHA-256 of every committed data file
```

Run ids follow `<material>-<device>-<yyyymmdd>-<seq>`, for example `ge-1n60p-20240412-01`.

## 3. Run record

Required fields (see `data/schema.json` for the authoritative list):

| Field | Meaning |
|---|---|
| `run_id` | stable identifier, also the filename stem |
| `material` | `ge`, `si`, `led-red`, `led-green`, `led-blue` |
| `device` | part number, e.g. `1N4007`, `1N60P` |
| `recorded_at` | ISO 8601 date |
| `instrument` | ADC gain/FSR, `R_set`, `R_temp`, `R1`, supply limits |
| `samples` | arrays of `t_s`, `adc_counts`, `voltage_v`, `current_a`, `temp_c`, `lm234_current_a` |
| `notes` | free text, including anything that went wrong |
| `provenance` | operator, consent/attribution status, source file hash |

## 4. Units and conversions

- Internal model units are SI. Temperature is kelvin inside the model; run files may store Celsius but must say so and are converted on load.
- ADC counts are converted with the recorded FSR and gain, never a hard-coded constant.
- `I = V_Rtemp / R_temp`; resistor tolerance is recorded, not inferred.
- Wavelength uses `λ[nm] = 1239.84 / E[eV]`.

## 5. Hashing and the manifest

`scripts/hash-data.mjs` (Phase 0) rewrites `data/manifest.json` with the SHA-256 of every file under `data/raw` and `data/clean`. CI runs the same script in check mode and fails if the manifest is stale. The site footer's "Receipts" line shows the data hash alongside the commit hash.

## 6. Known issues in the recorded data

- Thermocouple sensitivity dominates the reported uncertainty; temperature is the weakest variable.
- The Peltier was paused during logging, so each point is quasi-static but not a continuous ramp.
- A ceiling of 383.15 K was applied so the simplified Shockley equation stays usable; this truncates the temperature range and worsens the `E_g`/`ln B` degeneracy.
- The report lists two `R_set` values (`220 Ω` and `2200 Ω`); which run used which must be recovered from the raw files.
- The report's red wavelength (613 nm) does not match `1239.84 / 2.030 eV ≈ 610.7 nm`; recompute, do not copy.

## 7. Consent and attribution

The dataset and figures originate in coursework. Publication requires the operator's consent and correct attribution. Until that is recorded in `data/manifest.json` (`provenance.consent`), the site must run on simulator mode only. This is tracked as a Phase 0 task.

## 8. Privacy

No personal data is collected by the site: no accounts, no cookies, no analytics by default. If cookieless analytics are ever added, they are disclosed on the About page. The optional hardware mode is local-network and read-only; it is never part of the public deployment.
