# Data and provenance

Status: **no recorded dataset**. The five original PHYC30021 diode-thermometry runs no longer exist. This page records what survives, what cannot be checked, and how the repository prevents a substitute dataset from being mistaken for the measurements.

## 1. What survives

| Source | Path | Use |
|---|---|---|
| PHYC30021 lab report | `source/Journal_Report/main.md` | narrative, the quoted Table 1 summary and instrument description |
| Report figures | `source/Journal_Report/images/` | provenance only; not shipped and not a data source |
| Lab handout | `source/Semiconductors.pdf` | method and parameter targets |
| Design document | `source/bandgap-observatory-design.md` | specification, not data |

The report is a secondary summary of the lost runs. A number copied from it is labelled **report summary**, never "the measured runs" and never a hash-pinned observation.

## 2. What is unavailable

- The raw `t`, ADC counts, derived voltage, current, thermocouple temperature and LM234 current for every run.
- The instrument configuration per run: ADC gain/FSR, `R_set`, `R_temp`, `R1`, supply and diode part number.
- The per-point uncertainties needed to reproduce the report's max–min trend-line spread.
- Any way to resolve which run used 220 Ω versus 2200 Ω.
- Any way to tell whether the red 613 nm entry is a measured emission wavelength, a typo, or a different rounding convention. The measured gap 2.030 eV gives 610.76 nm.
- Thermocouple calibration and the 1N4007 ideality at the recorded currents.

Digitizing `fig_3a` does not recover the runs: it produces a derived curve with no per-point provenance. Fabricating or interpolating rows would be worse. ADR [0006](adr/0006-no-recorded-dataset.md) records the decision not to do either.

## 3. Data slots are reserved, not populated

```text
data/
├─ raw/.gitkeep               # reserved; no run files
├─ clean/                     # reserved; no derived files
├─ schema.json                # contract for any future authentic run
└─ manifest.json              # intentionally empty
```

There are no raw or clean runs, and `data/manifest.json` has an empty `files` map. The validator and schema remain useful only as a gate for authentic data that might surface later.

## 4. Run contract for a future recovery

If original logs are ever recovered, each run must provide:

| Field | Meaning |
|---|---|
| `run_id` | stable identifier, also the filename stem |
| `material` | `ge`, `si`, `led-red`, `led-green`, `led-blue` |
| `device` | part number, e.g. `1N4007`, `1N60P` |
| `recorded_at` | ISO 8601 date |
| `instrument` | ADC gain/FSR, `R_set`, `R_temp`, `R1`, supply limits |
| `samples` | arrays of `t_s`, `adc_counts`, `voltage_v`, `current_a`, `temp_c`, `lm234_current_a` |
| `notes` | free text, including anything that went wrong |
| `provenance` | operator, granted consent, attribution and source file |

The authoritative constraints are in [`data/schema.json`](../data/schema.json). Internal units are SI; Celsius may be stored but must be declared and converted on load. `I = V_Rtemp / R_temp` and `λ[nm] = 1239.84 / E[eV]`.

## 5. Validation and hashing

`scripts/validate-data.mjs` checks every `data/raw/*.json` against the schema and the audit rules: filename and `run_id` must agree, all sample arrays must be the same length, each run needs a non-empty sibling `.md`, and consent must be `granted` with an attribution string. Its tests feed malformed fixtures through the same validator in CI.

`scripts/hash-data.mjs` would pin any committed file in `data/manifest.json`. Today the manifest is empty by design.

`--require-complete` demands all five materials. It is reserved for authentic recordings; it must not be used to make an empty or reconstructed dataset look complete.

## 6. Report summary and known limits

| Diode | Report intercept (eV) | Quoted spread (eV) |
|---|---|---|
| Ge | 0.646 | 0.0071 |
| Si | 1.002 | 0.0113 |
| Red LED | 2.030 | 0.0120 |
| Blue LED | 2.796 | 0.0126 |
| Green LED | 2.303 | 0.0112 |

These are quoted summary values, not observations this repository can refit.

- The quoted spread is a max–min trend-line spread, not a total uncertainty.
- The report compares Ge and Si with 0.67 and 1.12 eV, which are 300 K gaps, not 0 K gaps.
- The LEDs are compared with textbook bins, not the dies actually measured.
- The red wavelength is internally inconsistent: 613 nm is quoted, while `1239.84 / 2.030 = 610.76 nm`.
- No per-run current, temperature calibration or point-level uncertainty exists.

The physics consequences are in [`docs/physics-audit.md`](physics-audit.md).

## 7. Consent and attribution

Publication of the measured runs with attribution was approved, but the recordings no longer exist, so there is no dataset to publish. The report text and figures remain coursework material: they are cited and linked in `source/`, not copied into the deployed site. Any future use of the report's prose or figures requires the attribution wording to be confirmed and recorded first.

## 8. Privacy

No personal data is collected by the site: no accounts, no cookies, no analytics by default. If cookieless analytics are ever added, they are disclosed on the About page. The optional hardware mode is local-network and read-only; it is never part of the public deployment.
