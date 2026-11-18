# Physics audit: what the lab numbers actually mean

Status: **open**. This note exists so that no recalled number reaches public copy before it is checked. Each item below is a `[VERIFY]` from the design document, with the action needed and the current verdict.

## 1. The central question

The report fits a straight line to `V` against `T` and reads the intercept as `E_g/q`. Under the ideal assumptions that is correct; with a `T^η` prefactor it is not. The audit must establish, with citations, what the intercept estimates and what the 0 K and room-temperature gaps actually are.

| Quantity | Report / handout | Suspected issue | Verdict |
|---|---|---|---|
| Si `E_g(0)` | 1.12 eV | may be the room-temperature gap; 0 K value often quoted ≈ 1.16–1.17 eV | **unresolved** |
| Si `α` | 7.02×10⁻⁴ eV/K | usually paired with `E_g(0) ≈ 1.16 eV` | **unresolved** |
| Si `β` | 1108 K | consistent with common Varshni sets | **unresolved** |
| Ge `E_g(0)` | 0.67 eV | may be the room-temperature gap; 0 K value often quoted ≈ 0.74 eV | **unresolved** |
| Ge Varshni set | not given in report | literature set often `0.7437 eV, 4.774×10⁻⁴ eV/K, 235 K` | **unresolved** |

Until resolved, public copy must say "the fitted intercept" and must not say "the band gap at 0 K".

## 2. Items to verify against primary sources

1. Varshni (1967) parameter sets for Si and Ge, and the temperature at which quoted gaps are defined.
2. Viña, Logothetidis and Cardona (1984) Bose–Einstein form and parameter definitions.
3. The `T^η` prefactor: which of mobility, diffusion length and effective-mass temperature dependence contribute, and whether `η ≈ 3` is defensible or merely illustrative.
4. EPM form factors for Si and Ge (Cohen and Bergstresser 1966): `V3`, `V8`, `V11` and lattice constants.
5. The report's own arithmetic: `1239.84 / 2.030 eV`, the two `R_set` values, and the stated uncertainty method.
6. Whether the 1N4007 is diffusion- or recombination-dominated at the currents used, which sets `n_a`.
7. Instrument specifications: ADS1115 gain/FSR, LM234 `γ`, multimeter and thermocouple datasheets.

## 3. What the app will show

Once resolved, the "What does the straight line actually estimate?" panel shows three numbers side by side per diode:

1. the fitted intercept from the lab's M0 model,
2. the 0 K gap from the cited parameter set,
3. the room-temperature gap at the measured mean temperature.

The difference between them is the point of the panel. If the audit finds the report's targets are not like-for-like, the app says so and cites the source; it does not quietly "correct" the report's table.

## 4. Parking rule

An item that cannot be resolved before Phase 1 ships is labelled **do not publish** and excluded from copy. It may still appear in an explicitly marked "open questions" note.
