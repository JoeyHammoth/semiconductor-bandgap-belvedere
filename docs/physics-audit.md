# Physics audit: what the lab numbers actually mean

Status: **complete for the literature and arithmetic items; raw-data items are unavailable, not pending**. The five original runs no longer exist. Every claim below carries a source, and every computed number is reproduced by `python scripts/physics-audit.py`.

This note exists because the design document flags several recalled values `[VERIFY]`. Publishing any of them without a source would undermine the project's honesty claim, so the rule is: no number reaches public copy until it appears here with a verdict.

## 0. Verdict in one paragraph

The handout's "`E_g(0)`" column is mislabelled: 1.12 eV (Si) and 0.67 eV (Ge) are the **300 K** gaps, while the α and β printed beside them are Varshni's coefficients for the **0 K** fit. Adding a temperature-independent prefactor does not make a straight-line intercept equal to either gap: over the lab's window the chord extrapolation of `E_g(T)` alone sits **43 mV above** `E_g(0)` for Si and **37 mV above** for Ge, and a `T^η` prefactor adds about **29 mV per unit η**. The report's intercepts sit **132 mV (Ge) and 197 mV (Si) below** those chord values, which would require a *negative* η of −4.5 and −6.7. The prefactor cannot explain that. The quoted 7–13 mV uncertainties are trend-line spreads, not total uncertainties, and the report's agreement with 300 K literature values is partly a coincidence of two errors cancelling. The correct public claim is "the fitted intercept", never "the band gap at 0 K".

## 1. Method

- Primary sources were obtained and read, not recalled: Varshni (1967) [V67], O'Donnell and Chen (1991) [OC91], Viña, Logothetidis and Cardona (1984) [VLC84], Cohen and Bergstresser (1966) [CB66], plus the two TI datasheets [TI-ADC, TI-LM] and the lab handout [HO].
- The lab report [RPT] is the object of the audit, not a source of physical constants.
- All arithmetic uses `k = 8.617333262×10⁻⁵ eV/K`, `q = 1`, and `λ[nm] = 1239.84 / E[eV]`. The window used for the "lab window" columns is 300–383.15 K, the report's stated ceiling; the report never records the actual per-run start temperature.
- Reproduce every table with `python scripts/physics-audit.py`.

## 2. The handout's parameter table is mislabelled

The handout gives [HO]:

| Semiconductor | "E_g(0)" | α (eV/K) | β (K) |
|---|---|---|---|
| Ge | 0.67 | 4.56×10⁻⁴ | 210 |
| Si | 1.12 | 7.02×10⁻⁴ | 1108 |

Varshni's Table I gives, for the **indirect** gap [V67]:

| Semiconductor | E_g(0) | α (eV/K) | β (K) |
|---|---|---|---|
| Ge | 0.7412 | 4.56×10⁻⁴ | 210 |
| Si | 1.1557 | 7.021×10⁻⁴ | 1108 |

The α and β match Varshni exactly; the `E_g(0)` entries do not. Evaluating Varshni with the handout's numbers at 300 K:

| Set | E_g(0) | E_g(300 K) | E_g(383.15 K) |
|---|---|---|---|
| Si, correct 0 K set | 1.1557 | **1.1108** | 1.0866 |
| Si, handout treated literally | 1.1200 | 1.0751 | 1.0509 |
| Ge, correct 0 K set | 0.7412 | **0.6607** | 0.6283 |
| Ge, handout treated literally | 0.6700 | 0.5895 | 0.5571 |

So 1.12 and 0.67 are the **rounded 300 K gaps**, substituted into the `E_g(0)` slot of a 0 K parameter set. Taken literally, the handout's Si row predicts 1.075 eV at 300 K, contradicting the 1.12 eV written in the same row.

Consequences:

1. The report's statement that "the values of `E_g` can be denoted as `E_g(0)`" [RPT] is wrong for both diodes.
2. The comparison table against 0.67/1.12 [RPT] is a comparison of a fitted intercept with **300 K** gaps, not with 0 K gaps. It is not the like-for-like test the report presents.
3. The lab's own Question 1 ("calculate `E_g(T)` at 0 °C, 20 °C, 100 °C") is unanswerable from the printed table without knowing which value is which.

**Verdict:** resolved. Use 1.1557 eV (Si) and 0.7412 eV (Ge) with the printed α, β. Public copy must not call 1.12/0.67 the 0 K gaps.

## 3. What a straight line through `V(T)` actually estimates

Starting from the lab's own equations [RPT] and keeping the prefactor explicit:

```text
I      = I_s exp(qV / kT)
I_s(T) = B T^η exp(−E_g(T) / kT)
⇒ V(T) = E_g(T)/q + (k T/q)[ ln(I/B) − η ln T ]
```

A least-squares line `a + bT` fitted over `[T₁, T₂]` returns an intercept `a` that is **not** `E_g(0)/q`:

1. **Gap curvature.** For Varshni, `E_g` is concave down, so its chord extrapolates above `E_g(0)` by

   ```text
   ΔE_chord = α β T₁ T₂ / [(T₁ + β)(T₂ + β)]
   ```

   Over 300–383.15 K this is **+43 mV** for Si and **+37 mV** for Ge. Varshni himself documented the effect for Ge: extrapolating the `E_g`–`T` plot from data above 195 K gives "≈ 0.78 eV", whereas the measured 0 K gap is 0.7412 eV [V67, p. 152]. The computed chord value 0.7778 eV reproduces that statement.

2. **Prefactor curvature.** The `−η(kT/q)ln T` term is not linear. Over the same window its chord intercept is **+29.3 mV per unit η**: +44 mV for η = 1.5, +88 mV for η = 3, +147 mV for η = 5. The `ln(I/B)` term is exactly linear in `T` and therefore contributes **zero** to the intercept; it only changes the slope.

Computed chord intercepts for the gap models (300–383.15 K, uniform weighting):

| Model | E_g(0) | Chord intercept, η=0 | Chord intercept, η=3 | Chord slope, η=3 |
|---|---|---|---|---|
| Si (Varshni 1967) | 1.1557 | 1.1987 | 1.2866 | −2.06 meV/K |
| Si (1.166 / 4.73×10⁻⁴ / 636) | 1.1660 | 1.2025 | 1.2905 | −2.04 meV/K |
| Ge (Varshni 1967) | 0.7412 | 0.7778 | 0.8657 | −2.16 meV/K |

The tangent construction quoted in design §4.2 gives 1.197 eV (Si) and 0.777 eV (Ge) at 330 K, within 2 mV of the chord values. So the design's back-of-envelope "≈ 1.2 eV and ≈ 0.78 eV" is confirmed.

**Verdict:** resolved. The intercept of the lab's straight line is `E_g(0) + 40–90 mV` for plausible η, not `E_g(0)`. It is also not `E_g(T_mean)`: at 300 K the gaps are 1.111 (Si) and 0.661 (Ge), both below the chord.

## 4. The report's five intercepts, compared

Report values and quoted uncertainties [RPT]; gaps from [V67]; LED reference values from the report's cited Collings table [RPT].

| Diode | Reported intercept ± δ | E_g(0) | E_g(300 K) | Chord, η=0 | Intercept − chord | Verdict |
|---|---|---|---|---|---|---|
| Ge | 0.646 ± 0.0071 | 0.7412 | 0.6607 | 0.7778 | **−0.132** | not `E_g(0)`, not 300 K gap |
| Si | 1.002 ± 0.0113 | 1.1557 | 1.1108 | 1.1987 | **−0.197** | not `E_g(0)`, not 300 K gap |
| Red LED | 2.030 ± 0.0120 | n/a (alloy) | 1.97 (reference) | n/a | — | near 300 K emission, not 0 K |
| Blue LED | 2.796 ± 0.0126 | n/a (alloy) | 2.67 (reference) | n/a | — | 443 nm part, not the 465 nm reference |
| Green LED | 2.303 ± 0.0112 | n/a (alloy) | 2.30 (reference) | n/a | — | near 300 K emission |

### 4.1 The Si and Ge intercepts cannot be rescued by η

The deficit is 132 mV (Ge) and 197 mV (Si). Since each unit of η raises the intercept by 29.3 mV, closing the gap would need **η ≈ −4.5** (Ge) and **η ≈ −6.7** (Si). Negative η is unphysical: the prefactor would have to *decrease* with temperature faster than any transport term allows. The discrepancy is therefore not a prefactor effect; the measured `V(T)` is steeper than `E_g(T)` predicts.

Candidate causes, none of which the report can separate:

- **Temperature scale.** A thermocouple or plate/junction offset changes both slope and intercept. The report says thermocouple sensitivity dominates its uncertainty but gives no calibration.
- **Current.** `R_set` is ambiguous (Section 5); a 10× current error moves `V` by 60 mV at n = 1, which changes the fitted slope and therefore the extrapolated intercept.
- **Ideality and series resistance.** `n > 1` multiplies the prefactor term and the gap term; `R_s` adds a constant. Neither is recorded.
- **Fit window.** The actual per-run temperature ranges are not in the report; curvature bias depends on them.

The honest treatment is to publish the intercept as a *fitted* quantity and show the three candidate gaps beside it, which is what the app's "What does the straight line actually estimate?" panel does with the report summary. It must not quietly replace the report's numbers with corrected ones, and it cannot refit them without the lost recordings.

### 4.2 The quoted uncertainties are not total uncertainties

δ is described as a max–min trend-line spread [RPT]. It does not include:

- the η term: ±0.5 in η alone moves the intercept ±15 mV, larger than every quoted δ;
- the gap parameters: two published Si sets give 1.1987 vs 1.2025 (4 mV), small by comparison;
- temperature calibration and current setting: unquantified, and the 132–197 mV residual above shows they dominate.

**Verdict:** publish δ only as "trend-line spread", never as the uncertainty on `E_g`.

### 4.3 The LED comparisons are not like-for-like either

LED `E_g(0)` is not defined without the die composition, and the report does not record part numbers for the LEDs. The "expected" values (1.97, 2.67, 2.30 eV) are textbook examples, not the bins of the actual parts:

- The blue LED measured 2.796 eV = 443.4 nm. The 2.67 eV reference corresponds to 465 nm, a different wavelength bin. The 4.7% "error" is mostly a part-selection difference, not a measurement error.
- The red LED measured 2.030 eV = 610.8 nm; the reference 630 nm is again a different bin. The report's own 613 nm entry does not follow from its `E_g` (Section 7).
- Green agrees with its reference (2.303 vs 2.30) but that is one point, not evidence the method is unbiased.

What *is* meaningful for an LED is internal consistency: the measured forward-voltage intercept should track the photon energy of the same die at the same temperature. The report's red and green values do; the blue value is consistent with its own 443 nm emission. The percentage errors against a textbook bin should not be presented as accuracy.

## 5. `R_set`, current and the 10× ambiguity — parked

The report says "arbitrarily, `R_set = 220 Ω`, `R_set = 2200 Ω` and `R_temp = 150 Ω`" [RPT], but never says which run used which resistor, and the raw logs no longer exist.

Using the LM234 relation `I_set = γT / R_set` with γ = 227 µV/K (Section 6) [TI-LM, HO]:

| `R_set` | `I_set` at 298 K | `I_set` at 383 K | `V_Rtemp` at 150 Ω |
|---|---|---|---|
| 220 Ω | 307 µA | 395 µA | 46 mV |
| 2200 Ω | 30.7 µA | 39.5 µA | 4.6 mV |

A 10× current step shifts the forward voltage by `n(kT/q)ln 10` = **59.5 mV** at 300 K for n = 1, or 119 mV for n = 2. Both are 5–10× the quoted δ, so the ambiguity is material, not cosmetic. At 30 µA the 1N4007 is also deep in the recombination-dominated regime, where n ≈ 2 and the ideal-diode assumption is weakest [SZE].

**Verdict:** unavailable, **do not publish per-run current or `R_set`**. The logs needed to resolve the assignment are lost. The app may show the two candidate currents as a sensitivity toggle, labelled as a hypothetical assumption.

## 6. Bose–Einstein form and attribution — corrected

The design's form [DES] is:

```text
E_g(T) = E_B − a_B [ 1 + 2 / (exp(Θ_B/T) − 1) ]
```

This is algebraically the same curve as the standard single-oscillator form [OC91]:

```text
E_g(T) = E_g(0) − 2 a_B / (exp(Θ_B/T) − 1),        a_B = S⟨ħω⟩
```

but only if `E_B = E_g(0) + a_B`. **`E_B` is not the 0 K gap.** Writing the first form and calling `E_B` `E_g(0)` shifts the fitted gap by `a_B` — 38 meV for Si using the published fit below. The model must use the second form, or rename the parameter.

Attribution also needs correcting:

- Viña, Logothetidis and Cardona (1984) measured the **dielectric function of Ge** from 1.25 to 5.6 eV over 100–850 K and fitted the temperature dependence of the **E₁, E₁+Δ₁, E₀′ and E₂ critical points**, not the indirect fundamental gap [VLC84]. They do show that the critical-point energies can be represented by a Bose–Einstein statistical factor, which is the precedent the design cites, but they are not the source of fundamental-gap parameters.
- The three-parameter gap form and its justification are O'Donnell and Chen (1991) [OC91]. Their Si fit over 0–300 K gives `E_g(0) = 1.170 eV`, `S = 1.49`, `⟨ħω⟩ = 25.5 meV`, i.e. `a_B = 38.0 meV`.
- Varshni's relation is empirical and, as O'Donnell and Chen note, its β can even be negative; at low temperature it predicts a quadratic dependence where experiment is nearly flat [OC91]. Over 300–383 K the two models are practically indistinguishable, so comparing them by evidence is honest but cannot be expected to prefer one.

**Verdict:** form corrected; Viña's role narrowed to the critical-point precedent; O'Donnell–Chen cited for the gap form and Si parameters. `E_g(0)`, not `E_B`, is the 0 K gap.

## 7. Report arithmetic

| Quantity | Report | Recomputed | Verdict |
|---|---|---|---|
| Red wavelength from 2.030 eV | 613 nm | **610.76 nm** | report value is a typo or a measured wavelength, not `1239.84/E_g` |
| Blue wavelength from 2.796 eV | 444 nm | 443.43 nm | rounding, fine |
| Green wavelength from 2.303 eV | 539 nm | 538.36 nm | rounding, fine |
| Ge percentage error vs 0.67 | 3.58% | 3.58% | arithmetic correct, comparison label wrong |
| Si percentage error vs 1.12 | 10.54% | 10.54% | arithmetic correct, comparison label wrong |
| Red percentage error vs 630 nm | 2.70% | 2.70% (from 613) / 3.06% (from `E_g`) | two different numerators, not stated |
| Blue percentage error vs 465 nm | 4.52% | 4.52% (from 444) | correct arithmetic, wrong bin comparison |

**Verdict:** resolved. Publish 610.8 nm from the measured gap; if the report's 613 nm is a measured emission wavelength, label it as such and note the 2.3 nm discrepancy.

## 8. EPM form factors — verified

Cohen and Bergstresser Table II [CB66], symmetric form factors in rydbergs, with the lattice constants of their Table III:

| Material | a (Å) | V₃ | V₈ | V₁₁ |
|---|---|---|---|---|
| Si | 5.43 | −0.21 | +0.04 | +0.08 |
| Ge | **5.65** | −0.23 | +0.01 | +0.06 |

The design's recalled values are correct except the Ge lattice constant, which it gives as 5.66 Å; Cohen and Bergstresser use 5.65 Å. The paper states form-factor accuracy of 0.01 Ry and band energies good to ~0.01 Ry near the gap [CB66] — 0.01 Ry = 0.136 eV. The validation tolerance of 0.1 eV for Si/Ge band edges is therefore at the method's stated accuracy; use 0.15 eV, or require correct ordering and gap type rather than a tight absolute match.

**Verdict:** resolved, with the Ge lattice constant corrected to 5.65 Å and the validation tolerance flagged for revision.

## 9. The `T^η` prefactor — illustrative, not derived

The standard diffusion-dominated saturation current is [SZE, PS02]:

```text
I_s = A T^(3 + γ/2) exp(−E_g / kT)
```

where γ comes from the temperature dependence of `D/τ`. Comparing with `I_s = B T^η exp(−E_g/kT)` gives `η = 3 + γ/2`. Nothing in the lab measures γ, and the regime matters:

- diffusion-dominated: n ≈ 1, η = 3 + γ/2;
- recombination-dominated: n ≈ 2, and the current scales closer to `n_i`, i.e. an effective exponent near 3/2;
- at the currents implied by `R_set` (30–400 µA), a 1N4007 is likely recombination-dominated at the low end [SZE].

So η ≈ 3 is a defensible *central* value for the diffusion picture with weak `D/τ` dependence, but it is not derived, and it is degenerate with `ln B` over a 60–100 K window. The design's prior `Normal(3, 0.5)` is too tight: ±0.5 moves the M0 intercept ±15 mV, more than the report's quoted uncertainty, and a plausible range 1.5–5 spans 103 mV.

**Verdict:** parked as illustrative. Recommended prior: `Normal(3, 1)` truncated to [1.5, 5], with the η-sensitivity flag on every ledger entry. Do not present η = 3 as measured or derived.

## 10. Instrument arithmetic — partly verified

| Item | Claim | Source | Verdict |
|---|---|---|---|
| ADS1115 resolution | signed 16-bit; 125 µV LSB at ±4.096 V; quantization σ = LSB/√12 = 36 µV | [TI-ADC], Table 7-1 | verified |
| LM234 set current | 64 mV at 25 °C, slope 227 µV/K; `I_set = 67.7 mV / R_set` at 25 °C | [TI-LM] | verified |
| Handout Eq. 5 | `I_set = T / R_set` | [HO] | dimensionally incomplete: it omits γ = 227 µV/K |
| `R_temp` current readout | 150 Ω gives 46 mV (307 µA) or 4.6 mV (30.7 µA) | computed | one LSB is 0.27% and 2.7% of the readout respectively |
| Thermocouple + multimeter | offset/gain/lag priors | not in `source/` | **parked — do not publish**; needs the actual meter datasheet |
| 1N4007 ideality at these currents | n ≈ 1 | not measured | **parked**; lab Question 9 is the measurement that settles it |

The design's instrument constants are correct. The thermocouple model cannot be completed from the material in `source/`, so the site must not claim a calibrated temperature uncertainty until the meter datasheet is added.

## 11. Verdict on every `[VERIFY]` item

| Design item | Verdict | Evidence | Publish rule |
|---|---|---|---|
| Si/Ge `E_g(0)` and Varshni sets | **resolved** | §2, [V67] | use 1.1557 / 0.7412, never 1.12 / 0.67 as 0 K |
| Varshni vs Bose–Einstein form | **resolved** | §6, [OC91], [VLC84] | use `E_g(0) − 2a_B/(e^{Θ/T}−1)`; `E_B` is not `E_g(0)` |
| `T^η` prefactor, η ≈ 3 | **parked (illustrative)** | §9, [SZE], [PS02] | label illustrative; prior 3 ± 1, range 1.5–5 |
| EPM form factors | **resolved** | §8, [CB66] | Ge a = 5.65 Å, not 5.66 |
| Report wavelengths and `R_set` | wavelengths resolved; `R_set` unavailable | §5, §7 | 610.8 nm from the measured gap; no per-run current claim |
| 1N4007 diffusion vs recombination | **parked** | §9, [SZE] | no ideality claim until measured |
| Instrument specs | **partly resolved** | §10, [TI-ADC], [TI-LM] | no thermocouple-uncertainty claim without datasheet |

## 12. What changes in the repository

- `docs/model.md`: Bose–Einstein equation replaced with the standard form; `E_B` renamed; η prior annotated as illustrative with the audit's recommended range.
- `docs/validation.md`: the "intercept is not the room-temperature gap" row is now backed by this note and the script.
- Public copy rules:
  1. say "fitted intercept", never "band gap at 0 K";
  2. for Si and Ge, show intercept, `E_g(0)` and `E_g(300 K)` side by side (the app panel in §4);
  3. for LEDs, compare with the same die's emission energy at the measurement temperature, not a textbook bin;
  4. quote δ as trend-line spread, not total uncertainty;
  5. no per-run `R_set`/current, no thermocouple calibration, and no 1N4007 ideality; the original recordings are lost and the meter datasheet is still missing.

## 13. References

- [V67] Y. P. Varshni, "Temperature dependence of the energy gap in semiconductors", *Physica* **34**, 149–154 (1967). doi:[10.1016/0031-8914(67)90062-6](https://doi.org/10.1016/0031-8914(67)90062-6). Table I and p. 152.
- [OC91] K. P. O'Donnell and X. Chen, "Temperature dependence of semiconductor band gaps", *Appl. Phys. Lett.* **58**, 2924–2926 (1991). doi:[10.1063/1.104723](https://doi.org/10.1063/1.104723). Eq. 5 and Table I.
- [VLC84] L. Viña, S. Logothetidis and M. Cardona, "Temperature dependence of the dielectric function of germanium", *Phys. Rev. B* **30**, 1979–1982 (1984). doi:[10.1103/PhysRevB.30.1979](https://doi.org/10.1103/PhysRevB.30.1979).
- [CB66] M. L. Cohen and T. K. Bergstresser, "Band structures and pseudopotential form factors for fourteen semiconductors of the diamond and zinc-blende structures", *Phys. Rev.* **141**, 789–796 (1966). doi:[10.1103/PhysRev.141.789](https://doi.org/10.1103/PhysRev.141.789). Tables II and III.
- [SZE] S. M. Sze and K. K. Ng, *Physics of Semiconductor Devices*, 3rd ed., Wiley (2007), ch. 2: `J_s ∝ T^(3+γ/2) exp(−E_g/kT)` and ideality 1 vs 2.
- [PS02] J. W. Precker and M. A. da Silva, "Experimental estimation of the band gap in silicon and germanium from the temperature–voltage curve of diode thermometers", *Am. J. Phys.* **70**, 1150–1153 (2002). doi:[10.1119/1.1512658](https://doi.org/10.1119/1.1512658). Eq. 3.
- [TI-ADC] Texas Instruments, *ADS111x Ultra-Small, Low-Power, I²C-Compatible, 860-SPS, 16-Bit ADCs*, SBAS444E, Table 7-1 (LSB size by FSR).
- [TI-LM] Texas Instruments, *LM134/LM234/LM334 3-Terminal Adjustable Current Sources*, SNVS746E, sense voltage 64 mV at 25 °C and 227 µV/°C slope.
- [HO] PHYC30021 lab handout, `source/Semiconductors.pdf`, §1 and §3.4.
- [RPT] PHYC30021 lab report, `source/Journal_Report/main.md`.
- [DES] Design document v2, `source/bandgap-observatory-design.md`, §§3.10, 3.11, 4.2, 4.5.
