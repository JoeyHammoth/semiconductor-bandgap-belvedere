# Model specification

This is the normative description of the physics and the statistical model. If code and this document disagree, one of them is a bug. The originating design document is [`source/bandgap-observatory-design.md`](../source/bandgap-observatory-design.md); this file extracts the parts that must not be re-invented per module.

Status: extracted from design v2. Numbers marked **[verify]** are still awaiting the audit tracked in the Phase 0 physics audit issue.

## 1. Forward model (constant-current direction)

```
V = I·R_s + n·V_T·ln(I / I_s(T) + 1),        V_T = k·T / q
I_s(T) = B·T^η·exp( −E_g(T) / (n_a·k·T) )
```

Ideal limit (`n = n_a = 1`, `R_s = 0`, `I ≫ I_s`):

```
V(T) = (k·T/q)·[ln(I/B) − η·ln T] + E_g(T)/q
```

With `η = 0` and constant `E_g`, this reduces to the lab's straight line (report Eq. 8), so the fitted intercept is `E_g/q` only under those assumptions. That distinction is the scientific point of the project, not an implementation detail.

### Symbols

| Symbol | Meaning | Units |
|---|---|---|
| `V` | forward voltage across the diode | V |
| `I` | forward current held by the current source | A |
| `T` | junction temperature | K |
| `E_g(T)` | band-gap energy | eV |
| `B` | prefactor (material and device dependent) | A·K^-η |
| `η` | prefactor temperature exponent | dimensionless |
| `n` | ideality factor | dimensionless |
| `n_a` | exponent in the saturation-current activation | dimensionless |
| `R_s` | series resistance | Ω |
| `k`, `q` | Boltzmann constant, elementary charge | J/K, C |

Internally all quantities are SI; `E_g` is stored in eV only at presentation boundaries. Conversions live in `bgo-physics::units`.

## 2. Gap models

Varshni:

```
E_g(T) = E_g(0) − α·T² / (T + β)
```

Bose–Einstein (single effective phonon, after Viña, Logothetidis and Cardona 1984) **[verify form and parameter definitions]**:

```
E_g(T) = E_B − a_B·[ 1 + 2 / (exp(Θ_B / T) − 1) ]
```

The two are alternatives, not nested. `M2` and `M2b` are compared by evidence and by out-of-sample prediction, never by a likelihood-ratio test.

## 3. Model variants

| ID | Description | Free parameters (beyond noise) |
|---|---|---|
| M0 | Lab model: linear `V`–`T`, ideal diode | `E_g`, `ln B` |
| M1 | + prefactor `T^η` | + `η` |
| M2 | + Varshni `E_g(T)` | + `α`, `β` |
| M2b | Bose–Einstein `E_g(T)` instead of Varshni | + `a_B`, `Θ_B` |
| M3 | + ideality `n` | + `n` |
| M4 | + series resistance `R_s` | + `R_s` |
| M5 | + sensor lag / latent junction temperature | + thermal states |
| M6 | + state-space GP discrepancy | + GP hyperparameters |

Every variant shares the same data, likelihood and prior construction so that evidence comparisons are meaningful.

## 4. Instrument model

- **ADC (ADS1115):** signed 16-bit, `LSB = FSR / 32768` (125 µV at ±4.096 V); quantisation `σ_q = LSB/√12` plus a fitted noise floor.
- **Current:** `I = V_Rtemp / R_temp` with propagated uncertainty from resistor tolerance.
- **LM234 thermometer:** `T = I_set·R_set / γ`, `γ ≈ 227 µV/K` at 293 K, fitted per device.
- **Thermocouple and multimeter:** offset/gain priors from the meter datasheet **[fill in]**, plus a first-order lag.
- **Current-source drift:** slow multiplicative drift of `I_set`, measured per step where possible.

## 5. Priors

| Parameter | Prior | Notes |
|---|---|---|
| `E_g(0)` | wide Normal (weak) or literature-centred (informative) | both variants reported with a sensitivity flag |
| `ln B` | weak Normal scaled by simulation | scale recorded with every run |
| `η` | Normal(3, 0.5) | **[verify]** once mobility and diffusion-length terms are included |
| `n` | LogNormal around 1, truncated at `n ≥ 1` | by construction |
| `R_s` | HalfNormal at rectifier scale | |
| instrument terms | truncated from datasheets | no negative resistances or gains |

Prior sensitivity is a first-class output: every ledger entry carries a flag, and any conclusion that flips under the weak/informative prior swap is reported as prior-dependent.

## 6. Likelihood and noise

Voltage residuals are modelled as Gaussian with an instrument-derived scale plus an optional GP discrepancy term. When the latent-temperature model (M5) is active, the likelihood is the Kalman filter marginal likelihood over the thermal state, not a plug-in temperature.

## 7. Identifiability

`E_g`, `ln B` and `η` are nearly collinear over the lab's 60–100 K window around room temperature. The Fisher spectrum (design §3.1) quantifies this; the app's job is to show it rather than to hide it behind a straight-line fit. Any claim that a wider temperature range would "fix" the gap must be demonstrated with the temperature-range slider before it appears in copy.

## 8. Units and numerical conventions

- Temperatures in kelvin, never Celsius, inside the model.
- `ln T` and `ln B` require positive quantities; parameters are sampled unconstrained and transformed with Jacobians.
- All random draws are seeded; seeds are part of the view state and of every validation run.
- Floating point: `f64` in Rust core and Python references; `f32` only at the WebGPU/visual boundary.
