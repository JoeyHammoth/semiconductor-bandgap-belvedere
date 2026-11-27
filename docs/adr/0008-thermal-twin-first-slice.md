# ADR 0008 — Start Phase 4 with a deterministic thermal chain

- Status: accepted
- Date: 2026-11-26

## Context

The S6 design calls for a latent junction temperature, sensor lag, a UKF/RTS smoother, a
state-space GP discrepancy and DMDc time constants. Those pieces depend on a trustworthy forward
thermal model. Starting with the full estimator stack would make the first scene impossible to
audit and would mix model error with inference error.

The original lab recordings do not exist (ADR 0006), so the S6 scene cannot use a measured plate or
sensor trace.

## Decision

Ship the first Phase 4 slice as a deterministic synthetic chain:

```
T_plate    -> first-order response to the heater/ambient
T_junction -> first-order response to T_plate
T_sensor   -> first-order response to T_junction
```

The discrete update uses the exact first-order factor `1 - exp(-dt/tau)` for each link. The model
also recovers the sensor time constant from the latent junction/sensor series using the exact
discrete relation `ΔT_sensor = α (T_junction,new - T_sensor,old)`, then converts `α` back to `tau`.

The S6 page draws the three states and the noisy synthetic sensor trace, reports the recovered and
true lag, and includes an accessible table.

## Consequences

- The first S6 scene has a reference model that later UKF/RTS and DMDc work can be tested against.
- The page states that this is not yet the UKF/RTS, GP-discrepancy or DMDc stage.
- The lag estimate is data-derived from the synthetic series, but it uses the latent junction state;
  it is not a claim that the real experiment can observe the junction directly.
- No measured data is introduced.
