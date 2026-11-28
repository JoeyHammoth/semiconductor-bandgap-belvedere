# ADR 0012 — DMDc stays least-squares; the Matérn discrepancy gets a hyperprior

- Status: accepted
- Date: 2026-11-28
- Supersedes the fixed-hyperparameter note in [ADR 0011](0011-multi-pulse-schedule-and-matern-state.md)

## Context

Two loose ends from ADR 0011.

The fastest DMDc pole reads 25–34 s against the chain's 25 s across seeds. The obvious suspect is
errors-in-variables bias: measurement noise sits on both sides of an ARX regression, and the fastest
pole is the one with the least signal to support it.

The Matérn discrepancy was also run at fixed hyperparameters — 180 s length scale, 0.20 K signal
scale — which is a stated prior on the residual, not an estimate of it.

## Decision

### DMDc: least squares, with the noise sensitivity measured rather than hidden

Three estimators were measured on the same three-pulse schedule:

| Estimator | Default seed | Across six seeds |
|---|---|---|
| ordinary least squares | 433/74/29 s, max error 15.5% | fastest pole 25–34 s, max error 10–35% |
| two-stage least squares (lagged observations and heater lags as instruments) | 383/93/15 s, max error 39% | fastest pole 12–34 s, max error up to 90% |
| total least squares on the augmented `X`–`y` system (inverse iteration on its Gram matrix) | 358/236/115 s, max error 360% | no stable third mode |

Both errors-in-variables estimators were rejected. The instruments are nearly collinear with the
delay-embedded regressors at this sample size, so the variance they add exceeds the bias they
remove; and the embedding is collinear enough that the smallest singular direction of the augmented
system is the collinearity, not the noise.

On a noiseless trace the ordinary fit returns 420/75/25 s to 1e-7, which locates the error in the
observation noise rather than in the regression. Ordinary least squares therefore stays, and S6
keeps stating that the fastest pole is noise-limited.

### Matérn: a grid hyperprior over both scales

`matern_posterior` puts a Jeffreys (log-uniform) hyperprior on the length scale and the signal scale,
evaluates the marginal likelihood of the residual on an 8 × 6 grid (60–720 s, 0.005–0.16 K), and
moment-matches the mixture of the per-grid smoothed states. The mixture variance carries both the
within-grid uncertainty and the spread between grids, and `at_grid_edge` flags a posterior mean that
sits on a grid bound instead of at an interior maximum.

On the default run: length scale 163 s, signal scale 0.0052 K, 95.6% of residuals inside the
predictive 95% band, and a smoothed discrepancy RMS of 0.2 mK. The signal scale is at the grid floor,
so the honest reading is that the post-smoother residual is consistent with white sensor noise and
needs no smooth discrepancy. Raising the sensor noise to 0.1 K gives 151 s, 0.0066 K and 92.8%
coverage, so the fit is not pinned to one noise level.

Not claimed: an MCMC posterior over the hyperparameters, evidence comparable across different noise
assumptions, or a calibrated discrepancy for a real device.

## Consequences

- S6 draws a residual strip under the temperature plot with the smoothed discrepancy, its 1σ band
  and the residual itself, on a scale magnified to the residual, because a 5 mK band is invisible on
  a 21 K axis.
- S6 shows the fitted signal scale, length scale and 95% coverage next to the RBF diagnostic.
- A grid posterior is coarse. The reported signal scale sits on the floor of this trace, and the page
  says so instead of presenting it as an interior estimate.
- No measured data is introduced; the synthetic labels remain mandatory.
