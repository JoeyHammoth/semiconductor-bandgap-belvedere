<script lang="ts">
  import { onMount } from 'svelte';
  import { EnginePool, defaultPoolSize } from '../../lib/engine/pool';
  import type { ThermalTwinResult } from '../../lib/engine/protocol';
  import { wasmBase } from '../../lib/infer/case';
  import StatusBar from '../../lib/ui/StatusBar.svelte';

  let pool: EnginePool | null = null;
  let result = $state<ThermalTwinResult | null>(null);
  let error = $state<string | null>(null);
  let running = $state(false);
  let canvas = $state<HTMLCanvasElement | null>(null);
  let daylight = $state(false);

  const commit = typeof __COMMIT_HASH__ === 'string' ? __COMMIT_HASH__ : 'dev';
  const rows = $derived(
    result
      ? Array.from({ length: Math.ceil(result.sampleCount / 20) }, (_, index) => {
          const point = Math.min(result!.sampleCount - 1, index * 20);
          return {
            time: result!.timeS[point],
            plate: result!.plateK[point],
            junction: result!.junctionK[point],
            sensor: result!.sensorK[point],
            observed: result!.observedK[point],
            rts: result!.rtsJunctionK[point],
            rtsSd: result!.rtsSdK[point],
            gp: result!.gpMeanK[point],
            matern: result!.maternMeanK[point],
            maternSd: result!.maternSdK[point],
          };
        })
      : [],
  );
  const dmdModes = $derived(
    result
      ? Array.from(result.dmdTauS, (tau, index) => {
          const low = result!.dmdTauLowS[index];
          const high = result!.dmdTauHighS[index];
          const band =
            Number.isFinite(low) && Number.isFinite(high)
              ? ` (${low.toFixed(0)}–${high.toFixed(0)})`
              : '';
          return `${tau.toFixed(0)}${band}`;
        }).join(' · ')
      : '',
  );
  const greyBoxModes = $derived(
    result
      ? Array.from(result.greyboxTauS, (tau, index) => {
          const low = result!.greyboxTauLowS[index];
          const high = result!.greyboxTauHighS[index];
          const band =
            Number.isFinite(low) && Number.isFinite(high)
              ? ` (${low.toFixed(0)}–${high.toFixed(0)})`
              : '';
          return `${tau.toFixed(1)}${band}`;
        }).join(' · ')
      : '',
  );
  const lagError = $derived(
    result ? Math.abs(result.lagTauS - result.trueTauS) / Math.max(1e-9, result.trueTauS) : null,
  );

  function applyTheme() {
    document.documentElement.dataset.theme = daylight ? 'daylight' : 'darkroom';
    localStorage.setItem('bgo-theme', daylight ? 'daylight' : 'darkroom');
  }

  function toggleTheme() {
    daylight = !daylight;
    applyTheme();
    draw();
  }

  async function run() {
    if (running) return;
    running = true;
    error = null;
    try {
      pool ??= new EnginePool(defaultPoolSize(), wasmBase());
      const response = await pool.thermalTwin(0x5eed_7e11);
      if (!response.ok) throw new Error(response.error);
      const value = response.value;
      if (typeof value !== 'object' || value === null || !('timeS' in value)) {
        throw new Error('thermal worker returned no trace');
      }
      result = value as ThermalTwinResult;
      requestAnimationFrame(draw);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      running = false;
    }
  }

  function draw() {
    const element = canvas;
    if (!element || !result) return;
    const width = Math.max(320, Math.floor(element.clientWidth));
    const height = Math.max(300, Math.floor(element.clientHeight));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    element.width = width * dpr;
    element.height = height * dpr;
    const context = element.getContext('2d');
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    const styles = getComputedStyle(document.documentElement);
    const ink = styles.getPropertyValue('--ink').trim() || 'CanvasText';
    const dim = styles.getPropertyValue('--ink-dim').trim() || 'CanvasText';
    const accent = styles.getPropertyValue('--accent').trim() || 'CanvasText';
    const hair = styles.getPropertyValue('--hair').trim() || 'CanvasText';
    const sensorColour = 'oklch(0.82 0.15 40)';
    const smootherColour = 'oklch(0.72 0.12 260)';
    const discrepancyColour = 'oklch(0.78 0.11 160)';
    const greyboxColour = 'oklch(0.86 0.1 90)';

    const margin = { left: 58, right: 18, top: 24, bottom: 48 };
    const plotWidth = width - margin.left - margin.right;
    const available = height - margin.top - margin.bottom;
    const stripGap = 26;
    const ladderGap = 22;
    const ladderHeight = 72;
    const stripHeight = Math.max(52, Math.round(available * 0.2));
    const plotHeight = available - stripHeight - stripGap - ladderHeight - ladderGap;
    const values = [
      ...result.plateK,
      ...result.junctionK,
      ...result.sensorK,
      ...result.observedK,
      ...result.rtsJunctionK,
    ];
    const low = Math.min(...values) - 0.5;
    const high = Math.max(...values) + 0.5;
    const xMax = result.timeS[result.sampleCount - 1] || 1;
    const px = (time: number) => margin.left + (time / xMax) * plotWidth;
    const py = (temperature: number) =>
      margin.top + (1 - (temperature - low) / (high - low)) * plotHeight;

    context.strokeStyle = hair;
    context.lineWidth = 1;
    context.strokeRect(margin.left, margin.top, plotWidth, plotHeight);
    context.font = '11px JetBrains Mono, monospace';
    context.fillStyle = dim;
    for (let tick = 0; tick <= 3; tick += 1) {
      const temperature = low + ((high - low) * tick) / 3;
      const y = py(temperature);
      context.beginPath();
      context.moveTo(margin.left, y);
      context.lineTo(margin.left + plotWidth, y);
      context.stroke();
      context.fillText(`${temperature.toFixed(1)} K`, 6, y + 4);
    }
    for (let tick = 0; tick <= 3; tick += 1) {
      const time = (xMax * tick) / 3;
      context.fillText(`${Math.round(time)} s`, px(time) - 14, height - 24);
    }

    const line = (series: Float64Array, colour: string, widthPx: number, alpha = 1) => {
      context.globalAlpha = alpha;
      context.strokeStyle = colour;
      context.lineWidth = widthPx;
      context.beginPath();
      for (let index = 0; index < result!.sampleCount; index += 1) {
        const x = px(result!.timeS[index]);
        const y = py(series[index]);
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
      context.globalAlpha = 1;
    };
    const band = (centre: Float64Array, spread: Float64Array, colour: string, alpha: number) => {
      context.beginPath();
      for (let index = 0; index < result!.sampleCount; index += 1) {
        const x = px(result!.timeS[index]);
        const y = py(centre[index] + spread[index]);
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      for (let index = result!.sampleCount - 1; index >= 0; index -= 1) {
        context.lineTo(px(result!.timeS[index]), py(centre[index] - spread[index]));
      }
      context.closePath();
      context.fillStyle = colour;
      context.globalAlpha = alpha;
      context.fill();
      context.globalAlpha = 1;
    };
    const gpCentre = result.sensorK.map((value, index) => value + result!.gpMeanK[index]);
    band(result.rtsJunctionK, result.rtsSdK, smootherColour, 0.12);
    band(gpCentre, result.gpSdK, sensorColour, 0.08);
    line(result.plateK, dim, 1.25, 0.75);
    line(result.junctionK, accent, 2);
    line(result.sensorK, sensorColour, 1.5);
    context.save();
    context.setLineDash([6, 4]);
    line(result.rtsJunctionK, smootherColour, 1.6, 0.95);
    context.restore();

    context.fillStyle = sensorColour;
    for (let index = 0; index < result.sampleCount; index += 4) {
      context.beginPath();
      context.arc(px(result.timeS[index]), py(result.observedK[index]), 1.4, 0, Math.PI * 2);
      context.fill();
    }

    // Residual strip: the post-smoother sensor residual with the fitted
    // Matérn-3/2 1σ band drawn around its smoothed mean.
    const stripTop = margin.top + plotHeight + stripGap;
    const residual = result.observedK.map((value, index) => value - result!.rtsSensorK[index]);
    let extent = 0;
    for (let index = 0; index < result.sampleCount; index += 1) {
      extent = Math.max(
        extent,
        Math.abs(residual[index]),
        Math.abs(result.maternMeanK[index]) + result.maternSdK[index],
      );
    }
    extent = extent > 0 ? extent * 1.25 : 0.1;
    const ry = (value: number) => stripTop + (1 - (value + extent) / (2 * extent)) * stripHeight;

    context.globalAlpha = 1;
    context.strokeStyle = hair;
    context.lineWidth = 1;
    context.strokeRect(margin.left, stripTop, plotWidth, stripHeight);
    context.beginPath();
    context.moveTo(margin.left, ry(0));
    context.lineTo(margin.left + plotWidth, ry(0));
    context.stroke();

    context.beginPath();
    for (let index = 0; index < result.sampleCount; index += 1) {
      const x = px(result.timeS[index]);
      const y = ry(result.maternMeanK[index] + result.maternSdK[index]);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    for (let index = result.sampleCount - 1; index >= 0; index -= 1) {
      context.lineTo(
        px(result.timeS[index]),
        ry(result.maternMeanK[index] - result.maternSdK[index]),
      );
    }
    context.closePath();
    context.fillStyle = discrepancyColour;
    context.globalAlpha = 0.2;
    context.fill();
    context.globalAlpha = 1;

    context.fillStyle = dim;
    for (let index = 0; index < result.sampleCount; index += 1) {
      context.beginPath();
      context.arc(px(result.timeS[index]), ry(residual[index]), 1.1, 0, Math.PI * 2);
      context.fill();
    }
    context.strokeStyle = discrepancyColour;
    context.lineWidth = 1.4;
    context.beginPath();
    for (let index = 0; index < result.sampleCount; index += 1) {
      const x = px(result.timeS[index]);
      const y = ry(result.maternMeanK[index]);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
    context.fillStyle = dim;
    context.fillText(`residual ±${extent.toFixed(2)} K`, 6, stripTop + 12);

    // Mode ladder: both estimators' time constants and their 5–95% bands on a
    // log axis, so the modal comparison lives on the plot and not only in cards.
    const ladderTop = stripTop + stripHeight + ladderGap;
    const ladderTaus = [
      ...result.dmdTauS,
      ...result.dmdTauLowS,
      ...result.dmdTauHighS,
      ...result.greyboxTauS,
      ...result.greyboxTauLowS,
      ...result.greyboxTauHighS,
    ].filter((value) => Number.isFinite(value) && value > 0);
    const domainLow = ladderTaus.length ? Math.min(...ladderTaus) * 0.8 : 10;
    const domainHigh = ladderTaus.length ? Math.max(...ladderTaus) * 1.25 : 1000;
    const mx = (tau: number) =>
      margin.left + (Math.log(tau / domainLow) / Math.log(domainHigh / domainLow)) * plotWidth;
    const rowCount = Math.min(result.dmdTauS.length, result.greyboxTauS.length);
    context.strokeStyle = hair;
    context.lineWidth = 1;
    context.strokeRect(margin.left, ladderTop, plotWidth, ladderHeight);
    context.fillStyle = dim;
    context.fillText('τ (s, log)', 6, ladderTop + 11);
    for (let index = 0; index < rowCount; index += 1) {
      const y = ladderTop + 18 + index * 16;
      context.globalAlpha = 0.18;
      context.strokeStyle = hair;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(mx(domainLow), y);
      context.lineTo(mx(domainHigh), y);
      context.stroke();
      context.globalAlpha = 1;
      const dmdLow = result.dmdTauLowS[index];
      const dmdHigh = result.dmdTauHighS[index];
      if (Number.isFinite(dmdLow) && Number.isFinite(dmdHigh)) {
        context.strokeStyle = accent;
        context.lineWidth = 5;
        context.beginPath();
        context.moveTo(mx(dmdLow), y);
        context.lineTo(mx(dmdHigh), y);
        context.stroke();
      }
      const greyLow = result.greyboxTauLowS[index];
      const greyHigh = result.greyboxTauHighS[index];
      if (Number.isFinite(greyLow) && Number.isFinite(greyHigh)) {
        context.strokeStyle = greyboxColour;
        context.lineWidth = 3;
        context.beginPath();
        context.moveTo(mx(greyLow), y);
        context.lineTo(mx(greyHigh), y);
        context.stroke();
      }
      context.fillStyle = accent;
      context.beginPath();
      context.arc(mx(result.dmdTauS[index]), y, 3, 0, Math.PI * 2);
      context.fill();
      context.strokeStyle = greyboxColour;
      context.lineWidth = 1.5;
      context.strokeRect(mx(result.greyboxTauS[index]) - 2.5, y - 2.5, 5, 5);
      context.fillStyle = dim;
      context.fillText(`m${index + 1}`, 8, y + 3);
    }
    for (const tick of [20, 50, 100, 200, 400, 800]) {
      if (tick < domainLow || tick > domainHigh) continue;
      const x = mx(tick);
      context.strokeStyle = hair;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(x, ladderTop + ladderHeight - 20);
      context.lineTo(x, ladderTop + ladderHeight - 16);
      context.stroke();
      context.fillText(`${tick}`, x - 8, ladderTop + ladderHeight - 5);
    }

    context.fillStyle = ink;
    context.fillText('time (s)', margin.left + plotWidth / 2 - 24, height - 6);
    context.save();
    context.translate(15, margin.top + plotHeight / 2 + 34);
    context.rotate(-Math.PI / 2);
    context.fillText('temperature (K)', 0, 0);
    context.restore();
  }

  onMount(() => {
    daylight = localStorage.getItem('bgo-theme') === 'daylight';
    applyTheme();
    void run();
    const resize = () => draw();
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      pool?.dispose();
      pool = null;
    };
  });
</script>

<svelte:head>
  <title>Thermal twin — Band-Gap Observatory</title>
  <meta
    name="description"
    content="A synthetic first-order thermal twin showing plate, junction and lagging sensor temperature, with a recovered sensor time constant."
  />
</svelte:head>

<div class="shell">
  <header class="top">
    <a class="mark" href="../" aria-label="Band-Gap Observatory home">
      <span class="aperture" aria-hidden="true"></span><span>Band-Gap Observatory</span>
    </a>
    <nav aria-label="Sections">
      <a href="../">Home</a>
      <a href="../infer/">Infer</a>
      <a href="../evidence/">Evidence</a>
      <a href="../npe/">Instant</a>
      <a href="../thermal/" aria-current="page">Thermal</a>
      <a href="../story/">Story</a>
      <button class="theme" type="button" onclick={toggleTheme} aria-pressed={daylight}>
        {daylight ? 'Darkroom' : 'Daylight'}
      </button>
    </nav>
  </header>

  <main>
    <p class="eyebrow">Phase 4 · S6 · synthetic thermal twin</p>
    <h1>Where the sensor lags the junction</h1>
    <p class="lede">
      The plate heats, the diode junction follows, and the sensor follows more slowly. The first
      slice of the M5 thermal model is a deterministic three-state chain driven by a three-pulse
      heater schedule, followed by a UKF/RTS state estimate of the hidden junction, a GP residual
      diagnostic and a Matérn-3/2 latent discrepancy.
    </p>

    <section class="run panel" aria-labelledby="run-title">
      <div>
        <h2 id="run-title">Run the thermal twin</h2>
        <p class="caveat">
          Synthetic only. This is not the lost lab trace; the GP uses fixed RBF hyperparameters and
          the Matérn discrepancy a grid-fitted length and signal scale, both as diagnostics rather
          than calibrated physical models.
        </p>
      </div>
      <button type="button" onclick={run} disabled={running} data-testid="thermal-run">
        {running ? 'Running…' : 'Run twin'}
      </button>
      {#if error}<p class="error" role="status">{error}</p>{/if}
    </section>

    {#if result}
      <section class="result" data-testid="thermal" aria-labelledby="result-title">
        <div class="section-head">
          <h2 id="result-title">Plate → junction → sensor</h2>
          <span class="badge" data-testid="thermal-lag">
            τ sensor {result.lagTauS.toFixed(1)} s · true {result.trueTauS.toFixed(1)} s
          </span>
        </div>
        <figure class="plot">
          <canvas bind:this={canvas} data-testid="thermal-plot" aria-hidden="true"></canvas>
          <figcaption>
            Thin dim line: plate. Accent line: latent junction. Orange line: sensor. Dots: noisy
            synthetic observations. Dashed blue: RTS-smoothed junction. Bands: RTS 1σ around the
            junction and the GP residual 1σ around the sensor. Bottom strip: the post-smoother
            sensor residual as dots, with the smoothed Matérn-3/2 discrepancy (green line) and its
            1σ band on a scale magnified to the residual. Bottom ladder: fitted time constants on a
            log scale — filled bars and dots are DMDc with its 5–95% band, thin bars and open
            squares are the grey-box fit with its residual-bootstrap band.
          </figcaption>
        </figure>
        <div class="metrics">
          <p>
            <span class="label">Recovered lag</span>
            <strong class="tabular">{result.lagTauS.toFixed(2)} s</strong>
          </p>
          <p>
            <span class="label">True lag</span>
            <strong class="tabular">{result.trueTauS.toFixed(2)} s</strong>
          </p>
          <p>
            <span class="label">Relative error</span>
            <strong class="tabular">
              {lagError === null ? '—' : `${(lagError * 100).toFixed(2)}%`}
            </strong>
          </p>
          <p>
            <span class="label">Peak junction−sensor</span>
            <strong class="tabular">{result.maxLagK.toFixed(2)} K</strong>
          </p>
          <p data-testid="thermal-smoother">
            <span class="label">RTS junction RMSE</span>
            <strong class="tabular">{result.junctionRmseK.toFixed(3)} K</strong>
          </p>
          <p data-testid="thermal-gp">
            <span class="label">GP residual RMS</span>
            <strong class="tabular">{result.gpRmsK.toFixed(3)} K</strong>
          </p>
          <p data-testid="thermal-matern">
            <span class="label">Matérn σ (fitted)</span>
            <strong class="tabular">{result.maternSignalSdK.toFixed(4)} K</strong>
          </p>
          <p data-testid="thermal-matern-fit">
            <span class="label">Matérn ℓ · 95% coverage</span>
            <strong class="tabular">
              {result.maternLengthScaleS.toFixed(0)} s · {(result.maternCoverage95 * 100).toFixed(
                0,
              )}%
            </strong>
          </p>
          <p data-testid="thermal-likelihood">
            <span class="label">Kalman log-likelihood</span>
            <strong class="tabular">{result.kalmanLogLikelihood.toFixed(1)}</strong>
          </p>
          <p data-testid="thermal-modes">
            <span class="label">DMDc τ (5–95% band)</span>
            <strong class="tabular">
              {dmdModes} s
            </strong>
          </p>
          <p data-testid="thermal-greybox">
            <span class="label">Grey-box τ (5–95% band)</span>
            <strong class="tabular">{greyBoxModes} s</strong>
          </p>
        </div>
        <table class="trace-table" data-testid="thermal-table">
          <caption>
            S6 accessible equivalent: synthetic thermal state every 200 s, with the RTS and Matérn
            1σ bands drawn on the plot
          </caption>
          <thead>
            <tr>
              <th scope="col">Time (s)</th>
              <th scope="col">Plate (K)</th>
              <th scope="col">Junction (K)</th>
              <th scope="col">Sensor (K)</th>
              <th scope="col">Observed (K)</th>
              <th scope="col">RTS junction (K)</th>
              <th scope="col">RTS 1σ (K)</th>
              <th scope="col">GP mean (K)</th>
              <th scope="col">Matérn mean (mK)</th>
              <th scope="col">Matérn 1σ (mK)</th>
            </tr>
          </thead>
          <tbody>
            {#each rows as row}
              <tr>
                <td class="tabular">{row.time.toFixed(0)}</td>
                <td class="tabular">{row.plate.toFixed(2)}</td>
                <td class="tabular">{row.junction.toFixed(2)}</td>
                <td class="tabular">{row.sensor.toFixed(2)}</td>
                <td class="tabular">{row.observed.toFixed(2)}</td>
                <td class="tabular">{row.rts.toFixed(2)}</td>
                <td class="tabular">{row.rtsSd.toFixed(3)}</td>
                <td class="tabular">{row.gp.toFixed(3)}</td>
                <td class="tabular">{(row.matern * 1000).toFixed(1)}</td>
                <td class="tabular">{(row.maternSd * 1000).toFixed(1)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
        <p class="receipt">
          Ambient <span class="tabular">{result.ambientK.toFixed(2)} K</span>, heater
          <span class="tabular">+{result.heaterK.toFixed(1)} K</span>. The lag estimate uses the
          first-order relation between the latent junction and sensor series. The UKF runs on the
          noisy sensor only; RTS smoothing uses the full trace. Sensor RMSE is
          <span class="tabular">{result.sensorRmseK.toFixed(3)} K</span>. The post-smoother residual
          is carried as a two-state Matérn-3/2 SDE whose length and signal scales come from a grid
          hyperprior: ℓ = <span class="tabular">{result.maternLengthScaleS.toFixed(0)} s</span>, σ =
          <span class="tabular">{result.maternSignalSdK.toFixed(4)} K</span>{result.maternAtGridEdge
            ? ' (at the grid floor, so the residual needs no smooth discrepancy beyond sensor noise)'
            : ''}, with <span class="tabular">{(result.maternCoverage95 * 100).toFixed(1)}%</span>
          of residuals inside the 95% predictive band. Hankel-DMDc on the observed series fits a fourth-order
          companion model to the three-pulse schedule and reports the three distinct modes within
          <span class="tabular">{(result.dmdMaxRelativeError * 100).toFixed(1)}%</span> of the chain
          constants (fit residual
          <span class="tabular">{(result.dmdRelativeResidual * 100).toFixed(1)}%</span>). The bands
          come from
          <span class="tabular">{result.dmdBootstrapReplicates}</span> parametric-bootstrap refits
          of the fitted model over the trace; the middle and fastest poles are the least certain, so
          this is a diagnostic, not an identification claim. A grey-box output-error fit of the same
          chain, which assumes the three-state structure and the known heater schedule instead of
          identifying it, returns
          <span class="tabular">{greyBoxModes} s</span> (worst modal error
          <span class="tabular">{(result.greyboxMaxRelativeError * 100).toFixed(1)}%</span>,
          output-error RMS <span class="tabular">{result.greyboxOutputRmsK.toFixed(3)} K</span>,
          bands from <span class="tabular">{result.greyboxBootstrapReplicates}</span>
          residual-bootstrap refits). It recovers the sensor pole that the blind ARX fit biases; it is
          shown beside DMDc rather than replacing it, because it only validates the chain it already assumes.
        </p>
      </section>
    {/if}

    <section class="method" aria-labelledby="method-title">
      <p class="eyebrow">Estimator boundary</p>
      <h2 id="method-title">A reference chain, not a calibrated device model</h2>
      <p>
        The UKF sees only the noisy sensor and uses the synthetic first-order dynamics; RTS uses the
        full trace. The GP is fitted to the post-smoother residual with fixed RBF hyperparameters.
        The Kalman marginal likelihood is the filter's own predictive log-density, not a
        model-evidence estimate. Hankel-DMDc reports damping constants from a delay embedding. The
        Matérn-3/2 discrepancy is carried as a two-state latent SDE whose length and signal scales
        are fitted on a grid hyperprior, so the discrepancy is a state of the model rather than a
        kernel fit. The grey-box fit is the opposite trade: it assumes the chain structure to remove
        the errors-in-variables bias, so it cannot discover a structure the chain does not have.
      </p>
    </section>
  </main>
  <StatusBar backend="wasm" core="thermal" phase="Phase 4" {commit} />
</div>

<style>
  .shell {
    min-height: 100dvh;
    display: flex;
    flex-direction: column;
  }
  .top {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    justify-content: space-between;
    align-items: center;
    width: var(--page);
    margin: 0 auto;
    padding: 1.2rem 0;
    border-bottom: var(--hairline);
  }
  .mark {
    display: inline-flex;
    gap: 0.6rem;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink);
    text-decoration: none;
  }
  .aperture {
    width: 0.7rem;
    height: 0.7rem;
    border-radius: 50%;
    border: 1px solid var(--accent);
    box-shadow: 0 0 12px var(--accent);
  }
  nav {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    font-size: 0.82rem;
  }
  nav a {
    color: var(--ink-dim);
    text-decoration: none;
  }
  nav a[aria-current='page'] {
    color: var(--accent);
  }
  .theme {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-dim);
    background: transparent;
    border: var(--hairline);
    border-radius: 999px;
    padding: 0.35rem 0.7rem;
    cursor: pointer;
  }
  main {
    flex: 1;
    width: var(--page);
    margin: 0 auto;
    padding: clamp(2rem, 6vw, 5rem) 0 4rem;
  }
  h1 {
    font-size: var(--step-3);
    margin: 0.5rem 0 1rem;
    max-width: 18ch;
  }
  h2 {
    font-size: var(--step-2);
    margin: 0;
  }
  .lede {
    max-width: 70ch;
    color: var(--ink-dim);
  }
  .run {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    justify-content: space-between;
    align-items: center;
    padding: 1.2rem;
    margin: 2rem 0;
  }
  .run h2 {
    font-size: var(--step-1);
  }
  button {
    min-height: 44px;
    padding: 0.7rem 1rem;
    border-radius: 999px;
    border: 1px solid var(--accent);
    background: color-mix(in oklch, var(--accent), transparent 75%);
    color: var(--ink);
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.55;
    cursor: wait;
  }
  .caveat,
  .error,
  .receipt {
    font-size: 0.8rem;
    color: var(--ink-dim);
    max-width: 72ch;
  }
  .error {
    color: oklch(0.78 0.12 40);
    flex-basis: 100%;
  }
  .result {
    margin-top: 3rem;
  }
  .section-head {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    align-items: center;
    justify-content: space-between;
  }
  .badge {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 0.3rem 0.6rem;
    border-radius: 999px;
    border: 1px solid var(--hairline);
    color: oklch(0.82 0.15 40);
  }
  .plot {
    margin: 1rem 0 1.5rem;
  }
  canvas {
    width: 100%;
    height: 520px;
    background: var(--surface);
    border: var(--hairline);
    border-radius: var(--radius);
  }
  figcaption {
    font-size: 0.78rem;
    color: var(--ink-dim);
    padding-top: 0.4rem;
  }
  .metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.7rem;
    margin: 1rem 0 1.5rem;
  }
  .metrics p {
    margin: 0;
    padding: 0.7rem;
    border: var(--hairline);
    border-radius: 8px;
  }
  .label {
    display: block;
    font-family: var(--font-mono);
    font-size: 0.64rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-dim);
  }
  .metrics strong {
    display: block;
    margin-top: 0.2rem;
  }
  .trace-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.78rem;
  }
  .trace-table caption {
    text-align: left;
    color: var(--ink-dim);
    font-size: 0.75rem;
    padding-bottom: 0.5rem;
  }
  .trace-table th,
  .trace-table td {
    text-align: left;
    padding: 0.45rem 0.5rem;
    border-bottom: var(--hairline);
  }
  .method {
    border-top: var(--hairline);
    padding-top: 2rem;
    margin-top: 3rem;
  }
  .method p {
    max-width: 72ch;
    color: var(--ink-dim);
  }
  @media (max-width: 760px) {
    .metrics {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
