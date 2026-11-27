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
            gp: result!.gpMeanK[point],
          };
        })
      : [],
  );
  const lagError = $derived(
    result
      ? Math.abs(result.lagTauS - result.trueTauS) / Math.max(1e-9, result.trueTauS)
      : null,
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

    const margin = { left: 58, right: 18, top: 24, bottom: 48 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
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
        context.lineTo(
          px(result!.timeS[index]),
          py(centre[index] - spread[index]),
        );
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
      slice of the M5 thermal model is a deterministic three-state chain, followed by a UKF/RTS
      state estimate of the hidden junction and a GP residual discrepancy.
    </p>

    <section class="run panel" aria-labelledby="run-title">
      <div>
        <h2 id="run-title">Run the thermal twin</h2>
        <p class="caveat">
          Synthetic only. This is not the lost lab trace; the GP uses fixed RBF hyperparameters as
          a diagnostic, not a calibrated physical discrepancy model.
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
            junction and the GP residual 1σ around the sensor.
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
            <strong class="tabular">{lagError === null ? '—' : `${(lagError * 100).toFixed(2)}%`}</strong>
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
        </div>
        <table class="trace-table" data-testid="thermal-table">
          <caption>S6 accessible equivalent: synthetic thermal state every 200 s</caption>
          <thead>
            <tr>
              <th scope="col">Time (s)</th>
              <th scope="col">Plate (K)</th>
              <th scope="col">Junction (K)</th>
              <th scope="col">Sensor (K)</th>
              <th scope="col">Observed (K)</th>
              <th scope="col">RTS junction (K)</th>
              <th scope="col">GP mean (K)</th>
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
                <td class="tabular">{row.gp.toFixed(3)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
        <p class="receipt">
          Ambient <span class="tabular">{result.ambientK.toFixed(2)} K</span>, heater
          <span class="tabular">+{result.heaterK.toFixed(1)} K</span>. The lag estimate uses the
          first-order relation between the latent junction and sensor series. The UKF runs on the
          noisy sensor only; RTS smoothing uses the full trace. Sensor RMSE is
          <span class="tabular">{result.sensorRmseK.toFixed(3)} K</span>.
        </p>
      </section>
    {/if}

    <section class="method" aria-labelledby="method-title">
      <p class="eyebrow">Estimator boundary</p>
      <h2 id="method-title">A reference chain, not a calibrated device model</h2>
      <p>
        The UKF sees only the noisy sensor and uses the synthetic first-order dynamics; RTS uses
        the full trace. The GP is fitted to the post-smoother residual with fixed RBF
        hyperparameters. Hankel-DMDc time constants and a learned discrepancy model remain future
        work.
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
    height: 360px;
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
