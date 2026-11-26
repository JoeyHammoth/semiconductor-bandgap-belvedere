<script lang="ts">
  import { onMount } from 'svelte';
  import { EnginePool, defaultPoolSize } from '../../lib/engine/pool';
  import type { NpeResult } from '../../lib/engine/protocol';
  import { loadInferCase, wasmBase, type InferCase } from '../../lib/infer/case';
  import { npeSummaries } from '../../lib/infer/npe';
  import StatusBar from '../../lib/ui/StatusBar.svelte';

  interface Calibration {
    pass: boolean;
    synthetic: boolean;
    note: string;
    sbc: { p_values: number[]; pass: boolean; threshold: number; simulations: number };
    tarp: { max_deviation: number; pass: boolean; threshold: number };
    ood: { tpr: number; fpr: number; pass: boolean; threshold: number };
  }

  let pool: EnginePool | null = null;
  let caseData = $state<InferCase | null>(null);
  let manifestJson = $state('');
  let weights = $state<Uint8Array | null>(null);
  let calibration = $state<Calibration | null>(null);
  let result = $state<NpeResult | null>(null);
  let error = $state<string | null>(null);
  let running = $state(false);
  let canvas = $state<HTMLCanvasElement | null>(null);
  let daylight = $state(false);

  const commit = typeof __COMMIT_HASH__ === 'string' ? __COMMIT_HASH__ : 'dev';
  const ready = $derived(Boolean(caseData && manifestJson && weights));
  const summary = $derived(
    result
      ? result.parameterNames.map((name, index) => ({
          name: name === 'eg_ev' ? 'E_g (eV)' : name === 'ln_b' ? 'ln B (A K⁻¹)' : name,
          mean: result!.mean[index],
          sd: Math.sqrt(result!.variance[index]),
          q05: result!.q05[index],
          q50: result!.q50[index],
          q95: result!.q95[index],
        }))
      : [],
  );
  const sbcP = $derived(calibration ? Math.min(...calibration.sbc.p_values).toFixed(3) : '—');

  function applyTheme() {
    document.documentElement.dataset.theme = daylight ? 'daylight' : 'darkroom';
  }

  async function loadArtifacts() {
    const [manifestResponse, weightResponse, calibrationResponse] = await Promise.all([
      fetch(new URL('../weights/m0-flow.json', document.baseURI)),
      fetch(new URL('../weights/m0-flow.bin', document.baseURI)),
      fetch(new URL('../data/npe-calibration.json', document.baseURI)),
    ]);
    if (!manifestResponse.ok || !weightResponse.ok || !calibrationResponse.ok) {
      throw new Error('NPE artifacts could not be fetched');
    }
    manifestJson = await manifestResponse.text();
    weights = new Uint8Array(await weightResponse.arrayBuffer());
    calibration = (await calibrationResponse.json()) as Calibration;
    if (calibration.synthetic !== true || calibration.pass !== true) {
      throw new Error('committed NPE calibration is not marked as passing');
    }
  }

  async function run() {
    if (!caseData || !pool || !weights || !manifestJson || running) return;
    running = true;
    error = null;
    result = null;
    try {
      const condition = npeSummaries(caseData.observations.voltage_v);
      const response = await pool.npe({
        manifestJson,
        weights,
        condition,
        samples: 2_000,
        seed: 0x4e5045,
      });
      if (!response.ok) throw new Error(response.error);
      const value = response.value;
      if (typeof value !== 'object' || value === null || !('draws' in value)) {
        throw new Error('NPE worker returned no draws');
      }
      result = value as NpeResult;
      requestAnimationFrame(draw);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      running = false;
    }
  }

  function draw() {
    const element = canvas;
    if (!element || !result || result.sampleCount < 2) return;
    const width = Math.max(320, Math.floor(element.clientWidth));
    const height = Math.max(280, Math.floor(element.clientHeight));
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

    const first = result.draws.subarray(0, result.sampleCount);
    const second = result.draws.subarray(result.sampleCount, result.sampleCount * 2);
    const x0 = result.q05[0] - (result.q95[0] - result.q05[0]) * 0.25;
    const x1 = result.q95[0] + (result.q95[0] - result.q05[0]) * 0.25;
    const y0 = result.q05[1] - (result.q95[1] - result.q05[1]) * 0.25;
    const y1 = result.q95[1] + (result.q95[1] - result.q05[1]) * 0.25;
    const margin = { left: 58, right: 18, top: 18, bottom: 44 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const px = (value: number) => margin.left + ((value - x0) / (x1 - x0)) * plotWidth;
    const py = (value: number) => margin.top + (1 - (value - y0) / (y1 - y0)) * plotHeight;

    context.strokeStyle = hair;
    context.strokeRect(margin.left, margin.top, plotWidth, plotHeight);
    context.fillStyle = accent;
    context.globalAlpha = 0.45;
    for (let index = 0; index < result.sampleCount; index += 1) {
      context.beginPath();
      context.arc(px(first[index]), py(second[index]), 1.2, 0, Math.PI * 2);
      context.fill();
    }
    context.globalAlpha = 1;
    context.fillStyle = dim;
    context.font = '11px JetBrains Mono, monospace';
    context.fillText('E_g (eV)', margin.left + plotWidth / 2 - 24, height - 10);
    context.save();
    context.translate(14, margin.top + plotHeight / 2 + 20);
    context.rotate(-Math.PI / 2);
    context.fillText('ln B (A K⁻¹)', 0, 0);
    context.restore();
    context.fillStyle = ink;
    context.fillText(x0.toFixed(3), margin.left - 8, height - 27);
    context.fillText(x1.toFixed(3), margin.left + plotWidth - 34, height - 27);
    context.fillText(y0.toFixed(2), 8, margin.top + plotHeight);
    context.fillText(y1.toFixed(2), 8, margin.top + 10);
  }

  onMount(() => {
    daylight = localStorage.getItem('bgo-theme') === 'daylight';
    applyTheme();
    pool = new EnginePool(defaultPoolSize(), wasmBase());
    void (async () => {
      try {
        const [value] = await Promise.all([loadInferCase(), loadArtifacts()]);
        caseData = value;
      } catch (cause) {
        error = cause instanceof Error ? cause.message : String(cause);
      }
    })();
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
  <title>Instant posterior — Band-Gap Observatory</title>
  <meta
    name="description"
    content="A frozen conditional flow for the synthetic M0 diode case, with SBC, TARP and OOD calibration."
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
      <a href="../ledger/">Ledger</a>
      <a href="../npe/" aria-current="page">Instant</a>
      <button
        class="theme"
        type="button"
        onclick={() => {
          daylight = !daylight;
          applyTheme();
          localStorage.setItem('bgo-theme', daylight ? 'daylight' : 'darkroom');
          draw();
        }}
        aria-pressed={daylight}
      >
        {daylight ? 'Darkroom' : 'Daylight'}
      </button>
    </nav>
  </header>

  <main>
    <p class="eyebrow">Phase 3 · S5 · synthetic M0 local amortised posterior</p>
    <h1>Instant posterior</h1>
    <p class="lede">
      A frozen conditional affine flow turns five design-aware sweep summaries into posterior draws.
      It is local to the report-anchored synthetic M0 case, not a substitute for the full
      broad-prior MCLMC run. The calibration report is shipped with the weights and is required
      before the page will run.
    </p>

    <section class="run panel" aria-labelledby="run-title">
      <div>
        <h2 id="run-title">Run the frozen flow</h2>
        <p class="caveat">
          Synthetic observations only. The original recordings do not exist (ADR 0006).
        </p>
      </div>
      <button type="button" onclick={run} disabled={!ready || running} data-testid="npe-run">
        {running ? 'Running…' : 'Run instant posterior'}
      </button>
      {#if error}<p class="error" role="status">{error}</p>{/if}
    </section>

    {#if result}
      <section class="result" data-testid="npe" aria-labelledby="result-title">
        <div class="section-head">
          <h2 id="result-title">2,000 draws from the conditional flow</h2>
          <span class:badge-ok={!result.ood} class:badge-warn={result.ood} data-testid="npe-ood">
            {result.ood ? 'OOD input — do not trust this posterior' : 'In-distribution input'}
          </span>
        </div>
        <figure class="plot">
          <canvas bind:this={canvas} data-testid="npe-plot" aria-hidden="true"></canvas>
          <figcaption>
            Joint posterior in physical coordinates. The flow output is a normalizing-flow
            approximation, not a converged MCMC sample.
          </figcaption>
        </figure>
        <table class="summary" data-testid="npe-table">
          <caption>S5 accessible equivalent: marginal posterior summaries</caption>
          <thead>
            <tr>
              <th scope="col">Parameter</th><th scope="col">Mean</th><th scope="col">SD</th>
              <th scope="col">5%</th><th scope="col">Median</th><th scope="col">95%</th>
            </tr>
          </thead>
          <tbody>
            {#each summary as row}
              <tr>
                <th scope="row">{row.name}</th>
                <td class="tabular">{row.mean.toFixed(6)}</td>
                <td class="tabular">{row.sd.toFixed(6)}</td>
                <td class="tabular">{row.q05.toFixed(6)}</td>
                <td class="tabular">{row.q50.toFixed(6)}</td>
                <td class="tabular">{row.q95.toFixed(6)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
        <p class="receipt">
          OOD distance <span class="tabular">{result.oodDistance.toFixed(2)}</span> vs threshold
          <span class="tabular">{result.oodThreshold.toFixed(2)}</span>. Weights
          <code>{result.weightsSha256.slice(0, 12)}…</code>.
        </p>
      </section>
    {/if}

    {#if calibration}
      <section class="calibration" aria-labelledby="cal-title">
        <h2 id="cal-title">Calibration is a gate, not a badge</h2>
        <p>
          SBC minimum p-value <span class="tabular">{sbcP}</span>; TARP maximum deviation
          <span class="tabular">{calibration.tarp.max_deviation.toFixed(3)}</span>; OOD TPR
          <span class="tabular">{calibration.ood.tpr.toFixed(2)}</span> with FPR
          <span class="tabular">{calibration.ood.fpr.toFixed(2)}</span>. The report is generated
          offline from 1,000 synthetic simulations and 100 posterior draws each.
        </p>
        <p class="caveat">
          Local synthetic M0 only. SBC/TARP do not certify the broad M0 prior or any measured
          dataset.
          <a
            href="https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/blob/main/docs/validation.md"
          >
            Validation record
          </a>.
        </p>
      </section>
    {/if}
  </main>
  <StatusBar backend="wasm" core={ready ? 'npe' : 'loading'} phase="Phase 3" {commit} />
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
  .result,
  .calibration {
    margin-top: 3rem;
  }
  .section-head {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    align-items: center;
    justify-content: space-between;
  }
  .badge-ok,
  .badge-warn {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 0.3rem 0.6rem;
    border-radius: 999px;
    border: 1px solid var(--hairline);
  }
  .badge-ok {
    color: oklch(0.8 0.13 150);
  }
  .badge-warn {
    color: oklch(0.8 0.16 40);
    border-color: oklch(0.8 0.16 40);
  }
  .plot {
    margin: 1rem 0 1.5rem;
  }
  canvas {
    width: 100%;
    height: 340px;
    background: var(--surface);
    border: var(--hairline);
    border-radius: var(--radius);
  }
  figcaption {
    font-size: 0.78rem;
    color: var(--ink-dim);
    padding-top: 0.4rem;
  }
  .summary {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.8rem;
  }
  .summary caption {
    text-align: left;
    color: var(--ink-dim);
    font-size: 0.75rem;
    padding-bottom: 0.5rem;
  }
  .summary th,
  .summary td {
    text-align: left;
    padding: 0.45rem 0.5rem;
    border-bottom: var(--hairline);
  }
  .calibration p {
    max-width: 70ch;
    color: var(--ink-dim);
  }
</style>
