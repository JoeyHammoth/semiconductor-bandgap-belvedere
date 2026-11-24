<script lang="ts">
  import { onMount } from 'svelte';
  import { defaultPoolSize, EnginePool } from '../../lib/engine/pool';
  import type { PredictiveResult, SamplerResult } from '../../lib/engine/protocol';
  import {
    PRODUCTION_SAMPLER,
    loadInferCase,
    wasmBase,
    type InferCase,
  } from '../../lib/infer/case';
  import CornerPlot from '../../lib/infer/CornerPlot.svelte';
  import DiagnosticsBadge from '../../lib/infer/DiagnosticsBadge.svelte';
  import EnergyTrace from '../../lib/infer/EnergyTrace.svelte';
  import PredictivePlot from '../../lib/infer/PredictivePlot.svelte';
  import { fixed } from '../../lib/infer/stats';
  import StatusBar from '../../lib/ui/StatusBar.svelte';

  type Phase = 'loading' | 'idle' | 'running' | 'done' | 'error';

  let phase = $state<Phase>('loading');
  let caseData = $state<InferCase | null>(null);
  let result = $state<SamplerResult | null>(null);
  let predictive = $state<PredictiveResult | null>(null);
  let error = $state<string | null>(null);
  let predictiveError = $state<string | null>(null);
  let elapsed = $state(0);
  let daylight = $state(false);
  let core = $state('starting');
  let backend = $state('idle');
  let pool: EnginePool | null = null;
  let ticker: ReturnType<typeof setInterval> | undefined;
  let runToken = 0;
  let mounted = false;

  const commit = typeof __COMMIT_HASH__ === 'string' ? __COMMIT_HASH__ : 'dev';
  const theme = $derived(daylight ? 'daylight' : 'darkroom');
  const elapsedLabel = $derived((elapsed / 1_000).toFixed(1));
  const seedHex = PRODUCTION_SAMPLER.seed.toString(16);

  function applyTheme() {
    document.documentElement.dataset.theme = daylight ? 'daylight' : 'darkroom';
    localStorage.setItem('bgo-theme', daylight ? 'daylight' : 'darkroom');
  }

  function toggleTheme() {
    daylight = !daylight;
    applyTheme();
  }

  async function run() {
    if (!caseData || phase === 'running') return;
    runToken += 1;
    const token = runToken;
    phase = 'running';
    result = null;
    predictive = null;
    error = null;
    predictiveError = null;
    elapsed = 0;
    const started = performance.now();
    if (ticker) clearInterval(ticker);
    ticker = setInterval(() => {
      elapsed = performance.now() - started;
    }, 100);

    try {
      pool ??= new EnginePool(defaultPoolSize(), wasmBase());
      backend = 'wasm';
      await pool.prime();
      const ready = await pool.version();
      if (token !== runToken) return;
      if (!ready.ok) throw new Error(ready.error);
      core = String(ready.value);

      const observations = caseData.observations;
      const sampled = await pool.samplerParallel({
        model: caseData.model,
        temperatureK: new Float64Array(observations.temperature_k),
        voltageV: new Float64Array(observations.voltage_v),
        currentA: observations.current_a,
        sigmaV: observations.noise.likelihood_sigma_v,
        config: { ...PRODUCTION_SAMPLER },
      });
      if (token !== runToken) return;
      if (!sampled.ok) throw new Error(sampled.error);
      const sampledValue = sampled.value;
      if (typeof sampledValue !== 'object' || sampledValue === null || !('draws' in sampledValue)) {
        throw new Error('sampler returned no draws');
      }
      result = sampledValue as SamplerResult;

      const predicted = await pool.posteriorPredictive({
        model: caseData.model,
        temperatureK: new Float64Array(observations.temperature_k),
        currentA: observations.current_a,
        sigmaV: observations.noise.likelihood_sigma_v,
        draws: result.draws,
        parameterCount: result.parameterCount,
        drawsPerParameter: result.drawsPerParameter,
        seed: (PRODUCTION_SAMPLER.seed ^ 0x51ed_270b) >>> 0,
        lineCount: 500,
      });
      if (token !== runToken) return;
      if (!predicted.ok) {
        predictiveError = predicted.error;
      } else if (
        typeof predicted.value !== 'object' ||
        predicted.value === null ||
        !('lines' in predicted.value)
      ) {
        predictiveError = 'predictive output was empty';
      } else {
        predictive = predicted.value as unknown as PredictiveResult;
      }
      phase = 'done';
    } catch (cause) {
      if (token !== runToken) return;
      phase = 'error';
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      if (token === runToken) {
        elapsed = performance.now() - started;
        if (ticker) clearInterval(ticker);
        ticker = undefined;
      }
    }
  }

  function cancel() {
    runToken += 1;
    pool?.cancelAll();
    if (ticker) clearInterval(ticker);
    ticker = undefined;
    phase = caseData ? 'idle' : 'loading';
    error = null;
    predictiveError = null;
    elapsed = 0;
  }

  onMount(() => {
    daylight = localStorage.getItem('bgo-theme') === 'daylight';
    applyTheme();
    mounted = true;
    void loadInferCase()
      .then((value) => {
        if (!mounted) return;
        caseData = value;
        phase = 'idle';
      })
      .catch((cause) => {
        if (!mounted) return;
        phase = 'error';
        error = cause instanceof Error ? cause.message : String(cause);
      });

    return () => {
      mounted = false;
      runToken += 1;
      if (ticker) clearInterval(ticker);
      pool?.dispose();
      pool = null;
      backend = 'idle';
    };
  });
</script>

<svelte:head>
  <title>Infer — Band-Gap Observatory</title>
  <meta
    name="description"
    content="Run a synthetic M0 diode-thermometry posterior in the browser with fail-closed diagnostics, a corner plot, posterior predictive and energy-error trace."
  />
</svelte:head>

<div class="shell">
  <header class="top">
    <a class="mark" href="../" aria-label="Band-Gap Observatory home">
      <span class="aperture" aria-hidden="true"></span>
      <span>Band-Gap Observatory</span>
    </a>
    <nav aria-label="Sections">
      <a href="../">Observatory</a>
      <a href="../evidence/">Evidence</a>
      <a href="../ledger/">Ledger</a>
      {#if caseData}<a href="#run">Run</a>{/if}
      {#if result}<a href="#diagnostics">Diagnostics</a>{/if}
      {#if predictive}<a href="#s1">S1</a>{/if}
      <button class="theme" type="button" onclick={toggleTheme} aria-pressed={daylight}>
        {daylight ? 'Darkroom' : 'Daylight'}
      </button>
    </nav>
  </header>

  <main>
    <section class="hero" aria-labelledby="infer-title">
      <p class="eyebrow">Infer · synthetic M0 · browser-only</p>
      <h1 id="infer-title">The posterior,<br />not the point estimate.</h1>
      <p class="lede">
        Four MCLMC chains sample a synthetic diode-thermometry posterior in this tab. The
        diagnostics are computed in Rust, and the interface will not show a converged state when any
        gate fails.
      </p>
      <div class="chips" aria-label="Case labels">
        <span class="chip">Synthetic observations</span>
        <span class="chip">Report-summary anchor, not recovered data</span>
        <span class="chip">No server</span>
      </div>
    </section>

    {#if phase === 'loading'}
      <p class="loading panel" role="status">Loading the committed synthetic case…</p>
    {:else if caseData}
      <section class="provenance panel" aria-labelledby="provenance-title">
        <div class="section-head">
          <p class="eyebrow">Receipt</p>
          <h2 id="provenance-title">What this run uses</h2>
        </div>
        <div class="provenance-grid">
          <div>
            <span class="label">Case</span>
            <strong>{caseData.case_id}</strong>
            <p>
              {caseData.observations.temperature_k.length} points · {fixed(
                caseData.observations.temperature_k[0],
                2,
              )}–{fixed(
                caseData.observations.temperature_k[caseData.observations.temperature_k.length - 1],
                2,
              )} K
            </p>
          </div>
          <div>
            <span class="label">Current</span>
            <strong class="tabular">{fixed(caseData.observations.current_a * 1_000, 3)} mA</strong>
            <p>
              Likelihood σV {fixed(caseData.observations.noise.likelihood_sigma_v * 1_000, 3)} mV
            </p>
          </div>
          <div>
            <span class="label">Anchor</span>
            <strong class="tabular">{fixed(caseData.anchor.value_ev, 3)} eV</strong>
            <p>{caseData.anchor.source}</p>
          </div>
          <div>
            <span class="label">Reference hash</span>
            <strong class="hash" title={caseData.reference_sha256}>
              {caseData.reference_sha256.slice(0, 12)}…
            </strong>
            <p>Config {caseData.config_sha256.slice(0, 12)}… · seed {caseData.seed}</p>
          </div>
        </div>
        <p class="note">
          {caseData.anchor.note} The observations are generated, not the lost lab recordings. The posterior
          is compared with the committed NumPyro summary offline; this view does not claim a recovered
          measurement or a Table 1 reproduction.
        </p>
      </section>

      <section id="run" class="run-panel panel" aria-labelledby="run-title">
        <div class="section-head">
          <p class="eyebrow">Production settings</p>
          <h2 id="run-title">Run the sampler</h2>
        </div>
        <div class="settings">
          <p><strong>{PRODUCTION_SAMPLER.chains}</strong> chains</p>
          <p><strong>{PRODUCTION_SAMPLER.warmup.toLocaleString()}</strong> warmup</p>
          <p><strong>{PRODUCTION_SAMPLER.draws.toLocaleString()}</strong> draws</p>
          <p><strong>{PRODUCTION_SAMPLER.meanTrajectorySteps}</strong> mean steps</p>
          <p><strong>0x{seedHex}</strong> seed</p>
          <p><strong>{PRODUCTION_SAMPLER.targetAccept}</strong> target accept</p>
        </div>
        <div class="actions">
          <button class="primary" type="button" onclick={run} disabled={phase === 'running'}>
            {phase === 'running' ? 'Sampling…' : 'Run M0 inference'}
          </button>
          {#if phase === 'running'}
            <button class="secondary" type="button" onclick={cancel}>Cancel</button>
          {/if}
        </div>
        <p class="status-line" role="status" aria-live="polite">
          {#if phase === 'running'}
            Running four chains in parallel. The first worker compile happens before sampling.
          {:else if phase === 'done'}
            Completed in {elapsedLabel} s.
          {:else if phase === 'error'}
            The run did not complete.
          {:else}
            Nothing has been sampled yet. The run is opt-in.
          {/if}
          {#if phase === 'running'}<span class="tabular elapsed"> {elapsedLabel} s</span>{/if}
        </p>
        {#if error}
          <p class="error" role="alert">{error}</p>
        {/if}
      </section>

      {#if result}
        <section id="diagnostics" class="result-section" aria-labelledby="diagnostics-title">
          <div class="section-head">
            <p class="eyebrow">Fail-closed gate</p>
            <h2 id="diagnostics-title">Diagnostics</h2>
          </div>
          <DiagnosticsBadge {result} />
        </section>

        <section class="result-section" aria-labelledby="posterior-title">
          <div class="section-head">
            <p class="eyebrow">M0 posterior</p>
            <h2 id="posterior-title">Corner plot</h2>
          </div>
          <CornerPlot
            draws={result.draws}
            parameterNames={result.parameterNames}
            parameterCount={result.parameterCount}
            drawsPerParameter={result.drawsPerParameter}
            rhat={result.rhat}
            essBulk={result.essBulk}
            {theme}
          />
        </section>

        {#if predictive}
          <section id="s1" class="result-section" aria-labelledby="s1-title">
            <div class="section-head">
              <p class="eyebrow">S1 · simulation-labelled</p>
              <h2 id="s1-title">The Noisy Line</h2>
            </div>
            <p class="section-lede">
              Every thin curve is a posterior draw; the band is the posterior predictive. The points
              are synthetic observations, not the lost lab runs.
            </p>
            <PredictivePlot
              temperatures={caseData.observations.temperature_k}
              observed={caseData.observations.voltage_v}
              {predictive}
              {theme}
            />
          </section>
        {:else if predictiveError}
          <p class="error panel" role="alert">Predictive plot unavailable: {predictiveError}</p>
        {/if}

        {#if result.energyErrorTrace.length > 0}
          <section class="result-section" aria-labelledby="energy-title">
            <div class="section-head">
              <p class="eyebrow">MAMS energy error</p>
              <h2 id="energy-title">Oscilloscope</h2>
            </div>
            <p class="section-lede">
              This is the per-draw trace behind the energy-error gate. Spikes are visible; the
              verdict uses the mean, not a cleaned-up subset.
            </p>
            <EnergyTrace
              trace={result.energyErrorTrace}
              chainCount={result.chainCount}
              drawsPerChain={result.drawsPerChain}
              energyErrorMean={result.energyErrorMean}
              limit={PRODUCTION_SAMPLER.energyErrorLimit}
              divergences={result.divergences}
              {theme}
            />
          </section>
        {/if}
      {/if}
    {:else if error}
      <p class="error panel" role="alert">{error}</p>
    {/if}

    <section class="method" aria-labelledby="method-title">
      <p class="eyebrow">Read this before quoting a number</p>
      <h2 id="method-title">What the view does not claim</h2>
      <p>
        The original recordings were lost. This page uses the committed synthetic M0 case anchored
        to the report's quoted Si intercept; it is not a recovered measurement, a refit of the
        report, or a Table 1 reproduction. The reference comparison is a separate offline test
        against committed NumPyro summaries.
      </p>
    </section>
  </main>

  <StatusBar {backend} {core} phase="Phase 2" {commit} />
</div>

<style>
  .shell {
    display: flex;
    flex-direction: column;
    min-height: 100dvh;
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
    gap: 1.1rem;
    align-items: center;
    font-size: 0.82rem;
  }

  nav a {
    color: var(--ink-dim);
    text-decoration: none;
  }

  nav a:hover {
    color: var(--ink);
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

  .theme:hover {
    color: var(--ink);
    border-color: var(--accent);
  }

  main {
    flex: 1;
    width: var(--page);
    margin: 0 auto;
    padding: clamp(2rem, 6vw, 4.5rem) 0 4rem;
  }

  .hero h1 {
    font-size: var(--step-3);
    max-width: 16ch;
    margin: 0.6rem 0 1.2rem;
    text-wrap: balance;
  }

  .lede,
  .section-lede,
  .method p {
    max-width: 68ch;
    color: var(--ink-dim);
  }

  .lede {
    font-size: 1.1rem;
  }

  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.4rem;
  }

  .chip {
    padding: 0.3rem 0.65rem;
    border: var(--hairline);
    border-radius: 999px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--ink-dim);
  }

  section {
    margin-bottom: clamp(2.4rem, 5vw, 4.5rem);
  }

  .provenance,
  .run-panel {
    padding: clamp(1.1rem, 2.5vw, 1.6rem);
  }

  .section-head h2 {
    font-size: var(--step-2);
    margin: 0.25rem 0 0.7rem;
  }

  .provenance-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin: 1.2rem 0;
  }

  .provenance-grid > div {
    padding-right: 1rem;
    border-right: var(--hairline);
  }

  .provenance-grid > div:last-child {
    border-right: 0;
  }

  .label {
    display: block;
    margin-bottom: 0.35rem;
    font-family: var(--font-mono);
    font-size: 0.64rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-dim);
  }

  .provenance-grid strong {
    display: block;
    font-size: 0.98rem;
    overflow-wrap: anywhere;
  }

  .provenance-grid p {
    margin: 0.35rem 0 0;
    font-size: 0.75rem;
    color: var(--ink-dim);
  }

  .hash {
    font-family: var(--font-mono);
    font-size: 0.82rem !important;
  }

  .note {
    max-width: 78ch;
    margin: 0.5rem 0 0;
    font-size: 0.78rem;
    color: var(--ink-dim);
  }

  .settings {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.7rem;
    margin: 1rem 0 1.3rem;
  }

  .settings p {
    margin: 0;
    padding: 0.55rem;
    border: var(--hairline);
    border-radius: 8px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--ink-dim);
  }

  .settings strong {
    display: block;
    color: var(--ink);
    font-size: 0.86rem;
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
  }

  button {
    min-height: 44px;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    cursor: pointer;
  }

  .primary {
    color: var(--bg);
    background: var(--accent);
    border: 1px solid var(--accent);
  }

  .primary:disabled {
    cursor: progress;
    opacity: 0.65;
  }

  .secondary {
    color: var(--ink);
    background: transparent;
    border: var(--hairline);
  }

  .status-line {
    margin: 1rem 0 0;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--ink-dim);
  }

  .elapsed {
    color: var(--ink);
  }

  .error {
    max-width: 74ch;
    margin: 0.8rem 0 0;
    padding: 0.8rem;
    border: 1px solid oklch(0.72 0.15 35);
    border-radius: 8px;
    color: var(--ink);
    font-family: var(--font-mono);
    font-size: 0.76rem;
  }

  .result-section {
    margin-top: 2.4rem;
  }

  .result-section h2 {
    font-size: var(--step-2);
    margin: 0.25rem 0 0.6rem;
  }

  .loading {
    padding: 1rem;
    color: var(--ink-dim);
  }

  .method {
    border-top: var(--hairline);
    padding-top: 2rem;
  }

  .method h2 {
    font-size: var(--step-2);
    margin: 0.25rem 0 0.8rem;
  }

  @media (max-width: 900px) {
    .provenance-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .provenance-grid > div {
      border-right: 0;
    }

    .settings {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
  }

  @media (max-width: 600px) {
    .settings {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
