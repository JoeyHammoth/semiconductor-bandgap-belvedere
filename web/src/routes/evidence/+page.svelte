<script lang="ts">
  import { onMount } from 'svelte';
  import { wasmBase } from '../../lib/engine/base';
  import { EnginePool } from '../../lib/engine/pool';
  import type { FisherResult, NestedTraceResult } from '../../lib/engine/protocol';
  import Sloppiness from '../../lib/scenes/Sloppiness.svelte';
  import ShrinkingLight from '../../lib/scenes/ShrinkingLight.svelte';
  import StatusBar from '../../lib/ui/StatusBar.svelte';

  interface SloppinessCase {
    case: { case_id: string; model: string; seed: number; synthetic: boolean };
    observations: {
      current_a: number;
      noise: { likelihood_sigma_v: number };
      synthetic: boolean;
      temperature_k: number[];
      voltage_v: number[];
    };
    reference: { mean: Record<string, number> };
    truth: Record<string, number>;
  }

  const NESTED_SEED = 0x5eed;
  const NESTED_LIVE_POINTS = 128;

  let pool: EnginePool | null = null;
  let caseData = $state<SloppinessCase | null>(null);
  let fisher = $state<FisherResult | null>(null);
  let nested = $state<NestedTraceResult | null>(null);
  let low = $state(220);
  let high = $state(383.15);
  let min = $state(220);
  let max = $state(383.15);
  let fisherError = $state<string | null>(null);
  let nestedError = $state<string | null>(null);
  let loadError = $state<string | null>(null);
  let nestedRunning = $state(false);
  let fisherRunning = $state(false);
  let core = $state('starting');
  let backend = $state('detecting');
  let daylight = $state(false);
  let reducedMotion = $state(false);
  let mounted = false;
  let fisherToken = 0;
  let debounce: ReturnType<typeof setTimeout> | undefined;

  const commit = typeof __COMMIT_HASH__ === 'string' ? __COMMIT_HASH__ : 'dev';

  function applyTheme() {
    document.documentElement.dataset.theme = daylight ? 'daylight' : 'darkroom';
    localStorage.setItem('bgo-theme', daylight ? 'daylight' : 'darkroom');
  }

  function toggleTheme() {
    daylight = !daylight;
    applyTheme();
  }

  async function loadCase(): Promise<SloppinessCase> {
    const response = await fetch(new URL('../data/m1-sloppiness.json', window.location.href));
    if (!response.ok) throw new Error(`synthetic M1 case fetch failed: ${response.status}`);
    const value = (await response.json()) as SloppinessCase;
    if (
      value.case.synthetic !== true ||
      value.observations.synthetic !== true ||
      value.case.model !== 'M1'
    ) {
      throw new Error('synthetic M1 case failed validation');
    }
    return value;
  }

  async function computeFisher() {
    if (!caseData || !pool) return;
    const token = ++fisherToken;
    const temperatures = caseData.observations.temperature_k.filter(
      (value) => value >= low - 1e-9 && value <= high + 1e-9,
    );
    if (temperatures.length < 2) {
      fisherError = 'The selected window contains fewer than two observations.';
      fisher = null;
      return;
    }
    fisherRunning = true;
    fisherError = null;
    const mean = caseData.reference.mean;
    const result = await pool.fisherReport({
      model: 'M1',
      temperatureK: new Float64Array(temperatures),
      currentA: caseData.observations.current_a,
      sigmaV: caseData.observations.noise.likelihood_sigma_v,
      reported: new Float64Array([mean.eg_ev, mean.ln_b, mean.eta]),
    });
    if (token !== fisherToken) return;
    fisherRunning = false;
    if (!result.ok) {
      fisherError = result.error;
      fisher = null;
      return;
    }
    fisher = result.value as FisherResult;
  }

  function scheduleFisher() {
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => {
      void computeFisher();
    }, 40);
  }

  function handleRange(nextLow: number, nextHigh: number) {
    low = Math.max(min, Math.min(nextLow, nextHigh - 5));
    high = Math.min(max, Math.max(nextHigh, low + 5));
    scheduleFisher();
  }

  async function runNested() {
    if (!pool || nestedRunning) return;
    nestedRunning = true;
    nestedError = null;
    const result = await pool.nestedEvidence(NESTED_LIVE_POINTS, NESTED_SEED);
    nestedRunning = false;
    if (!result.ok) {
      nestedError = result.error;
      return;
    }
    nested = result.value as NestedTraceResult;
  }

  onMount(() => {
    daylight = localStorage.getItem('bgo-theme') === 'daylight';
    applyTheme();
    reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    mounted = true;
    pool = new EnginePool(1, wasmBase());
    void (async () => {
      try {
        const value = await loadCase();
        if (!mounted) return;
        caseData = value;
        min = value.observations.temperature_k[0];
        max = value.observations.temperature_k[value.observations.temperature_k.length - 1];
        low = min;
        high = max;

        await pool!.prime();
        const ready = await pool!.version();
        if (!mounted) return;
        if (!ready.ok) throw new Error(ready.error);
        core = String(ready.value);
        backend = 'wasm';
        await computeFisher();
      } catch (cause) {
        if (!mounted) return;
        backend = 'unavailable';
        core = 'unavailable';
        loadError = cause instanceof Error ? cause.message : String(cause);
      }
    })();

    return () => {
      mounted = false;
      fisherToken += 1;
      if (debounce) clearTimeout(debounce);
      pool?.dispose();
      pool = null;
    };
  });
</script>

<svelte:head>
  <title>Evidence — Band-Gap Observatory</title>
  <meta
    name="description"
    content="Nested-sampling live points and Fisher sloppiness computed in Rust/WASM from labelled synthetic cases."
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
      <a href="../infer/">Infer</a>
      <a href="#s2">S2</a>
      <a href="#s4">S4</a>
      <a href="../ledger/">Ledger</a>
      <button class="theme" type="button" onclick={toggleTheme} aria-pressed={daylight}>
        {daylight ? 'Darkroom' : 'Daylight'}
      </button>
    </nav>
  </header>

  <main>
    <section class="hero" aria-labelledby="evidence-title">
      <p class="eyebrow">Phase 2 · evidence and information geometry · browser-only</p>
      <h1 id="evidence-title">What the data<br />can pin down.</h1>
      <p class="lede">
        Nested sampling contracts a live set while the evidence accumulates. Fisher information
        turns the same synthetic model into a sloppy ellipsoid: a few parameter combinations are
        stiff, and the rest are nearly free. Both numbers are computed in Rust, not in the scene.
      </p>
      <div class="chips" aria-label="Provenance labels">
        <span class="chip">Synthetic cases</span>
        <span class="chip">No measured data</span>
        <span class="chip">Validated kernel</span>
      </div>
    </section>

    {#if loadError}
      <p class="error panel" role="alert">{loadError}</p>
    {/if}

    <section id="s2" aria-labelledby="s2-title">
      <div class="section-head">
        <p class="eyebrow">S2 · synthetic Gaussian calibration</p>
        <h2 id="s2-title">Shrinking Light</h2>
      </div>
      <p class="section-lede">
        The live set is a uniform-prior sample constrained above the current likelihood threshold.
        Scrub the event timeline to watch it contract; the final evidence is compared with the
        closed-form Gaussian value.
      </p>
      <ShrinkingLight
        trace={nested}
        theme={daylight ? 'daylight' : 'darkroom'}
        {reducedMotion}
        running={nestedRunning}
        error={nestedError}
        onrun={runNested}
      />
    </section>

    <section id="s4" aria-labelledby="s4-title">
      <div class="section-head">
        <p class="eyebrow">S4 · synthetic M1 · local Fisher geometry</p>
        <h2 id="s4-title">Sloppiness</h2>
      </div>
      <p class="section-lede">
        Narrow the temperature sweep and the covariance ellipse grows; widen it and the same
        parameters become better separated. The slider sends the selected observations back to Rust
        for a fresh Fisher report.
      </p>
      {#if fisherRunning}
        <p class="status-line" role="status">Recomputing Fisher information…</p>
      {/if}
      <Sloppiness {fisher} {low} {high} {min} {max} error={fisherError} onrange={handleRange} />
    </section>

    <section class="method" aria-labelledby="method-title">
      <p class="eyebrow">Read this before quoting a number</p>
      <h2 id="method-title">Local geometry, synthetic input</h2>
      <p>
        S2 uses a two-dimensional Gaussian fixture whose evidence is available in closed form; it is
        not the M0 posterior. S4 linearises the M1 forward model at the committed synthetic
        posterior mean. The Fisher inverse is a local Gaussian approximation, not a replacement for
        the full posterior when the eta prior or parameter bounds dominate. The M0
        Fisher-versus-posterior check is the committed near-Gaussian validation.
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
    max-width: 15ch;
    margin: 0.6rem 0 1.2rem;
    text-wrap: balance;
  }

  .lede,
  .section-lede,
  .method p {
    max-width: 70ch;
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
    margin-bottom: clamp(3rem, 7vw, 5rem);
  }

  .section-head h2,
  .method h2 {
    font-size: var(--step-2);
    margin: 0.25rem 0 0.7rem;
  }

  .status-line {
    margin: 0 0 0.8rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--ink-dim);
  }

  .error {
    max-width: 74ch;
    margin: 1rem 0;
    padding: 0.8rem;
    border: 1px solid oklch(0.72 0.15 35);
    border-radius: 8px;
    color: var(--ink);
    font-family: var(--font-mono);
    font-size: 0.76rem;
  }

  .method {
    border-top: var(--hairline);
    padding-top: 2rem;
  }
</style>
