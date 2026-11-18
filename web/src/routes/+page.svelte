<script lang="ts">
  import { onMount } from 'svelte';
  import { EnginePool, defaultPoolSize } from '../lib/engine/pool';
  import HeroGap from '../lib/scenes/HeroGap.svelte';
  import StatusBar from '../lib/ui/StatusBar.svelte';
  import { DEFAULT_STATE, decodeState, encodeState } from '../lib/state/permalink';

  // Si literature Varshni set, resolved by the Phase 0 physics audit and
  // labelled as literature in the UI; never presented as a measured value.
  const SI = { eg0: 1.1557, alpha: 7.021e-4, beta: 1108 };

  let pool: EnginePool | null = null;
  let backend = $state<'detecting' | 'webgpu' | 'webgl2' | 'static'>('detecting');
  let daylight = $state(false);
  let core = $state('starting');
  let temperature = $state(DEFAULT_STATE.temperature);
  let gap = $state<number | null>(null);
  let table = $state<{ t: number; eg: number }[]>([]);
  let error = $state<string | null>(null);
  let available = $state(false);

  const commit = typeof __COMMIT_HASH__ === 'string' ? __COMMIT_HASH__ : 'dev';

  function applyTheme() {
    document.documentElement.dataset.theme = daylight ? 'daylight' : 'darkroom';
    localStorage.setItem('bgo-theme', daylight ? 'daylight' : 'darkroom');
  }

  function toggleTheme() {
    daylight = !daylight;
    applyTheme();
  }

  async function readGap(t: number) {
    if (!pool) return;
    const result = await pool.varshni(SI.eg0, SI.alpha, SI.beta, t);
    if (result.ok && typeof result.value === 'number') {
      gap = result.value;
      error = null;
    } else if (!result.ok) {
      error = result.error;
    }
  }

  onMount(() => {
    daylight = localStorage.getItem('bgo-theme') === 'daylight';
    applyTheme();

    const hash = window.location.hash.slice(1);
    if (hash) temperature = decodeState(hash).temperature;

    // Workers resolve relative imports against their own chunk URL, so the
    // base must be absolute. The glue is generated, so it is not part of the
    // typed asset manifest; on this root route the document base is the app
    // base (including any configured subpath).
    const glue = new URL('wasm/bgo_wasm.js', document.baseURI);
    const basePath = new URL('../', glue).href.replace(/\/$/, '');
    pool = new EnginePool(defaultPoolSize(), basePath);
    void (async () => {
      const ready = await pool!.version();
      if (!ready.ok) {
        available = false;
        core = 'unavailable';
        error = `core not built: ${ready.error}`;
        return;
      }
      available = true;
      core = String(ready.value);
      const temps = [250, 300, 350];
      const values = await Promise.all(
        temps.map((t) => pool!.varshni(SI.eg0, SI.alpha, SI.beta, t)),
      );
      table = temps.map((t, i) => ({
        t,
        eg: values[i].ok && typeof values[i].value === 'number' ? (values[i].value as number) : NaN,
      }));
      await readGap(temperature);
    })();

    return () => {
      pool?.dispose();
      pool = null;
    };
  });

  function onTemperature(event: Event) {
    temperature = Number((event.currentTarget as HTMLInputElement).value);
    void readGap(temperature);
    const state = { ...DEFAULT_STATE, temperature };
    history.replaceState(null, '', `#${encodeState(state)}`);
  }

  const display = $derived(gap === null ? '—' : gap.toFixed(3));
</script>

<svelte:head>
  <title>Band-Gap Observatory</title>
</svelte:head>

<div class="shell">
  <header class="top">
    <a class="mark" href="./" aria-label="Band-Gap Observatory home">
      <span class="aperture" aria-hidden="true"></span>
      <span>Band-Gap Observatory</span>
    </a>
    <nav aria-label="Sections">
      <a href="#question">Question</a>
      <a href="#method">Method</a>
      <a href="#honesty">Honesty</a>
      <a href="https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere#readme">Docs</a>
      <button class="theme" type="button" onclick={toggleTheme} aria-pressed={daylight}>
        {daylight ? 'Darkroom' : 'Daylight'}
      </button>
    </nav>
  </header>

  <main>
    <section class="hero" aria-labelledby="hero-title">
      <p class="eyebrow">Diode thermometry · PHYC30021 · static, zero-backend</p>
      <h1 id="hero-title">How wide is the gap,<br />and how sure are we?</h1>
      <p class="lede">
        A classic, imperfect measurement protocol: a diode's forward voltage against temperature.
        The lab fits a straight line and calls the intercept the band gap. This instrument asks
        which assumptions that line hides, and what the surviving report and simulations can
        actually support.
      </p>

      <div class="readout panel">
        <div class="readout-main">
          <span class="eyebrow">Varshni gap at <span class="tabular">{temperature.toFixed(0)}</span> K</span>
          <p class="value tabular" aria-live="polite">
            {display}<span class="unit">eV</span>
          </p>
          <p class="caveat">
            Silicon Varshni (1967) parameters, verified in the
            <a href="https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/blob/main/docs/physics-audit.md">physics audit</a>.
            Computed in Rust/WASM, not in this component.
          </p>
        </div>

        <div class="controls">
          <label for="temp">Temperature</label>
          <input
            id="temp"
            type="range"
            min="150"
            max="400"
            step="1"
            value={temperature}
            oninput={onTemperature}
            disabled={!available}
          />
          <div class="range-labels tabular" aria-hidden="true"><span>150 K</span><span>400 K</span></div>
          {#if error}
            <p class="error" role="status">{error}</p>
          {:else if !available}
            <p class="error" role="status">Starting the Rust core…</p>
          {/if}
        </div>
      </div>

      <HeroGap {gap} {temperature} onbackend={(value) => (backend = value)} />

      {#if table.length > 0}
        <table class="readout-table">
          <caption>Static equivalent of the readout above</caption>
          <thead>
            <tr><th scope="col">Temperature (K)</th><th scope="col">Gap (eV)</th></tr>
          </thead>
          <tbody>
            {#each table as row (row.t)}
              <tr><th scope="row">{row.t}</th><td class="tabular">{row.eg.toFixed(3)}</td></tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </section>

    <section id="question" aria-labelledby="q-title">
      <p class="eyebrow">01 — The question</p>
      <h2 id="q-title">The intercept is not the band gap</h2>
      <div class="prose">
        <p>
          Under the ideal assumptions — no series resistance, unit ideality, a prefactor with no
          temperature dependence — the forward voltage falls on a straight line whose intercept is
          <code>E<sub>g</sub>/q</code>. Relax any one of those and the intercept becomes a mixture of
          the gap, the prefactor and the ideality.
        </p>
        <p>
          Over a 60–100 K window around room temperature, those parameters are nearly collinear.
          That is <em>sloppiness</em>, and it is the honest headline: the data constrain a
          combination far better than they constrain the gap itself.
        </p>
      </div>
    </section>

    <section id="receipt" aria-labelledby="r-title">
      <p class="eyebrow">Receipt 01 — the intercept</p>
      <h2 id="r-title">What the straight line actually estimates</h2>
      <div class="prose">
        <p>
          The lab reports the y-intercept as the band gap. It is not: Varshni curvature alone puts
          the chord extrapolation 37–43 mV above the true 0 K gap over 300–383 K, and a
          temperature-dependent prefactor adds about 29 mV per unit η. The measured intercepts sit
          below even that chord.
        </p>
      </div>
      <table class="readout-table">
        <caption>Report summary intercepts against Varshni (1967) gaps</caption>
        <thead>
          <tr>
            <th scope="col">Diode</th>
            <th scope="col">Fitted intercept (eV)</th>
            <th scope="col">E<sub>g</sub>(0 K) (eV)</th>
            <th scope="col">E<sub>g</sub>(300 K) (eV)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Si (1N4007)</th>
            <td class="tabular">1.002 ± 0.011</td>
            <td class="tabular">1.1557</td>
            <td class="tabular">1.1108</td>
          </tr>
          <tr>
            <th scope="row">Ge (1N60P)</th>
            <td class="tabular">0.646 ± 0.007</td>
            <td class="tabular">0.7412</td>
            <td class="tabular">0.6607</td>
          </tr>
        </tbody>
      </table>
      <p class="caveat">
        The original per-run recordings no longer exist, so this table quotes the report's summary
        and cannot be refitted here. Uncertainties are trend-line spreads, not total uncertainties.
        LEDs are compared with the same die's room-temperature emission instead, because an alloy
        LED has no well-defined 0 K gap. Sources and arithmetic:
        <a href="https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/blob/main/docs/physics-audit.md">physics audit</a>.
      </p>
    </section>

    <section id="method" aria-labelledby="m-title">
      <p class="eyebrow">02 — Method</p>
      <h2 id="m-title">Everything runs in your browser</h2>
      <div class="cards">
        <article class="panel card">
          <h3>Inference</h3>
          <p>Microcanonical HMC, nested-sampling evidence, Stein particles and amortised posteriors — validated against NumPyro offline.</p>
        </article>
        <article class="panel card">
          <h3>Information</h3>
          <p>Fisher spectra and Cramér–Rao ellipsoids show which parameter combinations the data pin down, and which they do not.</p>
        </article>
        <article class="panel card">
          <h3>Design</h3>
          <p>Expected information gain ranks the next measurement, so the recommendation is a calculation rather than an opinion.</p>
        </article>
      </div>
    </section>

    <section id="honesty" aria-labelledby="h-title">
      <p class="eyebrow">03 — Honesty</p>
      <h2 id="h-title">What this is not</h2>
      <div class="prose">
        <p>
          Not new physics. Not a better band gap than the literature. The measurement is a classic
          teaching experiment, and the straight-line analysis is a known simplification. The
          contribution is the interrogation: which assumptions the protocol supports, how much
          simulations and the surviving summary actually constrain, and what to measure next.
        </p>
        <p>
          Every number traces to code, a cited source or a hash-pinned dataset, and a commit. Claims
          that are not verified are labelled, not hidden. If a sampler diagnostic fails, the site
          does not show a "converged" badge.
        </p>
      </div>
    </section>

    <section class="next" aria-labelledby="n-title">
      <h2 id="n-title">Status</h2>
      <p>
        Phase 0 — walking skeleton. The repository, CI, physics audit and data-provenance audit
        are complete. The original recordings no longer exist, so later scenes use labelled
        simulated and report-anchored data. Full plan in
        <a href="https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/blob/main/docs/roadmap.md">the roadmap</a>.
      </p>
    </section>
  </main>

  <StatusBar {backend} {core} phase="Phase 0" {commit} />
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
    font-size: 0.82rem;
  }

  nav a {
    color: var(--ink-dim);
    text-decoration: none;
  }

  nav a:hover {
    color: var(--ink);
  }

  main {
    flex: 1;
    width: var(--page);
    margin: 0 auto;
    padding: clamp(2rem, 6vw, 5rem) 0 4rem;
  }

  section {
    margin-bottom: clamp(3rem, 7vw, 6rem);
  }

  .hero h1 {
    font-size: var(--step-3);
    max-width: 18ch;
    margin: 0.6rem 0 1.2rem;
    text-wrap: balance;
  }

  .lede {
    max-width: 62ch;
    font-size: 1.1rem;
    color: var(--ink-dim);
  }

  .readout {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
    gap: 1.5rem;
    padding: clamp(1.2rem, 3vw, 2rem);
    margin: 2rem 0 1rem;
    border-top: 1px solid color-mix(in oklch, var(--accent), transparent 40%);
  }

  @media (max-width: 720px) {
    .readout {
      grid-template-columns: 1fr;
    }
  }

  .value {
    font-family: var(--font-display);
    font-size: clamp(3.4rem, 9vw, 6rem);
    line-height: 0.95;
    margin: 0.4rem 0 0.6rem;
    color: var(--accent);
    text-shadow: 0 0 40px color-mix(in oklch, var(--accent), transparent 70%);
  }

  .unit {
    font-family: var(--font-mono);
    font-size: 1rem;
    margin-left: 0.6rem;
    color: var(--ink-dim);
    letter-spacing: 0.08em;
  }

  .caveat,
  .error {
    font-size: 0.78rem;
    color: var(--ink-dim);
    max-width: 40ch;
  }

  .error {
    color: oklch(0.78 0.12 40);
  }

  .controls {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    justify-content: center;
  }

  .controls label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-dim);
  }

  input[type='range'] {
    width: 100%;
    accent-color: var(--accent);
    min-height: 44px;
  }

  .range-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: var(--ink-dim);
  }

  .readout-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    margin-top: 1rem;
  }

  .readout-table caption {
    text-align: left;
    color: var(--ink-dim);
    font-size: 0.75rem;
    padding-bottom: 0.5rem;
  }

  .readout-table th,
  .readout-table td {
    text-align: left;
    padding: 0.4rem 0.6rem;
    border-bottom: var(--hairline);
  }

  h2 {
    font-size: var(--step-2);
    margin: 0.4rem 0 1rem;
    max-width: 24ch;
  }

  .prose {
    max-width: 68ch;
    color: var(--ink-dim);
  }

  .prose code {
    color: var(--ink);
    font-size: 0.92em;
  }

  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1rem;
  }

  .card {
    padding: 1.25rem;
  }

  .card h3 {
    font-size: 1.4rem;
    margin-bottom: 0.5rem;
  }

  .card p {
    color: var(--ink-dim);
    font-size: 0.92rem;
    margin: 0;
  }

  .next {
    border-top: var(--hairline);
    padding-top: 2rem;
    color: var(--ink-dim);
  }

  .next p {
    max-width: 68ch;
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
</style>
