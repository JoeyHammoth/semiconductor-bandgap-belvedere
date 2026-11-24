<script lang="ts">
  import { onMount } from 'svelte';
  import StatusBar from '../../lib/ui/StatusBar.svelte';

  interface PriorSensitivity {
    weak_log_evidence: number;
    tight_log_evidence: number;
    delta_log_evidence: number;
    flag: boolean;
    description: string;
  }

  interface LedgerModel {
    id: string;
    prior_variant: string;
    parameter_count: number;
    log_evidence: number;
    log_evidence_error: number;
    information: number;
    elpd_loo: number;
    elpd_loo_se: number;
    p_loo: number;
    pareto_k_max: number;
    loo_reliable: boolean;
    nuts_divergences: number;
    prior_sensitivity?: PriorSensitivity;
  }

  interface LedgerDocument {
    schema_version: number;
    synthetic: boolean;
    generated_by: string;
    source_reference: string;
    note: string;
    loo_method: string;
    dataset: {
      case_id: string;
      model: string;
      seed: number;
      points: number;
      temperature_start_k: number;
      temperature_stop_k: number;
      current_a: number;
      likelihood_sigma_v: number;
    };
    settings: {
      engine: string;
      arviz: string;
      nlive: number;
      dlogz: number;
      posterior_draws: number;
      seed: number;
    };
    models: LedgerModel[];
    comparison: {
      baseline: string;
      delta_log_evidence: number;
      log10_bayes_factor: number;
      bayes_factor: number | null;
      elpd_difference: number;
      elpd_difference_se: number;
      direction: string;
      strength: string;
    };
    verdict: string;
  }

  let ledger = $state<LedgerDocument | null>(null);
  let error = $state<string | null>(null);
  let daylight = $state(false);

  const commit = typeof __COMMIT_HASH__ === 'string' ? __COMMIT_HASH__ : 'dev';
  const backend = 'reference';
  const core = $derived(ledger ? ledger.settings.engine : 'loading');

  function applyTheme() {
    document.documentElement.dataset.theme = daylight ? 'daylight' : 'darkroom';
    localStorage.setItem('bgo-theme', daylight ? 'daylight' : 'darkroom');
  }

  function toggleTheme() {
    daylight = !daylight;
    applyTheme();
  }

  function sci(value: number, digits = 3): string {
    return Number.isFinite(value) ? value.toExponential(digits) : '—';
  }

  function decimal(value: number, digits = 2): string {
    return Number.isFinite(value) ? value.toFixed(digits) : '—';
  }

  function log10Bayes(value: number): string {
    return Number.isFinite(value) ? value.toFixed(2) : '—';
  }

  function bayesFactor(value: number | null, log10Value: number): string {
    if (value !== null && Number.isFinite(value)) {
      return value >= 1_000 ? value.toExponential(2) : value.toFixed(2);
    }
    return `10^${log10Bayes(log10Value)}`;
  }

  onMount(() => {
    daylight = localStorage.getItem('bgo-theme') === 'daylight';
    applyTheme();
    void (async () => {
      try {
        const response = await fetch(new URL('../data/evidence-ledger.json', window.location.href));
        if (!response.ok) throw new Error(`ledger fetch failed: ${response.status}`);
        const value = (await response.json()) as LedgerDocument;
        if (value.schema_version !== 1 || value.synthetic !== true) {
          throw new Error('ledger failed its synthetic-data contract');
        }
        ledger = value;
      } catch (cause) {
        error = cause instanceof Error ? cause.message : String(cause);
      }
    })();
  });
</script>

<svelte:head>
  <title>Evidence Ledger — Band-Gap Observatory</title>
  <meta
    name="description"
    content="A synthetic model-comparison ledger with nested-sampling evidence, Bayes factors, PSIS-LOO and prior-sensitivity flags."
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
      <a href="../evidence/">Evidence</a>
      <button class="theme" type="button" onclick={toggleTheme} aria-pressed={daylight}>
        {daylight ? 'Darkroom' : 'Daylight'}
      </button>
    </nav>
  </header>

  <main>
    <section class="hero" aria-labelledby="ledger-title">
      <p class="eyebrow">S10 · model comparison · committed reference</p>
      <h1 id="ledger-title">The evidence<br />ledger.</h1>
      <p class="lede">
        Nested-sampling evidence, Bayes factors, PSIS-LOO and a computed prior-sensitivity flag for
        one synthetic comparison. The table is the accessible form of the scene; the prose only
        repeats what the numbers support.
      </p>
      <div class="chips" aria-label="Provenance labels">
        <span class="chip">Synthetic M1 observations</span>
        <span class="chip">M0 deliberately misspecified</span>
        <span class="chip">No measured-data claim</span>
      </div>
    </section>

    {#if error}
      <p class="error panel" role="alert">{error}</p>
    {:else if ledger}
      <section class="provenance panel" aria-labelledby="provenance-title">
        <div class="section-head">
          <p class="eyebrow">Receipt</p>
          <h2 id="provenance-title">What this ledger compares</h2>
        </div>
        <div class="provenance-grid">
          <div>
            <span class="label">Case</span>
            <strong>{ledger.dataset.case_id}</strong>
            <p>
              {ledger.dataset.points} synthetic points · {decimal(
                ledger.dataset.temperature_start_k,
                2,
              )}–{decimal(ledger.dataset.temperature_stop_k, 2)} K
            </p>
          </div>
          <div>
            <span class="label">Data-generating model</span>
            <strong>{ledger.dataset.model}</strong>
            <p>Seed {ledger.dataset.seed} · σV {sci(ledger.dataset.likelihood_sigma_v)} V</p>
          </div>
          <div>
            <span class="label">Evidence engine</span>
            <strong>{ledger.settings.engine}</strong>
            <p>{ledger.settings.nlive} live points · dlogz {ledger.settings.dlogz}</p>
          </div>
          <div>
            <span class="label">Source reference</span>
            <strong class="hash">{ledger.source_reference}</strong>
            <p>{ledger.settings.posterior_draws} posterior draws for PSIS-LOO</p>
            <p>{ledger.loo_method}</p>
          </div>
        </div>
        <p class="note">{ledger.note} Generated by <code>{ledger.generated_by}</code>.</p>
      </section>

      <section class="verdict panel" aria-labelledby="verdict-title">
        <p class="eyebrow">Plain-language verdict</p>
        <h2 id="verdict-title">
          {ledger.comparison.strength} preference for {ledger.comparison.direction}
        </h2>
        <p>{ledger.verdict}</p>
      </section>

      <section aria-labelledby="models-title">
        <div class="section-head">
          <p class="eyebrow">Evidence and predictive score</p>
          <h2 id="models-title">The comparison table</h2>
        </div>
        <div class="table-wrap">
          <table>
            <caption>All values from the committed synthetic ledger reference</caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Params</th>
                <th scope="col">ln Z</th>
                <th scope="col">Error</th>
                <th scope="col">H</th>
                <th scope="col">ELPD<sub>LOO</sub></th>
                <th scope="col">SE</th>
                <th scope="col">p<sub>LOO</sub></th>
                <th scope="col">max k</th>
                <th scope="col">NUTS div.</th>
                <th scope="col">LOO reliable</th>
              </tr>
            </thead>
            <tbody>
              {#each ledger.models as model (model.id)}
                <tr>
                  <th scope="row">{model.id}</th>
                  <td class="tabular">{model.parameter_count}</td>
                  <td class="tabular">{decimal(model.log_evidence, 3)}</td>
                  <td class="tabular">{decimal(model.log_evidence_error, 3)}</td>
                  <td class="tabular">{decimal(model.information, 2)}</td>
                  <td class="tabular">{decimal(model.elpd_loo, 1)}</td>
                  <td class="tabular">{decimal(model.elpd_loo_se, 1)}</td>
                  <td class="tabular">{decimal(model.p_loo, 1)}</td>
                  <td class="tabular">{decimal(model.pareto_k_max, 2)}</td>
                  <td class="tabular">{model.nuts_divergences}</td>
                  <td>{model.loo_reliable ? 'yes' : 'no'}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
        <p class="table-note">
          Pareto-k above 0.7 or a nonzero NUTS divergence count makes the magnitude of that model's
          PSIS-LOO score unreliable. The table reports those diagnostics rather than hiding them.
        </p>
      </section>

      <section aria-labelledby="bf-title">
        <div class="section-head">
          <p class="eyebrow">Relative comparison</p>
          <h2 id="bf-title">Bayes factor and LOO difference</h2>
        </div>
        <div class="comparison-grid">
          <article class="panel metric">
            <span class="label">Δ ln Z vs M0</span>
            <strong class="tabular">{decimal(ledger.comparison.delta_log_evidence, 1)}</strong>
            <p>Positive favours M1, the data-generating model.</p>
          </article>
          <article class="panel metric">
            <span class="label">log<sub>10</sub> Bayes factor</span>
            <strong class="tabular">{log10Bayes(ledger.comparison.log10_bayes_factor)}</strong>
            <p>
              BF<sub>10</sub> = {bayesFactor(
                ledger.comparison.bayes_factor,
                ledger.comparison.log10_bayes_factor,
              )} in favour of {ledger.comparison.direction}.
            </p>
          </article>
          <article class="panel metric">
            <span class="label">Δ ELPD<sub>LOO</sub> ± SE</span>
            <strong class="tabular">
              {decimal(ledger.comparison.elpd_difference, 1)} ± {decimal(
                ledger.comparison.elpd_difference_se,
                1,
              )}
            </strong>
            <p>Positive favours M1. Check the Pareto-k column before quoting the size.</p>
          </article>
        </div>
      </section>

      <section aria-labelledby="prior-title">
        <div class="section-head">
          <p class="eyebrow">Computed, not assumed</p>
          <h2 id="prior-title">Prior sensitivity</h2>
        </div>
        {#each ledger.models as model (model.id)}
          {#if model.prior_sensitivity}
            <div class="prior panel">
              <div class="prior-head">
                <h3>{model.id} · eta prior swap</h3>
                <span class="flag" class:flagged={model.prior_sensitivity.flag}>
                  {model.prior_sensitivity.flag ? 'prior-sensitive' : 'stable under this swap'}
                </span>
              </div>
              <p>{model.prior_sensitivity.description}.</p>
              <div class="prior-values">
                <span>
                  weak: <strong class="tabular">
                    {decimal(model.prior_sensitivity.weak_log_evidence, 3)}
                  </strong>
                </span>
                <span>
                  tight: <strong class="tabular">
                    {decimal(model.prior_sensitivity.tight_log_evidence, 3)}
                  </strong>
                </span>
                <span>
                  Δ: <strong class="tabular">
                    {decimal(model.prior_sensitivity.delta_log_evidence, 3)}
                  </strong>
                </span>
              </div>
            </div>
          {/if}
        {/each}
      </section>

      <section class="method" aria-labelledby="method-title">
        <p class="eyebrow">Read this before quoting a number</p>
        <h2 id="method-title">What the ledger does not claim</h2>
        <p>
          These are synthetic observations generated from the committed M1 case. M0 is deliberately
          misspecified; a large Bayes factor here does not tell us which model the lost lab runs
          support. The evidence is conditional on the stated priors, the likelihood width and the
          synthetic design, and the prior-sensitivity flag is part of the result.
        </p>
      </section>
    {:else}
      <p class="loading panel" role="status">Loading the committed evidence ledger…</p>
    {/if}
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
    margin-bottom: clamp(2.4rem, 5vw, 4rem);
  }

  .provenance,
  .verdict {
    padding: clamp(1.1rem, 2.5vw, 1.6rem);
  }

  .section-head h2,
  .verdict h2,
  .method h2 {
    font-size: var(--step-2);
    margin: 0.25rem 0 0.7rem;
  }

  .verdict h2 {
    font-size: var(--step-1);
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
    font-size: 0.94rem;
    overflow-wrap: anywhere;
  }

  .provenance-grid p,
  .note,
  .table-note,
  .metric p,
  .prior p {
    color: var(--ink-dim);
    font-size: 0.76rem;
  }

  .provenance-grid p {
    margin: 0.35rem 0 0;
  }

  .note {
    max-width: 84ch;
    margin: 0.5rem 0 0;
  }

  .hash {
    font-family: var(--font-mono);
    font-size: 0.8rem !important;
  }

  .verdict {
    border-left: 3px solid var(--accent);
  }

  .verdict p {
    max-width: 86ch;
    margin: 0;
    color: var(--ink);
  }

  .table-wrap {
    overflow: auto;
    border: var(--hairline);
    border-radius: var(--radius);
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.72rem;
  }

  caption {
    text-align: left;
    padding: 0.65rem;
    color: var(--ink-dim);
    font-size: 0.7rem;
  }

  th,
  td {
    padding: 0.5rem 0.55rem;
    border-bottom: var(--hairline);
    text-align: right;
    white-space: nowrap;
  }

  th:first-child,
  td:first-child {
    text-align: left;
  }

  thead th {
    color: var(--ink-dim);
    font-weight: 500;
    background: var(--surface);
  }

  .table-note {
    max-width: 82ch;
    margin: 0.6rem 0 0;
  }

  .comparison-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
  }

  .metric {
    padding: 1.1rem;
  }

  .metric strong {
    display: block;
    margin: 0.2rem 0 0.5rem;
    font-size: 1.4rem;
  }

  .metric p {
    margin: 0;
  }

  .prior {
    padding: 1.1rem;
  }

  .prior-head {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
    justify-content: space-between;
    align-items: center;
  }

  .prior h3 {
    font-family: var(--font-ui);
    font-size: 1rem;
    font-weight: 500;
  }

  .flag {
    padding: 0.25rem 0.6rem;
    border: var(--hairline);
    border-radius: 999px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--ink-dim);
  }

  .flag.flagged {
    border-color: oklch(0.72 0.15 35);
    color: var(--ink);
  }

  .prior-values {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-top: 0.6rem;
    font-family: var(--font-mono);
    font-size: 0.76rem;
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

  .loading {
    padding: 1rem;
    color: var(--ink-dim);
  }

  .method {
    border-top: var(--hairline);
    padding-top: 2rem;
  }

  @media (max-width: 900px) {
    .provenance-grid,
    .comparison-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .provenance-grid > div {
      border-right: 0;
    }
  }

  @media (max-width: 600px) {
    .provenance-grid,
    .comparison-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
