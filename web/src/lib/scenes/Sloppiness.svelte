<script lang="ts">
  import type { FisherResult } from '../engine/protocol';

  interface Props {
    fisher: FisherResult | null;
    low: number;
    high: number;
    min: number;
    max: number;
    error: string | null;
    onrange: (low: number, high: number) => void;
  }

  let { fisher, low, high, min, max, error, onrange }: Props = $props();

  const labels = $derived(
    (fisher?.parameterNames ?? []).map((name) =>
      name === 'eg_ev' ? 'E_g' : name === 'ln_b' ? 'ln B' : name === 'eta' ? 'η' : name,
    ),
  );

  const geometry = $derived.by(() => {
    if (!fisher || fisher.ellipse2.length < 3) return null;
    const major = fisher.ellipse2[0];
    const minor = fisher.ellipse2[1];
    if (!(major > 0) || !(minor > 0)) return null;
    const scale = Math.min(440 / (2.8 * major), 260 / (2.8 * minor));
    return {
      cx: 280,
      cy: 158,
      major1: fisher.ellipse1[0] * scale,
      minor1: fisher.ellipse1[1] * scale,
      major2: major * scale,
      minor2: minor * scale,
      angle: (fisher.ellipse2[2] * 180) / Math.PI,
      major,
      minor,
    };
  });

  function parameterSigma(index: number): number {
    if (!fisher) return Number.NaN;
    const n = fisher.parameterCount;
    return Math.sqrt(fisher.covariance[index * n + index]);
  }

  function direction(index: number): string {
    if (!fisher) return '';
    const n = fisher.parameterCount;
    return Array.from(fisher.eigenvectors.subarray(index * n, (index + 1) * n))
      .map((value, parameter) => `${value >= 0 ? '+' : ''}${value.toFixed(2)} ${labels[parameter]}`)
      .join('  ');
  }

  function sci(value: number): string {
    return Number.isFinite(value) ? value.toExponential(2) : '—';
  }

  function ratio(value: number): string {
    return Number.isFinite(value) ? value.toExponential(2) : '—';
  }

  const summary = $derived.by(() => {
    if (!fisher || labels.length === 0) return 'Fisher information is not available yet.';
    const stiffest = fisher.stiffestIndex;
    const sloppiest = fisher.sloppiestIndex;
    return (
      `At ${low.toFixed(0)}–${high.toFixed(0)} K the stiffest combination is ` +
      `${direction(stiffest)}; the sloppiest is ${direction(sloppiest)}. ` +
      `The Fisher condition number is ${ratio(fisher.conditionNumber)}, so the stiffest ` +
      `direction has that much smaller variance than the sloppiest. This is a local ` +
      `Gaussian approximation to a synthetic M1 posterior.`
    );
  });

  function setLow(event: Event) {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    onrange(Math.min(value, high - 5), high);
  }

  function setHigh(event: Event) {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    onrange(low, Math.max(value, low + 5));
  }

  const tableRows = $derived(
    (fisher?.parameterNames ?? []).map((name, index) => ({
      name: labels[index] ?? name,
      sigma: parameterSigma(index),
    })),
  );
</script>

<figure class="sloppiness" data-testid="sloppiness">
  <div class="range-controls">
    <div>
      <label for="sloppiness-low">Lower temperature</label>
      <input id="sloppiness-low" type="range" {min} {max} step="5" value={low} oninput={setLow} />
      <span class="tabular">{low.toFixed(0)} K</span>
    </div>
    <div>
      <label for="sloppiness-high">Upper temperature</label>
      <input
        id="sloppiness-high"
        type="range"
        {min}
        {max}
        step="5"
        value={high}
        oninput={setHigh}
      />
      <span class="tabular">{high.toFixed(0)} K</span>
    </div>
    <button type="button" onclick={() => onrange(min, max)}>Full sweep</button>
  </div>

  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}

  {#if fisher && geometry}
    <div class="visual">
      <svg
        viewBox="0 0 560 320"
        role="img"
        aria-label={`Fisher covariance ellipse for ${labels.join(', ')} over ${low.toFixed(
          0,
        )} to ${high.toFixed(0)} kelvin. The 2-sigma major axis is ${sci(
          geometry.major,
        )} in parameter units; condition number ${ratio(fisher.conditionNumber)}.`}
      >
        <line x1="40" y1={geometry.cy} x2="520" y2={geometry.cy} class="axis" />
        <line x1={geometry.cx} y1="18" x2={geometry.cx} y2="298" class="axis" />
        <ellipse
          cx={geometry.cx}
          cy={geometry.cy}
          rx={geometry.major2}
          ry={geometry.minor2}
          transform={`rotate(${geometry.angle} ${geometry.cx} ${geometry.cy})`}
          class="outer"
        />
        <ellipse
          cx={geometry.cx}
          cy={geometry.cy}
          rx={geometry.major1}
          ry={geometry.minor1}
          transform={`rotate(${geometry.angle} ${geometry.cx} ${geometry.cy})`}
          class="inner"
        />
        <circle cx={geometry.cx} cy={geometry.cy} r="2.5" class="centre" />
        <text x="40" y="308" class="label">{labels[0] ?? 'parameter 1'}</text>
        <text x="446" y="308" class="label">{labels[1] ?? 'parameter 2'}</text>
        <text x="22" y="28" class="label">2σ / 1σ covariance</text>
      </svg>
      <p class="plot-note">
        Auto-fit to the current range; the eigen-table below carries the absolute 1σ scale.
      </p>
    </div>

    <p class="summary" role="status" aria-live="polite">{summary}</p>

    <div class="table-grid">
      <div class="table-wrap">
        <table>
          <caption>Marginal Fisher 1σ uncertainty</caption>
          <thead>
            <tr><th scope="col">Parameter</th><th scope="col">1σ</th></tr>
          </thead>
          <tbody>
            {#each tableRows as row (row.name)}
              <tr><th scope="row">{row.name}</th><td class="tabular">{sci(row.sigma)}</td></tr>
            {/each}
          </tbody>
        </table>
      </div>

      <div class="table-wrap">
        <table>
          <caption>Fisher spectrum, stiffest first</caption>
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">Eigenvalue</th>
              <th scope="col">1σ length</th>
              <th scope="col">Direction</th>
            </tr>
          </thead>
          <tbody>
            {#each Array.from(fisher.eigenvalues) as eigenvalue, index (index)}
              <tr>
                <th scope="row">{index + 1}</th>
                <td class="tabular">{sci(eigenvalue)}</td>
                <td class="tabular">{eigenvalue > 0 ? sci(1 / Math.sqrt(eigenvalue)) : '—'}</td>
                <td class="direction">{direction(index)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </div>

    <figcaption>
      Computed in Rust from the M1 forward-model Jacobian over the selected temperatures. The input
      is a labelled synthetic case. The local Fisher inverse is not assumed to equal the full
      posterior when the eta prior or parameter bounds dominate; the M0 validation uses the
      near-Gaussian committed reference instead.
    </figcaption>
  {:else}
    <p class="placeholder">
      Waiting for the Rust Fisher report. The controls recompute the spectrum from the selected
      synthetic temperature window.
    </p>
  {/if}
</figure>

<style>
  .sloppiness {
    margin: 0;
    padding: 1.25rem;
    background: var(--surface);
    border: var(--hairline);
    border-radius: var(--radius);
  }

  .range-controls {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr)) auto;
    gap: 1rem;
    align-items: end;
    margin-bottom: 1rem;
  }

  .range-controls > div {
    display: grid;
    gap: 0.35rem;
  }

  label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-dim);
  }

  input[type='range'] {
    width: 100%;
    min-height: 44px;
    accent-color: var(--accent);
  }

  button {
    min-height: 44px;
    padding: 0.55rem 0.85rem;
    color: var(--ink);
    background: transparent;
    border: var(--hairline);
    border-radius: 8px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    cursor: pointer;
  }

  .visual {
    display: grid;
    justify-items: center;
  }

  svg {
    width: 100%;
    max-width: 620px;
    height: auto;
  }

  .axis {
    stroke: var(--hair);
    stroke-width: 1;
  }

  .outer {
    fill: color-mix(in oklch, var(--accent), transparent 86%);
    stroke: var(--accent);
    stroke-width: 1.2;
    opacity: 0.65;
  }

  .inner {
    fill: color-mix(in oklch, var(--accent), transparent 62%);
    stroke: var(--accent);
    stroke-width: 1.6;
  }

  .centre {
    fill: var(--ink);
  }

  .label {
    fill: var(--ink-dim);
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .plot-note,
  .placeholder,
  figcaption {
    color: var(--ink-dim);
    font-size: 0.76rem;
  }

  .plot-note {
    margin: 0;
  }

  .summary {
    max-width: 82ch;
    margin: 0.9rem 0;
    color: var(--ink-dim);
    font-size: 0.82rem;
  }

  .table-grid {
    display: grid;
    grid-template-columns: minmax(200px, 0.6fr) minmax(0, 1.4fr);
    gap: 1rem;
  }

  .table-wrap {
    overflow: auto;
    max-height: 250px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.7rem;
  }

  caption {
    text-align: left;
    padding-bottom: 0.4rem;
    color: var(--ink-dim);
    font-size: 0.68rem;
  }

  th,
  td {
    padding: 0.35rem 0.4rem;
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
    position: sticky;
    top: 0;
    background: var(--surface);
  }

  .direction {
    max-width: 260px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .error {
    margin: 0.8rem 0;
    padding: 0.8rem;
    border: 1px solid oklch(0.72 0.15 35);
    border-radius: 8px;
    color: var(--ink);
    font-family: var(--font-mono);
    font-size: 0.76rem;
  }

  figcaption {
    margin-top: 0.9rem;
    max-width: 82ch;
  }

  @media (max-width: 760px) {
    .range-controls,
    .table-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
