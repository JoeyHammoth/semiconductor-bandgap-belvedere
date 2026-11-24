<script lang="ts">
  import type { SamplerResult } from '../engine/protocol';
  import { fixed } from './stats';

  interface Props {
    result: SamplerResult;
  }

  let { result }: Props = $props();
</script>

<section
  class="diagnostics panel"
  data-testid="diagnostics-badge"
  data-pass={result.passes}
  aria-live="polite"
  aria-label="Sampler diagnostics"
>
  <div class="headline">
    <span class="dot" aria-hidden="true"></span>
    <div>
      <p class="eyebrow">Diagnostics gate</p>
      <h2>{result.passes ? 'Converged' : 'Not converged'}</h2>
    </div>
  </div>

  <p class="verdict">
    {#if result.passes}
      Every gate passed. These draws can be read as a posterior.
    {:else}
      At least one gate failed. Do not treat these draws as a converged posterior.
    {/if}
  </p>

  <dl class="metrics">
    <div>
      <dt>R-hat max</dt>
      <dd class="tabular">{fixed(result.rhatMax, 5)}</dd>
      <span>&lt; 1.01</span>
    </div>
    <div>
      <dt>ESS min</dt>
      <dd class="tabular">{fixed(result.essBulkMin, 1)}</dd>
      <span>&gt; 400</span>
    </div>
    <div>
      <dt>Energy error</dt>
      <dd class="tabular">{fixed(result.energyErrorMean, 4)}</dd>
      <span>&lt; 0.1</span>
    </div>
    <div>
      <dt>Divergences</dt>
      <dd class="tabular">{result.divergences}</dd>
      <span>= 0</span>
    </div>
  </dl>

  <p class="rule">
    The badge is computed from the Rust diagnostics over all chains. The interface cannot
    override it.
  </p>
</section>

<style>
  .diagnostics {
    padding: 1.25rem;
    border-top: 2px solid var(--accent);
  }

  .diagnostics[data-pass='false'] {
    border-top-color: oklch(0.72 0.15 35);
  }

  .headline {
    display: flex;
    gap: 0.8rem;
    align-items: center;
  }

  .dot {
    width: 0.8rem;
    height: 0.8rem;
    flex: 0 0 auto;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 14px var(--accent);
  }

  [data-pass='false'] .dot {
    background: oklch(0.72 0.15 35);
    box-shadow: 0 0 14px oklch(0.72 0.15 35);
  }

  h2 {
    margin: 0.15rem 0 0;
    font-size: clamp(1.7rem, 3vw, 2.3rem);
  }

  .verdict {
    max-width: 48ch;
    margin: 0.9rem 0;
    color: var(--ink-dim);
  }

  .metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.7rem;
    margin: 1rem 0;
  }

  .metrics div {
    padding: 0.65rem;
    border: var(--hairline);
    border-radius: 8px;
  }

  dt,
  .metrics span,
  .rule {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--ink-dim);
  }

  dt {
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  dd {
    margin: 0.2rem 0;
    font-size: 1rem;
    color: var(--ink);
  }

  .rule {
    margin: 0;
    line-height: 1.5;
  }

  @media (max-width: 600px) {
    .metrics {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
