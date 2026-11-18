<script lang="ts">
  interface Props {
    backend: string;
    core: string;
    phase: string;
    commit: string;
  }

  let { backend, core, phase, commit }: Props = $props();
</script>

<footer class="status" aria-label="Compute status">
  <span class="dot" aria-hidden="true"></span>
  <span>{phase}</span>
  <span class="sep" aria-hidden="true">·</span>
  <span>backend <strong>{backend}</strong></span>
  <span class="sep" aria-hidden="true">·</span>
  <span>core <strong>{core}</strong></span>
  <span class="sep" aria-hidden="true">·</span>
  <span class="hash" title="Commit">{commit.slice(0, 7)}</span>
</footer>

<style>
  .status {
    position: sticky;
    bottom: 0;
    z-index: 10;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 0.6rem;
    align-items: center;
    padding: 0.55rem clamp(0.75rem, 3vw, 2rem);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.02em;
    color: var(--ink-dim);
    background: color-mix(in oklch, var(--bg), black 12%);
    border-top: var(--hairline);
    backdrop-filter: blur(8px);
  }

  strong {
    font-weight: 500;
    color: var(--ink);
  }

  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent);
  }

  .sep {
    opacity: 0.4;
  }

  .hash {
    margin-left: auto;
  }
</style>
