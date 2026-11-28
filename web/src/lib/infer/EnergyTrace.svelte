<script lang="ts">
  import { onMount } from 'svelte';
  import { fixed, mean } from './stats';

  interface Props {
    trace: Float64Array;
    chainCount: number;
    drawsPerChain: number;
    energyErrorMean: number;
    limit: number;
    divergences: number;
    theme: string;
  }

  let { trace, chainCount, drawsPerChain, energyErrorMean, limit, divergences, theme }: Props =
    $props();

  let canvas = $state<HTMLCanvasElement | null>(null);
  let ready = $state(false);

  const chains = $derived.by(() => {
    if (chainCount < 1 || drawsPerChain < 1) return [];
    return Array.from({ length: chainCount }, (_, index) => {
      const values = trace.subarray(index * drawsPerChain, (index + 1) * drawsPerChain);
      return {
        index: index + 1,
        mean: mean(values),
        max: values.length > 0 ? Math.max(...values) : Number.NaN,
      };
    });
  });

  function draw() {
    const element = canvas;
    if (!element || trace.length === 0 || chainCount < 1) return;
    const width = Math.max(320, Math.floor(element.clientWidth));
    const height = Math.max(150, Math.floor(element.clientHeight));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (element.width !== width * dpr) element.width = width * dpr;
    if (element.height !== height * dpr) element.height = height * dpr;

    const context = element.getContext('2d');
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);

    const styles = getComputedStyle(document.documentElement);
    const dim = styles.getPropertyValue('--ink-dim').trim() || 'CanvasText';
    const accent = styles.getPropertyValue('--accent').trim() || 'CanvasText';
    const hair = styles.getPropertyValue('--hair').trim() || 'CanvasText';
    const alert = styles.getPropertyValue('--ir').trim() || 'CanvasText';

    const logLimit = Math.log10(limit);
    let low = logLimit - 2;
    let high = logLimit + 1;
    for (const value of trace) {
      const logged = Math.log10(Math.max(value, 1e-9));
      if (logged < low) low = logged;
      if (logged > high) high = logged;
    }
    const plot = { x: 54, y: 12, width: width - 72, height: height - 38 };
    const xScale = (draw: number) => plot.x + (draw / Math.max(1, drawsPerChain - 1)) * plot.width;
    const yScale = (value: number) =>
      plot.y +
      plot.height -
      ((Math.log10(Math.max(value, 1e-9)) - low) / (high - low)) * plot.height;

    context.font = '10px "JetBrains Mono Variable", ui-monospace, monospace';
    context.textBaseline = 'middle';
    context.strokeStyle = hair;
    context.strokeRect(plot.x, plot.y, plot.width, plot.height);

    for (let exponent = Math.ceil(low); exponent <= Math.floor(high); exponent += 1) {
      const y = yScale(10 ** exponent);
      context.strokeStyle = hair;
      context.beginPath();
      context.moveTo(plot.x, y);
      context.lineTo(plot.x + plot.width, y);
      context.stroke();
      context.fillStyle = dim;
      context.textAlign = 'right';
      context.fillText(`1e${exponent}`, plot.x - 8, y);
    }

    const limitY = yScale(limit);
    context.strokeStyle = alert;
    context.setLineDash([5, 4]);
    context.beginPath();
    context.moveTo(plot.x, limitY);
    context.lineTo(plot.x + plot.width, limitY);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = dim;
    context.textAlign = 'left';
    context.fillText(`mean gate ${limit}`, plot.x + 6, limitY - 8);

    const dashes = [[], [4, 3], [2, 3], [8, 3, 2, 3]];
    for (let chain = 0; chain < chainCount; chain += 1) {
      const start = chain * drawsPerChain;
      context.strokeStyle = accent;
      context.lineWidth = chain === 0 ? 1.2 : 0.8;
      context.globalAlpha = chain === 0 ? 0.8 : 0.45;
      context.setLineDash(dashes[chain % dashes.length]);
      context.beginPath();
      for (let draw = 0; draw < drawsPerChain; draw += 1) {
        const x = xScale(draw);
        const y = yScale(trace[start + draw]);
        if (draw === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    }
    context.setLineDash([]);
    context.globalAlpha = 1;
    context.fillStyle = dim;
    context.textAlign = 'center';
    context.fillText('post-warmup draw', plot.x + plot.width / 2, height - 10);
  }

  onMount(() => {
    ready = true;
    const element = canvas;
    if (!element) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(element);
    draw();
    return () => observer.disconnect();
  });

  $effect(() => {
    theme;
    trace;
    chainCount;
    drawsPerChain;
    if (ready) draw();
  });
</script>

<figure class="energy" data-testid="energy-trace">
  <div
    class="canvas-wrap"
    role="img"
    aria-label={`Energy-error trace for ${chainCount} chains over ${drawsPerChain} post-warmup draws. Mean ${fixed(
      energyErrorMean,
      4,
    )}; gate ${limit}; ${divergences} divergences.`}
  >
    <canvas bind:this={canvas} aria-hidden="true"></canvas>
  </div>
  <div class="table-wrap">
    <table>
      <caption>Energy-error audit by chain</caption>
      <thead>
        <tr>
          <th scope="col">Chain</th>
          <th scope="col">Mean</th>
          <th scope="col">Max draw</th>
        </tr>
      </thead>
      <tbody>
        {#each chains as chain (chain.index)}
          <tr>
            <th scope="row">{chain.index}</th>
            <td class="tabular">{fixed(chain.mean, 5)}</td>
            <td class="tabular">{fixed(chain.max, 3)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <figcaption>
    Absolute Metropolis energy error per post-warmup draw. The gate uses the mean across chains, not
    any single draw. Divergences: <strong>{divergences}</strong>.
  </figcaption>
</figure>

<style>
  .energy {
    margin: 0;
    padding: 1.25rem;
    background: var(--surface);
    border: var(--hairline);
    border-radius: var(--radius);
  }

  canvas {
    display: block;
    width: 100%;
    height: 190px;
  }

  figcaption {
    margin: 0.5rem 0 0.8rem;
    color: var(--ink-dim);
    font-size: 0.78rem;
    max-width: 70ch;
  }

  .table-wrap {
    max-height: 170px;
    overflow: auto;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.72rem;
  }

  caption {
    text-align: left;
    padding-bottom: 0.4rem;
    color: var(--ink-dim);
    font-size: 0.7rem;
  }

  th,
  td {
    padding: 0.35rem 0.45rem;
    border-bottom: var(--hairline);
    text-align: right;
  }

  th:first-child,
  td:first-child {
    text-align: left;
  }

  thead th {
    color: var(--ink-dim);
    font-weight: 500;
  }
</style>
