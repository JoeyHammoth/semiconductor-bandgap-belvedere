<script lang="ts">
  import { onMount } from 'svelte';
  import { compact, extent, fixed, padExtent, quantile } from './stats';

  interface Props {
    draws: Float64Array;
    parameterNames: string[];
    parameterCount: number;
    drawsPerParameter: number;
    rhat: Float64Array;
    essBulk: Float64Array;
    theme: string;
  }

  let { draws, parameterNames, parameterCount, drawsPerParameter, rhat, essBulk, theme }: Props =
    $props();

  let canvas = $state<HTMLCanvasElement | null>(null);
  let ready = $state(false);

  const labels = $derived(
    parameterNames.map((name) =>
      name === 'eg_ev' ? 'E_g(0 K) (eV)' : name === 'ln_b' ? 'ln B (A K⁻¹)' : name,
    ),
  );

  const summary = $derived.by(() => {
    if (parameterCount !== 2 || draws.length < 2) return [];
    return parameterNames.map((name, index) => {
      const values = draws.subarray(index * drawsPerParameter, (index + 1) * drawsPerParameter);
      return {
        name: labels[index] ?? name,
        low: quantile(values, 0.05),
        median: quantile(values, 0.5),
        high: quantile(values, 0.95),
      };
    });
  });

  const description = $derived(
    `Corner plot of ${compact(drawsPerParameter)} posterior draws for ${labels.join(
      ' and ',
    )}. R-hat maximum ${fixed(rhat[0] ?? Number.NaN, 5)}; ESS minimum ${fixed(
      essBulk[0] ?? Number.NaN,
      1,
    )}.`,
  );

  function draw() {
    const element = canvas;
    if (!element || parameterCount !== 2 || drawsPerParameter < 2) return;
    const width = Math.max(320, Math.floor(element.clientWidth));
    const height = Math.max(300, Math.floor(element.clientHeight));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (element.width !== width * dpr) element.width = width * dpr;
    if (element.height !== height * dpr) element.height = height * dpr;

    const context = element.getContext('2d');
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);

    const styles = getComputedStyle(document.documentElement);
    const ink = styles.getPropertyValue('--ink').trim() || 'CanvasText';
    const dim = styles.getPropertyValue('--ink-dim').trim() || 'CanvasText';
    const accent = styles.getPropertyValue('--accent').trim() || 'CanvasText';
    const hair = styles.getPropertyValue('--hair').trim() || 'CanvasText';

    const first = draws.subarray(0, drawsPerParameter);
    const second = draws.subarray(drawsPerParameter, drawsPerParameter * 2);
    const [x0, x1] = padExtent(...extent(first));
    const [y0, y1] = padExtent(...extent(second));

    const margin = { left: 58, right: 16, top: 16, bottom: 44 };
    const gap = 12;
    const panelWidth = (width - margin.left - margin.right - gap) / 2;
    const panelHeight = (height - margin.top - margin.bottom - gap) / 2;
    const panel = (column: number, row: number) => ({
      x: margin.left + column * (panelWidth + gap),
      y: margin.top + row * (panelHeight + gap),
      width: panelWidth,
      height: panelHeight,
    });

    const frame = (rect: { x: number; y: number; width: number; height: number }) => {
      context.strokeStyle = hair;
      context.lineWidth = 1;
      context.strokeRect(rect.x, rect.y, rect.width, rect.height);
    };

    const xScale = (value: number) => panel(0, 0).x + ((value - x0) / (x1 - x0)) * panelWidth;
    const yScale = (value: number) => panel(1, 1).y + panelHeight - ((value - y0) / (y1 - y0)) * panelHeight;

    context.font = '10px "JetBrains Mono Variable", ui-monospace, monospace';
    context.textBaseline = 'middle';

    const diagonal = panel(0, 0);
    frame(diagonal);
    const bins = 34;
    const counts = new Array<number>(bins).fill(0);
    for (const value of first) {
      const bin = Math.min(bins - 1, Math.max(0, Math.floor(((value - x0) / (x1 - x0)) * bins)));
      counts[bin] += 1;
    }
    const maxCount = Math.max(...counts, 1);
    context.fillStyle = accent;
    context.globalAlpha = 0.72;
    counts.forEach((count, index) => {
      const barWidth = diagonal.width / bins;
      const barHeight = (count / maxCount) * (diagonal.height - 10);
      context.fillRect(
        diagonal.x + index * barWidth + 1,
        diagonal.y + diagonal.height - barHeight,
        Math.max(barWidth - 2, 1),
        barHeight,
      );
    });
    context.globalAlpha = 1;
    context.fillStyle = dim;
    context.fillText(labels[0] ?? 'parameter 1', diagonal.x, diagonal.y + diagonal.height + 16);
    context.textAlign = 'right';
    context.fillText(fixed(x0, 3), diagonal.x + 4, diagonal.y + diagonal.height - 4);
    context.fillText(fixed(x1, 3), diagonal.x + diagonal.width, diagonal.y + diagonal.height - 4);
    context.textAlign = 'left';

    const scatter = panel(1, 0);
    frame(scatter);
    context.fillStyle = accent;
    context.globalAlpha = 0.32;
    const stride = Math.max(1, Math.floor(drawsPerParameter / 6_000));
    for (let index = 0; index < drawsPerParameter; index += stride) {
      const x = scatter.x + ((first[index] - x0) / (x1 - x0)) * scatter.width;
      const y = yScale(second[index]);
      context.fillRect(x - 0.7, y - 0.7, 1.5, 1.5);
    }
    context.globalAlpha = 1;
    context.strokeStyle = hair;
    for (let tick = 0; tick <= 4; tick += 1) {
      const x = scatter.x + (tick / 4) * scatter.width;
      const y = scatter.y + scatter.height - (tick / 4) * scatter.height;
      context.beginPath();
      context.moveTo(x, scatter.y);
      context.lineTo(x, scatter.y + scatter.height);
      context.stroke();
      context.beginPath();
      context.moveTo(scatter.x, y);
      context.lineTo(scatter.x + scatter.width, y);
      context.stroke();
    }
    context.fillStyle = dim;
    context.fillText(labels[0] ?? 'parameter 1', scatter.x, scatter.y + scatter.height + 16);
    context.save();
    context.translate(scatter.x - 12, scatter.y + scatter.height / 2);
    context.rotate(-Math.PI / 2);
    context.textAlign = 'center';
    context.fillText(labels[1] ?? 'parameter 2', 0, 0);
    context.restore();

    const secondDiagonal = panel(1, 1);
    frame(secondDiagonal);
    const yCounts = new Array<number>(bins).fill(0);
    for (const value of second) {
      const bin = Math.min(bins - 1, Math.max(0, Math.floor(((value - y0) / (y1 - y0)) * bins)));
      yCounts[bin] += 1;
    }
    const maxYCount = Math.max(...yCounts, 1);
    context.fillStyle = accent;
    context.globalAlpha = 0.72;
    yCounts.forEach((count, index) => {
      const barHeight = secondDiagonal.height / bins;
      const barWidth = (count / maxYCount) * (secondDiagonal.width - 10);
      context.fillRect(
        secondDiagonal.x,
        secondDiagonal.y + secondDiagonal.height - (index + 1) * barHeight + 1,
        barWidth,
        Math.max(barHeight - 2, 1),
      );
    });
    context.globalAlpha = 1;
    context.fillStyle = dim;
    context.textAlign = 'center';
    context.fillText(labels[1] ?? 'parameter 2', secondDiagonal.x + secondDiagonal.width / 2, secondDiagonal.y + secondDiagonal.height + 16);
    context.textAlign = 'left';

    const notes = panel(0, 1);
    context.fillStyle = dim;
    context.font = '11px "Hanken Grotesk Variable", sans-serif';
    context.fillText('M0 posterior', notes.x, notes.y + 18);
    context.font = '10px "JetBrains Mono Variable", ui-monospace, monospace';
    context.fillText(`${compact(drawsPerParameter)} draws`, notes.x, notes.y + 40);
    context.fillText(`R-hat max ${fixed(rhat[0] ?? Number.NaN, 5)}`, notes.x, notes.y + 58);
    context.fillText(`ESS min ${fixed(essBulk[0] ?? Number.NaN, 1)}`, notes.x, notes.y + 76);
    context.fillStyle = accent;
    context.fillText('lower triangle · 5–95% table', notes.x, notes.y + 100);
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
    draws;
    parameterCount;
    if (ready) draw();
  });
</script>

<figure class="corner" data-testid="corner-plot">
  <div class="canvas-wrap" role="img" aria-label={description}>
    <canvas bind:this={canvas} aria-hidden="true"></canvas>
  </div>
  <div class="table-wrap">
    <table>
      <caption>Posterior summary for the pooled chains</caption>
      <thead>
        <tr>
          <th scope="col">Parameter</th>
          <th scope="col">Median</th>
          <th scope="col">5%</th>
          <th scope="col">95%</th>
          <th scope="col">R-hat</th>
          <th scope="col">ESS</th>
        </tr>
      </thead>
      <tbody>
        {#each summary as row, index (row.name)}
          <tr>
            <th scope="row">{row.name}</th>
            <td class="tabular">{fixed(row.median, 6)}</td>
            <td class="tabular">{fixed(row.low, 6)}</td>
            <td class="tabular">{fixed(row.high, 6)}</td>
            <td class="tabular">{fixed(rhat[index] ?? Number.NaN, 5)}</td>
            <td class="tabular">{fixed(essBulk[index] ?? Number.NaN, 1)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <figcaption>
    Marginal posteriors and the joint lower triangle. The canvas is not the accessible source;
    the table carries the interval summary.
  </figcaption>
</figure>

<style>
  .corner {
    margin: 0;
    padding: 1.25rem;
    background: var(--surface);
    border: var(--hairline);
    border-radius: var(--radius);
  }

  canvas {
    display: block;
    width: 100%;
    height: 430px;
  }

  figcaption {
    margin: 0.6rem 0 0.9rem;
    color: var(--ink-dim);
    font-size: 0.78rem;
    max-width: 70ch;
  }

  .table-wrap {
    max-height: 210px;
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
    padding: 0.4rem 0.45rem;
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
</style>
