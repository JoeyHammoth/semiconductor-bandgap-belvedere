<script lang="ts">
  import { onMount } from 'svelte';
  import type { PredictiveResult } from '../engine/protocol';
  import { extent, fixed, padExtent } from './stats';

  interface Props {
    temperatures: number[];
    observed: number[];
    predictive: PredictiveResult;
    theme: string;
  }

  let { temperatures, observed, predictive, theme }: Props = $props();

  let canvas = $state<HTMLCanvasElement | null>(null);
  let ready = $state(false);

  const rows = $derived(
    temperatures.map((temperature, index) => ({
      temperature,
      observed: observed[index],
      low: predictive.q05[index],
      median: predictive.q50[index],
      high: predictive.q95[index],
    })),
  );

  const description = $derived(
    `S1 noisy line. ${temperatures.length} synthetic voltage observations at 0.3 mA, with ` +
      `${predictive.lineCount} posterior curves and a 90% posterior predictive band.`,
  );

  function draw() {
    const element = canvas;
    if (!element || temperatures.length < 2 || predictive.pointCount !== temperatures.length)
      return;
    const width = Math.max(320, Math.floor(element.clientWidth));
    const height = Math.max(260, Math.floor(element.clientHeight));
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

    const [temperatureLow, temperatureHigh] = padExtent(...extent(temperatures), 0.02);
    const allVoltages = [...observed, ...predictive.q05, ...predictive.q95];
    const [voltageLow, voltageHigh] = padExtent(...extent(allVoltages));
    const plot = {
      x: 64,
      y: 14,
      width: width - 82,
      height: height - 56,
    };
    const xScale = (value: number) =>
      plot.x + ((value - temperatureLow) / (temperatureHigh - temperatureLow)) * plot.width;
    const yScale = (value: number) =>
      plot.y + plot.height - ((value - voltageLow) / (voltageHigh - voltageLow)) * plot.height;

    context.font = '10px "JetBrains Mono Variable", ui-monospace, monospace';
    context.textBaseline = 'middle';
    context.strokeStyle = hair;
    context.lineWidth = 1;
    context.strokeRect(plot.x, plot.y, plot.width, plot.height);

    context.fillStyle = dim;
    const xTicks = 5;
    const yTicks = 5;
    for (let index = 0; index <= xTicks; index += 1) {
      const x = plot.x + (index / xTicks) * plot.width;
      context.strokeStyle = hair;
      context.beginPath();
      context.moveTo(x, plot.y);
      context.lineTo(x, plot.y + plot.height);
      context.stroke();
      context.textAlign = 'center';
      context.fillText(
        fixed(temperatureLow + (index / xTicks) * (temperatureHigh - temperatureLow), 0),
        x,
        plot.y + plot.height + 16,
      );
    }
    for (let index = 0; index <= yTicks; index += 1) {
      const y = plot.y + plot.height - (index / yTicks) * plot.height;
      context.strokeStyle = hair;
      context.beginPath();
      context.moveTo(plot.x, y);
      context.lineTo(plot.x + plot.width, y);
      context.stroke();
      context.textAlign = 'right';
      context.fillText(
        fixed(voltageLow + (index / yTicks) * (voltageHigh - voltageLow), 3),
        plot.x - 8,
        y,
      );
    }

    const band = new Path2D();
    predictive.q05.forEach((value, index) => {
      const x = xScale(temperatures[index]);
      const y = yScale(value);
      if (index === 0) band.moveTo(x, y);
      else band.lineTo(x, y);
    });
    for (let index = predictive.q95.length - 1; index >= 0; index -= 1) {
      band.lineTo(xScale(temperatures[index]), yScale(predictive.q95[index]));
    }
    band.closePath();
    context.fillStyle = accent;
    context.globalAlpha = 0.14;
    context.fill(band);
    context.globalAlpha = 1;

    context.strokeStyle = accent;
    context.lineWidth = 0.8;
    context.globalAlpha = 0.07;
    for (let line = 0; line < predictive.lineCount; line += 1) {
      const start = line * predictive.pointCount;
      context.beginPath();
      for (let point = 0; point < predictive.pointCount; point += 1) {
        const x = xScale(temperatures[point]);
        const y = yScale(predictive.lines[start + point]);
        if (point === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    }
    context.globalAlpha = 1;

    context.strokeStyle = accent;
    context.lineWidth = 2;
    context.beginPath();
    predictive.q50.forEach((value, index) => {
      const x = xScale(temperatures[index]);
      const y = yScale(value);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();

    context.fillStyle = ink;
    observed.forEach((value, index) => {
      context.beginPath();
      context.arc(xScale(temperatures[index]), yScale(value), 2.2, 0, Math.PI * 2);
      context.fill();
    });

    context.fillStyle = dim;
    context.textAlign = 'center';
    context.fillText('temperature (K)', plot.x + plot.width / 2, height - 12);
    context.save();
    context.translate(15, plot.y + plot.height / 2);
    context.rotate(-Math.PI / 2);
    context.fillText('forward voltage (V)', 0, 0);
    context.restore();
    context.textAlign = 'left';
    context.fillStyle = dim;
    context.fillText(
      'S1 · observed  ·  median  ·  90% predictive  ·  500 posterior curves',
      plot.x,
      8,
    );
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
    temperatures;
    observed;
    predictive;
    if (ready) draw();
  });
</script>

<figure class="predictive" data-testid="s1-plot">
  <div class="canvas-wrap" role="img" aria-label={description}>
    <canvas bind:this={canvas} aria-hidden="true"></canvas>
  </div>
  <div class="table-wrap">
    <table>
      <caption>S1 accessible equivalent: observed and predictive voltage</caption>
      <thead>
        <tr>
          <th scope="col">T (K)</th>
          <th scope="col">Observed V</th>
          <th scope="col">Median V</th>
          <th scope="col">5% V</th>
          <th scope="col">95% V</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as row (row.temperature)}
          <tr>
            <th scope="row" class="tabular">{fixed(row.temperature, 2)}</th>
            <td class="tabular">{fixed(row.observed, 5)}</td>
            <td class="tabular">{fixed(row.median, 5)}</td>
            <td class="tabular">{fixed(row.low, 5)}</td>
            <td class="tabular">{fixed(row.high, 5)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <figcaption>
    Labelled synthetic observations. Thin curves are noiseless posterior draws; the shaded band is
    the 5–95% posterior predictive interval and includes the likelihood noise.
  </figcaption>
</figure>

<style>
  .predictive {
    margin: 0;
    padding: 1.25rem;
    background: var(--surface);
    border: var(--hairline);
    border-radius: var(--radius);
  }

  canvas {
    display: block;
    width: 100%;
    height: 340px;
  }

  figcaption {
    margin: 0.6rem 0 0.9rem;
    color: var(--ink-dim);
    font-size: 0.78rem;
    max-width: 72ch;
  }

  .table-wrap {
    max-height: 230px;
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
