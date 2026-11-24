<script lang="ts">
  import { onMount } from 'svelte';
  import type { NestedTraceResult } from '../engine/protocol';
  import { fixed } from '../infer/stats';

  interface Props {
    trace: NestedTraceResult | null;
    theme: string;
    reducedMotion: boolean;
    running: boolean;
    error: string | null;
    onrun: () => void;
  }

  let { trace, theme, reducedMotion, running, error, onrun }: Props = $props();

  let canvas = $state<HTMLCanvasElement | null>(null);
  let ready = $state(false);
  let frame = $state(0);
  let playTimer: ReturnType<typeof setInterval> | undefined;

  const lastFrame = $derived(trace ? Math.max(0, trace.frameCount - 1) : 0);
  const currentLogEvidence = $derived(
    trace && trace.frameCount > 0 ? trace.eventLogEvidence[frame] : Number.NaN,
  );
  const currentIteration = $derived(
    trace && trace.frameCount > 0 ? trace.eventIterations[frame] : 0,
  );

  const sampledFrames = $derived.by(() => {
    if (!trace || trace.frameCount === 0) return [];
    const candidates = [0, 0.25, 0.5, 0.75, 1].map((fraction) =>
      Math.round(fraction * (trace.frameCount - 1)),
    );
    return [...new Set(candidates)].map((index) => ({
      index,
      iteration: trace.eventIterations[index],
      logEvidence: trace.eventLogEvidence[index],
      threshold: trace.eventThresholds[index],
      logPriorVolume: trace.eventLogPriorVolume[index],
    }));
  });

  const meterLow = $derived(
    trace && trace.eventLogEvidence.length > 0 ? Math.min(...trace.eventLogEvidence) : 0,
  );
  const meterHigh = $derived(
    trace && trace.eventLogEvidence.length > 0 ? Math.max(...trace.eventLogEvidence) : 1,
  );
  const meter = $derived(
    Number.isFinite(currentLogEvidence) && meterHigh > meterLow
      ? Math.max(0, Math.min(1, (currentLogEvidence - meterLow) / (meterHigh - meterLow)))
      : 0,
  );

  const description = $derived(
    trace
      ? `S2 Shrinking Light. Frame ${frame + 1} of ${trace.frameCount}, iteration ${currentIteration}, ` +
          `${trace.livePoints} live points. Accumulated log evidence ${fixed(currentLogEvidence, 3)}. ` +
          `The synthetic Gaussian fixture has closed-form log evidence ${fixed(trace.expectedLogEvidence, 3)}.`
      : 'S2 Shrinking Light has not run yet.',
  );

  function stopPlayback() {
    if (playTimer) clearInterval(playTimer);
    playTimer = undefined;
  }

  function replay() {
    if (!trace || trace.frameCount === 0 || reducedMotion) return;
    stopPlayback();
    frame = 0;
    playTimer = setInterval(() => {
      if (!trace || frame >= trace.frameCount - 1) {
        stopPlayback();
        return;
      }
      frame += 1;
    }, 45);
  }

  function draw() {
    const element = canvas;
    if (!element || !trace || trace.frameCount === 0) return;
    const width = Math.max(320, Math.floor(element.clientWidth));
    const height = Math.max(280, Math.floor(element.clientHeight));
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

    const plot = { x: 62, y: 18, width: width - 82, height: height - 58 };
    const lowX = trace.priorLow[0];
    const highX = trace.priorHigh[0];
    const lowY = trace.priorLow[1];
    const highY = trace.priorHigh[1];
    const xScale = (value: number) => plot.x + ((value - lowX) / (highX - lowX)) * plot.width;
    const yScale = (value: number) =>
      plot.y + plot.height - ((value - lowY) / (highY - lowY)) * plot.height;

    context.font = '10px "JetBrains Mono Variable", ui-monospace, monospace';
    context.textBaseline = 'middle';
    context.strokeStyle = hair;
    context.lineWidth = 1;
    context.strokeRect(plot.x, plot.y, plot.width, plot.height);

    context.fillStyle = dim;
    for (let tick = 0; tick <= 4; tick += 1) {
      const x = plot.x + (tick / 4) * plot.width;
      const y = plot.y + plot.height - (tick / 4) * plot.height;
      context.strokeStyle = hair;
      context.beginPath();
      context.moveTo(x, plot.y);
      context.lineTo(x, plot.y + plot.height);
      context.stroke();
      context.beginPath();
      context.moveTo(plot.x, y);
      context.lineTo(plot.x + plot.width, y);
      context.stroke();
      context.textAlign = 'center';
      context.fillText(fixed(lowX + (tick / 4) * (highX - lowX), 3), x, plot.y + plot.height + 16);
      context.textAlign = 'right';
      context.fillText(fixed(lowY + (tick / 4) * (highY - lowY), 2), plot.x - 8, y);
    }

    const pointCount = trace.livePoints;
    const parameterCount = trace.parameterNames.length || 2;
    const offset = frame * pointCount * parameterCount;
    const points: { x: number; y: number }[] = [];
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (let point = 0; point < pointCount; point += 1) {
      const base = offset + point * parameterCount;
      const x = trace.livePositions[base];
      const y = trace.livePositions[base + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      points.push({ x: xScale(x), y: yScale(y) });
    }

    if (points.length > 0) {
      context.strokeStyle = accent;
      context.globalAlpha = 0.28;
      context.setLineDash([4, 4]);
      context.strokeRect(
        xScale(minX),
        yScale(maxY),
        Math.max(1, xScale(maxX) - xScale(minX)),
        Math.max(1, yScale(minY) - yScale(maxY)),
      );
      context.setLineDash([]);
      context.globalAlpha = 1;
    }

    context.fillStyle = accent;
    for (const point of points) {
      context.beginPath();
      context.arc(point.x, point.y, 1.7, 0, Math.PI * 2);
      context.fill();
    }

    context.fillStyle = ink;
    context.textAlign = 'left';
    context.fillText(`frame ${frame + 1}/${trace.frameCount}`, plot.x, 8);
    context.fillStyle = dim;
    context.textAlign = 'right';
    context.fillText(
      `${trace.livePoints} live points · ln Z ${fixed(currentLogEvidence, 3)}`,
      plot.x + plot.width,
      8,
    );
    context.textAlign = 'center';
    context.fillText('E_g (eV)', plot.x + plot.width / 2, height - 12);
    context.save();
    context.translate(15, plot.y + plot.height / 2);
    context.rotate(-Math.PI / 2);
    context.fillText('ln B', 0, 0);
    context.restore();
    context.textAlign = 'left';
    context.fillStyle = dim;
    context.fillText('synthetic Gaussian fixture · full prior box', plot.x, height - 12);
  }

  onMount(() => {
    ready = true;
    const element = canvas;
    if (!element) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(element);
    draw();
    return () => {
      observer.disconnect();
      stopPlayback();
    };
  });

  $effect(() => {
    theme;
    trace;
    frame;
    if (ready) draw();
  });

  $effect(() => {
    if (trace) {
      stopPlayback();
      frame = Math.max(0, trace.frameCount - 1);
    }
  });
</script>

<figure class="shrinking" data-testid="nested-trace">
  <div class="canvas-wrap" role="img" aria-label={description}>
    <canvas bind:this={canvas} aria-hidden="true"></canvas>
  </div>

  <div class="controls" aria-label="Timeline controls">
    <button class="primary" type="button" onclick={onrun} disabled={running}>
      {running ? 'Sampling…' : trace ? 'Re-run nested sampling' : 'Run nested sampling'}
    </button>
    {#if !reducedMotion}
      <button class="secondary" type="button" onclick={replay} disabled={!trace || running}>
        Replay
      </button>
    {:else}
      <span class="static-note">Reduced motion: static final frame</span>
    {/if}
    <label class="scrub" for="nested-frame">
      <span>Timeline</span>
      <input
        id="nested-frame"
        data-testid="nested-scrub"
        type="range"
        min="0"
        max={lastFrame}
        step="1"
        value={frame}
        disabled={!trace}
        oninput={(event) => {
          stopPlayback();
          frame = Number((event.currentTarget as HTMLInputElement).value);
        }}
      />
      <span class="tabular">{frame + 1} / {trace ? trace.frameCount : 0}</span>
    </label>
  </div>

  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}

  {#if trace}
    <div class="meter-row" aria-hidden="true">
      <span class="meter-label">ln Z</span>
      <div class="meter"><span style={`width: ${meter * 100}%`}></span></div>
      <span class="tabular">{fixed(currentLogEvidence, 3)}</span>
    </div>
    <p class="readout" role="status" aria-live="polite">
      Frame <strong>{frame + 1}</strong> of <strong>{trace.frameCount}</strong>; iteration
      <strong>{currentIteration}</strong>; live points <strong>{trace.livePoints}</strong>;
      threshold <strong>{fixed(trace.eventThresholds[frame], 3)}</strong>. Final
      <strong>ln Z = {fixed(trace.logEvidence, 3)} ± {fixed(trace.logEvidenceError, 3)}</strong>
      against the closed-form <strong>{fixed(trace.expectedLogEvidence, 3)}</strong>.
    </p>
    <div class="table-wrap">
      <table>
        <caption>S2 accessible equivalent: sampled nested-sampling events</caption>
        <thead>
          <tr>
            <th scope="col">Frame</th>
            <th scope="col">Iteration</th>
            <th scope="col">Live points</th>
            <th scope="col">ln Z</th>
            <th scope="col">Threshold</th>
            <th scope="col">ln prior volume</th>
          </tr>
        </thead>
        <tbody>
          {#each sampledFrames as row (row.index)}
            <tr>
              <th scope="row" class="tabular">{row.index + 1}</th>
              <td class="tabular">{row.iteration}</td>
              <td class="tabular">{trace.livePoints}</td>
              <td class="tabular">{fixed(row.logEvidence, 3)}</td>
              <td class="tabular">{fixed(row.threshold, 3)}</td>
              <td class="tabular">{fixed(row.logPriorVolume, 3)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <figcaption>
      The live set is a projection of the Rust event stream. The input is a labelled synthetic
      Gaussian calibration fixture, not a measured-data result; its closed-form evidence is
      committed in the nested-sampling validation.
    </figcaption>
  {:else}
    <p class="placeholder">
      No trace yet. The scene runs the same constrained-prior nested sampler that passed the
      closed-form and dynesty checks; the canvas receives every live-point snapshot from Rust.
    </p>
  {/if}
</figure>

<style>
  .shrinking {
    margin: 0;
    padding: 1.25rem;
    background: var(--surface);
    border: var(--hairline);
    border-radius: var(--radius);
  }

  canvas {
    display: block;
    width: 100%;
    height: 360px;
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    align-items: center;
    margin: 0.8rem 0;
  }

  button {
    min-height: 44px;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.76rem;
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

  .scrub {
    display: flex;
    flex: 1 1 260px;
    gap: 0.6rem;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--ink-dim);
  }

  .scrub input {
    flex: 1;
    min-width: 100px;
    accent-color: var(--accent);
    min-height: 44px;
  }

  .static-note {
    color: var(--ink-dim);
    font-family: var(--font-mono);
    font-size: 0.7rem;
  }

  .meter-row {
    display: flex;
    gap: 0.7rem;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--ink-dim);
  }

  .meter {
    flex: 1;
    height: 8px;
    overflow: hidden;
    border: var(--hairline);
    border-radius: 999px;
    background: var(--surface-2);
  }

  .meter span {
    display: block;
    height: 100%;
    background: var(--accent);
  }

  .readout,
  .placeholder,
  figcaption {
    max-width: 78ch;
    color: var(--ink-dim);
    font-size: 0.78rem;
  }

  .readout {
    margin: 0.7rem 0;
  }

  .placeholder {
    margin: 0.8rem 0 0;
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

  .table-wrap {
    max-height: 220px;
    overflow: auto;
    margin: 0.8rem 0;
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

  figcaption {
    margin-top: 0.4rem;
  }
</style>
