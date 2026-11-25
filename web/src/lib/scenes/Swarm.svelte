<script lang="ts">
  import { onMount } from 'svelte';
  import type { SamplerResult, SvgdComparisonResult } from '../engine/protocol';
  import { fixed } from '../infer/stats';

  interface Props {
    result: SvgdComparisonResult | null;
    mclmc: SamplerResult | null;
    ready: boolean;
    running: boolean;
    error: string | null;
    reducedMotion: boolean;
    theme: string;
    onrun: (bandwidth: number, step: number) => void;
  }

  let { result, mclmc, ready, running, error, reducedMotion, theme, onrun }: Props = $props();

  let canvas = $state<HTMLCanvasElement | null>(null);
  let readyToDraw = $state(false);
  let frame = $state(0);
  let bandwidth = $state(0.5);
  let step = $state(0.08);
  let playTimer: ReturnType<typeof setInterval> | undefined;

  const lastFrame = $derived(result ? Math.max(0, result.frameCount - 1) : 0);
  const currentFrame = $derived(result ? Math.min(frame, lastFrame) : 0);
  const labels = $derived(
    (result?.parameterNames ?? []).map((name) =>
      name === 'eg_ev' ? 'E_g' : name === 'ln_b' ? 'ln B' : name === 'eta' ? 'η' : name,
    ),
  );

  const rows = $derived.by(() => {
    if (!result) return [];
    return result.parameterNames.map((name, index) => {
      const svgdSd = Math.sqrt(Math.max(0, result.svgdVariance[index] ?? 0));
      const mclmcSd = Math.sqrt(Math.max(0, result.mclmcVariance[index] ?? 0));
      const ratio = mclmcSd > 0 ? svgdSd / mclmcSd : Number.NaN;
      return {
        name: labels[index] ?? name,
        svgdMean: result.svgdMean[index] ?? Number.NaN,
        mclmcMean: result.mclmcMean[index] ?? Number.NaN,
        svgdSd,
        mclmcSd,
        ratio,
      };
    });
  });

  const varianceText = $derived.by(() => {
    if (!result) return '';
    const ratios = result.parameterNames
      .map((_, index) => {
        const svgd = result.svgdVariance[index] ?? 0;
        const mclmc = result.mclmcVariance[index] ?? 0;
        return mclmc > 0 ? Math.sqrt(Math.max(0, svgd) / mclmc) : Number.NaN;
      })
      .filter((value) => Number.isFinite(value));
    if (ratios.length === 0) {
      return 'MCLMC variance is zero, so the variance ratio is undefined.';
    }
    const low = Math.min(...ratios) * 100;
    const high = Math.max(...ratios) * 100;
    if (low >= 98 && high <= 102) {
      return `SVGD's marginal standard deviation is within 2% of MCLMC's on this two-dimensional target. The table still reports the per-parameter ratio.`;
    }
    return `SVGD's marginal standard deviation is ${low.toFixed(0)}–${high.toFixed(
      0,
    )}% of MCLMC's across parameters. The table reports each ratio; this is the variance bias, not a hidden correction.`;
  });

  const description = $derived(
    result
      ? `S3 The Swarm. Frame ${currentFrame + 1} of ${result.frameCount}; ${
          result.particleCount
        } SVGD particles. MCLMC overlay has ${result.mclmcParticleCount} draws. Sliced-W2 is ${fixed(
          result.w2Distance,
          4,
        )} plus or minus ${fixed(result.w2BootstrapSe, 4)} in prior-standardised units. ${varianceText}`
      : 'S3 The Swarm has not run yet.',
  );

  const summary = $derived(
    result
      ? `Sliced-W2 between SVGD and MCLMC is ${fixed(result.w2Distance, 4)} ± ${fixed(
          result.w2BootstrapSe,
          4,
        )} in prior-standardised units. ${varianceText}`
      : '',
  );

  function stopPlayback() {
    if (playTimer) clearInterval(playTimer);
    playTimer = undefined;
  }

  function startPlayback() {
    const data = result;
    if (!data || reducedMotion) return;
    stopPlayback();
    frame = 0;
    playTimer = setInterval(() => {
      if (!result || frame >= result.frameCount - 1) {
        stopPlayback();
        return;
      }
      frame += 1;
    }, 55);
  }

  function replay() {
    if (!result || reducedMotion) return;
    startPlayback();
  }

  function draw() {
    const element = canvas;
    if (!element) return;
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
    const surface = styles.getPropertyValue('--surface-2').trim() || 'Canvas';

    const data = result;
    if (!data || data.frameCount === 0 || data.parameterCount < 2) return;

    const plot = { x: 58, y: 20, width: width - 82, height: height - 62 };
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    const include = (x: number, y: number) => {
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    };
    for (let index = 0; index < data.trajectory.length; index += data.parameterCount) {
      include(data.trajectory[index], data.trajectory[index + 1]);
    }
    for (let index = 0; index < data.mclmcParticles.length; index += data.parameterCount) {
      include(data.mclmcParticles[index], data.mclmcParticles[index + 1]);
    }
    if (!Number.isFinite(minX) || !Number.isFinite(minY)) return;
    const padX = (maxX - minX) * 0.08 || 0.005;
    const padY = (maxY - minY) * 0.08 || 0.005;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;
    const xScale = (value: number) => plot.x + ((value - minX) / (maxX - minX)) * plot.width;
    const yScale = (value: number) =>
      plot.y + plot.height - ((value - minY) / (maxY - minY)) * plot.height;

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
      context.fillText(fixed(minX + (tick / 4) * (maxX - minX), 3), x, plot.y + plot.height + 15);
      context.textAlign = 'right';
      context.fillText(fixed(minY + (tick / 4) * (maxY - minY), 3), plot.x - 8, y);
    }

    context.globalAlpha = 0.42;
    context.fillStyle = dim;
    for (let index = 0; index < data.mclmcParticles.length; index += data.parameterCount) {
      const x = data.mclmcParticles[index];
      const y = data.mclmcParticles[index + 1];
      if (x < minX || x > maxX || y < minY || y > maxY) continue;
      context.beginPath();
      context.arc(xScale(x), yScale(y), 1.15, 0, Math.PI * 2);
      context.fill();
    }

    const offset = currentFrame * data.particleCount * data.parameterCount;
    context.globalAlpha = 0.92;
    context.fillStyle = accent;
    for (let point = 0; point < data.particleCount; point += 1) {
      const index = offset + point * data.parameterCount;
      const x = data.trajectory[index];
      const y = data.trajectory[index + 1];
      if (x < minX || x > maxX || y < minY || y > maxY) continue;
      context.beginPath();
      context.arc(xScale(x), yScale(y), 1.8, 0, Math.PI * 2);
      context.fill();
    }
    context.globalAlpha = 1;

    context.fillStyle = ink;
    context.textAlign = 'left';
    context.fillText(`SVGD frame ${currentFrame + 1}/${data.frameCount}`, plot.x, 9);
    context.fillStyle = dim;
    context.textAlign = 'right';
    context.fillText(
      `sliced-W2 ${fixed(data.w2Distance, 4)} ± ${fixed(data.w2BootstrapSe, 4)}`,
      plot.x + plot.width,
      9,
    );
    context.textAlign = 'center';
    context.fillText(labels[0] ?? 'parameter 1', plot.x + plot.width / 2, height - 12);
    context.save();
    context.translate(15, plot.y + plot.height / 2);
    context.rotate(-Math.PI / 2);
    context.fillText(labels[1] ?? 'parameter 2', 0, 0);
    context.restore();

    const inset = {
      x: plot.x + plot.width - 146,
      y: plot.y + 12,
      width: 134,
      height: 92,
    };
    context.fillStyle = surface;
    context.globalAlpha = 0.92;
    context.fillRect(inset.x, inset.y, inset.width, inset.height);
    context.globalAlpha = 1;
    context.strokeStyle = hair;
    context.strokeRect(inset.x, inset.y, inset.width, inset.height);

    let finalMinX = Number.POSITIVE_INFINITY;
    let finalMaxX = Number.NEGATIVE_INFINITY;
    let finalMinY = Number.POSITIVE_INFINITY;
    let finalMaxY = Number.NEGATIVE_INFINITY;
    const includeFinal = (x: number, y: number) => {
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      finalMinX = Math.min(finalMinX, x);
      finalMaxX = Math.max(finalMaxX, x);
      finalMinY = Math.min(finalMinY, y);
      finalMaxY = Math.max(finalMaxY, y);
    };
    for (let index = 0; index < data.particles.length; index += data.parameterCount) {
      includeFinal(data.particles[index], data.particles[index + 1]);
    }
    for (let index = 0; index < data.mclmcParticles.length; index += data.parameterCount) {
      includeFinal(data.mclmcParticles[index], data.mclmcParticles[index + 1]);
    }
    if (Number.isFinite(finalMinX) && Number.isFinite(finalMinY)) {
      const finalPadX = (finalMaxX - finalMinX) * 0.18 || 0.001;
      const finalPadY = (finalMaxY - finalMinY) * 0.18 || 0.001;
      finalMinX -= finalPadX;
      finalMaxX += finalPadX;
      finalMinY -= finalPadY;
      finalMaxY += finalPadY;
      const insetX = (value: number) =>
        inset.x + ((value - finalMinX) / (finalMaxX - finalMinX)) * inset.width;
      const insetY = (value: number) =>
        inset.y + inset.height - ((value - finalMinY) / (finalMaxY - finalMinY)) * inset.height;
      context.globalAlpha = 0.5;
      context.fillStyle = dim;
      for (let index = 0; index < data.mclmcParticles.length; index += data.parameterCount) {
        const x = data.mclmcParticles[index];
        const y = data.mclmcParticles[index + 1];
        if (x < finalMinX || x > finalMaxX || y < finalMinY || y > finalMaxY) continue;
        context.beginPath();
        context.arc(insetX(x), insetY(y), 1.1, 0, Math.PI * 2);
        context.fill();
      }
      context.globalAlpha = 0.9;
      context.fillStyle = accent;
      for (let index = 0; index < data.particles.length; index += data.parameterCount) {
        const x = data.particles[index];
        const y = data.particles[index + 1];
        if (x < finalMinX || x > finalMaxX || y < finalMinY || y > finalMaxY) continue;
        context.beginPath();
        context.arc(insetX(x), insetY(y), 1.4, 0, Math.PI * 2);
        context.fill();
      }
      context.globalAlpha = 1;
      context.fillStyle = dim;
      context.textAlign = 'left';
      context.fillText('final overlay', inset.x + 5, inset.y + 10);
    }
  }

  onMount(() => {
    readyToDraw = true;
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
    const data = result;
    if (!data) return;
    stopPlayback();
    frame = reducedMotion ? Math.max(0, data.frameCount - 1) : 0;
    if (!reducedMotion) startPlayback();
  });

  $effect(() => {
    theme;
    result;
    currentFrame;
    if (readyToDraw) draw();
  });
</script>

<figure class="swarm" data-testid="swarm">
  <div class="canvas-wrap" role="img" aria-label={description}>
    <canvas bind:this={canvas} aria-hidden="true"></canvas>
  </div>

  <div class="controls" aria-label="SVGD controls">
    <button
      class="primary"
      type="button"
      onclick={() => onrun(bandwidth, step)}
      disabled={running || !ready}
    >
      {running ? 'Running SVGD…' : result ? 'Re-run SVGD' : 'Run SVGD'}
    </button>
    {#if !reducedMotion}
      <button class="secondary" type="button" onclick={replay} disabled={!result || running}>
        Replay
      </button>
    {:else}
      <span class="static-note">Reduced motion: static final frame</span>
    {/if}

    <label class="control" for="swarm-bandwidth">
      <span>Bandwidth <span class="unit">(standardised)</span></span>
      <input
        id="swarm-bandwidth"
        type="range"
        min="0.15"
        max="1.5"
        step="0.05"
        bind:value={bandwidth}
      />
      <span class="tabular">{bandwidth.toFixed(2)}</span>
    </label>

    <label class="control" for="swarm-step">
      <span>Step <span class="unit">(standardised)</span></span>
      <input id="swarm-step" type="range" min="0.01" max="0.2" step="0.005" bind:value={step} />
      <span class="tabular">{step.toFixed(3)}</span>
    </label>

    <label class="control timeline" for="swarm-frame">
      <span>Frame</span>
      <input
        id="swarm-frame"
        type="range"
        min="0"
        max={lastFrame}
        step="1"
        value={currentFrame}
        disabled={!result}
        oninput={(event) => {
          stopPlayback();
          frame = Number((event.currentTarget as HTMLInputElement).value);
        }}
      />
      <span class="tabular">{currentFrame + 1} / {result ? result.frameCount : 0}</span>
    </label>
  </div>

  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}

  {#if result}
    <p class="w2" data-testid="swarm-w2" role="status" aria-live="polite">{summary}</p>
    {#if mclmc}
      <p class="diagnostics" class:fail={!mclmc.passes}>
        MCLMC reference: {mclmc.passes
          ? 'diagnostics pass'
          : 'diagnostics fail — comparison is not a validation'}. R-hat max {fixed(
          mclmc.rhatMax,
          3,
        )}; bulk ESS min {fixed(mclmc.essBulkMin, 0)}.
      </p>
    {/if}

    <div class="table-wrap">
      <table>
        <caption>S3 accessible equivalent: SVGD and MCLMC posterior summaries</caption>
        <thead>
          <tr>
            <th scope="col">Parameter</th>
            <th scope="col">SVGD mean</th>
            <th scope="col">MCLMC mean</th>
            <th scope="col">SVGD sd</th>
            <th scope="col">MCLMC sd</th>
            <th scope="col">sd ratio</th>
          </tr>
        </thead>
        <tbody>
          {#each rows as row (row.name)}
            <tr>
              <th scope="row">{row.name}</th>
              <td class="tabular">{fixed(row.svgdMean, 4)}</td>
              <td class="tabular">{fixed(row.mclmcMean, 4)}</td>
              <td class="tabular">{fixed(row.svgdSd, 4)}</td>
              <td class="tabular">{fixed(row.mclmcSd, 4)}</td>
              <td class="tabular">{Number.isFinite(row.ratio) ? row.ratio.toFixed(3) : '—'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    <figcaption>
      SVGD is deterministic: the RBF kernel is expected to contract the cloud relative to MCLMC,
      especially as dimension grows. This is a two-parameter synthetic M0 target; the sliced-W2 is
      computed in prior-standardised coordinates and the bootstrap is over random projections, not
      over the posterior. The MCLMC overlay is the comparison sample, not a second generative model.
    </figcaption>
  {:else}
    <p class="placeholder">
      No swarm yet. The button runs the Rust M0 sampler, then evolves SVGD particles in the same
      standardised coordinates and returns the trajectory, MCLMC overlay and sliced-W2 comparison.
    </p>
  {/if}
</figure>

<style>
  .swarm {
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
    display: grid;
    grid-template-columns: auto auto minmax(180px, 1fr) minmax(180px, 1fr) minmax(200px, 1.2fr);
    gap: 0.8rem;
    align-items: end;
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

  .control {
    display: grid;
    gap: 0.25rem;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--ink-dim);
  }

  .control input {
    width: 100%;
    min-height: 44px;
    accent-color: var(--accent);
  }

  .unit {
    text-transform: none;
    letter-spacing: 0;
    opacity: 0.75;
  }

  .static-note {
    align-self: center;
    color: var(--ink-dim);
    font-family: var(--font-mono);
    font-size: 0.7rem;
  }

  .w2,
  .diagnostics,
  .placeholder,
  figcaption {
    max-width: 82ch;
    color: var(--ink-dim);
    font-size: 0.78rem;
  }

  .w2 {
    margin: 0.8rem 0;
  }

  .diagnostics {
    margin: 0.3rem 0 0.8rem;
    font-family: var(--font-mono);
    font-size: 0.7rem;
  }

  .diagnostics.fail {
    color: oklch(0.78 0.13 35);
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
    max-height: 260px;
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

  @media (max-width: 900px) {
    .controls {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .timeline {
      grid-column: 1 / -1;
    }
  }

  @media (max-width: 560px) {
    .controls {
      grid-template-columns: 1fr;
    }

    .timeline {
      grid-column: auto;
    }
  }
</style>
