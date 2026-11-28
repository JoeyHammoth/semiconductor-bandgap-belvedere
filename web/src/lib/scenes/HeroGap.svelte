<script lang="ts" module>
  interface Renderer {
    update: (gap: number, temperature: number) => void;
    dispose: () => void;
  }

  const WGSL = /* wgsl */ `
struct Uniforms {
  time: f32,
  gap: f32,
  aspect: f32,
  pad: f32,
  accent: vec4f,
}
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@location(0) p: vec2f) -> VOut {
  var out: VOut;
  out.position = vec4f(p.x / u.aspect, p.y, 0.0, 1.0);
  out.uv = p;
  return out;
}

@fragment
fn fs(input: VOut) -> @location(0) vec4f {
  let t = clamp(input.uv.y * 0.5 + 0.5, 0.0, 1.0);
  let band = smoothstep(0.07, 0.0, abs(t - (0.55 + u.gap * 0.25)));
  let pulse = 0.85 + 0.15 * sin(u.time * 0.8);
  let low = vec3f(0.10, 0.05, 0.22);
  let high = vec3f(1.0, 0.62, 0.20);
  let base = mix(low, high, t);
  let colour = mix(base, vec3f(0.02, 0.02, 0.03), band * 0.85) * pulse;
  return vec4f(colour, 1.0);
}
`;

  function resize(canvas: HTMLCanvasElement) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    return { width, height };
  }

  async function startWebGPU(
    canvas: HTMLCanvasElement,
    isReduced: () => boolean,
  ): Promise<Renderer | null> {
    const gpu = (navigator as Navigator & { gpu?: any }).gpu;
    if (!gpu) return null;
    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu') as any;
    if (!context) return null;

    const format = gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    const module = device.createShaderModule({ code: WGSL });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [
          { arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] },
        ],
      },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });

    const vertices = device.createBuffer({
      size: 24,
      usage: 0x20, // GPUBufferUsage.VERTEX
      mappedAtCreation: true,
    });
    new Float32Array(vertices.getMappedRange()).set([-0.85, -0.72, 0.85, -0.72, 0.0, 0.78]);
    vertices.unmap();

    const uniforms = device.createBuffer({
      size: 32,
      usage: 0x40 | 0x8, // GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const data = new Float32Array(8);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniforms } }],
    });

    let raf = 0;
    let currentGap = 1.11;
    const started = performance.now();

    const draw = (now: number) => {
      const { width, height } = resize(canvas);
      data[0] = (now - started) / 1000;
      data[1] = Math.min(Math.max(currentGap / 3, 0.05), 1);
      data[2] = width / height;
      device.queue.writeBuffer(uniforms, 0, data);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0.04, g: 0.03, b: 0.06, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.setVertexBuffer(0, vertices);
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
      if (!isReduced()) raf = requestAnimationFrame(draw);
    };

    const observer = new ResizeObserver(() => {
      resize(canvas);
      if (isReduced()) draw(performance.now());
    });
    observer.observe(canvas);
    draw(performance.now());

    return {
      update(nextGap: number) {
        currentGap = nextGap;
        if (isReduced()) draw(performance.now());
      },
      dispose() {
        cancelAnimationFrame(raf);
        observer.disconnect();
        context.unconfigure();
        device.destroy();
      },
    };
  }

  function startWebGL2(canvas: HTMLCanvasElement, isReduced: () => boolean): Renderer | null {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) return null;

    const vertexShader = `#version 300 es
      in vec2 a_position;
      uniform float u_aspect;
      out vec2 v_uv;
      void main() {
        v_uv = a_position;
        gl_Position = vec4(a_position.x / u_aspect, a_position.y, 0.0, 1.0);
      }`;

    const fragmentShader = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 out_colour;
      uniform float u_time;
      uniform float u_gap;
      void main() {
        float t = clamp(v_uv.y * 0.5 + 0.5, 0.0, 1.0);
        float band = smoothstep(0.07, 0.0, abs(t - (0.55 + u_gap * 0.25)));
        float pulse = 0.85 + 0.15 * sin(u_time * 0.8);
        vec3 low = vec3(0.10, 0.05, 0.22);
        vec3 high = vec3(1.0, 0.62, 0.20);
        vec3 base = mix(low, high, t);
        vec3 colour = mix(base, vec3(0.02, 0.02, 0.03), band * 0.85) * pulse;
        out_colour = vec4(colour, 1.0);
      }`;

    const compile = (type: number, source: string) => {
      const shader = gl.createShader(type);
      if (!shader) throw new Error('shader allocation failed');
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader) ?? 'shader compile failed');
      }
      return shader;
    };

    const program = gl.createProgram();
    if (!program) return null;
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexShader));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentShader));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) ?? 'program link failed');
    }
    gl.useProgram(program);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-0.85, -0.72, 0.85, -0.72, 0.0, 0.78]),
      gl.STATIC_DRAW,
    );
    const position = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uGap = gl.getUniformLocation(program, 'u_gap');
    const uAspect = gl.getUniformLocation(program, 'u_aspect');

    let raf = 0;
    let currentGap = 1.11;
    const started = performance.now();

    const draw = (now: number) => {
      const { width, height } = resize(canvas);
      gl.viewport(0, 0, width, height);
      gl.clearColor(0.04, 0.03, 0.06, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1f(uTime, (now - started) / 1000);
      gl.uniform1f(uGap, Math.min(Math.max(currentGap / 3, 0.05), 1));
      gl.uniform1f(uAspect, width / height);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      if (!isReduced()) raf = requestAnimationFrame(draw);
    };

    const observer = new ResizeObserver(() => {
      resize(canvas);
      if (isReduced()) draw(performance.now());
    });
    observer.observe(canvas);
    draw(performance.now());

    return {
      update(nextGap: number) {
        currentGap = nextGap;
        if (isReduced()) draw(performance.now());
      },
      dispose() {
        cancelAnimationFrame(raf);
        observer.disconnect();
        gl.getExtension('WEBGL_lose_context')?.loseContext();
      },
    };
  }
</script>

<script lang="ts">
  import { onMount } from 'svelte';

  interface Props {
    gap?: number | null;
    temperature?: number;
    onbackend?: (backend: 'webgpu' | 'webgl2' | 'static') => void;
  }

  let { gap = null, temperature = 300, onbackend = () => {} }: Props = $props();

  type Backend = 'detecting' | 'webgpu' | 'webgl2' | 'static';

  let canvas = $state<HTMLCanvasElement | null>(null);
  let backend = $state<Backend>('detecting');
  let update = (_gap: number, _temperature: number) => {};
  let reduced = false;

  $effect(() => {
    update(gap ?? 1.11, temperature);
  });

  onMount(() => {
    reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {
      backend = 'static';
      onbackend('static');
      return;
    }
    let dispose = () => {};
    let cancelled = false;

    void (async () => {
      const element = canvas;
      if (!element) {
        backend = 'static';
        onbackend('static');
        return;
      }
      try {
        const gpu = await startWebGPU(element, () => reduced);
        if (cancelled) {
          gpu?.dispose();
          return;
        }
        if (gpu) {
          update = gpu.update;
          dispose = gpu.dispose;
          backend = 'webgpu';
          onbackend('webgpu');
          gpu.update(gap ?? 1.11, temperature);
          return;
        }
      } catch {
        // fall through to WebGL2
      }
      let gl = null;
      try {
        gl = startWebGL2(element, () => reduced);
      } catch {
        gl = null;
      }
      if (cancelled) return gl?.dispose();
      if (gl) {
        update = gl.update;
        dispose = gl.dispose;
        backend = 'webgl2';
        onbackend('webgl2');
        gl.update(gap ?? 1.11, temperature);
        return;
      }
      backend = 'static';
      onbackend('static');
    })();

    return () => {
      cancelled = true;
      dispose();
    };
  });

  const bandY = $derived(262 - Math.min(Math.max((gap ?? 1.11) / 3, 0.1), 1) * 150);
</script>

<div class="scene panel" data-backend={backend}>
  <canvas bind:this={canvas} aria-hidden="true"></canvas>
  {#if backend !== 'webgpu' && backend !== 'webgl2'}
    <svg
      viewBox="0 0 400 300"
      role="img"
      aria-label={`Sketch of the gap: ${gap === null ? 'unknown' : gap.toFixed(3)} electronvolts at ${temperature.toFixed(0)} kelvin`}
    >
      <defs>
        <linearGradient id="hero-gap" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="oklch(0.55 0.12 300)" />
          <stop offset="100%" stop-color="oklch(0.82 0.16 70)" />
        </linearGradient>
      </defs>
      <rect width="400" height="300" fill="oklch(0.12 0.01 265)" rx="10" />
      <polygon points="200,22 376,278 24,278" fill="url(#hero-gap)" opacity="0.92" />
      <line
        x1="40"
        y1={bandY}
        x2="360"
        y2={bandY}
        stroke="oklch(0.1 0.01 265)"
        stroke-width="3"
        stroke-dasharray="6 5"
      />
      <line
        x1="40"
        y1={bandY + 14}
        x2="360"
        y2={bandY + 14}
        stroke="oklch(0.98 0.01 90)"
        stroke-width="1"
        opacity="0.7"
      />
    </svg>
  {/if}
  <p class="scene-caption">
    Gap sketch · <span class="tabular">{backend === 'detecting' ? 'detecting' : backend}</span>
  </p>
</div>

<style>
  .scene {
    position: relative;
    margin: 0.5rem 0 1rem;
    overflow: hidden;
    background: oklch(0.12 0.01 265);
  }

  canvas,
  svg {
    display: block;
    width: 100%;
    height: clamp(220px, 32vh, 320px);
  }

  .scene-caption {
    position: absolute;
    left: 0.9rem;
    bottom: 0.7rem;
    margin: 0;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: color-mix(in oklch, var(--ink), transparent 35%);
    text-shadow: 0 1px 4px oklch(0 0 0 / 0.6);
  }
</style>
