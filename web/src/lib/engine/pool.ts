import { compileCore } from './core';
import type {
  DiagnosticsResult,
  FisherResult,
  NestedTraceResult,
  NpeRequest,
  NpeResult,
  PredictiveRequest,
  PredictiveResult,
  Request,
  Response,
  SamplerRequest,
  SamplerResult,
  SvgdComparisonRequest,
  SvgdComparisonResult,
  ThermalTwinResult,
} from './protocol';

export interface EngineSuccess {
  ok: true;
  value?:
    | number
    | string
    | SamplerResult
    | DiagnosticsResult
    | PredictiveResult
    | FisherResult
    | NestedTraceResult
    | NpeResult
    | SvgdComparisonResult
    | ThermalTwinResult;
}

export interface EngineFailure {
  ok: false;
  error: string;
}

export type EngineResult = EngineSuccess | EngineFailure;

interface Pending {
  resolve: (value: EngineResult) => void;
  timer: ReturnType<typeof setTimeout>;
}

/** Distribute `Omit` over the request union so each variant keeps its fields. */
type WithoutId<T> = T extends { id: number } ? Omit<T, 'id'> : never;
type RequestBody = WithoutId<Request>;

type WorkerFactory = () => Worker;

/**
 * A small pool of module workers, each with its own WASM instance.
 *
 * No SharedArrayBuffer: GitHub Pages cannot send COOP/COEP, so parallelism is
 * embarrassingly parallel workers (ADR 0005). The binary is compiled once on
 * the main thread and the `WebAssembly.Module` is posted to every worker;
 * cancellation is worker termination, followed by a clean respawn.
 */
export class EnginePool {
  #workers: Worker[] = [];
  #pending = new Map<number, Pending>();
  #nextId = 1;
  #cursor = 0;
  #timeoutMs: number;
  #disposed = false;
  #base: string;
  #size: number;
  #factory: WorkerFactory;
  #module: WebAssembly.Module | null = null;
  #initSent = false;
  #primed: Promise<void>;

  constructor(size: number, base: string, timeoutMs = 15_000, factory?: WorkerFactory) {
    this.#timeoutMs = timeoutMs;
    this.#base = base;
    this.#size = Math.max(1, size);
    this.#factory =
      factory ?? (() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }));
    for (let i = 0; i < this.#size; i += 1) this.#workers.push(this.#spawn());
    this.#primed = this.#prime();
  }

  get size() {
    return this.#workers.length;
  }

  get pendingCount() {
    return this.#pending.size;
  }

  /** Resolves once the core has been compiled once and init has been posted. */
  prime() {
    return this.#primed;
  }

  #spawn(): Worker {
    const worker = this.#factory();
    worker.onmessage = (event: MessageEvent<Response>) => this.#handle(event.data);
    worker.onerror = (event) => {
      console.error('engine worker error', event.message);
    };
    if (this.#initSent) this.#postInit(worker);
    return worker;
  }

  #postInit(worker: Worker) {
    const id = this.#nextId++;
    worker.postMessage({
      id,
      kind: 'init',
      base: this.#base,
      module: this.#module ?? undefined,
    } satisfies Request);
  }

  async #prime() {
    try {
      this.#module = await compileCore(this.#base);
    } catch {
      this.#module = null;
    }
    this.#initSent = true;
    for (const worker of this.#workers) this.#postInit(worker);
  }

  #handle(message: Response) {
    const pending = this.#pending.get(message.id);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.#pending.delete(message.id);
    if (message.ok) {
      pending.resolve({
        ok: true,
        value: 'value' in message ? message.value : undefined,
      });
    } else {
      pending.resolve({ ok: false, error: message.error });
    }
  }

  #failAll(error: string) {
    for (const pending of this.#pending.values()) {
      clearTimeout(pending.timer);
      pending.resolve({ ok: false, error });
    }
    this.#pending.clear();
  }

  #send(request: RequestBody, timeoutMs = this.#timeoutMs): Promise<EngineResult> {
    if (this.#disposed) return Promise.resolve({ ok: false, error: 'engine disposed' });
    if (this.#workers.length === 0)
      return Promise.resolve({ ok: false, error: 'no engine workers' });

    const worker = this.#workers[this.#cursor % this.#workers.length];
    this.#cursor += 1;
    const id = this.#nextId++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id);
        resolve({ ok: false, error: `timeout after ${timeoutMs} ms` });
      }, timeoutMs);
      this.#pending.set(id, { resolve, timer });

      // Register the request before the worker is ready so that cancelAll() can
      // reject it, then post once init has been sent to the pool.
      if (this.#initSent) {
        worker.postMessage({ ...request, id } as Request);
      } else {
        void this.#primed.then(() => {
          if (!this.#pending.has(id)) return;
          worker.postMessage({ ...request, id } as Request);
        });
      }
    });
  }

  version() {
    return this.#send({ kind: 'version' });
  }

  varshni(eg0: number, alpha: number, beta: number, temperature: number) {
    return this.#send({ kind: 'varshni', eg0, alpha, beta, temperature });
  }

  sampler(request: SamplerRequest, timeoutMs = this.#timeoutMs) {
    return this.#send({ kind: 'sampler', ...request }, timeoutMs);
  }

  posteriorPredictive(request: PredictiveRequest) {
    return this.#send({ kind: 'predictive', ...request }, 120_000);
  }

  fisherReport(request: {
    model: string;
    temperatureK: Float64Array;
    currentA: number;
    sigmaV: number;
    reported: Float64Array;
  }) {
    return this.#send({ kind: 'fisher', ...request }, 30_000);
  }

  nestedEvidence(livePoints: number, seed: number, timeoutMs = 120_000) {
    return this.#send({ kind: 'nested', livePoints, seed }, timeoutMs);
  }

  thermalTwin(seed: number, timeoutMs = 10_000) {
    return this.#send({ kind: 'thermal', seed }, timeoutMs);
  }

  npe(request: NpeRequest, timeoutMs = 10_000) {
    return this.#send({ kind: 'npe', ...request }, timeoutMs);
  }

  svgdComparison(request: SvgdComparisonRequest, timeoutMs = 120_000) {
    return this.#send({ kind: 'svgd', ...request }, timeoutMs);
  }

  /**
   * Run one chain per available worker, then combine the draws and compute
   * diagnostics in Rust. A pool of one remains correct, just sequential.
   * The returned draws keep the parameter-major layout of `sampler()`.
   */
  async samplerParallel(request: SamplerRequest): Promise<EngineResult> {
    const chains = request.config.chains;
    if (!Number.isInteger(chains) || chains < 1) {
      return { ok: false, error: 'chain count must be a positive integer' };
    }
    if (chains === 1) return this.sampler(request);

    const results = await Promise.all(
      Array.from({ length: chains }, (_, chain) =>
        this.sampler(
          {
            ...request,
            config: {
              ...request.config,
              chains: 1,
              seed: (request.config.seed + chain * 0x9e37_79b9) >>> 0,
            },
          },
          120_000,
        ),
      ),
    );

    const values: SamplerResult[] = [];
    for (const result of results) {
      if (!result.ok) return result;
      const value = result.value;
      if (
        typeof value !== 'object' ||
        value === null ||
        !('draws' in value) ||
        !('drawsPerParameter' in value)
      ) {
        return { ok: false, error: 'sampler worker returned no draws' };
      }
      values.push(value as SamplerResult);
    }

    const first = values[0];
    const parameterCount = first.parameterCount;
    const drawsPerChain = first.drawsPerParameter;
    const chainDrawsLength = parameterCount * drawsPerChain;
    for (const value of values) {
      if (
        value.parameterCount !== parameterCount ||
        value.drawsPerParameter !== drawsPerChain ||
        value.chainCount !== 1 ||
        value.draws.length !== chainDrawsLength ||
        value.energyErrorTrace.length !== drawsPerChain
      ) {
        return { ok: false, error: 'sampler workers returned inconsistent shapes' };
      }
    }

    const chainDraws = new Float64Array(chainDrawsLength * chains);
    const combinedDraws = new Float64Array(chainDrawsLength * chains);
    const energyErrorTrace = new Float64Array(drawsPerChain * chains);
    for (let chain = 0; chain < chains; chain += 1) {
      chainDraws.set(values[chain].draws, chain * chainDrawsLength);
      energyErrorTrace.set(values[chain].energyErrorTrace, chain * drawsPerChain);
      for (let parameter = 0; parameter < parameterCount; parameter += 1) {
        const source = chain * chainDrawsLength + parameter * drawsPerChain;
        const target = parameter * chains * drawsPerChain + chain * drawsPerChain;
        combinedDraws.set(chainDraws.subarray(source, source + drawsPerChain), target);
      }
    }

    const energyErrorMean = values.reduce((sum, value) => sum + value.energyErrorMean, 0) / chains;
    const divergences = values.reduce((sum, value) => sum + value.divergences, 0);
    const diagnosticResult = await this.#send(
      {
        kind: 'diagnostics',
        chains,
        parameterCount,
        drawsPerChain,
        draws: chainDraws,
        energyErrorMean,
        divergences,
      },
      120_000,
    );
    if (!diagnosticResult.ok) return diagnosticResult;
    const diagnostics = diagnosticResult.value as DiagnosticsResult;

    return {
      ok: true,
      value: {
        ...first,
        draws: combinedDraws,
        drawsPerParameter: drawsPerChain * chains,
        chainCount: chains,
        drawsPerChain,
        energyErrorTrace,
        rhat: diagnostics.rhat,
        essBulk: diagnostics.essBulk,
        rhatMax: diagnostics.rhatMax,
        essBulkMin: diagnostics.essBulkMin,
        energyErrorMean: diagnostics.energyErrorMean,
        divergences: diagnostics.divergences,
        passes: diagnostics.passes,
        acceptanceRate: values.reduce((sum, value) => sum + value.acceptanceRate, 0) / chains,
        stepSize: values.reduce((sum, value) => sum + value.stepSize, 0) / chains,
      },
    };
  }

  async warm() {
    const results = await Promise.all(this.#workers.map(() => this.version()));
    return results.every((r) => r.ok);
  }

  /**
   * Terminate every worker, reject in-flight requests, and respawn the pool.
   * The compiled module is reused, so a respawn does not refetch the binary.
   */
  cancelAll() {
    if (this.#disposed) return;
    this.#failAll('cancelled');
    for (const worker of this.#workers) worker.terminate();
    this.#workers = [];
    for (let i = 0; i < this.#size; i += 1) this.#workers.push(this.#spawn());
  }

  dispose() {
    this.#disposed = true;
    this.#failAll('engine disposed');
    for (const worker of this.#workers) worker.terminate();
    this.#workers = [];
  }
}

/** `min(hardwareConcurrency - 1, 4)`, and 2 on small screens. */
export function defaultPoolSize(): number {
  const cores = typeof navigator === 'undefined' ? 4 : navigator.hardwareConcurrency || 4;
  const size = Math.min(Math.max(cores - 1, 1), 4);
  const narrow = typeof window !== 'undefined' && window.innerWidth < 720;
  return narrow ? Math.min(size, 2) : size;
}
