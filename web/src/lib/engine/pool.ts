import { compileCore } from './core';
import type { Request, Response, SamplerRequest, SamplerResult } from './protocol';

export interface EngineSuccess {
  ok: true;
  value?: number | string | SamplerResult;
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
    worker.postMessage(
      { id, kind: 'init', base: this.#base, module: this.#module ?? undefined } satisfies Request,
    );
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

  #send(request: RequestBody): Promise<EngineResult> {
    if (this.#disposed) return Promise.resolve({ ok: false, error: 'engine disposed' });
    if (this.#workers.length === 0) return Promise.resolve({ ok: false, error: 'no engine workers' });

    const worker = this.#workers[this.#cursor % this.#workers.length];
    this.#cursor += 1;
    const id = this.#nextId++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id);
        resolve({ ok: false, error: `timeout after ${this.#timeoutMs} ms` });
      }, this.#timeoutMs);
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

  sampler(request: SamplerRequest) {
    return this.#send({ kind: 'sampler', ...request });
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
