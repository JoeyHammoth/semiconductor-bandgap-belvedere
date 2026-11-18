import type { Request, Response } from './protocol';

export interface EngineSuccess {
  ok: true;
  value?: number | string;
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

/**
 * A small pool of module workers, each with its own WASM instance.
 *
 * No SharedArrayBuffer: GitHub Pages cannot send COOP/COEP, so parallelism is
 * embarrassingly parallel workers (ADR 0005). Cancellation is termination.
 */
export class EnginePool {
  #workers: Worker[] = [];
  #pending = new Map<number, Pending>();
  #nextId = 1;
  #cursor = 0;
  #timeoutMs: number;
  #disposed = false;

  constructor(size: number, base: string, timeoutMs = 15_000) {
    this.#timeoutMs = timeoutMs;
    for (let i = 0; i < size; i += 1) {
      const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = (event: MessageEvent<Response>) => this.#handle(event.data);
      worker.onerror = (event) => {
        console.error('engine worker error', event.message);
      };
      const id = this.#nextId++;
      this.#pending.set(id, {
        resolve: () => undefined,
        timer: setTimeout(() => undefined, 0),
      });
      worker.postMessage({ id, kind: 'init', base } satisfies Request);
      this.#workers.push(worker);
    }
  }

  get size() {
    return this.#workers.length;
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

  #send(request: RequestBody): Promise<EngineResult> {
    if (this.#disposed) return Promise.resolve({ ok: false, error: 'engine disposed' });
    const worker = this.#workers[this.#cursor % this.#workers.length];
    this.#cursor += 1;
    const id = this.#nextId++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id);
        resolve({ ok: false, error: `timeout after ${this.#timeoutMs} ms` });
      }, this.#timeoutMs);
      this.#pending.set(id, { resolve, timer });
      worker.postMessage({ ...request, id } as Request);
    });
  }

  version() {
    return this.#send({ kind: 'version' });
  }

  varshni(eg0: number, alpha: number, beta: number, temperature: number) {
    return this.#send({ kind: 'varshni', eg0, alpha, beta, temperature });
  }

  async warm() {
    const results = await Promise.all(this.#workers.map(() => this.version()));
    return results.every((r) => r.ok);
  }

  dispose() {
    this.#disposed = true;
    for (const pending of this.#pending.values()) {
      clearTimeout(pending.timer);
      pending.resolve({ ok: false, error: 'engine disposed' });
    }
    this.#pending.clear();
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
