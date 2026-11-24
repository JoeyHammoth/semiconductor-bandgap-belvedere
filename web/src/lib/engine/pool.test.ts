import { describe, expect, it } from 'vitest';
import { EnginePool } from './pool';

class FakeWorker {
  posted: any[] = [];
  terminated = false;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;

  postMessage(message: unknown) {
    this.posted.push(message);
  }

  terminate() {
    this.terminated = true;
  }

  respond(message: unknown) {
    this.onmessage?.({ data: message } as MessageEvent);
  }
}

function makePool(size = 2) {
  const workers: FakeWorker[] = [];
  const pool = new EnginePool(size, '/base', 1000, () => {
    const worker = new FakeWorker();
    workers.push(worker);
    return worker as unknown as Worker;
  });
  return { pool, workers };
}

function findRequest(worker: FakeWorker, kind: string) {
  return worker.posted.find((message) => message.kind === kind);
}

describe('EnginePool', () => {
  it('sends init once the core has been primed', async () => {
    const { pool, workers } = makePool();
    await pool.prime();
    expect(pool.size).toBe(2);
    expect(workers).toHaveLength(2);
    for (const worker of workers) {
      expect(findRequest(worker, 'init')).toMatchObject({ base: '/base', kind: 'init' });
    }
    expect(pool.pendingCount).toBe(0);
  });

  it('resolves a version request from a worker', async () => {
    const { pool, workers } = makePool();
    await pool.prime();
    const promise = pool.version();
    const worker = workers.find((candidate) => findRequest(candidate, 'version'));
    expect(worker).toBeDefined();
    const request = findRequest(worker!, 'version');
    worker!.respond({ id: request.id, ok: true, kind: 'version', value: '0.0.0' });
    await expect(promise).resolves.toEqual({ ok: true, value: '0.0.0' });
  });

  it('routes a sampler request without accepting a converged flag', async () => {
    const { pool, workers } = makePool();
    await pool.prime();
    const promise = pool.sampler({
      model: 'M0',
      temperatureK: new Float64Array([300, 320]),
      voltageV: new Float64Array([0.65, 0.63]),
      currentA: 3e-4,
      sigmaV: 1.25e-4,
      config: {
        chains: 4,
        warmup: 5000,
        draws: 4000,
        seed: 7,
        targetAccept: 0.995,
        initialStep: 0.4,
        meanTrajectorySteps: 16,
        energyErrorLimit: 0.1,
      },
    });
    const worker = workers.find((candidate) => findRequest(candidate, 'sampler'))!;
    const request = findRequest(worker, 'sampler');
    const value = {
      parameterNames: ['eg_ev', 'ln_b'],
      draws: new Float64Array([1, 2]),
      parameterCount: 2,
      drawsPerParameter: 1,
      chainCount: 1,
      drawsPerChain: 1,
      energyErrorTrace: new Float64Array([0.01]),
      rhat: new Float64Array([1.01, 1.02]),
      essBulk: new Float64Array([399, 400]),
      rhatMax: 1.02,
      essBulkMin: 399,
      energyErrorMean: 0.01,
      divergences: 0,
      passes: false,
      acceptanceRate: 0.99,
      stepSize: 0.1,
    };
    worker.respond({ id: request.id, ok: true, kind: 'sampler', value });
    await expect(promise).resolves.toEqual({ ok: true, value });
  });

  it('runs chains in parallel and computes combined diagnostics', async () => {
    const { pool, workers } = makePool(2);
    await pool.prime();
    const promise = pool.samplerParallel({
      model: 'M0',
      temperatureK: new Float64Array([300, 320]),
      voltageV: new Float64Array([0.65, 0.63]),
      currentA: 3e-4,
      sigmaV: 1.25e-4,
      config: {
        chains: 2,
        warmup: 20,
        draws: 2,
        seed: 7,
        targetAccept: 0.995,
        initialStep: 0.4,
        meanTrajectorySteps: 16,
        energyErrorLimit: 0.1,
      },
    });

    const requests = workers
      .flatMap((worker) => worker.posted)
      .filter((message) => message.kind === 'sampler');
    expect(requests).toHaveLength(2);
    expect(requests.every((request) => request.config.chains === 1)).toBe(true);
    expect(new Set(requests.map((request) => request.config.seed)).size).toBe(2);

    for (const [index, request] of requests.entries()) {
      const worker = workers.find((candidate) => candidate.posted.includes(request))!;
      worker.respond({
        id: request.id,
        ok: true,
        kind: 'sampler',
        value: {
          parameterNames: ['eg_ev'],
          draws: new Float64Array(index === 0 ? [1, 2] : [3, 4]),
          parameterCount: 1,
          drawsPerParameter: 2,
          chainCount: 1,
          drawsPerChain: 2,
          energyErrorTrace: new Float64Array([0.01, 0.02]),
          rhat: new Float64Array([Number.NaN]),
          essBulk: new Float64Array([0]),
          rhatMax: Number.NaN,
          essBulkMin: 0,
          energyErrorMean: index === 0 ? 0.01 : 0.03,
          divergences: 0,
          passes: false,
          acceptanceRate: 0.99,
          stepSize: 0.1,
        },
      });
    }

    await new Promise((resolve) => setTimeout(resolve, 0));
    const diagnosticRequest = workers
      .flatMap((worker) => worker.posted)
      .find((message) => message.kind === 'diagnostics');
    expect(diagnosticRequest).toBeDefined();
    expect(Array.from(diagnosticRequest.draws)).toEqual([1, 2, 3, 4]);
    expect(diagnosticRequest.energyErrorMean).toBeCloseTo(0.02);

    const diagnosticWorker = workers.find((candidate) =>
      candidate.posted.includes(diagnosticRequest),
    )!;
    diagnosticWorker.respond({
      id: diagnosticRequest.id,
      ok: true,
      kind: 'diagnostics',
      value: {
        rhat: new Float64Array([1.001]),
        essBulk: new Float64Array([500]),
        rhatMax: 1.001,
        essBulkMin: 500,
        energyErrorMean: 0.02,
        divergences: 0,
        passes: true,
      },
    });

    const result = await promise;
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const value = result.value as {
      draws: Float64Array;
      drawsPerParameter: number;
      energyErrorTrace: Float64Array;
      passes: boolean;
    };
    expect(Array.from(value.draws)).toEqual([1, 2, 3, 4]);
    expect(value.drawsPerParameter).toBe(4);
    expect(value.energyErrorTrace).toEqual(new Float64Array([0.01, 0.02, 0.01, 0.02]));
    expect(value.passes).toBe(true);
  });

  it('terminates in-flight work and respawns on cancelAll', async () => {
    const { pool, workers } = makePool();
    await pool.prime();
    const promise = pool.version();
    pool.cancelAll();

    await expect(promise).resolves.toEqual({ ok: false, error: 'cancelled' });
    expect(workers.slice(0, 2).every((worker) => worker.terminated)).toBe(true);
    expect(pool.size).toBe(2);
    expect(pool.pendingCount).toBe(0);

    const fresh = workers.slice(2);
    expect(fresh).toHaveLength(2);
    expect(fresh.every((worker) => findRequest(worker, 'init'))).toBe(true);

    const next = pool.version();
    const worker = fresh.find((candidate) => findRequest(candidate, 'version'))!;
    worker.respond({ id: findRequest(worker, 'version').id, ok: true, kind: 'version', value: '0.0.1' });
    await expect(next).resolves.toEqual({ ok: true, value: '0.0.1' });
  });

  it('rejects pending work and terminates everything on dispose', async () => {
    const { pool, workers } = makePool();
    await pool.prime();
    const promise = pool.version();
    pool.dispose();

    await expect(promise).resolves.toEqual({ ok: false, error: 'engine disposed' });
    expect(pool.pendingCount).toBe(0);
    expect(pool.size).toBe(0);
    expect(workers.every((worker) => worker.terminated)).toBe(true);
    await expect(pool.version()).resolves.toEqual({ ok: false, error: 'engine disposed' });
  });
});
