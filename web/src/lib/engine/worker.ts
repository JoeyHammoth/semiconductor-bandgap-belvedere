/// <reference lib="webworker" />
import { loadCore } from './core';
import type { CoreModule, DiagnosticsResult, Request, Response, SamplerResult } from './protocol';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

let core: CoreModule | null = null;
let base = '';

function reply(message: Response, transfer: Transferable[] = []) {
  ctx.postMessage(message, transfer);
}

let transferred: WebAssembly.Module | undefined;

async function ensureCore(): Promise<CoreModule> {
  if (!core) core = await loadCore(base, transferred);
  return core;
}

ctx.onmessage = async (event: MessageEvent<Request>) => {
  const request = event.data;
  try {
    if (request.kind === 'init') {
      base = request.base;
      transferred = request.module;
      core = null;
      core = await ensureCore();
      reply({ id: request.id, ok: true, kind: 'ready' });
      return;
    }

    const module = await ensureCore();

    switch (request.kind) {
      case 'version':
        reply({ id: request.id, ok: true, kind: 'version', value: module.core_version() });
        break;
      case 'varshni':
        reply({
          id: request.id,
          ok: true,
          kind: 'varshni',
          value: module.varshni_gap(
            request.eg0,
            request.alpha,
            request.beta,
            request.temperature,
          ),
        });
        break;
      case 'diagnostics': {
        const output = module.diagnose_chains(
          request.chains,
          request.parameterCount,
          request.drawsPerChain,
          request.draws,
          request.energyErrorMean,
          request.divergences,
        );
        try {
          const value: DiagnosticsResult = {
            rhat: output.rhat,
            essBulk: output.ess_bulk,
            rhatMax: output.rhat_max,
            essBulkMin: output.ess_bulk_min,
            energyErrorMean: output.energy_error_mean,
            divergences: output.divergences,
            passes: output.passes,
          };
          reply({ id: request.id, ok: true, kind: 'diagnostics', value }, [
            value.rhat.buffer as ArrayBuffer,
            value.essBulk.buffer as ArrayBuffer,
          ]);
        } finally {
          output.free();
        }
        break;
      }
      case 'sampler': {
        const output = module.run_sampler(
          request.model,
          request.temperatureK,
          request.voltageV,
          request.currentA,
          request.sigmaV,
          request.config.chains,
          request.config.warmup,
          request.config.draws,
          request.config.seed,
          request.config.targetAccept,
          request.config.initialStep,
          request.config.meanTrajectorySteps,
          request.config.energyErrorLimit,
        );
        try {
          const value: SamplerResult = {
            parameterNames: output.parameter_names.split(',').filter(Boolean),
            draws: output.draws,
            parameterCount: output.parameter_count,
            drawsPerParameter: output.draws_per_parameter,
            rhat: output.rhat,
            essBulk: output.ess_bulk,
            rhatMax: output.rhat_max,
            essBulkMin: output.ess_bulk_min,
            energyErrorMean: output.energy_error_mean,
            divergences: output.divergences,
            passes: output.passes,
            acceptanceRate: output.acceptance_rate,
            stepSize: output.step_size,
          };
          reply({ id: request.id, ok: true, kind: 'sampler', value }, [
            value.draws.buffer as ArrayBuffer,
            value.rhat.buffer as ArrayBuffer,
            value.essBulk.buffer as ArrayBuffer,
          ]);
        } finally {
          output.free();
        }
        break;
      }
      default: {
        const exhaustive: never = request;
        throw new Error(`unknown request: ${JSON.stringify(exhaustive)}`);
      }
    }
  } catch (error) {
    reply({
      id: request.id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
