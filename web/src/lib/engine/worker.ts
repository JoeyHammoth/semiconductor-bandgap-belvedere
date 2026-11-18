/// <reference lib="webworker" />
import { loadCore } from './core';
import type { CoreModule, Request, Response } from './protocol';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

let core: CoreModule | null = null;
let base = '';

function reply(message: Response) {
  ctx.postMessage(message);
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
