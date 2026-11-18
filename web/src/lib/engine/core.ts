import type { CoreModule } from './protocol';

/**
 * Compile the core once on the main thread. The resulting module is posted to
 * every worker, so N workers do not fetch and compile the binary N times.
 * Returns null when the binary is unavailable; the caller then falls back to
 * per-worker loading (or reports an honest failure).
 */
export async function compileCore(base: string): Promise<WebAssembly.Module | null> {
  const url = `${base}/wasm/bgo_wasm_bg.wasm`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`core fetch failed: ${response.status}`);
  if (typeof WebAssembly.compileStreaming === 'function') {
    try {
      return await WebAssembly.compileStreaming(response.clone());
    } catch {
      // fall through to the ArrayBuffer path
    }
  }
  return WebAssembly.compile(await response.arrayBuffer());
}

/**
 * Load the generated wasm-pack glue for the current base path.
 *
 * The glue and the binary are build artifacts, not source: `wasm-pack build
 * crates/bgo-wasm --target web --out-dir ../../web/static/wasm`. When the core
 * has not been built, this rejects and the UI reports an honest fallback rather
 * than inventing numbers.
 */
export async function loadCore(base: string, module?: WebAssembly.Module): Promise<CoreModule> {
  const glue = `${base}/wasm/bgo_wasm.js`;
  const mod = (await import(/* @vite-ignore */ glue)) as CoreModule;

  if (module && typeof mod.initSync === 'function') {
    mod.initSync({ module });
    return mod;
  }

  await mod.default({ module_or_path: `${base}/wasm/bgo_wasm_bg.wasm` });
  return mod;
}
