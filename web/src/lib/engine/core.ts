import type { CoreModule } from './protocol';

/**
 * Load the generated wasm-pack glue for the current base path.
 *
 * The glue and the binary are build artifacts, not source: `wasm-pack build
 * crates/bgo-wasm --target web --out-dir ../../web/static/wasm`. When the core
 * has not been built, this rejects and the UI reports an honest fallback rather
 * than inventing numbers.
 *
 * Compiling once on the main thread and transferring the `WebAssembly.Module`
 * to every worker is tracked in the Phase 0 WASM ticket.
 */
export async function loadCore(base: string): Promise<CoreModule> {
  const glue = `${base}/wasm/bgo_wasm.js`;
  const module = (await import(/* @vite-ignore */ glue)) as CoreModule;
  await module.default({ module_or_path: `${base}/wasm/bgo_wasm_bg.wasm` });
  return module;
}
