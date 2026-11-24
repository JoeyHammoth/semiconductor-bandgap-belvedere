/** Resolve the generated WASM directory from a route one level below the app root. */
export function wasmBase(): string {
  const root = new URL('..', window.location.href);
  const glue = new URL('wasm/bgo_wasm.js', root);
  return new URL('..', glue).href.replace(/\/$/, '');
}
