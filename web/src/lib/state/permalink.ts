/** Shareable view state, carried in the URL hash. No server, no accounts. */

export interface ViewState {
  model: string;
  dataset: string | null;
  temperature: number;
  seed: number;
}

export const DEFAULT_STATE: ViewState = {
  model: 'M0',
  dataset: null,
  temperature: 300,
  seed: 1,
};

const VERSION = 1;

function toBase64Url(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(value: string): Uint8Array {
  const padded = value.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(value.length / 4) * 4, '=');
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export function encodeState(state: ViewState): string {
  const payload = { v: VERSION, ...state };
  return toBase64Url(new TextEncoder().encode(JSON.stringify(payload)));
}

export function decodeState(encoded: string): ViewState {
  try {
    const payload = JSON.parse(new TextDecoder().decode(fromBase64Url(encoded))) as Partial<ViewState> & {
      v?: number;
    };
    if (payload.v !== VERSION) return { ...DEFAULT_STATE };
    return {
      model: typeof payload.model === 'string' ? payload.model : DEFAULT_STATE.model,
      dataset: typeof payload.dataset === 'string' || payload.dataset === null ? payload.dataset : null,
      temperature:
        typeof payload.temperature === 'number' ? payload.temperature : DEFAULT_STATE.temperature,
      seed: typeof payload.seed === 'number' ? payload.seed : DEFAULT_STATE.seed,
    };
  } catch {
    return { ...DEFAULT_STATE };
  }
}
