/** Worker protocol. Mirrored by the exports in `crates/bgo-wasm`. */

export type Request =
  | { id: number; kind: 'init'; base: string }
  | { id: number; kind: 'version' }
  | {
      id: number;
      kind: 'varshni';
      eg0: number;
      alpha: number;
      beta: number;
      temperature: number;
    };

export type Response =
  | { id: number; ok: true; kind: 'ready' }
  | { id: number; ok: true; kind: 'version'; value: string }
  | { id: number; ok: true; kind: 'varshni'; value: number }
  | { id: number; ok: false; error: string };

export interface CoreModule {
  default: (init: unknown) => Promise<unknown>;
  core_version: () => string;
  varshni_gap: (eg0: number, alpha: number, beta: number, temperature: number) => number;
  ideal_diode_voltage: (
    current: number,
    temperature: number,
    lnB: number,
    eta: number,
    egEv: number,
  ) => number;
}
