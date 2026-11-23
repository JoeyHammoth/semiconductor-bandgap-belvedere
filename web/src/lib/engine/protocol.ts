/** Worker protocol. Mirrored by the exports in `crates/bgo-wasm`. */

export interface SamplerConfig {
  chains: number;
  warmup: number;
  draws: number;
  seed: number;
  targetAccept: number;
  initialStep: number;
  meanTrajectorySteps: number;
  energyErrorLimit: number;
}

export interface SamplerRequest {
  model: string;
  temperatureK: Float64Array;
  voltageV: Float64Array;
  currentA: number;
  sigmaV: number;
  config: SamplerConfig;
}

export interface SamplerResult {
  parameterNames: string[];
  /** Parameter-major flattened draws. */
  draws: Float64Array;
  parameterCount: number;
  drawsPerParameter: number;
  rhat: Float64Array;
  essBulk: Float64Array;
  rhatMax: number;
  essBulkMin: number;
  energyErrorMean: number;
  divergences: number;
  /** Fail-closed gate. Never render a converged badge when this is false. */
  passes: boolean;
  acceptanceRate: number;
  stepSize: number;
}

export type Request =
  | { id: number; kind: 'init'; base: string; module?: WebAssembly.Module }
  | { id: number; kind: 'version' }
  | {
      id: number;
      kind: 'varshni';
      eg0: number;
      alpha: number;
      beta: number;
      temperature: number;
    }
  | ({ id: number; kind: 'sampler' } & SamplerRequest);

export type Response =
  | { id: number; ok: true; kind: 'ready' }
  | { id: number; ok: true; kind: 'version'; value: string }
  | { id: number; ok: true; kind: 'varshni'; value: number }
  | { id: number; ok: true; kind: 'sampler'; value: SamplerResult }
  | { id: number; ok: false; error: string };

/** The wasm-bindgen class shape, including its snake_case getters. */
export interface WasmSamplerOutput {
  readonly parameter_names: string;
  readonly draws: Float64Array;
  readonly parameter_count: number;
  readonly draws_per_parameter: number;
  readonly rhat: Float64Array;
  readonly ess_bulk: Float64Array;
  readonly rhat_max: number;
  readonly ess_bulk_min: number;
  readonly energy_error_mean: number;
  readonly divergences: number;
  readonly passes: boolean;
  readonly acceptance_rate: number;
  readonly step_size: number;
  free: () => void;
}

export interface CoreModule {
  default: (init: unknown) => Promise<unknown>;
  initSync?: (init: { module: WebAssembly.Module }) => unknown;
  core_version: () => string;
  varshni_gap: (eg0: number, alpha: number, beta: number, temperature: number) => number;
  ideal_diode_voltage: (
    current: number,
    temperature: number,
    lnB: number,
    eta: number,
    egEv: number,
  ) => number;
  run_sampler: (
    model: string,
    temperatureK: Float64Array,
    voltageV: Float64Array,
    currentA: number,
    sigmaV: number,
    chains: number,
    warmup: number,
    draws: number,
    seed: number,
    targetAccept: number,
    initialStep: number,
    meanTrajectorySteps: number,
    energyErrorLimit: number,
  ) => WasmSamplerOutput;
}
