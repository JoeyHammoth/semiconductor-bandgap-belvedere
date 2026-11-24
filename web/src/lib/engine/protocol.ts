/** Worker protocol. Mirrored by the exports in `crates/bgo-wasm`. */

export interface SamplerConfig {
  chains: number;
  warmup: number;
  draws: number;
  seed: number;
  /** Shared base seed for the covariance pilot; workers keep their own `seed`. */
  pilotSeed: number;
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
  chainCount: number;
  drawsPerChain: number;
  /** Chain-major absolute energy error, one value per post-warmup draw. */
  energyErrorTrace: Float64Array;
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

export interface DiagnosticsRequest {
  chains: number;
  parameterCount: number;
  drawsPerChain: number;
  /** Chain-major; within each chain, parameter-major. */
  draws: Float64Array;
  energyErrorMean: number;
  divergences: number;
}

export interface PredictiveRequest {
  model: string;
  temperatureK: Float64Array;
  currentA: number;
  sigmaV: number;
  /** Parameter-major draws, matching `SamplerResult.draws`. */
  draws: Float64Array;
  parameterCount: number;
  drawsPerParameter: number;
  seed: number;
  lineCount: number;
}

export interface PredictiveResult {
  mean: Float64Array;
  q05: Float64Array;
  q50: Float64Array;
  q95: Float64Array;
  /** Line-major noiseless posterior curves: `lineCount * pointCount`. */
  lines: Float64Array;
  lineCount: number;
  pointCount: number;
}

export interface DiagnosticsResult {
  rhat: Float64Array;
  essBulk: Float64Array;
  rhatMax: number;
  essBulkMin: number;
  energyErrorMean: number;
  divergences: number;
  passes: boolean;
}

export interface FisherResult {
  parameterNames: string[];
  parameterCount: number;
  /** Row-major covariance in reported-parameter coordinates. */
  covariance: Float64Array;
  /** Descending: stiffest first, sloppiest last. */
  eigenvalues: Float64Array;
  /** Eigenvectors in eigenvalue order, flattened vector-major. */
  eigenvectors: Float64Array;
  conditionNumber: number;
  stiffestIndex: number;
  sloppiestIndex: number;
  /** `[semi-major, semi-minor, angle]` for the 1-sigma ellipse. */
  ellipse1: Float64Array;
  /** `[semi-major, semi-minor, angle]` for the 2-sigma ellipse. */
  ellipse2: Float64Array;
}

export interface NestedTraceResult {
  logEvidence: number;
  logEvidenceError: number;
  expectedLogEvidence: number;
  information: number;
  iterations: number;
  converged: boolean;
  livePoints: number;
  frameCount: number;
  parameterNames: string[];
  eventIterations: Uint32Array;
  eventLogEvidence: Float64Array;
  eventThresholds: Float64Array;
  eventLogPriorVolume: Float64Array;
  /** Frame-major, point-major, parameter-major physical positions. */
  livePositions: Float64Array;
  priorLow: Float64Array;
  priorHigh: Float64Array;
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
  | ({ id: number; kind: 'sampler' } & SamplerRequest)
  | ({ id: number; kind: 'predictive' } & PredictiveRequest)
  | ({ id: number; kind: 'diagnostics' } & DiagnosticsRequest)
  | {
      id: number;
      kind: 'fisher';
      model: string;
      temperatureK: Float64Array;
      currentA: number;
      sigmaV: number;
      reported: Float64Array;
    }
  | { id: number; kind: 'nested'; livePoints: number; seed: number };

export type Response =
  | { id: number; ok: true; kind: 'ready' }
  | { id: number; ok: true; kind: 'version'; value: string }
  | { id: number; ok: true; kind: 'varshni'; value: number }
  | { id: number; ok: true; kind: 'sampler'; value: SamplerResult }
  | { id: number; ok: true; kind: 'predictive'; value: PredictiveResult }
  | { id: number; ok: true; kind: 'diagnostics'; value: DiagnosticsResult }
  | { id: number; ok: true; kind: 'fisher'; value: FisherResult }
  | { id: number; ok: true; kind: 'nested'; value: NestedTraceResult }
  | { id: number; ok: false; error: string };

/** The wasm-bindgen class shape, including its snake_case getters. */
export interface WasmSamplerOutput {
  readonly parameter_names: string;
  readonly draws: Float64Array;
  readonly parameter_count: number;
  readonly draws_per_parameter: number;
  readonly chain_count: number;
  readonly draws_per_chain: number;
  readonly energy_error_trace: Float64Array;
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

export interface WasmPredictiveOutput {
  readonly mean: Float64Array;
  readonly q05: Float64Array;
  readonly q50: Float64Array;
  readonly q95: Float64Array;
  readonly lines: Float64Array;
  readonly line_count: number;
  readonly point_count: number;
  free: () => void;
}

export interface WasmDiagnosticsOutput {
  readonly rhat: Float64Array;
  readonly ess_bulk: Float64Array;
  readonly rhat_max: number;
  readonly ess_bulk_min: number;
  readonly energy_error_mean: number;
  readonly divergences: number;
  readonly passes: boolean;
  free: () => void;
}

export interface WasmFisherOutput {
  readonly parameter_names: string;
  readonly parameter_count: number;
  readonly covariance: Float64Array;
  readonly eigenvalues: Float64Array;
  readonly eigenvectors: Float64Array;
  readonly condition_number: number;
  readonly stiffest_index: number;
  readonly sloppiest_index: number;
  readonly ellipse_1: Float64Array;
  readonly ellipse_2: Float64Array;
  free: () => void;
}

export interface WasmNestedTraceOutput {
  readonly log_evidence: number;
  readonly log_evidence_error: number;
  readonly expected_log_evidence: number;
  readonly information: number;
  readonly iterations: number;
  readonly converged: boolean;
  readonly live_points: number;
  readonly frame_count: number;
  readonly parameter_names: string;
  readonly event_iterations: Uint32Array;
  readonly event_log_evidence: Float64Array;
  readonly event_thresholds: Float64Array;
  readonly event_log_prior_volume: Float64Array;
  readonly live_positions: Float64Array;
  readonly prior_low: Float64Array;
  readonly prior_high: Float64Array;
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
  diagnose_chains: (
    chains: number,
    parameterCount: number,
    drawsPerChain: number,
    draws: Float64Array,
    energyErrorMean: number,
    divergences: number,
  ) => WasmDiagnosticsOutput;
  posterior_predictive: (
    model: string,
    temperatureK: Float64Array,
    currentA: number,
    sigmaV: number,
    draws: Float64Array,
    parameterCount: number,
    drawsPerParameter: number,
    seed: number,
    lineCount: number,
  ) => WasmPredictiveOutput;
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
    pilotSeed: number,
    targetAccept: number,
    initialStep: number,
    meanTrajectorySteps: number,
    energyErrorLimit: number,
  ) => WasmSamplerOutput;
  fisher_report: (
    model: string,
    temperatureK: Float64Array,
    currentA: number,
    sigmaV: number,
    reported: Float64Array,
  ) => WasmFisherOutput;
  run_nested_gaussian: (livePoints: number, seed: number) => WasmNestedTraceOutput;
}
