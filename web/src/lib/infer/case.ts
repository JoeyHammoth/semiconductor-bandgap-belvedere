export interface InferNoise {
  adc_lsb_v: number;
  likelihood_sigma_v: number;
  voltage_sigma_v: number;
}

export interface InferAnchor {
  kind: string;
  label: string;
  note: string;
  source: string;
  value_ev: number;
}

export interface InferObservations {
  current_a: number;
  noise: InferNoise;
  synthetic: boolean;
  temperature_k: number[];
  voltage_v: number[];
}

export interface InferCase {
  schema_version: number;
  case_id: string;
  model: string;
  synthetic: boolean;
  description: string;
  anchor: InferAnchor;
  config_sha256: string;
  reference_sha256: string;
  seed: number;
  source_reference: string;
  noise: InferNoise;
  observations: InferObservations;
}

export interface ProductionSampler {
  chains: number;
  warmup: number;
  draws: number;
  seed: number;
  targetAccept: number;
  initialStep: number;
  meanTrajectorySteps: number;
  energyErrorLimit: number;
}

/** The production defaults used by the native reference tests. */
export const PRODUCTION_SAMPLER: ProductionSampler = {
  chains: 4,
  warmup: 5_000,
  draws: 4_000,
  seed: 0x6d63_6c6d,
  targetAccept: 0.995,
  initialStep: 0.4,
  meanTrajectorySteps: 16,
  energyErrorLimit: 0.1,
};

export function wasmBase(): string {
  // `/infer/` is one level below the app root, including any configured base path.
  const root = new URL('..', window.location.href);
  const glue = new URL('wasm/bgo_wasm.js', root);
  return new URL('..', glue).href.replace(/\/$/, '');
}

export async function loadInferCase(): Promise<InferCase> {
  const url = new URL('../data/m0-report-si.json', window.location.href);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`synthetic case fetch failed: ${response.status}`);
  const value = (await response.json()) as InferCase;
  if (
    value.schema_version !== 1 ||
    value.model !== 'M0' ||
    value.synthetic !== true ||
    value.observations.synthetic !== true ||
    value.observations.temperature_k.length !== value.observations.voltage_v.length ||
    value.observations.temperature_k.length < 2
  ) {
    throw new Error('synthetic case failed validation');
  }
  return value;
}
