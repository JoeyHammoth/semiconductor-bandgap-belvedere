/// <reference lib="webworker" />
import { loadCore } from './core';
import type {
  CoreModule,
  DiagnosticsResult,
  FisherResult,
  NestedTraceResult,
  PredictiveResult,
  Request,
  Response,
  SamplerResult,
  SvgdComparisonResult,
} from './protocol';

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
          value: module.varshni_gap(request.eg0, request.alpha, request.beta, request.temperature),
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
      case 'predictive': {
        const output = module.posterior_predictive(
          request.model,
          request.temperatureK,
          request.currentA,
          request.sigmaV,
          request.draws,
          request.parameterCount,
          request.drawsPerParameter,
          request.seed,
          request.lineCount,
        );
        try {
          const value: PredictiveResult = {
            mean: output.mean,
            q05: output.q05,
            q50: output.q50,
            q95: output.q95,
            lines: output.lines,
            lineCount: output.line_count,
            pointCount: output.point_count,
          };
          reply({ id: request.id, ok: true, kind: 'predictive', value }, [
            value.mean.buffer as ArrayBuffer,
            value.q05.buffer as ArrayBuffer,
            value.q50.buffer as ArrayBuffer,
            value.q95.buffer as ArrayBuffer,
            value.lines.buffer as ArrayBuffer,
          ]);
        } finally {
          output.free();
        }
        break;
      }
      case 'fisher': {
        const output = module.fisher_report(
          request.model,
          request.temperatureK,
          request.currentA,
          request.sigmaV,
          request.reported,
        );
        try {
          const value: FisherResult = {
            parameterNames: output.parameter_names.split(',').filter(Boolean),
            parameterCount: output.parameter_count,
            covariance: output.covariance,
            eigenvalues: output.eigenvalues,
            eigenvectors: output.eigenvectors,
            conditionNumber: output.condition_number,
            stiffestIndex: output.stiffest_index,
            sloppiestIndex: output.sloppiest_index,
            ellipse1: output.ellipse_1,
            ellipse2: output.ellipse_2,
          };
          reply({ id: request.id, ok: true, kind: 'fisher', value }, [
            value.covariance.buffer as ArrayBuffer,
            value.eigenvalues.buffer as ArrayBuffer,
            value.eigenvectors.buffer as ArrayBuffer,
            value.ellipse1.buffer as ArrayBuffer,
            value.ellipse2.buffer as ArrayBuffer,
          ]);
        } finally {
          output.free();
        }
        break;
      }
      case 'nested': {
        const output = module.run_nested_gaussian(request.livePoints, request.seed);
        try {
          const value: NestedTraceResult = {
            logEvidence: output.log_evidence,
            logEvidenceError: output.log_evidence_error,
            expectedLogEvidence: output.expected_log_evidence,
            information: output.information,
            iterations: output.iterations,
            converged: output.converged,
            livePoints: output.live_points,
            frameCount: output.frame_count,
            parameterNames: output.parameter_names.split(',').filter(Boolean),
            eventIterations: output.event_iterations,
            eventLogEvidence: output.event_log_evidence,
            eventThresholds: output.event_thresholds,
            eventLogPriorVolume: output.event_log_prior_volume,
            livePositions: output.live_positions,
            priorLow: output.prior_low,
            priorHigh: output.prior_high,
          };
          reply({ id: request.id, ok: true, kind: 'nested', value }, [
            value.eventIterations.buffer as ArrayBuffer,
            value.eventLogEvidence.buffer as ArrayBuffer,
            value.eventThresholds.buffer as ArrayBuffer,
            value.eventLogPriorVolume.buffer as ArrayBuffer,
            value.livePositions.buffer as ArrayBuffer,
            value.priorLow.buffer as ArrayBuffer,
            value.priorHigh.buffer as ArrayBuffer,
          ]);
        } finally {
          output.free();
        }
        break;
      }
      case 'svgd': {
        const output = module.run_svgd_comparison(
          request.model,
          request.temperatureK,
          request.voltageV,
          request.currentA,
          request.sigmaV,
          request.mclmcDraws,
          request.parameterCount,
          request.drawsPerParameter,
          request.particles,
          request.iterations,
          request.stepSize,
          request.bandwidth,
          request.seed,
          request.traceFrames,
        );
        try {
          const value: SvgdComparisonResult = {
            parameterNames: output.parameter_names.split(',').filter(Boolean),
            parameterCount: output.parameter_count,
            particleCount: output.particle_count,
            frameCount: output.frame_count,
            iterations: output.iterations,
            trajectory: output.trajectory,
            particles: output.particles,
            svgdMean: output.svgd_mean,
            svgdVariance: output.svgd_variance,
            mclmcMean: output.mclmc_mean,
            mclmcVariance: output.mclmc_variance,
            mclmcParticles: output.mclmc_particles,
            mclmcParticleCount: output.mclmc_particle_count,
            w2Distance: output.w2_distance,
            w2BootstrapSe: output.w2_bootstrap_se,
            bandwidth: output.bandwidth,
            stepSize: output.step_size,
            finalLogpMean: output.final_logp_mean,
          };
          reply({ id: request.id, ok: true, kind: 'svgd', value }, [
            value.trajectory.buffer as ArrayBuffer,
            value.particles.buffer as ArrayBuffer,
            value.svgdMean.buffer as ArrayBuffer,
            value.svgdVariance.buffer as ArrayBuffer,
            value.mclmcMean.buffer as ArrayBuffer,
            value.mclmcVariance.buffer as ArrayBuffer,
            value.mclmcParticles.buffer as ArrayBuffer,
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
          request.config.pilotSeed,
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
            chainCount: output.chain_count,
            drawsPerChain: output.draws_per_chain,
            energyErrorTrace: output.energy_error_trace,
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
            value.energyErrorTrace.buffer as ArrayBuffer,
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
