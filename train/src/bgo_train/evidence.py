"""Generate the S10 evidence ledger from the committed synthetic M1 case.

Run from ``train/`` with::

    uv run --with dynesty python -m bgo_train.evidence

Dynesty supplies ``ln Z`` and ``H``. NumPyro NUTS supplies posterior draws for
PSIS-LOO, because a nested-sampling equal-weight resample can collapse onto one
point when the likelihood is much narrower than the prior. That collapse is
reported through the Pareto-k flag rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import arviz as az
import dynesty
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from scipy.stats import norm, truncnorm

jax.config.update("jax_enable_x64", True)

K_BOLTZMANN = 1.380649e-23
ELEMENTARY_CHARGE = 1.602176634e-19
REFERENCE_TEMPERATURE_K = 300.0
NLIVE = 120
DLOGZ = 0.5
MAXITER = 20_000
NUTS_CHAINS = 4
NUTS_WARMUP = 3_000
NUTS_DRAWS = 2_000
TRAIN_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = TRAIN_ROOT.parent / ".scratch" / "evidence-cache.json"
DEFAULT_REFERENCE = TRAIN_ROOT / "references" / "m1.json"
DEFAULT_OUTPUT = TRAIN_ROOT.parent / "web" / "static" / "data" / "evidence-ledger.json"


def _truncated(low: float, high: float, loc: float, scale: float, u: float) -> float:
    return float(
        truncnorm.ppf(
            np.clip(u, 1.0e-12, 1.0 - 1.0e-12),
            (low - loc) / scale,
            (high - loc) / scale,
            loc=loc,
            scale=scale,
        )
    )


def _forward(model: str, theta: np.ndarray, temperature: np.ndarray, current: float) -> np.ndarray:
    vt = K_BOLTZMANN * temperature / ELEMENTARY_CHARGE
    log_current = math.log(current)
    if model == "M0":
        eg, ln_b = theta
        log_argument = log_current - (ln_b - eg / vt)
    else:
        eg, ln_b_ref, eta = theta
        log_argument = log_current - (
            ln_b_ref
            + eta * (np.log(temperature) - math.log(REFERENCE_TEMPERATURE_K))
            - eg / vt
        )
    return vt * np.logaddexp(0.0, log_argument)


def _pointwise(model: str, theta: np.ndarray, data: dict[str, Any]) -> np.ndarray:
    predicted = _forward(
        model,
        theta,
        np.asarray(data["temperature_k"], dtype=np.float64),
        float(data["current_a"]),
    )
    sigma = float(data["noise"]["likelihood_sigma_v"])
    residual = (np.asarray(data["voltage_v"]) - predicted) / sigma
    return -0.5 * residual * residual - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def _loglike(model: str, theta: np.ndarray, data: dict[str, Any]) -> float:
    return float(np.sum(_pointwise(model, theta, data)))


def _prior_transform(model: str, variant: str, unit: np.ndarray) -> np.ndarray:
    eg = _truncated(0.05, 3.5, 0.9, 0.35, float(unit[0]))
    ln_b_ref = float(norm.ppf(np.clip(unit[1], 1.0e-12, 1.0 - 1.0e-12), 8.0, 8.0))
    if model == "M0":
        return np.asarray([eg, ln_b_ref])
    scale = 1.0 if variant == "weak" else 0.25
    eta = _truncated(1.5, 5.0, 3.0, scale, float(unit[2]))
    return np.asarray([eg, ln_b_ref, eta])


def _cache_load() -> dict[str, Any]:
    if not CACHE_PATH.is_file():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _cache_store(key: str, value: dict[str, Any]) -> None:
    cache = _cache_load()
    cache[key] = value
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_evidence(
    model: str, variant: str, data: dict[str, Any], seed: int
) -> dict[str, Any]:
    key = f"evidence:v2:{model}:{variant}:{seed}:{NLIVE}:{DLOGZ}"
    cached = _cache_load().get(key)
    if cached is not None:
        print(f"cached evidence {model} ({variant}): ln Z = {cached['log_evidence']:.3f}", flush=True)
        return cached
    dimension = 2 if model == "M0" else 3
    print(f"starting evidence {model} ({variant})...", flush=True)
    sampler = dynesty.NestedSampler(
        lambda theta: _loglike(model, theta, data),
        lambda unit: _prior_transform(model, variant, unit),
        dimension,
        nlive=NLIVE,
        bound="multi",
        sample="rslice",
        rstate=np.random.default_rng(seed),
    )
    sampler.run_nested(dlogz=DLOGZ, maxiter=MAXITER, print_progress=False)
    result = sampler.results
    log_evidence = float(result.logz[-1])
    log_evidence_error = float(result.logzerr[-1])
    weights = np.exp(result.logwt - log_evidence)
    weights /= weights.sum()
    information = float(np.sum(weights * (result.logl - log_evidence)))
    print(f"finished evidence {model} ({variant}): ln Z = {log_evidence:.3f}", flush=True)
    entry = {
        "id": model,
        "prior_variant": variant,
        "parameter_count": dimension,
        "log_evidence": log_evidence,
        "log_evidence_error": log_evidence_error,
        "information": information,
    }
    _cache_store(key, entry)
    return entry


def _numpyro_model(model: str, variant: str, data: dict[str, Any]):
    temperature = jnp.asarray(data["temperature_k"], dtype=jnp.float64)
    voltage = jnp.asarray(data["voltage_v"], dtype=jnp.float64)
    sigma = float(data["noise"]["likelihood_sigma_v"])
    current = float(data["current_a"])
    vt = K_BOLTZMANN * temperature / ELEMENTARY_CHARGE
    log_current = math.log(current)

    def model_fn():
        eg = numpyro.sample(
            "eg_ev",
            dist.TruncatedNormal(0.9, 0.35, low=0.05, high=3.5),
        )
        if model == "M0":
            ln_b = numpyro.sample("ln_b", dist.Normal(8.0, 8.0))
            log_argument = log_current - (ln_b - eg / vt)
        else:
            ln_b_ref = numpyro.sample("ln_b_ref", dist.Normal(8.0, 8.0))
            scale = 1.0 if variant == "weak" else 0.25
            eta = numpyro.sample(
                "eta",
                dist.TruncatedNormal(3.0, scale, low=1.5, high=5.0),
            )
            log_argument = log_current - (
                ln_b_ref
                + eta * (jnp.log(temperature) - math.log(REFERENCE_TEMPERATURE_K))
                - eg / vt
            )
        predicted = vt * jnp.logaddexp(0.0, log_argument)
        numpyro.sample("obs", dist.Normal(predicted, sigma), obs=voltage)

    return model_fn


def _run_loo(
    model: str,
    variant: str,
    data: dict[str, Any],
    seed: int,
    init: dict[str, float] | None = None,
) -> dict[str, Any]:
    key = f"loo:v2:{model}:{variant}:{seed}:{NUTS_CHAINS}:{NUTS_WARMUP}:{NUTS_DRAWS}"
    cached = _cache_load().get(key)
    if cached is not None:
        print(f"cached NUTS/LOO {model} ({variant}): elpd = {cached['elpd_loo']:.1f}", flush=True)
        return cached
    print(f"starting NUTS/LOO {model} ({variant})...", flush=True)
    nuts_kwargs: dict[str, Any] = {
        "target_accept_prob": 0.99,
        "max_tree_depth": 12,
        "dense_mass": True,
    }
    if init is not None:
        nuts_kwargs["init_strategy"] = init_to_value(values=init)
    kernel = NUTS(_numpyro_model(model, variant, data), **nuts_kwargs)
    mcmc = MCMC(
        kernel,
        num_warmup=NUTS_WARMUP,
        num_samples=NUTS_DRAWS,
        num_chains=NUTS_CHAINS,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed))
    samples = mcmc.get_samples(group_by_chain=True)
    draws = {name: np.asarray(values, dtype=np.float64) for name, values in samples.items()}
    chains = draws["eg_ev"].shape[0]
    draws_per_chain = draws["eg_ev"].shape[1]
    pointwise = np.empty((chains, draws_per_chain, len(data["temperature_k"])))
    for chain in range(chains):
        for draw in range(draws_per_chain):
            if model == "M0":
                theta = np.asarray([draws["eg_ev"][chain, draw], draws["ln_b"][chain, draw]])
            else:
                theta = np.asarray(
                    [
                        draws["eg_ev"][chain, draw],
                        draws["ln_b_ref"][chain, draw],
                        draws["eta"][chain, draw],
                    ]
                )
            pointwise[chain, draw] = _pointwise(model, theta, data)
    idata = az.from_dict(
        {"posterior": draws, "log_likelihood": {"obs": pointwise}},
        dims={"obs": ["observation"]},
    )
    loo = az.loo(idata, pointwise=True, var_name="obs")
    pareto_k = float(np.max(np.asarray(loo.pareto_k.values)))
    divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    print(
        f"finished NUTS/LOO {model} ({variant}): elpd = {float(loo.elpd):.1f}, "
        f"max k = {pareto_k:.2f}, divergences = {divergences}",
        flush=True,
    )
    entry = {
        "elpd_loo": float(loo.elpd),
        "elpd_loo_se": float(loo.se),
        "p_loo": float(loo.p),
        "pareto_k_max": pareto_k,
        "loo_reliable": bool(pareto_k < 0.7 and divergences == 0),
        "nuts_divergences": divergences,
        "nuts_chains": NUTS_CHAINS,
        "nuts_warmup": NUTS_WARMUP,
        "nuts_draws": NUTS_DRAWS,
    }
    _cache_store(key, entry)
    return entry


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args(argv)

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    data = reference["observations"]
    if reference.get("case", {}).get("synthetic") is not True or data.get("synthetic") is not True:
        raise ValueError("the evidence ledger only accepts a labelled synthetic case")

    m0 = _run_evidence("M0", "weak", data, args.seed)
    m0.update(_run_loo("M0", "weak", data, args.seed + 1))
    reference_mean = reference["reference"]["mean"]
    m1_init = {
        "eg_ev": float(reference_mean["eg_ev"]),
        "ln_b_ref": float(
            reference_mean["ln_b"]
            + reference_mean["eta"] * math.log(REFERENCE_TEMPERATURE_K)
        ),
        "eta": float(reference_mean["eta"]),
    }
    m1_weak = _run_evidence("M1", "weak", data, args.seed + 10)
    m1_weak.update(_run_loo("M1", "weak", data, args.seed + 11, init=m1_init))
    m1_tight = _run_evidence("M1", "tight", data, args.seed + 20)

    delta = m1_tight["log_evidence"] - m1_weak["log_evidence"]
    prior_sensitive = abs(delta) > 0.5
    m1 = dict(m1_weak)
    m1["prior_sensitivity"] = {
        "weak_log_evidence": m1_weak["log_evidence"],
        "tight_log_evidence": m1_tight["log_evidence"],
        "delta_log_evidence": delta,
        "flag": bool(prior_sensitive),
        "description": "eta prior scale 1.0 versus 0.25, both truncated to [1.5, 5]",
    }

    delta_log_evidence = m1["log_evidence"] - m0["log_evidence"]
    log10_bayes_factor = delta_log_evidence / math.log(10.0)
    elpd_difference = m1["elpd_loo"] - m0["elpd_loo"]
    se_difference = math.hypot(m1["elpd_loo_se"], m0["elpd_loo_se"])
    direction = "M1" if delta_log_evidence > 0.0 else "M0"
    strength = (
        "overwhelming"
        if abs(log10_bayes_factor) >= 2.0
        else "strong"
        if abs(log10_bayes_factor) >= 1.0
        else "moderate"
        if abs(log10_bayes_factor) >= 0.5
        else "inconclusive"
    )
    unreliable = []
    for compared in (m0, m1):
        if not compared["loo_reliable"]:
            reason = f"{compared['id']} max Pareto k {compared['pareto_k_max']:.2f}"
            if compared["nuts_divergences"]:
                reason += f" and {compared['nuts_divergences']} NUTS divergences"
            unreliable.append(reason)
    if elpd_difference > 0.0:
        loo_direction = "points in the same direction"
    else:
        loo_direction = "points the other way"
    if unreliable:
        verb = "fails" if len(unreliable) == 1 else "fail"
        loo_text = (
            f"PSIS-LOO {loo_direction} (delta ELPD = {elpd_difference:.1f} "
            f"+/- {se_difference:.1f}), but {'; '.join(unreliable)} {verb} the "
            "reliability check, so the LOO comparison does not independently confirm the evidence."
        )
    elif elpd_difference > 0.0:
        loo_text = (
            f"PSIS-LOO independently agrees (delta ELPD = {elpd_difference:.1f} "
            f"+/- {se_difference:.1f})."
        )
    else:
        loo_text = (
            f"PSIS-LOO disagrees (delta ELPD = {elpd_difference:.1f} "
            f"+/- {se_difference:.1f}); that disagreement is part of the result."
        )
    prior_text = (
        f"M1's evidence moves by {delta:.2f} nats under the tighter eta prior, so the strength "
        "of the preference is prior-sensitive."
        if prior_sensitive
        else f"M1's evidence moves by only {delta:.2f} nats under the tighter eta prior."
    )
    verdict = (
        f"The synthetic data {strength}ly favour {direction} over "
        f"{'M0' if direction == 'M1' else 'M1'} "
        f"(delta ln Z = {abs(delta_log_evidence):.1f}, log10 Bayes factor = "
        f"{abs(log10_bayes_factor):.2f}). {loo_text} {prior_text} "
        "This is a simulation result, not a measured-data claim."
    )

    document = {
        "schema_version": 1,
        "synthetic": True,
        "generated_by": "cd train && uv run --with dynesty python -m bgo_train.evidence",
        "source_reference": str(args.reference.relative_to(TRAIN_ROOT.parent)).replace("\\", "/"),
        "note": "Synthetic M1 observations; M0 is deliberately misspecified. No measured data.",
        "loo_method": "NumPyro NUTS posterior draws, PSIS-LOO via ArviZ",
        "dataset": {
            "case_id": reference["case"]["case_id"],
            "model": reference["case"]["model"],
            "seed": reference["case"]["seed"],
            "points": len(data["temperature_k"]),
            "temperature_start_k": data["temperature_k"][0],
            "temperature_stop_k": data["temperature_k"][-1],
            "current_a": data["current_a"],
            "likelihood_sigma_v": data["noise"]["likelihood_sigma_v"],
        },
        "settings": {
            "engine": f"dynesty {dynesty.__version__}",
            "arviz": az.__version__,
            "nlive": NLIVE,
            "dlogz": DLOGZ,
            "posterior_draws": NUTS_CHAINS * NUTS_DRAWS,
            "seed": args.seed,
        },
        "models": [m0, m1],
        "comparison": {
            "baseline": "M1",
            "delta_log_evidence": delta_log_evidence,
            "log10_bayes_factor": log10_bayes_factor,
            "bayes_factor": _finite_or_none(math.exp(delta_log_evidence))
            if delta_log_evidence < 700.0
            else None,
            "elpd_difference": elpd_difference,
            "elpd_difference_se": se_difference,
            "direction": direction,
            "strength": strength,
        },
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
