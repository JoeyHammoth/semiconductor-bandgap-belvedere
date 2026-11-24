"""Generate the S10 evidence ledger from the committed synthetic M1 case.

Run from ``train/`` with::

    uv run --with dynesty==3.2.0 python -m bgo_train.evidence

Dynesty supplies ``ln Z`` and ``H``. NumPyro NUTS supplies posterior draws for
PSIS-LOO, because a nested-sampling equal-weight resample can collapse onto one
point when the likelihood is much narrower than the prior. That collapse is
reported through the Pareto-k flag rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import arviz as az
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


def _cache_load(cache_path: Path) -> dict[str, Any]:
    if not cache_path.is_file():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _cache_store(cache_path: Path, key: str, value: dict[str, Any]) -> None:
    cache = _cache_load(cache_path)
    cache[key] = value
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_evidence(
    model: str,
    variant: str,
    data: dict[str, Any],
    seed: int,
    cache_path: Path,
) -> dict[str, Any]:
    import dynesty

    key = f"evidence:v2:{model}:{variant}:{seed}:{NLIVE}:{DLOGZ}"
    cached = _cache_load(cache_path).get(key)
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
    _cache_store(cache_path, key, entry)
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
    cache_path: Path,
    init: dict[str, float] | None = None,
) -> dict[str, Any]:
    key = f"loo:v2:{model}:{variant}:{seed}:{NUTS_CHAINS}:{NUTS_WARMUP}:{NUTS_DRAWS}"
    cached = _cache_load(cache_path).get(key)
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
    _cache_store(cache_path, key, entry)
    return entry


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def derive_ledger(
    m0: dict[str, Any], m1: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Derive the comparison and verdict from the two model entries."""

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
    prior = m1["prior_sensitivity"]
    prior_delta = float(prior["delta_log_evidence"])
    prior_sensitive = bool(prior["flag"])
    prior_text = (
        f"M1's evidence moves by {prior_delta:.2f} nats under the tighter eta prior, so the "
        "strength of the preference is prior-sensitive."
        if prior_sensitive
        else f"M1's evidence moves by only {prior_delta:.2f} nats under the tighter eta prior."
    )
    verdict = (
        f"The synthetic data {strength}ly favour {direction} over "
        f"{'M0' if direction == 'M1' else 'M1'} "
        f"(delta ln Z = {abs(delta_log_evidence):.1f}, log10 Bayes factor = "
        f"{abs(log10_bayes_factor):.2f}). {loo_text} {prior_text} "
        "This is a simulation result, not a measured-data claim."
    )
    comparison = {
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
    }
    return comparison, verdict


def build_ledger(reference_path: Path, seed: int, cache_path: Path) -> dict[str, Any]:
    """Run the synthetic evidence pipeline and return the ledger document."""

    import dynesty

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    data = reference["observations"]
    if reference.get("case", {}).get("synthetic") is not True or data.get("synthetic") is not True:
        raise ValueError("the evidence ledger only accepts a labelled synthetic case")

    m0 = _run_evidence("M0", "weak", data, seed, cache_path)
    m0.update(_run_loo("M0", "weak", data, seed + 1, cache_path))
    reference_mean = reference["reference"]["mean"]
    m1_init = {
        "eg_ev": float(reference_mean["eg_ev"]),
        "ln_b_ref": float(
            reference_mean["ln_b"] + reference_mean["eta"] * math.log(REFERENCE_TEMPERATURE_K)
        ),
        "eta": float(reference_mean["eta"]),
    }
    m1_weak = _run_evidence("M1", "weak", data, seed + 10, cache_path)
    m1_weak.update(_run_loo("M1", "weak", data, seed + 11, cache_path, init=m1_init))
    m1_tight = _run_evidence("M1", "tight", data, seed + 20, cache_path)

    prior_delta = m1_tight["log_evidence"] - m1_weak["log_evidence"]
    m1 = dict(m1_weak)
    m1["prior_sensitivity"] = {
        "weak_log_evidence": m1_weak["log_evidence"],
        "tight_log_evidence": m1_tight["log_evidence"],
        "delta_log_evidence": prior_delta,
        "flag": bool(abs(prior_delta) > 0.5),
        "description": "eta prior scale 1.0 versus 0.25, both truncated to [1.5, 5]",
    }

    comparison, verdict = derive_ledger(m0, m1)
    source_reference = reference_path.resolve().relative_to(TRAIN_ROOT.parent.resolve())
    return {
        "schema_version": 1,
        "synthetic": True,
        "generated_by": "cd train && uv run --with dynesty python -m bgo_train.evidence",
        "source_reference": str(source_reference).replace("\\", "/"),
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
            "seed": seed,
        },
        "models": [m0, m1],
        "comparison": comparison,
        "verdict": verdict,
    }


def _compare_exact(
    errors: list[str], label: str, committed: Any, regenerated: Any
) -> None:
    if committed != regenerated:
        errors.append(f"{label}: committed {committed!r}, regenerated {regenerated!r}")


def _compare_number(
    errors: list[str],
    label: str,
    committed: Any,
    regenerated: Any,
    *,
    atol: float,
    rtol: float = 0.0,
) -> None:
    if (
        isinstance(committed, bool)
        or isinstance(regenerated, bool)
        or not isinstance(committed, (int, float))
        or not isinstance(regenerated, (int, float))
    ):
        errors.append(f"{label}: expected numbers, got {committed!r} and {regenerated!r}")
        return
    if not math.isclose(float(committed), float(regenerated), rel_tol=rtol, abs_tol=atol):
        errors.append(
            f"{label}: committed {float(committed):.6g}, regenerated "
            f"{float(regenerated):.6g} (atol {atol:g}, rtol {rtol:g})"
        )


def _compare_comparison(
    errors: list[str],
    label: str,
    committed: dict[str, Any],
    regenerated: dict[str, Any],
    *,
    delta_atol: float,
    elpd_atol: float,
    se_atol: float,
    check_bayes_value: bool,
) -> None:
    for key in ("baseline", "direction", "strength"):
        _compare_exact(errors, f"{label}.{key}", committed.get(key), regenerated.get(key))
    _compare_number(
        errors,
        f"{label}.delta_log_evidence",
        committed.get("delta_log_evidence"),
        regenerated.get("delta_log_evidence"),
        atol=delta_atol,
    )
    _compare_number(
        errors,
        f"{label}.log10_bayes_factor",
        committed.get("log10_bayes_factor"),
        regenerated.get("log10_bayes_factor"),
        atol=delta_atol / math.log(10.0),
    )
    _compare_number(
        errors,
        f"{label}.elpd_difference",
        committed.get("elpd_difference"),
        regenerated.get("elpd_difference"),
        atol=elpd_atol,
    )
    _compare_number(
        errors,
        f"{label}.elpd_difference_se",
        committed.get("elpd_difference_se"),
        regenerated.get("elpd_difference_se"),
        atol=se_atol,
    )
    left_bayes = committed.get("bayes_factor")
    right_bayes = regenerated.get("bayes_factor")
    if left_bayes is None or right_bayes is None:
        _compare_exact(errors, f"{label}.bayes_factor", left_bayes, right_bayes)
    elif check_bayes_value:
        _compare_number(
            errors,
            f"{label}.bayes_factor",
            left_bayes,
            right_bayes,
            atol=0.0,
            rtol=1.0e-9,
        )


def _model_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = document.get("models")
    if not isinstance(models, list):
        return {}
    return {str(model.get("id")): model for model in models if isinstance(model, dict)}


_MODEL_EXACT = (
    "prior_variant",
    "parameter_count",
    "nuts_chains",
    "nuts_warmup",
    "nuts_draws",
    "nuts_divergences",
    "loo_reliable",
)
_MODEL_NUMERIC = {
    "log_evidence": 0.5,
    "log_evidence_error": 0.2,
    "information": 0.5,
    "elpd_loo": 5.0,
    "elpd_loo_se": 1.0,
    "p_loo": 1.0,
    "pareto_k_max": 0.1,
}


def compare_ledger(
    committed: dict[str, Any], regenerated: dict[str, Any]
) -> list[str]:
    """Compare a committed ledger with a fresh run within documented tolerances."""

    errors: list[str] = []
    for key in (
        "schema_version",
        "synthetic",
        "source_reference",
        "note",
        "loo_method",
        "generated_by",
    ):
        _compare_exact(errors, key, committed.get(key), regenerated.get(key))

    committed_dataset = committed.get("dataset", {})
    regenerated_dataset = regenerated.get("dataset", {})
    for key in sorted(set(committed_dataset) | set(regenerated_dataset)):
        _compare_exact(
            errors,
            f"dataset.{key}",
            committed_dataset.get(key),
            regenerated_dataset.get(key),
        )

    committed_settings = committed.get("settings", {})
    regenerated_settings = regenerated.get("settings", {})
    for key in ("engine", "arviz", "nlive", "dlogz", "posterior_draws", "seed"):
        _compare_exact(
            errors,
            f"settings.{key}",
            committed_settings.get(key),
            regenerated_settings.get(key),
        )

    committed_models = _model_map(committed)
    regenerated_models = _model_map(regenerated)
    for model_id in sorted(set(committed_models) | set(regenerated_models)):
        left = committed_models.get(model_id)
        right = regenerated_models.get(model_id)
        if left is None or right is None:
            errors.append(f"models.{model_id}: present in only one ledger")
            continue
        for key in _MODEL_EXACT:
            _compare_exact(errors, f"models.{model_id}.{key}", left.get(key), right.get(key))
        for key, atol in _MODEL_NUMERIC.items():
            _compare_number(
                errors,
                f"models.{model_id}.{key}",
                left.get(key),
                right.get(key),
                atol=atol,
            )
        expected_reliable = bool(
            float(left.get("pareto_k_max", math.inf)) < 0.7
            and left.get("nuts_divergences") == 0
        )
        if bool(left.get("loo_reliable")) != expected_reliable:
            errors.append(
                f"models.{model_id}.loo_reliable: committed flag does not match Pareto-k "
                "and divergence gate"
            )

        left_prior = left.get("prior_sensitivity")
        right_prior = right.get("prior_sensitivity")
        if (left_prior is None) != (right_prior is None):
            errors.append(f"models.{model_id}.prior_sensitivity: present in only one ledger")
        elif left_prior is not None:
            for key in ("description", "flag"):
                _compare_exact(
                    errors,
                    f"models.{model_id}.prior_sensitivity.{key}",
                    left_prior.get(key),
                    right_prior.get(key),
                )
            for key in ("weak_log_evidence", "tight_log_evidence", "delta_log_evidence"):
                _compare_number(
                    errors,
                    f"models.{model_id}.prior_sensitivity.{key}",
                    left_prior.get(key),
                    right_prior.get(key),
                    atol=0.5,
                )
            expected_flag = abs(float(left_prior.get("delta_log_evidence", 0.0))) > 0.5
            if bool(left_prior.get("flag")) != expected_flag:
                errors.append(
                    f"models.{model_id}.prior_sensitivity.flag: committed flag does not "
                    "match the 0.5-nat threshold"
                )

    if "M0" in committed_models and "M1" in committed_models:
        derived_comparison, derived_verdict = derive_ledger(
            committed_models["M0"], committed_models["M1"]
        )
        _compare_comparison(
            errors,
            "comparison (derived from committed models)",
            committed.get("comparison", {}),
            derived_comparison,
            delta_atol=1.0e-9,
            elpd_atol=1.0e-9,
            se_atol=1.0e-9,
            check_bayes_value=True,
        )
        _compare_exact(
            errors,
            "verdict (derived from committed models)",
            committed.get("verdict"),
            derived_verdict,
        )
    if "M0" in regenerated_models and "M1" in regenerated_models:
        regenerated_comparison, regenerated_verdict = derive_ledger(
            regenerated_models["M0"], regenerated_models["M1"]
        )
        _compare_comparison(
            errors,
            "comparison (derived from regenerated models)",
            regenerated.get("comparison", {}),
            regenerated_comparison,
            delta_atol=1.0e-9,
            elpd_atol=1.0e-9,
            se_atol=1.0e-9,
            check_bayes_value=True,
        )
        _compare_exact(
            errors,
            "verdict (derived from regenerated models)",
            regenerated.get("verdict"),
            regenerated_verdict,
        )

    _compare_comparison(
        errors,
        "comparison (committed vs regenerated)",
        committed.get("comparison", {}),
        regenerated.get("comparison", {}),
        delta_atol=0.5,
        elpd_atol=5.0,
        se_atol=1.0,
        check_bayes_value=False,
    )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate with a temporary cache and fail if the committed ledger is stale",
    )
    args = parser.parse_args(argv)

    if args.check:
        if not args.out.is_file():
            print(f"missing committed ledger: {args.out}")
            return 1
        with tempfile.TemporaryDirectory(prefix="bgo-evidence-") as directory:
            fresh = build_ledger(
                args.reference,
                args.seed,
                Path(directory) / "evidence-cache.json",
            )
        committed = json.loads(args.out.read_text(encoding="utf-8"))
        errors = compare_ledger(committed, fresh)
        if errors:
            print(f"stale evidence ledger: {args.out}")
            for error in errors[:50]:
                print(f"  - {error}")
            if len(errors) > 50:
                print(f"  ... and {len(errors) - 50} more")
            return 1
        print(f"{args.out} matches a fresh regeneration within tolerance")
        return 0

    document = build_ledger(args.reference, args.seed, args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    print(document["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
