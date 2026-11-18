"""Offline NumPyro NUTS references for the synthetic M0-M4 cases.

Run from ``train/`` with::

    uv run python -m bgo_train.reference --all
    uv run python -m bgo_train.reference --check

The output is deliberately a summary, not a posterior sample dump: quantiles,
correlations and convergence diagnostics are what the Rust sampler is checked
against.  The observations beside them are generated synthetic data, not a
reconstructed measurement.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import arviz as az
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from scipy.optimize import least_squares

from .config import CaseConfig, load_all, load_config
from .simulation import forward_voltage, generate_dataset, observation_json

SCHEMA_VERSION = 1
WARMUP = 3000
SAMPLES = 2000
CHAINS = 4
TARGET_ACCEPT = 0.99
MAX_TREE_DEPTH = 12
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
RHAT_LIMIT = 1.01
ESS_LIMIT = 400.0
REFERENCE_TEMPERATURE_K = 300.0

TRAIN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = TRAIN_ROOT / "configs"
DEFAULT_OUTPUT_DIR = TRAIN_ROOT / "references"


def _truncated_normal(
    name: str,
    loc: float,
    scale: float,
    low: float,
    high: float,
) -> jax.Array:
    return numpyro.sample(
        name,
        dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high),
    )


def _model(config: CaseConfig, dataset: Any):
    sigma = jnp.asarray(dataset.likelihood_sigma_v, dtype=jnp.float64)
    temperature = jnp.asarray(dataset.temperature_k, dtype=jnp.float64)
    voltage = jnp.asarray(dataset.voltage_v, dtype=jnp.float64)

    parameters: dict[str, jax.Array] = {}
    if config.model in ("M0", "M1"):
        parameters["eg_ev"] = _truncated_normal("eg_ev", 0.9, 0.35, 0.05, 3.5)
    else:
        parameters["eg0_ev"] = _truncated_normal("eg0_ev", 0.9, 0.35, 0.05, 3.5)
    if config.model == "M0":
        parameters["ln_b"] = numpyro.sample("ln_b", dist.Normal(8.0, 8.0))
    else:
        # Parameterising at 300 K removes the ln B-eta translation that makes
        # the raw (ln B, eta) posterior a narrow curved ridge.
        ln_b_ref = numpyro.sample("ln_b_ref", dist.Normal(8.0, 8.0))
        if config.model in ("M1", "M2"):
            eta = _truncated_normal("eta", 3.0, 1.0, 1.5, 5.0)
            parameters["eta"] = eta
        else:
            # A single constant-current sweep cannot separate eta from the
            # terminal ideality factor; M3/M4 hold eta at its illustrative
            # central value and infer ideality/Rs instead.
            eta = jnp.asarray(3.0, dtype=jnp.float64)
            parameters["eta"] = eta
        ln_b = ln_b_ref - eta * jnp.log(REFERENCE_TEMPERATURE_K)
        numpyro.deterministic("ln_b", ln_b)
        parameters["ln_b"] = ln_b
    if config.model in ("M2", "M3", "M4"):
        # The Si Varshni pair is resolved in docs/physics-audit.md section 2;
        # this informative prior keeps the weak alpha-beta ridge from dominating
        # the reference while still allowing a broad fit.
        parameters["alpha_ev_per_k"] = _truncated_normal(
            "alpha_ev_per_k", 7.021e-4, 1.0e-4, 0.0, 2.0e-3
        )
        parameters["beta_k"] = _truncated_normal("beta_k", 1108.0, 150.0, 100.0, 3000.0)
    if config.model in ("M3", "M4"):
        parameters["ideality"] = _truncated_normal("ideality", 1.0, 0.15, 0.7, 1.6)
    if config.model == "M4":
        parameters["series_resistance_ohm"] = numpyro.sample(
            "series_resistance_ohm", dist.HalfNormal(10.0)
        )

    predicted = forward_voltage(config.model, parameters, temperature, dataset.current_a)
    numpyro.sample("obs", dist.Normal(predicted, sigma), obs=voltage)


def _round(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("reference produced a non-finite value")
    return float(round(result, 10))


def _value(array: Any, name: str) -> float:
    values = np.asarray(array[name].values if hasattr(array[name], "values") else array[name])
    if values.size != 1:
        raise ValueError(f"diagnostic {name!r} has unexpected shape {values.shape}")
    return _round(values.reshape(-1)[0])


def _summary(mcmc: MCMC, config: CaseConfig) -> dict[str, Any]:
    samples = mcmc.get_samples(group_by_chain=True)
    parameters = list(config.free_parameters)
    draws = {name: np.asarray(samples[name]).reshape(-1) for name in parameters}
    summary: dict[str, Any] = {
        "parameters": parameters,
        "mean": {name: _round(np.mean(draws[name])) for name in parameters},
        "sd": {
            name: _round(np.std(draws[name], ddof=1)) for name in parameters
        },
        "quantiles": {
            f"{quantile:.2f}": {
                name: _round(np.quantile(draws[name], quantile)) for name in parameters
            }
            for quantile in QUANTILES
        },
    }

    matrix = np.corrcoef(np.stack([draws[name] for name in parameters], axis=0))
    summary["correlation"] = {
        "parameters": parameters,
        "matrix": [[_round(value) for value in row] for row in matrix],
    }

    diverging = np.asarray(mcmc.get_extra_fields()["diverging"]).reshape(CHAINS, SAMPLES)
    idata = az.from_dict(
        {
            "posterior": {name: np.asarray(samples[name]) for name in parameters},
            "sample_stats": {"diverging": diverging},
        }
    )
    rhat = az.rhat(idata)
    ess = az.ess(idata, method="bulk")
    divergences = int(diverging.sum())
    rhat_values = {name: _value(rhat, name) for name in parameters}
    ess_values = {name: _value(ess, name) for name in parameters}
    passes = (
        divergences == 0
        and max(rhat_values.values()) < RHAT_LIMIT
        and min(ess_values.values()) > ESS_LIMIT
    )
    summary["diagnostics"] = {
        "num_chains": CHAINS,
        "num_draws": SAMPLES,
        "divergences": divergences,
        "rhat": rhat_values,
        "ess_bulk": ess_values,
        "rhat_max": max(rhat_values.values()),
        "ess_bulk_min": min(ess_values.values()),
        "passes": bool(passes),
    }
    return summary


def _prior_center(config: CaseConfig) -> dict[str, float]:
    """A prior-centred start used when the unregularised fit hits a bound."""

    if config.model == "M0":
        return {"eg_ev": 0.9, "ln_b": 8.0}
    if config.model == "M1":
        return {"eg_ev": 0.9, "ln_b_ref": 8.0, "eta": 3.0}
    values = {
        "eg0_ev": 0.9,
        "ln_b_ref": 8.0,
        "alpha_ev_per_k": 7.021e-4,
        "beta_k": 1108.0,
    }
    if config.model == "M2":
        values["eta"] = 3.0
    if config.model in ("M3", "M4"):
        values["ideality"] = 1.0
    if config.model == "M4":
        values["series_resistance_ohm"] = 1.0
    return values


def _initial_values(config: CaseConfig, dataset: Any) -> dict[str, float]:
    """Find a bounded least-squares start, falling back to the prior centre.

    This is an initialization, not a posterior approximation: NUTS still
    explores the full prior-constrained target.  It avoids starting broad
    correlated cases at a random point far outside the typical set.
    """

    if config.model == "M0":
        names = ["eg_ev", "ln_b"]
    elif config.model == "M1":
        names = ["eg_ev", "ln_b_ref", "eta"]
    elif config.model == "M2":
        names = ["eg0_ev", "ln_b_ref", "eta", "alpha_ev_per_k", "beta_k"]
    else:
        names = ["eg0_ev", "ln_b_ref", "alpha_ev_per_k", "beta_k", "ideality"]
        if config.model == "M4":
            names.append("series_resistance_ohm")

    start = {
        "eg_ev": 0.9,
        "eg0_ev": 0.9,
        "ln_b": 8.0,
        "ln_b_ref": 8.0,
        "eta": 3.0,
        "alpha_ev_per_k": 7.0e-4,
        "beta_k": 1108.0,
        "ideality": 1.0,
        "series_resistance_ohm": 1.0,
    }
    lower = {
        "eg_ev": 0.05,
        "eg0_ev": 0.05,
        "ln_b": -20.0,
        "ln_b_ref": -20.0,
        "eta": 1.5,
        "alpha_ev_per_k": 1.0e-8,
        "beta_k": 100.0,
        "ideality": 0.7,
        "series_resistance_ohm": 0.0,
    }
    upper = {
        "eg_ev": 3.5,
        "eg0_ev": 3.5,
        "ln_b": 40.0,
        "ln_b_ref": 40.0,
        "eta": 5.0,
        "alpha_ev_per_k": 2.0e-3,
        "beta_k": 3000.0,
        "ideality": 1.6,
        "series_resistance_ohm": 50.0,
    }
    x0 = np.asarray([start[name] for name in names], dtype=np.float64)
    lo = np.asarray([lower[name] for name in names], dtype=np.float64)
    hi = np.asarray([upper[name] for name in names], dtype=np.float64)

    def unpack(values: np.ndarray) -> dict[str, Any]:
        fitted = dict(zip(names, values, strict=True))
        parameters: dict[str, Any] = dict(fitted)
        if config.model != "M0":
            eta = fitted.get("eta", 3.0)
            parameters["eta"] = eta
            parameters["ln_b"] = fitted["ln_b_ref"] - eta * math.log(
                REFERENCE_TEMPERATURE_K
            )
        return parameters

    def residual(values: np.ndarray) -> np.ndarray:
        predicted = np.asarray(
            forward_voltage(
                config.model,
                {key: jnp.asarray(value) for key, value in unpack(values).items()},
                dataset.temperature_k,
                dataset.current_a,
            ),
            dtype=np.float64,
        )
        return (predicted - dataset.voltage_v) / dataset.likelihood_sigma_v

    result = least_squares(
        residual,
        np.clip(x0, lo, hi),
        bounds=(lo, hi),
        x_scale="jac",
        max_nfev=2000,
    )
    if not result.success:
        raise RuntimeError(f"{config.case_id}: initialization fit failed: {result.message}")
    # A fit pinned to a prior bound is a poor NUTS starting point and usually
    # means the single-current design is sloppy.  Start at the prior centre
    # instead of entering warmup on the boundary.
    margin = 1e-6 * np.maximum(1.0, np.abs(hi - lo))
    if np.any(result.x <= lo + margin) or np.any(result.x >= hi - margin):
        return _prior_center(config)
    return {name: float(value) for name, value in zip(names, result.x, strict=True)}


def generate_reference(config: CaseConfig) -> dict[str, Any]:
    """Run one fixed NUTS reference and return the committed document."""

    dataset = generate_dataset(config)
    initial_values = _initial_values(config, dataset)
    kernel = NUTS(
        lambda: _model(config, dataset),
        target_accept_prob=TARGET_ACCEPT,
        max_tree_depth=MAX_TREE_DEPTH,
        dense_mass=True,
        init_strategy=init_to_value(values=initial_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=WARMUP,
        num_samples=SAMPLES,
        num_chains=CHAINS,
        chain_method="sequential",
        progress_bar=False,
    )
    # A fixed inference seed, distinct from the data seed.
    mcmc.run(jax.random.PRNGKey(config.seed + 1))
    summary = _summary(mcmc, config)
    document = {
        "schema_version": SCHEMA_VERSION,
        "case": config.as_json(),
        "observations": observation_json(dataset),
        "truth": dict(config.truth),
        "reference": {
            "engine": "numpyro",
            "numpyro_version": numpyro.__version__,
            "jax_version": jax.__version__,
            "arviz_version": az.__version__,
            "settings": {
                "warmup": WARMUP,
                "samples": SAMPLES,
                "chains": CHAINS,
                "target_accept_prob": TARGET_ACCEPT,
                "max_tree_depth": MAX_TREE_DEPTH,
            },
            **summary,
        },
    }
    if not summary["diagnostics"]["passes"]:
        raise RuntimeError(
            f"{config.case_id}: reference diagnostics failed: "
            f"{summary['diagnostics']}"
        )
    return document


def _manifest(configs: Iterable[CaseConfig], output_dir: Path) -> dict[str, Any]:
    cases = []
    for config in configs:
        reference_name = f"{config.case_id}.json"
        if not (output_dir / reference_name).is_file():
            raise FileNotFoundError(f"missing reference output: {reference_name}")
        cases.append(
            {
                "case_id": config.case_id,
                "model": config.model,
                "config": f"configs/{config.source_name}",
                "config_sha256": config.config_sha256,
                "seed": config.seed,
                "seed_sha256": config.seed_sha256,
                "reference": reference_name,
                "synthetic": True,
                "anchor": None if config.anchor is None else config.anchor.as_json(),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "generated_by": "python -m bgo_train.reference --all",
        "note": "Synthetic simulation references; no measured data and no data hash.",
        "engine": {
            "numpyro": numpyro.__version__,
            "jax": jax.__version__,
            "arviz": az.__version__,
        },
        "settings": {
            "warmup": WARMUP,
            "samples": SAMPLES,
            "chains": CHAINS,
            "target_accept_prob": TARGET_ACCEPT,
            "max_tree_depth": MAX_TREE_DEPTH,
        },
        "cases": cases,
    }


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def check_references(config_dir: Path, output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("synthetic") is not True:
        raise ValueError("manifest must be schema 1 and explicitly synthetic")
    configs = load_all(config_dir)
    by_id = {config.case_id: config for config in configs}
    entries = manifest.get("cases")
    if not isinstance(entries, list):
        raise TypeError("manifest.cases must be a list")
    seen: set[str] = set()
    for entry in entries:
        case_id = entry.get("case_id")
        if case_id not in by_id:
            raise ValueError(f"manifest references unknown case {case_id!r}")
        if case_id in seen:
            raise ValueError(f"manifest has duplicate case {case_id!r}")
        seen.add(case_id)
        config = by_id[case_id]
        if entry.get("config_sha256") != config.config_sha256:
            raise ValueError(f"{case_id}: config hash is stale; regenerate references")
        if entry.get("seed") != config.seed or entry.get("seed_sha256") != config.seed_sha256:
            raise ValueError(f"{case_id}: seed metadata is stale")
        if entry.get("synthetic") is not True:
            raise ValueError(f"{case_id}: manifest entry must be synthetic")
        reference_path = output_dir / entry["reference"]
        if not reference_path.is_file():
            raise FileNotFoundError(f"missing reference file: {reference_path}")
        document = json.loads(reference_path.read_text(encoding="utf-8"))
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{case_id}: reference schema is not {SCHEMA_VERSION}")
        if document.get("case", {}).get("synthetic") is not True:
            raise ValueError(f"{case_id}: reference is not labelled synthetic")
        if document.get("case", {}).get("config_sha256") != config.config_sha256:
            raise ValueError(f"{case_id}: reference config hash is stale")
        diagnostics = document.get("reference", {}).get("diagnostics", {})
        if diagnostics.get("passes") is not True:
            raise ValueError(f"{case_id}: reference diagnostics do not pass")
    if seen != set(by_id):
        missing = sorted(set(by_id) - seen)
        raise ValueError(f"manifest is missing cases: {missing}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="regenerate every committed case")
    parser.add_argument("--config", type=Path, help="regenerate one TOML case")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true", help="validate committed references only")
    args = parser.parse_args(argv)

    if args.check:
        if args.config or args.all:
            parser.error("--check cannot be combined with --config or --all")
        check_references(args.config_dir, args.out)
        print("reference manifest and cases are consistent")
        return 0

    if args.config:
        configs = [load_config(args.config)]
    else:
        configs = load_all(args.config_dir)
    for config in configs:
        document = generate_reference(config)
        path = args.out / f"{config.case_id}.json"
        write_json(path, document)
        print(
            f"{config.case_id}: {config.model}, "
            f"divergences={document['reference']['diagnostics']['divergences']}, "
            f"rhat_max={document['reference']['diagnostics']['rhat_max']:.4f}"
        )
    if not args.config:
        write_json(args.out / "manifest.json", _manifest(configs, args.out))
        print(f"wrote {len(configs)} references and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
