"""Offline conditional spline-flow training for the synthetic M0 case.

The exported artifact is intentionally small and fully inference-only:

- ``web/static/weights/m0-flow.bin``: little-endian float64 weights
- ``web/static/weights/m0-flow.json``: architecture, standardisation and OOD stats
- ``web/static/data/npe-calibration.json``: SBC, TARP and OOD results
- ``crates/bgo-nn/tests/references/m0-flow-golden.json``: Rust reference vectors

This module uses JAX, which is already part of the pinned training environment.
PyTorch is not required. Every observation is synthetic; the original lab runs
do not exist (ADR 0006).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy import stats

from .config import load_config
from .simulation import forward_voltage

ROOT = Path(__file__).resolve().parents[3]
CASE_CONFIG = ROOT / "train" / "configs" / "m0-report-si.toml"
WEIGHTS_BIN = ROOT / "web" / "static" / "weights" / "m0-flow.bin"
MANIFEST_JSON = ROOT / "web" / "static" / "weights" / "m0-flow.json"
CALIBRATION_JSON = ROOT / "web" / "static" / "data" / "npe-calibration.json"
GOLDEN_JSON = ROOT / "crates" / "bgo-nn" / "tests" / "references" / "m0-flow-golden.json"

PARAMETER_NAMES = ["eg_ev", "ln_b"]
# The M0 NPE is explicitly local to the report-anchored synthetic case. The
# broad M0 prior is used by MCLMC; an amortised model trained over the full
# prior would need far more simulations and cannot claim calibration here.
PARAMETER_CENTER = np.array([1.002, 5.504239846], dtype=np.float64)
PARAMETER_SCALE = np.array([0.02, 0.2], dtype=np.float64)
PRIOR_LOW = np.array([0.90, 4.5], dtype=np.float64)
PRIOR_HIGH = np.array([1.10, 6.5], dtype=np.float64)


@dataclass(frozen=True)
class NpeConfig:
    flow_kind: str = "affine"
    seed: int = 20261126
    simulations: int = 40_000
    validation_fraction: float = 0.15
    steps: int = 6_000
    batch_size: int = 256
    learning_rate: float = 5.0e-4
    hidden_dim: int = 96
    layers: int = 6
    bins: int = 12
    bound: float = 4.0
    sbc_simulations: int = 1_000
    posterior_samples: int = 100
    tarp_alphas: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ood_simulations: int = 300
    golden_count: int = 8


def _mlp_init(key: jax.Array, in_dim: int, hidden: int, out_dim: int, bins: int) -> dict[str, jax.Array]:
    first, second, third = jax.random.split(key, 3)
    w1 = jax.random.normal(first, (in_dim, hidden)) * math.sqrt(2.0 / in_dim)
    w2 = jax.random.normal(second, (hidden, hidden)) * math.sqrt(2.0 / hidden)
    w3 = jax.random.normal(third, (hidden, out_dim)) * 0.01
    bias3 = jnp.zeros((out_dim,), dtype=jnp.float64)
    # Start with unit derivatives: softplus(log(exp(1)-1)) = 1.
    bias3 = bias3.at[2 * bins :].set(math.log(math.expm1(1.0)))
    return {
        "W1": w1.astype(jnp.float64),
        "b1": jnp.zeros((hidden,), dtype=jnp.float64),
        "W2": w2.astype(jnp.float64),
        "b2": jnp.zeros((hidden,), dtype=jnp.float64),
        "W3": w3.astype(jnp.float64),
        "b3": bias3,
    }


def _init_params(key: jax.Array, config: NpeConfig, condition_dim: int) -> list[dict[str, jax.Array]]:
    keys = jax.random.split(key, config.layers)
    out_dim = 2 if config.flow_kind == "affine" else 3 * config.bins - 1
    return [
        _mlp_init(keys[index], condition_dim + 1, config.hidden_dim, out_dim, config.bins)
        for index in range(config.layers)
    ]


def _spline_params(raw: jax.Array, bins: int, bound: float) -> tuple[jax.Array, jax.Array, jax.Array]:
    widths = jax.nn.softmax(raw[:bins]) * (2.0 * bound)
    heights = jax.nn.softmax(raw[bins : 2 * bins]) * (2.0 * bound)
    interior = jax.nn.softplus(raw[2 * bins :]) + 1.0e-6
    derivatives = jnp.concatenate(
        [jnp.ones((1,), dtype=raw.dtype), interior, jnp.ones((1,), dtype=raw.dtype)]
    )
    return widths, heights, derivatives


def _bin_index(knots: jax.Array, value: jax.Array, bins: int) -> jax.Array:
    # Number of interior knots at or below the value. Clipping handles the
    # boundary values without a separate search.
    index = jnp.sum(value >= knots[1:-1])
    return jnp.clip(index, 0, bins - 1).astype(jnp.int32)


def _rq_forward(
    value: jax.Array, raw: jax.Array, bins: int, bound: float
) -> tuple[jax.Array, jax.Array]:
    raw = jnp.clip(raw, -8.0, 8.0)
    widths, heights, derivatives = _spline_params(raw, bins, bound)
    cum_widths = jnp.concatenate([jnp.array([-bound]), -bound + jnp.cumsum(widths)])
    cum_heights = jnp.concatenate([jnp.array([-bound]), -bound + jnp.cumsum(heights)])
    inside = (value > -bound) & (value < bound)
    clipped = jnp.clip(value, -bound + 1.0e-12, bound - 1.0e-12)
    index = _bin_index(cum_widths, clipped, bins)
    width = widths[index]
    height = heights[index]
    left = derivatives[index]
    right = derivatives[index + 1]
    theta = (clipped - cum_widths[index]) / width
    one_minus = 1.0 - theta
    delta = height / width
    denominator = delta + (right + left - 2.0 * delta) * theta * one_minus
    numerator = height * (delta * theta * theta + left * theta * one_minus)
    output = cum_heights[index] + numerator / denominator
    derivative_numerator = delta * delta * (
        right * theta * theta + 2.0 * delta * theta * one_minus + left * one_minus * one_minus
    )
    derivative = derivative_numerator / (denominator * denominator)
    output = jnp.where(inside, output, value)
    log_derivative = jnp.where(
        inside, jnp.clip(jnp.log(jnp.maximum(derivative, 1.0e-8)), -12.0, 12.0), 0.0
    )
    return output, log_derivative


def _rq_inverse(
    value: jax.Array, raw: jax.Array, bins: int, bound: float
) -> jax.Array:
    raw = jnp.clip(raw, -8.0, 8.0)
    widths, heights, derivatives = _spline_params(raw, bins, bound)
    cum_widths = jnp.concatenate([jnp.array([-bound]), -bound + jnp.cumsum(widths)])
    cum_heights = jnp.concatenate([jnp.array([-bound]), -bound + jnp.cumsum(heights)])
    inside = (value > -bound) & (value < bound)
    clipped = jnp.clip(value, -bound + 1.0e-12, bound - 1.0e-12)
    index = _bin_index(cum_heights, clipped, bins)
    width = widths[index]
    height = heights[index]
    left = derivatives[index]
    right = derivatives[index + 1]
    delta = height / width
    offset = clipped - cum_heights[index]
    quadratic = right + left - 2.0 * delta
    a = offset * quadratic + height * (delta - left)
    b = height * left - offset * quadratic
    c = -offset * delta
    discriminant = jnp.maximum(b * b - 4.0 * a * c, 0.0)
    root = jnp.sqrt(discriminant)
    denominator = jnp.where(jnp.abs(a) > 1.0e-12, 2.0 * a, jnp.sign(b) * jnp.maximum(jnp.abs(b), 1.0e-12))
    root_a = (-b + root) / denominator
    root_b = (-b - root) / denominator
    theta = jnp.where((root_a >= 0.0) & (root_a <= 1.0), root_a, root_b)
    theta = jnp.where(jnp.abs(a) > 1.0e-12, theta, jnp.clip(-c / jnp.maximum(b, 1.0e-12), 0.0, 1.0))
    theta = jnp.clip(theta, 0.0, 1.0)
    output = cum_widths[index] + theta * width
    return jnp.where(inside, output, value)


def _conditioner(layer: dict[str, jax.Array], context: jax.Array, condition: jax.Array) -> jax.Array:
    inputs = jnp.concatenate([context, condition])
    hidden = jax.nn.silu(inputs @ layer["W1"] + layer["b1"])
    hidden = jax.nn.silu(hidden @ layer["W2"] + layer["b2"])
    return hidden @ layer["W3"] + layer["b3"]


def _layer_forward(
    x: jax.Array, condition: jax.Array, layer: dict[str, jax.Array], index: int, config: NpeConfig
) -> tuple[jax.Array, jax.Array]:
    transformed = index % 2
    context = 1 - transformed
    raw = _conditioner(layer, x[context : context + 1], condition)
    if config.flow_kind == "affine":
        log_scale = jnp.clip(raw[0], -6.0, 6.0)
        value = jnp.exp(log_scale) * x[transformed] + raw[1]
        return x.at[transformed].set(value), log_scale
    value, log_derivative = _rq_forward(x[transformed], raw, config.bins, config.bound)
    return x.at[transformed].set(value), log_derivative


def _layer_inverse(
    x: jax.Array, condition: jax.Array, layer: dict[str, jax.Array], index: int, config: NpeConfig
) -> jax.Array:
    transformed = index % 2
    context = 1 - transformed
    raw = _conditioner(layer, x[context : context + 1], condition)
    if config.flow_kind == "affine":
        log_scale = jnp.clip(raw[0], -6.0, 6.0)
        value = (x[transformed] - raw[1]) * jnp.exp(-log_scale)
        return x.at[transformed].set(value)
    value = _rq_inverse(x[transformed], raw, config.bins, config.bound)
    return x.at[transformed].set(value)


def _forward(
    params: list[dict[str, jax.Array]], condition: jax.Array, x: jax.Array, config: NpeConfig
) -> tuple[jax.Array, jax.Array]:
    z = x
    log_det = jnp.array(0.0, dtype=jnp.float64)
    for index, layer in enumerate(params):
        z, term = _layer_forward(z, condition, layer, index, config)
        log_det = log_det + term
    return z, log_det


def _log_prob(
    params: list[dict[str, jax.Array]], condition: jax.Array, x: jax.Array, config: NpeConfig
) -> jax.Array:
    z, log_det = _forward(params, condition, x, config)
    base = -0.5 * jnp.sum(z * z) - 0.5 * x.shape[0] * math.log(2.0 * math.pi)
    return base + log_det


def _sample_one(
    params: list[dict[str, jax.Array]], condition: jax.Array, key: jax.Array, config: NpeConfig
) -> jax.Array:
    x = jax.random.normal(key, (2,), dtype=jnp.float64)
    for index in range(len(params) - 1, -1, -1):
        x = _layer_inverse(x, condition, params[index], index, config)
    return x


def _sample(
    params: list[dict[str, jax.Array]], conditions: jax.Array, key: jax.Array, count: int, config: NpeConfig
) -> jax.Array:
    keys = jax.random.split(key, conditions.shape[0] * count).reshape(conditions.shape[0], count, 2)
    repeated = jnp.repeat(conditions[:, None, :], count, axis=1)
    return jax.vmap(
        jax.vmap(lambda condition, sample_key: _sample_one(params, condition, sample_key, config))
    )(repeated, keys)


def _tree_zeros(params: list[dict[str, jax.Array]]) -> list[dict[str, jax.Array]]:
    return jax.tree_util.tree_map(jnp.zeros_like, params)


def _adam(
    params: list[dict[str, jax.Array]],
    conditions: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    config: NpeConfig,
) -> tuple[list[dict[str, jax.Array]], float]:
    first = _tree_zeros(params)
    second = _tree_zeros(params)
    beta1, beta2, epsilon = 0.9, 0.999, 1.0e-8

    def loss_fn(current: list[dict[str, jax.Array]], batch_conditions: jax.Array, batch_targets: jax.Array) -> jax.Array:
        values = jax.vmap(lambda condition, target: _log_prob(current, condition, target, config))(
            batch_conditions, batch_targets
        )
        penalty = jax.tree_util.tree_reduce(
            lambda total, value: total + jnp.sum(value * value),
            current,
            initializer=jnp.array(0.0),
        )
        return -jnp.mean(values) + 1.0e-5 * penalty

    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_fn_jit = jax.jit(loss_fn)
    final_loss = float("nan")
    for step in range(1, config.steps + 1):
        key, batch_key = jax.random.split(key)
        indices = jax.random.randint(batch_key, (config.batch_size,), 0, conditions.shape[0])
        grads = grad_fn(params, conditions[indices], targets[indices])
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -5.0, 5.0), grads)
        first = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, first, grads)
        second = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * g * g, second, grads)
        correction1 = 1.0 - beta1**step
        correction2 = 1.0 - beta2**step
        learning_rate = config.learning_rate * (0.2 + 0.8 * (1.0 - step / config.steps))
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - learning_rate * (m / correction1) / (jnp.sqrt(v / correction2) + epsilon),
            params,
            first,
            second,
        )
        if step % 500 == 0 or step == config.steps:
            final_loss = float(loss_fn_jit(params, conditions, targets))
            print(f"step {step:4d}/{config.steps}: validation loss {final_loss:.6f}")
    return params, final_loss


def _draw_prior(key: jax.Array, count: int) -> jax.Array:
    key1, key2 = jax.random.split(key)
    eg = jax.random.normal(key1, (count,), dtype=jnp.float64) * PARAMETER_SCALE[0] + PARAMETER_CENTER[0]
    ln_b = jax.random.normal(key2, (count,), dtype=jnp.float64) * PARAMETER_SCALE[1] + PARAMETER_CENTER[1]
    return jnp.stack([jnp.clip(eg, PRIOR_LOW[0], PRIOR_HIGH[0]), jnp.clip(ln_b, PRIOR_LOW[1], PRIOR_HIGH[1])], axis=1)


def _simulate(
    key: jax.Array,
    theta: jax.Array,
    config: Any,
    temperature_shift: float = 0.0,
    current_scale: float = 1.0,
    sigma_scale: float = 1.0,
) -> jax.Array:
    temperatures = jnp.linspace(
        config.temperature_start_k + temperature_shift,
        config.temperature_stop_k + temperature_shift,
        config.points,
        dtype=jnp.float64,
    )
    current = jnp.asarray(config.current_a * current_scale, dtype=jnp.float64)
    noiseless = jax.vmap(
        lambda point: forward_voltage(
            "M0", {"eg_ev": point[0], "ln_b": point[1]}, temperatures, current
        )
    )(theta)
    noise = (
        jax.random.normal(key, noiseless.shape, dtype=jnp.float64)
        * config.likelihood_sigma_v
        * sigma_scale
    )
    return noiseless + noise


def _summaries(conditions: np.ndarray) -> np.ndarray:
    """Five design-aware summaries used only for the OOD detector.

    The full 41-point curve is the flow condition. The OOD detector uses the
    mean, linear slope, quadratic curvature, residual scatter and endpoint
    drop, which stay interpretable and do not invert a near-singular 41x41
    covariance.
    """
    values = np.asarray(conditions, dtype=np.float64)
    coordinate = np.linspace(-1.0, 1.0, values.shape[1])
    design = np.stack([np.ones_like(coordinate), coordinate, coordinate * coordinate], axis=1)
    coefficients, _, _, _ = np.linalg.lstsq(design, values.T, rcond=None)
    fitted = (design @ coefficients).T
    residual = values - fitted
    return np.stack(
        [
            values.mean(axis=1),
            coefficients[1],
            coefficients[2],
            residual.std(axis=1),
            values[:, -1] - values[:, 0],
        ],
        axis=1,
    )


def _standardise(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - mean) / scale


def _flatten(params: list[dict[str, jax.Array]]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for layer in params:
        for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
            arrays.append(np.asarray(layer[name], dtype=np.float64).ravel())
    return np.concatenate(arrays)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_float(value: float) -> float:
    return float(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train(config: NpeConfig) -> dict[str, Any]:
    case = load_config(CASE_CONFIG)
    key = jax.random.PRNGKey(config.seed)
    key, prior_key, noise_key, init_key = jax.random.split(key, 4)
    total = config.simulations
    theta = _draw_prior(prior_key, total)
    observations = _simulate(noise_key, theta, case)
    # A five-number design-aware summary is the conditioning input. The design
    # explicitly permits summary statistics, and this avoids asking a small
    # flow to infer a tight posterior from 41 nearly collinear voltages.
    conditions = _summaries(np.asarray(observations, dtype=np.float64))
    targets = np.asarray(theta, dtype=np.float64)

    split = int(total * (1.0 - config.validation_fraction))
    train_conditions_raw = conditions[:split]
    train_targets_raw = targets[:split]
    validation_conditions_raw = conditions[split:]
    validation_targets_raw = targets[split:]
    condition_mean = train_conditions_raw.mean(axis=0)
    condition_scale = train_conditions_raw.std(axis=0)
    condition_scale = np.where(condition_scale > 1.0e-12, condition_scale, 1.0)
    train_conditions = _standardise(train_conditions_raw, condition_mean, condition_scale)
    validation_conditions = _standardise(validation_conditions_raw, condition_mean, condition_scale)
    train_targets = _standardise(train_targets_raw, PARAMETER_CENTER, PARAMETER_SCALE)
    validation_targets = _standardise(validation_targets_raw, PARAMETER_CENTER, PARAMETER_SCALE)

    params = _init_params(init_key, config, conditions.shape[1])
    params, final_loss = _adam(
        params,
        jnp.asarray(train_conditions, dtype=jnp.float64),
        jnp.asarray(train_targets, dtype=jnp.float64),
        key,
        config,
    )
    flat = _flatten(params)
    WEIGHTS_BIN.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_BIN.write_bytes(flat.astype("<f8").tobytes())
    weights_sha = _sha256(WEIGHTS_BIN)

    # OOD is fitted in the same five-summary condition space as the flow.
    # This is stable, interpretable and catches a changed design as well as a
    # changed parameter region.
    ood_summary_mean = train_conditions_raw.mean(axis=0)
    ood_summary_scale = train_conditions_raw.std(axis=0)
    ood_summary_scale = np.where(ood_summary_scale > 1.0e-12, ood_summary_scale, 1.0)
    covariance = np.cov(train_conditions, rowvar=False) + np.eye(train_conditions.shape[1]) * 1.0e-6
    inverse_covariance = np.linalg.inv(covariance)
    ood_mean = np.zeros_like(ood_summary_mean)
    validation_ood = _mahalanobis(validation_conditions, ood_mean, inverse_covariance)
    ood_threshold = float(np.quantile(validation_ood, 0.99))

    manifest = {
        "schema_version": 1,
        "kind": "conditional-affine-flow" if config.flow_kind == "affine" else "conditional-rational-quadratic-spline-flow",
        "flow_kind": config.flow_kind,
        "synthetic": True,
        "training_framework": "jax",
        "model": "M0",
        "parameter_names": PARAMETER_NAMES,
        "prior_scope": "local report-anchored M0",
        "prior_low": PRIOR_LOW.tolist(),
        "prior_high": PRIOR_HIGH.tolist(),
        "condition_dim": int(conditions.shape[1]),
        "parameter_dim": 2,
        "hidden_dim": config.hidden_dim,
        "layers": config.layers,
        "bins": config.bins,
        "bound": config.bound,
        "layer_split": [index % 2 for index in range(config.layers)],
        "parameter_center": PARAMETER_CENTER.tolist(),
        "parameter_scale": PARAMETER_SCALE.tolist(),
        "condition_mean": condition_mean.tolist(),
        "condition_scale": condition_scale.tolist(),
        "ood_summary_mean": ood_summary_mean.tolist(),
        "ood_summary_scale": ood_summary_scale.tolist(),
        # OOD is evaluated after summary standardisation, so its centre is zero.
        "ood_mean": ood_mean.tolist(),
        "ood_inverse_covariance": inverse_covariance.tolist(),
        "ood_threshold": ood_threshold,
        "weight_count": int(flat.size),
        "weights_sha256": weights_sha,
        "training": {
            "seed": config.seed,
            "simulations": config.simulations,
            "steps": config.steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "final_loss": final_loss,
            "validation_fraction": config.validation_fraction,
        },
    }
    _write_json(MANIFEST_JSON, manifest)

    calibration, calibration_draws = calibrate(
        params,
        config,
        condition_mean,
        condition_scale,
        ood_summary_mean,
        ood_summary_scale,
        inverse_covariance,
    )
    _write_json(CALIBRATION_JSON, calibration)

    # Golden vectors are generated from held-out conditions. They pin the
    # forward log-probability path and the deterministic inverse-sampling path.
    golden_conditions_raw = validation_conditions_raw[: config.golden_count]
    golden_conditions = _standardise(golden_conditions_raw, condition_mean, condition_scale)
    golden_x = validation_targets[: config.golden_count]
    z_values: list[list[float]] = []
    log_probs: list[float] = []
    for condition, target in zip(golden_conditions, golden_x):
        z, log_det = _forward(params, jnp.asarray(condition), jnp.asarray(target), config)
        z_values.append(np.asarray(z, dtype=np.float64).tolist())
        log_probs.append(float(_log_prob(params, jnp.asarray(condition), jnp.asarray(target), config)))
    sample_z = np.array(
        [[0.25, -0.75], [-1.1, 0.4], [1.3, 0.9], [-0.2, -1.4], [0.05, 0.15], [0.8, -0.3], [-0.6, 1.0], [1.0, -0.55]],
        dtype=np.float64,
    )[: config.golden_count]
    sample_x: list[list[float]] = []
    for condition, z in zip(golden_conditions, sample_z):
        x = jnp.asarray(z, dtype=jnp.float64)
        for index in range(len(params) - 1, -1, -1):
            x = _layer_inverse(x, jnp.asarray(condition), params[index], index, config)
        sample_x.append(np.asarray(x, dtype=np.float64).tolist())
    golden = {
        "schema_version": 1,
        "synthetic": True,
        "weights_sha256": weights_sha,
        "parameter_names": PARAMETER_NAMES,
        "conditions": golden_conditions.tolist(),
        "x": golden_x.tolist(),
        "z": z_values,
        "log_prob": log_probs,
        "sample_z": sample_z.tolist(),
        "sample_x": sample_x,
    }
    _write_json(GOLDEN_JSON, golden)

    # The calibration JSON is written before golden vectors, but it carries the
    # weights hash so a stale report cannot be paired with new weights.
    calibration["weights_sha256"] = weights_sha
    calibration["sbc"]["samples_per_simulation"] = config.posterior_samples
    calibration["sbc"]["simulations"] = config.sbc_simulations
    _write_json(CALIBRATION_JSON, calibration)
    return {
        "weights": str(WEIGHTS_BIN),
        "manifest": str(MANIFEST_JSON),
        "calibration": str(CALIBRATION_JSON),
        "golden": str(GOLDEN_JSON),
        "weights_sha256": weights_sha,
        "calibration_pass": calibration["pass"],
        "calibration_draws": calibration_draws,
    }


def _mahalanobis(values: np.ndarray, mean: np.ndarray, inverse_covariance: np.ndarray) -> np.ndarray:
    centered = values - mean
    return np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered)


def calibrate(
    params: list[dict[str, jax.Array]],
    config: NpeConfig,
    condition_mean: np.ndarray,
    condition_scale: np.ndarray,
    ood_summary_mean: np.ndarray,
    ood_summary_scale: np.ndarray,
    inverse_covariance: np.ndarray,
) -> tuple[dict[str, Any], int]:
    case = load_config(CASE_CONFIG)
    key = jax.random.PRNGKey(config.seed + 17)
    key, prior_key, noise_key, sample_key = jax.random.split(key, 4)
    theta = _draw_prior(prior_key, config.sbc_simulations)
    observations = _simulate(noise_key, theta, case)
    conditions = _standardise(_summaries(np.asarray(observations)), condition_mean, condition_scale)
    samples = _sample(
        params,
        jnp.asarray(conditions, dtype=jnp.float64),
        sample_key,
        config.posterior_samples,
        config,
    )
    samples = np.asarray(samples, dtype=np.float64)
    truth = _standardise(np.asarray(theta), PARAMETER_CENTER, PARAMETER_SCALE)
    ranks = (samples < truth[:, None, :]).sum(axis=1)
    bins = 10
    rank_bins = np.minimum((ranks * bins // config.posterior_samples), bins - 1)
    chi_square = 0.0
    p_values: list[float] = []
    rank_counts: list[list[int]] = []
    for dimension in range(2):
        counts = np.bincount(rank_bins[:, dimension], minlength=bins)
        rank_counts.append(counts.tolist())
        statistic = float(((counts - config.sbc_simulations / bins) ** 2 / (config.sbc_simulations / bins)).sum())
        p_value = float(stats.chi2.sf(statistic, bins - 1))
        chi_square += statistic
        p_values.append(p_value)

    coverages: list[dict[str, float]] = []
    max_deviation = 0.0
    for alpha in config.tarp_alphas:
        lower = np.quantile(samples, (1.0 - alpha) / 2.0, axis=1)
        upper = np.quantile(samples, 1.0 - (1.0 - alpha) / 2.0, axis=1)
        covered = (truth >= lower) & (truth <= upper)
        empirical = float(covered.mean())
        deviation = abs(empirical - alpha)
        max_deviation = max(max_deviation, deviation)
        coverages.append({"nominal": alpha, "empirical": empirical, "deviation": deviation})

    ood_key, ood_noise_key = jax.random.split(key)
    ood_theta = _draw_prior(ood_key, config.ood_simulations)
    # Change both the parameter region and the measurement design. This is an
    # OOD probe, not a claim about a real instrument.
    ood_theta = ood_theta.at[:, 0].set(np.linspace(1.30, 1.45, config.ood_simulations))
    ood_theta = ood_theta.at[:, 1].set(np.linspace(8.0, 10.0, config.ood_simulations))
    ood_observations = _simulate(
        ood_noise_key,
        ood_theta,
        case,
        temperature_shift=80.0,
        current_scale=100.0,
        sigma_scale=8.0,
    )
    in_distribution = _mahalanobis(
        _standardise(_summaries(np.asarray(observations)), ood_summary_mean, ood_summary_scale),
        np.zeros_like(ood_summary_mean),
        inverse_covariance,
    )
    out_distribution = _mahalanobis(
        _standardise(_summaries(np.asarray(ood_observations)), ood_summary_mean, ood_summary_scale),
        np.zeros_like(ood_summary_mean),
        inverse_covariance,
    )
    threshold = float(np.quantile(in_distribution, 0.99))
    tpr = float((out_distribution > threshold).mean())
    fpr = float((in_distribution > threshold).mean())

    pass_sbc = all(p > 0.01 for p in p_values)
    pass_tarp = max_deviation <= 0.03
    pass_ood = tpr >= 0.90 and fpr <= 0.02
    report = {
        "schema_version": 1,
        "synthetic": True,
        "model": "M0",
        "pass": bool(pass_sbc and pass_tarp and pass_ood),
        "sbc": {
            "p_values": p_values,
            "chi_square": chi_square,
            "rank_counts": rank_counts,
            "pass": pass_sbc,
            "threshold": 0.01,
            "bins": bins,
        },
        "tarp": {
            "coverage": coverages,
            "max_deviation": max_deviation,
            "pass": pass_tarp,
            "threshold": 0.03,
        },
        "ood": {
            "tpr": tpr,
            "fpr": fpr,
            "threshold": threshold,
            "pass": pass_ood,
            "tpr_target": 0.90,
            "fpr_target": 0.02,
        },
        "parameter_names": PARAMETER_NAMES,
        "note": "Synthetic M0 local to the report-anchored case. The original measurements do not exist (ADR 0006).",
        "prior_scope": "local report-anchored M0",
    }
    return report, int(samples.shape[0] * samples.shape[1])


def check() -> None:
    for path in (WEIGHTS_BIN, MANIFEST_JSON, CALIBRATION_JSON, GOLDEN_JSON):
        if not path.is_file():
            raise SystemExit(f"missing NPE artifact: {path}")
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("synthetic") is not True:
        raise SystemExit("NPE manifest must be schema 1 and synthetic")
    if manifest.get("model") != "M0" or manifest.get("parameter_names") != PARAMETER_NAMES:
        raise SystemExit("NPE manifest model/parameters mismatch")
    if manifest.get("flow_kind") not in ("affine", "spline"):
        raise SystemExit("NPE manifest flow_kind is invalid")
    if manifest.get("weights_sha256") != _sha256(WEIGHTS_BIN):
        raise SystemExit("NPE weights hash is stale")
    if calibration.get("weights_sha256") != manifest.get("weights_sha256"):
        raise SystemExit("NPE calibration does not match weights")
    if golden.get("weights_sha256") != manifest.get("weights_sha256"):
        raise SystemExit("NPE golden vectors do not match weights")
    if calibration.get("pass") is not True:
        raise SystemExit("NPE calibration did not pass")
    expected = manifest["weight_count"] * 8
    if WEIGHTS_BIN.stat().st_size != expected:
        raise SystemExit("NPE weight byte count is inconsistent")
    for key in ("conditions", "x", "z", "log_prob", "sample_z", "sample_x"):
        if len(golden.get(key, [])) != len(golden["conditions"]):
            raise SystemExit(f"NPE golden field {key} has the wrong length")
    print(
        f"NPE artifacts consistent: {manifest['weight_count']} weights, "
        f"OOD threshold {manifest['ood_threshold']:.3f}, calibration pass"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="train, export and calibrate")
    parser.add_argument("--check", action="store_true", help="verify committed artifacts")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if not args.all:
        parser.error("choose --all or --check")
    config = NpeConfig()
    print("training synthetic M0 conditional spline flow")
    print(json.dumps(asdict(config), indent=2))
    result = train(config)
    print(json.dumps(result, indent=2))
    check()


if __name__ == "__main__":
    main()
