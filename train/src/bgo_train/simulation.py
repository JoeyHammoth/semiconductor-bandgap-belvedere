"""M0-M4 forward models and seeded synthetic observations.

The equations match docs/model.md.  All work is in SI internally; the gap is
carried in electronvolts because E_g / q is numerically volts.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .config import CaseConfig

K_BOLTZMANN = 1.380649e-23
ELEMENTARY_CHARGE = 1.602176634e-19
ACTIVATION_EXPONENT = 1.0


def thermal_voltage(temperature_k: jax.Array) -> jax.Array:
    return K_BOLTZMANN * temperature_k / ELEMENTARY_CHARGE


def gap_ev(model: str, parameters: dict[str, jax.Array], temperature_k: jax.Array) -> jax.Array:
    """Return the model's junction gap at each temperature, in eV."""

    if model in ("M0", "M1"):
        return parameters["eg_ev"]
    e_g0 = parameters["eg0_ev"]
    alpha = parameters["alpha_ev_per_k"]
    beta = parameters["beta_k"]
    return e_g0 - alpha * temperature_k**2 / (temperature_k + beta)


def forward_voltage(
    model: str,
    parameters: dict[str, jax.Array],
    temperature_k: jax.Array,
    current_a: jax.Array,
) -> jax.Array:
    """Terminal voltage for a forced forward current, including all model terms."""

    temperature_k = jnp.asarray(temperature_k, dtype=jnp.float64)
    current_a = jnp.asarray(current_a, dtype=jnp.float64)
    vt = thermal_voltage(temperature_k)
    if model == "M0":
        eta = jnp.asarray(0.0, dtype=jnp.float64)
    elif model in ("M3", "M4"):
        # M3/M4 reference convention: eta is held at its illustrative central
        # value so the single-current synthetic case can identify ideality.
        eta = jnp.asarray(3.0, dtype=jnp.float64)
    else:
        eta = jnp.asarray(parameters["eta"], dtype=jnp.float64)
    ideality = jnp.asarray(
        parameters["ideality"] if model in ("M3", "M4") else 1.0,
        dtype=jnp.float64,
    )
    series_resistance = jnp.asarray(
        parameters["series_resistance_ohm"] if model == "M4" else 0.0,
        dtype=jnp.float64,
    )
    eg = gap_ev(model, parameters, temperature_k)
    ln_is = (
        parameters["ln_b"]
        + eta * jnp.log(temperature_k)
        - eg / (ACTIVATION_EXPONENT * vt)
    )
    # log(1 + I/I_s), evaluated without overflow when I >> I_s.
    log_term = jnp.logaddexp(0.0, jnp.log(current_a) - ln_is)
    return current_a * series_resistance + ideality * vt * log_term


@dataclass(frozen=True)
class SyntheticDataset:
    temperature_k: NDArray[np.float64]
    voltage_v: NDArray[np.float64]
    noiseless_voltage_v: NDArray[np.float64]
    current_a: float
    voltage_sigma_v: float
    adc_lsb_v: float
    likelihood_sigma_v: float


def generate_dataset(config: CaseConfig) -> SyntheticDataset:
    """Generate deterministic synthetic observations from config + seed alone."""

    temperature_k = np.linspace(
        config.temperature_start_k,
        config.temperature_stop_k,
        config.points,
        dtype=np.float64,
    )
    truth = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in config.truth.items()}
    noiseless = np.asarray(
        forward_voltage(config.model, truth, temperature_k, config.current_a),
        dtype=np.float64,
    )
    rng = np.random.default_rng(config.seed)
    voltage = noiseless + rng.normal(0.0, config.voltage_sigma_v, size=config.points)
    if config.adc_lsb_v > 0.0:
        voltage = np.round(voltage / config.adc_lsb_v) * config.adc_lsb_v
    return SyntheticDataset(
        temperature_k=temperature_k,
        voltage_v=voltage.astype(np.float64),
        noiseless_voltage_v=noiseless,
        current_a=config.current_a,
        voltage_sigma_v=config.voltage_sigma_v,
        adc_lsb_v=config.adc_lsb_v,
        likelihood_sigma_v=config.likelihood_sigma_v,
    )


def observation_json(dataset: SyntheticDataset) -> dict[str, object]:
    return {
        "synthetic": True,
        "current_a": dataset.current_a,
        "temperature_k": [_round(value) for value in dataset.temperature_k],
        "voltage_v": [_round(value) for value in dataset.voltage_v],
        "noise": {
            "voltage_sigma_v": dataset.voltage_sigma_v,
            "adc_lsb_v": dataset.adc_lsb_v,
            "likelihood_sigma_v": dataset.likelihood_sigma_v,
        },
    }


def _round(value: float) -> float:
    return float(round(float(value), 10))
