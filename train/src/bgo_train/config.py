"""Validated configuration for synthetic M0-M4 reference cases.

Configs are deliberately small and explicit.  A case is reproducible from the
config file plus its seed; no measured-data file or data hash is involved.
"""

from __future__ import annotations

import hashlib
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODELS = ("M0", "M1", "M2", "M3", "M4")

TRUTH_FIELDS: dict[str, tuple[str, ...]] = {
    "M0": ("eg_ev", "ln_b"),
    "M1": ("eg_ev", "ln_b", "eta"),
    "M2": ("eg0_ev", "ln_b", "eta", "alpha_ev_per_k", "beta_k"),
    # M3/M4 anchor eta at its illustrative central value: one constant-current
    # sweep cannot separate eta from the terminal ideality factor.
    "M3": ("eg0_ev", "ln_b", "alpha_ev_per_k", "beta_k", "ideality"),
    "M4": (
        "eg0_ev",
        "ln_b",
        "alpha_ev_per_k",
        "beta_k",
        "ideality",
        "series_resistance_ohm",
    ),
}

_TOP_LEVEL = {
    "schema_version",
    "case_id",
    "model",
    "synthetic",
    "description",
    "seed",
    "anchor",
    "data",
    "noise",
    "truth",
}
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class Anchor:
    """A labelled provenance anchor for a synthetic case."""

    kind: str
    source: str
    label: str
    note: str
    value_ev: float | None = None

    def as_json(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "kind": self.kind,
            "source": self.source,
            "label": self.label,
            "note": self.note,
        }
        if self.value_ev is not None:
            value["value_ev"] = self.value_ev
        return value


@dataclass(frozen=True)
class CaseConfig:
    """One synthetic case, including its truth and observation contract."""

    case_id: str
    model: str
    description: str
    seed: int
    synthetic: bool
    current_a: float
    temperature_start_k: float
    temperature_stop_k: float
    points: int
    voltage_sigma_v: float
    adc_lsb_v: float
    truth: dict[str, float]
    anchor: Anchor | None
    source_name: str
    config_sha256: str

    @property
    def free_parameters(self) -> tuple[str, ...]:
        return TRUTH_FIELDS[self.model]

    @property
    def seed_sha256(self) -> str:
        return hashlib.sha256(str(self.seed).encode("ascii")).hexdigest()

    @property
    def likelihood_sigma_v(self) -> float:
        # Quantisation is approximately uniform with variance LSB^2 / 12.
        return math.sqrt(self.voltage_sigma_v**2 + self.adc_lsb_v**2 / 12.0)

    def as_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model": self.model,
            "description": self.description,
            "seed": self.seed,
            "synthetic": self.synthetic,
            "source": self.source_name,
            "config_sha256": self.config_sha256,
            "data": {
                "current_a": self.current_a,
                "temperature_start_k": self.temperature_start_k,
                "temperature_stop_k": self.temperature_stop_k,
                "points": self.points,
            },
            "noise": {
                "voltage_sigma_v": self.voltage_sigma_v,
                "adc_lsb_v": self.adc_lsb_v,
                "likelihood_sigma_v": self.likelihood_sigma_v,
            },
            "truth": dict(self.truth),
            "anchor": None if self.anchor is None else self.anchor.as_json(),
        }


def _number(table: dict[str, Any], key: str, where: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where}.{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where}.{key} must be finite")
    return result


def _integer(table: dict[str, Any], key: str, where: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}.{key} must be an integer")
    return value


def _text(table: dict[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def _table(table: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = table.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{where}.{key} must be a table")
    return value


def _parse_anchor(raw: MappingLike, where: str = "anchor") -> Anchor:
    table = _table(raw, "anchor", where) if "anchor" in raw else raw
    allowed = {"kind", "source", "label", "note", "value_ev"}
    unknown = set(table) - allowed
    if unknown:
        raise ValueError(f"{where} has unknown fields: {sorted(unknown)}")
    kind = _text(table, "kind", where)
    source = _text(table, "source", where)
    label = _text(table, "label", where)
    note = _text(table, "note", where)
    if label != "synthetic":
        raise ValueError(f"{where}.label must be 'synthetic'")
    value = None
    if "value_ev" in table:
        value = _number(table, "value_ev", where)
        if value <= 0.0:
            raise ValueError(f"{where}.value_ev must be positive")
    return Anchor(kind=kind, source=source, label=label, note=note, value_ev=value)


MappingLike = dict[str, Any]


def load_config(path: Path) -> CaseConfig:
    """Load and validate one TOML case, including its model-specific truth."""

    raw_bytes = path.read_bytes()
    try:
        raw = tomllib.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"{path}: invalid TOML: {error}") from error

    unknown = set(raw) - _TOP_LEVEL
    if unknown:
        raise ValueError(f"{path}: unknown top-level fields: {sorted(unknown)}")
    if raw.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")

    case_id = _text(raw, "case_id", str(path))
    if case_id != path.stem:
        raise ValueError(f"{path}: case_id must match the file stem ({path.stem})")
    if not _CASE_ID.fullmatch(case_id):
        raise ValueError(f"{path}: case_id must match {_CASE_ID.pattern}")
    model = _text(raw, "model", str(path)).upper()
    if model not in MODELS:
        raise ValueError(f"{path}: model must be one of {MODELS}")
    if raw.get("synthetic") is not True:
        raise ValueError(f"{path}: synthetic must be true; measured data is not allowed")
    description = _text(raw, "description", str(path))
    seed = _integer(raw, "seed", str(path))
    if seed < 0:
        raise ValueError(f"{path}: seed must be non-negative")

    data = _table(raw, "data", str(path))
    allowed_data = {
        "current_a",
        "temperature_start_k",
        "temperature_stop_k",
        "points",
    }
    if set(data) - allowed_data:
        raise ValueError(f"{path}: unknown data fields: {sorted(set(data) - allowed_data)}")
    current_a = _number(data, "current_a", "data")
    t_start = _number(data, "temperature_start_k", "data")
    t_stop = _number(data, "temperature_stop_k", "data")
    points = _integer(data, "points", "data")
    if current_a <= 0.0:
        raise ValueError(f"{path}: data.current_a must be positive")
    if t_start <= 0.0 or t_stop <= t_start:
        raise ValueError(f"{path}: temperature range must be positive and increasing")
    if points < 2:
        raise ValueError(f"{path}: data.points must be at least 2")

    noise = _table(raw, "noise", str(path))
    allowed_noise = {"voltage_sigma_v", "adc_lsb_v"}
    if set(noise) - allowed_noise:
        raise ValueError(f"{path}: unknown noise fields: {sorted(set(noise) - allowed_noise)}")
    sigma = _number(noise, "voltage_sigma_v", "noise")
    lsb = _number(noise, "adc_lsb_v", "noise")
    if sigma <= 0.0:
        raise ValueError(f"{path}: noise.voltage_sigma_v must be positive")
    if lsb < 0.0:
        raise ValueError(f"{path}: noise.adc_lsb_v cannot be negative")

    truth = _table(raw, "truth", str(path))
    required = TRUTH_FIELDS[model]
    if set(truth) != set(required):
        raise ValueError(
            f"{path}: truth fields for {model} must be exactly {list(required)}"
        )
    parsed_truth = {key: _number(truth, key, "truth") for key in required}
    _validate_truth(path, model, parsed_truth)

    anchor = _parse_anchor(raw) if "anchor" in raw else None
    return CaseConfig(
        case_id=case_id,
        model=model,
        description=description,
        seed=seed,
        synthetic=True,
        current_a=current_a,
        temperature_start_k=t_start,
        temperature_stop_k=t_stop,
        points=points,
        voltage_sigma_v=sigma,
        adc_lsb_v=lsb,
        truth=parsed_truth,
        anchor=anchor,
        source_name=path.name,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _validate_truth(path: Path, model: str, truth: dict[str, float]) -> None:
    if truth.get("eg_ev", truth.get("eg0_ev", 0.0)) <= 0.0:
        raise ValueError(f"{path}: gap must be positive")
    if "eta" in truth and not 1.5 <= truth["eta"] <= 5.0:
        raise ValueError(f"{path}: eta must lie in the model prior support [1.5, 5]")
    if "alpha_ev_per_k" in truth and truth["alpha_ev_per_k"] <= 0.0:
        raise ValueError(f"{path}: alpha_ev_per_k must be positive")
    if "beta_k" in truth and truth["beta_k"] <= 0.0:
        raise ValueError(f"{path}: beta_k must be positive")
    if "ideality" in truth and truth["ideality"] <= 0.0:
        raise ValueError(f"{path}: ideality must be positive")
    if "series_resistance_ohm" in truth and truth["series_resistance_ohm"] < 0.0:
        raise ValueError(f"{path}: series_resistance_ohm cannot be negative")


def load_all(config_dir: Path) -> list[CaseConfig]:
    paths = sorted(config_dir.glob("*.toml"))
    if not paths:
        raise ValueError(f"no TOML cases found in {config_dir}")
    cases = [load_config(path) for path in paths]
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id values")
    return cases
