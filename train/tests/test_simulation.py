from dataclasses import replace
from pathlib import Path

import numpy as np

from bgo_train.config import load_all
from bgo_train.simulation import forward_voltage, generate_dataset

ROOT = Path(__file__).resolve().parents[1]
CASES = load_all(ROOT / "configs")


def _case(case_id: str):
    return next(case for case in CASES if case.case_id == case_id)


def test_generation_is_reproducible_from_config_and_seed() -> None:
    case = _case("m1")
    first = generate_dataset(case)
    second = generate_dataset(case)
    assert np.array_equal(first.temperature_k, second.temperature_k)
    assert np.array_equal(first.voltage_v, second.voltage_v)


def test_a_different_seed_changes_only_the_noise() -> None:
    case = _case("m1")
    changed = replace(case, seed=case.seed + 1)
    first = generate_dataset(case)
    second = generate_dataset(changed)
    assert np.array_equal(first.noiseless_voltage_v, second.noiseless_voltage_v)
    assert not np.array_equal(first.voltage_v, second.voltage_v)


def test_observations_are_on_the_adc_grid() -> None:
    dataset = generate_dataset(_case("m4"))
    assert np.allclose(dataset.voltage_v / dataset.adc_lsb_v, np.round(dataset.voltage_v / dataset.adc_lsb_v))


def test_m0_noiseless_curve_is_linear() -> None:
    case = _case("m0-report-si")
    dataset = generate_dataset(case)
    slope, intercept = np.polyfit(dataset.temperature_k, dataset.noiseless_voltage_v, 1)
    residual = dataset.noiseless_voltage_v - (slope * dataset.temperature_k + intercept)
    assert slope < 0.0
    assert np.max(np.abs(residual)) < 1e-9


def test_m2_gap_decreases_with_temperature() -> None:
    case = _case("m2")
    dataset = generate_dataset(case)
    assert dataset.noiseless_voltage_v[0] > dataset.noiseless_voltage_v[-1]


def test_m3_ideality_raises_the_voltage() -> None:
    case = _case("m3")
    dataset = generate_dataset(case)
    ideal = dict(case.truth)
    ideal["ideality"] = 1.0
    base = forward_voltage(case.model, ideal, dataset.temperature_k[:1], case.current_a)
    full = forward_voltage(case.model, case.truth, dataset.temperature_k[:1], case.current_a)
    assert float(full[0]) > float(base[0])


def test_m4_series_resistance_adds_i_r() -> None:
    case = _case("m4")
    dataset = generate_dataset(case)
    without = dict(case.truth)
    without["series_resistance_ohm"] = 0.0
    base = forward_voltage(case.model, without, dataset.temperature_k[:1], case.current_a)
    full = forward_voltage(case.model, case.truth, dataset.temperature_k[:1], case.current_a)
    assert np.isclose(float(full[0] - base[0]), case.current_a * 8.0, rtol=1e-12)


def test_all_cases_are_finite() -> None:
    for case in CASES:
        dataset = generate_dataset(case)
        assert np.isfinite(dataset.noiseless_voltage_v).all()
        assert np.isfinite(dataset.voltage_v).all()
