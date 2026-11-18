from dataclasses import replace
from pathlib import Path

import pytest

from bgo_train.config import MODELS, load_all, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_cases_cover_m0_to_m4_and_are_synthetic() -> None:
    cases = load_all(ROOT / "configs")
    assert {case.model for case in cases} == set(MODELS)
    assert all(case.synthetic and case.anchor is not None for case in cases)
    assert all(case.anchor.label == "synthetic" for case in cases)
    assert sum(case.anchor.kind == "report-summary" for case in cases) >= 2


def test_case_ids_and_truth_follow_the_schema() -> None:
    for case in load_all(ROOT / "configs"):
        assert case.case_id == case.source_name.removesuffix(".toml")
        assert set(case.truth) == set(case.free_parameters)
        assert case.likelihood_sigma_v >= case.voltage_sigma_v


def test_non_synthetic_case_is_rejected(tmp_path: Path) -> None:
    source = ROOT / "configs" / "m0-report-si.toml"
    target = tmp_path / source.name
    target.write_text(
        source.read_text(encoding="utf-8").replace("synthetic = true", "synthetic = false"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="synthetic must be true"):
        load_config(target)


def test_seed_and_config_hash_are_part_of_the_case() -> None:
    case = load_config(ROOT / "configs" / "m0-report-si.toml")
    changed = replace(case, seed=case.seed + 1)
    assert changed.seed_sha256 != case.seed_sha256
    assert case.config_sha256 == load_config(ROOT / "configs" / "m0-report-si.toml").config_sha256
