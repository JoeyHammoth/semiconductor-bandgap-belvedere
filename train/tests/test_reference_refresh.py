import json
import shutil
from pathlib import Path

from bgo_train.reference import compare_references

ROOT = Path(__file__).resolve().parents[1]


def _copy_references(tmp_path: Path) -> Path:
    candidate = tmp_path / "references"
    shutil.copytree(ROOT / "references", candidate)
    return candidate


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_committed_references_compare_to_themselves() -> None:
    errors, _ = compare_references(ROOT / "references", ROOT / "references")
    assert errors == []


def test_refresh_accepts_monte_carlo_scatter(tmp_path: Path) -> None:
    candidate = _copy_references(tmp_path)
    path = candidate / "m0-report-si.json"
    document = _load(path)
    reference = document["reference"]

    for name, value in reference["mean"].items():
        reference["mean"][name] = value + 0.01 * reference["sd"][name]
    for name in reference["sd"]:
        reference["sd"][name] *= 1.01
    for quantile in reference["quantiles"].values():
        for name, value in quantile.items():
            quantile[name] = value + 0.01 * reference["sd"][name]
    reference["correlation"]["matrix"][0][1] += 0.001
    reference["correlation"]["matrix"][1][0] += 0.001
    _write(path, document)

    errors, stats = compare_references(ROOT / "references", candidate)
    assert errors == []
    assert stats["mean"] > 0.0
    assert stats["sd"] > 0.0


def test_refresh_rejects_a_meaningful_mean_shift(tmp_path: Path) -> None:
    candidate = _copy_references(tmp_path)
    path = candidate / "m0-report-si.json"
    document = _load(path)
    reference = document["reference"]
    reference["mean"]["eg_ev"] += 0.2 * reference["sd"]["eg_ev"]
    _write(path, document)

    errors, _ = compare_references(ROOT / "references", candidate)
    assert any("mean eg_ev" in error for error in errors)
