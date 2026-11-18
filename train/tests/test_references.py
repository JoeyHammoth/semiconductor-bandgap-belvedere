import json
from pathlib import Path

from bgo_train.reference import check_references

ROOT = Path(__file__).resolve().parents[1]


def test_committed_references_are_consistent() -> None:
    check_references(ROOT / "configs", ROOT / "references")


def test_manifest_is_synthetic_and_has_no_data_hash() -> None:
    manifest = json.loads((ROOT / "references" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True
    assert "data_sha256" not in json.dumps(manifest)
    assert len(manifest["cases"]) == 6
    assert all(case["synthetic"] is True for case in manifest["cases"])


def test_reference_documents_contain_diagnostics_and_observations() -> None:
    for path in sorted((ROOT / "references").glob("*.json")):
        if path.name == "manifest.json":
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["case"]["synthetic"] is True
        assert document["observations"]["synthetic"] is True
        diagnostics = document["reference"]["diagnostics"]
        assert diagnostics["passes"] is True
        assert diagnostics["divergences"] == 0
        assert document["reference"]["correlation"]["matrix"]
