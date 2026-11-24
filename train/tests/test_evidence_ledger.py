import copy
import json
import math
from pathlib import Path

from bgo_train.evidence import compare_ledger, derive_ledger

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT.parent / "web" / "static" / "data" / "evidence-ledger.json"


def _load() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def test_committed_ledger_is_synthetic_and_internally_consistent() -> None:
    document = _load()
    assert document["synthetic"] is True
    assert document["dataset"]["case_id"] == "m1"
    assert document["models"][0]["id"] == "M0"
    assert document["models"][1]["id"] == "M1"

    comparison, verdict = derive_ledger(document["models"][0], document["models"][1])
    assert verdict == document["verdict"]
    for key in (
        "delta_log_evidence",
        "log10_bayes_factor",
        "elpd_difference",
        "elpd_difference_se",
    ):
        assert math.isclose(
            comparison[key],
            document["comparison"][key],
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    assert compare_ledger(document, copy.deepcopy(document)) == []


def test_ledger_check_rejects_stale_evidence() -> None:
    document = _load()
    stale = copy.deepcopy(document)
    stale["models"][0]["log_evidence"] += 1.0

    errors = compare_ledger(document, stale)

    assert any("models.M0.log_evidence" in error for error in errors)


def test_ledger_check_rejects_a_stale_verdict() -> None:
    document = _load()
    stale = copy.deepcopy(document)
    stale["verdict"] += " Stale."

    errors = compare_ledger(document, stale)

    assert any("verdict" in error for error in errors)


def test_ledger_check_rejects_engine_drift() -> None:
    document = _load()
    stale = copy.deepcopy(document)
    stale["settings"]["engine"] = "dynesty 0.0.0"

    errors = compare_ledger(document, stale)

    assert any("settings.engine" in error for error in errors)
