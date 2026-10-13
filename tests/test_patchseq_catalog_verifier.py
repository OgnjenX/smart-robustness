"""Independent catalog reconstruction rejects corruption and false promotion."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_patchseq_catalog.py"
spec = importlib.util.spec_from_file_location("catalog_verifier", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def url(page):
    return module.BASE + "assets/?" + urlencode({"metadata": "false", "order": "path", "page_size": 500, "page": page})


def source():
    assets = [{"asset_id": str(UUID(int=i+1)), "path": f"sub-10/record-{i:04d}.nwb", "size": 20} for i in range(4435)]
    pages = []
    for page in range(1, 10):
        document = {"results": assets[(page-1)*500:page*500], "previous": url(page-1) if page > 1 else None,
                    "next": url(page+1) if page < 9 else None}
        if page == 1:
            document["count"] = 4435
        elif page % 2 == 0:
            document["count"] = None
        pages.append(document)
    version = {"identifier": "DANDI:000020", "version": "0.210913.1639",
               "assetsSummary": {"numberOfFiles": 4435, "numberOfBytes": 88700, "numberOfSubjects": 1}}
    old = [{"specimen_id": i, "included": i <= 111} for i in range(1, 2334)]
    return version, pages, old


def test_independent_complete_reconstruction_and_count_missingness():
    result = module.reconstruct(*source())
    assert result["summary"]["assets"] == 4435
    assert result["summary"]["total_asset_bytes"] == 88700
    assert result["summary"]["potential_prior_included_specimen_ID_matches"] == [10]
    assert not result["summary"]["specimen_identity_confirmed"]
    assert not result["summary"]["biological_eligibility_established"]
    assert [item["count_is_reported"] for item in result["count_evidence"]] == [True] + [False]*8
    assert [item["count_key_present"] for item in result["count_evidence"]][1:3] == [True, False]


@pytest.mark.parametrize("fault", ["first-count", "later-count", "short-page", "missing-page", "duplicate-ID",
                                  "duplicate-path", "wrong-size", "wrong-URL", "wrong-order", "typed-size"])
def test_full_catalog_and_source_count_failures(fault):
    version, pages, old = source()
    if fault == "first-count":
        pages[0].pop("count")
    elif fault == "later-count":
        pages[1]["count"] = 4434
    elif fault == "short-page":
        pages[1]["results"].pop()
    elif fault == "missing-page":
        pages.pop()
    elif fault == "duplicate-ID":
        pages[1]["results"][0]["asset_id"] = pages[0]["results"][0]["asset_id"]
    elif fault == "duplicate-path":
        pages[1]["results"][0]["path"] = pages[0]["results"][0]["path"]
    elif fault == "wrong-size":
        pages[1]["results"][0]["size"] = 21
    elif fault == "wrong-URL":
        pages[1]["next"] = "https://example.com/recording.nwb"
    elif fault == "wrong-order":
        pages[1]["results"].reverse()
    else:
        pages[1]["results"][0]["size"] = 20.0
    with pytest.raises(ValueError):
        module.reconstruct(version, pages, old)


@pytest.mark.parametrize("field,value", [("identifier", "DANDI:000021"), ("version", "draft")])
def test_version_identity_not_inferred_from_catalog(field, value):
    version, pages, old = source()
    version[field] = value
    with pytest.raises(ValueError):
        module.reconstruct(version, pages, old)


@pytest.mark.parametrize("field,value", [("numberOfFiles", 4435.0), ("numberOfBytes", True), ("numberOfSubjects", 0)])
def test_declaration_types_and_bounds(field, value):
    version, pages, old = source()
    version["assetsSummary"][field] = value
    with pytest.raises(ValueError):
        module.reconstruct(version, pages, old)


def test_duplicate_raw_JSON_keys_are_not_silently_selected(tmp_path):
    path = tmp_path / "response.json"
    path.write_text('{"count":4435,"count":1}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        module.read_json(path)


def test_typed_comparison_rejects_false_identity_promotion():
    result = module.reconstruct(*source())
    changed = copy.deepcopy(result["summary"])
    changed["specimen_identity_confirmed"] = 0
    assert not module.typed_equal(changed, result["summary"])
    changed["specimen_identity_confirmed"] = True
    assert not module.typed_equal(changed, result["summary"])


def test_no_terminal_evidence_creates_no_verification_output(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MANIFEST", tmp_path / "absent")
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "output")
    with pytest.raises(ValueError, match="terminal catalog evidence absent"):
        module.main()
    assert not module.OUTPUT.exists()


def test_bounded_json_source_read(tmp_path):
    path = tmp_path / "too-large.json"
    path.write_bytes(b" " * 2097153)
    with pytest.raises(ValueError, match="byte bound"):
        module.read_json(path)


def test_independent_result_does_not_mutate_pages():
    data = source()
    before = json.dumps(data)
    module.reconstruct(*data)
    assert json.dumps(data) == before
