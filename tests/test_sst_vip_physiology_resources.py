"""Resource selection and body-free size preflight safeguards."""

from __future__ import annotations

import importlib.util
import sys
from email.message import Message
from pathlib import Path

import pytest

from smart_robustness.validation.sst_vip_physiology_resources import content_length, resource_plan


def specimen():
    def file(fid, aid, atype, kind):
        return {
            "id": fid,
            "attachable_id": aid,
            "attachable_type": atype,
            "download_link": f"/api/v2/well_known_file_download/{fid}",
            "well_known_file_type": {"name": kind},
        }

    return {
        "id": 1,
        "ephys_result": {"id": 2, "well_known_files": [file(3, 2, "EphysResult", "NWBDownload")]},
        "neuron_reconstructions": [
            {
                "id": 4,
                "well_known_files": [file(5, 4, "NeuronReconstruction", "3DNeuronReconstruction")],
            },
            {
                "id": 6,
                "well_known_files": [file(7, 6, "NeuronReconstruction", "3DNeuronReconstruction")],
            },
        ],
    }


def test_all_reconstructions_retained():
    plan = resource_plan([specimen()])
    assert [r["file_id"] for r in plan["resources"]] == [3, 5, 7]


def test_missing_morphology_not_dropped():
    s = specimen()
    s["neuron_reconstructions"] = []
    assert resource_plan([s])["missing_morphology_specimen_ids"] == [1]


def test_duplicate_and_wrong_attachment_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        resource_plan([specimen(), specimen()])
    s = specimen()
    s["ephys_result"]["well_known_files"][0]["attachable_id"] = 999
    with pytest.raises(ValueError, match="identity"):
        resource_plan([s])


@pytest.mark.parametrize("value", [None, "", "0", "-1", "1.2", "NaN", "１２"])
def test_unresolved_length_not_zero(value):
    with pytest.raises(ValueError):
        content_length(value)


def test_positive_length():
    assert content_length("123") == 123


def test_head_does_not_read_body(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "physiology_preflight_test", Path("scripts/preflight_sst_vip_physiology_resources.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    resource = resource_plan([specimen()])["resources"][0]
    url = "https://api.brain-map.org" + resource["resource"]["download_link"]

    class Response:
        status = 200
        headers = Message()
        headers["Content-Length"] = "123"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def geturl(self):
            return url

        def read(self, *args):
            pytest.fail("HEAD audit must not read body")

    class Opener:
        def open(self, request, timeout):
            assert request.get_method() == "HEAD"
            return Response()

    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    result = module.head_one(
        resource, reg={"source_base": "https://api.brain-map.org", "timeout_seconds": 30}
    )
    assert result["content_length"] == 123 and result["status"] == "size-resolved"
    assert not result["body_read"]
