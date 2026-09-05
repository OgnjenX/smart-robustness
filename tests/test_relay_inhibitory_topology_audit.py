import runpy
from pathlib import Path

import pytest

audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_relay_inhibitory_topology.py"))["audit"]
source_path = Path(__file__).resolve().parents[1] / "tmp/modeldb-112923/versaceGrossberg2008/SMART.nml"


def test_source_checksum_is_required():
    with pytest.raises(ValueError, match="checksum"):
        audit(b"<neuroml/>")


@pytest.mark.skipif(not source_path.exists(), reason="original archive is local, not redistributed")
def test_recovered_relay_inhibition_and_independent_gaussian_agree():
    report = audit(source_path.read_bytes())
    assert [p["inputs_per_cell"] for p in report["projections"]] == [81, 81, 5, 81]
    assert report["source_fields_and_selected_topology_consistent"]
    assert not report["baseline_promoted"]
