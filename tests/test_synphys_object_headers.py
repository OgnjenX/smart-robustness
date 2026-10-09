"""Header acquisition must not expand into database downloads."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("object_headers", SCRIPTS / "collect_synphys_object_headers.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    names = [f"synphys_r2.1_{variant}.sqlite" for variant in ("small", "medium", "full")]
    return {"objects": names}, {"catalog": {
        "default_url_path": "https://allen-synphys.s3-us-west-2.amazonaws.com",
        "entries": [{"file": name} for name in names],
    }}


def test_exact_declared_urls():
    reg, manifest = fixture()
    urls = module.object_urls(reg, manifest)
    assert len(urls) == 3
    assert all(url.endswith(name) for url, name in zip(urls, reg["objects"], strict=True))


@pytest.mark.parametrize("change", ["host", "missing", "duplicate", "unknown"])
def test_reject_expanded_scope(change):
    reg, manifest = fixture()
    if change == "host":
        manifest["catalog"]["default_url_path"] = "https://example.com"
    elif change == "missing":
        manifest["catalog"]["entries"].pop()
    elif change == "duplicate":
        reg["objects"][1] = reg["objects"][0]
    else:
        reg["objects"][0] = "other.sqlite"
        manifest["catalog"]["entries"].append({"file": "other.sqlite"})
    with pytest.raises(ValueError):
        module.object_urls(reg, manifest)
