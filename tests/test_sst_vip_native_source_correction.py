"""Static-only tree identity checks and no-execution source parsing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def collector(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    name = "native_source_correction_test"
    spec = importlib.util.spec_from_file_location(
        name, Path("scripts/collect_sst_vip_native_source_correction.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_exact_complete_tree(collector):
    entry = {"path": "utils.py", "type": "blob", "sha": "abc"}
    tree = {"sha": "revision", "truncated": False, "tree": [entry]}
    assert collector.validate_tree(tree, revision="revision", path="utils.py") == entry


@pytest.mark.parametrize(
    "tree",
    [
        {"sha": "wrong", "truncated": False, "tree": []},
        {"sha": "revision", "truncated": True, "tree": []},
        {"sha": "revision", "truncated": False, "tree": []},
        {"sha": "revision", "truncated": False, "tree": [{"path": "utils.py", "type": "tree"}]},
    ],
)
def test_changed_truncated_missing_and_nonfile_tree_rejected(collector, tree):
    with pytest.raises(ValueError):
        collector.validate_tree(tree, revision="revision", path="utils.py")


def test_static_parse_does_not_execute_source(collector):
    source = "raise RuntimeError('must not execute')\nclass Utils:\n    def setup(self): pass\n"
    assert collector.inspect_classes(source) == {"Utils": ["setup"]}
