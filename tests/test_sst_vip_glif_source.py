"""Nonexecuting source compatibility checks cannot alter numerical functions."""

from __future__ import annotations

import ast

import pytest

from smart_robustness.validation.sst_vip_glif_source import numerical_module_ast

SOURCE = """import simplejson as json
import allensdk.core.json_utilities as ju
class GlifNeuron:
    def __str__(self): return json.dumps(self, default=ju.json_handler)
    def dynamics(self, v): return v + 1
"""


def test_only_registered_imports_removed_without_execution():
    tree, report = numerical_module_ast(SOURCE)
    assert ast.dump(tree.body[0]) == ast.dump(ast.parse(SOURCE).body[2])
    assert not report["numerical_ast_nodes_modified"] and not report["native_source_executed"]


def test_missing_import_rejected():
    with pytest.raises(ValueError, match="exact"):
        numerical_module_ast(SOURCE.replace("import simplejson as json\n", ""))


def test_numerical_serialization_use_rejected():
    with pytest.raises(ValueError, match="outside"):
        numerical_module_ast(SOURCE.replace("return v + 1", "return json.dumps(v)"))
