"""Validate explicitly bounded serialization-import omission; never execute here."""

from __future__ import annotations

import ast
import hashlib


def numerical_module_ast(source: str):
    original = ast.parse(source)
    omitted, retained = [], []
    for node in original.body:
        if isinstance(node, ast.Import) and len(node.names) == 1:
            alias = node.names[0]
            if (alias.name, alias.asname) in {
                ("simplejson", "json"),
                ("allensdk.core.json_utilities", "ju"),
            }:
                omitted.append((alias.name, alias.asname))
                continue
        retained.append(node)
    if sorted(omitted) != [("allensdk.core.json_utilities", "ju"), ("simplejson", "json")]:
        raise ValueError("exact registered serialization imports not found")
    for node in retained:
        children = (
            node.body if isinstance(node, ast.ClassDef) and node.name == "GlifNeuron" else [node]
        )
        for child in children:
            if (
                isinstance(child, ast.FunctionDef)
                and child.name == "__str__"
                and isinstance(node, ast.ClassDef)
                and node.name == "GlifNeuron"
            ):
                continue
            if any(
                isinstance(item, ast.Name) and item.id in {"json", "ju"} for item in ast.walk(child)
            ):
                raise ValueError("serialization dependency used outside GlifNeuron.__str__")
    result = ast.Module(body=retained, type_ignores=original.type_ignores)
    return result, {
        "omitted_imports": omitted,
        "retained_ast_sha256": hashlib.sha256(ast.dump(result).encode()).hexdigest(),
        "numerical_ast_nodes_modified": False,
        "native_source_executed": False,
    }
