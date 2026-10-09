"""Independent acquisition checks use synthetic objects, never real outcomes."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_synphys_medium_acquisition.py"
spec = importlib.util.spec_from_file_location("independent_medium_verifier", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    path = tmp_path / "synthetic.sqlite"
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE "quoted""table" (id INTEGER PRIMARY KEY, data BLOB)')
        db.execute('INSERT INTO "quoted""table" VALUES (1,?)', (b"must-not-read",))
    raw = path.read_bytes()
    schema = module.reconstruct_schema(path)
    manifest = {"status": "medium-acquired-schema-only", "client_exit_code": 0,
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "schema": schema}
    return path, manifest


def test_exact_object_and_all_schema_fields(tmp_path):
    path, manifest = fixture(tmp_path)
    result = module.verify_object(path, manifest, path.stat().st_size)
    assert result["identity"]["sqlite_signature"]
    assert result["identity"]["sha256"] == manifest["sha256"]
    assert result["schema"]["data_rows_read"] is False


@pytest.mark.parametrize("field,value", [("status", "failure-retained"), ("client_exit_code", 28),
                                        ("bytes", 1), ("sha256", "wrong"), ("schema", {})])
def test_partial_tampered_or_changed_schema_rejected(tmp_path, field, value):
    path, manifest = fixture(tmp_path)
    manifest[field] = value
    with pytest.raises(ValueError):
        module.verify_object(path, manifest, path.stat().st_size)


def test_column_level_schema_difference_rejected(tmp_path):
    path, manifest = fixture(tmp_path)
    manifest = copy.deepcopy(manifest)
    manifest["schema"]["tables"][0]["columns"][0][1] = "changed-id"
    with pytest.raises(ValueError, match="schema"):
        module.verify_object(path, manifest, path.stat().st_size)


def test_schema_guard_denies_data_reads_and_writes(tmp_path):
    path, _ = fixture(tmp_path)
    with sqlite3.connect(path) as db:
        db.set_authorizer(module.schema_guard)
        assert db.execute("SELECT name FROM sqlite_master").fetchall()
        for query in ['SELECT id FROM "quoted""table"', 'SELECT data FROM "quoted""table"',
                      'DELETE FROM "quoted""table"', "PRAGMA user_version"]:
            with pytest.raises(sqlite3.DatabaseError):
                db.execute(query)


def test_wrong_signature_rejected_before_sqlite_open(tmp_path):
    path = tmp_path / "not-sqlite"
    raw = b"not SQLite format"
    path.write_bytes(raw)
    manifest = {"status": "medium-acquired-schema-only", "client_exit_code": 0,
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "schema": {}}
    with pytest.raises(ValueError, match="identity"):
        module.verify_object(path, manifest, len(raw))


def test_identity_read_requests_are_bounded():
    raw = b"SQLite format 3\x00" + b"x" * 1048580

    class Guarded(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 1048576
            return super().read(size)

    class SyntheticPath:
        def open(self, mode):
            assert mode == "rb"
            return Guarded(raw)

    result = module.object_identity(SyntheticPath())
    assert result["bytes"] == len(raw)
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()


def source_fixture(tmp_path):
    reg = yaml.safe_load(module.REGISTRATION.read_text())
    for field in ("header_manifest", "certificate_authority_bundle"):
        path = tmp_path / field
        path.write_text("synthetic-lineage-only")
        reg[field] = str(path)
        key = "certificate_authority_sha256" if field == "certificate_authority_bundle" else field + "_sha256"
        reg[key] = module.checksum(path)
    registration = tmp_path / "registration.yaml"
    registration.write_text(yaml.safe_dump(reg))
    manifest = {"registration_sha256": module.checksum(registration),
                "collector_sha256": module.checksum(SCRIPT.with_name("acquire_synphys_medium_schema.py")),
                "schema_inspector_sha256": module.checksum(SCRIPT.with_name("acquire_synphys_small_schema.py")),
                "raw_path": "results/synphys-medium-schema-1131/synphys_r2.1_medium.sqlite",
                "row_values_read": False}
    return registration, manifest


def test_source_dependencies_without_opening_real_database(tmp_path):
    registration, manifest = source_fixture(tmp_path)
    reg, path = module.verify_sources(registration, manifest)
    assert reg["expected_bytes"] == 11125997568
    assert str(path) == manifest["raw_path"]


@pytest.mark.parametrize("field", ["registration_sha256", "collector_sha256", "schema_inspector_sha256",
                                  "raw_path", "row_values_read"])
def test_each_source_boundary_is_checked(tmp_path, field):
    registration, manifest = source_fixture(tmp_path)
    manifest[field] = "changed"
    with pytest.raises(ValueError):
        module.verify_sources(registration, manifest)


@pytest.mark.parametrize("field", ["row_reads_authorized", "parameter_fitting_authorized",
                                  "cell_or_network_execution_authorized", "full_release_fallback_authorized"])
def test_execution_permissions_cannot_expand(tmp_path, field):
    registration, manifest = source_fixture(tmp_path)
    reg = yaml.safe_load(registration.read_text())
    reg[field] = True
    registration.write_text(yaml.safe_dump(reg))
    manifest["registration_sha256"] = module.checksum(registration)
    with pytest.raises(ValueError, match="scope"):
        module.verify_sources(registration, manifest)


def test_premature_observation_never_creates_verification_output(tmp_path, monkeypatch):
    output = tmp_path / "verification"
    monkeypatch.setattr(module, "MANIFEST", tmp_path / "not-terminal.yaml")
    monkeypatch.setattr(module, "OUTPUT", output)
    monkeypatch.setattr(sys, "argv", ["verifier"])
    with pytest.raises(ValueError, match="absent"):
        module.main()
    assert not output.exists()


def test_failed_acquisition_is_retained_with_nonzero_exit(tmp_path, monkeypatch):
    registration, manifest = source_fixture(tmp_path)
    manifest["status"] = "failure-retained"
    manifest["client_exit_code"] = 28
    terminal = tmp_path / "terminal.yaml"
    terminal.write_text(yaml.safe_dump(manifest))
    output = tmp_path / "verification"
    monkeypatch.setattr(module, "REGISTRATION", registration)
    monkeypatch.setattr(module, "MANIFEST", terminal)
    monkeypatch.setattr(module, "OUTPUT", output)
    monkeypatch.setattr(sys, "argv", ["verifier"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    retained = yaml.safe_load((output / "manifest.yaml").read_text())
    assert retained["status"] == "verification-failure-retained"
    assert retained["data_rows_read"] is False
    original = (output / "manifest.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        module.main()
    assert (output / "manifest.yaml").read_bytes() == original
