"""Metadata projection uses synthetic SQLite only, never scientific releases."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import synphys_baseline_lineage
    import synphys_baseline_projection

    spec = importlib.util.spec_from_file_location("lineage_fixture", Path(__file__).with_name("test_synphys_baseline_lineage.py"))
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    return synphys_baseline_projection, synphys_baseline_lineage, fixture_module.fixture


def database(path, columns, tables):
    with sqlite3.connect(path) as db:
        for table, names in columns.items():
            declaration = ",".join('"id" INTEGER PRIMARY KEY' if name == "id" else f'"{name}"' for name in names)
            db.execute(f'CREATE TABLE "{table}" ({declaration}, data BLOB)')
            for row in tables[table]:
                db.execute(f'INSERT INTO "{table}" ({",".join(names)}) VALUES ({",".join("?" for _ in names)})',
                           tuple(row[name] for name in names))


def test_exact_bounded_metadata_projection_and_supported_mean(tmp_path, modules):
    reader, lineage, fixture = modules
    context, tables = fixture()
    path = tmp_path / "source.sqlite"
    database(path, lineage.COLUMNS, tables)
    result = reader.project_original_pulse_metadata(path, [1], context)
    assert result["tables"] == tables
    assert result["projected_scalar_payload_bytes"] > 0
    assert not result["waveform_values_read"]
    assert lineage.summarize_original_pulse_baselines([1], context, result["tables"])["metadata_supported_source_mean_volts"] == -0.068


def test_unrelated_rows_not_selected_and_missing_original_pulse_retained(tmp_path, modules):
    reader, lineage, fixture = modules
    context, tables = fixture()
    tables["pulse_response"].append(dict(tables["pulse_response"][0], id=999))
    path = tmp_path / "source.sqlite"
    database(path, lineage.COLUMNS, tables)
    result = reader.project_original_pulse_metadata(path, [2, 1], context)
    assert result["original_pulse_response_ids"] == [2, 1]
    assert [row["id"] for row in result["tables"]["pulse_response"]] == [1]
    assert not lineage.join_original_pulse_metadata([2, 1], context, result["tables"])["all_contributor_identities_complete"]


def test_multiple_patch_parents_are_retained(tmp_path, modules):
    reader, lineage, fixture = modules
    context, tables = fixture()
    tables["patch_clamp_recording"].append(dict(tables["patch_clamp_recording"][0], id=71))
    path = tmp_path / "source.sqlite"
    database(path, lineage.COLUMNS, tables)
    result = reader.project_original_pulse_metadata(path, [1], context)
    assert len(result["tables"]["patch_clamp_recording"]) == 2
    assert not lineage.join_original_pulse_metadata([1], context, result["tables"])["all_contributor_identities_complete"]


@pytest.mark.parametrize("table", ["cell", "recording", "pulse_response", "stim_pulse", "patch_clamp_recording", "sync_rec", "electrode"])
def test_each_missing_parent_stays_missing(tmp_path, modules, table):
    reader, lineage, fixture = modules
    context, tables = fixture()
    tables[table] = []
    path = tmp_path / "source.sqlite"
    database(path, lineage.COLUMNS, tables)
    result = reader.project_original_pulse_metadata(path, [1], context)
    assert not lineage.join_original_pulse_metadata([1], context, result["tables"])["all_contributor_identities_complete"]


@pytest.mark.parametrize("query", ["SELECT data FROM pulse_response", "SELECT * FROM sqlite_master", "DELETE FROM cell", "PRAGMA table_info(cell)", "SELECT count(*) FROM cell"])
def test_authorizer_denies_waveforms_schema_writes_and_unregistered_functions(tmp_path, modules, query):
    reader, lineage, fixture = modules
    _, tables = fixture()
    path = tmp_path / "source.sqlite"
    database(path, lineage.COLUMNS, tables)
    with sqlite3.connect(path) as db:
        db.set_authorizer(reader.metadata_authorizer)
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(query).fetchall()


@pytest.mark.parametrize("field,value", [("baseline_potential", b"x" * 100000),
                                       ("clamp_mode", b"ic"), ("clamp_mode", "x" * 4097),
                                       ("clamp_mode", "é" * 3000)])
def test_scalar_types_and_text_bounds_fail_without_promotion(tmp_path, modules, field, value):
    reader, lineage, fixture = modules
    context, tables = fixture()
    tables["patch_clamp_recording"][0][field] = value
    path = tmp_path / "source.sqlite"
    database(path, lineage.COLUMNS, tables)
    with pytest.raises(ValueError):
        reader.project_original_pulse_metadata(path, [1], context)


@pytest.mark.parametrize("bound,value", [("MAX_IDS", 0), ("MAX_ROWS_PER_TABLE", 0),
                                       ("MAX_ROWS_PER_PARENT", 0), ("MAX_PAYLOAD_BYTES", 1)])
def test_fixed_bounds_are_not_relaxed_on_failure(tmp_path, modules, monkeypatch, bound, value):
    reader, lineage, fixture = modules
    context, tables = fixture()
    path = tmp_path / "source.sqlite"
    database(path, lineage.COLUMNS, tables)
    monkeypatch.setattr(reader, bound, value)
    with pytest.raises(ValueError):
        reader.project_original_pulse_metadata(path, [1], context)


def test_readonly_source_not_created_if_missing(tmp_path, modules):
    reader, _, fixture = modules
    context, _ = fixture()
    path = tmp_path / "absent.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        reader.project_original_pulse_metadata(path, [1], context)
    assert not path.exists()
