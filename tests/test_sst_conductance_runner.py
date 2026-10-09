"""Protocol, corruption and authorization controls without native execution."""

from __future__ import annotations

import fcntl
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture
def modules(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    loaded = []
    for name in ("run_sst_conductance_translation", "verify_sst_conductance_translation"):
        spec = importlib.util.spec_from_file_location(name, scripts / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def parameters(dt=0.00005):
    return {"dt": dt, "R_input": 1e8, "th_inf": 0.02, "El": 0, "coeffs": {"G": 2, "th_inf": 1.5}}


def arm():
    return {
        "name": "fixture",
        "bias_scale_multiple": 2,
        "excitation_leak_multiple": 0.5,
        "inhibition_leak_multiple": 1,
    }


@pytest.mark.parametrize(
    "dt, n, begin, end", [(0.00005, 2000, 400, 1600), (0.000005, 20000, 4000, 16000)]
)
def test_registered_input_recipe(modules, dt, n, begin, end):
    runner, _ = modules
    bias, ge, gi = runner.inputs(parameters(dt), arm())
    assert len(bias) == n and np.all(bias == 1.2e-9)
    assert np.all(ge[begin:end] == 1e-8) and np.all(gi[begin:end] == 2e-8)
    assert not np.any(ge[:begin]) and not np.any(ge[end:])


def test_unregistered_dt_rejected(modules):
    with pytest.raises(ValueError, match="timestep"):
        modules[0].inputs(parameters(0.001), arm())


def test_missing_seal_stops_before_native_execution(modules, tmp_path):
    with pytest.raises(FileNotFoundError):
        modules[0].prepare(seal_path=tmp_path / "missing.yaml")


def test_false_authorization_rejected(modules, tmp_path):
    runner, _ = modules
    seal = tmp_path / "seal.yaml"
    seal.write_text(
        yaml.safe_dump(
            {
                "registration_sha256": runner.digest(runner.REGISTRATION),
                "synthetic_execution_authorized": False,
            }
        )
    )
    with pytest.raises(ValueError, match="not authorized"):
        runner.prepare(seal_path=seal)


def test_changed_source_rejected_before_loading_native(modules, tmp_path):
    runner, _ = modules
    seal = tmp_path / "seal.yaml"
    source = tmp_path / "source.py"
    source.write_text("changed")
    seal.write_text(
        yaml.safe_dump(
            {
                "registration_sha256": runner.digest(runner.REGISTRATION),
                "synthetic_execution_authorized": True,
                "network_execution_authorized": False,
                "parameter_fitting_authorized": False,
                "implementation_sha256": {str(source): "wrong"},
            }
        )
    )
    with pytest.raises(ValueError, match="implementation changed"):
        runner.prepare(seal_path=seal)


def test_full_fixed_cohort_preparation_without_execution(modules, monkeypatch, tmp_path):
    runner, _ = modules
    registration = yaml.safe_load(runner.REGISTRATION.read_text())
    records = []
    for model in registration["cohort"]["model_ids"]:
        raw = tmp_path / f"{model}.json"
        raw.write_text(json.dumps(parameters()))
        records.append(
            {"source": {"model_id": model}, "raw_path": str(raw), "raw_sha256": runner.digest(raw)}
        )
    manifest = tmp_path / "parameters.yaml"
    runner.save(manifest, {"records": records})
    inventory = tmp_path / "inventory.yaml"
    runner.save(inventory, {})
    parent = tmp_path / "parent.yaml"
    runner.save(
        parent,
        {
            "cell_inventory": str(inventory),
            "cell_inventory_sha256": runner.digest(inventory),
            "parameter_manifest": str(manifest),
            "parameter_manifest_sha256": runner.digest(manifest),
        },
    )
    registration.update(parent=str(parent), parent_sha256=runner.digest(parent))
    reg_path = tmp_path / "registration.yaml"
    runner.save(reg_path, registration)
    native = {}
    for key in ("native_neuron", "native_methods"):
        path = tmp_path / (key + ".txt")
        path.write_text("NOT EXECUTABLE: fixture source only")
        native[key + "_source"] = str(path)
        native[key + "_sha256"] = runner.digest(path)
    native_reg = tmp_path / "native.yaml"
    runner.save(native_reg, native)
    monkeypatch.setattr(runner, "NATIVE_REGISTRATION", native_reg)
    seal_path = tmp_path / "seal.yaml"
    runner.save(
        seal_path,
        {
            "registration_sha256": runner.digest(reg_path),
            "synthetic_execution_authorized": True,
            "network_execution_authorized": False,
            "parameter_fitting_authorized": False,
            "implementation_sha256": {},
            "native_registration_sha256": runner.digest(native_reg),
        },
    )
    cases, _, context = runner.prepare(reg_path, seal_path)
    assert len(cases) == 90
    assert len({(c["model_id"], c["dt"], c["arm"]["name"]) for c in cases}) == 90
    assert context["registration_sha256"] == runner.digest(reg_path)
    # The test sources cannot be executed: prepare must only read/hash them.
    Path(records[0]["raw_path"]).write_text("changed parameters")
    with pytest.raises(ValueError, match="parameter bytes changed"):
        runner.prepare(reg_path, seal_path)


def test_checkpoint_resume_and_corruption(modules, monkeypatch, tmp_path):
    runner, _ = modules
    case = {"parameters": parameters(), "arm": arm()}
    context = {"fixture": True}
    calls = []

    def evaluate(*args):
        calls.append(1)
        bias, ge, gi = runner.inputs(case["parameters"], case["arm"])
        return {"passed": False}, {"bias": bias, "excitation": ge, "inhibition": gi}

    monkeypatch.setattr(runner, "evaluate", evaluate)
    saved = runner.checkpoint(0, case, context, None, tmp_path)
    assert runner.checkpoint(0, case, context, None, tmp_path) == saved
    assert len(calls) == 1
    with pytest.raises(ValueError, match="identity"):
        runner.checkpoint(0, case, {"fixture": False}, None, tmp_path)
    (tmp_path / "case-000.npz").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="arrays changed"):
        runner.checkpoint(0, case, context, None, tmp_path)
    assert len(calls) == 1


def test_orphan_arrays_never_overwritten(modules, tmp_path):
    runner, _ = modules
    raw = tmp_path / "case-000.npz"
    raw.write_bytes(b"preserved")
    with pytest.raises(ValueError, match="orphan"):
        runner.checkpoint(0, {"parameters": parameters(), "arm": arm()}, {}, None, tmp_path)
    assert raw.read_bytes() == b"preserved"


def test_duplicate_lock_rejects_runner(modules, monkeypatch, tmp_path):
    runner, _ = modules
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "prepare", lambda: ([], {}, {}))
    with (tmp_path / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runner.main()


def test_terminal_missing_sidecar_cannot_restart(modules, monkeypatch, tmp_path):
    runner, _ = modules
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "prepare", lambda: ([{}] * 90, {}, {}))
    runner.save(tmp_path / "manifest.yaml", {"context": {}, "records": [{}] * 90})
    with pytest.raises(ValueError, match="sidecar missing"):
        runner.main()


def test_complete_mock_matrix_resumes_without_loading_or_reexecuting(
    modules, monkeypatch, tmp_path
):
    runner, _ = modules
    case = {"parameters": parameters(), "arm": arm()}
    cases = [dict(case, fixture_case=n) for n in range(90)]
    calls = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "prepare", lambda: (cases, {}, {"fixture": True}))
    monkeypatch.setattr(runner, "load_native", lambda _: object())

    def evaluate(*args):
        calls.append(1)
        bias, ge, gi = runner.inputs(parameters(), arm())
        return {"passed": False}, {"bias": bias, "excitation": ge, "inhibition": gi}

    monkeypatch.setattr(runner, "evaluate", evaluate)
    runner.main()
    assert len(calls) == 90
    terminal = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert terminal["cases"] == 90 and terminal["attempts"] == 360
    assert len(terminal["records"]) == 90 and terminal["passed_cases"] == 0

    def forbidden(*args):
        raise AssertionError("terminal mock matrix must not execute native or candidate code")

    monkeypatch.setattr(runner, "load_native", forbidden)
    monkeypatch.setattr(runner, "evaluate", forbidden)
    runner.main()
    assert len(calls) == 90


def saved_arrays(verifier, tmp_path, *, null=False):
    report = {
        "errors": {},
        "states": {
            k: {"bad_reset_stop": False} for k in ("native0", "native1", "candidate0", "candidate1")
        },
        "passed": True,
        "native_exact_repeat": True,
        "candidate_exact_repeat": True,
        "bad_reset_stop": False,
        "null_equivalence": True,
    }
    arrays = {
        kind + "_" + field: np.array([0.0, 1.0])
        for kind in ("native0", "native1", "candidate0", "candidate1")
        for field in verifier.FIELDS
    }
    report["comparison"], _ = verifier.reconstruct(arrays)
    if null:
        arrays.update(
            {"current_only_" + f: arrays["candidate0_" + f].copy() for f in verifier.FIELDS}
        )
    arrays.update(
        bias=np.zeros(2),
        excitation=np.zeros(2),
        inhibition=np.zeros(2),
        report_json=np.asarray(json.dumps(report)),
    )
    path = tmp_path / "fixture.npz"
    np.savez(path, **arrays)
    return path, report


@pytest.mark.parametrize("null", [False, True])
def test_independent_report_reconstruction(modules, tmp_path, null):
    _, verifier = modules
    path, report = saved_arrays(verifier, tmp_path, null=null)
    with np.load(path, allow_pickle=False) as arrays:
        assert verifier.verify_report(arrays, report, null=null)
        report["candidate_exact_repeat"] = False
        with pytest.raises(ValueError, match="repeat differs"):
            verifier.verify_report(arrays, report, null=null)


def test_independent_rejects_missing_output_field(modules, tmp_path):
    _, verifier = modules
    path, report = saved_arrays(verifier, tmp_path)
    with np.load(path, allow_pickle=False) as original:
        data = {k: original[k] for k in original.files if k != "native0_voltage"}
    np.savez(path, **data)
    with (
        np.load(path, allow_pickle=False) as arrays,
        pytest.raises(ValueError, match="field coverage"),
    ):
        verifier.verify_report(arrays, report, null=False)


def test_independent_rejects_promoted_error(modules, tmp_path):
    _, verifier = modules
    report = {
        "errors": {
            k: {"message": "fixture"} for k in ("native0", "native1", "candidate0", "candidate1")
        },
        "states": {},
        "passed": True,
    }
    path = tmp_path / "errors.npz"
    np.savez(path, bias=[], excitation=[], inhibition=[], report_json="fixture")
    with (
        np.load(path, allow_pickle=False) as arrays,
        pytest.raises(ValueError, match="error case promoted"),
    ):
        verifier.verify_report(arrays, report, null=False)


@pytest.mark.parametrize("failed", ["native", "candidate", "current_only", None])
def test_evaluate_retains_failures_and_all_attempts(modules, monkeypatch, tmp_path, failed):
    runner, verifier = modules
    called = []
    out = {field: np.array([0.0]) for field in verifier.FIELDS}

    class Reference:
        @classmethod
        def from_dict(cls, parameters):
            return cls()

    def response(kind):
        def run(*args, **kwargs):
            called.append(kind)
            if kind == failed:
                raise RuntimeError("retained fixture failure")
            return {k: v.copy() for k, v in out.items()}, {"bad_reset_stop": False}

        return run

    monkeypatch.setattr(runner, "run_extended_native", response("native"))
    monkeypatch.setattr(runner, "simulate_conductance_glif", response("candidate"))
    monkeypatch.setattr(runner, "simulate_glif", response("current_only"))
    case_arm = dict(
        arm(), name="zero-conductance-null", excitation_leak_multiple=0, inhibition_leak_multiple=0
    )
    report, arrays = runner.evaluate({"parameters": parameters(), "arm": case_arm}, Reference)
    assert called[:4] == ["native", "native", "candidate", "candidate"]
    assert report["passed"] == (failed is None)
    path = tmp_path / "attempts.npz"
    np.savez(path, report_json=np.asarray(json.dumps(report)), **arrays)
    with np.load(path, allow_pickle=False) as data:
        assert verifier.verify_report(data, report, null=True) == (failed is None)
    if failed:
        assert report["errors"] and all(
            e["message"] == "retained fixture failure" for e in report["errors"].values()
        )
