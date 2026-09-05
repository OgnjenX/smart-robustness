"""Check relay inhibitory catalog facts and selected Gaussian implementation."""

import argparse
import math
import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.modeldb_projections import load_modeldb_first_order_catalog
from smart_robustness.synapses import modeldb_topology_pairs


def audit(source):
    catalog = load_modeldb_first_order_catalog()
    if sha256(source).hexdigest() != catalog.source_sha256:
        raise ValueError("source archive checksum mismatch")
    root = ET.fromstring(source)
    relay = root.find(".//population[@name='Relay']")
    records = [p for p in catalog.projections if p.target_population == "thalamic_relay"
               and p.source_population in ("trn", "thalamic_interneuron")]
    assert len(records) == 4
    compartments = {"soma": "Soma", "proximal_dendrite": "Dendrite 0",
                    "distal_dendrite": "Dendrite 1"}
    reports = []
    for record in records:
        compartment = relay.find(f".//OrientedSubstructure[@name='{compartments[record.target_compartment]}']")
        channel = compartment.find(f"channel[@name='{record.channel_name}']")
        gate = channel.find("gatingVariable")
        projection = gate.find("projection")
        ref = projection.find("refToPopulation")
        method = projection.find("refToSourceMethod")
        kernel = method.find("Kernel")
        expected_source = "Reticular" if record.source_population == "trn" else "Relay_INT"
        assert ref.get("target") == expected_source
        for value, expected in ((channel.get("g_bar"), record.channel_conductance_mS_cm2),
                                (channel.get("equilibriumPotential"), record.reversal_mV),
                                (gate.get("tau_r"), record.rise_ms),
                                (gate.get("tau_f"), record.fall_ms),
                                (ref.get("axonalDelay"), record.delay_ms),
                                (method.get("weight"), record.weight)):
            assert float(value) == expected
        assert kernel.get("ring") == "false" and record.kernel.ring is False
        assert kernel.get("borderEffect") == record.kernel.border_effect == "wrap"
        sx, sy = float(kernel.get("sigma_x")), float(kernel.get("sigma_y"))
        assert (sx, sy) == (record.kernel.sigma_x, record.kernel.sigma_y)
        assert method.get("target") == record.method == "connectFromMany"
        pre, post, factors = modeldb_topology_pairs(record, source_shape=(9, 9), target_shape=(9, 9))
        expected = {}
        for i in range(81):
            for j in range(81):
                dx = min(abs(i % 9 - j % 9), 9 - abs(i % 9 - j % 9))
                dy = min(abs(i // 9 - j // 9), 9 - abs(i // 9 - j // 9))
                factor = math.exp(-0.5 * ((dx / sx)**2 + (dy / sy)**2))
                if float(method.get("weight")) * factor >= 0.001:
                    expected[i, j] = factor
        actual = dict(zip(zip(pre.tolist(), post.tolist(), strict=True), factors, strict=True))
        assert actual.keys() == expected.keys()
        assert np.allclose(list(actual.values()), list(expected.values()), rtol=1e-14, atol=0)
        reports.append({"id": record.id, "compartment": record.target_compartment,
                        "source": record.source_population, "edges": len(actual),
                        "inputs_per_cell": len(actual) // 81,
                        "colocated_factor": float(actual[40, 40]),
                        "source_peak_weight": record.weight})
    return {"schema_version": 1, "source_sha256": catalog.source_sha256,
            "source_fields_and_selected_topology_consistent": True, "projections": reports,
            "scope": "Unscaled source weights; checks standard-deviation Gaussian with minimum wrapped distance and 0.001 cutoff. Does not recover legacy runtime semantics or validate calibrated weights or behavior.",
            "baseline_promoted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(Path(args.source).read_bytes())
    with Path(args.output).open("x") as stream:
        yaml.safe_dump(result, stream, sort_keys=False)
