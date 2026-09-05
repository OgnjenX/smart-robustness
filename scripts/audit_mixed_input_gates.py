"""Expose archived input gates that also retain nested projection metadata."""

import argparse
import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path

import yaml

from smart_robustness.models.modeldb112923 import SMART_NML_SHA256


def find_mixed_gates(source):
    root = ET.fromstring(source)
    findings = []
    for population in root.findall(".//population"):
        for compartment in population.findall("./neuron/structure/OrientedSubstructure"):
            for channel in compartment.findall("channel"):
                for gate in channel.findall("gatingVariable"):
                    if gate.get("dependency") != "input" or not gate.findall("projection"):
                        continue
                    findings.append({
                        "population": population.get("name"),
                        "compartment": compartment.get("name"),
                        "channel": dict(channel.attrib), "gate": dict(gate.attrib),
                        "direct_input_methods": [dict(m.attrib) for m in gate.findall("refToSourceMethod")],
                        "nested_projection_sources": [dict(m.attrib) for m in gate.findall("projection/refToPopulation")],
                    })
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source).read_bytes()
    if sha256(source).hexdigest() != SMART_NML_SHA256:
        raise ValueError("source checksum mismatch")
    report = {"schema_version": 1, "source_sha256": SMART_NML_SHA256,
              "mixed_input_gates": find_mixed_gates(source),
              "interpretation": "Extractor currently suppresses external input whenever nested projections exist and classifies those records as chemical. Mixed metadata needs explicit source-semantic resolution, not silent ligand conversion.",
              "baseline_promoted": False}
    with Path(args.output).open("x") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
