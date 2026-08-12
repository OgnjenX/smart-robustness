"""Extract derived first-order SMART facts from the integrity-pinned ModelDB XML."""

from __future__ import annotations

import argparse
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

EXPECTED_SMART_SHA256 = "16fc196f2ed68d1c02a39395fa83bb0ffab8fcf8edffe6fd11022bb6c6e299e1"

POPULATIONS = {
    "Relay": "thalamic_relay",
    "Reticular": "trn",
    "Layer_5": "layer5_excitatory_v1",
    "Layer_6_II": "layer6ii_excitatory_v1",
    "Layer_6_I": "layer6i_excitatory_v1",
    "Layer_4_INT": "layer4_inhibitory_v1",
    "Layer_2_3": "layer23_excitatory_v1",
    "Layer_4": "layer4_excitatory_v1",
    "Layer_2_3_INT": "layer23_inhibitory_v1",
    "Relay_INT": "thalamic_interneuron",
    "INTRALAMINAR": "thalamic_nonspecific",
    "Thalamic_MATRIX": "thalamic_matrix",
    "Layer_6_II_V2": "layer6ii_excitatory_v2",
    "Reticular_V2": "trn_v2",
    "Relay_V2": "thalamic_relay_v2",
    "Layer_5_V2": "layer5_excitatory_v2",
    "Layer_6_I_V2": "layer6i_excitatory_v2",
    "Layer_2_3_V2": "layer23_excitatory_v2",
    "Layer_4_V2": "layer4_excitatory_v2",
    "Layer_2_3_INT_V2": "layer23_inhibitory_v2",
    "Relay_INT_V2": "thalamic_interneuron_v2",
    "INTRALAMINAR_V2": "thalamic_nonspecific_v2",
    "Thalamic_MATRIX_V2": "thalamic_matrix_v2",
    "Layer_4_INT_V2": "layer4_inhibitory_v2",
}
COMPARTMENTS = {
    "Soma": "soma",
    "Dendrite 0": "proximal_dendrite",
    "Dendrite 1": "distal_dendrite",
}


def _number(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() == "true"


def _source_endpoint(raw: str) -> tuple[str, str | None]:
    for source_name in sorted(POPULATIONS, key=len, reverse=True):
        if raw == source_name:
            return POPULATIONS[source_name], None
        prefix = f"{source_name} "
        if raw.startswith(prefix):
            compartment = raw[len(prefix) :]
            return POPULATIONS[source_name], COMPARTMENTS.get(compartment, compartment)
    return raw, None


def extract(source: Path, *, population_limit: int | None = 12) -> dict[str, object]:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SMART_SHA256:
        raise ValueError(f"SMART.nml hash mismatch: {digest}")
    root = ET.parse(source).getroot()
    populations = root.findall(".//population")
    if population_limit is not None:
        populations = populations[:population_limit]
    records: list[dict[str, object]] = []
    external_channels: list[dict[str, object]] = []
    for population in populations:
        target_population = POPULATIONS[population.get("name", "")]
        for compartment in population.findall("./neuron/structure/OrientedSubstructure"):
            target_compartment = COMPARTMENTS[compartment.get("name", "")]
            for channel in compartment.findall("channel"):
                for gate in channel.findall("gatingVariable"):
                    projections = gate.findall("projection")
                    dependency = gate.get("dependency")
                    if dependency in {"input", "injection"} and not projections:
                        external_channels.append(
                            {
                                "id": f"modeldb112923.external.{len(external_channels):03d}",
                                "target_population": target_population,
                                "target_compartment": target_compartment,
                                "channel_name": channel.get("name"),
                                "dependency": dependency,
                                "channel": dict(channel.attrib),
                                "gate": dict(gate.attrib),
                            }
                        )
                    for projection in projections:
                        source = projection.find("refToPopulation")
                        method = projection.find("refToSourceMethod")
                        if source is None:
                            continue
                        source_population, source_compartment = _source_endpoint(
                            source.get("target", "")
                        )
                        kernel = method.find("Kernel") if method is not None else None
                        records.append(
                            {
                                "id": f"modeldb112923.projection.{len(records):03d}",
                                "kind": "gap_junction"
                                if dependency == "gap junction"
                                else "chemical",
                                "source_population": source_population,
                                "source_compartment": source_compartment,
                                "target_population": target_population,
                                "target_compartment": target_compartment,
                                "channel_name": channel.get("name"),
                                "dependency": dependency,
                                "channel_conductance_mS_cm2": _number(channel.get("g_bar")),
                                "reversal_mV": _number(channel.get("equilibriumPotential")),
                                "rise_ms": _number(gate.get("tau_r")),
                                "fall_ms": _number(gate.get("tau_f")),
                                "delay_ms": _number(source.get("axonalDelay")),
                                "method": None if method is None else method.get("target"),
                                "weight": None if method is None else _number(method.get("weight")),
                                "asymptotic_weight": None
                                if method is None
                                else _number(method.get("assymptoticWeight")),
                                "kernel": None
                                if kernel is None
                                else {
                                    "sigma_x": _number(kernel.get("sigma_x")),
                                    "sigma_y": _number(kernel.get("sigma_y")),
                                    "width": _number(kernel.get("width")),
                                    "height": _number(kernel.get("height")),
                                    "ring": _bool(kernel.get("ring")),
                                    "wrap": _bool(kernel.get("wrap")),
                                    "border_effect": kernel.get("borderEffect"),
                                },
                                "projection_attributes": dict(projection.attrib),
                                "method_attributes": {} if method is None else dict(method.attrib),
                                "gate_attributes": dict(gate.attrib),
                            }
                        )
    return {
        "schema_version": 1,
        "source": (
            "ModelDB 112923 SMART.nml full two-area network"
            if population_limit is None
            else "ModelDB 112923 SMART.nml first-order area"
        ),
        "population_count": len(populations),
        "source_sha256": digest,
        "projection_count": len(records),
        "external_channel_count": len(external_channels),
        "projections": records,
        "external_channels": external_channels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--all-populations",
        action="store_true",
        help="extract the complete V1-pulvinar-V2 network instead of the first 12 populations",
    )
    args = parser.parse_args()
    data = extract(args.source, population_limit=None if args.all_populations else 12)
    args.output.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
