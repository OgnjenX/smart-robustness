import runpy
from pathlib import Path

find_mixed_gates = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_mixed_input_gates.py"))["find_mixed_gates"]


def test_mixed_gate_preserves_both_routes_without_selecting_semantics():
    xml = b'''<neuroml><population name="Relay_INT"><neuron><structure>
    <OrientedSubstructure name="Dendrite 0"><channel name="Input">
    <gatingVariable dependency="input" input2="0.37">
    <projection><refToPopulation target="Layer_4"/></projection>
    <refToSourceMethod target="connectFromOne"/>
    </gatingVariable></channel></OrientedSubstructure>
    </structure></neuron></population></neuroml>'''
    result = find_mixed_gates(xml)
    assert len(result) == 1
    assert result[0]["gate"]["input2"] == "0.37"
    assert result[0]["nested_projection_sources"] == [{"target": "Layer_4"}]
    assert result[0]["direct_input_methods"] == [{"target": "connectFromOne"}]
    assert find_mixed_gates(xml.replace(b'dependency="input"', b'dependency="ligand"')) == []
    assert find_mixed_gates(xml.replace(b'<projection><refToPopulation target="Layer_4"/></projection>', b'')) == []
