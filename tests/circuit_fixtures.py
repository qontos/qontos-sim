"""Shared CircuitIR fixture builders for qontos-sim tests.

These builders always construct valid CircuitIR objects against the live
SDK schema. If the SDK adds or changes required fields, these builders
must be updated — and the schema compatibility check in CI will catch it.
"""
from __future__ import annotations
from qontos.models.circuit import CircuitIR, GateOperation, InputFormat


def make_qasm_circuit(
    qasm: str,
    num_qubits: int = 2,
    num_clbits: int | None = None,
    depth: int = 10,
    gate_count: int | None = None,
) -> CircuitIR:
    """Build a valid QASM-backed CircuitIR."""
    return CircuitIR(
        qasm_string=qasm,
        num_qubits=num_qubits,
        num_clbits=num_clbits or num_qubits,
        depth=depth,
        gate_count=gate_count or num_qubits * 2,
        gates=[],
        source_type=InputFormat.OPENQASM,
        circuit_hash="fixture",
    )


def make_gate_list_circuit(
    num_qubits: int,
    gates: list[tuple[str, list[int]] | tuple[str, list[int], list[float]]],
    *,
    num_clbits: int | None = None,
    include_measurements: bool = False,
) -> CircuitIR:
    """Build a valid gate-list-backed CircuitIR (no QASM string).

    Args:
        gates: List of (name, qubits) or (name, qubits, params) tuples.
        include_measurements: If True, append measure gates for all qubits.
    """
    gate_ops = []
    for g in gates:
        name, qubits = g[0], g[1]
        params = g[2] if len(g) > 2 else []
        gate_ops.append(GateOperation(name=name, qubits=qubits, params=params))

    if include_measurements:
        for i in range(num_qubits):
            gate_ops.append(GateOperation(name="measure", qubits=[i], params=[]))

    connectivity = []
    for g in gate_ops:
        if len(g.qubits) == 2:
            connectivity.append((g.qubits[0], g.qubits[1]))

    return CircuitIR(
        qasm_string="",
        num_qubits=num_qubits,
        num_clbits=num_clbits or num_qubits,
        depth=len(gate_ops),
        gate_count=len([g for g in gate_ops if g.name != "measure"]),
        gates=gate_ops,
        source_type=InputFormat.QISKIT,
        qubit_connectivity=connectivity,
        circuit_hash=f"fixture_gate_list_{num_qubits}q",
    )


# Standard QASM strings
BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    "qreg q[2];\ncreg c[2];\n"
    "h q[0];\ncx q[0],q[1];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)

GHZ3_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    "qreg q[3];\ncreg c[3];\n"
    "h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
)
