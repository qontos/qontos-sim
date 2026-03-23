"""Helpers for converting simulator outputs into QONTOS result models."""

from __future__ import annotations

import uuid

from qiskit import QuantumCircuit

from qontos.models.circuit import CircuitIR
from qontos.models.result import PartitionResult


def aer_result_to_partition_result(
    *,
    counts: dict[str, int],
    shots: int,
    elapsed_ms: float,
    backend_name: str,
    provider: str,
    transpiled_circuit: QuantumCircuit | None = None,
    circuit_ir: CircuitIR | None = None,
    partition_id: str | None = None,
    partition_index: int = 0,
) -> PartitionResult:
    """Convert raw Aer output into the canonical QONTOS partition result."""
    normalized_counts = {bitstring.replace(" ", ""): count for bitstring, count in counts.items()}

    transpiled_depth: int | None = None
    transpiled_gate_count: int | None = None
    if transpiled_circuit is not None:
        transpiled_depth = transpiled_circuit.depth()
        transpiled_gate_count = sum(transpiled_circuit.count_ops().values())

    return PartitionResult(
        partition_id=partition_id or uuid.uuid4().hex[:16],
        partition_index=partition_index,
        backend_id=backend_name,
        backend_name=backend_name,
        provider=provider,
        counts=normalized_counts,
        shots_completed=shots,
        execution_time_ms=elapsed_ms,
        transpiled_depth=transpiled_depth,
        transpiled_gate_count=transpiled_gate_count,
        metadata={
            "circuit_hash": circuit_ir.circuit_hash if circuit_ir else "",
        },
    )
