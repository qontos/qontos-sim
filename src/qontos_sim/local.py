"""Local simulator executor — thin wrapper around Qiskit Aer.

Executor stays thin: no partitioning, scheduling, or aggregation logic.
(Non-negotiable rule #2.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import logging
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qontos.models.circuit import CircuitIR
from qontos.models.result import PartitionResult
from qontos_sim.normalize import aer_result_to_partition_result

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Minimal validation response returned by the simulator executors."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class LocalSimulatorExecutor:
    """Executes a CircuitIR on the Qiskit Aer statevector/qasm simulator."""

    def __init__(self, backend_name: str = "aer_simulator") -> None:
        self._backend = AerSimulator()
        self._backend_name = backend_name

    @property
    def provider_name(self) -> str:
        return "local_simulator"

    @property
    def is_synchronous(self) -> bool:
        return True

    def validate(self, circuit_ir: CircuitIR, shots: int = 1024) -> ValidationResult:
        """Pre-flight validation for the local simulator."""
        errors: list[str] = []
        warnings: list[str] = []

        if circuit_ir.num_qubits < 1:
            errors.append("Circuit must have at least 1 qubit.")
        if circuit_ir.num_qubits > 32:
            warnings.append(
                f"Simulating {circuit_ir.num_qubits} qubits may be slow or exhaust memory."
            )
        if shots < 1:
            errors.append("shots must be >= 1.")
        if not circuit_ir.qasm_string and not circuit_ir.gates:
            errors.append("Circuit has no qasm_string and no gates; nothing to execute.")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def submit(
        self,
        circuit_ir: CircuitIR,
        shots: int = 1024,
        **kwargs,
    ) -> PartitionResult:
        """Transpile and run a circuit, returning a PartitionResult."""
        optimization_level = kwargs.get("optimization_level", 1)
        qc = _circuit_ir_to_qiskit(circuit_ir)

        transpiled = transpile(
            qc,
            backend=self._backend,
            optimization_level=optimization_level,
        )

        t0 = time.perf_counter()
        job = self._backend.run(transpiled, shots=shots)
        result = job.result()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        counts = result.get_counts(transpiled)
        return aer_result_to_partition_result(
            counts=counts,
            shots=shots,
            elapsed_ms=elapsed_ms,
            backend_name=self._backend_name,
            provider="simulator",
            transpiled_circuit=transpiled,
            circuit_ir=circuit_ir,
        )

    # Backwards-compat alias
    def execute(
        self,
        circuit_ir: CircuitIR,
        shots: int = 1024,
        optimization_level: int = 1,
    ) -> PartitionResult:
        """Alias for submit() — kept for backwards compatibility."""
        return self.submit(circuit_ir, shots=shots, optimization_level=optimization_level)

    def poll(self, provider_job_id: str) -> dict:
        """Local simulator is synchronous — poll is a no-op."""
        return {"status": "completed", "result": None}

    def cancel(self, provider_job_id: str) -> bool:
        """Local simulator is synchronous — cancel is a no-op."""
        return False

    def normalize(self, raw_result: Any) -> PartitionResult:
        """Delegate to the normalize module."""
        if isinstance(raw_result, PartitionResult):
            return raw_result
        # Attempt to pass through as kwargs if it's a dict
        if isinstance(raw_result, dict):
            return PartitionResult(**raw_result)
        raise TypeError(f"Cannot normalize result of type {type(raw_result)}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _circuit_ir_to_qiskit(circuit_ir: CircuitIR) -> QuantumCircuit:
    """Reconstruct a Qiskit QuantumCircuit from CircuitIR."""
    if circuit_ir.qasm_string:
        return QuantumCircuit.from_qasm_str(circuit_ir.qasm_string)

    qc = QuantumCircuit(circuit_ir.num_qubits, circuit_ir.num_clbits or circuit_ir.num_qubits)
    for gate in circuit_ir.gates:
        getattr(qc, gate.name)(*gate.params, *[qc.qubits[q] for q in gate.qubits])
    qc.measure_all(add_bits=False) if circuit_ir.num_clbits else qc.measure_all()
    return qc
