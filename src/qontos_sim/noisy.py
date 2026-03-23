"""Noisy simulator executor — Aer with configurable noise model.

Default noise parameters match the QONTOS whitepaper specs:
  - T1 = 500 us, T2 = 500 us
  - Single-qubit gate error: 0.001
  - Two-qubit gate error:   0.01

Executor stays thin: no partitioning, scheduling, or aggregation logic.
"""

from __future__ import annotations

import time
import logging
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
)

from qontos.models.circuit import CircuitIR
from qontos.models.result import PartitionResult
from qontos_sim.normalize import aer_result_to_partition_result
from qontos_sim.local import _circuit_ir_to_qiskit

logger = logging.getLogger(__name__)

# QONTOS whitepaper default noise parameters
_DEFAULT_NOISE_CONFIG: dict[str, Any] = {
    "t1_us": 500.0,
    "t2_us": 500.0,
    "single_qubit_error": 0.001,
    "two_qubit_error": 0.01,
    "single_qubit_gate_time_us": 0.05,
    "two_qubit_gate_time_us": 0.5,
}


class NoisySimulatorExecutor:
    """Executes a CircuitIR on Aer with a realistic noise model."""

    def __init__(
        self,
        noise_model_config: dict[str, Any] | None = None,
        backend_name: str = "aer_simulator_noisy",
    ) -> None:
        cfg = {**_DEFAULT_NOISE_CONFIG, **(noise_model_config or {})}
        self._noise_model = _build_noise_model(cfg)
        self._backend = AerSimulator(noise_model=self._noise_model)
        self._backend_name = backend_name
        self._config = cfg

    @property
    def provider_name(self) -> str:
        return "simulator_noisy"

    @property
    def is_synchronous(self) -> bool:
        return True

    def submit(
        self,
        circuit_ir: CircuitIR,
        shots: int = 1024,
        **kwargs,
    ) -> PartitionResult:
        """Compatibility wrapper matching the local simulator API."""
        return self.execute(
            circuit_ir,
            shots=shots,
            optimization_level=kwargs.get("optimization_level", 1),
        )

    def execute(
        self,
        circuit_ir: CircuitIR,
        shots: int = 1024,
        optimization_level: int = 1,
    ) -> PartitionResult:
        """Transpile and run a circuit with noise, returning a PartitionResult."""
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
        partition = aer_result_to_partition_result(
            counts=counts,
            shots=shots,
            elapsed_ms=elapsed_ms,
            backend_name=self._backend_name,
            provider="simulator_noisy",
            transpiled_circuit=transpiled,
            circuit_ir=circuit_ir,
        )
        partition.metadata["noise_config"] = self._config
        return partition


# ------------------------------------------------------------------
# Noise model builder
# ------------------------------------------------------------------
def _build_noise_model(cfg: dict[str, Any]) -> NoiseModel:
    """Construct a NoiseModel from the given config dict."""
    noise_model = NoiseModel()

    # --- Depolarizing errors ---
    error_1q = depolarizing_error(cfg["single_qubit_error"], 1)
    error_2q = depolarizing_error(cfg["two_qubit_error"], 2)

    # --- Thermal relaxation errors ---
    t1 = cfg["t1_us"] * 1e-6  # convert to seconds
    t2 = cfg["t2_us"] * 1e-6
    gate_time_1q = cfg["single_qubit_gate_time_us"] * 1e-6
    gate_time_2q = cfg["two_qubit_gate_time_us"] * 1e-6

    thermal_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    thermal_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
        thermal_relaxation_error(t1, t2, gate_time_2q)
    )

    # Compose depolarizing + thermal for each gate class
    combined_1q = error_1q.compose(thermal_1q)
    combined_2q = error_2q.compose(thermal_2q)

    # Apply to standard gate sets
    single_qubit_gates = ["u1", "u2", "u3", "rx", "ry", "rz", "x", "y", "z", "h", "s", "t", "id"]
    two_qubit_gates = ["cx", "cz", "swap", "ecr"]

    for gate in single_qubit_gates:
        noise_model.add_all_qubit_quantum_error(combined_1q, gate)

    for gate in two_qubit_gates:
        noise_model.add_all_qubit_quantum_error(combined_2q, gate)

    return noise_model
