"""QONTOS Simulators — local quantum simulation backends.

Public API
----------
LocalSimulatorExecutor : Noiseless Aer-backed circuit executor.
NoisySimulatorExecutor : Noisy Aer-backed circuit executor.
ValidationResult       : Pre-flight validation dataclass.
aer_result_to_partition_result : Normalize raw Aer counts to PartitionResult.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LocalSimulatorExecutor",
    "NoisySimulatorExecutor",
    "ValidationResult",
    "aer_result_to_partition_result",
]


def __getattr__(name: str):
    if name in {"LocalSimulatorExecutor", "ValidationResult"}:
        from qontos_sim.local import LocalSimulatorExecutor, ValidationResult

        exports = {
            "LocalSimulatorExecutor": LocalSimulatorExecutor,
            "ValidationResult": ValidationResult,
        }
        return exports[name]
    if name == "NoisySimulatorExecutor":
        from qontos_sim.noisy import NoisySimulatorExecutor

        return NoisySimulatorExecutor
    if name == "aer_result_to_partition_result":
        from qontos_sim.normalize import aer_result_to_partition_result

        return aer_result_to_partition_result
    raise AttributeError(f"module 'qontos_sim' has no attribute {name}")
