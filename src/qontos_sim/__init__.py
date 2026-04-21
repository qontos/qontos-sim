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


def _missing_dependency_proxy(name: str, dependency: str):
    class _MissingDependencyProxy:
        __name__ = name

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                f"{name} requires the optional dependency '{dependency}'. "
                "Install qontos-sim with the 'sim' extras to enable Aer-backed simulators."
            )

    _MissingDependencyProxy.__name__ = name
    return _MissingDependencyProxy


def __getattr__(name: str):
    if name in {"LocalSimulatorExecutor", "ValidationResult"}:
        try:
            from qontos_sim.local import LocalSimulatorExecutor, ValidationResult
        except ModuleNotFoundError as exc:
            if exc.name != "qiskit":
                raise
            exports = {
                "LocalSimulatorExecutor": _missing_dependency_proxy("LocalSimulatorExecutor", "qiskit"),
                "ValidationResult": _missing_dependency_proxy("ValidationResult", "qiskit"),
            }
            return exports[name]

        exports = {
            "LocalSimulatorExecutor": LocalSimulatorExecutor,
            "ValidationResult": ValidationResult,
        }
        return exports[name]
    if name == "NoisySimulatorExecutor":
        try:
            from qontos_sim.noisy import NoisySimulatorExecutor
        except ModuleNotFoundError as exc:
            if exc.name != "qiskit":
                raise
            return _missing_dependency_proxy("NoisySimulatorExecutor", "qiskit")

        return NoisySimulatorExecutor
    if name == "aer_result_to_partition_result":
        try:
            from qontos_sim.normalize import aer_result_to_partition_result
        except ModuleNotFoundError as exc:
            if exc.name != "qiskit":
                raise

            def _missing_normalizer(*args, **kwargs):
                raise ModuleNotFoundError(
                    "aer_result_to_partition_result requires the optional dependency 'qiskit'. "
                    "Install qontos-sim with the 'sim' extras to enable Aer-backed normalization."
                )

            return _missing_normalizer

        return aer_result_to_partition_result
    raise AttributeError(f"module 'qontos_sim' has no attribute {name}")
