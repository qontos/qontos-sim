"""QONTOS Simulators — local quantum simulation backends."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LocalSimulatorExecutor",
    "NoisySimulatorExecutor",
    "ValidationResult",
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
    raise AttributeError(f"module 'qontos_sim' has no attribute {name}")
