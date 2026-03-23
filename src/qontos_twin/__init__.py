"""QONTOS Digital Twin — modular hardware simulator for architecture studies."""

from qontos_twin.modular_simulator import (
    ModuleConfig,
    ModularSimulator,
    SimulationResult,
    SystemConfig,
    classify_degradation,
    run_scaling_analysis,
    simulate_workload,
    simulate_workload_calibrated,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ModuleConfig",
    "ModularSimulator",
    "SimulationResult",
    "SystemConfig",
    "classify_degradation",
    "run_scaling_analysis",
    "simulate_workload",
    "simulate_workload_calibrated",
]
