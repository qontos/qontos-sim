"""QONTOS Architecture Estimator: a modular-systems planning sandbox.

This is an ARCHITECTURE ESTIMATOR, not a measured-data-calibrated digital twin.
It uses scenario bands and analytic proxy formulas to give a stable software-side
signal for planning (Bell-pair budgets, cross-module gate fractions, an aggregate
fidelity indicator), NOT a literal hardware-fidelity prediction. The term "digital
twin" is reserved for a model that is calibrated against and validated on measured
device data (see MODEL_CARD.md for the valid operating domain). The public class
is named ModularSimulator for backward compatibility.
"""

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
