#!/usr/bin/env python3
"""QONTOS Digital Twin — Modular Hardware Simulation.

Simulates the QONTOS modular architecture at different scales
and transduction efficiencies. Based on whitepaper specifications.

Run: cd qontos && python -m v1.digital_twin.modular_simulator
"""

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class ModuleConfig:
    """Configuration for a single quantum module."""
    qubits_per_module: int = 50
    gate_fidelity_1q: float = 0.999
    gate_fidelity_2q: float = 0.999
    t1_us: float = 500  # tantalum-silicon target
    t2_us: float = 500
    gate_time_1q_ns: float = 25
    gate_time_2q_ns: float = 50


@dataclass
class SystemConfig:
    """Configuration for a modular quantum system."""
    num_modules: int = 4
    module: ModuleConfig = None
    transduction_efficiency: float = 0.15  # target from whitepaper
    bell_pair_rate_hz: float = 3000
    inter_module_gate_time_us: float = 100

    def __post_init__(self):
        if self.module is None:
            self.module = ModuleConfig()

    @property
    def total_qubits(self) -> int:
        return self.num_modules * self.module.qubits_per_module


@dataclass
class SimulationResult:
    """Result of a digital twin simulation."""
    config: SystemConfig
    total_qubits: int
    total_gates: int
    inter_module_gates: int
    intra_module_gates: int
    estimated_fidelity: float
    estimated_runtime_us: float
    inter_module_latency_us: float
    bell_pairs_needed: int
    effective_circuit_depth: int
    degradation_band: str  # STRETCH, TARGET, FALLBACK, MINIMUM


class ModularSimulator:
    """Lightweight wrapper around the digital twin workload estimators."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def simulate_workload(
        self,
        circuit_depth: int = 100,
        two_qubit_gate_ratio: float = 0.4,
    ) -> SimulationResult:
        return simulate_workload(
            self.config,
            circuit_depth=circuit_depth,
            two_qubit_gate_ratio=two_qubit_gate_ratio,
        )


# Degradation bands from whitepaper (page 9)
DEGRADATION_BANDS = [
    (0.25, "STRETCH",  "~5 kHz", "~500 ops/sec", "Full algorithm library"),
    (0.15, "TARGET",   "~3 kHz", "~300 ops/sec", "Most algorithms viable"),
    (0.10, "FALLBACK", "~2 kHz", "~200 ops/sec", "Sparse communication"),
    (0.05, "MINIMUM",  "~1 kHz", "~100 ops/sec", "Single-module operation"),
]


def classify_degradation(efficiency: float) -> tuple[str, str]:
    """Classify transduction efficiency into degradation band."""
    for threshold, band, rate, ops, capability in DEGRADATION_BANDS:
        if efficiency >= threshold:
            return band, capability
    return "BELOW_MINIMUM", "Non-functional"


def simulate_workload(
    config: SystemConfig,
    circuit_depth: int = 100,
    two_qubit_gate_ratio: float = 0.4,
) -> SimulationResult:
    """Simulate a quantum workload on the modular architecture.

    Estimates fidelity, runtime, and resource requirements.
    """
    total_qubits = config.total_qubits
    gates_per_layer = total_qubits  # approximate
    total_gates = circuit_depth * gates_per_layer
    two_q_gates = int(total_gates * two_qubit_gate_ratio)

    # Estimate inter-module gates (depends on connectivity)
    # For a linear chain of modules, ~1/num_modules of 2Q gates cross boundaries
    if config.num_modules > 1:
        cross_module_ratio = 1.0 / config.num_modules
        inter_module_gates = int(two_q_gates * cross_module_ratio)
    else:
        inter_module_gates = 0

    intra_module_gates = total_gates - inter_module_gates

    # Fidelity calculation
    f_1q = config.module.gate_fidelity_1q
    f_2q = config.module.gate_fidelity_2q
    one_q_gates = total_gates - two_q_gates

    # Intra-module fidelity
    intra_fidelity = (f_1q ** one_q_gates) * (f_2q ** (two_q_gates - inter_module_gates))

    # Inter-module fidelity (depends on transduction)
    inter_module_gate_fidelity = min(0.99, config.transduction_efficiency * 2)
    if inter_module_gates > 0:
        inter_fidelity = inter_module_gate_fidelity ** inter_module_gates
    else:
        inter_fidelity = 1.0

    # Decoherence
    total_time_us = (
        one_q_gates * config.module.gate_time_1q_ns / 1000
        + (two_q_gates - inter_module_gates) * config.module.gate_time_2q_ns / 1000
        + inter_module_gates * config.inter_module_gate_time_us / max(0.01, config.transduction_efficiency)
    )
    decoherence_factor = math.exp(-total_time_us / config.module.t1_us)

    estimated_fidelity = intra_fidelity * inter_fidelity * decoherence_factor

    # Bell pairs needed
    bell_pairs = inter_module_gates  # 1 Bell pair per inter-module gate

    # Inter-module latency
    inter_latency = inter_module_gates * config.inter_module_gate_time_us / max(0.01, config.transduction_efficiency)

    # Effective depth increase from inter-module serialization
    effective_depth = circuit_depth + inter_module_gates

    band, _ = classify_degradation(config.transduction_efficiency)

    return SimulationResult(
        config=config,
        total_qubits=total_qubits,
        total_gates=total_gates,
        inter_module_gates=inter_module_gates,
        intra_module_gates=intra_module_gates,
        estimated_fidelity=max(0, estimated_fidelity),
        estimated_runtime_us=total_time_us,
        inter_module_latency_us=inter_latency,
        bell_pairs_needed=bell_pairs,
        effective_circuit_depth=effective_depth,
        degradation_band=band,
    )


def run_scaling_analysis():
    """Run the full digital twin analysis across scales and efficiencies."""
    print("=" * 70)
    print("  QONTOS Digital Twin — Modular Architecture Simulation")
    print("  Based on whitepaper specifications (tantalum-silicon, photonic)")
    print("=" * 70)

    # Scenario 1: Module scaling at target efficiency (15%)
    print("\n  SCENARIO 1: Module Scaling @ 15% transduction")
    print(f"  {'Modules':<10} {'Qubits':<10} {'Inter-mod':<12} {'Fidelity':<12} {'Runtime(us)':<14} {'Band'}")
    print("  " + "-" * 68)

    for n_modules in [1, 2, 4, 8, 16]:
        config = SystemConfig(num_modules=n_modules, transduction_efficiency=0.15)
        result = simulate_workload(config, circuit_depth=50)
        print(f"  {n_modules:<10} {result.total_qubits:<10} {result.inter_module_gates:<12} "
              f"{result.estimated_fidelity:<12.6f} {result.estimated_runtime_us:<14.1f} {result.degradation_band}")

    # Scenario 2: Transduction efficiency sweep
    print("\n  SCENARIO 2: Transduction Efficiency Sweep (4 modules, 200 qubits)")
    print(f"  {'Efficiency':<12} {'Band':<12} {'Fidelity':<12} {'Inter-latency(us)':<20} {'Bell pairs'}")
    print("  " + "-" * 68)

    for eff in [0.05, 0.10, 0.15, 0.20, 0.25, 0.50]:
        config = SystemConfig(num_modules=4, transduction_efficiency=eff)
        result = simulate_workload(config, circuit_depth=50)
        band, _ = classify_degradation(eff)
        print(f"  {eff:<12.0%} {band:<12} {result.estimated_fidelity:<12.6f} "
              f"{result.inter_module_latency_us:<20.1f} {result.bell_pairs_needed}")

    # Scenario 3: Chemistry workload (target: FeMoco-scale)
    print("\n  SCENARIO 3: Chemistry Workload Scaling")
    print(f"  {'Molecule':<12} {'Qubits':<10} {'Modules':<10} {'Depth':<8} {'Fidelity':<12} {'Viable?'}")
    print("  " + "-" * 62)

    chemistry_targets = [
        ("H2", 4, 1, 20),
        ("LiH", 12, 1, 50),
        ("BeH2", 28, 1, 100),
        ("H2O", 52, 2, 200),
        ("N2", 100, 2, 500),
        ("FeMoco", 2000, 40, 5000),  # The whitepaper's flagship target
    ]

    for name, qubits, min_modules, depth in chemistry_targets:
        modules = max(min_modules, (qubits + 49) // 50)
        config = SystemConfig(num_modules=modules, transduction_efficiency=0.15,
                            module=ModuleConfig(qubits_per_module=max(50, qubits // modules)))
        result = simulate_workload(config, circuit_depth=depth)
        viable = "YES" if result.estimated_fidelity > 0.01 else "NEEDS EC"
        print(f"  {name:<12} {qubits:<10} {modules:<10} {depth:<8} "
              f"{result.estimated_fidelity:<12.6f} {viable}")

    print("\n  Note: FeMoco requires error correction (qLDPC codes)")
    print("  The whitepaper targets 100+ logical qubits by 2030")
    print("=" * 70)


def simulate_workload_calibrated(
    calibrated_module: ModuleConfig,
    num_modules: int = 4,
    transduction_efficiency: float = 0.15,
    circuit_depth: int = 100,
    two_qubit_gate_ratio: float = 0.4,
) -> SimulationResult:
    """Run a simulation with externally-calibrated module parameters.

    This entry point is used by ``ConnectedDigitalTwin`` to feed measured
    gate fidelities and coherence times into the modular simulator.

    Parameters
    ----------
    calibrated_module : ModuleConfig
        Module configuration whose fields have been calibrated against
        real benchmark data (e.g. gate_fidelity_1q/2q, t1_us, t2_us).
    num_modules : int
        Number of modules in the system.
    transduction_efficiency : float
        Microwave-to-optical transduction efficiency.
    circuit_depth : int
        Depth of the quantum circuit.
    two_qubit_gate_ratio : float
        Fraction of gates that are two-qubit gates.

    Returns
    -------
    SimulationResult
        Full simulation result with fidelity, runtime, and resource data.
    """
    config = SystemConfig(
        num_modules=num_modules,
        module=calibrated_module,
        transduction_efficiency=transduction_efficiency,
    )
    return simulate_workload(config, circuit_depth, two_qubit_gate_ratio)


if __name__ == "__main__":
    run_scaling_analysis()
