"""QGH-3011: Comprehensive tests for the Digital Twin modular simulator.

Tests cover SystemConfig creation, simulate_workload output fields,
single-module workloads, multi-module overhead, transduction efficiency
bands, fidelity degradation, Bell pair scaling, effective depth increase,
and scenario comparisons.
"""

from __future__ import annotations

import pytest

from qontos_twin.modular_simulator import (
    ModuleConfig,
    ModularSimulator,
    SimulationResult,
    SystemConfig,
    classify_degradation,
    simulate_workload,
)


# ===================================================================
# SystemConfig creation
# ===================================================================

class TestSystemConfig:
    """Test SystemConfig dataclass defaults and derived properties."""

    def test_default_config(self):
        cfg = SystemConfig()
        assert cfg.num_modules == 4
        assert cfg.module is not None
        assert cfg.module.qubits_per_module == 50
        assert cfg.transduction_efficiency == 0.15

    def test_custom_module_count(self):
        cfg = SystemConfig(num_modules=8)
        assert cfg.num_modules == 8
        assert cfg.total_qubits == 8 * 50

    def test_custom_transduction(self):
        cfg = SystemConfig(transduction_efficiency=0.25)
        assert cfg.transduction_efficiency == 0.25

    def test_total_qubits(self):
        cfg = SystemConfig(
            num_modules=3,
            module=ModuleConfig(qubits_per_module=100),
        )
        assert cfg.total_qubits == 300


# ===================================================================
# simulate_workload return fields
# ===================================================================

class TestSimulateWorkloadFields:
    """Verify that simulate_workload returns all expected fields."""

    def test_result_has_expected_fields(self):
        cfg = SystemConfig(num_modules=4)
        result = simulate_workload(cfg, circuit_depth=50)

        assert isinstance(result, SimulationResult)
        assert isinstance(result.total_gates, int)
        assert isinstance(result.estimated_fidelity, float)
        assert isinstance(result.estimated_runtime_us, float)
        assert isinstance(result.bell_pairs_needed, int)
        assert isinstance(result.inter_module_gates, int)
        assert isinstance(result.intra_module_gates, int)
        assert isinstance(result.effective_circuit_depth, int)
        assert isinstance(result.degradation_band, str)

    def test_gate_count_sums(self):
        cfg = SystemConfig(num_modules=4)
        result = simulate_workload(cfg, circuit_depth=50)
        assert result.total_gates == result.inter_module_gates + result.intra_module_gates


# ===================================================================
# Single-module workload (no inter-module overhead)
# ===================================================================

class TestSingleModule:
    """Single-module system should have zero inter-module overhead."""

    def test_no_inter_module_gates(self):
        cfg = SystemConfig(num_modules=1)
        result = simulate_workload(cfg, circuit_depth=100)
        assert result.inter_module_gates == 0
        assert result.bell_pairs_needed == 0
        assert result.inter_module_latency_us == 0.0

    def test_effective_depth_equals_circuit_depth(self):
        cfg = SystemConfig(num_modules=1)
        result = simulate_workload(cfg, circuit_depth=100)
        assert result.effective_circuit_depth == 100


# ===================================================================
# Multi-module workload (inter-module gates increase runtime)
# ===================================================================

class TestMultiModule:
    """Multi-module systems should show inter-module overhead."""

    def test_inter_module_gates_present(self):
        cfg = SystemConfig(num_modules=4)
        result = simulate_workload(cfg, circuit_depth=100)
        assert result.inter_module_gates > 0

    def test_runtime_increases_with_modules(self):
        r1 = simulate_workload(SystemConfig(num_modules=1), circuit_depth=50)
        r4 = simulate_workload(SystemConfig(num_modules=4), circuit_depth=50)
        assert r4.estimated_runtime_us > r1.estimated_runtime_us

    def test_effective_depth_increases(self):
        r1 = simulate_workload(SystemConfig(num_modules=1), circuit_depth=50)
        r4 = simulate_workload(SystemConfig(num_modules=4), circuit_depth=50)
        assert r4.effective_circuit_depth > r1.effective_circuit_depth


# ===================================================================
# Transduction efficiency bands
# ===================================================================

class TestTransductionBands:
    """Test the four degradation bands from the whitepaper."""

    def test_stretch_band(self):
        band, _ = classify_degradation(0.25)
        assert band == "STRETCH"

    def test_stretch_threshold(self):
        band, _ = classify_degradation(0.20)
        # 0.20 >= 0.15 => TARGET
        assert band == "TARGET"

    def test_target_band(self):
        band, _ = classify_degradation(0.15)
        assert band == "TARGET"

    def test_aggressive_band(self):
        """>=0.10 maps to FALLBACK."""
        band, _ = classify_degradation(0.10)
        assert band == "FALLBACK"

    def test_base_band(self):
        """>=0.05 maps to MINIMUM."""
        band, _ = classify_degradation(0.05)
        assert band == "MINIMUM"

    def test_below_minimum(self):
        band, _ = classify_degradation(0.01)
        assert band == "BELOW_MINIMUM"

    def test_simulation_result_band(self):
        cfg = SystemConfig(transduction_efficiency=0.25)
        result = simulate_workload(cfg, circuit_depth=50)
        assert result.degradation_band == "STRETCH"


# ===================================================================
# Fidelity degrades with more inter-module gates
# ===================================================================

class TestFidelityDegradation:
    """More inter-module gates should lower fidelity."""

    def test_fidelity_lower_with_more_modules(self):
        r1 = simulate_workload(SystemConfig(num_modules=1), circuit_depth=50)
        r4 = simulate_workload(SystemConfig(num_modules=4), circuit_depth=50)
        r8 = simulate_workload(SystemConfig(num_modules=8), circuit_depth=50)

        assert r1.estimated_fidelity > r4.estimated_fidelity
        assert r4.estimated_fidelity > r8.estimated_fidelity

    def test_fidelity_in_valid_range(self):
        cfg = SystemConfig(num_modules=4)
        result = simulate_workload(cfg, circuit_depth=50)
        assert 0.0 <= result.estimated_fidelity <= 1.0


# ===================================================================
# Bell pair count scales with cross-module entanglement
# ===================================================================

class TestBellPairScaling:
    """Bell pairs needed should scale with inter-module gate count."""

    def test_bell_pairs_equal_inter_module_gates(self):
        cfg = SystemConfig(num_modules=4)
        result = simulate_workload(cfg, circuit_depth=100)
        assert result.bell_pairs_needed == result.inter_module_gates

    def test_more_modules_more_bell_pairs(self):
        r2 = simulate_workload(SystemConfig(num_modules=2), circuit_depth=100)
        r8 = simulate_workload(SystemConfig(num_modules=8), circuit_depth=100)
        assert r8.bell_pairs_needed > r2.bell_pairs_needed


# ===================================================================
# Effective circuit depth increase from serialization
# ===================================================================

class TestEffectiveDepth:
    """Effective depth should increase by the inter-module gate count."""

    def test_effective_depth_formula(self):
        cfg = SystemConfig(num_modules=4)
        result = simulate_workload(cfg, circuit_depth=100)
        assert result.effective_circuit_depth == 100 + result.inter_module_gates


# ===================================================================
# Scenario comparison: base vs aggressive vs stretch
# ===================================================================

class TestScenarioComparison:
    """Compare results across transduction efficiency scenarios."""

    def test_stretch_better_than_target(self):
        stretch = simulate_workload(
            SystemConfig(num_modules=4, transduction_efficiency=0.25),
            circuit_depth=100,
        )
        target = simulate_workload(
            SystemConfig(num_modules=4, transduction_efficiency=0.15),
            circuit_depth=100,
        )
        assert stretch.estimated_fidelity >= target.estimated_fidelity

    def test_target_better_than_fallback(self):
        target = simulate_workload(
            SystemConfig(num_modules=4, transduction_efficiency=0.15),
            circuit_depth=100,
        )
        fallback = simulate_workload(
            SystemConfig(num_modules=4, transduction_efficiency=0.10),
            circuit_depth=100,
        )
        assert target.estimated_fidelity >= fallback.estimated_fidelity

    def test_modular_simulator_wrapper(self):
        """ModularSimulator class should delegate to simulate_workload."""
        cfg = SystemConfig(num_modules=4)
        sim = ModularSimulator(cfg)
        result = sim.simulate_workload(circuit_depth=50)
        assert isinstance(result, SimulationResult)
        assert result.total_qubits == 200
