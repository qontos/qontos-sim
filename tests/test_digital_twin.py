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

    def test_new_hybrid_knobs_exist(self):
        cfg = SystemConfig(
            transduction_loss=0.1,
            transduction_calibration_quality=0.85,
            optical_coupling_efficiency=0.8,
            heralding_success_probability=0.75,
            detector_efficiency=0.9,
            phase_lock_duty_cycle=0.88,
            link_phase_stability=0.8,
            added_noise=0.05,
            bell_pair_retry_rate=1.5,
            entanglement_parallel_links=3,
            entanglement_buffer_pairs=128,
            memory_wait_time_us=12.0,
            control_jitter_us=3.0,
        )
        assert cfg.transduction_loss == 0.1
        assert cfg.transduction_calibration_quality == 0.85
        assert cfg.optical_coupling_efficiency == 0.8
        assert cfg.heralding_success_probability == 0.75
        assert cfg.detector_efficiency == 0.9
        assert cfg.phase_lock_duty_cycle == 0.88
        assert cfg.link_phase_stability == 0.8
        assert cfg.added_noise == 0.05
        assert cfg.bell_pair_retry_rate == 1.5
        assert cfg.entanglement_parallel_links == 3
        assert cfg.entanglement_buffer_pairs == 128
        assert cfg.memory_wait_time_us == 12.0
        assert cfg.control_jitter_us == 3.0

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
        assert isinstance(result.effective_transduction_efficiency, float)
        assert isinstance(result.transduction_channel_efficiency, float)
        assert isinstance(result.channel_margin_to_target, float)
        assert isinstance(result.link_quality, float)
        assert isinstance(result.dynamic_link_stability, float)
        assert isinstance(result.link_margin_to_target, float)
        assert isinstance(result.retry_adjusted_link_fidelity, float)
        assert isinstance(result.transduction_calibration_quality, float)
        assert isinstance(result.optical_coupling_efficiency, float)
        assert isinstance(result.heralding_success_probability, float)
        assert isinstance(result.detector_efficiency, float)
        assert isinstance(result.phase_lock_duty_cycle, float)
        assert isinstance(result.link_phase_stability, float)
        assert isinstance(result.channel_component_values, dict)
        assert isinstance(result.channel_component_margins, dict)
        assert isinstance(result.weakest_channel_component, str)
        assert isinstance(result.weakest_channel_value, float)
        assert isinstance(result.weakest_channel_margin, float)
        assert isinstance(result.expected_attempts_per_bell_pair, float)
        assert isinstance(result.entanglement_parallel_links, int)
        assert isinstance(result.entanglement_buffer_pairs, int)
        assert isinstance(result.buffered_bell_pairs_used, int)
        assert isinstance(result.entanglement_supply_time_us, float)
        assert isinstance(result.entanglement_supply_utilization, float)
        assert isinstance(result.effective_bell_pair_rate_hz, float)
        assert isinstance(result.throughput_ops_per_sec, float)
        assert isinstance(result.retry_overhead_us, float)
        assert isinstance(result.memory_wait_overhead_us, float)
        assert isinstance(result.control_jitter_overhead_us, float)
        assert isinstance(result.added_noise_penalty, float)
        assert isinstance(result.runtime_breakdown_us, dict)
        assert isinstance(result.bottleneck_scores, dict)
        assert isinstance(result.dominant_bottleneck, str)

    def test_runtime_breakdown_matches_total_runtime(self):
        cfg = SystemConfig(num_modules=4)
        result = simulate_workload(cfg, circuit_depth=50)
        assert abs(sum(result.runtime_breakdown_us.values()) - result.estimated_runtime_us) < 1e-6

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
        assert result.degradation_band == "TARGET"


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


class TestHybridKnobSensitivity:
    """Hybrid-specific knobs should visibly affect latency and fidelity proxies."""

    def test_transduction_loss_reduces_effective_efficiency(self):
        baseline = simulate_workload(SystemConfig(num_modules=4, transduction_efficiency=0.15))
        lossy = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.15,
                transduction_loss=0.2,
            )
        )
        assert lossy.effective_transduction_efficiency < baseline.effective_transduction_efficiency
        assert lossy.link_quality < baseline.link_quality
        assert lossy.inter_module_latency_us > baseline.inter_module_latency_us

    def test_retry_and_wait_penalties_increase_latency(self):
        baseline = simulate_workload(SystemConfig(num_modules=4), circuit_depth=100)
        stressed = simulate_workload(
            SystemConfig(
                num_modules=4,
                bell_pair_retry_rate=1.8,
                memory_wait_time_us=15.0,
                control_jitter_us=5.0,
            ),
            circuit_depth=100,
        )
        assert stressed.retry_overhead_us > 0.0
        assert stressed.memory_wait_overhead_us > 0.0
        assert stressed.control_jitter_overhead_us > 0.0
        assert stressed.expected_attempts_per_bell_pair > baseline.expected_attempts_per_bell_pair
        assert stressed.inter_module_latency_us > baseline.inter_module_latency_us
        assert stressed.effective_circuit_depth > baseline.effective_circuit_depth
        assert stressed.effective_bell_pair_rate_hz < baseline.effective_bell_pair_rate_hz

    def test_added_noise_degrades_fidelity(self):
        clean = simulate_workload(
            SystemConfig(num_modules=4, transduction_efficiency=0.2, added_noise=0.01),
            circuit_depth=50,
        )
        noisy = simulate_workload(
            SystemConfig(num_modules=4, transduction_efficiency=0.2, added_noise=0.12),
            circuit_depth=50,
        )
        assert noisy.added_noise_penalty > clean.added_noise_penalty
        assert noisy.estimated_fidelity < clean.estimated_fidelity

    def test_calibration_quality_improves_link_margin(self):
        weak = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.15,
                transduction_calibration_quality=0.7,
            ),
            circuit_depth=50,
        )
        strong = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.15,
                transduction_calibration_quality=1.0,
            ),
            circuit_depth=50,
        )
        assert strong.effective_transduction_efficiency > weak.effective_transduction_efficiency
        assert strong.link_margin_to_target > weak.link_margin_to_target
        assert strong.retry_adjusted_link_fidelity > weak.retry_adjusted_link_fidelity

    def test_optical_coupling_maps_to_channel_margin(self):
        weak = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.15,
                optical_coupling_efficiency=0.72,
            ),
            circuit_depth=50,
        )
        strong = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.15,
                optical_coupling_efficiency=1.0,
            ),
            circuit_depth=50,
        )
        assert weak.weakest_channel_component == "optical_coupling"
        assert strong.transduction_channel_efficiency > weak.transduction_channel_efficiency
        assert strong.channel_margin_to_target > weak.channel_margin_to_target

    def test_heralding_and_detector_limits_raise_attempt_count(self):
        weak = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.15,
                heralding_success_probability=0.7,
                detector_efficiency=0.8,
            ),
            circuit_depth=50,
        )
        strong = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.15,
                heralding_success_probability=1.0,
                detector_efficiency=1.0,
            ),
            circuit_depth=50,
        )
        assert weak.channel_component_values["heralding"] == pytest.approx(0.7)
        assert weak.channel_component_values["detector"] == pytest.approx(0.8)
        assert weak.expected_attempts_per_bell_pair > strong.expected_attempts_per_bell_pair
        assert weak.link_quality < strong.link_quality

    def test_phase_lock_duty_cycle_degrades_dynamic_link_stability(self):
        weak = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.15,
                phase_lock_duty_cycle=0.7,
            ),
            circuit_depth=50,
        )
        strong = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.15,
                phase_lock_duty_cycle=1.0,
            ),
            circuit_depth=50,
        )
        assert weak.weakest_channel_component == "phase_lock"
        assert strong.dynamic_link_stability > weak.dynamic_link_stability
        assert strong.retry_adjusted_link_fidelity > weak.retry_adjusted_link_fidelity

    def test_phase_stability_reduces_retry_pressure(self):
        unstable = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.15,
                link_phase_stability=0.7,
            ),
            circuit_depth=50,
        )
        stable = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.15,
                link_phase_stability=1.0,
            ),
            circuit_depth=50,
        )
        assert stable.expected_attempts_per_bell_pair < unstable.expected_attempts_per_bell_pair
        assert stable.retry_adjusted_link_fidelity > unstable.retry_adjusted_link_fidelity
        assert stable.link_quality > unstable.link_quality

    def test_dominant_bottleneck_tracks_memory_wait_stress(self):
        stressed = simulate_workload(
            SystemConfig(
                num_modules=4,
                memory_wait_time_us=120.0,
                control_jitter_us=1.0,
            ),
            circuit_depth=100,
        )
        assert stressed.dominant_bottleneck in {
            "memory_wait",
            "transduction_link",
            "entanglement_supply",
        }

    def test_throughput_drops_under_seam_stress(self):
        baseline = simulate_workload(
            SystemConfig(num_modules=4, transduction_efficiency=0.2),
            circuit_depth=100,
        )
        stressed = simulate_workload(
            SystemConfig(
                num_modules=4,
                transduction_efficiency=0.08,
                transduction_loss=0.15,
                bell_pair_retry_rate=1.8,
            ),
            circuit_depth=100,
        )
        assert stressed.throughput_ops_per_sec < baseline.throughput_ops_per_sec

    def test_parallel_links_increase_supply_capacity(self):
        baseline = simulate_workload(
            SystemConfig(num_modules=4, transduction_efficiency=0.12, entanglement_parallel_links=1),
            circuit_depth=100,
        )
        upgraded = simulate_workload(
            SystemConfig(num_modules=4, transduction_efficiency=0.12, entanglement_parallel_links=4),
            circuit_depth=100,
        )
        assert upgraded.effective_bell_pair_rate_hz > baseline.effective_bell_pair_rate_hz
        assert upgraded.entanglement_supply_time_us <= baseline.entanglement_supply_time_us
        assert upgraded.inter_module_latency_us <= baseline.inter_module_latency_us

    def test_buffered_pairs_reduce_supply_pressure(self):
        baseline = simulate_workload(
            SystemConfig(num_modules=4, transduction_efficiency=0.10, entanglement_buffer_pairs=0),
            circuit_depth=100,
        )
        buffered = simulate_workload(
            SystemConfig(num_modules=4, transduction_efficiency=0.10, entanglement_buffer_pairs=600),
            circuit_depth=100,
        )
        assert buffered.buffered_bell_pairs_used > 0
        assert buffered.entanglement_supply_time_us < baseline.entanglement_supply_time_us
        assert buffered.entanglement_supply_utilization <= baseline.entanglement_supply_utilization
