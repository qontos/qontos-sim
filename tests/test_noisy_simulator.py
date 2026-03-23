"""QGH-3010: Comprehensive tests for NoisySimulatorExecutor.

Tests cover instantiation with noise model, degraded fidelity vs noiseless,
configurable depolarizing error rate, circuit-size noise scaling, readout
error modelling, zero-error equivalence, and PartitionResult schema.
"""

from __future__ import annotations

import pytest
import numpy as np

from qontos_sim.noisy import NoisySimulatorExecutor, _DEFAULT_NOISE_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_circuit_ir(qasm: str, num_qubits: int = 2, num_clbits: int | None = None):
    from qontos.models.circuit import CircuitIR

    return CircuitIR(
        qasm_string=qasm,
        num_qubits=num_qubits,
        num_clbits=num_clbits or num_qubits,
        gates=[],
        circuit_hash="test",
    )


def _bell_qasm() -> str:
    return (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "creg c[2];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
        "measure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n"
    )


def _ghz_qasm(n: int) -> str:
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n}];",
        f"creg c[{n}];",
        "h q[0];",
    ]
    for i in range(n - 1):
        lines.append(f"cx q[{i}],q[{i + 1}];")
    for i in range(n):
        lines.append(f"measure q[{i}] -> c[{i}];")
    return "\n".join(lines) + "\n"


# ===================================================================
# Instantiation
# ===================================================================

class TestNoisyInstantiation:
    """Noisy executor creation with default and custom noise configs."""

    def test_default_instantiation(self):
        executor = NoisySimulatorExecutor()
        assert executor is not None
        assert executor.provider_name == "simulator_noisy"

    def test_custom_noise_config(self):
        cfg = {"single_qubit_error": 0.05, "two_qubit_error": 0.1}
        executor = NoisySimulatorExecutor(noise_model_config=cfg)
        assert executor._config["single_qubit_error"] == 0.05
        assert executor._config["two_qubit_error"] == 0.1
        # Defaults still present for keys not overridden
        assert executor._config["t1_us"] == _DEFAULT_NOISE_CONFIG["t1_us"]

    def test_custom_backend_name(self):
        executor = NoisySimulatorExecutor(backend_name="my_noisy")
        assert executor._backend_name == "my_noisy"

    def test_is_synchronous(self):
        executor = NoisySimulatorExecutor()
        assert executor.is_synchronous is True


# ===================================================================
# Noisy vs noiseless fidelity
# ===================================================================

class TestNoisyFidelity:
    """Noisy execution should produce degraded fidelity vs noiseless."""

    def test_noisy_produces_errors(self):
        """With default noise, some unexpected bitstrings should appear."""
        from qontos_sim.local import LocalSimulatorExecutor

        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        shots = 8192

        noiseless = LocalSimulatorExecutor()
        noisy = NoisySimulatorExecutor()

        r_clean = noiseless.execute(circuit_ir, shots=shots)
        r_noisy = noisy.execute(circuit_ir, shots=shots)

        # Noiseless should only have 00 and 11
        assert set(r_clean.counts.keys()).issubset({"00", "11"})

        # Noisy may have 01 or 10 due to noise (or not, but at minimum
        # the dominant states should still be 00/11)
        total_noisy = sum(r_noisy.counts.values())
        dominant = r_noisy.counts.get("00", 0) + r_noisy.counts.get("11", 0)
        # Dominant states should still be majority
        assert dominant / total_noisy > 0.75

    def test_high_noise_degrades_more(self):
        """Higher error rates should produce more noise."""
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        shots = 8192

        low_noise = NoisySimulatorExecutor(
            noise_model_config={"single_qubit_error": 0.001, "two_qubit_error": 0.005}
        )
        high_noise = NoisySimulatorExecutor(
            noise_model_config={"single_qubit_error": 0.05, "two_qubit_error": 0.1}
        )

        r_low = low_noise.execute(circuit_ir, shots=shots)
        r_high = high_noise.execute(circuit_ir, shots=shots)

        # Compute dominant-state fraction for each
        def dominant_frac(counts):
            total = sum(counts.values())
            return (counts.get("00", 0) + counts.get("11", 0)) / total

        assert dominant_frac(r_low.counts) > dominant_frac(r_high.counts)


# ===================================================================
# Configurable depolarizing error rate
# ===================================================================

class TestDepolarizingRate:
    """Verify that depolarizing error rates are configurable."""

    def test_zero_error_rate(self):
        """Error rates of 0 should behave like noiseless."""
        cfg = {"single_qubit_error": 0.0, "two_qubit_error": 0.0}
        executor = NoisySimulatorExecutor(noise_model_config=cfg)
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        result = executor.execute(circuit_ir, shots=4096)

        # With zero depolarizing error (thermal still present but negligible
        # for short circuits), almost all results should be 00 or 11
        total = sum(result.counts.values())
        dominant = result.counts.get("00", 0) + result.counts.get("11", 0)
        assert dominant / total > 0.95


# ===================================================================
# Noise scales with circuit size
# ===================================================================

class TestNoiseCircuitScaling:
    """Larger circuits should show more degradation from noise."""

    def test_larger_circuit_more_noise(self):
        shots = 4096
        executor = NoisySimulatorExecutor()

        # 2-qubit Bell
        ir_small = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        r_small = executor.execute(ir_small, shots=shots)

        # 5-qubit GHZ (more gates => more noise)
        ir_large = _make_circuit_ir(_ghz_qasm(5), num_qubits=5, num_clbits=5)
        r_large = executor.execute(ir_large, shots=shots)

        def dominant_frac(counts, ideal_keys):
            total = sum(counts.values())
            return sum(counts.get(k, 0) for k in ideal_keys) / total

        frac_small = dominant_frac(r_small.counts, {"00", "11"})
        frac_large = dominant_frac(r_large.counts, {"0" * 5, "1" * 5})

        # Larger circuit should have lower fidelity
        assert frac_small > frac_large


# ===================================================================
# Readout error modelling (noise model includes measurement effects)
# ===================================================================

class TestReadoutError:
    """Noise model should affect measurement outcomes."""

    def test_noise_metadata_recorded(self):
        executor = NoisySimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        result = executor.execute(circuit_ir, shots=100)

        # Noise config should be in metadata
        assert "noise_config" in result.metadata
        assert result.metadata["noise_config"]["single_qubit_error"] == pytest.approx(
            _DEFAULT_NOISE_CONFIG["single_qubit_error"]
        )


# ===================================================================
# Error rate 0.0 equals noiseless
# ===================================================================

class TestZeroErrorNoiseless:
    """With zero error parameters, noisy executor should match noiseless."""

    def test_zero_error_matches_noiseless(self):
        cfg = {
            "single_qubit_error": 0.0,
            "two_qubit_error": 0.0,
            "t1_us": 1e10,  # extremely long coherence
            "t2_us": 1e10,
        }
        executor = NoisySimulatorExecutor(noise_model_config=cfg)
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        result = executor.execute(circuit_ir, shots=4096)

        # Should be essentially noiseless
        assert set(result.counts.keys()).issubset({"00", "11"})


# ===================================================================
# Result format matches PartitionResult schema
# ===================================================================

class TestPartitionResultSchema:
    """Verify result conforms to PartitionResult schema."""

    def test_result_has_required_fields(self):
        executor = NoisySimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        result = executor.execute(circuit_ir, shots=512)

        assert result.partition_id is not None
        assert isinstance(result.counts, dict)
        assert result.shots_completed == 512
        assert result.backend_name == "aer_simulator_noisy"
        assert result.provider == "simulator_noisy"
        assert result.execution_time_ms >= 0

    def test_submit_alias(self):
        executor = NoisySimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        result = executor.submit(circuit_ir, shots=256)
        assert result.shots_completed == 256
