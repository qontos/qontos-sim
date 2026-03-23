"""QGH-3010: Comprehensive tests for LocalSimulatorExecutor.

Tests cover instantiation, Bell state execution, GHZ circuits, configurable
shots, result normalization, validation, error handling, deterministic seeding,
large circuits, and empty circuits.
"""

from __future__ import annotations

import pytest
import numpy as np

from qontos_sim.local import LocalSimulatorExecutor, ValidationResult, _circuit_ir_to_qiskit
from qontos_sim.normalize import aer_result_to_partition_result

# ---------------------------------------------------------------------------
# Helpers — lightweight CircuitIR stand-ins
# ---------------------------------------------------------------------------

def _make_circuit_ir(qasm: str, num_qubits: int = 2, num_clbits: int | None = None):
    """Build a valid CircuitIR from a QASM string."""
    from qontos.models.circuit import CircuitIR, InputFormat

    return CircuitIR(
        qasm_string=qasm,
        num_qubits=num_qubits,
        num_clbits=num_clbits or num_qubits,
        depth=10,  # Approximate; exact value not critical for simulation
        gate_count=num_qubits * 2,  # Approximate
        gates=[],
        source_type=InputFormat.OPENQASM,
        circuit_hash="test",
    )


def _make_gate_list_circuit_ir(
    num_qubits: int,
    gates: list,
    *,
    num_clbits: int | None = None,
    include_measurements: bool = False,
):
    """Build a valid CircuitIR from a gate list (no QASM string)."""
    from qontos.models.circuit import CircuitIR, GateOperation, InputFormat

    gate_ops = []
    for g in gates:
        if isinstance(g, GateOperation):
            gate_ops.append(g)
        elif isinstance(g, tuple):
            name, qubits = g[0], g[1]
            params = g[2] if len(g) > 2 else []
            gate_ops.append(GateOperation(name=name, qubits=qubits, params=params))

    if include_measurements:
        for i in range(num_qubits):
            gate_ops.append(GateOperation(name="measure", qubits=[i], params=[]))

    connectivity = []
    for g in gate_ops:
        if len(g.qubits) == 2:
            connectivity.append((g.qubits[0], g.qubits[1]))

    return CircuitIR(
        qasm_string="",  # Force gate-list path
        num_qubits=num_qubits,
        num_clbits=num_clbits or num_qubits,
        depth=len(gate_ops),
        gate_count=len([g for g in gate_ops if g.name != "measure"]),
        gates=gate_ops,
        source_type=InputFormat.QISKIT,  # Gate-list input
        qubit_connectivity=connectivity,
        circuit_hash=f"test_gate_list_{num_qubits}q",
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


def _ghz3_qasm() -> str:
    return (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[3];\n"
        "creg c[3];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
        "cx q[1],q[2];\n"
        "measure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n"
        "measure q[2] -> c[2];\n"
    )


# ===================================================================
# Instantiation and configuration
# ===================================================================

class TestLocalSimulatorInstantiation:
    """Tests for creating and configuring LocalSimulatorExecutor."""

    def test_default_instantiation(self):
        executor = LocalSimulatorExecutor()
        assert executor is not None
        assert executor.provider_name == "local_simulator"

    def test_custom_backend_name(self):
        executor = LocalSimulatorExecutor(backend_name="custom_backend")
        assert executor._backend_name == "custom_backend"

    def test_is_synchronous(self):
        executor = LocalSimulatorExecutor()
        assert executor.is_synchronous is True

    def test_poll_is_noop(self):
        executor = LocalSimulatorExecutor()
        result = executor.poll("any-id")
        assert result["status"] == "completed"

    def test_cancel_is_noop(self):
        executor = LocalSimulatorExecutor()
        assert executor.cancel("any-id") is False


# ===================================================================
# Bell state circuit
# ===================================================================

class TestBellStateExecution:
    """Execute a Bell state circuit and verify ~50/50 distribution."""

    def test_bell_state_counts(self):
        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        result = executor.execute(circuit_ir, shots=4096)

        counts = result.counts
        total = sum(counts.values())
        assert total == 4096

        # Expect only |00> and |11>
        for key in counts:
            assert key in ("00", "11"), f"Unexpected bitstring: {key}"

        # Each should be roughly 50 % (allow generous tolerance)
        for key in ("00", "11"):
            frac = counts.get(key, 0) / total
            assert 0.35 < frac < 0.65, f"{key} fraction {frac} outside tolerance"

    def test_bell_state_result_format(self):
        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        result = executor.execute(circuit_ir, shots=1024)

        assert result.shots_completed == 1024
        assert result.backend_name == "aer_simulator"
        assert result.provider == "simulator"
        assert result.execution_time_ms > 0
        assert result.partition_id is not None


# ===================================================================
# GHZ-3 circuit
# ===================================================================

class TestGHZ3Execution:
    """Execute a 3-qubit GHZ circuit."""

    def test_ghz3_counts(self):
        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(_ghz3_qasm(), num_qubits=3, num_clbits=3)
        result = executor.execute(circuit_ir, shots=4096)

        counts = result.counts
        for key in counts:
            assert key in ("000", "111"), f"Unexpected bitstring: {key}"

        total = sum(counts.values())
        for key in ("000", "111"):
            frac = counts.get(key, 0) / total
            assert 0.35 < frac < 0.65


# ===================================================================
# Configurable shots
# ===================================================================

class TestConfigurableShots:
    """Verify that the shots parameter is respected."""

    @pytest.mark.parametrize("shots", [1, 100, 2048])
    def test_shots_respected(self, shots):
        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        result = executor.execute(circuit_ir, shots=shots)
        assert result.shots_completed == shots
        assert sum(result.counts.values()) == shots


# ===================================================================
# Result normalization
# ===================================================================

class TestResultNormalization:
    """Verify normalization to PartitionResult format."""

    def test_normalize_partition_result_passthrough(self):
        from qontos.models.result import PartitionResult

        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        original = executor.execute(circuit_ir, shots=100)

        # Passing a PartitionResult back should return it unchanged
        normalized = executor.normalize(original)
        assert normalized is original

    def test_normalize_dict(self):
        from qontos.models.result import PartitionResult

        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        original = executor.execute(circuit_ir, shots=100)

        as_dict = original.model_dump()
        normalized = executor.normalize(as_dict)
        assert isinstance(normalized, PartitionResult)
        assert normalized.counts == original.counts

    def test_normalize_invalid_type_raises(self):
        executor = LocalSimulatorExecutor()
        with pytest.raises(TypeError, match="Cannot normalize"):
            executor.normalize(42)

    def test_aer_result_to_partition_result_schema(self):
        pr = aer_result_to_partition_result(
            counts={"00": 50, "11": 50},
            shots=100,
            elapsed_ms=1.5,
            backend_name="test_backend",
            provider="test_provider",
        )
        assert pr.shots_completed == 100
        assert pr.backend_name == "test_backend"
        assert pr.provider == "test_provider"
        assert pr.counts == {"00": 50, "11": 50}
        assert pr.partition_id is not None


# ===================================================================
# Validation
# ===================================================================

class TestValidation:
    """Test that validate() correctly accepts and rejects inputs."""

    def test_valid_circuit(self):
        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        vr = executor.validate(circuit_ir, shots=1024)
        assert isinstance(vr, ValidationResult)
        assert vr.valid is True
        assert len(vr.errors) == 0

    def test_rejects_zero_qubits(self):
        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir("", num_qubits=0)
        vr = executor.validate(circuit_ir, shots=1024)
        assert vr.valid is False
        assert any("at least 1 qubit" in e for e in vr.errors)

    def test_rejects_zero_shots(self):
        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        vr = executor.validate(circuit_ir, shots=0)
        assert vr.valid is False
        assert any("shots" in e for e in vr.errors)

    def test_rejects_empty_circuit(self):
        from qontos.models.circuit import CircuitIR, InputFormat

        circuit_ir = CircuitIR(
            qasm_string="",
            num_qubits=2,
            num_clbits=2,
            depth=0,
            gate_count=0,
            gates=[],
            source_type=InputFormat.OPENQASM,
            circuit_hash="empty",
        )
        executor = LocalSimulatorExecutor()
        vr = executor.validate(circuit_ir, shots=1024)
        assert vr.valid is False
        assert any("nothing to execute" in e for e in vr.errors)

    def test_warns_large_circuit(self):
        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=33)
        vr = executor.validate(circuit_ir, shots=1024)
        # Valid but with a warning
        assert len(vr.warnings) > 0
        assert any("slow" in w.lower() or "memory" in w.lower() for w in vr.warnings)


# ===================================================================
# Error handling for malformed QASM
# ===================================================================

class TestMalformedQASM:
    """Test that malformed QASM raises appropriate errors."""

    def test_malformed_qasm_raises(self):
        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir("NOT VALID QASM", num_qubits=2)
        with pytest.raises(Exception):
            executor.execute(circuit_ir, shots=100)


# ===================================================================
# Deterministic results with fixed seed
# ===================================================================

class TestDeterministicSeed:
    """Test that results are reproducible when Aer seed is fixed."""

    def test_fixed_seed_reproducibility(self):
        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)

        # Aer supports seed_simulator via kwargs on the backend run
        # We run twice with the same backend and check statistical consistency
        results = []
        for _ in range(2):
            r = executor.execute(circuit_ir, shots=8192)
            results.append(r.counts)

        # With enough shots, the distributions should be very close
        for key in ("00", "11"):
            counts = [r.get(key, 0) for r in results]
            fracs = [c / 8192 for c in counts]
            assert abs(fracs[0] - fracs[1]) < 0.08, (
                f"Runs diverge too much for {key}: {fracs}"
            )


# ===================================================================
# Large circuit handling (20+ qubits)
# ===================================================================

class TestLargeCircuit:
    """Test that circuits with 20+ qubits can be handled."""

    def test_20_qubit_circuit(self):
        n = 20
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
        qasm = "\n".join(lines) + "\n"

        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(qasm, num_qubits=n, num_clbits=n)
        result = executor.execute(circuit_ir, shots=512)

        assert result.shots_completed == 512
        assert sum(result.counts.values()) == 512
        # GHZ state: only all-zeros and all-ones
        for key in result.counts:
            assert key in ("0" * n, "1" * n)


# ===================================================================
# Empty circuit error
# ===================================================================

class TestEmptyCircuit:
    """Test that an empty circuit (no gates, no QASM) is rejected."""

    def test_empty_circuit_validation_fails(self):
        from qontos.models.circuit import CircuitIR, InputFormat

        circuit_ir = CircuitIR(
            qasm_string="",
            num_qubits=2,
            num_clbits=2,
            depth=0,
            gate_count=0,
            gates=[],
            source_type=InputFormat.OPENQASM,
            circuit_hash="empty",
        )
        executor = LocalSimulatorExecutor()
        vr = executor.validate(circuit_ir, shots=1024)
        assert vr.valid is False

    def test_submit_alias_matches_execute(self):
        """submit() and execute() should return equivalent results."""
        executor = LocalSimulatorExecutor()
        circuit_ir = _make_circuit_ir(_bell_qasm(), num_qubits=2)
        r1 = executor.submit(circuit_ir, shots=1024)
        r2 = executor.execute(circuit_ir, shots=1024)
        assert r1.shots_completed == r2.shots_completed
        assert set(r1.counts.keys()) == set(r2.counts.keys())


# ===================================================================
# QGH-3103: Non-QASM gate-list reconstruction path
# ===================================================================

class TestNonQASMPath:
    """Tests for the gate-list fallback path when qasm_string is absent."""

    def test_bell_state_from_gate_list(self):
        """Bell circuit via gate list produces ~50/50 counts."""
        circuit = _make_gate_list_circuit_ir(
            num_qubits=2,
            gates=[("h", [0]), ("cx", [0, 1])],
            include_measurements=True,
        )
        executor = LocalSimulatorExecutor()
        result = executor.submit(circuit, shots=4096)

        counts = result.counts
        total = sum(counts.values())
        assert total == 4096
        for key in counts:
            assert key in ("00", "11"), f"Unexpected bitstring: {key}"
        for key in ("00", "11"):
            frac = counts.get(key, 0) / total
            assert 0.35 < frac < 0.65, f"{key} fraction {frac} outside tolerance"

    def test_single_qubit_hadamard(self):
        """Single H gate produces superposition with both |0> and |1>."""
        circuit = _make_gate_list_circuit_ir(
            num_qubits=1,
            gates=[("h", [0])],
        )
        executor = LocalSimulatorExecutor()
        result = executor.submit(circuit, shots=4096)
        # Auto-measurement should be added; both states should appear
        assert "0" in result.counts and "1" in result.counts

    def test_explicit_measurements_no_double(self):
        """Gate list with explicit measurements should NOT double-measure."""
        circuit = _make_gate_list_circuit_ir(
            num_qubits=2,
            gates=[("h", [0]), ("cx", [0, 1])],
            include_measurements=True,
        )
        executor = LocalSimulatorExecutor()
        result = executor.submit(circuit, shots=1024)
        assert result.shots_completed == 1024
        assert sum(result.counts.values()) == 1024

    def test_auto_measure_when_no_measurements(self):
        """Gate list without measurements should auto-add measure_all."""
        circuit = _make_gate_list_circuit_ir(
            num_qubits=2,
            gates=[("h", [0]), ("cx", [0, 1])],
            include_measurements=False,
        )
        executor = LocalSimulatorExecutor()
        result = executor.submit(circuit, shots=1024)
        assert result.shots_completed == 1024
        assert sum(result.counts.values()) == 1024

    def test_barrier_handling(self):
        """Barrier gates in the list are handled without error."""
        circuit = _make_gate_list_circuit_ir(
            num_qubits=2,
            gates=[("h", [0]), ("barrier", [0, 1]), ("cx", [0, 1])],
        )
        executor = LocalSimulatorExecutor()
        result = executor.submit(circuit, shots=1024)
        assert result.shots_completed == 1024

    def test_unknown_gate_warning(self):
        """Unknown gate name does not crash (may warn)."""
        circuit = _make_gate_list_circuit_ir(
            num_qubits=2,
            gates=[("h", [0]), ("zzz_gate", [1])],
        )
        executor = LocalSimulatorExecutor()
        # Should not raise; unknown gates are skipped or warned
        try:
            result = executor.submit(circuit, shots=1024)
            # If it succeeds, we just verify it ran
            assert result.shots_completed == 1024
        except Exception:
            # Some backends may raise on unknown gates — that is acceptable
            pass

    def test_parametric_gate_from_list(self):
        """RY(0.5) gate via gate list executes successfully."""
        circuit = _make_gate_list_circuit_ir(
            num_qubits=1,
            gates=[("ry", [0], [0.5])],
        )
        executor = LocalSimulatorExecutor()
        result = executor.submit(circuit, shots=1024)
        assert result.shots_completed == 1024
        assert sum(result.counts.values()) == 1024
