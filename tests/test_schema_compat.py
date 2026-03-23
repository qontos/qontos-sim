"""Schema compatibility check between qontos SDK and qontos-sim.

Verifies that the CircuitIR model fields used by ALL simulator fixtures
match the live SDK schema. Fails fast if the SDK adds required fields that
fixture builders don't provide.

QGH-3303: Initial schema drift prevention
QGH-3402: Extended to cover noisy simulator tests
"""
import ast
import inspect
import pathlib
import pytest
from pydantic import ValidationError

from qontos.models.circuit import CircuitIR, InputFormat, GateOperation


class TestCircuitIRSchemaCompat:
    """Verify simulator fixtures track live SDK schema."""

    def test_qasm_fixture_valid(self):
        """QASM fixture produces valid CircuitIR."""
        from tests.circuit_fixtures import make_qasm_circuit, BELL_QASM
        ir = make_qasm_circuit(BELL_QASM)
        assert isinstance(ir, CircuitIR)
        assert ir.source_type == InputFormat.OPENQASM
        assert ir.depth > 0
        assert ir.gate_count >= 0

    def test_gate_list_fixture_valid(self):
        """Gate-list fixture produces valid CircuitIR."""
        from tests.circuit_fixtures import make_gate_list_circuit
        ir = make_gate_list_circuit(2, [("h", [0]), ("cx", [0, 1])], include_measurements=True)
        assert isinstance(ir, CircuitIR)
        assert ir.source_type == InputFormat.QISKIT
        assert len(ir.gates) == 4  # h, cx, measure, measure

    def test_required_fields_present(self):
        """All required CircuitIR fields are set by fixtures."""
        from tests.circuit_fixtures import make_qasm_circuit, BELL_QASM
        ir = make_qasm_circuit(BELL_QASM)
        # These are the fields that have caused drift before
        assert hasattr(ir, 'depth') and ir.depth is not None
        assert hasattr(ir, 'gate_count') and ir.gate_count is not None
        assert hasattr(ir, 'source_type') and ir.source_type is not None
        assert hasattr(ir, 'num_qubits') and ir.num_qubits > 0

    def test_missing_required_field_rejected(self):
        """CircuitIR rejects construction without required fields."""
        with pytest.raises(ValidationError):
            CircuitIR(
                num_qubits=2,
                gates=[],
                circuit_hash="bad",
                # Missing: depth, gate_count, source_type
            )

    def test_all_required_fields_documented(self):
        """Enumerate required fields so drift is visible."""
        required = []
        for name, field_info in CircuitIR.model_fields.items():
            if field_info.is_required():
                required.append(name)
        # These must always be present — if this set changes, fixtures must update
        assert "num_qubits" in required
        assert "depth" in required
        assert "gate_count" in required
        assert "gates" in required
        assert "source_type" in required


class TestNoAdHocFixtures:
    """Enforce: all CircuitIR construction lives in circuit_fixtures.py.

    Policy: No simulator test file may define its own CircuitIR builder
    function or construct CircuitIR directly. All fixture creation must
    go through the shared builders in tests/circuit_fixtures.py.

    If this test fails, the fix is:
    1. Remove the local builder from the test file
    2. Import make_qasm_circuit or make_gate_list_circuit from circuit_fixtures
    3. If new builder behavior is needed, add it to circuit_fixtures.py
    """

    # Files that are allowed to construct CircuitIR directly
    ALLOWED_FILES = {"circuit_fixtures.py", "test_schema_compat.py"}

    def test_no_local_circuit_ir_builders(self):
        """No test file may define _make_circuit_ir or _make_gate_list_circuit_ir."""
        test_dir = pathlib.Path(__file__).parent
        violations = []
        for py_file in sorted(test_dir.glob("test_*.py")):
            if py_file.name in self.ALLOWED_FILES:
                continue
            source = py_file.read_text()
            for pattern in ["def _make_circuit_ir", "def _make_gate_list_circuit_ir"]:
                if pattern in source:
                    violations.append(
                        f"{py_file.name} defines '{pattern.split('def ')[1]}()' — "
                        f"use circuit_fixtures.py instead"
                    )
        assert not violations, (
            "Stale local CircuitIR builders found. "
            "All fixture construction must use tests/circuit_fixtures.py:\n"
            + "\n".join(f"  ✗ {v}" for v in violations)
        )

    def test_local_tests_import_shared_fixtures(self):
        """test_local_simulator.py must import from circuit_fixtures."""
        test_dir = pathlib.Path(__file__).parent
        source = (test_dir / "test_local_simulator.py").read_text()
        assert "circuit_fixtures" in source, (
            "test_local_simulator.py must import from tests.circuit_fixtures — "
            "found no reference to circuit_fixtures"
        )

    def test_noisy_tests_import_shared_fixtures(self):
        """test_noisy_simulator.py must import from circuit_fixtures."""
        test_dir = pathlib.Path(__file__).parent
        source = (test_dir / "test_noisy_simulator.py").read_text()
        assert "circuit_fixtures" in source, (
            "test_noisy_simulator.py must import from tests.circuit_fixtures — "
            "found no reference to circuit_fixtures"
        )
