"""QGH-3303: Schema compatibility check between qontos SDK and qontos-sim.

Verifies that the CircuitIR model fields used by simulator fixtures match
the live SDK schema. Fails fast if the SDK adds required fields that
fixture builders don't provide.
"""
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
