"""QGH-3012: Comprehensive tests for the Tensor Network engine.

Tests cover Tensor creation/contraction, MPS initialisation, single- and
two-qubit gate application, inner products, MPO construction, DMRG ground
state search, TNSimulator circuit execution, bond dimension truncation,
memory scaling, and stability labelling.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from qontos_tensor.tensor_core import Tensor, TensorNetwork, contract_pair, random_tensor
from qontos_tensor.mps import MatrixProductState, ghz_state_mps
from qontos_tensor.mpo import (
    MatrixProductOperator,
    heisenberg_xxz,
    transverse_field_ising,
    from_pauli_string,
    identity_mpo,
)
from qontos_tensor.dmrg import DMRG, DMRGConfig, DMRGResult
from qontos_tensor.circuit_simulator import (
    GateInstruction,
    SimulationResult,
    TNSimulator,
    resolve_gate,
)


# ===================================================================
# Tensor creation and contraction
# ===================================================================

class TestTensorCore:
    """Basic Tensor operations."""

    def test_tensor_creation(self):
        t = Tensor(np.eye(2), indices=["a", "b"])
        assert t.rank == 2
        assert t.shape == (2, 2)
        assert t.indices == ["a", "b"]

    def test_tensor_auto_indices(self):
        t = Tensor(np.zeros((3, 4, 5)))
        assert t.indices == ["i0", "i1", "i2"]

    def test_tensor_index_mismatch_raises(self):
        with pytest.raises(ValueError, match="index labels"):
            Tensor(np.eye(2), indices=["a"])

    def test_tensor_contraction_identity(self):
        """I @ I = I."""
        a = Tensor(np.eye(2), indices=["i", "j"])
        b = Tensor(np.eye(2), indices=["j", "k"])
        c = a.contract_with(b)
        assert c.shape == (2, 2)
        np.testing.assert_allclose(c.data, np.eye(2), atol=1e-12)

    def test_tensor_contraction_vector(self):
        """Matrix-vector contraction."""
        mat = Tensor(np.array([[1, 2], [3, 4]], dtype=complex), indices=["i", "j"])
        vec = Tensor(np.array([1, 0], dtype=complex), indices=["j"])
        result = mat.contract_with(vec)
        assert result.shape == (2,)
        np.testing.assert_allclose(result.data, [1, 3], atol=1e-12)

    def test_contract_pair_returns_cost(self):
        a = Tensor(np.eye(3), indices=["i", "j"])
        b = Tensor(np.eye(3), indices=["j", "k"])
        result, cost = contract_pair(a, b)
        assert cost > 0
        assert result.shape == (3, 3)

    def test_tensor_network_contract(self):
        tn = TensorNetwork("test")
        tn.add_tensor(Tensor(np.eye(2), ["a", "b"]))
        tn.add_tensor(Tensor(np.eye(2), ["b", "c"]))
        result = tn.contract()
        assert result.shape == (2, 2)


# ===================================================================
# MPS initialization
# ===================================================================

class TestMPSInit:
    """Matrix Product State creation for n-qubit states."""

    def test_zero_state(self):
        mps = MatrixProductState.zero_state(4)
        assert mps.n_sites == 4
        assert mps.d == 2

    def test_zero_state_bond_dim(self):
        mps = MatrixProductState.zero_state(6)
        assert mps.max_bond_dim == 1

    def test_plus_state(self):
        mps = MatrixProductState.plus_state(3)
        assert mps.n_sites == 3

    def test_ghz_state_mps(self):
        mps = ghz_state_mps(4)
        assert mps.n_sites == 4
        assert mps.max_bond_dim == 2

    def test_from_statevector_bell(self):
        """Convert Bell state statevector to MPS and back."""
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        mps = MatrixProductState.from_statevector(bell, n=2)
        recovered = mps.to_statevector()
        np.testing.assert_allclose(np.abs(recovered), np.abs(bell), atol=1e-10)

    def test_product_state_factory(self):
        states = [np.array([1, 0], dtype=complex)] * 5
        mps = MatrixProductState.from_product_state(states)
        assert mps.n_sites == 5
        assert mps.max_bond_dim == 1


# ===================================================================
# MPS apply_gate single-qubit
# ===================================================================

class TestMPSSingleQubitGate:
    """Apply single-qubit gates via MPS."""

    def test_x_gate_flips_state(self):
        mps = MatrixProductState.zero_state(2)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        mps.apply_single_qubit_gate(X, 0)
        sv = mps.to_statevector()
        # |00> -> |10> (qubit 0 flipped)
        expected = np.array([0, 0, 1, 0], dtype=complex)
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-10)

    def test_hadamard_creates_superposition(self):
        mps = MatrixProductState.zero_state(1)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        mps.apply_single_qubit_gate(H, 0)
        sv = mps.to_statevector()
        np.testing.assert_allclose(np.abs(sv) ** 2, [0.5, 0.5], atol=1e-10)


# ===================================================================
# MPS apply_gate two-qubit
# ===================================================================

class TestMPSTwoQubitGate:
    """Apply two-qubit gates via MPS."""

    def test_cnot_bell_state(self):
        mps = MatrixProductState.zero_state(2)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)

        mps.apply_single_qubit_gate(H, 0)
        mps.apply_two_qubit_gate(CNOT, 0)
        sv = mps.to_statevector()

        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(np.abs(sv), np.abs(bell), atol=1e-10)

    def test_two_qubit_gate_returns_truncation_error(self):
        mps = MatrixProductState.zero_state(2)
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        err = mps.apply_two_qubit_gate(CNOT, 0)
        assert isinstance(err, float)
        assert err >= 0.0


# ===================================================================
# MPS inner product
# ===================================================================

class TestMPSInnerProduct:
    """MPS inner product <psi|phi>."""

    def test_norm_of_zero_state(self):
        mps = MatrixProductState.zero_state(4)
        norm = mps.norm()
        assert abs(norm - 1.0) < 1e-10

    def test_inner_product_orthogonal(self):
        """<00|10> = 0."""
        psi = MatrixProductState.zero_state(2)
        phi = MatrixProductState.zero_state(2)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        phi.apply_single_qubit_gate(X, 0)
        overlap = psi.inner_product(phi)
        assert abs(overlap) < 1e-10

    def test_inner_product_same_state(self):
        mps = MatrixProductState.zero_state(3)
        overlap = mps.inner_product(mps)
        assert abs(overlap - 1.0) < 1e-10


# ===================================================================
# MPO construction for Heisenberg model
# ===================================================================

class TestMPOConstruction:
    """MPO factory functions."""

    def test_heisenberg_xxz_creation(self):
        H = heisenberg_xxz(6, delta=1.0)
        assert isinstance(H, MatrixProductOperator)
        assert H.n_sites == 6
        assert H.max_bond_dim == 5  # exact for Heisenberg

    def test_ising_creation(self):
        H = transverse_field_ising(4, J=1.0, h=0.5)
        assert isinstance(H, MatrixProductOperator)
        assert H.n_sites == 4
        assert H.max_bond_dim == 3

    def test_identity_mpo(self):
        I_mpo = identity_mpo(5)
        assert I_mpo.n_sites == 5
        assert I_mpo.max_bond_dim == 1

    def test_from_pauli_string(self):
        mpo = from_pauli_string("XYZ")
        assert mpo.n_sites == 3


# ===================================================================
# DMRG ground state search
# ===================================================================

class TestDMRG:
    """DMRG ground state search convergence."""

    def test_heisenberg_ground_state_converges(self):
        n = 6
        H = heisenberg_xxz(n, delta=1.0, h=0.0)
        config = DMRGConfig(
            max_bond_dim=32,
            max_sweeps=20,
            convergence_threshold=1e-6,
        )
        dmrg = DMRG(H, config)
        result = dmrg.ground_state()

        assert isinstance(result, DMRGResult)
        assert result.energy < 0  # antiferromagnetic ground state is negative
        assert result.n_sweeps > 0
        assert result.state is not None

    def test_ising_ground_state(self):
        n = 4
        H = transverse_field_ising(n, J=1.0, h=0.5)
        config = DMRGConfig(max_bond_dim=16, max_sweeps=15)
        dmrg = DMRG(H, config)
        result = dmrg.ground_state()
        assert result.energy < 0


# ===================================================================
# TNSimulator circuit execution
# ===================================================================

class TestTNSimulator:
    """Circuit simulation via MPS."""

    def test_simulate_identity(self):
        sim = TNSimulator(n_qubits=2, chi_max=32)
        result = sim.simulate([], n_shots=0)
        assert isinstance(result, SimulationResult)
        assert result.gate_count == 0

    def test_simulate_bell_state(self):
        sim = TNSimulator(n_qubits=2, chi_max=32)
        gates = [
            GateInstruction("H", [0]),
            GateInstruction("CNOT", [0, 1]),
        ]
        result = sim.simulate(gates, n_shots=1000)

        # Count measurements
        counts = {}
        for m in result.measurements:
            key = "".join(str(b) for b in m)
            counts[key] = counts.get(key, 0) + 1

        assert len(result.measurements) == 1000
        # Expect only 00 and 11
        for key in counts:
            assert key in ("00", "11")

    def test_simulate_returns_final_state(self):
        sim = TNSimulator(n_qubits=3, chi_max=32)
        gates = [GateInstruction("H", [0])]
        result = sim.simulate(gates, n_shots=0)
        assert result.final_state is not None
        assert result.final_state.n_sites == 3


# ===================================================================
# TNSimulator results match expected for Bell state
# ===================================================================

class TestTNSimulatorBellAccuracy:
    """Bell state amplitudes should match exact values."""

    def test_bell_state_amplitudes(self):
        sim = TNSimulator(n_qubits=2, chi_max=64)
        gates = [
            GateInstruction("H", [0]),
            GateInstruction("CNOT", [0, 1]),
        ]
        result = sim.simulate(gates, n_shots=0)
        sv = result.final_state.to_statevector()
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-10)


# ===================================================================
# Bond dimension truncation effect on fidelity
# ===================================================================

class TestBondDimensionTruncation:
    """Lower chi_max should introduce truncation error."""

    def test_truncation_error_increases_with_low_chi(self):
        # Build an entangled state that requires high bond dim
        sim_high = TNSimulator(n_qubits=6, chi_max=64)
        sim_low = TNSimulator(n_qubits=6, chi_max=2)

        # Random-ish circuit that creates entanglement
        gates = []
        for i in range(5):
            gates.append(GateInstruction("H", [i]))
            gates.append(GateInstruction("CNOT", [i, i + 1]))

        r_high = sim_high.simulate(gates, n_shots=0)
        r_low = sim_low.simulate(gates, n_shots=0)

        # Low bond dim should have more truncation error
        assert r_low.total_truncation_error >= r_high.total_truncation_error

    def test_mps_truncate_reduces_bond_dim(self):
        mps = ghz_state_mps(8)
        assert mps.max_bond_dim == 2
        # Truncating to chi=1 should lose information
        err = mps.truncate(chi_max=1)
        assert err > 0 or mps.max_bond_dim <= 1


# ===================================================================
# Memory usage scales with bond dimension, not qubit count
# ===================================================================

class TestMemoryScaling:
    """MPS memory should be O(n * chi^2), not O(2^n)."""

    def test_memory_linear_in_n(self):
        mps_10 = MatrixProductState.zero_state(10)
        mps_100 = MatrixProductState.zero_state(100)

        # For product states (chi=1), params should scale linearly
        ratio = mps_100.total_parameters / mps_10.total_parameters
        # Should be close to 10x (100/10)
        assert 8 < ratio < 12

    def test_memory_quadratic_in_chi(self):
        """GHZ state has chi=2, product state has chi=1."""
        product = MatrixProductState.zero_state(20)
        ghz = ghz_state_mps(20)

        # GHZ has chi=2, so ~4x more params per site (chi^2 factor)
        ratio = ghz.total_parameters / product.total_parameters
        assert ratio > 2  # at least 2x, typically ~4x


# ===================================================================
# Stability label: experimental features
# ===================================================================

class TestStabilityLabel:
    """Verify that the tensor package marks experimental status."""

    def test_version_is_pre_1(self):
        import qontos_tensor

        major = int(qontos_tensor.__version__.split(".")[0])
        assert major < 1, "Pre-1.0 indicates experimental status"

    def test_stability_marker_in_init(self):
        import qontos_tensor

        assert hasattr(qontos_tensor, "__stability__")
        assert qontos_tensor.__stability__ == "experimental"


# ===================================================================
# Additional gate resolution tests
# ===================================================================

class TestGateResolution:
    """Test the gate library lookup."""

    def test_resolve_h(self):
        H = resolve_gate("H")
        assert H.shape == (2, 2)

    def test_resolve_cnot(self):
        cnot = resolve_gate("CNOT")
        assert cnot.shape == (4, 4)

    def test_resolve_rx_with_params(self):
        rx = resolve_gate("Rx", [np.pi / 2])
        assert rx.shape == (2, 2)

    def test_unknown_gate_raises(self):
        with pytest.raises(ValueError, match="Unknown gate"):
            resolve_gate("NonExistentGate")

    def test_parametric_gate_without_params_raises(self):
        with pytest.raises(ValueError, match="requires parameters"):
            resolve_gate("Rx")
