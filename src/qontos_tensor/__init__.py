"""QONTOS Tensor Network Engine — pure NumPy MPS/MPO/DMRG simulation.

.. warning::
   This package is **experimental** (pre-1.0).  APIs may change between
   minor releases without deprecation warnings.

Public API
----------
Tensor, TensorNetwork        : Core tensor primitives (tensor_core).
MatrixProductState            : MPS data structure and operations (mps).
MatrixProductOperator         : MPO data structure and Hamiltonian factories (mpo).
DMRG, DMRGConfig, DMRGResult : Ground-state search via DMRG (dmrg).
TNSimulator, GateInstruction, SimulationResult : Circuit simulation (circuit_simulator).
"""

from __future__ import annotations

__version__ = "0.1.0"

__stability__ = "experimental"

from qontos_tensor.tensor_core import Tensor, TensorNetwork
from qontos_tensor.mps import MatrixProductState, ghz_state_mps, w_state_mps
from qontos_tensor.mpo import (
    MatrixProductOperator,
    heisenberg_xxz,
    transverse_field_ising,
    from_pauli_string,
    from_hamiltonian,
    identity_mpo,
    molecular_hamiltonian,
)
from qontos_tensor.dmrg import DMRG, DMRGConfig, DMRGResult
from qontos_tensor.circuit_simulator import (
    GateInstruction,
    SimulationResult,
    TNSimulator,
    resolve_gate,
    depolarizing_channel,
    amplitude_damping_channel,
    dephasing_channel,
    ScalabilityDemo,
)

__all__ = [
    # meta
    "__version__",
    "__stability__",
    # tensor_core
    "Tensor",
    "TensorNetwork",
    # mps
    "MatrixProductState",
    "ghz_state_mps",
    "w_state_mps",
    # mpo
    "MatrixProductOperator",
    "heisenberg_xxz",
    "transverse_field_ising",
    "from_pauli_string",
    "from_hamiltonian",
    "identity_mpo",
    "molecular_hamiltonian",
    # dmrg
    "DMRG",
    "DMRGConfig",
    "DMRGResult",
    # circuit_simulator
    "GateInstruction",
    "SimulationResult",
    "TNSimulator",
    "resolve_gate",
    "depolarizing_channel",
    "amplitude_damping_channel",
    "dephasing_channel",
    "ScalabilityDemo",
]
