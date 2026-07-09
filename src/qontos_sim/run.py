"""The one function developers call: ``simulate(circuit, shots)``.

Two backends behind one call:

- ``method="statevector"`` (default): exact, numpy only, correct for any circuit,
  costs 2**n memory. Right for small to moderate circuits.
- ``method="mps"``: the matrix-product-state / tensor-network engine, for larger
  circuits whose entanglement stays bounded. Approximate (bond-dimension limited)
  but far cheaper in memory.

Both return the same ``Result`` so switching backends never changes how you read
the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .circuit import Circuit
from . import statevector as _sv


@dataclass
class Result:
    """The outcome of a simulation.

    ``counts`` maps measured bitstrings (qubit 0 leftmost) to occurrences.
    ``statevector`` is the exact amplitude vector when the statevector backend ran,
    else None (the MPS backend does not return a dense vector).
    """

    counts: Dict[str, int]
    shots: int
    method: str
    statevector: Optional[np.ndarray] = None

    def probabilities(self) -> Dict[str, float]:
        return {k: v / self.shots for k, v in self.counts.items()}

    def most_common(self, n: int = 1):
        return sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def __repr__(self) -> str:
        return f"Result(method={self.method!r}, shots={self.shots}, counts={self.counts})"


# gate-name map from this SDK's names to the tensor engine's names
_MPS_NAMES = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "t": "T",
    "cx": "CNOT",
    "cz": "CZ",
    "swap": "SWAP",
    "rx": "Rx",
    "ry": "Ry",
    "rz": "Rz",
}


def _run_mps(circuit: Circuit, shots: int) -> Result:
    try:
        from qontos_tensor.circuit_simulator import TNSimulator, GateInstruction
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "the 'mps' backend needs the qontos_tensor engine, which ships with "
            "qontos-sim; reinstall the package"
        ) from exc

    gates = []
    for inst in circuit.instructions:
        if inst.name not in _MPS_NAMES:
            raise ValueError(
                f"the mps backend does not support gate {inst.name!r}; "
                f"use method='statevector' for this circuit"
            )
        gates.append(
            GateInstruction(
                name=_MPS_NAMES[inst.name],
                qubits=list(inst.qubits),
                params=list(inst.params) or None,
                matrix=None,
            )
        )

    res = TNSimulator(n_qubits=circuit.num_qubits).run(gates, n_shots=shots)
    measured = circuit.measured or list(range(circuit.num_qubits))
    counts: Dict[str, int] = {}
    for shot in res.measurements:
        key = "".join(str(int(shot[q])) for q in measured)
        counts[key] = counts.get(key, 0) + 1
    return Result(counts=dict(sorted(counts.items())), shots=shots, method="mps")


def simulate(
    circuit: Circuit, shots: int = 1024, method: str = "statevector", seed: Optional[int] = None
) -> Result:
    """Run ``circuit`` and return a :class:`Result`.

    Parameters
    ----------
    circuit : Circuit
    shots : number of measurement samples (default 1024).
    method : "statevector" (exact, default) or "mps" (tensor-network, approximate).
    seed : optional RNG seed for reproducible sampling (statevector backend).
    """
    if shots < 1:
        raise ValueError("shots must be at least 1")
    if method == "statevector":
        rng = np.random.default_rng(seed)
        counts = _sv.sample_counts(circuit, shots, rng)
        return Result(
            counts=counts, shots=shots, method="statevector", statevector=_sv.final_state(circuit)
        )
    if method == "mps":
        return _run_mps(circuit, shots)
    raise ValueError(f"unknown method {method!r}; choose 'statevector' or 'mps'")
