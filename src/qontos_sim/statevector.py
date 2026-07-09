"""Exact statevector simulator (numpy only).

The state is held as an ``n``-dimensional complex tensor of shape ``(2,) * n``, so
qubit ``i`` is axis ``i``. A gate is applied by contracting its matrix against the
relevant axes with ``np.tensordot`` and moving the new axes back into place. This
is exact (no truncation) and correct for any circuit; it costs ``2**n`` memory, so
it is the right default for small to moderate circuits and the reference the
approximate backends are checked against.

Convention matches ``qontos_sim.circuit``: bitstring position ``i`` is qubit ``i``
(qubit 0 leftmost).
"""

from __future__ import annotations

import cmath
import math
from typing import Dict

import numpy as np

from .circuit import Circuit

_INV_SQRT2 = 1.0 / math.sqrt(2.0)

# Static single-qubit matrices.
_ONE_Q = {
    "h": np.array([[_INV_SQRT2, _INV_SQRT2], [_INV_SQRT2, -_INV_SQRT2]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, cmath.exp(-1j * math.pi / 4)]], dtype=complex),
}


def _param_matrix(name: str, theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if name == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "ry":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz":
        return np.array(
            [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]], dtype=complex
        )
    if name == "p":
        return np.array([[1, 0], [0, cmath.exp(1j * theta)]], dtype=complex)
    raise ValueError(f"unknown parameterised gate {name!r}")


def _apply_1q(state: np.ndarray, mat: np.ndarray, q: int) -> np.ndarray:
    """Contract a 2x2 gate onto axis q."""
    out = np.tensordot(mat, state, axes=([1], [q]))
    # tensordot puts the new axis at position 0; move it back to q
    return np.moveaxis(out, 0, q)


def _apply_2q(state: np.ndarray, mat4: np.ndarray, q0: int, q1: int) -> np.ndarray:
    """Contract a 4x4 gate (ordered as |q0 q1>) onto axes q0, q1."""
    g = mat4.reshape(2, 2, 2, 2)  # out0, out1, in0, in1
    out = np.tensordot(g, state, axes=([2, 3], [q0, q1]))
    # new axes out0,out1 are at positions 0,1; move them to q0,q1
    return np.moveaxis(out, [0, 1], [q0, q1])


# Two-qubit gate matrices in |control target> (or |a b>) basis, row/col = 00,01,10,11.
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_TWO_Q = {"cx": _CX, "cz": _CZ, "swap": _SWAP}


def final_state(circuit: Circuit) -> np.ndarray:
    """Return the exact amplitude vector (length 2**n, flat, qubit 0 most significant)."""
    n = circuit.num_qubits
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0
    for inst in circuit.instructions:
        if inst.name in _ONE_Q:
            state = _apply_1q(state, _ONE_Q[inst.name], inst.qubits[0])
        elif inst.name in ("rx", "ry", "rz", "p"):
            state = _apply_1q(state, _param_matrix(inst.name, inst.params[0]), inst.qubits[0])
        elif inst.name in _TWO_Q:
            state = _apply_2q(state, _TWO_Q[inst.name], inst.qubits[0], inst.qubits[1])
        else:
            raise ValueError(f"statevector simulator cannot apply gate {inst.name!r}")
    return state.reshape(-1)


def probabilities(circuit: Circuit) -> np.ndarray:
    amp = final_state(circuit)
    return np.abs(amp) ** 2


def sample_counts(circuit: Circuit, shots: int, rng: np.random.Generator) -> Dict[str, int]:
    """Sample measurement outcomes on the measured qubits and return a counts dict.

    Only the qubits in ``circuit.measured`` appear in the bitstrings, in qubit order.
    If nothing was explicitly measured, all qubits are measured.
    """
    n = circuit.num_qubits
    measured = circuit.measured or list(range(n))
    probs = probabilities(circuit)
    # guard against tiny negative numbers from floating point
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    draws = rng.choice(len(probs), size=shots, p=probs)

    counts: Dict[str, int] = {}
    # precompute, for each full basis index, the projected bitstring on measured qubits
    for idx in np.unique(draws):
        bits = format(idx, f"0{n}b")  # qubit 0 is leftmost
        key = "".join(bits[q] for q in measured)
        counts[key] = counts.get(key, 0) + int(np.count_nonzero(draws == idx))
    return dict(sorted(counts.items()))
