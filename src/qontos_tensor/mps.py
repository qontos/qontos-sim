"""
QONTOS Q-TENSOR: Matrix Product States (MPS)
=============================================

The Matrix Product State is the central data structure for scalable quantum
simulation.  An *n*-qubit state |psi> is represented as a chain of rank-3
tensors::

    A[0] -- A[1] -- A[2] -- ... -- A[n-1]
     |        |       |               |
    phys0   phys1   phys2          phys(n-1)

Each tensor ``A[i]`` has shape ``(d, chi_left, chi_right)`` where:
- ``d`` is the local (physical) dimension (2 for qubits),
- ``chi_left``, ``chi_right`` are the *bond dimensions* connecting to
  neighbouring tensors.

The key insight is that states with limited entanglement can be represented
**exactly** with small bond dimensions, and even highly entangled states can
be **approximated** by truncating to a maximum bond dimension ``chi_max``.

Memory scaling: O(n * d * chi^2) vs O(d^n) for statevector.
For 1000 qubits with chi=256: ~500 MB vs 10^{300} bytes.

References
----------
- Vidal, G. "Efficient classical simulation of slightly entangled quantum
  computations." Physical Review Letters 91.14 (2003): 147902.
- Schollwoeck, U. "The density-matrix renormalization group in the age of
  matrix product states." Annals of Physics 326.1 (2011): 96-192.
- Orus, R. "A practical introduction to tensor networks." Annals of Physics
  349 (2014): 117-158. arXiv:1306.2164

Copyright (c) 2024-2026 QONTOS Inc. All rights reserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Standard Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_PAULIS = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}

# Maximum supported bond dimension
MAX_BOND_DIM = 4096


class MatrixProductState:
    """
    Matrix Product State for *n* qudits of local dimension *d*.

    The MPS is stored as a list of rank-3 numpy arrays.  Convention:
    ``tensors[i]`` has shape ``(d, chi_left, chi_right)`` where axis 0
    is the physical index and axes 1,2 are the virtual (bond) indices.

    The boundary tensors have ``chi_left=1`` (site 0) and ``chi_right=1``
    (site n-1).

    Parameters
    ----------
    tensors : list[np.ndarray]
        Rank-3 tensors in the chain.
    d : int
        Physical dimension (default 2 for qubits).
    """

    def __init__(self, tensors: List[np.ndarray], d: int = 2) -> None:
        self.tensors: List[np.ndarray] = [
            np.asarray(t, dtype=np.complex128) for t in tensors
        ]
        self.d = d
        self._validate()

    def _validate(self) -> None:
        """Basic shape consistency check."""
        n = len(self.tensors)
        for i, t in enumerate(self.tensors):
            if t.ndim != 3:
                raise ValueError(
                    f"Tensor at site {i} has rank {t.ndim}, expected 3"
                )
            if t.shape[0] != self.d:
                raise ValueError(
                    f"Physical dim at site {i} is {t.shape[0]}, expected {self.d}"
                )
        # Check bond compatibility
        for i in range(n - 1):
            if self.tensors[i].shape[2] != self.tensors[i + 1].shape[1]:
                raise ValueError(
                    f"Bond dimension mismatch between sites {i} and {i+1}: "
                    f"{self.tensors[i].shape[2]} != {self.tensors[i+1].shape[1]}"
                )
        # Boundary conditions
        if self.tensors[0].shape[1] != 1:
            raise ValueError("Left boundary must have chi_left=1")
        if self.tensors[-1].shape[2] != 1:
            raise ValueError("Right boundary must have chi_right=1")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_sites(self) -> int:
        """Number of sites (qubits)."""
        return len(self.tensors)

    @property
    def bond_dimensions(self) -> List[int]:
        """Bond dimensions chi_0, chi_1, ..., chi_{n-2} between sites."""
        return [self.tensors[i].shape[2] for i in range(self.n_sites - 1)]

    @property
    def max_bond_dim(self) -> int:
        """Current maximum bond dimension."""
        if self.n_sites <= 1:
            return 1
        return max(self.bond_dimensions)

    @property
    def total_parameters(self) -> int:
        """Total number of complex parameters in the MPS."""
        return sum(t.size for t in self.tensors)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_product_state(
        cls,
        states: List[np.ndarray],
        d: int = 2,
    ) -> "MatrixProductState":
        """
        Build an MPS from a product state |psi_0> x |psi_1> x ... x |psi_{n-1}>.

        Each element of *states* is a 1-D array of length *d*.
        The resulting MPS has bond dimension 1 everywhere.

        Parameters
        ----------
        states : list[np.ndarray]
            Local states, each of shape (d,).
        d : int
            Physical dimension.

        Returns
        -------
        MatrixProductState
        """
        tensors = []
        for s in states:
            s = np.asarray(s, dtype=np.complex128).reshape(d)
            tensors.append(s.reshape(d, 1, 1))
        return cls(tensors, d=d)

    @classmethod
    def zero_state(cls, n: int, d: int = 2) -> "MatrixProductState":
        """All-zero product state |00...0>."""
        s = np.zeros(d, dtype=np.complex128)
        s[0] = 1.0
        return cls.from_product_state([s] * n, d=d)

    @classmethod
    def plus_state(cls, n: int) -> "MatrixProductState":
        """All |+> product state."""
        s = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
        return cls.from_product_state([s] * n, d=2)

    @classmethod
    def from_statevector(
        cls,
        psi: np.ndarray,
        n: int,
        d: int = 2,
        chi_max: int = MAX_BOND_DIM,
        cutoff: float = 1e-12,
    ) -> "MatrixProductState":
        """
        Convert a full statevector to MPS via sequential SVD.

        This peels off one site at a time from the left using SVD,
        truncating singular values below *cutoff* or to *chi_max*.

        Parameters
        ----------
        psi : np.ndarray
            Statevector of shape (d^n,).
        n : int
            Number of sites.
        d : int
            Physical dimension.
        chi_max : int
            Maximum bond dimension.
        cutoff : float
            SVD truncation threshold.

        Returns
        -------
        MatrixProductState
        """
        psi = np.asarray(psi, dtype=np.complex128).ravel()
        expected = d ** n
        if psi.size != expected:
            raise ValueError(f"Statevector size {psi.size} != d^n = {expected}")

        tensors: List[np.ndarray] = []
        remainder = psi.copy()
        chi_left = 1

        for i in range(n - 1):
            remainder = remainder.reshape(chi_left * d, -1)
            U, S, Vh = np.linalg.svd(remainder, full_matrices=False)

            # Truncate
            keep = min(len(S), chi_max)
            mask = S[:keep] > cutoff
            if not np.any(mask):
                keep = 1
            else:
                keep = int(np.sum(mask))
                keep = max(keep, 1)

            U = U[:, :keep]
            S = S[:keep]
            Vh = Vh[:keep, :]

            # Reshape U into (d, chi_left, chi_right)
            A = U.reshape(chi_left, d, keep)
            A = A.transpose(1, 0, 2)  # -> (d, chi_left, keep)
            tensors.append(A)

            remainder = np.diag(S) @ Vh
            chi_left = keep

        # Last site
        last = remainder.reshape(d, chi_left, 1)
        tensors.append(last)

        return cls(tensors, d=d)

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    def apply_single_qubit_gate(self, gate: np.ndarray, site: int) -> None:
        """
        Apply a single-qubit gate (2x2 unitary) at the given site.

        This is an O(d^2 * chi^2) operation -- no bond dimension growth.

        Parameters
        ----------
        gate : np.ndarray
            (d, d) unitary matrix.
        site : int
        """
        gate = np.asarray(gate, dtype=np.complex128)
        # tensors[site] shape: (d, chi_l, chi_r)
        # new_tensor[s, l, r] = sum_s' gate[s, s'] * old[s', l, r]
        self.tensors[site] = np.einsum(
            "ij,jkl->ikl", gate, self.tensors[site]
        )

    def apply_two_qubit_gate(
        self,
        gate: np.ndarray,
        site: int,
        chi_max: int = MAX_BOND_DIM,
        cutoff: float = 1e-12,
    ) -> float:
        """
        Apply a two-qubit gate to adjacent sites (site, site+1).

        The method:
        1. Contract tensors at site and site+1 into a rank-4 object.
        2. Apply the gate.
        3. SVD to split back into two rank-3 tensors, truncating to chi_max.

        Parameters
        ----------
        gate : np.ndarray
            (d^2, d^2) unitary or (d, d, d, d) tensor.
        site : int
            Left site index.
        chi_max : int
            Maximum bond dimension after truncation.
        cutoff : float
            Singular value cutoff.

        Returns
        -------
        float
            Truncation error (sum of discarded squared singular values).
        """
        d = self.d
        gate = np.asarray(gate, dtype=np.complex128)
        if gate.shape == (d * d, d * d):
            gate = gate.reshape(d, d, d, d)

        A = self.tensors[site]       # (d, chi_l, chi_m)
        B = self.tensors[site + 1]   # (d, chi_m, chi_r)
        chi_l = A.shape[1]
        chi_r = B.shape[2]

        # Contract: theta[s1, s2, chi_l, chi_r] = sum_chi_m A[s1,l,m] B[s2,m,r]
        theta = np.einsum("ilm,jmr->ijlr", A, B)

        # Apply gate: theta'[s1', s2', l, r] = sum_{s1,s2} gate[s1',s2',s1,s2] theta[s1,s2,l,r]
        theta = np.einsum("abij,ijlr->ablr", gate, theta)

        # Reshape for SVD: (d*chi_l) x (d*chi_r)
        theta = theta.reshape(d * chi_l, d * chi_r)

        U, S, Vh = np.linalg.svd(theta, full_matrices=False)

        # Truncate
        chi_new = min(len(S), chi_max)
        trunc_error = float(np.sum(S[chi_new:] ** 2))

        # Apply cutoff
        above_cutoff = np.sum(S[:chi_new] > cutoff)
        chi_new = max(int(above_cutoff), 1)

        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]

        # Absorb singular values into U (left-canonical)
        US = U @ np.diag(S)

        # Reshape back
        self.tensors[site] = US.reshape(d, chi_l, chi_new)
        self.tensors[site + 1] = Vh.reshape(chi_new, d, chi_r).transpose(1, 0, 2)

        return trunc_error

    def apply_two_qubit_gate_distant(
        self,
        gate: np.ndarray,
        site_a: int,
        site_b: int,
        chi_max: int = MAX_BOND_DIM,
        cutoff: float = 1e-12,
    ) -> float:
        """
        Apply a two-qubit gate to *non-adjacent* sites using a SWAP network.

        The qubit at site_b is SWAPped adjacent to site_a, the gate is
        applied, and the SWAPs are reversed.

        Parameters
        ----------
        gate : np.ndarray
            (4, 4) or (2, 2, 2, 2) gate.
        site_a, site_b : int
            Target sites (site_a < site_b).
        chi_max : int
        cutoff : float

        Returns
        -------
        float
            Cumulative truncation error.
        """
        if site_a > site_b:
            site_a, site_b = site_b, site_a

        SWAP = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.complex128)

        total_error = 0.0

        # SWAP site_b leftward to site_a + 1
        for i in range(site_b - 1, site_a, -1):
            total_error += self.apply_two_qubit_gate(SWAP, i - 1, chi_max=chi_max, cutoff=cutoff)

        # Apply the actual gate to (site_a, site_a+1)
        total_error += self.apply_two_qubit_gate(gate, site_a, chi_max=chi_max, cutoff=cutoff)

        # SWAP back
        for i in range(site_a + 1, site_b):
            total_error += self.apply_two_qubit_gate(SWAP, i, chi_max=chi_max, cutoff=cutoff)

        return total_error

    # ------------------------------------------------------------------
    # Canonical forms
    # ------------------------------------------------------------------

    def left_canonicalize(self, start: int = 0, stop: Optional[int] = None) -> None:
        """
        Bring sites [start, stop) into left-canonical form via QR.

        After this, A[i]^dagger @ A[i] = I for each site in the range
        (up to numerical precision).
        """
        if stop is None:
            stop = self.n_sites - 1
        for i in range(start, stop):
            d = self.d
            chi_l = self.tensors[i].shape[1]
            chi_r = self.tensors[i].shape[2]
            mat = self.tensors[i].reshape(d * chi_l, chi_r)
            Q, R = np.linalg.qr(mat)
            chi_new = Q.shape[1]
            self.tensors[i] = Q.reshape(d, chi_l, chi_new)
            # Absorb R into next tensor
            self.tensors[i + 1] = np.einsum(
                "ij,kjl->kil", R, self.tensors[i + 1]
            )

    def right_canonicalize(self, start: Optional[int] = None, stop: int = 0) -> None:
        """
        Bring sites (stop, start] into right-canonical form via QR on
        the transposed matrix (equivalent to LQ decomposition).
        """
        if start is None:
            start = self.n_sites - 1
        for i in range(start, stop, -1):
            d = self.d
            chi_l = self.tensors[i].shape[1]
            chi_r = self.tensors[i].shape[2]
            mat = self.tensors[i].reshape(chi_l, d * chi_r)
            # LQ decomposition via QR of transpose
            Q, R = np.linalg.qr(mat.T)
            # L = R.T, Q_actual = Q.T  but we use: mat = L @ Q_actual
            L = R.T
            Q_act = Q.T
            chi_new = Q_act.shape[0]
            self.tensors[i] = Q_act.reshape(chi_new, d, chi_r).transpose(1, 0, 2)
            # Absorb L into previous tensor
            self.tensors[i - 1] = np.einsum(
                "ijk,kl->ijl", self.tensors[i - 1], L
            )

    def mixed_canonicalize(self, center: int) -> None:
        """
        Bring the MPS into mixed-canonical form with orthogonality center
        at *center*.  Sites left of center are left-canonical, sites right
        of center are right-canonical.
        """
        self.left_canonicalize(0, center)
        self.right_canonicalize(self.n_sites - 1, center)

    def normalize(self) -> float:
        """
        Normalize the MPS so that <psi|psi> = 1.  Returns the old norm.
        """
        nrm = self.norm()
        if nrm > 0:
            # Distribute normalization factor to the first tensor
            self.tensors[0] = self.tensors[0] / nrm
        return nrm

    def norm(self) -> float:
        """Compute <psi|psi> via transfer matrix contraction."""
        return float(np.sqrt(np.abs(self.inner_product(self))))

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------

    def truncate(self, chi_max: int, cutoff: float = 1e-12) -> float:
        """
        Global SVD truncation to reduce bond dimension.

        Bring to left-canonical form, then sweep right-to-left with SVD
        truncation.

        Returns
        -------
        float
            Total truncation error.
        """
        self.left_canonicalize()
        total_error = 0.0
        for i in range(self.n_sites - 1, 0, -1):
            d = self.d
            chi_l = self.tensors[i].shape[1]
            chi_r = self.tensors[i].shape[2]
            mat = self.tensors[i].reshape(chi_l, d * chi_r)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            chi_new = min(len(S), chi_max)
            above = np.sum(S[:chi_new] > cutoff)
            chi_new = max(int(above), 1)
            trunc_err = float(np.sum(S[chi_new:] ** 2))
            total_error += trunc_err
            U = U[:, :chi_new]
            S = S[:chi_new]
            Vh = Vh[:chi_new, :]
            self.tensors[i] = Vh.reshape(chi_new, d, chi_r).transpose(1, 0, 2)
            self.tensors[i - 1] = np.einsum(
                "ijk,kl->ijl", self.tensors[i - 1], U @ np.diag(S)
            )
        return total_error

    # ------------------------------------------------------------------
    # Measurements & observables
    # ------------------------------------------------------------------

    def inner_product(self, other: "MatrixProductState") -> complex:
        """
        Compute <self|other> via transfer matrix contraction.

        Contracts left-to-right:  T_i = sum_s  conj(A_i[s]) x B_i[s]
        where x denotes the matrix-like bond contraction.

        Complexity: O(n * d * chi^3).
        """
        if self.n_sites != other.n_sites:
            raise ValueError("MPS lengths differ")

        # Initialize transfer matrix: shape (chi_self_left, chi_other_left) = (1, 1)
        T = np.ones((1, 1), dtype=np.complex128)

        for i in range(self.n_sites):
            A = self.tensors[i]   # (d, chi_l_a, chi_r_a)
            B = other.tensors[i]  # (d, chi_l_b, chi_r_b)
            # T[a,b] -> sum_s conj(A[s,a,a']) * B[s,b,b'] * T[a_old, b_old]
            # = einsum('ab,sac,sbd->cd', T, conj(A), B)
            T = np.einsum("ab,sac,sbd->cd", T, A.conj(), B)

        return complex(T[0, 0])

    def measure(self, n_shots: int = 1024) -> List[List[int]]:
        """
        Sample computational basis measurements from the MPS.

        Uses the sequential conditional probability method:
        1. Left-canonicalize.
        2. For each site i from left to right, compute p(s_i | s_0,...,s_{i-1})
           by contracting the reduced density matrix.
        3. Sample s_i from p(s_i).

        Parameters
        ----------
        n_shots : int

        Returns
        -------
        list of list of int
            Each inner list is a measurement outcome [s_0, s_1, ..., s_{n-1}].
        """
        self.left_canonicalize()
        results: List[List[int]] = []

        for _ in range(n_shots):
            outcome: List[int] = []
            # Track the left environment as we collapse sites
            env = np.ones((1, 1), dtype=np.complex128)

            for i in range(self.n_sites):
                A = self.tensors[i]  # (d, chi_l, chi_r)
                # Reduced density matrix for site i: rho[s, s']
                # rho = sum_{l,l'} env[l, l'] * A[s, l, r] * conj(A[s', l', r])
                rho = np.einsum("ab,sac,tbc->st", env, A, A.conj())
                probs = np.real(np.diag(rho))
                probs = np.abs(probs)
                total = probs.sum()
                if total < 1e-15:
                    probs = np.ones(self.d) / self.d
                else:
                    probs /= total

                s = int(np.random.choice(self.d, p=probs))
                outcome.append(s)

                # Collapse: update environment for the chosen outcome
                # env_new[r, r'] = sum_{l, l'} env[l, l'] * A[s, l, r] * conj(A[s, l', r'])
                A_s = A[s, :, :]  # (chi_l, chi_r)
                env = np.einsum("ab,ac,bd->cd", env, A_s, A_s.conj())
                # Renormalize
                tr = np.trace(env)
                if np.abs(tr) > 1e-15:
                    env /= tr

            results.append(outcome)

        return results

    def expectation_value(self, pauli_string: str) -> complex:
        """
        Compute <psi|O|psi> where O is a Pauli string like "XIZZY".

        Each character must be one of {I, X, Y, Z}.  The string length
        must equal the number of sites.

        Complexity: O(n * d^2 * chi^2).

        Parameters
        ----------
        pauli_string : str

        Returns
        -------
        complex
        """
        if len(pauli_string) != self.n_sites:
            raise ValueError(
                f"Pauli string length {len(pauli_string)} != n_sites {self.n_sites}"
            )

        # Transfer matrix contraction with operator insertion
        T = np.ones((1, 1), dtype=np.complex128)

        for i in range(self.n_sites):
            A = self.tensors[i]  # (d, chi_l, chi_r)
            op = _PAULIS[pauli_string[i]]  # (d, d)
            # OA[s, l, r] = sum_t op[s, t] * A[t, l, r]
            OA = np.einsum("st,tlr->slr", op, A)
            # T_new[a', b'] = sum_{a, b, s} conj(A[s, a, a']) * OA[s, b, b'] * T[a, b]
            T = np.einsum("ab,sac,sbd->cd", T, A.conj(), OA)

        return complex(T[0, 0])

    def entanglement_entropy(self) -> List[float]:
        """
        Compute the von Neumann entanglement entropy at each bond.

        Brings the MPS to left-canonical form and then reads off singular
        values at each bipartition.

        Returns
        -------
        list[float]
            S[i] = entropy of the bipartition (sites 0..i | sites i+1..n-1).
        """
        # Work on a copy
        tensors_copy = [t.copy() for t in self.tensors]
        entropies: List[float] = []

        for i in range(self.n_sites - 1):
            d = self.d
            chi_l = tensors_copy[i].shape[1]
            chi_r = tensors_copy[i].shape[2]
            mat = tensors_copy[i].reshape(d * chi_l, chi_r)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)

            # Von Neumann entropy
            s2 = S ** 2
            s2 = s2[s2 > 1e-20]  # avoid log(0)
            s2 /= s2.sum()
            entropy = -float(np.sum(s2 * np.log2(s2)))
            entropies.append(entropy)

            # Put into left-canonical form for next step
            chi_new = len(S)
            tensors_copy[i] = U.reshape(d, chi_l, chi_new).copy()
            tensors_copy[i + 1] = np.einsum(
                "ij,kjl->kil",
                np.diag(S) @ Vh,
                tensors_copy[i + 1],
            )

        return entropies

    # ------------------------------------------------------------------
    # State reconstruction (small systems only)
    # ------------------------------------------------------------------

    def to_statevector(self) -> np.ndarray:
        """
        Reconstruct the full statevector from the MPS.

        WARNING: This is O(d^n) and only feasible for small systems
        (n <= ~25).

        Returns
        -------
        np.ndarray
            Shape (d^n,).
        """
        if self.n_sites > 25:
            raise ValueError(
                f"to_statevector() refuses to run on {self.n_sites} sites "
                f"(would require {self.d ** self.n_sites} amplitudes)"
            )

        # Contract left to right
        # Start with shape (d, 1, chi_r) -> treat as (d, chi_r)
        result = self.tensors[0][:, 0, :]  # (d, chi_r)

        for i in range(1, self.n_sites):
            A = self.tensors[i]  # (d, chi_l, chi_r)
            # result has shape (d^i, chi_l) after previous steps
            # new_result[..., s, r] = sum_m result[..., m] * A[s, m, r]
            result = np.einsum("...m,smr->...sr", result, A)
            # Merge the last two physical dims
            shape = result.shape
            result = result.reshape(-1, shape[-1])

        return result.ravel()

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> "MatrixProductState":
        """Deep copy."""
        return MatrixProductState(
            [t.copy() for t in self.tensors], d=self.d
        )

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        bds = self.bond_dimensions
        max_chi = max(bds) if bds else 1
        return (
            f"MatrixProductState(n={self.n_sites}, d={self.d}, "
            f"max_chi={max_chi}, params={self.total_parameters})"
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"MPS: {self.n_sites} sites, d={self.d}",
            f"  Bond dimensions: {self.bond_dimensions}",
            f"  Max bond dim:    {self.max_bond_dim}",
            f"  Total params:    {self.total_parameters}",
            f"  Memory (MB):     {self.total_parameters * 16 / 1e6:.2f}",
        ]
        return "\n".join(lines)


# ===================================================================
# Utility: common initial states
# ===================================================================

def ghz_state_mps(n: int, chi_max: int = MAX_BOND_DIM) -> MatrixProductState:
    """
    Construct the MPS for the GHZ state (|00...0> + |11...1>) / sqrt(2).

    The exact MPS has bond dimension 2.

    Parameters
    ----------
    n : int
        Number of qubits.
    chi_max : int
        Not used for GHZ (always chi=2), kept for API consistency.

    Returns
    -------
    MatrixProductState
    """
    tensors: List[np.ndarray] = []

    # First site: shape (2, 1, 2)
    A0 = np.zeros((2, 1, 2), dtype=np.complex128)
    A0[0, 0, 0] = 1.0 / np.sqrt(2)
    A0[1, 0, 1] = 1.0 / np.sqrt(2)
    tensors.append(A0)

    # Bulk sites: shape (2, 2, 2) -- diagonal transfer matrices
    for _ in range(n - 2):
        A = np.zeros((2, 2, 2), dtype=np.complex128)
        A[0, 0, 0] = 1.0
        A[1, 1, 1] = 1.0
        tensors.append(A)

    # Last site: shape (2, 2, 1)
    AN = np.zeros((2, 2, 1), dtype=np.complex128)
    AN[0, 0, 0] = 1.0
    AN[1, 1, 0] = 1.0
    tensors.append(AN)

    return MatrixProductState(tensors)


def w_state_mps(n: int) -> MatrixProductState:
    """
    Construct the MPS for the W state (1/sqrt(n)) * sum_i |0..1_i..0>.

    Exact MPS with bond dimension 2.
    """
    tensors: List[np.ndarray] = []

    # First site
    A0 = np.zeros((2, 1, 2), dtype=np.complex128)
    A0[0, 0, 0] = 1.0
    A0[1, 0, 1] = 1.0 / np.sqrt(n)
    tensors.append(A0)

    # Bulk
    for k in range(1, n - 1):
        A = np.zeros((2, 2, 2), dtype=np.complex128)
        A[0, 0, 0] = 1.0
        A[0, 1, 1] = 1.0
        A[1, 0, 1] = 1.0 / np.sqrt(n)
        tensors.append(A)

    # Last site
    AN = np.zeros((2, 2, 1), dtype=np.complex128)
    AN[0, 1, 0] = 1.0
    AN[1, 0, 0] = 1.0 / np.sqrt(n)
    tensors.append(AN)

    mps = MatrixProductState(tensors)
    mps.normalize()
    return mps
