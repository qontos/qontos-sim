"""
QONTOS Q-TENSOR: Density Matrix Renormalization Group (DMRG)
=============================================================

A genuine two-site DMRG implementation for finding ground states (and low-lying
excited states) of 1D quantum Hamiltonians represented as MPOs.

Algorithm overview (two-site DMRG):

1. Initialize an MPS |psi> (random or product state).
2. Build left and right environment tensors by contracting <psi|H|psi>
   from the edges inward.
3. Sweep left-to-right then right-to-left.  At each pair of adjacent sites
   (i, i+1):
   a. Form the effective Hamiltonian H_eff acting on the local 2-site space.
   b. Solve the local eigenvalue problem H_eff |theta> = E |theta> via Lanczos.
   c. SVD-split |theta> back into two MPS tensors, truncating to chi_max.
   d. Update the environment tensors.
4. Repeat sweeps until the energy converges.

For excited states, we use the penalty method: H' = H + w * |psi_0><psi_0|
where |psi_0> is the previously found ground state and w is a large weight.

The Lanczos implementation is pure numpy (no scipy.sparse.linalg) so that
DMRG works with zero external dependencies beyond numpy.

References
----------
- White, S.R. "Density matrix formulation for quantum renormalization groups."
  Physical Review Letters 69.19 (1992): 2863.
- White, S.R. "Density-matrix algorithms for quantum renormalization groups."
  Physical Review B 48.14 (1993): 10345.
- Schollwoeck, U. "The density-matrix renormalization group in the age of
  matrix product states." Annals of Physics 326.1 (2011): 96-192.
- Hubig, C., et al. "Strictly single-site DMRG is equivalent to subspace
  expansion." Physical Review B 91.15 (2015): 155115.

Copyright (c) 2024-2026 QONTOS Inc. All rights reserved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from qontos_tensor.mps import MatrixProductState
from qontos_tensor.mpo import MatrixProductOperator

logger = logging.getLogger(__name__)


# ===================================================================
# Configuration and result data classes
# ===================================================================

@dataclass
class DMRGConfig:
    """
    Configuration for a DMRG run.

    Parameters
    ----------
    max_bond_dim : int
        Maximum MPS bond dimension (chi_max).  Controls accuracy vs cost.
        Typical values: 32--256 for quick runs, 512--4096 for production.
    max_sweeps : int
        Maximum number of left-right sweep pairs.
    convergence_threshold : float
        Stop when |E_{new} - E_{old}| < threshold.
    noise : float
        Noise amplitude added to the density matrix during early sweeps
        to escape local minima.  Decays exponentially over sweeps.
    noise_decay : float
        Multiplicative factor applied to noise each sweep.
    lanczos_max_iter : int
        Maximum Lanczos iterations for the local eigensolver.
    lanczos_tol : float
        Convergence tolerance for Lanczos.
    svd_cutoff : float
        Discard singular values smaller than this.
    initial_bond_dim : int
        Bond dimension for the random initial MPS.
    """

    max_bond_dim: int = 256
    max_sweeps: int = 30
    convergence_threshold: float = 1e-8
    noise: float = 1e-4
    noise_decay: float = 0.5
    lanczos_max_iter: int = 20
    lanczos_tol: float = 1e-12
    svd_cutoff: float = 1e-14
    initial_bond_dim: int = 16


@dataclass
class DMRGResult:
    """
    Results of a DMRG computation.

    Attributes
    ----------
    energy : float
        Final variational energy.
    state : MatrixProductState
        Optimized MPS.
    convergence_history : list[float]
        Energy after each sweep.
    bond_dimensions : list[int]
        Max bond dimension after each sweep.
    entanglement_profile : list[float]
        Von Neumann entropy at each bond of the final state.
    wall_time : float
        Total wall-clock time in seconds.
    converged : bool
        Whether the energy converged within max_sweeps.
    n_sweeps : int
        Number of sweeps performed.
    truncation_errors : list[float]
        Cumulative SVD truncation error per sweep.
    """

    energy: float = 0.0
    state: Optional[MatrixProductState] = None
    convergence_history: List[float] = field(default_factory=list)
    bond_dimensions: List[int] = field(default_factory=list)
    entanglement_profile: List[float] = field(default_factory=list)
    wall_time: float = 0.0
    converged: bool = False
    n_sweeps: int = 0
    truncation_errors: List[float] = field(default_factory=list)


# ===================================================================
# DMRG Engine
# ===================================================================

class DMRG:
    """
    Two-site Density Matrix Renormalization Group solver.

    Parameters
    ----------
    hamiltonian : MatrixProductOperator
        The Hamiltonian as an MPO.
    config : DMRGConfig, optional
        Algorithmic parameters.

    Examples
    --------
    >>> from qontos_tensor.mpo import transverse_field_ising
    >>> H = transverse_field_ising(20, J=1.0, h=1.0)
    >>> dmrg = DMRG(H, DMRGConfig(max_bond_dim=64, max_sweeps=20))
    >>> result = dmrg.ground_state()
    >>> print(f"Ground state energy: {result.energy:.10f}")
    """

    def __init__(
        self,
        hamiltonian: MatrixProductOperator,
        config: Optional[DMRGConfig] = None,
    ) -> None:
        self.H = hamiltonian
        self.config = config or DMRGConfig()
        self.n_sites = hamiltonian.n_sites
        self.d = hamiltonian.d

        # Environment tensors (built during sweeps)
        self._left_envs: List[Optional[np.ndarray]] = [None] * (self.n_sites + 1)
        self._right_envs: List[Optional[np.ndarray]] = [None] * (self.n_sites + 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ground_state(
        self,
        initial_state: Optional[MatrixProductState] = None,
    ) -> DMRGResult:
        """
        Find the ground state energy and MPS via two-site DMRG.

        Parameters
        ----------
        initial_state : MatrixProductState, optional
            Starting MPS.  If None, a random MPS is used.

        Returns
        -------
        DMRGResult
        """
        t0 = time.time()
        cfg = self.config

        # Initialize MPS
        if initial_state is not None:
            psi = initial_state.copy()
        else:
            psi = self._random_mps(cfg.initial_bond_dim)

        # Bring to right-canonical form
        psi.right_canonicalize()
        psi.normalize()

        # Build initial environments
        self._build_environments(psi)

        result = DMRGResult()
        energy = 0.0
        noise = cfg.noise

        for sweep in range(cfg.max_sweeps):
            sweep_error = 0.0

            # Left-to-right sweep
            for i in range(self.n_sites - 1):
                energy, trunc_err = self._optimize_two_site(psi, i, "right", noise)
                sweep_error += trunc_err

            # Right-to-left sweep
            for i in range(self.n_sites - 2, -1, -1):
                energy, trunc_err = self._optimize_two_site(psi, i, "left", noise)
                sweep_error += trunc_err

            result.convergence_history.append(energy)
            result.bond_dimensions.append(psi.max_bond_dim)
            result.truncation_errors.append(sweep_error)

            logger.info(
                "DMRG sweep %d: E = %.12f, max_chi = %d, trunc_err = %.2e",
                sweep + 1,
                energy,
                psi.max_bond_dim,
                sweep_error,
            )

            # Check convergence
            if len(result.convergence_history) >= 2:
                dE = abs(result.convergence_history[-1] - result.convergence_history[-2])
                if dE < cfg.convergence_threshold:
                    result.converged = True
                    logger.info(
                        "DMRG converged after %d sweeps (dE = %.2e)",
                        sweep + 1,
                        dE,
                    )
                    break

            # Decay noise
            noise *= cfg.noise_decay

        result.energy = energy
        result.state = psi
        result.n_sweeps = len(result.convergence_history)
        result.wall_time = time.time() - t0
        result.entanglement_profile = psi.entanglement_entropy()

        return result

    def excited_states(
        self,
        n_states: int = 3,
        penalty_weight: float = 100.0,
        initial_states: Optional[List[MatrixProductState]] = None,
    ) -> List[DMRGResult]:
        """
        Find excited states via the penalty method.

        For each excited state k, we solve:
            H_k = H + w * sum_{j<k} |psi_j><psi_j|

        This is implemented by modifying the effective Hamiltonian in the
        local eigensolve to include penalty overlaps with previously found
        states.

        Parameters
        ----------
        n_states : int
            Total number of states to find (including ground state).
        penalty_weight : float
            Weight w for the penalty terms.
        initial_states : list[MatrixProductState], optional
            Starting states for each solve.

        Returns
        -------
        list[DMRGResult]
        """
        results: List[DMRGResult] = []
        found_states: List[MatrixProductState] = []

        for k in range(n_states):
            logger.info("DMRG: finding state %d of %d", k + 1, n_states)

            if k == 0:
                # Ground state: standard DMRG
                init = initial_states[0] if initial_states and len(initial_states) > 0 else None
                result = self.ground_state(initial_state=init)
            else:
                # Excited state: penalty method
                init = initial_states[k] if initial_states and len(initial_states) > k else None
                result = self._excited_state_dmrg(
                    found_states, penalty_weight, initial_state=init
                )

            results.append(result)
            if result.state is not None:
                found_states.append(result.state.copy())

        return results

    # ------------------------------------------------------------------
    # Core: two-site optimization
    # ------------------------------------------------------------------

    def _optimize_two_site(
        self,
        psi: MatrixProductState,
        site: int,
        direction: str,
        noise: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Optimize the two-site tensor at (site, site+1).

        Parameters
        ----------
        psi : MatrixProductState
        site : int
            Left site index.
        direction : str
            "right" or "left" -- determines how to split the updated tensor.
        noise : float
            Noise amplitude for the density matrix.

        Returns
        -------
        energy : float
        trunc_error : float
        """
        cfg = self.config
        d = self.d

        A = psi.tensors[site]        # (d, chi_l, chi_m)
        B = psi.tensors[site + 1]    # (d, chi_m, chi_r)
        chi_l = A.shape[1]
        chi_r = B.shape[2]

        # Form the two-site tensor theta: (d, d, chi_l, chi_r)
        theta = np.einsum("ilm,jmr->ijlr", A, B)

        # Get effective Hamiltonian action as a function
        L = self._left_envs[site]         # (chi_l_psi, chi_l_H, chi_l_psi)
        R = self._right_envs[site + 2]    # (chi_r_psi, chi_r_H, chi_r_psi)
        W1 = self.H.tensors[site]         # (d, d, chi_l_H, chi_r_H_mid)
        W2 = self.H.tensors[site + 1]     # (d, d, chi_r_H_mid, chi_r_H)

        def h_eff(vec: np.ndarray) -> np.ndarray:
            """Apply the effective Hamiltonian to a flattened 2-site vector."""
            psi_2site = vec.reshape(d, d, chi_l, chi_r)
            # Contract: L[a,w,b] * W1[s1',s1,w,x] * W2[s2',s2,x,y] * R[c,y,d] * psi[s1,s2,a,c]
            # -> result[s1',s2',b,d]
            # Step by step:
            # 1. Contract L with psi: tmp1[s1,s2,w,c] = L[a,w,b->skip] ... let me use einsum
            # result = einsum('awb, stuv, stxy, cyd, svbd -> ...')
            # More carefully:

            # tmp1[s1,s2,w,r] = sum_l L[l,w,l'] * psi[s1,s2,l',r]  -- but L has 3 indices
            # Wait, L shape: (chi_l_psi, chi_l_H, chi_l_psi) -- indices (bra_l, H_l, ket_l)

            # Full contraction:
            # H_eff * psi = sum over contracted indices:
            # L[a, w, b] * W1[s1out, s1in, w, x] * W2[s2out, s2in, x, y]
            # * R[c, y, d] * psi[s1in, s2in, b, d]
            # -> result[s1out, s2out, a, c]

            result = np.einsum(
                "awb,pqwx,rsxy,cyd,qsbd->prac",
                L, W1, W2, R, psi_2site,
                optimize="greedy",
            )
            return result.ravel()

        # Solve local eigenvalue problem via Lanczos
        vec_size = d * d * chi_l * chi_r
        theta_flat = theta.ravel()

        eigenvalue, eigenvector = self._lanczos(
            h_eff,
            theta_flat,
            vec_size,
            max_iter=cfg.lanczos_max_iter,
            tol=cfg.lanczos_tol,
        )

        # Reshape back to 2-site tensor
        theta_opt = eigenvector.reshape(d, d, chi_l, chi_r)

        # SVD split
        if direction == "right":
            # Left-canonical: theta -> A * SV^dag
            mat = theta_opt.reshape(d * chi_l, d * chi_r)
            # Transpose to group (d, chi_l) as rows and (d, chi_r) as cols
            mat = theta_opt.transpose(0, 2, 1, 3).reshape(d * chi_l, d * chi_r)
        else:
            mat = theta_opt.transpose(0, 2, 1, 3).reshape(d * chi_l, d * chi_r)

        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Add noise to singular values (helps convergence in early sweeps)
        if noise > 0 and len(S) > 1:
            noise_vec = noise * np.random.randn(len(S))
            S = np.abs(S + noise_vec * S[0])

        # Truncate
        chi_new = min(len(S), cfg.max_bond_dim)
        above_cutoff = np.sum(S[:chi_new] > cfg.svd_cutoff)
        chi_new = max(int(above_cutoff), 1)
        trunc_error = float(np.sum(S[chi_new:] ** 2))

        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]

        # Normalize singular values
        norm = np.linalg.norm(S)
        if norm > 0:
            S /= norm

        if direction == "right":
            # A is left-canonical, absorb S*Vh into B
            psi.tensors[site] = U.reshape(d, chi_l, chi_new)
            SV = np.diag(S) @ Vh
            psi.tensors[site + 1] = SV.reshape(chi_new, d, chi_r).transpose(1, 0, 2)
            # Update left environment
            self._update_left_env(psi, site)
        else:
            # B is right-canonical, absorb U*S into A
            US = U @ np.diag(S)
            psi.tensors[site] = US.reshape(d, chi_l, chi_new)
            psi.tensors[site + 1] = Vh.reshape(chi_new, d, chi_r).transpose(1, 0, 2)
            # Update right environment
            self._update_right_env(psi, site + 1)

        return eigenvalue, trunc_error

    # ------------------------------------------------------------------
    # Lanczos eigensolver
    # ------------------------------------------------------------------

    def _lanczos(
        self,
        matvec,
        initial_vec: np.ndarray,
        vec_size: int,
        max_iter: int = 20,
        tol: float = 1e-12,
    ) -> Tuple[float, np.ndarray]:
        """
        Lanczos algorithm for finding the smallest eigenvalue/eigenvector.

        This is a numpy-only implementation of the Lanczos algorithm for
        Hermitian operators.  We build a Krylov subspace and diagonalize
        the tridiagonal matrix to extract the ground state.

        Parameters
        ----------
        matvec : callable
            Function that applies the operator to a vector.
        initial_vec : np.ndarray
            Starting vector (will be normalized).
        vec_size : int
            Dimension of the vector space.
        max_iter : int
            Maximum number of Lanczos steps.
        tol : float
            Convergence tolerance for the eigenvalue.

        Returns
        -------
        eigenvalue : float
        eigenvector : np.ndarray
        """
        # Normalize initial vector
        v = initial_vec.astype(np.complex128).copy()
        norm = np.linalg.norm(v)
        if norm < 1e-15:
            v = np.random.randn(vec_size) + 1j * np.random.randn(vec_size)
            norm = np.linalg.norm(v)
        v /= norm

        # Lanczos vectors
        V = np.zeros((max_iter + 1, vec_size), dtype=np.complex128)
        V[0] = v

        # Tridiagonal matrix elements
        alpha = np.zeros(max_iter + 1, dtype=np.float64)  # diagonal
        beta = np.zeros(max_iter + 1, dtype=np.float64)   # off-diagonal

        prev_eigenvalue = float("inf")

        for j in range(min(max_iter, vec_size)):
            w = matvec(V[j])

            # alpha_j = <v_j | w>
            alpha[j] = np.real(np.dot(V[j].conj(), w))

            # Orthogonalize
            w = w - alpha[j] * V[j]
            if j > 0:
                w = w - beta[j] * V[j - 1]

            # Full reorthogonalization (crucial for numerical stability)
            for k in range(j + 1):
                overlap = np.dot(V[k].conj(), w)
                w -= overlap * V[k]

            beta[j + 1] = np.linalg.norm(w)

            if beta[j + 1] < 1e-14:
                # Invariant subspace found
                max_iter = j + 1
                break

            V[j + 1] = w / beta[j + 1]

            # Check convergence by diagonalizing the tridiagonal matrix
            if j >= 1:
                T = np.diag(alpha[: j + 1]) + np.diag(beta[1: j + 1], 1) + np.diag(
                    beta[1: j + 1], -1
                )
                evals, evecs = np.linalg.eigh(T)
                eigenvalue = evals[0]

                if abs(eigenvalue - prev_eigenvalue) < tol:
                    # Converged: reconstruct eigenvector
                    coeffs = evecs[:, 0]
                    eigenvector = np.zeros(vec_size, dtype=np.complex128)
                    for k in range(j + 1):
                        eigenvector += coeffs[k] * V[k]
                    eigenvector /= np.linalg.norm(eigenvector)
                    return float(eigenvalue), eigenvector

                prev_eigenvalue = eigenvalue

        # Return best result from final tridiagonal matrix
        k = min(max_iter, vec_size)
        if k < 1:
            return float(alpha[0]), V[0]

        T = np.diag(alpha[:k]) + np.diag(beta[1:k], 1) + np.diag(beta[1:k], -1)
        evals, evecs = np.linalg.eigh(T)

        coeffs = evecs[:, 0]
        eigenvector = np.zeros(vec_size, dtype=np.complex128)
        for i in range(k):
            eigenvector += coeffs[i] * V[i]
        eigenvector /= np.linalg.norm(eigenvector)

        return float(evals[0]), eigenvector

    # ------------------------------------------------------------------
    # Environment construction
    # ------------------------------------------------------------------

    def _build_environments(self, psi: MatrixProductState) -> None:
        """
        Build all left and right environment tensors from scratch.

        Left environment at site i:
            L[i] has shape (chi_psi, chi_H, chi_psi) and represents
            the contraction of <psi| H |psi> over sites 0..i-1.

        Right environment at site i:
            R[i] has shape (chi_psi, chi_H, chi_psi) and represents
            the contraction over sites i..n-1.

        Boundary conditions:
            L[0] = R[n] = [[[1]]]  (trivial 1x1x1 tensors).
        """
        n = self.n_sites

        # Trivial boundaries
        self._left_envs[0] = np.ones((1, 1, 1), dtype=np.complex128)
        self._right_envs[n] = np.ones((1, 1, 1), dtype=np.complex128)

        # Build right environments from right to left
        for i in range(n - 1, -1, -1):
            self._update_right_env_from_scratch(psi, i)

    def _update_right_env_from_scratch(self, psi: MatrixProductState, site: int) -> None:
        """Compute R[site] from R[site+1]."""
        A = psi.tensors[site]       # (d, chi_l, chi_r)
        W = self.H.tensors[site]    # (d, d, chi_l_H, chi_r_H)
        R = self._right_envs[site + 1]  # (chi_r, chi_r_H, chi_r)

        # R_new[a, w, b] = sum_{s,s',c,y,d} conj(A[s',a,c]) * W[s',s,w,y] * A[s,b,d] * R[c,y,d]
        # Step 1: contract A with R
        tmp1 = np.einsum("sbd,cyd->sbcy", A, R)
        # Step 2: contract with W
        tmp2 = np.einsum("sbcy,tswx->tbwxcy", tmp1, W)
        # Hmm, this is getting tangled. Let me do it more carefully.

        # R_new[a, w, b] = sum_{s, s', c, y, d}
        #   conj(A[s', a, c]) * W[s', s, w, y] * A[s, b, d] * R_old[c, y, d]

        # Contract step by step:
        # tmp1[s, b, y] = sum_d A[s, b, d] * R_old[:, y, d] -- no, R is (chi_r, chi_r_H, chi_r)
        # Let's index R as R[c, y, d]

        # Step 1: tmp[s, b, c, y] = sum_d A[s, b, d] * R[c, y, d]
        tmp = np.einsum("sbd,cyd->sbcy", A, R)
        # Step 2: tmp2[s', a, w] = sum_{s, b, c, y} conj(A[s',a,c]) * W[s',s,w,y] * tmp[s,b,c,y]
        # -- too many indices. Let me merge differently.

        # Alternative order:
        # Step 1: tmp1[s, w, b, d] = sum_y W[:, s, w, y] * ... nope.
        # Let me use one big einsum:
        R_new = np.einsum(
            "tac,tswx,sbd,cxd->awb",
            A.conj(), W, A, R,
            optimize="greedy",
        )
        self._right_envs[site] = R_new

    def _update_left_env(self, psi: MatrixProductState, site: int) -> None:
        """Update L[site+1] from L[site] after optimizing site."""
        A = psi.tensors[site]       # (d, chi_l, chi_r)
        W = self.H.tensors[site]    # (d, d, chi_l_H, chi_r_H)
        L = self._left_envs[site]   # (chi_l, chi_l_H, chi_l)

        # L_new[c, y, d] = sum_{s, s', a, w, b}
        #   conj(A[s', a, c]) * W[s', s, w, y] * A[s, b, d] * L[a, w, b]
        L_new = np.einsum(
            "awb,tac,tswx,sbd->cxd",
            L, A.conj(), W, A,
            optimize="greedy",
        )
        self._left_envs[site + 1] = L_new

    def _update_right_env(self, psi: MatrixProductState, site: int) -> None:
        """Update R[site] from R[site+1] after optimizing site."""
        self._update_right_env_from_scratch(psi, site)

    # ------------------------------------------------------------------
    # Helper: random initial MPS
    # ------------------------------------------------------------------

    def _random_mps(self, bond_dim: int) -> MatrixProductState:
        """
        Create a random MPS with the given bond dimension.

        The bond dimension ramps up from 1 at the boundaries to
        min(d^i, d^(n-i), bond_dim) in the bulk.
        """
        n = self.n_sites
        d = self.d
        tensors: List[np.ndarray] = []

        chi_left = 1
        for i in range(n):
            chi_max_left = d ** (i + 1)
            chi_max_right = d ** (n - i)
            if i < n - 1:
                chi_right = min(bond_dim, chi_max_left, chi_max_right)
            else:
                chi_right = 1

            A = np.random.randn(d, chi_left, chi_right) + \
                1j * np.random.randn(d, chi_left, chi_right)
            A /= np.linalg.norm(A)
            tensors.append(A)
            chi_left = chi_right

        return MatrixProductState(tensors, d=d)

    # ------------------------------------------------------------------
    # Excited states via penalty method
    # ------------------------------------------------------------------

    def _excited_state_dmrg(
        self,
        lower_states: List[MatrixProductState],
        penalty_weight: float,
        initial_state: Optional[MatrixProductState] = None,
    ) -> DMRGResult:
        """
        DMRG for an excited state using the penalty method.

        H_eff' = H_eff + w * sum_k |psi_k><psi_k|

        The penalty term is added to the local effective Hamiltonian by
        computing overlaps of the two-site tensor with the corresponding
        two-site tensors of the lower states.
        """
        t0 = time.time()
        cfg = self.config

        if initial_state is not None:
            psi = initial_state.copy()
        else:
            psi = self._random_mps(cfg.initial_bond_dim)

        psi.right_canonicalize()
        psi.normalize()
        self._build_environments(psi)

        # Build environments for penalty states
        penalty_left_envs = []
        penalty_right_envs = []
        for state in lower_states:
            l_envs, r_envs = self._build_penalty_environments(psi, state)
            penalty_left_envs.append(l_envs)
            penalty_right_envs.append(r_envs)

        result = DMRGResult()
        energy = 0.0
        noise = cfg.noise

        for sweep in range(cfg.max_sweeps):
            sweep_error = 0.0

            for i in range(self.n_sites - 1):
                energy, trunc_err = self._optimize_two_site_excited(
                    psi, i, "right", noise, lower_states,
                    penalty_weight, penalty_left_envs, penalty_right_envs,
                )
                sweep_error += trunc_err

            for i in range(self.n_sites - 2, -1, -1):
                energy, trunc_err = self._optimize_two_site_excited(
                    psi, i, "left", noise, lower_states,
                    penalty_weight, penalty_left_envs, penalty_right_envs,
                )
                sweep_error += trunc_err

            result.convergence_history.append(energy)
            result.bond_dimensions.append(psi.max_bond_dim)
            result.truncation_errors.append(sweep_error)

            if len(result.convergence_history) >= 2:
                dE = abs(result.convergence_history[-1] - result.convergence_history[-2])
                if dE < cfg.convergence_threshold:
                    result.converged = True
                    break

            noise *= cfg.noise_decay

        result.energy = energy
        result.state = psi
        result.n_sweeps = len(result.convergence_history)
        result.wall_time = time.time() - t0
        result.entanglement_profile = psi.entanglement_entropy()

        return result

    def _build_penalty_environments(
        self,
        psi: MatrixProductState,
        target: MatrixProductState,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Build left and right overlap environments between |psi> and |target>.

        L_pen[i] has shape (chi_psi, chi_target) representing <target|psi>
        contracted over sites 0..i-1.
        """
        n = self.n_sites
        left_envs: List[np.ndarray] = [None] * (n + 1)
        right_envs: List[np.ndarray] = [None] * (n + 1)

        left_envs[0] = np.ones((1, 1), dtype=np.complex128)
        right_envs[n] = np.ones((1, 1), dtype=np.complex128)

        for i in range(n):
            A = psi.tensors[i]      # (d, chi_l_psi, chi_r_psi)
            B = target.tensors[i]   # (d, chi_l_tar, chi_r_tar)
            L = left_envs[i]        # (chi_l_psi, chi_l_tar)
            left_envs[i + 1] = np.einsum("ab,sac,sbd->cd", L, A, B.conj())

        for i in range(n - 1, -1, -1):
            A = psi.tensors[i]
            B = target.tensors[i]
            R = right_envs[i + 1]
            right_envs[i] = np.einsum("sac,sbd,cd->ab", A, B.conj(), R)

        return left_envs, right_envs

    def _optimize_two_site_excited(
        self,
        psi: MatrixProductState,
        site: int,
        direction: str,
        noise: float,
        lower_states: List[MatrixProductState],
        penalty_weight: float,
        penalty_left_envs: List[List[np.ndarray]],
        penalty_right_envs: List[List[np.ndarray]],
    ) -> Tuple[float, float]:
        """
        Two-site optimization with penalty terms for excited states.
        """
        cfg = self.config
        d = self.d

        A = psi.tensors[site]
        B = psi.tensors[site + 1]
        chi_l = A.shape[1]
        chi_r = B.shape[2]

        theta = np.einsum("ilm,jmr->ijlr", A, B)

        L = self._left_envs[site]
        R = self._right_envs[site + 2]
        W1 = self.H.tensors[site]
        W2 = self.H.tensors[site + 1]

        def h_eff(vec: np.ndarray) -> np.ndarray:
            psi_2site = vec.reshape(d, d, chi_l, chi_r)

            # Hamiltonian part
            result = np.einsum(
                "awb,pqwx,rsxy,cyd,qsbd->prac",
                L, W1, W2, R, psi_2site,
                optimize="greedy",
            )

            # Penalty terms: w * |phi_k><phi_k| * |psi>
            for k, state in enumerate(lower_states):
                Ak = state.tensors[site]
                Bk = state.tensors[site + 1]
                Lp = penalty_left_envs[k][site]
                Rp = penalty_right_envs[k][site + 2]

                # <phi_k|psi> at the two-site level
                theta_k = np.einsum("ilm,jmr->ijlr", Ak, Bk)  # target 2-site tensor
                # overlap = <phi_k_local|psi_local> with environments
                overlap = np.einsum(
                    "ab,ijac,ijbd,cd->",
                    Lp, theta_k.conj(), psi_2site, Rp,
                )
                # penalty: w * overlap * |phi_k_local>
                proj = np.einsum(
                    "ab,ijac,cd->ijbd",
                    Lp, theta_k.conj(), Rp,
                )
                result += penalty_weight * overlap * proj

            return result.ravel()

        vec_size = d * d * chi_l * chi_r
        theta_flat = theta.ravel()

        eigenvalue, eigenvector = self._lanczos(
            h_eff, theta_flat, vec_size,
            max_iter=cfg.lanczos_max_iter, tol=cfg.lanczos_tol,
        )

        # SVD split (same as ground state)
        theta_opt = eigenvector.reshape(d, d, chi_l, chi_r)
        mat = theta_opt.transpose(0, 2, 1, 3).reshape(d * chi_l, d * chi_r)

        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        if noise > 0 and len(S) > 1:
            noise_vec = noise * np.random.randn(len(S))
            S = np.abs(S + noise_vec * S[0])

        chi_new = min(len(S), cfg.max_bond_dim)
        above_cutoff = np.sum(S[:chi_new] > cfg.svd_cutoff)
        chi_new = max(int(above_cutoff), 1)
        trunc_error = float(np.sum(S[chi_new:] ** 2))

        U, S, Vh = U[:, :chi_new], S[:chi_new], Vh[:chi_new, :]
        norm = np.linalg.norm(S)
        if norm > 0:
            S /= norm

        if direction == "right":
            psi.tensors[site] = U.reshape(d, chi_l, chi_new)
            SV = np.diag(S) @ Vh
            psi.tensors[site + 1] = SV.reshape(chi_new, d, chi_r).transpose(1, 0, 2)
            self._update_left_env(psi, site)
            # Update penalty left envs
            for k, state in enumerate(lower_states):
                Ak = psi.tensors[site]
                Bk_target = state.tensors[site]
                Lp = penalty_left_envs[k][site]
                penalty_left_envs[k][site + 1] = np.einsum(
                    "ab,sac,sbd->cd", Lp, Ak, Bk_target.conj()
                )
        else:
            US = U @ np.diag(S)
            psi.tensors[site] = US.reshape(d, chi_l, chi_new)
            psi.tensors[site + 1] = Vh.reshape(chi_new, d, chi_r).transpose(1, 0, 2)
            self._update_right_env(psi, site + 1)
            for k, state in enumerate(lower_states):
                Bk = psi.tensors[site + 1]
                Bk_target = state.tensors[site + 1]
                Rp = penalty_right_envs[k][site + 2]
                penalty_right_envs[k][site + 1] = np.einsum(
                    "sac,sbd,cd->ab", Bk, Bk_target.conj(), Rp
                )

        return eigenvalue, trunc_error
