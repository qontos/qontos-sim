"""
QONTOS Q-TENSOR: Core Tensor Operations
========================================

Foundation layer providing arbitrary-rank tensor manipulation, contraction,
and decomposition. All higher-level structures (MPS, MPO, DMRG) build on
these primitives.

The contraction engine uses numpy's einsum with an optimized greedy path
finder that minimizes total FLOP count rather than peak memory, which is
the correct objective for simulation throughput.

References
----------
- Gray, J. "opt_einsum - A Python package for optimizing contraction order
  for einsum-like expressions." JOSS 3.26 (2018): 753.
- Pfeifer, R.N.C., et al. "Faster identification of optimal contraction
  sequences for tensor networks." Physical Review E 90.3 (2014): 033315.

Copyright (c) 2024-2026 QONTOS Inc. All rights reserved.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index label helpers
# ---------------------------------------------------------------------------

_EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _fresh_labels(n: int, exclude: Set[str] = frozenset()) -> List[str]:
    """Return *n* unique single-character einsum labels not in *exclude*."""
    labels: List[str] = []
    for ch in _EINSUM_CHARS:
        if ch not in exclude:
            labels.append(ch)
            if len(labels) == n:
                return labels
    raise ValueError(f"Ran out of einsum labels (requested {n}, excluded {len(exclude)})")


# ===================================================================
# Tensor
# ===================================================================

class Tensor:
    """
    Arbitrary-rank dense tensor backed by a numpy ndarray.

    Each axis is optionally labelled with a string *index name* so that
    contractions can be specified symbolically (match shared index names)
    rather than positionally.

    Parameters
    ----------
    data : array_like
        The tensor data.  Copied once on construction.
    indices : list[str], optional
        Human-readable names for each axis.  If omitted, auto-generated
        as ``["i0", "i1", ...]``.

    Examples
    --------
    >>> t = Tensor(np.eye(2), indices=["row", "col"])
    >>> t.rank
    2
    >>> t.shape
    (2, 2)
    """

    __slots__ = ("data", "indices")

    def __init__(
        self,
        data: np.ndarray,
        indices: Optional[List[str]] = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.complex128)
        if indices is None:
            indices = [f"i{k}" for k in range(self.data.ndim)]
        if len(indices) != self.data.ndim:
            raise ValueError(
                f"Number of index labels ({len(indices)}) does not match "
                f"tensor rank ({self.data.ndim})"
            )
        self.indices: List[str] = list(indices)

    # -- properties --------------------------------------------------------

    @property
    def rank(self) -> int:
        """Number of axes (order) of the tensor."""
        return self.data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def norm(self) -> float:
        """Frobenius norm."""
        return float(np.linalg.norm(self.data))

    # -- algebra -----------------------------------------------------------

    def conjugate(self) -> "Tensor":
        """Element-wise complex conjugate (indices unchanged)."""
        return Tensor(self.data.conj(), list(self.indices))

    def rename_index(self, old: str, new: str) -> "Tensor":
        """Return a copy with one index renamed."""
        new_indices = [new if idx == old else idx for idx in self.indices]
        return Tensor(self.data.copy(), new_indices)

    def reorder(self, new_order: List[str]) -> "Tensor":
        """Transpose to match the given index order."""
        perm = [self.indices.index(n) for n in new_order]
        return Tensor(np.transpose(self.data, perm), new_order)

    def scale(self, alpha: complex) -> "Tensor":
        """Return alpha * self."""
        return Tensor(self.data * alpha, list(self.indices))

    # -- contraction -------------------------------------------------------

    def contract_with(
        self,
        other: "Tensor",
        contract_indices: Optional[List[str]] = None,
    ) -> "Tensor":
        """
        Contract *self* with *other* over shared (or specified) indices.

        Uses ``np.einsum`` for the heavy lifting with ``optimize='greedy'``.

        Parameters
        ----------
        other : Tensor
        contract_indices : list[str], optional
            Indices to sum over.  Defaults to all indices shared by name.

        Returns
        -------
        Tensor
            Result with the un-contracted indices from both tensors.
        """
        if contract_indices is None:
            contract_indices = [i for i in self.indices if i in other.indices]

        used: Set[str] = set()
        label_map: Dict[str, str] = {}

        def _get_label(name: str) -> str:
            if name not in label_map:
                label_map[name] = _fresh_labels(1, used)[0]
                used.add(label_map[name])
            return label_map[name]

        lhs = "".join(_get_label(i) for i in self.indices)
        rhs = "".join(_get_label(i) for i in other.indices)

        out_indices: List[str] = []
        for i in self.indices:
            if i not in contract_indices:
                out_indices.append(i)
        for i in other.indices:
            if i not in contract_indices and i not in out_indices:
                out_indices.append(i)

        out = "".join(label_map[i] for i in out_indices)
        subscripts = f"{lhs},{rhs}->{out}"

        result = np.einsum(subscripts, self.data, other.data, optimize="greedy")
        return Tensor(result, out_indices)

    # -- decompositions ----------------------------------------------------

    def svd(
        self,
        left_indices: List[str],
        right_indices: Optional[List[str]] = None,
        max_rank: Optional[int] = None,
        cutoff: float = 0.0,
    ) -> Tuple["Tensor", np.ndarray, "Tensor"]:
        """
        Singular Value Decomposition across a bipartition of the indices.

        The tensor is reshaped into a matrix ``(left | right)`` and then
        decomposed as ``U @ diag(S) @ Vh``.

        Parameters
        ----------
        left_indices : list[str]
            Indices that form the "row" space.
        right_indices : list[str], optional
            Indices for the "column" space.  Inferred from the complement
            of *left_indices* if omitted.
        max_rank : int, optional
            Truncate to at most this many singular values.
        cutoff : float
            Discard singular values below this threshold.

        Returns
        -------
        U : Tensor
            Left unitary, indices = left_indices + [bond_label].
        S : ndarray
            1-D array of singular values.
        Vh : Tensor
            Right unitary, indices = [bond_label] + right_indices.
        """
        if right_indices is None:
            right_indices = [i for i in self.indices if i not in left_indices]

        all_idx = left_indices + right_indices
        reordered = self.reorder(all_idx)

        left_shape = tuple(reordered.data.shape[k] for k in range(len(left_indices)))
        right_shape = tuple(
            reordered.data.shape[k] for k in range(len(left_indices), reordered.rank)
        )
        mat = reordered.data.reshape(int(np.prod(left_shape)), int(np.prod(right_shape)))

        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # -- truncation ----------------------------------------------------
        if cutoff > 0:
            keep = np.sum(S > cutoff)
            U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]
        if max_rank is not None and len(S) > max_rank:
            U, S, Vh = U[:, :max_rank], S[:max_rank], Vh[:max_rank, :]
        if len(S) == 0:
            S = np.array([0.0])
            U = np.zeros((mat.shape[0], 1), dtype=np.complex128)
            Vh = np.zeros((1, mat.shape[1]), dtype=np.complex128)

        bond = f"_svd_{id(self) % 10000}"

        U_tensor = Tensor(
            U.reshape(*left_shape, len(S)),
            left_indices + [bond],
        )
        Vh_tensor = Tensor(
            Vh.reshape(len(S), *right_shape),
            [bond] + right_indices,
        )
        return U_tensor, S, Vh_tensor

    def qr(
        self,
        left_indices: List[str],
        right_indices: Optional[List[str]] = None,
    ) -> Tuple["Tensor", "Tensor"]:
        """
        QR decomposition across a bipartition, returning (Q, R).

        Parameters
        ----------
        left_indices, right_indices : list[str]
            As in :meth:`svd`.

        Returns
        -------
        Q : Tensor   (left_indices + [bond])
        R : Tensor   ([bond] + right_indices)
        """
        if right_indices is None:
            right_indices = [i for i in self.indices if i not in left_indices]

        all_idx = left_indices + right_indices
        reordered = self.reorder(all_idx)

        left_shape = tuple(reordered.data.shape[k] for k in range(len(left_indices)))
        right_shape = tuple(
            reordered.data.shape[k] for k in range(len(left_indices), reordered.rank)
        )
        rows = int(np.prod(left_shape))
        cols = int(np.prod(right_shape))

        mat = reordered.data.reshape(rows, cols)
        Q, R = np.linalg.qr(mat, mode="reduced")

        bond = f"_qr_{id(self) % 10000}"
        k = Q.shape[1]

        Q_tensor = Tensor(Q.reshape(*left_shape, k), left_indices + [bond])
        R_tensor = Tensor(R.reshape(k, *right_shape), [bond] + right_indices)
        return Q_tensor, R_tensor

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        shape_str = "x".join(str(s) for s in self.shape)
        return f"Tensor({shape_str}, indices={self.indices})"

    def __add__(self, other: "Tensor") -> "Tensor":
        if self.indices != other.indices:
            other = other.reorder(self.indices)
        return Tensor(self.data + other.data, list(self.indices))

    def __mul__(self, scalar: complex) -> "Tensor":
        return self.scale(scalar)

    def __rmul__(self, scalar: complex) -> "Tensor":
        return self.scale(scalar)


# ===================================================================
# Free-standing contraction helpers
# ===================================================================

def _contraction_cost(
    shape_a: Tuple[int, ...],
    indices_a: List[str],
    shape_b: Tuple[int, ...],
    indices_b: List[str],
) -> int:
    """
    Estimate the FLOP cost of contracting two tensors.

    FLOPs ~ product of all unique dimension sizes appearing in the
    contraction (each contracted index contributes a factor, as does
    each free index).
    """
    dim_map: Dict[str, int] = {}
    for idx, s in zip(indices_a, shape_a):
        dim_map[idx] = s
    for idx, s in zip(indices_b, shape_b):
        dim_map[idx] = max(dim_map.get(idx, 0), s)
    return int(np.prod(list(dim_map.values())))


def contract_pair(
    a: Tensor,
    b: Tensor,
    contract_indices: Optional[List[str]] = None,
) -> Tuple[Tensor, int]:
    """
    Contract two tensors and return ``(result, flop_cost)``.

    Parameters
    ----------
    a, b : Tensor
    contract_indices : list[str], optional

    Returns
    -------
    result : Tensor
    cost : int   (estimated FLOPs)
    """
    if contract_indices is None:
        contract_indices = [i for i in a.indices if i in b.indices]
    cost = _contraction_cost(a.shape, a.indices, b.shape, b.indices)
    result = a.contract_with(b, contract_indices)
    return result, cost


# ===================================================================
# TensorNetwork
# ===================================================================

@dataclass
class _TNEdge:
    """An internal edge (bond) in the tensor network."""
    index_name: str
    tensor_ids: List[int]
    dimension: int


class TensorNetwork:
    """
    A graph of :class:`Tensor` objects connected by named bonds.

    Tensors are added via :meth:`add_tensor` and the full network can be
    contracted with :meth:`contract`, which uses :func:`optimal_contraction_order`
    to choose the pairwise sequence that minimises total FLOP count.

    Parameters
    ----------
    name : str, optional
        Descriptive label for logging.

    Examples
    --------
    >>> tn = TensorNetwork("demo")
    >>> tn.add_tensor(Tensor(np.random.randn(2,3), ["a","b"]))
    >>> tn.add_tensor(Tensor(np.random.randn(3,4), ["b","c"]))
    >>> result = tn.contract()
    >>> result.shape
    (2, 4)
    """

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
        self._tensors: Dict[int, Tensor] = {}
        self._next_id: int = 0

    # -- mutation ----------------------------------------------------------

    def add_tensor(self, tensor: Tensor) -> int:
        """Add a tensor; return its integer id."""
        tid = self._next_id
        self._next_id += 1
        self._tensors[tid] = tensor
        return tid

    def remove_tensor(self, tid: int) -> Tensor:
        return self._tensors.pop(tid)

    # -- queries -----------------------------------------------------------

    @property
    def num_tensors(self) -> int:
        return len(self._tensors)

    @property
    def tensor_ids(self) -> List[int]:
        return list(self._tensors.keys())

    def get_tensor(self, tid: int) -> Tensor:
        return self._tensors[tid]

    def bonds(self) -> Dict[str, _TNEdge]:
        """Return a dict of internal edges (indices shared by >=2 tensors)."""
        idx_to_tids: Dict[str, List[int]] = {}
        idx_to_dim: Dict[str, int] = {}
        for tid, t in self._tensors.items():
            for idx, dim in zip(t.indices, t.shape):
                idx_to_tids.setdefault(idx, []).append(tid)
                idx_to_dim[idx] = dim
        return {
            name: _TNEdge(name, tids, idx_to_dim[name])
            for name, tids in idx_to_tids.items()
            if len(tids) >= 2
        }

    def bond_dimensions(self) -> Dict[str, int]:
        """Map of bond name -> dimension for all internal bonds."""
        return {name: e.dimension for name, e in self.bonds().items()}

    def open_indices(self) -> List[str]:
        """Indices that appear in exactly one tensor (external legs)."""
        count: Dict[str, int] = {}
        for t in self._tensors.values():
            for idx in t.indices:
                count[idx] = count.get(idx, 0) + 1
        return [idx for idx, c in count.items() if c == 1]

    def total_memory(self) -> int:
        """Total number of complex128 elements across all tensors."""
        return sum(t.size for t in self._tensors.values())

    # -- contraction -------------------------------------------------------

    def contract(self) -> Tensor:
        """
        Contract the entire network into a single tensor.

        Uses the greedy FLOP-optimal ordering from
        :func:`optimal_contraction_order`.
        """
        if self.num_tensors == 0:
            raise ValueError("Cannot contract an empty network")
        if self.num_tensors == 1:
            return list(self._tensors.values())[0]

        ids = list(self._tensors.keys())
        order = optimal_contraction_order(
            [self._tensors[i] for i in ids],
        )

        # Build a working copy
        working: Dict[int, Tensor] = {i: self._tensors[i] for i in ids}
        id_map = {k: k for k in ids}  # track merges

        total_cost = 0
        for i_pos, j_pos in order:
            tid_a = ids[i_pos]
            tid_b = ids[j_pos]
            a = working.pop(tid_a)
            b = working.pop(tid_b)
            result, cost = contract_pair(a, b)
            total_cost += cost
            new_id = self._next_id
            self._next_id += 1
            working[new_id] = result
            # Update ids list: replace i_pos with new_id, mark j_pos
            ids[i_pos] = new_id
            ids[j_pos] = -1  # sentinel

        logger.debug("TN '%s' contracted with total cost %d", self.name, total_cost)
        assert len(working) == 1
        return next(iter(working.values()))


# ===================================================================
# Optimal contraction order (greedy)
# ===================================================================

def _pair_cost(a: Tensor, b: Tensor) -> int:
    """FLOP cost of contracting tensors *a* and *b*."""
    return _contraction_cost(a.shape, a.indices, b.shape, b.indices)


def _pair_result_size(a: Tensor, b: Tensor) -> int:
    """Size (number of elements) of the result of contracting a with b."""
    shared = set(a.indices) & set(b.indices)
    result_indices = [i for i in a.indices if i not in shared] + \
                     [i for i in b.indices if i not in shared]
    dim_map = {}
    for idx, s in zip(a.indices, a.shape):
        dim_map[idx] = s
    for idx, s in zip(b.indices, b.shape):
        dim_map[idx] = s
    if not result_indices:
        return 1
    return int(np.prod([dim_map[i] for i in result_indices]))


def optimal_contraction_order(
    tensors: List[Tensor],
    strategy: str = "greedy_flops",
) -> List[Tuple[int, int]]:
    """
    Find a near-optimal pairwise contraction ordering for a list of tensors.

    This implements the *greedy* heuristic that at each step contracts the
    pair with the **lowest FLOP cost**, breaking ties by choosing the pair
    whose result has the smallest size.  This is O(n^3) in the number of
    tensors -- acceptable for networks up to ~1000 tensors.

    For MPS/MPO applications the network is a chain and the ordering is
    trivially sequential, so this function mainly matters for general
    tensor networks.

    Parameters
    ----------
    tensors : list[Tensor]
    strategy : str
        ``"greedy_flops"`` (default) -- minimise cumulative FLOP count.
        ``"greedy_size"`` -- minimise intermediate tensor size.

    Returns
    -------
    list of (int, int)
        Pairs of *position indices* into the (evolving) list to contract.
        Positions reference the original list; after each contraction the
        result replaces the first element and the second is invalidated.

    References
    ----------
    - Gray, J. "Hyper-optimized tensor network contraction."
      Quantum 5 (2021): 410. arXiv:2002.01935
    """
    n = len(tensors)
    if n <= 1:
        return []

    # We work on indices into a mutable list.
    active: List[int] = list(range(n))
    working: List[Tensor] = list(tensors)  # shallow copy of list
    order: List[Tuple[int, int]] = []

    while len(active) > 1:
        best_cost = float("inf")
        best_size = float("inf")
        best_pair: Tuple[int, int] = (active[0], active[1])

        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                a_idx, b_idx = active[ii], active[jj]
                a, b = working[a_idx], working[b_idx]

                # Only consider pairs that share at least one index
                shared = set(a.indices) & set(b.indices)
                if not shared and len(active) > 2:
                    continue

                cost = _pair_cost(a, b)
                size = _pair_result_size(a, b)

                if strategy == "greedy_flops":
                    better = (cost < best_cost) or (
                        cost == best_cost and size < best_size
                    )
                else:
                    better = (size < best_size) or (
                        size == best_size and cost < best_cost
                    )

                if better:
                    best_cost = cost
                    best_size = size
                    best_pair = (a_idx, b_idx)

        i_idx, j_idx = best_pair
        order.append((i_idx, j_idx))

        # Contract and replace
        result = working[i_idx].contract_with(working[j_idx])
        working[i_idx] = result
        active.remove(j_idx)

    return order


# ===================================================================
# Convenience factories
# ===================================================================

def random_tensor(
    shape: Tuple[int, ...],
    indices: Optional[List[str]] = None,
    real: bool = False,
) -> Tensor:
    """Generate a random tensor (Gaussian entries)."""
    if real:
        data = np.random.randn(*shape)
    else:
        data = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    data /= np.linalg.norm(data)
    return Tensor(data, indices)


def identity_tensor(d: int, index_in: str, index_out: str) -> Tensor:
    """Return a rank-2 identity tensor of dimension d."""
    return Tensor(np.eye(d, dtype=np.complex128), [index_in, index_out])


def delta_tensor(d: int, rank: int, indices: List[str]) -> Tensor:
    """Generalised Kronecker delta (nonzero only when all indices equal)."""
    data = np.zeros([d] * rank, dtype=np.complex128)
    for i in range(d):
        data[tuple([i] * rank)] = 1.0
    return Tensor(data, indices)


# ===================================================================
# Utility: einsum-based multi-tensor contraction
# ===================================================================

def multi_contract(tensors: List[Tensor], output_indices: List[str]) -> Tensor:
    """
    Contract a list of tensors in one ``np.einsum`` call.

    All shared index names are summed over; *output_indices* specifies the
    free legs of the result.

    Parameters
    ----------
    tensors : list[Tensor]
    output_indices : list[str]

    Returns
    -------
    Tensor
    """
    label_map: Dict[str, str] = {}
    used: Set[str] = set()

    def lbl(name: str) -> str:
        if name not in label_map:
            label_map[name] = _fresh_labels(1, used)[0]
            used.add(label_map[name])
        return label_map[name]

    subscript_parts = []
    for t in tensors:
        subscript_parts.append("".join(lbl(i) for i in t.indices))

    out_sub = "".join(label_map[i] for i in output_indices)
    subscripts = ",".join(subscript_parts) + "->" + out_sub

    arrays = [t.data for t in tensors]
    result = np.einsum(subscripts, *arrays, optimize="greedy")
    return Tensor(result, output_indices)
