"""A small, dependency-free circuit model for building quantum algorithms.

The design goal is a friendly, readable builder that a developer can pick up in
one minute::

    from qontos_sim import Circuit
    c = Circuit(2)
    c.h(0).cx(0, 1).measure_all()

Bit-ordering convention: in a result bitstring, position i holds the measured
value of qubit i (qubit 0 is the leftmost character). This is stated once here
and honoured everywhere so results are never ambiguous.

Nothing here imports numpy or qiskit; a Circuit is a pure description of gates.
The simulators consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

# Gate names the builder emits. The simulators map these to matrices. Keeping the
# set explicit means an unknown gate fails loudly at build time, not mid-simulation.
ONE_QUBIT = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}
ONE_QUBIT_PARAM = {"rx", "ry", "rz", "p"}
TWO_QUBIT = {"cx", "cz", "swap"}


@dataclass
class Instruction:
    """One gate (or measurement) in the circuit."""

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()

    def __repr__(self) -> str:
        p = f", params={list(self.params)}" if self.params else ""
        return f"{self.name}(qubits={list(self.qubits)}{p})"


@dataclass
class Circuit:
    """A quantum circuit over ``num_qubits`` qubits.

    Gate methods return ``self`` so calls chain. Measurement is explicit:
    ``measure_all`` (or ``measure`` per qubit) marks which qubits are read out.
    """

    num_qubits: int
    instructions: List[Instruction] = field(default_factory=list)
    measured: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError("a circuit needs at least one qubit")

    # -- internal helpers ---------------------------------------------------
    def _check(self, *qubits: int) -> None:
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"qubit {q} is out of range for a {self.num_qubits}-qubit circuit")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"a gate cannot act on the same qubit twice: {qubits}")

    def _add(self, name: str, qubits: Tuple[int, ...], params: Tuple[float, ...] = ()) -> "Circuit":
        self._check(*qubits)
        self.instructions.append(Instruction(name, qubits, params))
        return self

    # -- single-qubit gates -------------------------------------------------
    def h(self, q: int) -> "Circuit":
        return self._add("h", (q,))

    def x(self, q: int) -> "Circuit":
        return self._add("x", (q,))

    def y(self, q: int) -> "Circuit":
        return self._add("y", (q,))

    def z(self, q: int) -> "Circuit":
        return self._add("z", (q,))

    def s(self, q: int) -> "Circuit":
        return self._add("s", (q,))

    def sdg(self, q: int) -> "Circuit":
        return self._add("sdg", (q,))

    def t(self, q: int) -> "Circuit":
        return self._add("t", (q,))

    def tdg(self, q: int) -> "Circuit":
        return self._add("tdg", (q,))

    # -- parameterised single-qubit gates -----------------------------------
    def rx(self, theta: float, q: int) -> "Circuit":
        return self._add("rx", (q,), (float(theta),))

    def ry(self, theta: float, q: int) -> "Circuit":
        return self._add("ry", (q,), (float(theta),))

    def rz(self, theta: float, q: int) -> "Circuit":
        return self._add("rz", (q,), (float(theta),))

    def p(self, lam: float, q: int) -> "Circuit":
        return self._add("p", (q,), (float(lam),))

    # -- two-qubit gates ----------------------------------------------------
    def cx(self, control: int, target: int) -> "Circuit":
        return self._add("cx", (control, target))

    def cz(self, control: int, target: int) -> "Circuit":
        return self._add("cz", (control, target))

    def swap(self, a: int, b: int) -> "Circuit":
        return self._add("swap", (a, b))

    # aliases developers coming from other SDKs expect
    cnot = cx

    # -- measurement --------------------------------------------------------
    def measure(self, q: int) -> "Circuit":
        self._check(q)
        if q not in self.measured:
            self.measured.append(q)
        return self

    def measure_all(self) -> "Circuit":
        self.measured = list(range(self.num_qubits))
        return self

    # -- introspection ------------------------------------------------------
    @property
    def depth(self) -> int:
        """Number of gate instructions (measurements excluded)."""
        return len(self.instructions)

    def __repr__(self) -> str:
        head = (
            f"Circuit(num_qubits={self.num_qubits}, gates={self.depth}, measured={self.measured})"
        )
        body = "\n".join("  " + repr(i) for i in self.instructions)
        return head + ("\n" + body if body else "")
