"""QONTOS Simulators: build and run quantum algorithms in a few lines.

Quick start
-----------
::

    from qontos_sim import Circuit, simulate

    c = Circuit(2)
    c.h(0).cx(0, 1).measure_all()      # a Bell pair
    result = simulate(c, shots=1000)
    print(result.counts)               # {'00': ~500, '11': ~500}

The whole package depends only on numpy and ships self-contained: a friendly
circuit builder, an exact statevector simulator, and a matrix-product-state
(tensor-network) backend for larger, low-entanglement circuits.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .circuit import Circuit, Instruction
from .run import simulate, Result

__all__ = [
    "__version__",
    "Circuit",
    "Instruction",
    "simulate",
    "Result",
]
