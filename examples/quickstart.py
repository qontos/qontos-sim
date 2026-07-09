"""Quickstart: build and run two entangled states in a few lines.

Run:  python examples/quickstart.py
"""

from qontos_sim import Circuit, simulate


def bell():
    c = Circuit(2)
    c.h(0).cx(0, 1).measure_all()
    result = simulate(c, shots=1000, seed=1)
    print("Bell pair:", result.counts)
    # only 00 and 11 appear: the qubits are perfectly correlated
    assert set(result.counts) == {"00", "11"}


def ghz(n=4):
    c = Circuit(n).h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    c.measure_all()
    # the MPS backend handles this cheaply even as n grows
    result = simulate(c, shots=1000, method="mps")
    print(f"{n}-qubit GHZ:", result.counts)
    assert set(result.counts) <= {"0" * n, "1" * n}


if __name__ == "__main__":
    bell()
    ghz()
    print("\nQuickstart OK: entanglement built and measured on both backends.")
