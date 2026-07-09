"""CHSH: reproduce a known result of quantum mechanics on the SDK.

A shared Bell pair, measured in cleverly chosen bases, violates the CHSH
inequality: a classical (local hidden variable) theory can never exceed S = 2,
yet quantum mechanics reaches S = 2 * sqrt(2) ~ 2.828 (Tsirelson's bound). This
script builds the circuits, runs them, and computes S, showing the SDK
reproducing real physics rather than just shuffling bitstrings.

Run:  python examples/chsh_bell_inequality.py
"""

import math

from qontos_sim import Circuit, simulate


def correlation(theta_a: float, theta_b: float, seed: int, shots: int = 40000) -> float:
    """Measure <A(theta_a) B(theta_b)> on a Bell pair.

    Measuring qubit q in a basis rotated by theta is the same as rotating the
    state by -2*theta about Y and then measuring in the computational basis.
    """
    c = Circuit(2).h(0).cx(0, 1)
    c.ry(-2 * theta_a, 0).ry(-2 * theta_b, 1).measure_all()
    counts = simulate(c, shots=shots, seed=seed).counts
    e = 0
    for bits, n in counts.items():
        e += (1 if bits.count("1") % 2 == 0 else -1) * n  # +1 if outcomes agree
    return e / shots


def main():
    # the standard optimal CHSH angles
    a, a_prime = 0.0, math.pi / 4
    b, b_prime = math.pi / 8, -math.pi / 8

    e_ab = correlation(a, b, seed=1)
    e_abp = correlation(a, b_prime, seed=2)
    e_apb = correlation(a_prime, b, seed=3)
    e_apbp = correlation(a_prime, b_prime, seed=4)
    s = e_ab + e_abp + e_apb - e_apbp

    print("Correlations:")
    print(f"  E(a,  b)   = {e_ab:+.3f}")
    print(f"  E(a,  b')  = {e_abp:+.3f}")
    print(f"  E(a', b)   = {e_apb:+.3f}")
    print(f"  E(a', b')  = {e_apbp:+.3f}")
    print(f"\nCHSH  S = {s:.3f}")
    print("Classical bound:  2.000")
    print(f"Tsirelson bound:  {2 * math.sqrt(2):.3f}")
    print("Violated!" if s > 2.0 else "No violation.")


if __name__ == "__main__":
    main()
