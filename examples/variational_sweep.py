"""A one-qubit variational energy sweep: the kernel of a VQE-style algorithm.

We prepare the state ry(theta)|0>, measure the energy of the Hamiltonian H = Z,
whose expectation is <Z> = cos(theta), and sweep theta to find the minimum at
theta = pi (energy -1). This is the smallest honest example of "build a
parameterised circuit, run it, read out an expectation, optimise."

Run:  python examples/variational_sweep.py
"""

import math

from qontos_sim import Circuit, simulate


def energy_z(theta: float, shots: int = 20000, seed: int = 0) -> float:
    """<Z> for ry(theta)|0>, estimated from measurement counts."""
    c = Circuit(1).ry(theta, 0).measure_all()
    counts = simulate(c, shots=shots, seed=seed).counts
    p0 = counts.get("0", 0) / shots
    p1 = counts.get("1", 0) / shots
    return p0 - p1  # <Z> = P(0) - P(1)


def main():
    thetas = [i * math.pi / 12 for i in range(13)]  # 0 .. pi
    print("  theta      <Z> (measured)   cos(theta) (exact)")
    best_theta, best_e = None, 1.0
    for t in thetas:
        e = energy_z(t, seed=int(t * 100))
        print(f"  {t:5.3f}      {e:+.3f}            {math.cos(t):+.3f}")
        if e < best_e:
            best_theta, best_e = t, e
    print(
        f"\nMinimum energy {best_e:+.3f} near theta = {best_theta:.3f} "
        f"(exact minimum: -1 at theta = pi = {math.pi:.3f})"
    )


if __name__ == "__main__":
    main()
