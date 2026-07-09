"""Core SDK tests: the circuit builder, the exact statevector simulator, and the
matrix-product-state backend, checked against known physics and against each other."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qontos_sim import Circuit, simulate
from qontos_sim import statevector as sv


# --------------------------------------------------------------------- builder
def test_circuit_builds_and_chains():
    c = Circuit(2).h(0).cx(0, 1).measure_all()
    assert c.num_qubits == 2
    assert c.depth == 2
    assert c.measured == [0, 1]
    assert [i.name for i in c.instructions] == ["h", "cx"]


def test_circuit_rejects_bad_qubits():
    with pytest.raises(ValueError):
        Circuit(2).h(5)
    with pytest.raises(ValueError):
        Circuit(2).cx(0, 0)  # same qubit twice
    with pytest.raises(ValueError):
        Circuit(0)  # needs at least one qubit


# ------------------------------------------------------------- statevector math
def test_bell_amplitudes_exact():
    c = Circuit(2).h(0).cx(0, 1)
    amp = sv.final_state(c)
    assert np.allclose(np.abs(amp) ** 2, [0.5, 0.0, 0.0, 0.5], atol=1e-12)


def test_x_gate_flips():
    amp = sv.final_state(Circuit(1).x(0))
    assert np.allclose(amp, [0, 1], atol=1e-12)


def test_rotation_probability():
    # ry(theta) on |0> gives P(1) = sin^2(theta/2)
    for theta in (0.3, 1.0, math.pi / 2, 2.4):
        p = sv.probabilities(Circuit(1).ry(theta, 0))
        assert abs(p[1] - math.sin(theta / 2) ** 2) < 1e-12


def test_ghz_only_all_zero_or_all_one():
    amp = sv.final_state(Circuit(3).h(0).cx(0, 1).cx(1, 2))
    probs = np.abs(amp) ** 2
    nonzero = {i for i, x in enumerate(probs) if x > 1e-9}
    assert nonzero == {0, 7}  # |000> and |111>


def test_measured_bitstring_ordering():
    # qubit 0 set to 1, qubit 1 left 0; measured "10" (qubit 0 leftmost)
    c = Circuit(2).x(0).measure_all()
    r = simulate(c, shots=100, seed=0)
    assert r.counts == {"10": 100}


# ------------------------------------------------------------------- sampling
def test_bell_counts_statevector():
    r = simulate(Circuit(2).h(0).cx(0, 1).measure_all(), shots=8000, seed=7)
    assert set(r.counts) == {"00", "11"}
    assert abs(r.counts["00"] - r.counts["11"]) < 400  # ~50/50


def test_seed_is_reproducible():
    c = Circuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
    assert simulate(c, shots=2000, seed=42).counts == simulate(c, shots=2000, seed=42).counts


def test_partial_measurement():
    # measure only qubit 0 of a Bell pair -> marginal is 50/50 over a 1-bit string
    c = Circuit(2).h(0).cx(0, 1).measure(0)
    r = simulate(c, shots=4000, seed=1)
    assert set(r.counts) == {"0", "1"}


# ----------------------------------------------------------------- mps backend
def test_mps_matches_statevector_on_bell():
    c = Circuit(2).h(0).cx(0, 1).measure_all()
    mps = simulate(c, shots=6000, method="mps")
    assert set(mps.counts) == {"00", "11"}  # same support as exact


def test_mps_ghz():
    c = Circuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3).measure_all()
    r = simulate(c, shots=4000, method="mps")
    assert set(r.counts) <= {"0000", "1111"}


def test_mps_rejects_unsupported_gate():
    c = Circuit(1).t(0).sdg(0).measure_all()  # sdg not in the mps gate set
    with pytest.raises(ValueError):
        simulate(c, shots=100, method="mps")


# --------------------------------------------------------------------- Result
def test_result_helpers():
    r = simulate(Circuit(2).h(0).cx(0, 1).measure_all(), shots=1000, seed=3)
    probs = r.probabilities()
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    top = r.most_common(1)[0]
    assert top[0] in {"00", "11"}


def test_simulate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        simulate(Circuit(1).h(0), shots=0)
    with pytest.raises(ValueError):
        simulate(Circuit(1).h(0), method="nonsense")


# ------------------------------------------------ known-physics: CHSH inequality
def _expectation_zz(theta_a: float, theta_b: float, seed: int) -> float:
    """<A B> for a Bell pair measured in rotated bases (rotate then measure Z)."""
    c = Circuit(2).h(0).cx(0, 1).ry(-2 * theta_a, 0).ry(-2 * theta_b, 1).measure_all()
    r = simulate(c, shots=20000, seed=seed)
    e = 0.0
    for bits, n in r.counts.items():
        parity = 1 if bits.count("1") % 2 == 0 else -1
        e += parity * n
    return e / r.shots


def test_chsh_violates_classical_bound():
    # optimal CHSH angles; quantum value approaches 2*sqrt(2) ~ 2.828, beating 2
    a, ap, b, bp = 0.0, math.pi / 4, math.pi / 8, -math.pi / 8
    s = (
        _expectation_zz(a, b, 1)
        + _expectation_zz(a, bp, 2)
        + _expectation_zz(ap, b, 3)
        - _expectation_zz(ap, bp, 4)
    )
    assert s > 2.4  # clearly violates the classical bound of 2
    assert s < 2 * math.sqrt(2) + 0.1  # and does not exceed Tsirelson's bound
