# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-08

First public release: a small, self-contained SDK for building and running quantum
algorithms. The whole package depends only on NumPy.

### Added

- **qontos_sim**: `Circuit`, a friendly circuit builder (h, x, y, z, s, sdg, t, tdg,
  rx, ry, rz, p, cx, cz, swap, measure, measure_all) with method chaining and clear
  errors.
- **qontos_sim**: `simulate(circuit, shots, method)`, one call with two backends:
  an exact NumPy statevector simulator (`method="statevector"`, the default) and a
  matrix-product-state backend (`method="mps"`) for larger, low-entanglement circuits.
- **qontos_sim**: `Result` with `counts`, `probabilities()`, `most_common()`, and the
  exact `statevector` when available.
- **qontos_twin**: `ModularSimulator` digital twin for modular architecture studies
  (fidelity estimation, Bell-pair budgeting, transduction efficiency bands).
- **qontos_tensor**: pure-NumPy tensor-network engine (`Tensor`, `TensorNetwork`,
  `MatrixProductState`, `MatrixProductOperator`, `DMRG`, `TNSimulator`) that powers the
  MPS backend.
- Runnable examples: `quickstart.py` (Bell and GHZ), `chsh_bell_inequality.py`
  (reproduces the CHSH violation, S approaches 2.83), and `variational_sweep.py`.
- Test suite covering the builder, both backends and their agreement, known-physics
  reproductions (CHSH), the tensor engine, and the digital twin.
- CI: lint, tests on Python 3.10 to 3.12, and wheel build.

### Bit-ordering convention

In a result bitstring, position `i` is the measured value of qubit `i` (qubit 0 is the
leftmost character). Stated once and honoured everywhere.
