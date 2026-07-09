<div align="center">
  <a href="https://github.com/qontos">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qontos/.github/main/assets/qontos-logo-white.png">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/qontos/.github/main/assets/qontos-logo.png">
      <img src="https://raw.githubusercontent.com/qontos/.github/main/assets/qontos-logo.png" alt="QONTOS" width="260">
    </picture>
  </a>

  <h3>QONTOS Simulators</h3>
  <p><strong>Simulation, digital twin, and tensor-network modeling for the QONTOS platform.</strong></p>
  <p>Public validation and planning tools for the software stack today and the modular hardware roadmap ahead.</p>

  <p>
    <img src="https://img.shields.io/badge/Visibility-Public-0f766e?style=flat-square" alt="Visibility: Public">
    <img src="https://img.shields.io/badge/Track-Simulation-0b3b8f?style=flat-square" alt="Track: Simulation">
    <img src="https://img.shields.io/badge/Status-Pre--release-c2410c?style=flat-square" alt="Status: Pre-release">
    <a href="https://github.com/qontos/qontos-sim/actions"><img src="https://img.shields.io/github/actions/workflow/status/qontos/qontos-sim/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI"></a>
  </p>

  <p>
    <a href="#overview">Overview</a> &middot;
    <a href="#installation">Installation</a> &middot;
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="docs/index.md">Docs Hub</a> &middot;
    <a href="#simulators">Simulators</a> &middot;
    <a href="#digital-twin">Digital Twin</a> &middot;
    <a href="#tensor-network-engine">Tensor Engine</a> &middot;
    <a href="#related-packages">Related Packages</a>
  </p>
</div>

---

## Overview

QONTOS Simulators is a small, self-contained SDK for building and running quantum algorithms. The whole package depends only on NumPy: a friendly circuit builder, an exact statevector simulator, and a matrix-product-state (tensor-network) backend for larger, low-entanglement circuits. It also ships a modular-architecture digital twin for system-level planning.

Start with [docs/index.md](docs/index.md) for the lightweight docs hub.

It provides:

1. **`qontos_sim`** — the developer SDK: a `Circuit` builder and one `simulate` call, with an exact statevector backend and a tensor-network (MPS) backend.
2. **`qontos_twin`** — a modular-hardware digital twin for architecture and throughput studies.
3. **`qontos_tensor`** — a pure NumPy tensor-network engine (MPS, MPO, DMRG) that powers the MPS backend.

## Installation

Requires Python 3.10+ and NumPy. Nothing else.

### Pre-release (current)

Not yet on PyPI. Install from source or a pinned release tag:

```bash
pip install "qontos-sim @ git+https://github.com/qontos/qontos-sim.git@v0.1.0"
```

Once published to PyPI this becomes `pip install qontos-sim`.

## Quick Start

Build a Bell pair and run it in three lines:

```python
from qontos_sim import Circuit, simulate

c = Circuit(2)
c.h(0).cx(0, 1).measure_all()
result = simulate(c, shots=1000)
print(result.counts)          # {'00': ~500, '11': ~500}
```

A 3-qubit GHZ state on the tensor-network backend:

```python
from qontos_sim import Circuit, simulate

ghz = Circuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
print(simulate(ghz, shots=1000, method="mps").counts)   # {'000': ~500, '111': ~500}
```

Bitstring convention: position `i` is the measured value of qubit `i` (qubit 0 leftmost).

### Digital Twin

```python
from qontos_twin import ModularSimulator, SystemConfig

config = SystemConfig(
    num_modules=4,
    transduction_efficiency=0.15,
)
sim = ModularSimulator(config)
workload = sim.simulate_workload(circuit_depth=250)
print(f"Estimated fidelity: {workload.estimated_fidelity:.4f}")
print(f"Bell pairs required: {workload.bell_pairs_needed}")
```

### Tensor Network Simulation

```python
from qontos_tensor import GateInstruction, TNSimulator

# Simulate bounded-entanglement circuits with an MPS backend
sim = TNSimulator(n_qubits=2, chi_max=256)
result = sim.run(
    [
        GateInstruction(name="H", qubits=[0]),
        GateInstruction(name="CNOT", qubits=[0, 1]),
    ],
    n_shots=1024,
)
print(result.measurements[:5])
```

## Simulators

| How to call | Backend | Qubits | Use case |
| :--- | :--- | :--- | :--- |
| `simulate(c, method="statevector")` | Exact statevector (NumPy) | Up to ~25 | The default; exact, any circuit |
| `simulate(c, method="mps")` | Tensor network (MPS) | Larger, bounded entanglement | Big low-entanglement circuits |
| `ModularSimulator` (`qontos_twin`) | Digital twin | Unlimited (modeled) | Architecture and throughput studies |

## Digital Twin

The digital twin simulates workloads on modular architecture candidates. For a given system configuration, it estimates:

- Total gate count (intra-module and inter-module)
- Circuit fidelity (based on gate fidelity, transduction, and decoherence)
- Runtime in microseconds
- Bell pairs required for inter-module operations
- Effective circuit depth increase from serialization

### Transduction Scenario Bands

| Efficiency | Scenario | Interpretation |
| :--- | :--- | :--- |
| >= 20% | Stretch | Full modular planning |
| >= 10% | Aggressive | Meaningful multi-module operation |
| 1-10% | Base | Staged modular validation |
| < 1% | Research | Device and link R&D |

## Tensor Network Engine

Pure NumPy implementation — zero external tensor network dependencies.

- **MPS** (Matrix Product State) — Bond dimension up to 4096
- **MPO** (Matrix Product Operator) — Heisenberg, Ising, Hubbard, molecular Hamiltonians
- **DMRG** — Variational ground-state search for 100+ site systems
- **Circuit simulation** — Full circuit evolution via MPS

## Examples

Runnable scripts live in [`examples/`](examples):

- `quickstart.py` — Bell and GHZ states end to end.
- `chsh_bell_inequality.py` — reproduces the CHSH violation (S approaches 2.83, beating the classical bound of 2).
- `variational_sweep.py` — a one-qubit energy sweep, the kernel of a variational algorithm.

## Related repositories

More of the QONTOS open-source ecosystem (examples, benchmarks, research) is being prepared and will be linked here as each repository is published.

## License

[Apache License 2.0](LICENSE)

---

*Built by [Zhyra Quantum Research Institute (ZQRI)](https://zhyra.xyz) — Abu Dhabi, UAE*
