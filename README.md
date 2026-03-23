<div align="center">

# QONTOS Simulators

**Quantum simulators and modular digital twin for the QONTOS platform**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![CI](https://img.shields.io/github/actions/workflow/status/qontos/qontos-sim/ci.yml?branch=main&label=CI&logo=github)](https://github.com/qontos/qontos-sim/actions)

[Installation](#installation) &middot;
[Quick Start](#quick-start) &middot;
[Simulators](#simulators) &middot;
[Digital Twin](#digital-twin) &middot;
[Tensor Network Engine](#tensor-network-engine) &middot;
[Related Packages](#related-packages)

</div>

---

## What is this?

This repository provides three simulation backends for the QONTOS quantum orchestration platform:

1. **`qontos_sim`** — Qiskit Aer-based simulators (noiseless and noisy)
2. **`qontos_twin`** — Modular hardware digital twin for architecture studies
3. **`qontos_tensor`** — Pure NumPy tensor network engine (MPS, MPO, DMRG)

## Installation

```bash
pip install qontos-sim
```

For the full tensor network engine:

```bash
pip install "qontos-sim[tensor]"
```

Requires Python 3.10+.

## Quick Start

### Local Simulator

```python
from qontos_sim import LocalSimulatorExecutor

executor = LocalSimulatorExecutor()
result = executor.submit(circuit_ir, shots=8192)
print(result.counts)
```

### Noisy Simulation

```python
from qontos_sim import NoisySimulatorExecutor

executor = NoisySimulatorExecutor(
    depolarizing_rate=0.001,
    readout_error=0.01,
)
result = executor.submit(circuit_ir, shots=8192)
```

### Digital Twin

```python
from qontos_twin import ModularSimulator, SystemConfig

config = SystemConfig(
    module_count=4,
    qubits_per_module=2000,
    transduction_efficiency=0.15,
)
sim = ModularSimulator(config)
workload = sim.simulate_workload(circuit_ir)
print(f"Estimated fidelity: {workload.fidelity:.4f}")
print(f"Bell pairs required: {workload.bell_pairs}")
```

### Tensor Network Simulation

```python
from qontos_tensor import MatrixProductState, TNSimulator

# Simulate 100+ qubit circuits with bounded entanglement
sim = TNSimulator(max_bond_dimension=256)
result = sim.run(circuit_ir)
```

## Simulators

| Simulator | Backend | Qubits | Speed | Use Case |
|---|---|---|---|---|
| `LocalSimulatorExecutor` | Qiskit Aer (statevector) | Up to ~30 | Fast | Pipeline validation, unit tests |
| `NoisySimulatorExecutor` | Qiskit Aer (depolarizing) | Up to ~30 | Fast | Noise-aware testing |
| `ModularSimulator` | Digital twin | Unlimited (modeled) | Instant | Architecture studies, scenario planning |
| `TNSimulator` | Tensor network (MPS) | 1000+ | Varies | Large circuits, bounded entanglement |

## Digital Twin

The digital twin simulates workloads on modular architecture candidates. For a given system configuration, it estimates:

- Total gate count (intra-module and inter-module)
- Circuit fidelity (based on gate fidelity, transduction, and decoherence)
- Runtime in microseconds
- Bell pairs required for inter-module operations
- Effective circuit depth increase from serialization

### Transduction Scenario Bands

| Efficiency | Scenario | Interpretation |
|---|---|---|
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

## Related Packages

| Package | Repository | Description |
|---|---|---|
| `qontos` | [qontos/qontos](https://github.com/qontos/qontos) | Core orchestration SDK |
| `qontos-bench` | [qontos/qontos-benchmarks](https://github.com/qontos/qontos-benchmarks) | Benchmark framework |

## License

[Apache License 2.0](LICENSE)

---

*Built by [Zhyra Quantum Research Institute (ZQRI)](https://zhyra.xyz) — Abu Dhabi, UAE*
