<div align="center">

<img src="assets/qontos-logo.png" alt="QONTOS" width="400">

### Simulators

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

QONTOS Simulators provides the simulation and digital-twin layer for the QONTOS platform. It includes local simulators, noisy simulation, modular architecture models, and tensor-network tools used for validation, planning, and large-scale system studies. This repository supports both present-day software workflows and future native QONTOS hardware design work.

It provides three simulation backends:

1. **`qontos_sim`** — Qiskit Aer-based simulators (noiseless and noisy)
2. **`qontos_twin`** — Modular hardware digital twin for architecture studies
3. **`qontos_tensor`** — Pure NumPy tensor network engine (MPS, MPO, DMRG)

## Installation

```bash
pip install "git+https://github.com/qontos/qontos.git@main"
pip install "git+https://github.com/qontos/qontos-sim.git@main"
```

For local and noisy Aer execution:

```bash
pip install qiskit qiskit-aer
```

Requires Python 3.10+.

The simulator package is designed to work alongside the flagship [`qontos`](https://github.com/qontos/qontos) SDK because it consumes the public `CircuitIR` and `PartitionResult` schemas from that repo.

## Quick Start

### Local Simulator

```python
from qontos.circuit import CircuitNormalizer
from qontos_sim import LocalSimulatorExecutor

normalizer = CircuitNormalizer()
circuit_ir = normalizer.normalize(input_type="openqasm", source=qasm_source)
executor = LocalSimulatorExecutor()
result = executor.submit(circuit_ir, shots=8192)
print(result.counts)
```

### Noisy Simulation

```python
from qontos.circuit import CircuitNormalizer
from qontos_sim import NoisySimulatorExecutor

normalizer = CircuitNormalizer()
circuit_ir = normalizer.normalize(input_type="openqasm", source=qasm_source)
executor = NoisySimulatorExecutor()
result = executor.submit(circuit_ir, shots=8192)
```

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
