# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-23

### Added

- **qontos_sim**: `LocalSimulatorExecutor` for noiseless Qiskit Aer simulation.
- **qontos_sim**: `NoisySimulatorExecutor` with configurable depolarizing, thermal
  relaxation, and readout noise models matching whitepaper specifications.
- **qontos_sim**: `aer_result_to_partition_result` normaliser for converting raw
  Aer counts into the canonical `PartitionResult` format.
- **qontos_sim**: `ValidationResult` dataclass for pre-flight circuit validation.
- **qontos_twin**: `ModularSimulator` digital twin for modular quantum architecture
  studies including fidelity estimation, Bell pair budgeting, and transduction
  efficiency band classification.
- **qontos_twin**: `simulate_workload` and `simulate_workload_calibrated` entry
  points for system-level workload analysis.
- **qontos_tensor**: Pure-NumPy tensor network engine with `Tensor`,
  `TensorNetwork`, `MatrixProductState`, `MatrixProductOperator`, and DMRG.
- **qontos_tensor**: `TNSimulator` circuit simulator supporting 1000+ qubit
  MPS-based simulation with configurable bond dimension truncation.
- **qontos_tensor**: Noise channels (depolarizing, amplitude damping, dephasing)
  via Kraus operator quantum trajectories.
- **qontos_tensor**: `ScalabilityDemo` showcasing 1000-qubit GHZ preparation and
  random circuit simulation.
- Comprehensive test suites: `test_local_simulator`, `test_noisy_simulator`,
  `test_digital_twin`, `test_tensor_network`, `test_package`.
- Digital twin documentation (`docs/digital-twin.md`).
- CI pipeline with lint, test (Python 3.10--3.12), and wheel build verification.
- Optional dependency groups: `[sim]`, `[twin]`, `[tensor]`, `[all]`, `[dev]`.
