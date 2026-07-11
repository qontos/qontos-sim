# Model card: the `qontos_twin` architecture estimator

This card documents what the `qontos_twin` module (public class `ModularSimulator`)
is and is not, and the domain in which its numbers are meaningful. It exists because
the term "digital twin" implies a model calibrated against and validated on measured
device data, and this component is not that yet. It is an ARCHITECTURE ESTIMATOR: a
planning sandbox.

## What it is

A fast, analytic estimator for planning modular quantum architectures. Given a system
configuration (module count, qubits per module, transduction efficiency, gate and link
assumptions) and a workload depth, it returns planning signals: an estimated inter-module
Bell-pair budget, a cross-module gate fraction, an aggregate fidelity indicator, a
runtime estimate, and a serialization-driven depth increase.

## What it is NOT

- It is not a measured-data-calibrated digital twin. It has not been validated against
  held-out measurements from real hardware.
- Its aggregate fidelity is a stable software-side planning signal, not a literal
  prediction of the fidelity a physical machine would achieve.
- It does not simulate quantum states. For state-level simulation use the `qontos_sim`
  statevector or `qontos_tensor` MPS backends.

## Method and assumptions

- Transduction efficiency is bucketed into scenario bands (stretch, aggressive, base,
  research) rather than modelled continuously.
- The cross-module gate fraction is a heuristic function of topology and workload, not a
  compiled placement.
- The aggregate fidelity is an exponential composition of per-operation proxy terms.
- Several quantities are deliberately lower bounds or hand-constructed proxies.

## Valid operating domain

- Use it for relative comparisons between architecture candidates and for order-of-
  magnitude planning (does this workload need tens or thousands of Bell pairs; is a
  design communication-bound or coherence-bound).
- Do not use it to certify that a specific device meets a specific fidelity target, or as
  the basis for asking a laboratory to change a device specification. That requires a
  calibrated model with measured inputs, confidence intervals, and validation against
  held-out observations.

## Path to a validated digital twin

The name "digital twin" is reserved for a future version with: a versioned physical
model, traceable and unit-carrying parameters, calibration from measured device data,
predictive validation against held-out observations, a quantified model-discrepancy term,
and this card updated with the measured operating domain.
