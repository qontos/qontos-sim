# QONTOS Digital Twin

## Overview

The QONTOS Digital Twin models the behaviour of a modular quantum computer
architecture at the system level. It answers questions such as:

- How does fidelity degrade as we add more modules?
- What transduction efficiency do we need for a given algorithm?
- How many Bell pairs does a workload consume?
- What is the effective circuit depth after inter-module serialization?

The digital twin does **not** simulate quantum states. Instead it estimates
aggregate metrics (fidelity, runtime, resource counts) from the system
configuration and workload parameters.

It now also exposes a hybrid bottleneck decomposition so teams can see whether
the seam is currently limited by:

- base transduction-link runtime
- retry overhead from Bell-pair instability
- memory wait accumulation
- control jitter accumulation
- decoherence pressure
- raw entanglement-supply pressure

The seam model is now calibrated around the `TARGET` operating point instead of
dividing inter-module runtime by raw transduction efficiency. That keeps the
simulation anchored to the nominal system design while still making lower link
quality and higher retry pressure show up as slower Bell-pair supply, larger
latency, and lower throughput.

It now also exposes explicit entanglement-fabric controls:

- `entanglement_parallel_links`: parallel Bell-pair supply lanes
- `entanglement_buffer_pairs`: pre-generated Bell-pair inventory

That lets us ask software-side architecture questions like:

- how many parallel photonic lanes are needed before supply stops queueing
- how much Bell-pair buffering changes modular latency
- whether the seam is protocol-limited or entanglement-supply-limited

The transduction side now also has explicit quality controls:

- `transduction_calibration_quality`: how well the transducer is tuned to its nominal target
- `link_phase_stability`: how stable the microwave-to-optical link stays under operation

Those knobs feed directly into:

- effective transduction efficiency
- retry-adjusted link fidelity
- link margin versus the `TARGET` operating point
- the split between a supply bottleneck and a transduction-link bottleneck

## Key Concepts

### Modular Architecture

QONTOS uses a modular design where each module contains a fixed number of
qubits. Modules communicate via microwave-to-optical transduction to
distribute entanglement (Bell pairs) across inter-module links.

### Transduction Efficiency

The single most important parameter. It determines how quickly Bell pairs
can be generated and therefore how much inter-module communication costs.

## Scenario Bands

| Band     | Efficiency | Bell-pair Rate | Throughput   | Capability              |
|----------|-----------|----------------|--------------|-------------------------|
| STRETCH  | >= 25%    | ~5 kHz         | ~500 ops/sec | Full algorithm library   |
| TARGET   | >= 15%    | ~3 kHz         | ~300 ops/sec | Most algorithms viable   |
| FALLBACK | >= 10%    | ~2 kHz         | ~200 ops/sec | Sparse communication     |
| MINIMUM  | >= 5%     | ~1 kHz         | ~100 ops/sec | Single-module operation  |

Below 5% the system is classified as BELOW_MINIMUM and considered
non-functional for multi-module workloads.

## Hardware Planning

Use the digital twin to answer planning questions:

1. **Module count trade-off** -- More modules means more qubits but also
   more inter-module gates and lower fidelity.
2. **Transduction target** -- Determine the minimum efficiency needed for a
   target workload to remain viable (fidelity > threshold).
3. **Resource budgeting** -- Estimate Bell pair consumption and plan
   entanglement distribution hardware accordingly.

## Usage

```python
from qontos_twin import SystemConfig, ModuleConfig, simulate_workload

# 4-module system at target transduction efficiency
config = SystemConfig(
    num_modules=4,
    module=ModuleConfig(qubits_per_module=50),
    transduction_efficiency=0.15,
)

result = simulate_workload(config, circuit_depth=100)

print(f"Total qubits:    {result.total_qubits}")
print(f"Fidelity:        {result.estimated_fidelity:.6f}")
print(f"Runtime (us):    {result.estimated_runtime_us:.1f}")
print(f"Bell pairs:      {result.bell_pairs_needed}")
print(f"Effective depth: {result.effective_circuit_depth}")
print(f"Band:            {result.degradation_band}")
print(f"Link quality:    {result.link_quality:.3f}")
print(f"Link margin:     {result.link_margin_to_target:.2f}x")
print(f"Link fidelity*:  {result.retry_adjusted_link_fidelity:.3f}")
print(f"Calibration:     {result.transduction_calibration_quality:.2f}")
print(f"Phase stability: {result.link_phase_stability:.2f}")
print(f"Bell attempts:   {result.expected_attempts_per_bell_pair:.2f}")
print(f"Parallel links:  {result.entanglement_parallel_links}")
print(f"Buffered pairs:  {result.buffered_bell_pairs_used}")
print(f"Supply time:     {result.entanglement_supply_time_us:.1f} us")
print(f"Supply util.:    {result.entanglement_supply_utilization:.2f}")
print(f"Bell-pair rate:  {result.effective_bell_pair_rate_hz:.1f} Hz")
print(f"Throughput:      {result.throughput_ops_per_sec:.1f} ops/sec")
print(f"Bottleneck:      {result.dominant_bottleneck}")
print(f"Runtime split:   {result.runtime_breakdown_us}")
```

### Scaling Analysis

```python
from qontos_twin import SystemConfig, simulate_workload

for n_modules in [1, 2, 4, 8, 16]:
    cfg = SystemConfig(num_modules=n_modules, transduction_efficiency=0.15)
    r = simulate_workload(cfg, circuit_depth=50)
    print(f"{n_modules:>3} modules | {r.total_qubits:>5} qubits | "
          f"fidelity={r.estimated_fidelity:.6f} | band={r.degradation_band}")
```

### Calibrated Simulation

When real calibration data is available, pass measured gate fidelities and
coherence times via `ModuleConfig`:

```python
from qontos_twin import ModuleConfig, simulate_workload_calibrated

calibrated = ModuleConfig(
    gate_fidelity_1q=0.9985,
    gate_fidelity_2q=0.995,
    t1_us=450,
    t2_us=420,
)

result = simulate_workload_calibrated(
    calibrated_module=calibrated,
    num_modules=4,
    transduction_efficiency=0.15,
    circuit_depth=100,
)
```
