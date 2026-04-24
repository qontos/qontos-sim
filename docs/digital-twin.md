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
- converter setup overhead
- converter drift stabilization overhead
- converter bandwidth caps
- retry overhead from Bell-pair instability
- phase-lock reacquisition overhead
- detector dead-time overhead
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
- `transduction_setup_time_us`: per-operation converter setup latency before the link can run
- `transduction_drift_probability`: probability that conversion drift forces extra retry/stabilization pressure
- `transduction_stabilization_time_us`: recovery time paid when conversion drift is modeled
- `transduction_bandwidth_limit_hz`: optional per-lane Bell-pair supply cap from converter bandwidth
- `optical_coupling_efficiency`: how much of the converted optical signal survives into the link
- `heralding_success_probability`: how often the photonic detection path successfully declares a usable Bell event
- `detector_efficiency`: how much detector loss the photonic receive path pays
- `detector_dark_count_probability`: how much false-event pressure the receive path adds
- `detector_dead_time_us`: how much per-event detector recovery time is exposed to the interconnect
- `phase_lock_duty_cycle`: how much of the operating window stays phase-locked instead of falling back into reacquisition
- `phase_lock_reacquisition_time_us`: how long a modeled phase-lock slip takes to recover
- `phase_lock_reference_jitter_us`: how much reference timing jitter degrades the phase-lock margin
- `link_phase_stability`: how stable the microwave-to-optical link stays under operation

Those knobs feed directly into:

- transduction channel efficiency before dynamic phase/noise penalties
- effective transduction efficiency
- weakest channel-component margin versus the nominal target stack
- transduction setup and drift-stabilization runtime overhead
- uncapped versus bandwidth-capped Bell-pair supply rate
- converter bandwidth utilization and cap status
- detector false-positive penalty from dark counts
- detector dead-time overhead
- phase-lock slip probability and reference stability
- phase-lock reacquisition overhead
- retry-adjusted link fidelity
- dynamic link stability after phase-lock duty and added-noise penalties
- link margin versus the `TARGET` operating point
- the split between supply, transduction-link, phase-lock, detector, memory, and control bottlenecks

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
print(f"Channel eff.:    {result.transduction_channel_efficiency:.3f}")
print(f"Channel margin:  {result.channel_margin_to_target:.2f}x")
print(f"Tx setup:        {result.transduction_setup_overhead_us:.1f} us")
print(f"Tx drift:        {result.transduction_drift_probability:.3f}")
print(f"Tx stabilize:    {result.transduction_stabilization_overhead_us:.1f} us")
print(f"Tx bw cap:       {result.transduction_bandwidth_limit_hz:.1f} Hz")
print(f"Tx bw capped:    {result.transduction_bandwidth_capped}")
print(f"Tx bw util.:     {result.transduction_bandwidth_utilization:.2f}")
print(f"Link quality:    {result.link_quality:.3f}")
print(f"Link stability:  {result.dynamic_link_stability:.3f}")
print(f"Link margin:     {result.link_margin_to_target:.2f}x")
print(f"Link fidelity*:  {result.retry_adjusted_link_fidelity:.3f}")
print(f"Calibration:     {result.transduction_calibration_quality:.2f}")
print(f"Optical couple:  {result.optical_coupling_efficiency:.2f}")
print(f"Heralding:       {result.heralding_success_probability:.2f}")
print(f"Detector eff.:   {result.detector_efficiency:.2f}")
print(f"Detector dark:   {result.detector_dark_count_probability:.3f}")
print(f"Detector penalty:{result.detector_false_positive_penalty:.3f}")
print(f"Detector dead:   {result.detector_dead_time_overhead_us:.1f} us")
print(f"Phase-lock duty: {result.phase_lock_duty_cycle:.2f}")
print(f"Phase slip:      {result.phase_lock_slip_probability:.3f}")
print(f"Phase ref.:      {result.phase_lock_reference_stability:.3f}")
print(f"Phase reacq.:    {result.phase_lock_reacquisition_overhead_us:.1f} us")
print(f"Phase stability: {result.link_phase_stability:.2f}")
print(f"Weakest channel: {result.weakest_channel_component} ({result.weakest_channel_margin:.2f}x)")
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
