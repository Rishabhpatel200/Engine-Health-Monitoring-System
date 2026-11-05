# Digital Twin Architecture

## 1. Design principles

**Physics first, learning second.** The learned models are given a feature space in
which the operating point has already been removed by physics. A detector fed raw
telemetry spends its capacity learning that CHT rises with power; a detector fed
model-versus-measurement residuals spends it on health. This is the difference
between a system that works at one flight condition and one that works across the
envelope.

**Explainability is structural, not bolted on.** Both learned detectors decompose
into per-feature contributions, and fault isolation is a signature matrix a
maintainer can read and argue with. No stage produces a number without also
producing the evidence for it.

**Every stage answers a different question.**

| Stage | Question |
|---|---|
| threshold layer | has a certified limit been exceeded? |
| reference model | how does this engine compare with a new one, here and now? |
| UKF | what are the states, including the ones nobody instrumented? |
| PCA / autoencoder | does this engine resemble a healthy fleet engine? |
| CUSUM | has *this* engine changed since the start of the sortie? |
| signature matrix | which fault explains the symptom pattern? |
| trend filter | how long until it needs action? |

Fusing them at the end is what gives both sensitivity and a low false-alarm rate:
they fail in different ways.

**Nothing flows backwards.** `dt/twin.py` imports the plant model but never the
simulator, the fault injector or the ground-truth health state; `scripts/selftest.py`
asserts this by inspecting the module source. Detection numbers from
`scripts/evaluate.py` are therefore not circular.

---

## 2. Module map and interfaces

```
config.py       EngineSpec, Limits, SensorSpec        ← single source of truth
atmosphere.py   isa(h, ΔISA, RH) → AtmState           ← environment
engine_model.py PistonEngine.step(dt, Command, AtmState, tas) → Outputs
degradation.py  DegradationModel.advance(HealthParams, Outputs, dt)
faults.py       FaultInjector.apply(HealthParams, t) → HealthParams
sensors.py      SensorSuite.measure(Outputs, atm, t, dt_h, faults) → dict
canbus.py       BaseCanBus.publish(dict, t) / .recv() → [CanFrame]
                BaseCanBus.frames_to_signals([CanFrame]) → dict
twin.py         DigitalTwin.sync(signals, dt, atm, cmd, tas) → snapshot dict
```

The seam between the simulator and a real aircraft is exactly one function.
`MissionSimulator.step()` produces `signals` from a simulated plant; a fielded
installation produces the same dictionary from a CAN receive thread. `DigitalTwin`
cannot tell the difference, and nothing downstream of it changes.

---

## 3. The reference model

The single most important design decision. A second `PistonEngine` with pristine
`HealthParams` is **clamped** to the measured speed and manifold pressure each tick,
while its thermal states evolve freely under its own healthy heat balance.

Clamping matters: a freely-integrating reference would diverge from the real engine
within seconds and the comparison would be meaningless. Letting the thermal states
run free also matters: the difference between the reference's temperatures and the
measured ones is then attributable to degradation rather than to a different
operating point.

One consequence needs care. The healthy oil-pressure reference must be evaluated at
the **measured** oil temperature, not the reference's own — otherwise a hot oil
charge is indistinguishable from a worn pump. `PistonEngine.oil_pressure_at()` exists
for exactly this.

---

## 4. Observability

What the twin can conclude is bounded by what the ICD carries.

| Quantity | Observable from | If unavailable |
|---|---|---|
| breathing (VE) | fuel flow vs reference at same rpm/MAP | — |
| shaft power | governor blade angle + speed + density | power index → 1.0, power diagnostics off |
| cooling capacity | CHT rise over ambient vs reference | — |
| pump condition | oil pressure vs reference at measured oil temp | — |
| compressor efficiency | charge temperature rise | reported nominal below PR 1.18 |
| per-cylinder fuelling | EGT dispersion | — |
| misfire | 0.5-order vibration + EGT deficit | — |
| bearing condition | 1x vibration + oil pressure | — |
| blowby | inferred from breathing index | weaker evidence, flagged as inferred |

Compressor efficiency is derived from a temperature rise; below a useful pressure
ratio that rise is comparable to sensor noise, so the index is reported as nominal
rather than as a noisy number the trend filters would chase. Stating the limit is
more useful than producing a confident-looking figure.

---

## 5. Why an unscented filter

The plant contains hard saturations — wastegate authority, oil relief valve,
propeller pitch stops, the misfire threshold. These are not differentiable, so an
EKF's Jacobian is either wrong or undefined at exactly the operating points that
matter. The unscented transform needs no derivatives.

Two implementation details are load-bearing:

- **Sigma points are regenerated from the post-process-noise covariance.** Reusing
  the propagated set omits Q from the innovation covariance, which collapses the
  Kalman gain to zero whenever the model is locally deterministic. This was a real
  bug during development: the filter silently stopped tracking.
- **The shadow model's run state comes from the measurement.** Inferring it from the
  sigma-point state lets the model stall at zero speed, from which the torque balance
  cannot recover.

---

## 6. Sequential detection

CUSUM catches drift too slow to trip any instantaneous threshold. Making it work in
flight rather than on paper required three things, each found by measurement:

1. **Steady-state gating.** A CUSUM compares against a fixed reference; running it
   through a climb accumulates legitimate operating-point change until it saturates.
2. **1 Hz decimation.** The slack and decision interval are designed in samples. At
   20 Hz the detector reaches its decision boundary in 0.6 s.
3. **Per-airframe, per-regime reference mean with a fleet reference spread.** The mean
   must come from this engine (an older engine is a legitimate mean shift, not a
   fault) and from this regime (several indices are only approximately
   operating-point invariant). The spread must come from the fleet, because an
   airframe baseline learned at one operating point sets an absurdly tight scale.

Measured effect on the false-alarm rate over five unseen healthy engines:

| Configuration | False alarms |
|---|---|
| naive CUSUM on raw features | 100 % |
| + steady-state gating and 1 Hz decimation | 81 % |
| + fleet reference spread | 6.2 % |
| + per-regime reference mean | **0.7 %** |

---

## 7. Data rates and budget

| Path | Rate | Notes |
|---|---|---|
| CAN bus | 5 PGNs, 10–50 Hz, 2.6 % of 500 kbit/s | 21 signals, J1939 framing |
| twin synchronisation | 20 Hz | UKF with 17 sigma points |
| health indices | 20 Hz, held outside the governed envelope | |
| sequential detection | 1 Hz, steady state only | |
| trend filters / RUL | 1 per minute, median of the interval | |
| continuous FDR | 2 Hz, compact record | 760 B/sample compressed |
| summary FDR | 1/60 Hz, full snapshot | 1.8 kB/sample compressed |
| event FDR | 20 Hz ring buffer, 60 s | flushed on any warning |
| dashboard | ~2.5 Hz over SSE | |

**Storage, measured.** 5.6 MB per flight hour on the duration-scaled tiers, i.e.
**about 134 MB for a 24 h sortie**, plus roughly 3 MB for each triggered event. That
is the figure *after* splitting the record into tiers and compressing it; writing the
whole twin snapshot at 2 Hz, as the first implementation did, costs 0.9 GB per sortie
and is not carriable.

**Throughput, measured** on one core of an Apple M-series CPU with the trained
models loaded: **~700 pipeline steps/s, about 35× real time** at the 20 Hz
synchronisation rate. Enabling the recorder costs 3 %. An earlier figure of 90×
was measured with the ML layer *absent*, which is exactly the sort of benchmark that
flatters a system into an infeasible deployment plan; 35× is the number that should
inform Phase 3 sizing, and the margin still supports a modest ARM SBC.

---

## 8. Interface Control Document

29-bit extended identifiers, priority 3, source address 0x00, 8-byte payloads,
little-endian scaled unsigned integers. All-ones means "signal not available" and
decodes to `None`, which the UKF handles by skipping that channel in the update.

| CAN ID | PGN | Name | Hz | Signals |
|---|---|---|---|---|
| 0x0CF00400 | 0xF004 | EEC1_SpeedLoad | 50 | rpm, throttle, MAP, IAT, bus volts |
| 0x0CFEEE00 | 0xFEEE | EngineTemp1 | 10 | coolant, oil temp, CHT 1–2 |
| 0x0CFEEF00 | 0xFEEF | EngineTemp2 | 10 | CHT 3–4, oil pressure, ambient pressure |
| 0x0CFEF200 | 0xFEF2 | FuelEconomy | 10 | fuel flow, alternator current, OAT, vibration, prop load |
| 0x0CFD0E00 | 0xFD0E | ExhaustTemp1 | 20 | EGT 1–4 |

Adding a signal means editing `PGN_MAP` in `dt/canbus.py` and `default_sensor_suite`
in `dt/config.py`. Nothing else needs to change.

---

## 9. Extending to another engine

1. Edit `EngineSpec` and `Limits` in `dt/config.py` — displacement, cylinder count,
   rated point, boost schedule, thermal masses, limits.
2. Adjust the VE map shape and Chen–Flynn friction constants in `dt/engine_model.py`
   against test-rig data.
3. Re-run `scripts/selftest.py` section 5, which checks the model against the rated
   point, the critical altitude, and the idle point.
4. Re-run `scripts/train.py` to identify new healthy baselines.

Nothing in `health.py`, `diagnostics.py`, `rul.py` or `advisory.py` is engine-specific;
they consume indices, not engineering units.

For a naturally-aspirated or diesel engine the air-path and combustion sub-models
change, but the interfaces do not.
