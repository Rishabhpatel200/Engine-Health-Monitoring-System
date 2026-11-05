# Deployment Roadmap

From the present software demonstrator to a fielded engine health management system
for a MALE UAV fleet. Each phase states what is being proven, what has to be built,
and what would make it fail.

---

## Phase 0 — where this demonstrator stands

**Proven.** A complete, self-contained processing chain: physics model validated
against the published performance envelope of the engine class, CAN acquisition with
a real ICD, state estimation, learned baselines, fault isolation, RUL, advisories,
dashboard, flight data recording and replay. Measured false-alarm and detection
performance on simulated data. 34 automated verification checks.

**Not proven.** Everything that depends on a real engine. Every number in this
repository comes from a model, and models are wrong in ways that only hardware
reveals. The rest of this roadmap is about replacing modelled evidence with measured
evidence in the order that retires the most risk per unit of effort.

---

## Phase 1 — Test-rig correlation (3–4 months)

**Objective.** Replace the modelled engine with a real one and find out how wrong the
model is.

- Instrument a rig engine to the ICD in `docs/architecture.md`: per-cylinder CHT and
  EGT, MAP, IAT, oil pressure and temperature, coolant temperature, thermostat or
  shutter position, fuel flow, tri-axial vibration with a crank-synchronous tacho,
  knock sensor, bus voltage and alternator current, propeller governor blade angle.
- Run the `envelope_sweep` profile on the rig at several ambient conditions. Feed the
  recorded data through the twin in replay.
- Correlate and re-fit: the volumetric-efficiency map, the Chen–Flynn friction
  constants, the thermal UA values, the knock margin, the EGT probe location factor.
  These are the parameters most likely to be materially wrong.
- Re-identify healthy baselines from **rig data**, not simulation.

**Exit criteria.** Reference-model prediction error within ±3 % on fuel flow, ±5 K on
CHT and ±20 K on EGT across the envelope; UKF innovations zero-mean and within ±1.5σ
at stabilised points.

**Principal risk.** The thermal sub-model is the most uncertain part and the one most
of the diagnostics rest on. If rig CHT behaviour cannot be matched within ±5 K, the
cooling and combustion indices lose most of their sensitivity and Phase 2 is not
worth starting until it is fixed.

---

## Phase 2 — Seeded-fault campaign (4–6 months)

**Objective.** Replace the signature matrix's assumed symptom patterns with measured
ones.

Every entry in `SIGNATURES` is currently an engineering hypothesis derived from the
model. Each needs a rig run: shimmed injectors at graded flow losses, resistored or
gapped plugs, partially blocked radiator cores, restricted oil pickups, worn bearing
shells, off-spec fuel, deliberately mis-calibrated sensors.

- Record each at several power settings and ambient conditions.
- Re-derive the symptom thresholds and signature weights from the measurements.
- Re-run `scripts/evaluate.py` against rig data to get *real* detection, false-alarm,
  lead-time and isolation numbers.

**Exit criteria.** Detection above 90 % and false alarms below 1 % on rig data, with
isolation naming the seeded fault first in at least 80 % of detections.

**Principal risk.** Some faults will prove to have overlapping signatures that the
model separated cleanly — most likely injector clogging versus ring wear, and bearing
wear versus mount degradation. The honest response is to merge those hypotheses into
a single advisory with a wider inspection scope rather than to claim a discrimination
the data does not support.

---

## Phase 3 — Edge deployment (3–4 months, parallel with Phase 2)

**Objective.** Run the twin where the engine is.

The core is standard-library Python precisely so this is a porting exercise rather
than a rewrite. Measured throughput today is ~1800 pipeline steps per second on one
desktop core — roughly 90× the 20 Hz synchronisation rate — so the margin for a
modest ARM SBC exists but must be verified, not assumed.

- Target a DO-178C-partitionable Linux SBC with a CAN interface. `SocketCanBus` binds
  `AF_CAN` directly and needs no changes.
- Split the workload: the 20 Hz synchronisation, threshold layer and event recording
  run onboard; the trend filters, RUL and advisory generation can run onboard or at
  the GCS, since they operate at one update per minute.
- Profile on target hardware. If the UKF is too expensive, the honest reduction is to
  drop the two fast states (speed and manifold pressure are directly and accurately
  measured) and filter only the six thermal states, cutting sigma points from 17 to
  13 and removing the fastest dynamics from the propagation.
- Bound memory: the event ring buffer and the SQLite recorder are the only unbounded
  structures.

**Exit criteria.** Sustained 20 Hz with the worst-case CAN load, under 50 % CPU and a
bounded memory footprint, through a full 24 h endurance profile.

---

## Phase 4 — Secure telemetry and GCS integration (3 months)

- Replace the plain HTTP/SSE service with a mutually authenticated TLS channel;
  per-airframe certificates; the dashboard served from the GCS, never from the
  aircraft.
- Sign flight data recorder sessions so post-flight analysis is admissible in an
  investigation: hash-chain the telemetry rows and sign each session on close.
- Integrate with the existing GCS: the twin publishes a snapshot dictionary and
  nothing else, so integration is a matter of transporting that structure.
- Rate-limit and validate everything arriving from the bus. A malformed or hostile
  CAN frame must not be able to influence a maintenance decision; the J1939
  "not available" convention already gives a safe path for out-of-range values, and
  every decoded signal must be range-checked before it reaches the estimator.

**Deliberately out of scope.** The twin advises; it does not command. No path from
the analytics back to the FADEC. Any future authority to derate the engine
automatically would need its own safety case and is a different programme.

---

## Phase 5 — Fleet operation and federated learning (6+ months)

- Depot-level health records: per-tail health index history, RUL trajectories and
  realised maintenance actions.
- Close the loop on prediction quality. The single most valuable metric is *realised*
  RUL error: when a component is finally removed, how far out was the estimate? A
  system that cannot report this will never earn a maintainer's trust.
- Federated baseline updates: each airframe computes baseline statistics locally and
  shares only the sufficient statistics — means, covariances, sample counts — never
  raw telemetry. The PCA monitor needs only a covariance matrix to be refitted, so
  this is a natural fit and keeps mission data on the aircraft.
- Re-identify fleet baselines on a fixed cadence, with an explicit review gate. An
  automatically updated baseline will otherwise slowly absorb a fleet-wide
  degradation and stop reporting it.

---

## Certification posture

This is an advisory system. It does not have flight-control authority, so the
certification burden is proportionate, but three things need to be true:

1. **The threshold layer is independent.** `ExceedanceMonitor` deliberately shares no
   state with the learned analytics and can be certified on its own. If the whole
   predictive stack were removed, the conventional protection would be unchanged.
2. **Every advisory is traceable.** Fault isolation is a signature matrix, not a
   black box; each advisory carries the symptoms that produced it and their
   membership values. A finding can be audited, and disputed, after the fact.
3. **Uncertainty is reported, not hidden.** Health indices carry confidence, RUL
   carries a 90 % band and refuses to publish without a statistically significant
   slope, and indices are explicitly held outside the governed envelope rather than
   extrapolated. A system that always answers is more dangerous than one that
   sometimes says it does not know.

---

## Summary schedule

| Phase | Duration | Gate |
|---|---|---|
| 1 Test-rig correlation | 3–4 months | model error within ±3 % fuel, ±5 K CHT |
| 2 Seeded-fault campaign | 4–6 months | >90 % detection, <1 % false alarm on rig data |
| 3 Edge deployment | 3–4 months (parallel) | 20 Hz sustained, <50 % CPU on target |
| 4 Secure telemetry / GCS | 3 months | mutual TLS, signed recordings, GCS integration |
| 5 Fleet and federated | 6+ months | realised RUL error tracked and reported |

Roughly 18 months to a fielded fleet capability, with a usable rig-correlated
demonstrator at the end of Phase 2.
