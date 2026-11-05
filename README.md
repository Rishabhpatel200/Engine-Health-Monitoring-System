# AeroDT — Digital Twin for MALE UAV Aero Piston Engines

A modular digital twin for the turbocharged four-cylinder aero piston engines used
on Medium Altitude Long Endurance UAVs. It maintains a continuously synchronised
virtual engine from live CAN telemetry, estimates unmeasured states, detects and
isolates faults before they reach a certified limit, tracks degradation, estimates
remaining useful life, and replays complete sorties.

**Pure Python standard library.** No numpy, no scikit-learn, no web framework, no
CAN stack. It runs on a bare CPython install, which is what a GCS or an airborne
edge node on a closed network can actually be given.

```bash
python3 scripts/train.py          # identify healthy baselines (~4 min)
python3 scripts/serve.py          # dashboard on http://127.0.0.1:8080
python3 scripts/demo.py --scenario progressive_misfire
python3 scripts/evaluate.py       # detection / false-alarm / lead-time metrics
```

---

## What it does

| Capability | Where |
|---|---|
| Virtual engine synchronised to live data | `dt/twin.py`, `dt/engine_model.py` |
| Real-time parameter visualisation | `dt/web/index.html` |
| Health indices for 8 subsystems | `dt/health.py` |
| Abnormal-condition detection | `dt/anomaly.py` |
| Failure prediction before occurrence | `dt/diagnostics.py` |
| Degradation trends and RUL | `dt/degradation.py`, `dt/rul.py` |
| Mission and environment simulation | `dt/missions.py`, `dt/atmosphere.py` |
| Post-flight analysis and replay | `dt/recorder.py` |
| CAN / SocketCAN acquisition | `dt/canbus.py` |
| Maintenance advisories | `dt/advisory.py` |

---

## The central idea

Conventional engine monitoring compares each channel against a fixed limit. That
fails for two reasons: a limit is only breached *after* the damage, and a raw
channel value means nothing without its operating point — 118 °C CHT is routine on
a hot-day climb and alarming during a high-altitude loiter.

This twin removes the operating point before it does anything else. A **reference
model** — the same physics, with new-engine health parameters — is clamped to the
measured speed, manifold pressure, altitude and air temperature. Every quantity the
analytics consume is then a ratio against that reference:

```
breathing index      = measured fuel flow / new-engine fuel flow at this operating point
power index          = shaft power from blade angle / new-engine power here
cooling index        = new-engine CHT rise over ambient / measured CHT rise
oil pressure index   = measured pressure / new-engine pressure at this oil temperature
```

These are near-invariant across the whole flight envelope. A 5 % drop in breathing
index means the same thing at sea level and at FL250, which is what lets one set of
learned baselines cover the entire envelope, and what lets a slope in the index be
read as degradation rather than as a change of flight condition.

---

## Architecture

```
ENGINE ─► sensors ─► ECU/FADEC ─► CAN 2.0B, J1939 framing, 500 kbit/s ─► edge node
                                                                            │
   ┌────────────────────── DIGITAL TWIN CORE ───────────────────────────────┘
   │
   │  reference model    healthy physics clamped to the measured operating point
   │  UKF synchronisation 8-state unscented filter running the plant model;
   │                     innovations are the statistically correct residual basis
   │  feature build      26 physics-normalised residuals and dimensionless indices
   │  AI/ML              PCA (T² + SPE)  ∥  autoencoder  ∥  CUSUM / EWMA
   │  health             8 subsystem indices, each with a confidence
   │  fault isolation    structured-residual signature matrix
   │  RUL                local-linear-trend Kalman filter with a 90 % band
   │  advisory           urgency-ranked actions and a mission recommendation
   └────────────────────────────────────────────────────────────────────────
        │                        │                         │
  operator dashboard     flight data recorder        fleet / depot
  (SSE, live)            (SQLite, full replay)       health records
```

Each stage is a separate module with no back-references, so the physics model can be
swapped for a different engine, the ML layer replaced, or the CAN transport pointed
at a real bus, without touching the others.

---

## The physics model

A mean-value, per-cylinder-resolved model of a 1211 cc turbocharged boxer four
(`dt/engine_model.py`):

- **Air path** — speed-density with wastegate-controlled boost, compressor pressure
  ratio limited by spool state, intercooler effectiveness, manifold filling dynamics.
- **Combustion** — Otto-cycle indicated efficiency modulated by per-cylinder lambda,
  spark timing deviation from MBT, residual gas fraction and knock. Misfire is a
  threshold process on delivered spark energy, so plug-driven misfire appears
  suddenly rather than gradually — as it does in service.
- **Friction** — Chen–Flynn FMEP with oil-viscosity correction, so a cold-oil start
  and a hot worn engine both behave correctly.
- **Rotational** — `J dω/dt = T_brake − T_prop` with a PI constant-speed governor
  operating between blade-angle stops. Off-governor behaviour at idle and above the
  critical altitude falls out of the same equations.
- **Thermal** — lumped nodes: CHT per cylinder → coolant → radiator, plus an oil node
  fed by friction and piston cooling. Cooling mass flow combines ram air and
  propeller-induced flow, which is why the model overheats on the ground and not in
  the cruise.
- **Vibration** — order-resolved. A missing firing event on a four-stroke repeats
  every two revolutions, so misfire energy lands on the **0.5 order**, while rotating
  imbalance lands on **1x**. That separation is what discriminates a misfiring
  cylinder from a worn bearing.

### Validation against the engine class

| Condition | Model | 914-class reference |
|---|---|---|
| Takeoff, sea level | 80.3 kW @ 5800 rpm, 132 kPa MAP, 32.8 L/h | 84.5 kW, ~33 L/h |
| BSFC at takeoff | 292 g/kWh | ~290–320 g/kWh |
| Power at 4900 m (critical altitude) | 76.6 kW | full rated power maintained |
| Power at 7620 m | 59.3 kW | boost falls off above critical altitude |
| Hot day ISA+35 | 71.9 kW, oil 113 °C | derated, oil near its 130 °C limit |
| Throttle slam 35→100 % | 1.9 % overspeed, settled in 30 s | governor transient |
| Power loss at 1200 h TBO | 6.6 % | 5–8 % typical at overhaul |

ISA atmosphere reproduces the standard tables exactly (101.3 / 70.1 / 37.6 kPa and
−56.5 °C at the tropopause).

---

## State estimation

An **unscented** Kalman filter runs the engine model itself as its process model over
eight states (speed, manifold pressure, four cylinder head temperatures, coolant,
oil). Unscented rather than extended because the plant is full of non-differentiable
saturations — wastegate, oil relief valve, pitch stops.

It delivers three things a threshold system cannot:

1. estimates of states that are not instrumented, and of true states behind a lying
   sensor;
2. innovation sequences that are zero-mean and unit-variance on a healthy engine —
   the statistically correct basis for anomaly detection;
3. graceful handling of dropped CAN frames: a missing signal is simply skipped in the
   update and the prediction carries on.

Filter consistency is checked in the test suite (mean NIS ≈ 1.0).

---

## The AI/ML layer

Three complementary detectors, fused:

**PCA monitor** — Hotelling T² measures movement *inside* the healthy correlation
structure; SPE (Q) measures movement *orthogonal* to it, i.e. a relationship between
channels that a healthy engine never showed. Both decompose into per-feature
contributions, so the alert carries its own evidence rather than a bare score.

**Autoencoder** — a 26-15-5-15-26 bottleneck network catching nonlinear couplings the
linear subspace misses. Per-feature reconstruction error is again the explanation.

**CUSUM / EWMA** — sequential change detection for drift too slow to trip any
instantaneous threshold, which is precisely the failure mode conventional monitoring
misses. Two properties make it usable in flight rather than merely on paper:

- it is **gated on steady state** and decimated to 1 Hz. A CUSUM compares against a
  fixed reference; running it through a climb makes it accumulate legitimate
  operating-point changes until it saturates.
- its reference **mean** is learned per airframe from that sortie's first stabilised
  samples, while its reference **spread** comes from the training fleet. Using the
  airframe's own spread would set the scale from a single operating point, and every
  later power change would then read as many sigma of drift.

Both were found the hard way: the first two training runs produced a 100 % false-alarm
rate on healthy engines, entirely from these two effects.

---

## Fault isolation

Detection says something is wrong; isolation says what. A **structured-residual
signature matrix** (`dt/diagnostics.py`) gives each candidate fault a prescribed
pattern of symptoms it must produce, symptoms it must not, and a weight marking the
symptom that actually discriminates it:

| | EGT one low | EGT one high | 0.5-order vib | 1x vib | oil press low | CHT all high |
|---|---|---|---|---|---|---|
| misfire | **++** | | **++** | − | | |
| injector clog | | **++** | + | − | | + |
| bearing wear | − | | − | **++** | + | |
| cooling degradation | | | | | | **++** |
| sensor fault | | | − | − | | − |

A cylinder running cold *with* 0.5-order vibration is a misfire. Running hot *with*
reduced fuel flow is a clogged injector. Elevated 1x *without* 0.5-order is the
bottom end. One channel inconsistent with every other channel is the instrument, not
the engine — the twin's single most valuable discrimination, since it is what stops
crews replacing good hardware.

This is deliberately not a black-box classifier: a maintainer can read the evidence
line by line and disagree with it.

---

## Measured performance

All numbers from `scripts/evaluate.py`, on engines and profiles the models were
never trained on. Ground truth is read only by the scoring code; the twin sees
decoded CAN signals and nothing else.

| Scenario | Detection | Isolation | First detection | Threshold layer | Lead time |
|---|---|---|---|---|---|
| progressive misfire | 95.6 % | 98.8 % | t+1457 s | never tripped | 65.7 min |
| injector fouling | 68.0 % | 94.4 % | t+2540 s | never tripped | 47.7 min |
| cooling loss † | 98.3 % | 98.7 % | t+1711 s | never tripped | 61.5 min |
| lubrication failure | 92.9 % | 100 % | t+1583 s | never tripped | 63.6 min |
| sensor drift | 75.6 % | 86.8 % | t+2182 s | never tripped | 53.6 min |
| detonation † | 46.5 % | 100 % | t+462 s | never tripped | 82.3 min |
| turbo decay | 80.3 % | 99.7 % | t+2213 s | never tripped | 53.1 min |
| combined | 95.6 % | 98.8 % | t+1457 s | never tripped | 65.7 min |
| **mean** | **81.6 %** | **97.2 %** | | | |

† evaluated on the hot-day profile: neither fault is physically expressible at
6100 m, where intake air is 23 °C. Pairing them with a high-altitude sortie would
have measured the profile rather than the detector.

**False alarms: 1.0 %** over 28 800 samples from three healthy engines (470 h,
330 h and 880 h) on three profiles, using the identical detection criterion.

**"Never tripped" is the headline.** In every scenario the sortie ended before any
certified limit was reached. A conventional threshold system would have returned
each of these aircraft with no indication at all, and the fault would have been
found on the next inspection — or in flight.

Detonation detects on only 46.5 % of faulted samples because detonation is only
*expressed* part of the time: the same off-spec fuel that knocks heavily on a
hot-day climb knocks not at all in a cold high-altitude cruise. Isolation is 100 %
whenever it is expressed.

### What the control runs caught

The healthy-engine control runs are the most useful part of the harness. Three
separate design faults were found only because a healthy engine was scored with the
same criterion as a faulted one:

| Symptom | Cause | Fix |
|---|---|---|
| 100 % false alarms | CUSUM run at 20 Hz across changing operating points | steady-state gating, 1 Hz decimation, per-regime baselines |
| 100 % false alarms at 880 h | health indices scored against a *zero-time* engine, so every mid-life airframe read 70 and raised standing advisories | indices scored against serviceability limits |
| all eight indices identical | one anomaly score capped every subsystem uniformly | penalty routed by contributor share to the subsystems it points at |
| contributor deviations all read 0σ | the per-airframe baseline was learned *after* the fault developed, so the sequential detector adopted the faulted state as normal | a sample joins the baseline only while the fleet-trained models agree the engine looks healthy |

---

## Honest engineering notes

Things that are modelled as observable because they genuinely are, and things that
are not:

- **Power requires a load observable.** Shaft power cannot be inferred from speed and
  manifold pressure alone. The ICD therefore carries propeller governor blade angle
  (`prop_load_pct`); power absorbed by a constant-speed propeller is `k(β)·σ·ω³`. On
  an installation without a blade-angle or torque signal, the power index degrades to
  1.0 and the power-based diagnostics are unavailable. This is stated rather than
  faked.
- **Crankcase blowby is inferred, not measured.** With no crankcase pressure sensor,
  the blowby index is derived from the breathing index and is correspondingly weaker
  evidence.
- **Health indices are only computed inside the governed envelope** (above 2200 rpm).
  Reference ratios at idle are meaningless; the twin holds them rather than letting
  them inject noise into the trends that drive RUL.
- **RUL is published only when the slope is statistically significant** (|slope| > 2σ)
  and is capped at the remaining time to overhaul. *"No significant trend"* is a valid
  and common answer, and the system says so rather than inventing a number.
- **Detonation needs a knock sensor.** It cannot be inferred from speed, load and
  temperature, and it certainly cannot be read from the reference model — which
  always has in-spec fuel and would therefore never report knock. The ICD carries a
  knock channel.
- **Cooling degradation is invisible in temperature until it is severe.** A 58 %
  loss of radiator capacity raises CHT by only 6.5 K because the thermostat opens to
  compensate; at 70 % it saturates and CHT jumps 12 K at once. The observable that
  moves early is regulator position, so the ICD carries it. Monitoring temperature
  alone catches the cliff and misses the whole approach to it.
- **The recorder is tiered because it has to be.** Writing whole twin snapshots at
  2 Hz costs 0.9 GB per 24 h sortie. Splitting into a compact 2 Hz stream, a full
  once-a-minute summary and a 20 Hz event ring, all zlib-compressed, brings it to a
  measured 134 MB per sortie plus ~3 MB per triggered event.
- **Measured throughput is ~700 steps/s, about 35× real time**, with the trained
  models loaded and the recorder running. A much higher figure is easy to obtain by
  benchmarking before the ML layer is loaded; that number would be useless for
  sizing an edge deployment.
- **Degradation runs in real time by default.** Real engines degrade over hundreds of
  hours. `SimConfig.degradation_accel` compresses wear for lifecycle studies; it is
  1.0 for mission runs, so an ordinary sortie shows the trend an ordinary sortie
  actually produces.

---

## Layout

```
dt/
  config.py        engine specification, certified limits, sensor suite
  atmosphere.py    ISA with off-standard deviation, ram recovery, cooling flow
  engine_model.py  the physics
  degradation.py   severity-driven wear, maintenance actions
  faults.py        fault catalogue and injection scheduler
  sensors.py       noise, bias, drift, quantisation, dropout
  canbus.py        J1939-style ICD, SocketCAN and loopback transports
  linalg.py        Cholesky, SPD solve, Jacobi eigendecomposition
  estimator.py     unscented Kalman filter
  anomaly.py       PCA monitor, autoencoder, CUSUM/EWMA, fusion
  health.py        condition indicators and subsystem health indices
  diagnostics.py   structured-residual fault isolation
  rul.py           local-linear-trend RUL with confidence bands
  advisory.py      maintenance advisories and mission recommendation
  missions.py      mission profiles and environment scheduling
  recorder.py      SQLite flight data recorder and replay
  twin.py          the core framework
  simulator.py     closed-loop plant → sensors → CAN → twin harness
  server.py        GCS HTTP/SSE service
  web/index.html   operator dashboard
scripts/
  train.py         baseline identification
  demo.py          narrated end-to-end scenario run
  evaluate.py      detection / false-alarm / lead-time / isolation metrics
  lifecycle.py     accelerated multi-sortie campaign for RUL demonstration
  selftest.py      model and pipeline verification
docs/
  architecture.md       design and interfaces
  deployment_roadmap.md path to a fielded system
```

## Using a real CAN bus

On Linux with a real or virtual interface:

```bash
sudo ip link add dev vcan0 type vcan && sudo ip link set up vcan0
python3 scripts/serve.py --socketcan --channel vcan0
```

`dt/canbus.py` binds `AF_CAN`/`SOCK_RAW` directly. On any other host it falls back to
an in-process loopback with identical framing, so the encode/decode path is exercised
either way. The ICD is `PGN_MAP`; changing an engine's signal set means editing that
one table.
