# Quick start

No installation, no dependencies. Python 3.9+ only.

```bash
cd aero-dt
python3 --version          # 3.9 or newer
python3 scripts/selftest.py
```

That runs 34 verification checks — ISA atmosphere against the standard tables,
linear algebra, CAN round-trip, filter consistency, the engine model against the
914-class performance envelope, energy balance, degradation to TBO, detector
calibration and the recorder. All 34 should pass in about a minute. If they do,
everything else will work.

---

## 1. Identify the healthy baselines (~4 minutes, once)

```bash
python3 scripts/train.py
```

Sweeps the flight envelope at four engine ages and three atmospheres, fits the PCA
monitor and the autoencoder, then reports the false-alarm rate on healthy engines it
has never seen. Writes `data/models.json` (~43 kB).

A pre-trained `data/models.json` ships with this archive, so you can skip this and
come back to it after you have changed something.

---

## 2. Run the dashboard

```bash
python3 scripts/serve.py
```

Open <http://127.0.0.1:8080>. Five tabs: **Live** (gauges, per-cylinder temperatures,
health indices, trends), **Diagnostics & AI** (anomaly decomposition, ranked fault
hypotheses with evidence, estimator innovations, fault injection), **Maintenance**
(advisories, RUL with confidence bands, service actions), **Replay & Reports**
(recorded sorties and mission health reports), **Architecture / ICD**.

Useful options:

```bash
python3 scripts/serve.py --profile hot_weather --scenario cooling_loss --hours 600
python3 scripts/serve.py --port 9000 --no-record
python3 scripts/serve.py --socketcan --channel vcan0    # Linux, real CAN bus
```

In the header: **Run/Pause**, a speed multiplier (1× to 600×), the mission profile,
the fault scenario and the engine's logged hours. **Reset** restarts with those
settings. On the Diagnostics tab you can inject any fault from the catalogue into a
running sortie and watch the twin find it.

---

## Deploying

See `DEPLOY.md` (Render, Railway, Heroku, Docker; `PORT` is honoured automatically).

---

## 3. Watch a fault develop, narrated

```bash
python3 scripts/demo.py --scenario progressive_misfire
python3 scripts/demo.py --scenario cooling_loss --profile hot_weather
python3 scripts/demo.py --list
```

Prints a running table, then the detection timeline, final health indices, RUL,
maintenance advisories, the mission recommendation, and finally the ground truth —
so you can check the twin's conclusion against what was actually injected.

---

## 4. Measure it

```bash
python3 scripts/evaluate.py              # detection, false alarms, lead time, isolation
python3 scripts/lifecycle.py             # accelerated multi-sortie campaign, RUL trend
```

`evaluate.py` takes about 15 minutes: eight fault scenarios plus three healthy control
runs, all scored with the same criterion. The control runs are the important half.

---

## Where to change things

| To change | Edit |
|---|---|
| engine, limits, sensor suite | `dt/config.py` |
| the physics | `dt/engine_model.py` |
| wear rates and service actions | `dt/degradation.py` |
| CAN ICD (PGNs, signals, rates) | `dt/canbus.py` |
| detectors and fusion | `dt/anomaly.py` |
| health index formulas | `dt/health.py` |
| fault signatures | `dt/diagnostics.py` — `SIGNATURES` |
| RUL thresholds | `dt/rul.py` — `RULEstimator.THRESHOLDS` |
| advisory wording and urgency | `dt/advisory.py` |
| mission profiles | `dt/missions.py` |
| the dashboard | `dt/web/index.html` (plain HTML/CSS/JS, no build step) |

After changing anything in the physics or the feature vector, re-run
`scripts/train.py` — the learned baselines are tied to both.

Read `README.md` for the design rationale and the measured numbers, and
`docs/architecture.md` for interfaces and data rates.
