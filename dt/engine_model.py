"""
Physics-based aero piston engine model (mean-value, per-cylinder resolved).

Structure
---------
  air path      speed-density with turbocharger + wastegate + intercooler
  combustion    Otto-cycle indicated efficiency modulated by lambda, spark
                timing, residual fraction and knock
  friction      Chen-Flynn FMEP with oil-viscosity scaling
  rotational    J dw/dt = T_brake - T_prop, constant-speed governor on pitch
  thermal       lumped nodes: CHT[n] -> coolant -> radiator, oil -> oil cooler
  exhaust       per-cylinder energy balance -> EGT with probe lag
  lubrication   pump curve x viscosity x wear, relief-valve limited
  electrical    alternator curve, battery SoC and terminal voltage
  vibration     order-resolved: 0.5x (misfire/imbalance), 1x (bearing), broadband

Every degradation and fault effect enters through `HealthParams`, so the same
equations serve the healthy twin, the degraded twin and the fault simulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

from .atmosphere import AtmState, ram_recovery, cooling_mass_flow
from .config import (AFR_STOICH, CP_AIR, CP_EXH, EngineSpec, GAMMA_AIR,
                     LHV_AVGAS, R_AIR)

# Fraction of released fuel energy that leaves through the combustion-chamber
# walls rather than the exhaust valve. Shared by the CHT and EGT paths so the
# twin's energy balance closes: brake + friction + wall + exhaust = released.
WALL_HEAT_FRACTION = 0.21
CYCLE_QUALITY = 0.70          # real-cycle derate against the ideal Otto cycle
INJECTOR_STATIC_FLOW_KGS = 0.0036
MISFIRE_ENERGY_THRESHOLD = 0.55   # spark energy below which misfire begins


# --------------------------------------------------------------------------
# Health / degradation parameters (1.0 = new engine, unless noted)
# --------------------------------------------------------------------------
@dataclass
class HealthParams:
    n_cyl: int = 4
    ring_seal: float = 1.0            # ring/bore seal quality -> VE, blowby
    valve_seal: float = 1.0           # seat recession -> VE, compression
    injector_flow: List[float] = field(default_factory=list)   # per cylinder
    ignition_energy: List[float] = field(default_factory=list) # per cylinder
    cooling_ua: float = 1.0           # radiator/fin fouling, coolant loss
    oil_cooler_ua: float = 1.0
    oil_pump: float = 1.0
    bearing_clearance: float = 1.0    # >1 = worn, leaks oil pressure
    oil_viscosity: float = 1.0        # <1 = sheared / fuel-diluted oil
    turbo_efficiency: float = 1.0
    wastegate_authority: float = 1.0
    alternator: float = 1.0
    battery_soh: float = 1.0
    mount_stiffness: float = 1.0
    fuel_octane: float = 100.0        # MON-ish; drives knock margin

    def __post_init__(self):
        if not self.injector_flow:
            self.injector_flow = [1.0] * self.n_cyl
        if not self.ignition_energy:
            self.ignition_energy = [1.0] * self.n_cyl

    def copy(self) -> "HealthParams":
        h = HealthParams(**{k: v for k, v in self.__dict__.items()
                            if k not in ("injector_flow", "ignition_energy")})
        h.injector_flow = list(self.injector_flow)
        h.ignition_energy = list(self.ignition_energy)
        return h


@dataclass
class Command:
    """ECU/FADEC command vector."""
    throttle_pct: float = 0.0
    rpm_target: float = 5000.0        # constant-speed governor setpoint
    lambda_cmd: float = 0.0           # 0 = let the ECU schedule choose
    spark_adv_trim_deg: float = 0.0   # test-rig / fault injection trim
    elec_load_w: float = 120.0        # avionics + payload draw
    ignition_on: bool = True
    fuel_pump_on: bool = True


@dataclass
class EngineState:
    """Continuous dynamic states integrated by the model."""
    rpm: float = 0.0
    map_kpa: float = 30.0
    cht_c: List[float] = field(default_factory=lambda: [15.0] * 4)
    egt_c: List[float] = field(default_factory=lambda: [15.0] * 4)
    coolant_c: float = 15.0
    oil_c: float = 15.0
    prop_k_eff: float = 3.45e-5
    batt_soc: float = 0.95
    hours: float = 0.0

    def copy(self) -> "EngineState":
        s = EngineState(**{k: v for k, v in self.__dict__.items()
                           if k not in ("cht_c", "egt_c")})
        s.cht_c = list(self.cht_c)
        s.egt_c = list(self.egt_c)
        return s

    def as_vector(self) -> List[float]:
        return [self.rpm, self.map_kpa, *self.cht_c, self.coolant_c, self.oil_c]

    @staticmethod
    def vector_names(n_cyl: int = 4) -> List[str]:
        return (["rpm", "map_kpa"] + [f"cht{i+1}_c" for i in range(n_cyl)]
                + ["coolant_c", "oil_c"])


@dataclass
class Outputs:
    """Algebraic outputs recomputed every step (the twin's observable set)."""
    rpm: float = 0.0
    prop_rpm: float = 0.0
    map_kpa: float = 0.0
    iat_c: float = 0.0
    boost_pr: float = 1.0
    volumetric_eff: float = 0.0
    air_flow_kgs: float = 0.0
    fuel_flow_kgs: float = 0.0
    fuel_flow_lph: float = 0.0
    lambda_avg: float = 1.0
    lambda_cyl: List[float] = field(default_factory=list)
    imep_kpa: List[float] = field(default_factory=list)
    bmep_kpa: float = 0.0
    fmep_kpa: float = 0.0
    torque_nm: float = 0.0
    power_kw: float = 0.0
    bsfc_g_kwh: float = 0.0
    cht_c: List[float] = field(default_factory=list)
    egt_c: List[float] = field(default_factory=list)
    coolant_c: float = 0.0
    oil_temp_c: float = 0.0
    oil_press_kpa: float = 0.0
    vib_rms_g: float = 0.0
    vib_halforder_g: float = 0.0
    vib_1x_g: float = 0.0
    vib_broadband_g: float = 0.0
    misfire_rate: List[float] = field(default_factory=list)
    knock_index: float = 0.0
    blowby_lpm: float = 0.0
    bus_v: float = 0.0
    alt_current_a: float = 0.0
    batt_soc: float = 0.0
    throttle_pct: float = 0.0
    inj_pulse_ms: float = 0.0
    spark_adv_deg: float = 0.0
    prop_load_pct: float = 0.0
    knock_level: float = 0.0
    coolant_valve_pct: float = 0.0


class PistonEngine:
    """Mean-value engine model integrated with a fixed-step explicit scheme.

    The step size (default 50 ms) is far above the fastest thermal mode and far
    below the slowest, so forward Euler is stable; the two fast states (crank
    speed, manifold pressure) use analytic exponential updates for robustness.
    """

    def __init__(self, spec: Optional[EngineSpec] = None,
                 health: Optional[HealthParams] = None, seed: int = 0):
        self.spec = spec or EngineSpec()
        self.health = health or HealthParams(n_cyl=self.spec.n_cyl)
        self.rng = random.Random(seed)
        n = self.spec.n_cyl
        self.state = EngineState(cht_c=[15.0] * n, egt_c=[15.0] * n,
                                 prop_k_eff=self.spec.prop_k)
        self.out = Outputs(lambda_cyl=[1.0] * n, imep_kpa=[0.0] * n,
                           cht_c=[15.0] * n, egt_c=[15.0] * n,
                           misfire_rate=[0.0] * n)
        self.running = False
        # Rotational inertia of crank + flywheel + gearbox + propeller, all
        # referred to the crankshaft (prop inertia divided by ratio squared).
        self.inertia = 0.30
        self._vd_m3 = self.spec.displacement_l / 1000.0
        self._vd_cyl = self._vd_m3 / n
        # small, fixed manufacturing maldistribution between cylinders
        self._air_split = [1.0 + 0.012 * math.sin(2.1 * i + 0.7) for i in range(n)]
        s = sum(self._air_split)
        self._air_split = [x * n / s for x in self._air_split]
        self._egt_probe_tau = 1.8

    # -- initial conditions -------------------------------------------------
    def soak(self, atm: AtmState) -> None:
        """Cold-soak the engine to ambient (pre-start condition)."""
        t = atm.temp_c
        self.state.cht_c = [t] * self.spec.n_cyl
        self.state.egt_c = [t] * self.spec.n_cyl
        self.state.coolant_c = t
        self.state.oil_c = t
        self.state.rpm = 0.0
        self.state.map_kpa = atm.press_kpa
        self.running = False

    def start(self, atm: AtmState) -> None:
        self.state.rpm = max(self.state.rpm, self.spec.idle_rpm * 0.85)
        self.running = True

    # -- component sub-models ----------------------------------------------
    def _air_path(self, cmd: Command, atm: AtmState, tas_ms: float, dt: float):
        """Turbocharger, wastegate, throttle plate and manifold filling."""
        sp = self.spec
        h = self.health
        inlet = ram_recovery(atm, tas_ms)
        p_amb = inlet.press_kpa

        # Wastegate holds a constant absolute plenum pressure until the
        # compressor runs out of pressure ratio (the critical altitude).
        pr_avail = sp.turbo_pressure_ratio_max * (0.55 + 0.45 * h.turbo_efficiency)
        # spool-up: available PR grows with exhaust energy, i.e. with speed/load
        spool = min(1.0, (self.state.rpm / sp.rated_rpm) ** 0.7
                    * (0.35 + 0.65 * cmd.throttle_pct / 100.0) * 1.35)
        pr_cap = 1.0 + (pr_avail - 1.0) * spool
        p_plenum_max = p_amb * pr_cap

        target = sp.boost_target_kpa * (0.5 + 0.5 * h.wastegate_authority)
        p_plenum = min(target, p_plenum_max)

        # Throttle plate: effective flow area vs. blade angle (non-linear).
        # Blade-angle flow area, then the saturating area-to-pressure relation
        # that makes MAP insensitive to throttle near wide-open.
        th = max(0.0, min(100.0, cmd.throttle_pct)) / 100.0
        a = 1.0 - math.cos(th * math.pi / 2.0)
        c = 0.18
        frac = (a / (a + c)) * (1.0 + c)
        idle_map = 0.26 * p_amb + 4.0
        map_target = idle_map + (p_plenum - idle_map) * frac
        map_target = max(12.0, min(map_target, p_plenum))
        # Retained so the twin can compare measured MAP against the pressure a
        # healthy air path would reach at this throttle and altitude. Comparing
        # against the reference model's own manifold state cannot work: that
        # state is clamped to the measurement and so can never disagree with it.
        self.last_map_target = map_target
        self.last_plenum_kpa = p_plenum

        # Manifold filling dynamics (fast, ~120 ms).
        tau = 0.12
        self.state.map_kpa += (map_target - self.state.map_kpa) * (1.0 - math.exp(-dt / tau))

        pr_actual = max(1.0, p_plenum / max(p_amb, 1e-3))
        eta_c = 0.74 * h.turbo_efficiency
        t_comp = inlet.temp_k * (1.0 + ((pr_actual ** ((GAMMA_AIR - 1) / GAMMA_AIR)) - 1.0)
                                 / max(eta_c, 0.2))
        t_iat = t_comp - sp.intercooler_effectiveness * (t_comp - inlet.temp_k)
        return t_iat, pr_actual, inlet

    def _volumetric_efficiency(self, rpm: float, map_kpa: float, p_exh_kpa: float) -> float:
        """VE map: breathing peak near 0.85 rated speed, residual-gas and
        wear corrections applied multiplicatively."""
        sp = self.spec
        x = rpm / sp.rated_rpm
        base = 0.70 + 0.52 * x - 0.30 * x * x          # cam/runner tuning shape
        base *= 1.0 - 0.80 * max(0.0, x - 1.05)        # valve float past redline
        # Residual burned gas left in the clearance volume displaces fresh
        # charge; it grows sharply once MAP falls below exhaust backpressure,
        # which is what collapses VE at idle and closed throttle.
        pr = map_kpa / max(p_exh_kpa, 1.0)
        resid = 0.050 / max(pr, 0.10) ** 1.5
        base *= (1.0 - min(0.60, resid))
        # wear: lost ring seal and valve seal both cost trapped mass
        base *= (0.55 + 0.45 * self.health.ring_seal)
        base *= (0.70 + 0.30 * self.health.valve_seal)
        return max(0.10, min(1.05, base))

    def _lambda_schedule(self, cmd: Command, load: float) -> float:
        """ECU open-loop lambda schedule: enrichment for charge cooling at
        high load, near-stoichiometric best-economy cruise."""
        if cmd.lambda_cmd > 0.0:
            return cmd.lambda_cmd
        if load > 0.85:
            return 0.82
        if load > 0.65:
            return 0.88 + 0.04 * (0.85 - load) / 0.20
        return 0.98 - 0.06 * max(0.0, (0.5 - load))

    @staticmethod
    def _combustion_efficiency(lam: float) -> float:
        """Fraction of injected fuel energy actually released."""
        if lam < 1.0:
            return max(0.05, 0.94 * lam / 1.0)          # oxygen-limited
        if lam <= 1.35:
            return 0.965 - 0.05 * (lam - 1.0)
        # beyond the lean limit combustion becomes partial then fails
        return max(0.0, 0.948 - 1.9 * (lam - 1.35))

    def _spark_advance(self, rpm: float, load: float, knock: float) -> float:
        """MBT-seeking advance with knock retard."""
        adv = 8.0 + 22.0 * (rpm / self.spec.rated_rpm) - 9.0 * load
        return max(0.0, min(34.0, adv - 12.0 * knock))

    def _knock_index(self, iat_c: float, cht_c: float, map_kpa: float,
                     lam: float, rpm: float) -> float:
        """0 = large margin, 1 = heavy detonation. Charge temperature, boost,
        mixture and octane set the margin; cruise-lean hot-and-high is the
        classic MALE knock corner."""
        h = self.health
        octane_margin = (h.fuel_octane - 87.0) / 13.0
        drive = (0.0125 * max(0.0, iat_c - 35.0)
                 + 0.0090 * max(0.0, cht_c - 105.0)
                 + 0.0130 * max(0.0, map_kpa - 118.0)
                 + 0.55 * max(0.0, lam - 1.02)
                 - 0.35 * max(0.0, 0.95 - lam)
                 - 0.00012 * max(0.0, rpm - 3000.0))
        return max(0.0, min(1.0, drive - 0.55 * octane_margin + 0.10))

    def _fmep_kpa(self, rpm: float, pmax_kpa: float) -> float:
        """Chen-Flynn friction correlation, viscosity-corrected."""
        sp = self.spec
        sp_piston = 2.0 * (sp.stroke_mm / 1000.0) * rpm / 60.0     # m/s
        # constants folded to include the reduction-gearbox and accessory drag
        fmep = 95.0 + 0.0080 * pmax_kpa + 2.20 * sp_piston + 0.0280 * sp_piston ** 2
        visc = self._viscosity_ratio()
        fmep *= (0.80 + 0.20 * visc ** 1.4)
        fmep *= (1.0 + 0.30 * max(0.0, self.health.bearing_clearance - 1.0))
        return fmep

    def _viscosity_ratio(self) -> float:
        """Kinematic viscosity relative to the 90 C reference (Vogel-like)."""
        t = max(-40.0, self.state.oil_c)
        vr = math.exp(1600.0 * (1.0 / (t + 273.15) - 1.0 / (self.spec.oil_visc_ref_temp_c + 273.15)))
        return max(0.25, min(8.0, vr * self.health.oil_viscosity))

    # -- main integration step ---------------------------------------------
    def step(self, dt: float, cmd: Command, atm: AtmState, tas_ms: float = 45.0) -> Outputs:
        sp, h, st = self.spec, self.health, self.state
        n = sp.n_cyl

        t_iat_k, pr, inlet = self._air_path(cmd, atm, tas_ms, dt)
        iat_c = t_iat_k - 273.15

        # --- trapped mass -------------------------------------------------
        p_exh = inlet.press_kpa * 1.10 + 6.0 * (st.rpm / sp.rated_rpm) ** 2
        ve = self._volumetric_efficiency(st.rpm, st.map_kpa, p_exh)
        rho_man = st.map_kpa * 1000.0 / (R_AIR * t_iat_k)
        cycles_per_s = st.rpm / 60.0 / (sp.strokes / 2.0)
        m_air = ve * self._vd_m3 * cycles_per_s * rho_man          # kg/s total
        m_air = max(0.0, m_air)

        load = min(1.0, st.map_kpa / max(sp.boost_target_kpa, 1.0))
        lam_cmd = self._lambda_schedule(cmd, load)

        # --- per-cylinder fuelling, combustion, work ----------------------
        cht_mean = sum(st.cht_c) / n
        knock = self._knock_index(iat_c, cht_mean, st.map_kpa, lam_cmd, st.rpm)
        spark = self._spark_advance(st.rpm, load, knock) + cmd.spark_adv_trim_deg
        spark_mbt = self._spark_advance(st.rpm, load, 0.0)
        # efficiency penalty for deviating from MBT (parabolic, ~0.35 %/deg^2)
        timing_eff = max(0.55, 1.0 - 0.0035 * (spark - spark_mbt) ** 2)

        eta_otto = 1.0 - sp.compression_ratio ** (1.0 - GAMMA_AIR)

        lam_cyl, imep, p_ind_cyl, q_fuel_cyl, m_fuel_cyl, misfire = [], [], [], [], [], []
        fired_any = False
        for i in range(n):
            m_air_i = m_air * self._air_split[i] / n
            if cmd.fuel_pump_on:
                m_f_i = m_air_i / (AFR_STOICH * max(lam_cmd, 0.05)) * h.injector_flow[i]
            else:
                m_f_i = 0.0
            lam_i = (m_air_i / max(m_f_i * AFR_STOICH, 1e-9)) if m_f_i > 1e-12 else 9.9
            lam_i = min(9.9, lam_i)

            # misfire probability: weak spark, mixture outside flammability,
            # low trapped mass at idle, and severe knock all contribute
            p_mis = 0.0
            if cmd.ignition_on and st.rpm > 200.0:
                # Ignition is a threshold process: a partly eroded plug still
                # lights the charge reliably. Misfire begins only once the
                # delivered spark energy falls below the mixture's requirement,
                # then rises steeply - which is why plug-driven misfire appears
                # suddenly rather than gradually.
                weak = max(0.0, (MISFIRE_ENERGY_THRESHOLD - h.ignition_energy[i])
                           / MISFIRE_ENERGY_THRESHOLD)
                p_mis += 0.95 * weak ** 1.6
                if lam_i > 1.45:
                    p_mis += min(0.95, (lam_i - 1.45) * 1.6)
                if lam_i < 0.62:
                    p_mis += min(0.9, (0.62 - lam_i) * 2.2)
                p_mis += 0.25 * max(0.0, knock - 0.85)
                p_mis *= (1.0 + 0.6 * max(0.0, 1.0 - st.rpm / sp.idle_rpm))
            else:
                p_mis = 1.0
            p_mis = max(0.0, min(1.0, p_mis))
            fire_frac = 1.0 - p_mis

            eta_c = self._combustion_efficiency(lam_i)
            q_i = m_f_i * LHV_AVGAS * eta_c * fire_frac                # W released
            eta_ind = eta_otto * CYCLE_QUALITY * timing_eff * (1.0 - 0.18 * knock)
            p_i = q_i * eta_ind
            if p_i > 0:
                fired_any = True
            imep_i = p_i / max(self._vd_cyl * cycles_per_s, 1e-9) / 1000.0  # kPa

            lam_cyl.append(lam_i); imep.append(imep_i); p_ind_cyl.append(p_i)
            q_fuel_cyl.append(q_i); m_fuel_cyl.append(m_f_i); misfire.append(p_mis)

        p_ind = sum(p_ind_cyl)
        imep_mean = sum(imep) / n
        pmax = st.map_kpa * (sp.compression_ratio ** GAMMA_AIR) * (0.55 + 0.45 * load)
        fmep = self._fmep_kpa(st.rpm, pmax)
        p_fric = fmep * 1000.0 * self._vd_m3 * cycles_per_s

        # --- electrical accessories ---------------------------------------
        bus_v, alt_i, alt_load_w = self._electrical(cmd, dt)
        p_acc = alt_load_w / 0.55                       # alternator drive power

        p_brake = p_ind - p_fric - p_acc
        omega = st.rpm * 2.0 * math.pi / 60.0

        # --- propeller load with constant-speed governor -------------------
        st.prop_k_eff = self._governor(cmd, atm, dt, p_brake, omega)
        t_prop = st.prop_k_eff * atm.sigma * omega ** 2 if omega > 0 else 0.0
        t_brake = p_brake / omega if omega > 1.0 else (p_brake / 1.0 if p_brake else 0.0)

        # --- crank dynamics -------------------------------------------------
        if self.running or fired_any:
            domega = (t_brake - t_prop) / self.inertia
            omega = max(0.0, omega + domega * dt)
            st.rpm = omega * 60.0 / (2.0 * math.pi)
            if st.rpm < 120.0 and not fired_any:
                self.running = False
                st.rpm = 0.0
            elif st.rpm > sp.redline_rpm * 1.15:
                st.rpm = sp.redline_rpm * 1.15
        else:
            st.rpm = max(0.0, st.rpm - 900.0 * dt)

        # --- thermal network ------------------------------------------------
        self._thermal(dt, atm, tas_ms, q_fuel_cyl, p_ind_cyl, p_fric, m_air, iat_c)

        # --- exhaust temperature --------------------------------------------
        for i in range(n):
            m_exh_i = (m_air * self._air_split[i] / n) + m_fuel_cyl[i]
            if m_exh_i > 1e-7:
                q_wall_frac = WALL_HEAT_FRACTION * (1.0 + 0.25 * (1.0 - load))
                q_exh = q_fuel_cyl[i] * (1.0 - q_wall_frac) - p_ind_cyl[i]
                t_exh = t_iat_k + max(0.0, q_exh) / (m_exh_i * CP_EXH)
                t_exh = min(t_exh, 1450.0)
                egt_target = (t_exh - 273.15) * 0.90 + 0.10 * st.cht_c[i]  # probe location
            else:
                egt_target = st.cht_c[i]
            a = 1.0 - math.exp(-dt / self._egt_probe_tau)
            st.egt_c[i] += (egt_target - st.egt_c[i]) * a

        # --- lubrication ------------------------------------------------------
        oil_p = self._oil_pressure()

        # --- vibration --------------------------------------------------------
        vib = self._vibration(imep, misfire, knock, st.rpm)

        # --- blowby (ring seal health indicator) -------------------------------
        blowby = (0.9 + 14.0 * (1.0 - h.ring_seal)) * (0.3 + 0.7 * load) * (st.rpm / sp.rated_rpm)

        st.hours += dt / 3600.0

        m_fuel = sum(m_fuel_cyl)
        o = self.out
        o.rpm = st.rpm
        o.prop_rpm = st.rpm / sp.gearbox_ratio
        o.map_kpa = st.map_kpa
        o.iat_c = iat_c
        o.boost_pr = pr
        o.volumetric_eff = ve
        o.air_flow_kgs = m_air
        o.fuel_flow_kgs = m_fuel
        o.fuel_flow_lph = m_fuel / 0.72 * 3600.0 * 1000.0 / 1000.0   # rho=0.72 kg/L
        o.lambda_cyl = lam_cyl
        o.lambda_avg = sum(lam_cyl) / n
        o.imep_kpa = imep
        o.fmep_kpa = fmep
        o.bmep_kpa = imep_mean - fmep
        o.torque_nm = t_brake
        o.power_kw = max(0.0, p_brake) / 1000.0
        o.bsfc_g_kwh = (m_fuel * 3.6e6 / max(o.power_kw, 1e-3)) if o.power_kw > 0.5 else 0.0
        o.cht_c = list(st.cht_c)
        o.egt_c = list(st.egt_c)
        o.coolant_c = st.coolant_c
        o.oil_temp_c = st.oil_c
        o.oil_press_kpa = oil_p
        o.vib_rms_g, o.vib_halforder_g, o.vib_1x_g, o.vib_broadband_g = vib
        o.misfire_rate = misfire
        o.coolant_valve_pct = getattr(self, 'last_valve_pct', 0.0)
        o.knock_index = knock
        # Knock-sensor output: a band-passed accelerometer channel scaled 0-100
        # by the ECU. Detonation is not inferable from speed, load and
        # temperature alone, so an installation without a knock sensor cannot
        # detect it - the ICD therefore carries one.
        o.knock_level = 100.0 * knock
        o.blowby_lpm = blowby
        o.bus_v = bus_v
        o.alt_current_a = alt_i
        o.batt_soc = st.batt_soc
        o.throttle_pct = cmd.throttle_pct
        o.spark_adv_deg = spark
        # Governor blade-angle position, telemetered by the propeller control
        # unit. This is the load observable: without it, delivered power cannot
        # be separated from the engine's speed and manifold pressure.
        span = sp.prop_pitch_max - sp.prop_pitch_min
        o.prop_load_pct = 100.0 * ((st.prop_k_eff / sp.prop_k) - sp.prop_pitch_min) / span
        o.prop_load_pct = max(0.0, min(100.0, o.prop_load_pct))
        o.inj_pulse_ms = ((m_fuel_cyl[0] / max(cycles_per_s, 1e-6))
                          / INJECTOR_STATIC_FLOW_KGS * 1000.0) if cycles_per_s > 0 else 0.0
        return o

    # -- sub-model helpers ---------------------------------------------------
    def _governor(self, cmd: Command, atm: AtmState, dt: float,
                  p_brake: float, omega: float) -> float:
        """Constant-speed propeller governor.

        A PI law drives blade pitch (represented by the torque coefficient) to
        null the speed error. Between the fine and coarse stops the governor
        holds the commanded RPM; outside them the engine is off-governor and
        speed is set by plain torque balance, which is the correct behaviour at
        idle, in a windmilling descent, and above the critical altitude.
        """
        sp = self.spec
        k0 = sp.prop_k
        lo, hi = k0 * sp.prop_pitch_min, k0 * sp.prop_pitch_max
        k = self.state.prop_k_eff
        rpm = omega * 60.0 / (2.0 * math.pi)
        if rpm < 300.0:
            return max(lo, min(hi, k))
        target = max(cmd.rpm_target, sp.idle_rpm)
        err = (rpm - target) / target                     # + = overspeed
        # Feed-forward: the pitch that balances the current brake torque at the
        # setpoint. Feedback then removes model error.
        omega_t = target * 2.0 * math.pi / 60.0
        k_ff = (p_brake / (max(atm.sigma, 0.05) * omega_t ** 3)) if p_brake > 0 else lo
        k_ff = max(lo, min(hi, k_ff))
        self._gov_i = max(-0.8, min(0.8, getattr(self, "_gov_i", 0.0) + 2.2 * err * dt))
        k_cmd = k_ff * (1.0 + 3.0 * err + self._gov_i)
        k_cmd = max(lo, min(hi, k_cmd))
        # blade-angle slew rate limit (a real governor cannot step)
        rate = 1.0 - math.exp(-dt / 0.20)
        return k + (k_cmd - k) * rate

    def _thermal(self, dt: float, atm: AtmState, tas_ms: float,
                 q_fuel_cyl: List[float], p_ind_cyl: List[float],
                 p_fric: float, m_air: float, iat_c: float) -> None:
        """CHT[i] -> coolant -> radiator, plus an oil node fed by friction."""
        sp, h, st = self.spec, self.health, self.state
        n = sp.n_cyl
        m_cool_air = cooling_mass_flow(atm, tas_ms, st.rpm, sp.rated_rpm)
        flow_factor = (m_cool_air / 0.55) ** 0.62          # forced-convection law
        t_cool_air = atm.temp_c + 0.5 * (tas_ms ** 2) / (2 * CP_AIR)

        # coolant pump flow follows engine speed
        pump = 0.25 + 0.75 * min(1.0, st.rpm / sp.rated_rpm)

        for i in range(n):
            q_wall = q_fuel_cyl[i] * WALL_HEAT_FRACTION + p_fric * 0.10 / n
            ua_head = sp.cht_ua_nominal * (0.35 + 0.65 * pump)
            q_out = ua_head * (st.cht_c[i] - st.coolant_c)
            dT = (q_wall - q_out) / (sp.cht_mass_kg * sp.cht_cp)
            st.cht_c[i] += dT * dt

        q_to_coolant = sum(sp.cht_ua_nominal * (0.35 + 0.65 * pump) * (st.cht_c[i] - st.coolant_c)
                           for i in range(n))
        # Thermostat / radiator shutter: closed below 80 C, fully open by 95 C.
        # Its position is the early indicator of cooling degradation: while the
        # regulator still has travel it holds temperature almost constant, so
        # coolant and CHT barely move until it saturates and the temperature
        # then rises abruptly. Monitoring temperature alone therefore misses the
        # whole degradation and catches only the cliff.
        thermo = max(0.05, min(1.0, (st.coolant_c - 80.0) / 15.0))
        self.last_valve_pct = 100.0 * thermo
        ua_rad = sp.coolant_ua_nominal * h.cooling_ua * flow_factor * thermo
        q_rad = ua_rad * (st.coolant_c - t_cool_air)
        st.coolant_c += (q_to_coolant - q_rad) / (sp.coolant_mass_kg * sp.coolant_cp) * dt

        # oil: absorbs friction heat plus piston-cooling heat, rejects to the
        # oil cooler and exchanges with the coolant loop
        q_oil_in = p_fric * 0.55 + sum(q_fuel_cyl) * 0.020
        ua_oil = sp.oil_ua_nominal * h.oil_cooler_ua * flow_factor
        q_oil_out = ua_oil * (st.oil_c - t_cool_air) + 25.0 * (st.oil_c - st.coolant_c)
        st.oil_c += (q_oil_in - q_oil_out) / (sp.oil_mass_kg * sp.oil_cp) * dt

        st.coolant_c = max(-60.0, min(180.0, st.coolant_c))
        st.oil_c = max(-60.0, min(220.0, st.oil_c))
        for i in range(n):
            st.cht_c[i] = max(-60.0, min(400.0, st.cht_c[i]))

    def oil_pressure_at(self, rpm: float, oil_temp_c: float,
                        health: Optional[HealthParams] = None) -> float:
        """Oil pressure the pump would deliver at a given speed and oil
        temperature. Exposed separately so the twin can evaluate its healthy
        reference at the *measured* oil temperature - comparing against a
        reference that is also at a different temperature would confuse a hot
        oil charge with a worn pump."""
        sp = self.spec
        h = health or HealthParams(n_cyl=sp.n_cyl)
        t = max(-40.0, oil_temp_c)
        visc = math.exp(1600.0 * (1.0 / (t + 273.15)
                                  - 1.0 / (sp.oil_visc_ref_temp_c + 273.15)))
        visc = max(0.25, min(8.0, visc * h.oil_viscosity))
        flow = sp.oil_press_gain_kpa_per_krpm * (rpm / 1000.0) * h.oil_pump
        p = flow * (0.55 + 0.45 * min(2.0, visc)) / (h.bearing_clearance ** 2.1)
        return max(0.0, min(p, sp.oil_press_bypass_kpa))

    def _oil_pressure(self) -> float:
        """Gear-pump delivery against a wear-dependent leakage path, limited by
        the relief valve. Hot thin oil in a worn engine is the classic low-
        oil-pressure signature."""
        sp, h, st = self.spec, self.health, self.state
        visc = self._viscosity_ratio()
        flow = sp.oil_press_gain_kpa_per_krpm * (st.rpm / 1000.0) * h.oil_pump
        p = flow * (0.55 + 0.45 * min(2.0, visc)) / (h.bearing_clearance ** 2.1)
        p = min(p, sp.oil_press_bypass_kpa)
        return max(0.0, p)

    def _vibration(self, imep: List[float], misfire: List[float],
                   knock: float, rpm: float):
        """Order-resolved vibration. On a four-stroke a missing firing event
        repeats every two revolutions, so misfire energy lands on the 0.5
        order - which is exactly how it is discriminated from imbalance (1x)."""
        n = len(imep)
        mean_imep = sum(imep) / n if n else 0.0
        speed = rpm / self.spec.rated_rpm
        if mean_imep > 1.0:
            spread = math.sqrt(sum((x - mean_imep) ** 2 for x in imep) / n) / mean_imep
        else:
            spread = 0.0
        mis = sum(misfire) / n
        stiff = 1.0 / max(0.4, self.health.mount_stiffness)

        half = (2.6 * mis + 1.5 * spread) * (0.3 + 0.7 * speed) * stiff
        one_x = (0.35 + 2.4 * max(0.0, self.health.bearing_clearance - 1.0)) * (0.2 + 0.8 * speed ** 2) * stiff
        broad = (0.55 * speed + 1.4 * knock ** 2 + 0.35 * spread) * stiff
        floor = 0.12
        rms = math.sqrt(half ** 2 + one_x ** 2 + broad ** 2 + floor ** 2)
        return rms, half, one_x, broad

    def _electrical(self, cmd: Command, dt: float):
        """Alternator curve, battery SoC integration, bus voltage."""
        sp, h, st = self.spec, self.health, self.state
        if st.rpm > sp.alternator_cutin_rpm:
            frac = min(1.0, (st.rpm - sp.alternator_cutin_rpm) /
                       (sp.rated_rpm * 0.55 - sp.alternator_cutin_rpm))
        else:
            frac = 0.0
        p_alt_max = sp.alternator_rated_w * frac * h.alternator
        load_w = cmd.elec_load_w
        p_alt = min(p_alt_max, load_w + 90.0 * (1.0 - st.batt_soc))   # + charge

        net_w = p_alt - load_w
        cap_wh = sp.battery_capacity_ah * sp.battery_nominal_v * max(0.25, h.battery_soh)
        st.batt_soc = max(0.0, min(1.0, st.batt_soc + net_w * dt / 3600.0 / cap_wh))

        ocv = 11.6 + 2.0 * st.batt_soc
        r_int = sp.battery_r_internal_ohm / max(0.2, h.battery_soh)
        if p_alt_max > load_w:
            bus_v = 14.2 * (0.985 + 0.015 * h.alternator)
            cur = p_alt / max(bus_v, 1.0)
        else:
            i_batt = (load_w - p_alt) / max(ocv, 1.0)
            bus_v = ocv - i_batt * r_int * 6.0
            cur = -i_batt
        return bus_v, cur, load_w
