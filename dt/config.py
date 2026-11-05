"""
Engine and platform configuration.

The reference powerplant is a turbocharged, liquid/air-cooled, four-cylinder
horizontally-opposed four-stroke of the class used on MALE UAVs (Rotax 914 /
912-iS family: 1211 cc, ~84 kW at 5800 rpm, 2.4286:1 reduction gearbox).
Every number below is a configuration value, not a hard-coded constant, so the
same twin can be re-pointed at a different engine by editing one dataclass.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List

R_AIR = 287.05          # J/(kg K)
R_EXH = 290.0           # J/(kg K), burned-gas approximation
CP_AIR = 1005.0         # J/(kg K)
CP_EXH = 1150.0         # J/(kg K)
GAMMA_AIR = 1.40
LHV_AVGAS = 43.5e6      # J/kg, AVGAS 100LL / MOGAS
AFR_STOICH = 14.7
G0 = 9.80665


@dataclass
class EngineSpec:
    name: str = "MALE-UAV Turbocharged Boxer-4 (914-class)"
    n_cyl: int = 4
    displacement_l: float = 1.211          # total, litres
    bore_mm: float = 79.5
    stroke_mm: float = 61.0
    compression_ratio: float = 9.0
    strokes: int = 4
    rated_power_kw: float = 84.5
    rated_rpm: float = 5800.0
    max_continuous_rpm: float = 5500.0
    idle_rpm: float = 1400.0
    redline_rpm: float = 5900.0
    gearbox_ratio: float = 2.4286          # crank : prop

    # --- Turbocharger / air path -------------------------------------------
    boost_target_kpa: float = 132.0        # absolute MAP held by wastegate
    boost_takeoff_kpa: float = 145.0       # 5 min limit
    critical_altitude_m: float = 4900.0    # above this the wastegate is shut
    turbo_pressure_ratio_max: float = 2.6
    intercooler_effectiveness: float = 0.62

    # --- Thermal ------------------------------------------------------------
    cht_mass_kg: float = 1.9               # lumped head+barrel thermal mass/cyl
    cht_cp: float = 900.0                  # J/(kg K), aluminium alloy
    cht_ua_nominal: float = 460.0          # W/K per cylinder at reference flow
    oil_mass_kg: float = 3.0
    oil_cp: float = 2000.0
    oil_ua_nominal: float = 95.0
    coolant_mass_kg: float = 2.4
    coolant_cp: float = 3600.0             # 50/50 glycol
    coolant_ua_nominal: float = 700.0

    # --- Lubrication --------------------------------------------------------
    oil_press_gain_kpa_per_krpm: float = 105.0
    oil_press_bypass_kpa: float = 500.0    # relief valve setting (5 bar)
    oil_visc_ref_temp_c: float = 90.0

    # --- Electrical ---------------------------------------------------------
    alternator_rated_w: float = 250.0
    alternator_cutin_rpm: float = 1800.0
    battery_capacity_ah: float = 12.0
    battery_nominal_v: float = 12.8
    battery_r_internal_ohm: float = 0.018

    # --- Propeller load (absorbed torque = k * omega^2 * rho/rho0) ----------
    prop_k: float = 0.000377               # N m s^2 referred to the crank, at rated
    prop_pitch_min: float = 0.50           # coarse/fine pitch stops as a
    prop_pitch_max: float = 2.20           # fraction of the rated-point value


@dataclass
class Limits:
    """Certified operating limits used by the threshold layer and the advisory
    engine. The DT is predictive, but these remain the last-ditch safety net."""
    cht_c: float = 135.0
    cht_caution_c: float = 120.0
    egt_c: float = 900.0
    egt_caution_c: float = 850.0
    oil_temp_c: float = 130.0
    oil_temp_caution_c: float = 120.0
    oil_temp_min_c: float = 50.0
    oil_press_kpa_min: float = 80.0
    oil_press_kpa_max: float = 520.0
    coolant_temp_c: float = 120.0
    rpm_max: float = 5900.0
    map_kpa_max: float = 150.0
    vib_rms_g: float = 2.5
    vib_caution_g: float = 1.5
    bus_voltage_min: float = 12.0
    bus_voltage_max: float = 15.2


@dataclass
class SensorSpec:
    """Per-channel measurement model: noise sigma (engineering units), bias,
    drift rate per flight hour, resolution, and CAN update rate."""
    name: str
    unit: str
    sigma: float
    resolution: float
    rate_hz: float
    drift_per_hr: float = 0.0
    range_lo: float = -1e9
    range_hi: float = 1e9


def default_sensor_suite(n_cyl: int = 4) -> Dict[str, SensorSpec]:
    s: Dict[str, SensorSpec] = {
        "rpm":         SensorSpec("rpm", "rpm", 4.0, 1.0, 50.0, 0.0, 0, 7000),
        "map_kpa":     SensorSpec("map_kpa", "kPa", 0.6, 0.25, 50.0, 0.05, 10, 200),
        "iat_c":       SensorSpec("iat_c", "degC", 0.5, 0.25, 10.0, 0.02, -60, 200),
        "oil_press_kpa": SensorSpec("oil_press_kpa", "kPa", 4.0, 1.0, 20.0, 0.30, 0, 1000),
        "oil_temp_c":  SensorSpec("oil_temp_c", "degC", 0.4, 0.25, 5.0, 0.04, -60, 200),
        "coolant_temp_c": SensorSpec("coolant_temp_c", "degC", 0.4, 0.25, 5.0, 0.04, -60, 200),
        "fuel_flow_lph": SensorSpec("fuel_flow_lph", "L/h", 0.12, 0.01, 10.0, 0.02, 0, 60),
        "vib_rms_g":   SensorSpec("vib_rms_g", "g", 0.04, 0.01, 20.0, 0.0, 0, 20),
        # Order-resolved bands and knock level are computed in the vibration
        # monitoring unit from a crank-synchronous tacho and published on the
        # bus, rather than being reconstructed downstream from broadband RMS.
        "vib_half_g":  SensorSpec("vib_half_g", "g", 0.020, 0.005, 20.0, 0.0, 0, 20),
        "vib_1x_g":    SensorSpec("vib_1x_g", "g", 0.020, 0.005, 20.0, 0.0, 0, 20),
        "vib_broad_g": SensorSpec("vib_broad_g", "g", 0.025, 0.005, 20.0, 0.0, 0, 20),
        "knock_level": SensorSpec("knock_level", "%", 1.2, 0.5, 20.0, 0.0, 0, 100),
        # Thermostat / radiator shutter position feedback.
        "coolant_valve_pct": SensorSpec("coolant_valve_pct", "%", 0.8, 0.4, 5.0, 0.0, 0, 100),
        "bus_v":       SensorSpec("bus_v", "V", 0.03, 0.01, 5.0, 0.005, 0, 20),
        "alt_current_a": SensorSpec("alt_current_a", "A", 0.15, 0.05, 5.0, 0.01, -50, 50),
        "throttle_pct": SensorSpec("throttle_pct", "%", 0.3, 0.1, 20.0, 0.0, 0, 100),
        "prop_load_pct": SensorSpec("prop_load_pct", "%", 0.5, 0.2, 20.0, 0.0, 0, 100),
        "amb_press_kpa": SensorSpec("amb_press_kpa", "kPa", 0.15, 0.05, 5.0, 0.02, 5, 110),
        "oat_c":       SensorSpec("oat_c", "degC", 0.4, 0.25, 5.0, 0.02, -80, 70),
    }
    for i in range(1, n_cyl + 1):
        s[f"cht{i}_c"] = SensorSpec(f"cht{i}_c", "degC", 1.1, 0.5, 10.0, 0.09, -60, 400)
        s[f"egt{i}_c"] = SensorSpec(f"egt{i}_c", "degC", 5.5, 1.0, 20.0, 0.55, 0, 1200)
    return s


@dataclass
class TwinConfig:
    engine: EngineSpec = field(default_factory=EngineSpec)
    limits: Limits = field(default_factory=Limits)
    sim_dt: float = 0.05                   # 20 Hz physics / DT sync rate
    can_bitrate: int = 500_000
    can_channel: str = "vcan0"
    fdr_path: str = "data/fdr.sqlite"
    model_path: str = "data/models.json"

    @property
    def sensors(self) -> Dict[str, SensorSpec]:
        return default_sensor_suite(self.engine.n_cyl)

    def to_dict(self) -> dict:
        return {"engine": asdict(self.engine), "limits": asdict(self.limits),
                "sim_dt": self.sim_dt}


# Subsystems the health layer scores independently.
SUBSYSTEMS: List[str] = [
    "combustion", "air_path", "fuel_injection", "cooling",
    "lubrication", "mechanical", "electrical", "sensing",
]
