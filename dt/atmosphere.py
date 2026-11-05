"""International Standard Atmosphere with off-standard temperature deviation.

MALE UAV missions routinely span sea level to FL250 and ISA+35 hot-and-high
conditions, both of which dominate turbocharger and cooling behaviour, so the
atmosphere is a first-class model input rather than a constant.
"""

import math
from dataclasses import dataclass

from .config import R_AIR, G0

T0 = 288.15      # K
P0 = 101325.0    # Pa
LAPSE = -0.0065  # K/m, troposphere
TROPOPAUSE_M = 11000.0
T_TROPO = T0 + LAPSE * TROPOPAUSE_M      # 216.65 K


@dataclass
class AtmState:
    altitude_m: float
    press_pa: float
    temp_k: float
    density: float
    isa_dev_k: float
    rel_humidity: float = 0.0

    @property
    def press_kpa(self) -> float:
        return self.press_pa / 1000.0

    @property
    def temp_c(self) -> float:
        return self.temp_k - 273.15

    @property
    def sigma(self) -> float:
        """Density ratio to sea-level standard - drives prop load and cooling."""
        return self.density / 1.225


def isa(altitude_m: float, isa_dev_k: float = 0.0, rel_humidity: float = 0.0) -> AtmState:
    """Return atmospheric state at geopotential altitude with an ISA deviation."""
    if altitude_m <= TROPOPAUSE_M:
        t_std = T0 + LAPSE * altitude_m
        p = P0 * (t_std / T0) ** (-G0 / (LAPSE * R_AIR))
    else:
        t_std = T_TROPO
        p_tropo = P0 * (T_TROPO / T0) ** (-G0 / (LAPSE * R_AIR))
        p = p_tropo * math.exp(-G0 * (altitude_m - TROPOPAUSE_M) / (R_AIR * T_TROPO))

    t = t_std + isa_dev_k
    # Humid air is less dense: subtract the water-vapour partial pressure share.
    if rel_humidity > 0.0:
        p_sat = 610.94 * math.exp(17.625 * (t - 273.15) / (t - 273.15 + 243.04))
        pv = rel_humidity * p_sat
        rho = (p - pv) / (R_AIR * t) + pv / (461.5 * t)
    else:
        rho = p / (R_AIR * t)
    return AtmState(altitude_m, p, t, rho, isa_dev_k, rel_humidity)


def ram_recovery(atm: AtmState, tas_ms: float, efficiency: float = 0.85) -> AtmState:
    """Total (stagnation) conditions at the induction and cooling inlets.

    At MALE cruise speeds (~60-100 kt TAS) ram rise is small but not negligible
    for cooling mass flow, which is what the CHT model actually consumes.
    """
    if tas_ms <= 0.0:
        return atm
    dt = efficiency * tas_ms ** 2 / (2 * 1005.0)
    t_tot = atm.temp_k + dt
    p_tot = atm.press_pa * (t_tot / atm.temp_k) ** 3.5
    rho = p_tot / (R_AIR * t_tot)
    return AtmState(atm.altitude_m, p_tot, t_tot, rho, atm.isa_dev_k, atm.rel_humidity)


def cooling_mass_flow(atm: AtmState, tas_ms: float, rpm: float, rated_rpm: float,
                      duct_area_m2: float = 0.028) -> float:
    """Cooling-air mass flow through the cowl duct: ram component plus the
    propeller-induced component that keeps the engine alive at low airspeed."""
    v_induced = 6.0 * (rpm / max(rated_rpm, 1.0))
    v_eff = math.sqrt(tas_ms ** 2 + v_induced ** 2)
    return max(0.02, atm.density * duct_area_m2 * v_eff)
