"""Layer 5a: powertrain and longitudinal forces (RWD, M10 + Getrag + ZF LSD).

All inputs from the parameter file (engine.torque curve, driveline ratios/efficiency/LSD, inertias,
aero, tyre rolling circumference). SI units throughout (rad/s, N, m).
"""
from __future__ import annotations

import numpy as np

from .params import Params, load_tyre

G = 9.80665


class Powertrain:
    def __init__(self, p: Params, tyre_overrides=None):
        e, d = p.engine, p.driveline
        self.w_curve = np.asarray(e.torque_rpm, float)          # rad/s
        self.T_curve = np.asarray(e.torque, float)              # Nm
        self.w_max = e.rev_limit
        self.w_idle = e.idle
        self.gears = np.asarray(d.gear_ratios, float)
        self.fd = d.final_drive
        self.eta = d.efficiency
        t = load_tyre(p.wheels.tyre_model, tyre_overrides)
        self.r = t.geometry.rolling_circumference / (2 * np.pi)
        self.f_rr = t.geometry.rolling_resistance
        self.mu_x = t.longitudinal.mu_x0
        self.lsd = d.diff_type == "clutch_lsd"
        L = d.lsd_lock_ratio if self.lsd else 0.0
        self.tbr = (1 + L) / (1 - L)                             # torque bias ratio (40 % -> 2.33)
        # rotating inertia reflected to the wheels, excluding the engine side (per car)
        I_w = 4 * p.wheels.wheel_spin_inertia + 2 * p.brakes.rotor_inertia_front + 2 * p.brakes.drum_inertia_rear
        self.I_wheel_side = I_w + d.driveline_inertia_at_wheels
        self.I_engine = e.inertia
        a = p.aero
        self.CdA, self.rho = a.Cd * a.frontal_area, a.air_density

    def torque(self, w):
        """Full-load engine torque (Nm) at crank speed w (rad/s); 0 above the rev limit."""
        T = np.interp(w, self.w_curve, self.T_curve, left=self.T_curve[0])
        return np.where(w > self.w_max * 1.0001, 0.0, T)

    def overall(self, g):
        return self.gears[g] * self.fd

    def engine_speed(self, v, g):
        return v / self.r * self.overall(g)

    def drive_force(self, v):
        """Best gear and wheel force (N) at speed v (full throttle, ideal shifting)."""
        best_F, best_g = 0.0, len(self.gears) - 1
        for g in range(len(self.gears)):
            w = max(self.engine_speed(v, g), self.w_idle)       # clutch slip below idle
            if w > self.w_max:
                continue
            F = self.torque(w) * self.overall(g) * self.eta / self.r
            if F > best_F:
                best_F, best_g = F, g
        return best_F, best_g

    def mass_factor(self, m, g):
        """Effective mass incl. rotating inertia in gear g."""
        return m + (self.I_wheel_side + self.I_engine * self.overall(g) ** 2) / self.r ** 2

    def resistance(self, v, m, grade):
        th = np.arctan(grade)
        return 0.5 * self.rho * self.CdA * v * v + self.f_rr * m * G * np.cos(th) + m * G * np.sin(th)

    def traction_limit(self, m, Fz_rear, dFz_rear, ay_use):
        """Max drive force at the rear axle: open diff = 2x inner wheel; LSD = inner x (1 + TBR), capped by both wheels.
        ay_use = lateral utilisation of the rear tyres (0..1), friction ellipse per wheel."""
        k = np.sqrt(max(0.0, 1 - ay_use ** 2))
        Fi = self.mu_x * max(Fz_rear / 2 - dFz_rear, 0.0) * k
        Fo = self.mu_x * (Fz_rear / 2 + dFz_rear) * k
        return min(Fi * (1 + self.tbr), Fi + Fo) if self.lsd else 2 * Fi

    def top_speed(self, m):
        vs = np.linspace(5, 70, 1301)
        surplus = np.array([self.drive_force(v)[0] - self.resistance(v, m, 0.0) for v in vs])
        i = np.where(surplus > 0)[0]
        return float(vs[i[-1]]) if len(i) else 0.0


def gearing_table(p, pt: Powertrain | None = None) -> dict:
    """Gear ratios, overall ratios, km/h per 1000 rpm and speed at the rev limit, plus the engine curve."""
    pt = pt or Powertrain(p)
    rows = []
    for g, ratio in enumerate(pt.gears):
        ov = ratio * pt.fd
        kmh_1000 = 1000 * 2 * np.pi / 60 / ov * pt.r * 3.6
        rows.append({"gear": g + 1, "ratio": float(ratio), "overall": float(ov), "kmh_per_1000": float(kmh_1000),
                     "v_at_limit": float(kmh_1000 * pt.w_max * 60 / (2 * np.pi) / 1000)})
    rpm = pt.w_curve * 60 / (2 * np.pi)
    lf = p.leaf("engine.torque")
    return {"gearbox": str(p.driveline.gearbox), "final_drive": float(pt.fd), "rows": rows,
            "rev_limit": float(pt.w_max * 60 / (2 * np.pi)), "tyre_circumference_m": float(2 * np.pi * pt.r),
            "engine": str(p.engine.type), "rpm": rpm.round(0).tolist(), "torque_Nm": pt.T_curve.round(1).tolist(),
            "power_PS": (pt.T_curve * pt.w_curve / 735.5).round(1).tolist(),
            "torque_conf": lf.get("conf"), "torque_note": lf.get("note", "")}
