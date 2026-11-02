"""Layer 5b: quasi-steady-state lap / route simulation (OpenLAP-style point mass + our own limits).

Per point along the road: cornering speed limit from the steady-state max lateral acceleration
(layer 2, setup dependent), forward pass at full throttle (engine, gearing, rear traction with the
LSD/open diff and load transfer, aero, rolling resistance, gradient, rotating inertia), backward pass
at the braking limit, friction ellipse between lateral and longitudinal use.
Assumptions: ideal instant gear changes, no driver margin, dry road, no aero lift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .powertrain import G, Powertrain
from .steady_state import Vehicle


@dataclass
class Result:
    s: np.ndarray
    v: np.ndarray
    t: np.ndarray
    gear: np.ndarray
    rpm: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    limit: np.ndarray          # 'corner' / 'power' / 'traction' / 'brake'

    @property
    def time(self):
        return float(self.t[-1])


class LapSim:
    def __init__(self, p, vehicle: Vehicle | None = None, tyre_overrides=None, brake_eff=0.92, driver_margin=1.0):
        self.p = p
        self.veh = vehicle or Vehicle(p, tyre_overrides=tyre_overrides)
        self.pt = Powertrain(p, tyre_overrides)
        self.m = self.veh.m
        self.ay_max = self.veh.max_ay()[0] * driver_margin
        self.ax_brake = self.pt.mu_x * G * brake_eff * driver_margin
        self.h, self.L, self.a = self.veh.h, self.veh.L, self.veh.a
        lt, _ = self.veh.load_transfer(1.0)                      # rear ΔFz per wheel per (m/s²)
        self.dFz_rear_per_ay = lt["rear"]["total"]
        self.v_top = self.pt.top_speed(self.m)

    def _accel(self, v, ay, grade, ax_guess=0.0):
        """Forward acceleration available at speed v with lateral acceleration ay on a gradient."""
        m, pt = self.m, self.pt
        th = np.arctan(grade)
        F_eng, g = pt.drive_force(v)
        # rear axle load: static + longitudinal transfer (accel + gradient) — 2 fixed-point iterations
        ax = ax_guess
        for _ in range(2):
            Fz_r = m * G * np.cos(th) * self.a / self.L + m * (ax + G * np.sin(th)) * self.h / self.L
            use = min(abs(ay) / self.ay_max, 1.0)
            F_trac = pt.traction_limit(m, Fz_r, self.dFz_rear_per_ay * abs(ay), use)
            F = min(F_eng, F_trac)
            ax = (F - pt.resistance(v, m, grade)) / pt.mass_factor(m, g)
        lim = "power" if F_eng <= F_trac else "traction"
        return ax, g, lim

    def _brake(self, v, ay, grade):
        k = np.sqrt(max(0.0, 1 - (ay / self.ay_max) ** 2))
        res = self.pt.resistance(v, self.m, grade)
        return self.ax_brake * k + res / self.m                 # deceleration magnitude

    def run(self, track, v0=0.0, v_end=None):
        s, kap, grd = track.s, track.kappa, track.grade
        n = len(s)
        ds = np.diff(s)
        v_c = np.minimum(np.sqrt(self.ay_max / np.maximum(np.abs(kap), 1e-6)), self.v_top)
        # forward pass
        vf = np.empty(n); vf[0] = min(v0, v_c[0])
        gear = np.zeros(n, int); lim = np.empty(n, object); axf = np.zeros(n)
        ax_prev = 0.0
        for i in range(n - 1):
            ay = vf[i] ** 2 * kap[i]
            ax, g, l = self._accel(vf[i], ay, grd[i], ax_prev)
            ax_prev = ax
            gear[i], lim[i], axf[i] = g, l, ax
            vf[i + 1] = min(np.sqrt(max(vf[i] ** 2 + 2 * ax * ds[i], 0.01)), v_c[i + 1])
            if vf[i + 1] == v_c[i + 1]:
                lim[i + 1] = "corner"
        # backward pass
        vb = np.empty(n); vb[-1] = v_c[-1] if v_end is None else min(v_end, v_c[-1])
        for i in range(n - 1, 0, -1):
            ay = vb[i] ** 2 * kap[i]
            d = self._brake(vb[i], ay, grd[i])
            vb[i - 1] = min(np.sqrt(vb[i] ** 2 + 2 * d * ds[i - 1]), v_c[i - 1])
        v = np.minimum(vf, vb)
        lim = np.where(vb < vf - 1e-6, "brake", lim)
        lim = np.where(v >= v_c * 0.995, "corner", lim)          # at the cornering limit (apexes, bends)
        lim[0] = lim[0] or "power"
        vm = 0.5 * (v[:-1] + v[1:])
        t = np.r_[0, np.cumsum(ds / np.maximum(vm, 0.3))]
        ax = np.gradient(v * v / 2, s)
        gear = np.array([self.pt.drive_force(max(vv, 0.5))[1] for vv in v])
        rpm = np.array([self.pt.engine_speed(vv, g) for vv, g in zip(v, gear)]) * 60 / (2 * np.pi)
        return Result(s, v, t, gear + 1, np.maximum(rpm, self.pt.w_idle * 60 / (2 * np.pi)), ax, v * v * kap, lim)

    def accel_times(self, speeds_kmh=(60, 80, 100, 120, 140), dist_m=(400, 1000)):
        from .roads import straight
        r = self.run(straight(1500.0), v0=0.0, v_end=1e3)
        out = {f"0-{k} km/h [s]": float(np.interp(k / 3.6, r.v, r.t)) if r.v.max() >= k / 3.6 else None for k in speeds_kmh}
        out.update({f"{d} m standing [s]": float(np.interp(d, r.s, r.t)) for d in dist_m})
        out["top speed [km/h]"] = self.v_top * 3.6
        return out
