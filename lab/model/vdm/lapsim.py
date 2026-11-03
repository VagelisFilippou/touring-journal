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
        # g-g envelope from the four-wheel model (longitudinal + lateral load transfer, combined grip per tyre,
        # rear drive, brake balance from the hardware). brake_eff kept for API compatibility (unused).
        gx, gy = self.veh.gg_envelope()
        self.gg_ax, self.gg_ay = gx, gy * driver_margin
        self.ay_max = float(np.interp(0.0, gx, self.gg_ay))
        self.ay_peak = float(self.gg_ay.max())                   # light braking can add a little grip
        neg, pos = gx <= 0, gx >= 0
        self._brk = (self.gg_ay[neg], -gx[neg] * driver_margin)  # ay ascending from full braking to coasting
        self._acc = (self.gg_ay[pos][::-1], gx[pos][::-1] * driver_margin)
        self.ax_brake = float(-gx[0] * driver_margin)
        self.h, self.L, self.a = self.veh.h, self.veh.L, self.veh.a
        lt, _ = self.veh.load_transfer(1.0)                      # rear ΔFz per wheel per (m/s²)
        self.dFz_rear_per_ay = lt["rear"]["total"]
        self.v_top = self.pt.top_speed(self.m)
        if "driver" in p:                                         # shift rules from the parameter file
            self.DOWNSHIFT_RPM = p.driver.downshift_rpm * 60 / (2 * np.pi)
            self.MIN_STINT_S = p.driver.min_gear_time

    DOWNSHIFT_RPM = 5500.0   # under braking: change down one gear at a time once the lower gear lands at or below this

    MIN_STINT_S = 1.5        # no gear held shorter than this (a short stint takes the higher neighbouring gear)

    def _gears(self, v, lim, s=None):
        """Gear per point (0-based). Driving: the gear with the most drive force (ideal shifting).
        Braking: hold the gear, change down one at a time when the lower gear is at or below DOWNSHIFT_RPM
        (heel-and-toe), and never below the gear needed for the drive that follows."""
        best = np.array([self.pt.drive_force(max(vv, 0.5))[1] for vv in v])
        g = best.copy()
        w_dn = self.DOWNSHIFT_RPM * 2 * np.pi / 60
        for i in range(1, len(v)):
            if lim[i] == "brake":
                cur = g[i - 1]
                if cur > best[i] and self.pt.engine_speed(v[i], cur - 1) <= w_dn:
                    cur -= 1
                g[i] = max(cur, best[i]) if cur < best[i] else cur
        if s is not None:                                      # merge stints shorter than MIN_STINT_S into the higher neighbour
            dt = np.r_[np.diff(s) / np.maximum(0.5 * (v[:-1] + v[1:]), 0.5), 0.0]
            for _ in range(3):
                edges = np.r_[0, np.where(np.diff(g) != 0)[0] + 1, len(g)]
                changed = False
                for a_, b_ in zip(edges[:-1], edges[1:]):
                    if a_ == 0 or b_ == len(g) or dt[a_:b_].sum() >= self.MIN_STINT_S:
                        continue
                    cand = max(g[a_ - 1], g[b_])
                    if cand == g[a_] or any(self.pt.engine_speed(v[k], cand) > self.pt.w_max * 1.002 for k in range(a_, b_)):
                        continue                                # would over-rev (or no change): keep the short stint
                    g[a_:b_] = cand
                    changed = True
                if not changed:
                    break
        return g

    def _gg_accel(self, ay):
        """Tyre-limited forward acceleration at lateral acceleration ay (drive grip side of the envelope)."""
        a = abs(ay)
        if a >= self.ay_max:
            return 0.0
        return float(np.interp(a, *self._acc))

    def _gg_brake(self, ay):
        """Tyre-limited deceleration magnitude at lateral acceleration ay (braking side of the envelope)."""
        a = abs(ay)
        if a >= self.ay_peak:
            return 0.0
        ays, dec = self._brk
        i = int(np.argmax(ays))                                  # use the monotonic part (full braking -> peak)
        return float(np.interp(a, ays[: i + 1], dec[: i + 1]))

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
            F_tyre = m * self._gg_accel(ay)                      # combined grip of the driven tyres (g-g envelope)
            F = min(F_eng, F_trac, F_tyre)
            ax = (F - pt.resistance(v, m, grade)) / pt.mass_factor(m, g)
        lim = "power" if F_eng <= min(F_trac, F_tyre) else "traction"
        return ax, g, lim

    def _brake(self, v, ay, grade):
        res = self.pt.resistance(v, self.m, grade)
        return self._gg_brake(ay) + res / self.m                # deceleration magnitude

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
        gear = self._gears(v, lim, s)
        rpm = np.array([self.pt.engine_speed(vv, g) for vv, g in zip(v, gear)]) * 60 / (2 * np.pi)
        return Result(s, v, t, gear + 1, np.maximum(rpm, self.pt.w_idle * 60 / (2 * np.pi)), ax, v * v * kap, lim)

    def accel_times(self, speeds_kmh=(60, 80, 100, 120, 140), dist_m=(400, 1000)):
        from .roads import straight
        r = self.run(straight(1500.0), v0=0.0, v_end=1e3)
        out = {f"0-{k} km/h [s]": float(np.interp(k / 3.6, r.v, r.t)) if r.v.max() >= k / 3.6 else None for k in speeds_kmh}
        out.update({f"{d} m standing [s]": float(np.interp(d, r.s, r.t)) for d in dist_m})
        out["top speed [km/h]"] = self.v_top * 3.6
        return out
