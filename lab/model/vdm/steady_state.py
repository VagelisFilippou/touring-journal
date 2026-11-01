"""Layer 2: steady-state cornering (constant radius, no drive/brake torque, no aero yet).

Per lateral acceleration ay:
  roll angle from roll stiffnesses (springs + ARBs, in series with tyres),
  lateral load transfer per axle = unsprung + geometric (roll centres) + elastic,
  wheel travel ±t/2·φ -> camber & toe from layer-1 kinematics (+ body roll),
  axle slip angles needed for the required axle forces,
  steer angle δ = L/R + α_f − α_r − rollsteer_f + rollsteer_r  -> understeer gradient.
Max ay = where either axle saturates; that axle tells the limit balance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import kinematics as K
from .params import Params
from .tyre import Tyre

G = 9.80665
G_STEEL = 79.3e9


def coil_rate(spring):
    if "rate" in spring and spring.rate is not None:
        return spring.rate
    D = spring.outer_D - spring.wire_d
    return G_STEEL * spring.wire_d ** 4 / (8 * D ** 3 * spring.coils_active)


def arb_roll_stiffness(arb, mr_arb, track):
    """Roll stiffness (Nm/rad) of an anti-roll bar, from diameter, lever and torsion length."""
    J = np.pi * arb.diameter ** 4 / 32
    k_bar = G_STEEL * J / arb.active_length                     # Nm/rad torsion
    return arb.efficiency * k_bar * (mr_arb * track / arb.lever_length) ** 2


@dataclass
class Axle:
    name: str
    sweep: K.Sweep
    track: float          # m, contact patch at static
    k_spring: float       # N/m at spring
    mr_spring: float
    k_wheel: float        # N/m at wheel (spring only)
    k_tyre: float
    K_roll_spring: float
    K_roll_arb: float
    rc: float             # m above ground
    m_unsprung: float     # per corner
    r_wheel: float

    @property
    def K_roll_susp(self):
        return self.K_roll_spring + self.K_roll_arb

    @property
    def K_roll_tyre(self):
        return self.k_tyre * self.track ** 2 / 2

    @property
    def K_roll(self):  # suspension in series with tyres
        return 1 / (1 / self.K_roll_susp + 1 / self.K_roll_tyre)

    def camber_toe(self, z):
        """Kinematic camber/toe (left-side convention) at wheel travel z."""
        s = self.sweep
        return np.interp(z, s.travel, s.camber), np.interp(z, s.travel, s.toe)


class Vehicle:
    def __init__(self, p: Params, driver=True, tyre_overrides: dict | None = None, sweeps=None):
        self.p = p
        L = p.chassis.wheelbase
        m = p.mass.kerb
        Wf = m * p.mass.front_fraction_kerb
        h = p.mass.cog_height_kerb
        x_cg = -L * (1 - p.mass.front_fraction_kerb)            # CoG x (front axle = 0)
        if driver:
            md, xd = p.mass.driver, p.mass.driver_position
            x_cg = (m * x_cg + md * xd[0]) / (m + md)
            h = (m * h + md * xd[2]) / (m + md)
            m += md
        self.m, self.L, self.h = m, L, h
        self.a = -x_cg                     # CoG to front axle
        self.b = L - self.a
        self.tyre = Tyre(p.wheels.tyre_model, tyre_overrides)
        sf, sr = sweeps if sweeps else (K.MacPherson(p).sweep(), K.SemiTrailingArm(p).sweep())
        mf, mr = p.mass.unsprung_front_corner, p.mass.unsprung_rear_corner
        self.ms = m - 2 * mf - 2 * mr
        # sprung CoG (unsprung masses at wheel-centre height on their axles)
        r_f, r_r = p.suspension.front.loaded_radius, p.suspension.rear.loaded_radius
        self.hs = (m * h - 2 * mf * r_f - 2 * mr * r_r) / self.ms
        self.as_ = (m * self.a - 2 * mr * L) / self.ms           # sprung CoG to front axle
        self.axles = {}
        for name, s, susp, mu, r in (("front", sf, p.suspension.front, mf, r_f),
                                     ("rear", sr, p.suspension.rear, mr, r_r)):
            st = s.at(0.0)
            t = 2 * st["half_track"]
            k_s = coil_rate(susp.spring)
            k_w = k_s * st["spring_mr"] ** 2
            self.axles[name] = Axle(name, s, t, k_s, st["spring_mr"], k_w, self.tyre.kz,
                                    k_w * t ** 2 / 2, arb_roll_stiffness(susp.arb, st["arb_mr"], t),
                                    st["rc_height"], mu, r)

    # ------------------------------------------------------------------------------------
    @property
    def static_axle_loads(self):
        return self.m * G * self.b / self.L, self.m * G * self.a / self.L

    def roll_gradient(self):
        F, R = self.axles["front"], self.axles["rear"]
        rc_cg = F.rc + (R.rc - F.rc) * self.as_ / self.L
        h_arm = self.hs - rc_cg
        return self.ms * h_arm / (F.K_roll + R.K_roll - self.ms * G * h_arm), h_arm   # rad per (m/s²)

    def load_transfer(self, ay):
        """ΔFz per wheel (outer +, inner −) for each axle, and the parts."""
        F, R = self.axles["front"], self.axles["rear"]
        dphi, h_arm = self.roll_gradient()
        phi = dphi * ay
        out = {}
        for ax, ms_share in ((F, self.ms * (self.L - self.as_) / self.L), (R, self.ms * self.as_ / self.L)):
            uns = 2 * ax.m_unsprung * ay * ax.r_wheel / ax.track
            geo = ms_share * ay * ax.rc / ax.track
            el = ax.K_roll * phi / ax.track
            out[ax.name] = dict(total=uns + geo + el, unsprung=uns, geometric=geo, elastic=el)
        return out, phi

    def axle_state(self, ax: Axle, ay, phi, dFz, Fz_static_axle):
        z = ax.track / 2 * phi                                # outer wheel bump, inner rebound
        cam_o, toe_o = ax.camber_toe(+z)
        cam_i, toe_i = ax.camber_toe(-z)
        # camber leaning toward turn centre (outer = right wheel, inner = left wheel)
        g_out = -(cam_o + phi)
        g_in = cam_i - phi
        roll_steer = (toe_o - toe_i) / 2                      # + = axle steers into the turn
        Fz_o, Fz_i = Fz_static_axle / 2 + dFz, Fz_static_axle / 2 - dFz
        return Fz_o, Fz_i, g_out, g_in, roll_steer

    def solve(self, ay):
        lt, phi = self.load_transfer(ay)
        Wf, Wr = self.static_axle_loads
        req = {"front": self.m * ay * self.b / self.L, "rear": self.m * ay * self.a / self.L}
        res = {"ay": ay, "phi": phi}
        for name, W in (("front", Wf), ("rear", Wr)):
            ax = self.axles[name]
            Fz_o, Fz_i, g_o, g_i, rs = self.axle_state(ax, ay, phi, lt[name]["total"], W)
            Fy = lambda al: self.tyre.Fy(al, Fz_o, g_o) + self.tyre.Fy(al, Fz_i, g_i)
            grid = np.linspace(-0.05, 0.35, 801)
            fy = Fy(grid)
            al_pk, cap = grid[np.argmax(fy)], fy.max()
            res[name] = dict(Fz_out=Fz_o, Fz_in=Fz_i, cam_out=g_o, cam_in=g_i, roll_steer=rs,
                             capacity=cap, required=req[name], lt=lt[name])
            if req[name] > cap or Fz_i < 0:
                res[name]["alpha"] = np.nan
            else:
                res[name]["alpha"] = brentq(lambda al: Fy(al) - req[name], -0.05, al_pk)
        res["delta_kin"] = res["front"]["alpha"] - res["rear"]["alpha"] \
            - res["front"]["roll_steer"] + res["rear"]["roll_steer"]      # excl. L/R
        return res

    def max_ay(self):
        def margin(ay):
            r = self.solve(ay)
            return min(r["front"]["capacity"] - r["front"]["required"],
                       r["rear"]["capacity"] - r["rear"]["required"],
                       min(r["front"]["Fz_in"], r["rear"]["Fz_in"]))
        ay = brentq(margin, 0.5, 2.0 * G, xtol=1e-3)
        r = self.solve(ay * 0.999)
        lim = "front (understeer)" if (r["front"]["capacity"] - r["front"]["required"]) < \
            (r["rear"]["capacity"] - r["rear"]["required"]) else "rear (oversteer)"
        if min(r["front"]["Fz_in"], r["rear"]["Fz_in"]) < 1:
            lim += " / inner wheel lift"
        return ay, lim

    def understeer_gradient(self, ay1=0.5, ay2=3.0):
        """deg of road-wheel steer per g (beyond Ackermann), linear range."""
        d1, d2 = self.solve(ay1)["delta_kin"], self.solve(ay2)["delta_kin"]
        return np.degrees((d2 - d1) / (ay2 - ay1) * G)

    # ------------------------------------------------------------------------------------
    def summary(self) -> dict:
        F, R = self.axles["front"], self.axles["rear"]
        Wf, Wr = self.static_axle_loads
        dphi, h_arm = self.roll_gradient()
        aymax, lim = self.max_ay()
        lt, _ = self.load_transfer(G * 0.5)
        lltd = lt["front"]["total"] * F.track / (lt["front"]["total"] * F.track + lt["rear"]["total"] * R.track)
        out = {"mass_kg": self.m, "front_weight_%": 100 * Wf / (Wf + Wr), "cog_height_mm": self.h * 1e3,
               "roll_gradient_deg_per_g": np.degrees(dphi * G), "roll_arm_mm": h_arm * 1e3,
               "LLTD_front_%": 100 * lltd,
               "roll_stiffness_front_%": 100 * F.K_roll_susp / (F.K_roll_susp + R.K_roll_susp),
               "understeer_grad_deg_per_g": self.understeer_gradient(),
               "max_ay_g": aymax / G, "limit": lim}
        for ax, W in ((F, Wf), (R, Wr)):
            m_corner = W / 2 / G - ax.m_unsprung
            k_ride = ax.k_wheel * ax.k_tyre / (ax.k_wheel + ax.k_tyre)
            out[ax.name] = {"spring_N/mm": ax.k_spring / 1e3, "MR": ax.mr_spring,
                            "wheel_rate_N/mm": ax.k_wheel / 1e3,
                            "ride_freq_Hz": np.sqrt(k_ride / m_corner) / (2 * np.pi),
                            "K_roll_spring_Nm/deg": ax.K_roll_spring * np.pi / 180,
                            "K_roll_arb_Nm/deg": ax.K_roll_arb * np.pi / 180,
                            "roll_centre_mm": ax.rc * 1e3, "track_mm": ax.track * 1e3}
        return out
