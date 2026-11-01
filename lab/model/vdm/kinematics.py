"""Layer 1: 3D suspension kinematics (left side; right side mirrors).

Front: MacPherson strut. Lower arm + bolted tension strut = rigid A-arm rotating about the axis
       lca_inner -> tension_strut_body. The upright (strut tube + spindle) is a rigid body that
       hangs on the ball joint; its strut axis must pass through the strut top mount; the track
       rod has constant length. One DOF (heave); steering via a lateral shift of the inner tie rod.
Rear:  semi-trailing arm, rigid, rotating about the pivot axis outer -> inner. One DOF.

All positions in m in the body frame (docs/conventions.md). Wheel travel z > 0 = bump.
Outputs per travel step: camber, toe, wheel centre & contact patch, spring/damper lengths,
motion ratios, kinematic roll centre (front-view, from contact patch path), side-view path slopes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares

from .params import Params

EZ = np.array([0.0, 0.0, 1.0])
TOE_REF_DIAM = 0.368   # m: BMW 1.5 mm total toe = 0°14' -> reference diameter for mm<->deg


def rot(axis, ang):
    """Rodrigues rotation matrix."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def rotvec(v):
    a = np.linalg.norm(v)
    return np.eye(3) if a < 1e-14 else rot(v / a, a)


def spin_axis(camber, toe):
    """Left-wheel spin axis (pointing outboard) from camber (+ top out) and toe (+ in)."""
    return np.array([np.sin(toe) * np.cos(camber), np.cos(toe) * np.cos(camber), -np.sin(camber)])


def camber_toe(a):
    return np.arcsin(-a[2]), np.arctan2(a[0], a[1])


def contact_point(wc, a, r):
    d = -EZ + a[2] * a
    return wc + r * d / np.linalg.norm(d)


def toe_to_rad(total):
    """Total toe (m, BMW convention measured at rim) -> per-wheel toe angle (rad)."""
    return np.arctan(total / 2 / TOE_REF_DIAM)


@dataclass
class Sweep:
    """Results vs wheel travel (relative to the static position). SI units, angles in rad."""
    travel: np.ndarray
    camber: np.ndarray
    toe: np.ndarray
    wc: np.ndarray           # (n,3)
    cp: np.ndarray           # (n,3)
    spring_len: np.ndarray
    damper_len: np.ndarray
    arb_z: np.ndarray        # vertical position of ARB link point
    ground_z: float          # ground height in body frame
    extra: dict = field(default_factory=dict)

    def _d(self, y):
        return np.gradient(y, self.travel)

    @property
    def spring_mr(self):  # spring compression per wheel bump
        return -self._d(self.spring_len)

    @property
    def damper_mr(self):
        return -self._d(self.damper_len)

    @property
    def arb_mr(self):
        return self._d(self.arb_z)

    @property
    def rc_height(self):
        """Front-view kinematic roll centre height above ground (heave-based, symmetric car)."""
        dy = self._d(self.cp[:, 1])
        return self.cp[:, 1] * dy + (self.cp[:, 2] - self.ground_z)

    @property
    def half_track(self):
        return self.cp[:, 1]

    @property
    def svsa_cp(self):  # side-view path slope of contact patch dx/dz (anti-dive, outboard brakes)
        return self._d(self.cp[:, 0])

    @property
    def svsa_wc(self):  # side-view slope of wheel centre dx/dz (anti-squat/-lift via halfshafts)
        return self._d(self.wc[:, 0])

    def at(self, z=0.0):
        """Interpolate all scalar outputs at travel z (default: static)."""
        keys = dict(camber=self.camber, toe=self.toe, spring_mr=self.spring_mr,
                    damper_mr=self.damper_mr, arb_mr=self.arb_mr, rc_height=self.rc_height,
                    half_track=self.half_track, svsa_cp=self.svsa_cp, svsa_wc=self.svsa_wc,
                    camber_gain=self._d(self.camber), bump_steer=self._d(self.toe),
                    spring_len=self.spring_len, damper_len=self.damper_len)
        return {k: float(np.interp(z, self.travel, v)) for k, v in keys.items()}


# ----------------------------------------------------------------------------------------
class MacPherson:
    def __init__(self, p: Params):
        f = p.suspension.front
        hp = f.hardpoints
        self.r = f.loaded_radius
        self.ground_z = -p.chassis.ride_height_change_front
        self.T = hp.strut_top + np.r_[f.camber_plate_offset, 0.0]
        A0 = np.asarray(hp.lower_ball_joint)            # stock BJ, on strut axis
        u0 = (hp.strut_top - A0) / np.linalg.norm(hp.strut_top - A0)
        self.A0, self.u0 = A0, u0
        L_axis = np.linalg.norm(hp.strut_top - A0)
        self.perch = A0 + (L_axis - f.spring_seat_lower_from_top) * u0
        # arm (rigid A-arm: transverse arm + tension strut)
        self.arm_o, self.arm_k = hp.lca_inner, hp.tension_strut_body - hp.lca_inner
        self.BJ_arm0 = A0.copy()
        self.arb0 = np.asarray(hp.arb_link_lower)
        self.ts_arm0 = np.asarray(hp.tension_strut_arm)
        # upright-local points (= positions in the stock design pose)
        self.BJ_up = A0 - np.r_[0, 0, f.rc_spacer]
        self.TRO_up = hp.tie_rod_outer - np.r_[0, 0, f.bump_steer_spacer]
        self.TRI = np.asarray(hp.tie_rod_inner)
        self.L_tr = np.linalg.norm(hp.tie_rod_outer - hp.tie_rod_inner)  # stock length
        hub_y = f.hub_face_y
        self.H = np.array([0.0, hub_y, self.r])
        cam = f.alignment.camber + f.strut_camber_offset
        self.toe_target = toe_to_rad(f.alignment.toe)
        self.a0 = spin_axis(cam, self.toe_target)
        self.cam_offset = f.strut_camber_offset
        self.ET = p.wheels.offset_ET
        self.wc_static_z = self.r + self.ground_z
        self._x = np.zeros(4)

    # pose -------------------------------------------------------------------------------
    def _bj_arm(self, th):
        return self.arm_o + rot(self.arm_k, th) @ (self.BJ_arm0 - self.arm_o)

    def _pose(self, x):
        th, R = x[0], rotvec(x[1:4])
        bj = self._bj_arm(th)
        return th, R, (lambda q: bj + R @ (np.asarray(q) - self.BJ_up))

    def _residual(self, x, wc_z, L_tr, tri):
        th, R, X = self._pose(x)
        u = R @ self.u0
        d = self.T - X(self.A0)
        perp = d - (d @ u) * u                       # strut axis through top mount (2 eq)
        e1 = np.cross(u, [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        wc = X(self.H) - self.ET * (R @ self.a0)
        return [perp @ e1, perp @ e2,
                np.linalg.norm(X(self.TRO_up) - tri) - L_tr,
                wc[2] - wc_z]

    def solve(self, wc_z, L_tr=None, steer_shift=0.0, x0=None):
        L_tr = self.L_tr if L_tr is None else L_tr
        tri = self.TRI + np.r_[0, steer_shift, 0]
        sol = least_squares(self._residual, self._x if x0 is None else x0,
                            args=(wc_z, L_tr, tri), xtol=1e-12, ftol=1e-12)
        if np.max(np.abs(sol.fun)) > 1e-7:
            raise RuntimeError(f"MacPherson did not converge at wc_z={wc_z:.4f}")
        self._x = sol.x
        th, R, X = self._pose(sol.x)
        a = R @ self.a0
        wc = X(self.H) - self.ET * a
        return dict(a=a, wc=wc, cp=contact_point(wc, a, self.r),
                    spring_len=np.linalg.norm(self.T - X(self.perch)),
                    damper_len=np.linalg.norm(self.T - X(self.A0)),
                    arb=self.arm_o + rot(self.arm_k, th) @ (self.arb0 - self.arm_o),
                    bj=X(self.BJ_up))

    def points(self, wc_z, steer_shift=0.0) -> dict:
        """All drawing points (body frame, m) at wheel-centre height wc_z (left side)."""
        r = self.solve(wc_z, steer_shift=steer_shift)
        th, R, X = self._pose(self._x)
        Ra = rot(self.arm_k, th)
        return dict(r, strut_top=self.T, strut_foot=X(self.A0), perch=X(self.perch),
                    lca_inner=self.arm_o, ts_body=self.arm_o + self.arm_k,
                    ts_arm=self.arm_o + Ra @ (self.ts_arm0 - self.arm_o),
                    tro=X(self.TRO_up), tri=self.TRI + np.r_[0, steer_shift, 0], hub=X(self.H),
                    axis_u=R @ self.u0)

    def realign_toe(self):
        """Adjust track-rod length so static toe equals the alignment spec (as a workshop would)."""
        def f(L):
            self._x = np.zeros(4)
            return camber_toe(self.solve(self.wc_static_z, L_tr=L)["a"])[1] - self.toe_target
        self.L_tr = brentq(f, self.L_tr - 0.02, self.L_tr + 0.02, xtol=1e-9)

    def sweep(self, travel=np.linspace(-0.06, 0.06, 49), realign=True) -> Sweep:
        if realign:
            self.realign_toe()
        out = []
        order = np.argsort(np.abs(travel))       # start near static for robust convergence
        res = [None] * len(travel)
        self._x = np.zeros(4)
        for i in order:
            res[i] = self.solve(self.wc_static_z + travel[i])
        for r in res:
            out.append(r)
        cam, toe = np.array([camber_toe(r["a"]) for r in out]).T
        return Sweep(np.asarray(travel), cam, toe, np.array([r["wc"] for r in out]),
                     np.array([r["cp"] for r in out]), np.array([r["spring_len"] for r in out]),
                     np.array([r["damper_len"] for r in out]), np.array([r["arb"][2] for r in out]),
                     self.ground_z)


# ----------------------------------------------------------------------------------------
class SemiTrailingArm:
    def __init__(self, p: Params):
        r_ = p.suspension.rear
        hp = r_.hardpoints
        self.r = r_.loaded_radius
        self.ground_z = -p.chassis.ride_height_change_rear
        self.o = np.asarray(hp.tsa_pivot_outer)
        self.k = hp.tsa_pivot_inner - hp.tsa_pivot_outer
        x_wc = -p.chassis.wheelbase
        self.H = np.array([x_wc, r_.hub_face_y, self.r])
        self.a0 = spin_axis(r_.alignment.camber, toe_to_rad(r_.alignment.toe))
        self.ET = p.wheels.offset_ET
        self.pts = dict(spring_lower=hp.spring_lower, damper_lower=hp.damper_lower, arb=hp.arb_link_arm)
        self.spring_upper, self.damper_upper = hp.spring_upper, hp.damper_upper
        self.wc_static_z = self.r + self.ground_z

    def _at(self, th):
        R = rot(self.k, th)
        X = lambda q: self.o + R @ (np.asarray(q) - self.o)
        a = R @ self.a0
        wc = X(self.H) - self.ET * a
        return R, X, a, wc

    def solve(self, wc_z):
        th = brentq(lambda t: self._at(t)[3][2] - wc_z, -0.6, 0.6, xtol=1e-12)
        R, X, a, wc = self._at(th)
        return dict(a=a, wc=wc, cp=contact_point(wc, a, self.r), theta=th,
                    spring_len=np.linalg.norm(self.spring_upper - X(self.pts["spring_lower"])),
                    damper_len=np.linalg.norm(self.damper_upper - X(self.pts["damper_lower"])),
                    arb=X(self.pts["arb"]))

    def points(self, wc_z) -> dict:
        r = self.solve(wc_z)
        R, X, a, wc = self._at(r["theta"])
        return dict(r, pivot_outer=self.o, pivot_inner=self.o + self.k, hub=X(self.H),
                    spring_lower=X(self.pts["spring_lower"]), spring_upper=self.spring_upper,
                    damper_lower=X(self.pts["damper_lower"]), damper_upper=self.damper_upper)

    def sweep(self, travel=np.linspace(-0.06, 0.06, 49)) -> Sweep:
        out = [self.solve(self.wc_static_z + t) for t in travel]
        cam, toe = np.array([camber_toe(r["a"]) for r in out]).T
        return Sweep(np.asarray(travel), cam, toe, np.array([r["wc"] for r in out]),
                     np.array([r["cp"] for r in out]), np.array([r["spring_len"] for r in out]),
                     np.array([r["damper_len"] for r in out]), np.array([r["arb"][2] for r in out]),
                     self.ground_z)

    @property
    def sweep_angle(self):
        k = self.k / np.linalg.norm(self.k)
        return np.degrees(np.arctan2(-k[0], -k[1]))   # 0 = pure trailing arm (transverse axis)


def summary(p: Params, sweeps=None) -> dict:
    """Static kinematic numbers for both axles, in mm / deg / deg-per-m units for reporting.
    `sweeps` = (front Sweep, rear Sweep) to reuse already computed sweeps."""
    sf, sr = sweeps if sweeps else (MacPherson(p).sweep(), SemiTrailingArm(p).sweep())
    F, R = sf.at(0.0), sr.at(0.0)
    fmt = {}
    for ax, s in (("front", F), ("rear", R)):
        fmt[ax] = {
            "camber_deg": np.degrees(s["camber"]),
            "toe_total_mm": 2 * TOE_REF_DIAM * np.tan(s["toe"]) * 1e3,
            "camber_gain_deg_per_10mm": np.degrees(s["camber_gain"]) * 0.01,
            "bump_steer_deg_per_10mm": np.degrees(s["bump_steer"]) * 0.01,
            "roll_centre_mm": s["rc_height"] * 1e3,
            "track_mm": 2 * s["half_track"] * 1e3,
            "spring_MR": s["spring_mr"], "damper_MR": s["damper_mr"], "arb_MR": s["arb_mr"],
            "svsa_cp_dxdz": s["svsa_cp"], "svsa_wc_dxdz": s["svsa_wc"],
        }
    return fmt
