"""Layer 5c: brakes — hydraulics, friction vs temperature, disc/drum temperatures and fade.

Hydraulics: pedal force × pedal ratio × booster → master-cylinder pressure p. Per wheel
  front (opposed pistons):  T = 2 faces · μ_pad(T) · p · n_pistons_per_side · A_piston · r_eff
  rear (simplex drum):      T = p · A_wheel_cyl · BF(μ_lining(T)) · r_drum, the brake factor taken proportional to μ
Friction vs temperature: tables of μ / μ_cold (brakes.pad_mu_factor over pad_mu_T, rear lining likewise),
scaled by the cold values front_pad_mu and rear_brake_factor, so the sliders move the whole curve.
Thermal: one lumped node per front disc and per rear drum,
  m c dT/dt = η · share · braking power − h(v) A (T − T_amb) − ε σ A (T⁴ − T_amb⁴),  h(v) = h0 + hv · v^0.8.
  Braking power = (deceleration force − drag − rolling − gradient − engine braking) × speed; engine braking from
  engine.friction_torque_closed_throttle through the gear in use.
Coupling with the lap simulation (QSS cannot integrate temperature itself): run the lap → temperatures along
it → per-point limits (hot brakes give less torque, and the front/rear torque split moves, which also changes the
tyre-limited deceleration) → run the lap again; two passes per lap. A session repeats this lap after lap with
the temperatures carried over.
Temperatures are in °C. Many inputs are estimates (conf C/D in baseline.yaml): read trends, not exact degrees.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIGMA = 5.670e-8
C_IRON = 460.0          # J/kg K, grey cast iron


@dataclass
class Session:
    laps: list = field(default_factory=list)       # per lap: dict(time, T_front_max, T_rear_max, mu_front_min, brake_limited, v_start)
    s: np.ndarray = None                            # cumulative distance over all laps [m]
    T_front: np.ndarray = None
    T_rear: np.ndarray = None
    v: np.ndarray = None                            # km/h
    lap_index: np.ndarray = None
    results: list = field(default_factory=list)     # lapsim Result per lap


class Brakes:
    def __init__(self, p, m, r_tyre):
        b = p.brakes
        self.p, self.m, self.r = p, m, r_tyre
        self.A_mc = np.pi / 4 * b.master_cyl_d ** 2
        self.F_pedal = p.driver.max_pedal_force if "driver" in p and "max_pedal_force" in p.driver else 400.0
        self.p_max = self.F_pedal * b.pedal_ratio * b.booster_ratio / self.A_mc               # Pa at full pedal
        self.Kf = 2 * b.front_pistons_per_side * np.pi / 4 * b.front_piston_d ** 2 * b.front_effective_radius
        self.Kr = np.pi / 4 * b.rear_wheel_cyl_d ** 2 * b.rear_drum_diameter / 2
        self.mu_f0, self.BF0 = b.front_pad_mu, b.rear_brake_factor
        self.padT, self.padF = np.asarray(b.pad_mu_T, float), np.asarray(b.pad_mu_factor, float)
        self.linT, self.linF = np.asarray(b.lining_mu_T, float), np.asarray(b.lining_mu_factor, float)
        self.T_amb = float(b.ambient_temperature)
        e = p.engine                                                    # closed-throttle (engine braking) torque at the crank
        self.eb_rpm = np.array([1000.0, 3500.0, 6000.0]) * 2 * np.pi / 60
        self.eb_T = np.abs(np.asarray(e.friction_torque_closed_throttle, float)) if "friction_torque_closed_throttle" in e else np.zeros(3)
        self.eta = float(b.heat_to_rotor)
        f, r = b.front_thermal, b.rear_thermal
        self.th = {"front": (f.mass * C_IRON, f.area, f.h0, f.hv * f.cooling_factor, f.emissivity),
                   "rear": (r.mass * C_IRON, r.area, r.h0, r.hv * r.cooling_factor, r.emissivity)}

    # ---- friction and torque
    def mu_front(self, T):
        return self.mu_f0 * np.interp(T, self.padT, self.padF)

    def bf_rear(self, T):
        return self.BF0 * np.interp(T, self.linT, self.linF)

    def torques(self, T_f, T_r, pressure=None):
        """Brake torque per front / rear wheel [Nm] at line pressure (default: full pedal)."""
        pr = self.p_max if pressure is None else pressure
        return pr * self.Kf * self.mu_front(T_f), pr * self.Kr * self.bf_rear(T_r)

    def balance(self, T_f, T_r):
        tf, tr = self.torques(T_f, T_r)
        return tf / (tf + tr)

    def max_decel(self, T_f, T_r):
        """Deceleration the brakes can give at full pedal force [m/s²] (tyres not considered)."""
        tf, tr = self.torques(T_f, T_r)
        return 2 * (tf + tr) / self.r / self.m

    # ---- thermal
    def _cool(self, axle, T, v):
        C, A, h0, hv, eps = self.th[axle]
        Tk, Ta = T + 273.15, self.T_amb + 273.15
        return (h0 + hv * max(v, 0.0) ** 0.8) * A * (T - self.T_amb) + eps * SIGMA * A * (Tk ** 4 - Ta ** 4)

    def thermal(self, res, track, pt, T0):
        """Disc / drum temperatures along a lap result, starting at T0 = (front, rear). Returns arrays at res.s."""
        n = len(res.s)
        Tf, Tr = np.empty(n), np.empty(n)
        Tf[0], Tr[0] = T0
        Cf, Cr = self.th["front"][0], self.th["rear"][0]
        w = np.asarray(res.rpm, float) * 2 * np.pi / 60                    # engine speed along the lap
        F_eb = np.interp(w, self.eb_rpm, self.eb_T) * np.array([pt.overall(int(g) - 1) for g in res.gear]) * pt.eta / pt.r
        for i in range(n - 1):
            v = 0.5 * (res.v[i] + res.v[i + 1])
            dt = res.t[i + 1] - res.t[i]
            # force the brakes take = deceleration force not covered by drag, rolling, gradient and engine braking
            F = -self.m * res.ax[i] - pt.resistance(v, self.m, track.grade[i]) - F_eb[i]
            P = max(F, 0.0) * v * self.eta
            beta = self.balance(Tf[i], Tr[i])
            Tf[i + 1] = Tf[i] + dt * (0.5 * beta * P - self._cool("front", Tf[i], v)) / Cf
            Tr[i + 1] = Tr[i] + dt * (0.5 * (1 - beta) * P - self._cool("rear", Tr[i], v)) / Cr
        return Tf, Tr

    # ---- limits for the lap simulation
    def limits(self, lapsim, Tf, Tr):
        """Per-point (deceleration cap from brake torque [m/s²], factor on the tyre-limited braking from the
        front/rear split moving away from the cold split)."""
        cap = self.max_decel(Tf, Tr)
        beta0 = self.balance(self.T_amb, self.T_amb)
        beta = self.balance(Tf, Tr)
        return cap, _split_factor(lapsim, beta) / _split_factor(lapsim, beta0)


def _split_factor(lapsim, beta):
    """Straight-line tyre-limited deceleration for a front brake share beta (whichever axle locks first)."""
    g, mu = 9.80665, lapsim.ax_brake / 9.80665
    L, h, a = lapsim.L, lapsim.h, lapsim.a
    b = L - a
    beta = np.clip(beta, 0.05, 0.98)
    front = mu * g * b / np.maximum(L * beta - mu * h, 1e-6)
    rear = mu * g * a / (L * (1 - beta) + mu * h)
    return np.minimum(front, rear)


def run_session(lapsim, brakes, track, laps=3, flying=True, v0=0.0, passes=2):
    """Lap after lap with the brake temperatures carried over. Open roads run once (laps = 1)."""
    closed = getattr(track, "closed", False)
    laps = laps if closed else 1
    T = (brakes.T_amb, brakes.T_amb)
    v_fly = lapsim.run_road(track, v0=v0, flying=True).v[0] if (closed and flying) else None
    ses = Session()
    parts = []
    for k in range(laps):
        cap = fac = None
        for _ in range(passes):
            if v_fly is not None:
                r = lapsim.run(track, v0=v_fly, v_end=v_fly, dec_cap=cap, brake_factor=fac)
            else:
                r = lapsim.run(track, v0=v0 if k == 0 else parts[-1][0].v[-1], dec_cap=cap, brake_factor=fac)
            Tf, Tr = brakes.thermal(r, track, lapsim.pt, T)
            cap, fac = brakes.limits(lapsim, Tf, Tr)
        limited = float(np.mean((r.limit == "brake") & (cap < lapsim._gg_brake(0.0) * fac + 1e-6))) if cap is not None else 0.0
        ses.laps.append({"lap": k + 1, "time": r.time, "T_front_max": float(Tf.max()), "T_rear_max": float(Tr.max()),
                         "mu_front_min": float(brakes.mu_front(Tf).min()), "brake_limited": limited,
                         "v_start": float(r.v[0] * 3.6), "T_front_end": float(Tf[-1]), "T_rear_end": float(Tr[-1])})
        ses.results.append(r)
        parts.append((r, Tf, Tr))
        T = (Tf[-1], Tr[-1])
    off = np.cumsum([0.0] + [p[0].s[-1] for p in parts[:-1]])
    ses.s = np.concatenate([p[0].s + o for p, o in zip(parts, off)])
    ses.T_front = np.concatenate([p[1] for p in parts])
    ses.T_rear = np.concatenate([p[2] for p in parts])
    ses.v = np.concatenate([p[0].v * 3.6 for p in parts])
    ses.lap_index = np.concatenate([np.full(len(p[0].s), i + 1) for i, p in enumerate(parts)])
    return ses


def hydraulics_table(brakes, lapsim):
    """Numbers for the Brakes tab: pressure, torques, split and decelerations, cold."""
    Ta = brakes.T_amb
    tf, tr = brakes.torques(Ta, Ta)
    return {"pedal_force_N": brakes.F_pedal, "line_pressure_bar": brakes.p_max / 1e5,
            "torque_front_wheel_Nm": tf, "torque_rear_wheel_Nm": tr, "front_share_%": 100 * brakes.balance(Ta, Ta),
            "brake_decel_cold_g": brakes.max_decel(Ta, Ta) / 9.80665, "tyre_decel_g": lapsim.ax_brake / 9.80665,
            "fade_onset_front_C": fade_onset(brakes.padT, brakes.padF), "fade_onset_rear_C": fade_onset(brakes.linT, brakes.linF)}


def fade_onset(T, f, level=0.9):
    """Temperature where friction has dropped to `level` × its best value (on the hot side), or None."""
    i = int(np.argmax(f))
    T, f = np.asarray(T[i:], float), np.asarray(f[i:], float) / f[i]
    if f[-1] >= level:
        return None
    return float(np.interp(level, f[::-1], T[::-1]))
