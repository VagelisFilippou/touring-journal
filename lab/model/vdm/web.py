"""Browser entry point (Pyodide) for the web 'Suspension lab'. Same model code as the desktop GUI.

JS calls (all return JSON strings):
  setup(variants_json)                 -> slider spec with base values for the chosen variants
  compute(variants_json, sliders_json) -> results + curves (baseline vs current)
  pose(heave_mm, roll_deg, steer_mm)   -> drawing data for the front and rear views
"""
from __future__ import annotations

import json
import re

import numpy as np

from . import drawing as D, kinematics as K, params, steady_state as S

G = 9.80665
MM = 1e3

# (group, label, path, min, max, step) — subset of the desktop GUI sliders; values in input-file units
SLIDERS = [
    ("Front", "Ride height change [mm]", "chassis.ride_height_change_front", -60, 30, 1),
    ("Front", "Roll-centre spacer [mm]", "suspension.front.rc_spacer", 0, 40, 1),
    ("Front", "Bump-steer spacer [mm]", "suspension.front.bump_steer_spacer", 0, 40, 1),
    ("Front", "Strut built-in camber [deg]", "suspension.front.strut_camber_offset", -3, 1, 0.1),
    ("Front", "Camber plate, top inboard [mm]", "@camber_plate_in", -10, 30, 1),
    ("Front", "Static toe, total [mm]", "suspension.front.alignment.toe", -3, 5, 0.1),
    ("Front", "Lower arm inner pivot z [mm]", "suspension.front.hardpoints.lca_inner[2]", 140, 240, 1),
    ("Rear", "Ride height change [mm]", "chassis.ride_height_change_rear", -60, 30, 1),
    ("Rear", "Trailing-arm sweep [deg]", "@rear_sweep", 5, 35, 0.5),
    ("Rear", "Pivot height [mm]", "@rear_pivot_z", 170, 290, 1),
    ("Rear", "Static camber [deg]", "suspension.rear.alignment.camber", -4, 0.5, 0.1),
    ("Springs & bars", "Front spring rate [N/mm]", "suspension.front.spring.rate", 10, 80, 0.5),
    ("Springs & bars", "Rear spring rate [N/mm]", "suspension.rear.spring.rate", 10, 100, 0.5),
    ("Springs & bars", "Front anti-roll bar Ø [mm] (0 = none)", "suspension.front.arb.diameter", 0, 28, 0.5),
    ("Springs & bars", "Rear anti-roll bar Ø [mm] (0 = none)", "suspension.rear.arb.diameter", 0, 25, 0.5),
    ("Springs & bars", "Front damper setting (0 soft – 1 hard)", "suspension.front.damper.setting", 0, 1, 0.05),
    ("Springs & bars", "Rear damper setting (0 soft – 1 hard)", "suspension.rear.damper.setting", 0, 1, 0.05),
    ("Mass & tyres", "CoG height [mm]", "mass.cog_height_kerb", 400, 600, 5),
    ("Mass & tyres", "Front weight fraction [-]", "mass.front_fraction_kerb", 0.46, 0.60, 0.005),
    ("Mass & tyres", "Wheel offset ET [mm]", "wheels.offset_ET", -10, 40, 1),
    ("Mass & tyres", "Tyre grip μ [-]", "tyre:lateral.mu_y0", 0.6, 1.6, 0.01),
]
VARIANT_LABELS = {
    "wheels_FPS_5J_ET26.yaml": "FPS alloys 5J ET26",
    "lowered_WN_springs.yaml": "Lowering springs (−20 mm)",
    "rc_spacers_20.yaml": "Roll-centre spacers 20 mm",
    "alpina_bilstein_struts.yaml": "Alpina struts (−1.5° camber)",
    "arb_ST_22_19.yaml": "Anti-roll bars 22/19 mm",
}

_state = {}


# ---------------------------------------------------------------- helpers
def _leaf(raw, path):
    d = raw
    for k in path.split("."):
        d = d[k]
    return d


def _split(path):
    m = re.match(r"(.*)\[(\d)\]$", path)
    return (m.group(1), int(m.group(2))) if m else (path, None)


def _special(raw):
    hp = _leaf(raw, "suspension.rear.hardpoints")
    o, i = np.array(hp["tsa_pivot_outer"]["value"], float), np.array(hp["tsa_pivot_inner"]["value"], float)
    k = i - o
    return {"@rear_sweep": float(np.degrees(np.arctan2(-k[0], -k[1]))), "@rear_pivot_z": float(o[2]),
            "@camber_plate_in": -float(_leaf(raw, "suspension.front.camber_plate_offset")["value"][1])}


def _tyre_raw(raw):
    import yaml
    return yaml.safe_load(open(params.INPUTS / "tyres" / _leaf(raw, "wheels.tyre_model")["value"], encoding="utf-8"))


def _base_value(raw, tyre_raw, path):
    if path.startswith("@"):
        return _special(raw)[path]
    if path.startswith("tyre:"):
        return float(_leaf(tyre_raw, path[5:])["value"])
    p, idx = _split(path)
    v = _leaf(raw, p)["value"]
    return float(v[idx] if idx is not None else v)


def _overrides(raw, tyre_raw, values):
    ov, tov, vec = {}, {}, {}
    changed = {p: v for p, v in values.items() if abs(v - _base_value(raw, tyre_raw, p)) > 1e-3}  # below the smallest slider step
    for path, val in changed.items():
        if path.startswith("tyre:"):
            tov[path[5:]] = val
        elif not path.startswith("@"):
            p, idx = _split(path)
            if idx is None:
                ov[p] = val
            else:
                vec.setdefault(p, list(_leaf(raw, p)["value"]))[idx] = val
    ov.update(vec)
    if "@camber_plate_in" in changed:
        ov["suspension.front.camber_plate_offset"] = [0.0, -changed["@camber_plate_in"]]
    if "@rear_sweep" in changed or "@rear_pivot_z" in changed:
        hp = _leaf(raw, "suspension.rear.hardpoints")
        o = np.array(hp["tsa_pivot_outer"]["value"], float)
        i = np.array(hp["tsa_pivot_inner"]["value"], float)
        L = np.linalg.norm(i - o)
        sw = np.radians(values.get("@rear_sweep", _special(raw)["@rear_sweep"]))
        o[2] = values.get("@rear_pivot_z", o[2])
        i_new = o + L * np.array([-np.sin(sw), -np.cos(sw), 0.0])
        ov["suspension.rear.hardpoints.tsa_pivot_outer"] = [float(x) for x in o]
        ov["suspension.rear.hardpoints.tsa_pivot_inner"] = [float(x) for x in i_new]
    return ov, tov


class Model:
    def __init__(self, variants, ov=None, tov=None):
        self.p = params.load(*variants, overrides=ov or {})
        self.front = K.MacPherson(self.p)
        self.front.realign_toe()
        self.rear = K.SemiTrailingArm(self.p)
        self.sf = self.front.sweep(np.linspace(-0.08, 0.08, 33), realign=False)
        self.sr = self.rear.sweep(np.linspace(-0.08, 0.08, 33))
        self.veh = S.Vehicle(self.p, tyre_overrides=tov or {}, sweeps=(self.sf, self.sr))
        self.summary = self.veh.summary()
        self.kin = K.summary(self.p, sweeps=(self.sf, self.sr))
        self.ride = self._ride()

    def _ride(self):
        out = {}
        Wf, Wr = self.veh.static_axle_loads
        for axle, W, sw in (("front", Wf, self.sf), ("rear", Wr, self.sr)):
            ax = self.veh.axles[axle]
            d = self.p.suspension[axle].damper
            c = self.p.dampers[d.curve]
            reb = c.rebound_soft + d.setting * (c.rebound_hard - c.rebound_soft)
            cd = 0.5 * (np.interp(0.2, c.velocity, reb) + np.interp(0.2, c.velocity, c.compression)) / 0.2  # 0-0.2 m/s secant (ride)
            c_w = cd * sw.at(0)["damper_mr"] ** 2
            m_c = W / 2 / G - ax.m_unsprung
            k_ride = ax.k_wheel * ax.k_tyre / (ax.k_wheel + ax.k_tyre)
            out[axle] = {"body_freq": float(np.sqrt(k_ride / m_c) / (2 * np.pi)),
                         "body_damping": float(c_w / (2 * np.sqrt(k_ride * m_c)))}
        return out


def _curves(m):
    out = {}
    for axle, sw in (("front", m.sf), ("rear", m.sr)):
        out[axle] = {"travel": (sw.travel * MM).round(1).tolist(), "camber": np.degrees(sw.camber).round(3).tolist(),
                     "toe": (np.degrees(sw.toe) * 60).round(2).tolist(), "rc": (sw.rc_height * MM).round(1).tolist(),
                     "mr": sw.spring_mr.round(4).tolist()}
    return out


def _results(m):
    s, k = m.summary, m.kin
    r = {"Roll gradient [°/g]": s["roll_gradient_deg_per_g"], "Understeer gradient [°/g]": s["understeer_grad_deg_per_g"],
         "Max lateral acceleration [g]": s["max_ay_g"], "Front share of roll stiffness [%]": s["roll_stiffness_front_%"],
         "Front share of load transfer [%]": s["LLTD_front_%"]}
    for axle in ("front", "rear"):
        A = axle.capitalize()
        r[f"{A} roll centre [mm]"] = k[axle]["roll_centre_mm"]
        r[f"{A} static camber [°]"] = k[axle]["camber_deg"]
        r[f"{A} camber gain [°/10 mm]"] = k[axle]["camber_gain_deg_per_10mm"]
        r[f"{A} bump steer [°/10 mm]"] = k[axle]["bump_steer_deg_per_10mm"]
        r[f"{A} track [mm]"] = k[axle]["track_mm"]
        r[f"{A} wheel rate [N/mm]"] = s[axle]["wheel_rate_N/mm"]
        r[f"{A} ride frequency [Hz]"] = m.ride[axle]["body_freq"]
        r[f"{A} body damping ratio [-] (placeholder dampers)"] = m.ride[axle]["body_damping"]
    return {kk: float(v) for kk, v in r.items()}


# ---------------------------------------------------------------- API
def variants():
    files = sorted(f.name for f in (params.INPUTS / "variants").glob("*.yaml"))
    return json.dumps([{"file": f, "label": VARIANT_LABELS.get(f, f[:-5].replace("_", " "))} for f in files if f in VARIANT_LABELS])


def setup(variants_json):
    v = json.loads(variants_json)
    raw = params.load(*v).raw
    traw = _tyre_raw(raw)
    base = params.load().raw
    out = []
    for grp, lab, path, lo, hi, step in SLIDERS:
        out.append({"group": grp, "label": lab, "path": path, "min": lo, "max": hi, "step": step,
                    "value": round(_base_value(raw, traw, path), 4), "baseline": round(_base_value(base, traw, path), 4)})
    _state["raw"], _state["traw"], _state["variants"] = raw, traw, v
    return json.dumps(out)


def compute(variants_json, sliders_json):
    v = json.loads(variants_json)
    if v != _state.get("variants"):
        setup(variants_json)
    ov, tov = _overrides(_state["raw"], _state["traw"], json.loads(sliders_json))
    if "ref" not in _state:
        _state["ref"] = Model([])
    m = Model(v, ov, tov)
    _state["model"] = m
    ref = _state["ref"]
    return json.dumps({"results": _results(m), "baseline": _results(ref), "limit": m.summary["limit"],
                       "curves": _curves(m), "curves_ref": _curves(ref), "n_overrides": len(ov) + len(tov)})


def _ic_rc(axle, L, R):
    Y, Z = 1, 2

    def inter(p1, d1, p2, d2):
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-12:
            return None
        return p1 + np.linalg.solve(A, p2 - p1)[0] * d1

    ics = []
    for P in (L, R):
        if axle == "front":
            u = P["axis_u"][[Y, Z]]
            ic = inter(P["strut_top"][[Y, Z]], np.array([u[1], -u[0]]), P["lca_inner"][[Y, Z]], (P["bj"] - P["lca_inner"])[[Y, Z]])
        else:
            o, k = P["pivot_outer"], P["pivot_inner"] - P["pivot_outer"]
            ic = (o + (P["wc"][0] - o[0]) / k[0] * k)[[Y, Z]]
        ics.append(ic)
    rc = None
    if all(i is not None for i in ics):
        rc = inter(L["cp"][[Y, Z]], ics[0] - L["cp"][[Y, Z]], R["cp"][[Y, Z]], ics[1] - R["cp"][[Y, Z]])
    lines = [[(P["cp"][[Y, Z]] * MM).round(1).tolist(), (ic * MM).round(1).tolist()]
             for P, ic in zip((L, R), ics) if ic is not None and abs(ic[0]) < 6]
    return lines, (None if rc is None else (rc * MM).round(1).tolist())


def pose(heave_mm, roll_deg, steer_mm):
    m = _state["model"]
    h, phi, st = heave_mm / MM, np.radians(roll_deg), steer_mm / MM
    pts, views = {}, {}
    for axle, obj, sw in (("front", m.front, m.sf), ("rear", m.rear, m.sr)):
        t = 2 * sw.at(0)["half_track"]
        zl, zr = h - t / 2 * np.sin(phi), h + t / 2 * np.sin(phi)
        if axle == "front":
            L = obj.points(obj.wc_static_z + zl, steer_shift=st)
            R = obj.points(obj.wc_static_z + zr, steer_shift=-st)
        else:
            L, R = obj.points(obj.wc_static_z + zl), obj.points(obj.wc_static_z + zr)
        R = {k: (v * [1, -1, 1] if isinstance(v, np.ndarray) and v.shape == (3,) else v) for k, v in R.items()}
        pts[axle] = (L, R, zl, zr)
    for axle in ("front", "rear"):
        L, R, zl, zr = pts[axle]
        prims = D.car_prims(pts, m.p, m.veh.tyre.R0, include=(axle,))
        items = sorted(((D.project(p, axle), p) for p in prims), key=lambda it: it[0][2])
        shapes = [{"k": kind, "p": (xy * MM).round(1).tolist(), "c": p["color"], "a": p["alpha"], "w": p.get("lw", 1.5)}
                  for (kind, xy, _), p in items]
        lines, rc = _ic_rc(axle, L, R)
        read = []
        for side, P, z in (("L", L, zl), ("R", R, zr)):
            cam, toe = K.camber_toe(P["a"] * [1, 1 if side == "L" else -1, 1])
            read.append({"side": side, "travel": round(z * MM, 1), "camber": round(float(np.degrees(cam)), 2),
                         "toe": round(float(np.degrees(toe) * 60), 1)})
        ground = [(L["cp"][[1, 2]] * MM).round(1).tolist(), (R["cp"][[1, 2]] * MM).round(1).tolist()]
        rc_h = None if rc is None else round(rc[1] - 0.5 * (ground[0][1] + ground[1][1]), 1)
        views[axle] = {"shapes": shapes, "ic_lines": lines, "rc": rc, "rc_height": rc_h, "ground": ground, "read": read}
    return json.dumps(views)
