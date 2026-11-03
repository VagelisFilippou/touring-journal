"""Browser entry point (Pyodide) for the web lab. Same model code and the same setup controls
(vdm/setup_ui.py) as the desktop GUI.

JS calls (all return JSON strings):
  variants()                           -> variant catalogue grouped like the desktop app
  setup(variants_json)                 -> slider spec (tab, section, base values) for the chosen variants
  compute(variants_json, values_json)  -> results, curves, gearing (car as is vs this setup)
  pose(heave_mm, roll_deg, steer_mm, views_json) -> drawing data for the requested views
  dampers_tyre()                       -> damper, tyre and handling-diagram curves (on demand)
  brakes_session(points_json, line, width, laps, flying) -> brake hydraulics, temperatures and fade, lap after lap
"""
from __future__ import annotations

import json

import numpy as np

from . import drawing as D, kinematics as K, lapsim as LS, params, powertrain as PT, racing_line as RLN, roads as RD, steady_state as S
from . import brakes as BR, setup_ui as SU

G = 9.80665
MM = 1e3
D.VIEWS.setdefault("plan_up", (1, 0, 2, 1))       # plan view with the front of the car up (axle tabs)
WEB_GROUPS_SKIP = {"My setups"}                    # personal what-if files stay on the desktop


def web_variants():
    return [v for v in params.variant_catalog() if v["group"] not in WEB_GROUPS_SKIP]


VARIANT_LABELS = {v["file"]: v["label"] for v in web_variants()}

_state = {}


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
        self.lap = LS.LapSim(self.p, vehicle=self.veh, tyre_overrides=tov or {})
        self.acc = self.lap.accel_times(speeds_kmh=(60, 100, 140), dist_m=(400, 1000))

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
                         "hop_freq": float(np.sqrt((ax.k_wheel + ax.k_tyre) / ax.m_unsprung) / (2 * np.pi)),
                         "body_damping": float(c_w / (2 * np.sqrt(k_ride * m_c))),
                         "wheel_load": float(W / 2)}
        return out


def _curves(m):
    out = {}
    for axle, sw in (("front", m.sf), ("rear", m.sr)):
        i0 = lambda y: np.interp(0, sw.travel, y)
        out[axle] = {"travel": (sw.travel * MM).round(1).tolist(), "camber": np.degrees(sw.camber).round(3).tolist(),
                     "toe": (np.degrees(sw.toe) * 60).round(2).tolist(), "rc": (sw.rc_height * MM).round(1).tolist(),
                     "mr": sw.spring_mr.round(4).tolist(),
                     "track": ((sw.half_track - i0(sw.half_track)) * MM).round(2).tolist(),
                     "wcx": ((sw.wc[:, 0] - i0(sw.wc[:, 0])) * MM).round(2).tolist()}
    return out


# (label, getter, better: +1 higher is better, -1 lower is better, 0 neutral)
_PERF = [("0–60 km/h [s]", lambda m: m.acc["0-60 km/h [s]"], -1), ("0–100 km/h [s]", lambda m: m.acc["0-100 km/h [s]"], -1),
         ("0–140 km/h [s]", lambda m: m.acc["0-140 km/h [s]"], -1), ("400 m standing start [s]", lambda m: m.acc["400 m standing [s]"], -1),
         ("1 km standing start [s]", lambda m: m.acc["1000 m standing [s]"], -1), ("Top speed [km/h]", lambda m: m.acc["top speed [km/h]"], 1)]
_VEH = [("Mass with driver [kg]", "mass_kg", 0), ("Front weight [%]", "front_weight_%", 0), ("CoG height [mm]", "cog_height_mm", 0),
        ("Roll gradient [°/g]", "roll_gradient_deg_per_g", -1), ("Roll arm (CoG over roll axis) [mm]", "roll_arm_mm", 0),
        ("Front share of roll stiffness [%]", "roll_stiffness_front_%", 0), ("Front share of load transfer [%]", "LLTD_front_%", 0),
        ("Understeer gradient [°/g]", "understeer_grad_deg_per_g", 0), ("Max lateral acceleration [g]", "max_ay_g", 1)]
_AXLE = [("Static camber [°]", "kin", "camber_deg"), ("Static toe, total [mm]", "kin", "toe_total_mm"),
         ("Camber gain [°/10 mm bump]", "kin", "camber_gain_deg_per_10mm"), ("Bump steer [°/10 mm bump]", "kin", "bump_steer_deg_per_10mm"),
         ("Roll centre height [mm]", "kin", "roll_centre_mm"), ("Track [mm]", "kin", "track_mm"),
         ("Spring rate [N/mm]", "veh", "spring_N/mm"), ("Spring motion ratio [-]", "kin", "spring_MR"), ("Damper motion ratio [-]", "kin", "damper_MR"),
         ("Wheel rate [N/mm]", "veh", "wheel_rate_N/mm"), ("Roll stiffness, springs [Nm/°]", "veh", "K_roll_spring_Nm/deg"),
         ("Roll stiffness, anti-roll bar [Nm/°]", "veh", "K_roll_arb_Nm/deg"), ("Static wheel load [N]", "ride", "wheel_load"),
         ("Body (ride) frequency [Hz]", "ride", "body_freq"), ("Wheel-hop frequency [Hz]", "ride", "hop_freq"),
         ("Body damping ratio [-] (placeholder dampers)", "ride", "body_damping")]


def _results(m, ref):
    """Grouped like the desktop Results tab: [{group, rows: [[label, car as is, this setup, better]]}]."""
    f = lambda v: None if v is None else float(v)
    out = [{"group": "Performance (full throttle, ideal shifts)", "rows": [[lab, f(g(ref)), f(g(m)), b] for lab, g, b in _PERF]},
           {"group": "Vehicle (steady state)", "rows": [[lab, f(ref.summary[k]), f(m.summary[k]), b] for lab, k, b in _VEH]}]
    for axle in ("front", "rear"):
        def src(mm, kind, k, axle=axle):
            return {"kin": mm.kin[axle], "veh": mm.summary[axle], "ride": mm.ride[axle]}[kind][k]
        out.append({"group": f"{axle.capitalize()} axle", "rows": [[lab, f(src(ref, kd, k)), f(src(m, kd, k)), 0] for lab, kd, k in _AXLE]})
    return out


# ---------------------------------------------------------------- API
def variants():
    groups = {}
    for v in web_variants():
        groups.setdefault(v["group"], {"group": v["group"], "exclusive": v["exclusive"], "items": []})["items"].append(
            {"file": v["file"], "label": v["label"], "description": v["description"]})
    return json.dumps({"groups": list(groups.values()), "as_is": SU.AS_IS})


def road_presets():
    return json.dumps(RD.list_roads())


def parse_road(text):
    """Saved route file (YAML/JSON, our road format or a bare list of [lat, lon, ele]) -> {label, points}."""
    import yaml
    d = yaml.safe_load(text)
    pts = d.get("points") if isinstance(d, dict) else d
    if not pts or len(pts[0]) < 3:
        raise ValueError("no [lat, lon, elevation] points found")
    meta = (d.get("meta") or {}) if isinstance(d, dict) else {}
    return json.dumps({"label": meta.get("label", ""), "width": meta.get("width"),
                       "points": [[float(a), float(b), float(c)] for a, b, c, *_ in pts]})


def road_points(name):
    import yaml
    d = yaml.safe_load((RD.ROADS / f"{name}.yaml").read_text(encoding="utf-8"))
    return json.dumps(d["points"])


def _road_lines(points_json, width):
    """Base track + the three driving lines; cached (they depend on the road and width, not on the setup)."""
    import hashlib
    key = (hashlib.md5(points_json.encode()).hexdigest(), round(float(width), 2))
    c = _state.setdefault("lines_cache", {})
    if key not in c:
        base = RD.build_track(json.loads(points_json))
        lines = {k: RLN.line_track(base, k, width) for k in RLN.LINES}
        c.clear()
        c[key] = (base, lines, {})                                      # {} = memo of the reference ("car as is") times
    return c[key]


def road_sim(points_json, v0_kmh=0.0, line="centre", width=6.0, flying=True):
    """Lap-time sim of the current setup on a road given as [[lat, lon, elevation], ...], along the chosen line.
    Closed circuits run as a flying lap when `flying` (start speed = finish speed); otherwise from v0."""
    m, ref = _state["model"], _state["ref"]
    base, lines, ref_memo = _road_lines(points_json, width)
    fly = bool(flying) and base.closed
    run = lambda lap, t: lap.run_road(t, v0=v0_kmh / 3.6, flying=fly)
    tr = lines[line][0]
    r = run(m.lap, tr)
    times = {k: {"label": RLN.LINES[k], "time": run(m.lap, lt[0]).time if k != line else r.time,
                 "time_ref": ref_memo.setdefault((k, v0_kmh, fly), run(ref.lap, lt[0]).time)} for k, lt in lines.items()}
    step = max(1, int(round(10.0 / (tr.s[1] - tr.s[0]))))           # ~10 m spacing for the browser
    idx = np.r_[np.arange(0, len(tr.s), step), len(tr.s) - 1]
    share = {k: float((r.limit == k).mean()) for k in ("power", "traction", "corner", "brake")}
    # zoom geometry in the road's local frame (m, east/north), every 4 m along the centreline
    g = np.r_[np.arange(0, len(base.s), 2), len(base.s) - 1]
    le, ri = RLN.edges(base, width)
    geo = {"c": base.xy[g].round(2).tolist(), "left": le[g].round(2).tolist(), "right": ri[g].round(2).tolist(),
           "lines": {k: lt[2][g].round(2).tolist() for k, lt in lines.items()}, "s": base.s[g].round(1).tolist()}
    shifts, brakes = RLN.key_points(tr, r, base)
    return json.dumps({"time": r.time, "time_ref": times[line]["time_ref"], "line": line, "width": width, "lines": times, "geo": geo,
                       "closed": bool(base.closed), "flying": fly, "v_start": float(r.v[0] * 3.6),
                       "shifts": shifts, "brakes": brakes,
                       "length": tr.length, "vmax": float(r.v.max() * 3.6), "avg": tr.length / r.time * 3.6,
                       "climb": float(tr.elev[-1] - tr.elev[0]), "min_radius": float(1 / max(abs(tr.kappa).max(), 1e-6)),
                       "s": tr.s[idx].round(1).tolist(), "v": (r.v[idx] * 3.6).round(1).tolist(), "elev": tr.elev[idx].round(1).tolist(),
                       "gear": r.gear[idx].tolist(), "rpm": r.rpm[idx].round(0).tolist(), "latlon": tr.latlon[idx].round(6).tolist(), "share": share,
                       "gears_used": sorted(set(int(g) for g in r.gear))})


def brakes_session(points_json, line="centre", width=6.0, laps=3, flying=True, v0_kmh=0.0):
    """Brake temperatures and fade on a road / circuit for the current setup (open roads: one run)."""
    m = _state["model"]
    br = BR.Brakes(m.p, m.lap.m, m.lap.pt.r)
    hyd = {k: (None if v is None else float(v)) for k, v in BR.hydraulics_table(br, m.lap).items()}
    if not points_json:
        return json.dumps({"hyd": hyd, "laps": [], "series": None})
    base, lines, _ = _road_lines(points_json, width)
    tr = lines[line][0]
    ses = BR.run_session(m.lap, br, tr, laps=int(laps), flying=bool(flying), v0=v0_kmh / 3.6)
    step = max(1, int(round(20.0 / (tr.s[1] - tr.s[0]))))                 # ~20 m spacing for the browser
    idx = np.r_[np.arange(0, len(ses.s), step), len(ses.s) - 1]
    no_fade = m.lap.run_road(tr, v0=v0_kmh / 3.6, flying=bool(flying) and tr.closed).time
    return json.dumps({"hyd": hyd, "laps": ses.laps, "no_fade": no_fade, "closed": bool(tr.closed), "flying": bool(flying) and bool(tr.closed),
                       "series": {"s": (ses.s[idx] / 1e3).round(3).tolist(), "Tf": ses.T_front[idx].round(1).tolist(),
                                  "Tr": ses.T_rear[idx].round(1).tolist(), "v": ses.v[idx].round(1).tolist(),
                                  "lap": ses.lap_index[idx].tolist(), "lap_len": float(tr.length / 1e3)}})


def setup(variants_json):
    v = json.loads(variants_json)
    raw = params.load(*v).raw
    traw = SU.tyre_raw(raw)
    base = params.load().raw
    ng, ng_base = SU.n_gears(raw), SU.n_gears(base)
    out = []
    for grp, lab, path, lo, hi, step in SU.PARAMS:
        tab, sec = SU.TABS.get(grp, (grp, grp))
        val, ref = SU.base_value(raw, traw, path), SU.base_value(base, traw, path)
        gear = int(path[-1]) if path.startswith("@gear_") else 0
        out.append({"tab": tab, "section": sec, "label": lab, "path": path, "min": lo, "max": hi, "step": step,
                    "value": round(val, 4), "base": val, "baseline": round(ref, 4),
                    "disabled": gear > ng, "baseline_text": "no such gear" if gear > ng_base else None})
    _state["raw"], _state["traw"], _state["variants"] = raw, traw, v
    _state["bases"] = {s_["path"]: s_["base"] for s_ in out}
    return json.dumps(out)


def compute(variants_json, values_json):
    v = json.loads(variants_json)
    if v != _state.get("variants"):
        setup(variants_json)
    values = {p: x for p, x in json.loads(values_json).items() if p in _state["bases"]}
    ov, tov = SU.overrides(_state["raw"], values, _state["bases"])
    if "ref" not in _state:
        _state["ref"] = Model([])
    m = Model(v, ov, tov)
    _state["model"] = m
    ref = _state["ref"]
    gearing = {"current": PT.gearing_table(m.p, m.lap.pt), "ref": PT.gearing_table(ref.p, ref.lap.pt)}
    return json.dumps({"results": _results(m, ref), "gearing": gearing, "limit": m.summary["limit"], "max_ay": float(m.summary["max_ay_g"]),
                       "curves": _curves(m), "curves_ref": _curves(ref), "n_overrides": len(ov) + len(tov),
                       "sweep": float(m.rear.sweep_angle)})


def dampers_tyre():
    """Damper F-v curves, tyre lateral force and the handling diagram (this setup vs the car as is)."""
    m, ref = _state["model"], _state["ref"]
    out = {}
    for axle in ("front", "rear"):
        d = m.p.suspension[axle].damper
        c = m.p.dampers[d.curve]
        reb = c.rebound_soft + d.setting * (c.rebound_hard - c.rebound_soft)
        out[axle] = {"model": str(d.model), "v": [float(x) for x in c.velocity], "reb": [float(x) for x in reb],
                     "comp": [float(x) for x in c.compression], "soft": [float(x) for x in c.rebound_soft],
                     "hard": [float(x) for x in c.rebound_hard]}
    al = np.linspace(0, 0.25, 51)
    W = m.ride["front"]["wheel_load"]
    out["tyre"] = {"slip": np.degrees(al).round(3).tolist(), "loads": [round(W * f) for f in (0.5, 1.0, 1.5)],
                   "cur": [(m.veh.tyre.Fy(al, W * f) / MM).round(4).tolist() for f in (0.5, 1.0, 1.5)],
                   "ref": [(ref.veh.tyre.Fy(al, W * f) / MM).round(4).tolist() for f in (0.5, 1.0, 1.5)]}
    hd = {}
    for key, mm in (("cur", m), ("ref", ref)):
        ay = np.linspace(0.5, mm.summary["max_ay_g"] * G * 0.99, 25)
        hd[key] = {"ay": (ay / G).round(4).tolist(), "delta": [round(float(np.degrees(mm.veh.solve(a)["delta_kin"])), 4) for a in ay]}
    out["handling"] = hd
    return json.dumps(out)


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


def _shapes(prims, view):
    items = sorted(((D.project(p, view), p) for p in prims), key=lambda it: it[0][2])
    return [{"k": kind, "p": (xy * MM).round(1).tolist(), "c": p["color"], "a": p["alpha"], "w": p.get("lw", 1.5)}
            for (kind, xy, _), p in items]


def pose(heave_mm, roll_deg, steer_mm, views_json='["front", "rear"]'):
    m = _state["model"]
    want = set(json.loads(views_json))
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
        if axle not in want:
            continue
        L, R, zl, zr = pts[axle]
        prims = D.car_prims(pts, m.p, m.veh.tyre.R0, include=(axle,))
        lines, rc = _ic_rc(axle, L, R)
        read = []
        for side, P, z in (("L", L, zl), ("R", R, zr)):
            cam, toe = K.camber_toe(P["a"] * [1, 1 if side == "L" else -1, 1])
            read.append({"side": side, "travel": round(z * MM, 1), "camber": round(float(np.degrees(cam)), 2),
                         "toe": round(float(np.degrees(toe) * 60), 1), "spring": round(float(P["spring_len"] * MM))})
        ground = [(L["cp"][[1, 2]] * MM).round(1).tolist(), (R["cp"][[1, 2]] * MM).round(1).tolist()]
        rc_h = None if rc is None else round(rc[1] - 0.5 * (ground[0][1] + ground[1][1]), 1)
        sw = m.sf if axle == "front" else m.sr
        views[axle] = {"shapes": _shapes(prims, axle), "plan": _shapes(prims, "plan_up"), "ic_lines": lines, "rc": rc,
                       "rc_height": rc_h, "ground": ground, "read": read, "track": round(abs(L["cp"][1] - R["cp"][1]) * MM),
                       "travel_l": round(zl * MM, 1), "camber_l": round(float(np.degrees(np.interp(zl, sw.travel, sw.camber))), 3)}
    if want & {"side", "plan"}:
        prims = D.car_prims(pts, m.p, m.veh.tyre.R0)
        for view in ("side", "plan"):
            if view in want:
                views[view] = {"shapes": _shapes(prims, view), "ground_z": round(float(m.front.ground_z * MM), 1)}
    return json.dumps(views)
