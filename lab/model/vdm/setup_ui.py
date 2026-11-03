"""Setup controls shared by the desktop GUI (gui.py) and the web lab (vdm/web.py): the slider list, how the
sliders are grouped into tabs, the labels for the car as it is, and the conversion of slider values into
parameter overrides. Keep both front ends on this one definition.
"""
from __future__ import annotations

import re

import numpy as np

from . import params

# Tweakable parameters: (group, label, path, min, max, step). Values in input-file units.
# path "x.y[i]" = component i of a vector leaf; "tyre:..." = tyre file leaf; special paths start with '@'.
PARAMS = [
    ("Front geometry", "Ride height change front [mm]", "chassis.ride_height_change_front", -60, 30, 1),
    ("Front geometry", "Roll-centre spacer [mm]", "suspension.front.rc_spacer", 0, 40, 1),
    ("Front geometry", "Bump-steer spacer [mm]", "suspension.front.bump_steer_spacer", 0, 40, 1),
    ("Front geometry", "Strut built-in camber [deg]", "suspension.front.strut_camber_offset", -3, 1, 0.1),
    ("Front geometry", "Camber plate inboard shift [mm] (−y)", "@camber_plate_in", -10, 30, 1),
    ("Front geometry", "Caster plate rearward shift [mm]", "@caster_plate_back", -15, 30, 1),
    ("Front geometry", "Static toe front, total [mm]", "suspension.front.alignment.toe", -3, 5, 0.1),
    ("Front geometry", "Strut top z [mm]", "suspension.front.hardpoints.strut_top[2]", 680, 820, 1),
    ("Front geometry", "Lower ball joint z [mm]", "suspension.front.hardpoints.lower_ball_joint[2]", 110, 190, 1),
    ("Front geometry", "Lower arm inner pivot y [mm]", "suspension.front.hardpoints.lca_inner[1]", 280, 400, 1),
    ("Front geometry", "Lower arm inner pivot z [mm]", "suspension.front.hardpoints.lca_inner[2]", 140, 240, 1),
    ("Front geometry", "Tension strut body mount x [mm]", "suspension.front.hardpoints.tension_strut_body[0]", 150, 500, 5),
    ("Front geometry", "Tie rod inner z [mm]", "suspension.front.hardpoints.tie_rod_inner[2]", 170, 250, 1),
    ("Front geometry", "Tie rod outer z [mm]", "suspension.front.hardpoints.tie_rod_outer[2]", 140, 210, 1),
    ("Rear geometry", "Ride height change rear [mm]", "chassis.ride_height_change_rear", -60, 30, 1),
    ("Rear geometry", "Trailing-arm sweep angle [deg]", "@rear_sweep", 5, 35, 0.5),
    ("Rear geometry", "Pivot axis dihedral [deg] (+ inner up)", "@rear_dihedral", -5, 5, 0.1),
    ("Rear geometry", "Pivot height outer z [mm]", "@rear_pivot_z", 170, 290, 1),
    ("Rear geometry", "Subframe height offset [mm] (+ up, e.g. flipped mounts)", "suspension.rear.subframe_offset[2]", -20, 30, 0.5),
    ("Rear geometry", "Subframe fore/aft offset [mm] (+ forward)", "suspension.rear.subframe_offset[0]", -20, 20, 0.5),
    ("Rear geometry", "Static camber rear [deg] (brackets)", "suspension.rear.alignment.camber", -4, 0.5, 0.1),
    ("Rear geometry", "Static toe rear, total [mm]", "suspension.rear.alignment.toe", -2, 6, 0.1),
    ("Springs / ARB", "Front spring rate [N/mm]", "suspension.front.spring.rate", 10, 80, 0.5),
    ("Springs / ARB", "Rear spring rate [N/mm]", "suspension.rear.spring.rate", 10, 100, 0.5),
    ("Springs / ARB", "Front ARB diameter [mm] (0 = none)", "suspension.front.arb.diameter", 0, 28, 0.5),
    ("Springs / ARB", "Rear ARB diameter [mm] (0 = none)", "suspension.rear.arb.diameter", 0, 25, 0.5),
    ("Springs / ARB", "Front ARB lever [mm]", "suspension.front.arb.lever_length", 120, 260, 5),
    ("Springs / ARB", "Rear ARB lever [mm]", "suspension.rear.arb.lever_length", 120, 260, 5),
    ("Springs / ARB", "Rear ARB mount efficiency [-]", "suspension.rear.arb.efficiency", 0.4, 1.0, 0.05),
    ("Dampers", "Front damper setting (0 soft…1 hard)", "suspension.front.damper.setting", 0, 1, 0.05),
    ("Dampers", "Rear damper setting (0 soft…1 hard)", "suspension.rear.damper.setting", 0, 1, 0.05),
    ("Powertrain", "Gear 1 ratio [-]", "@gear_1", 2.5, 4.3, 0.005),
    ("Powertrain", "Gear 2 ratio [-]", "@gear_2", 1.4, 2.6, 0.005),
    ("Powertrain", "Gear 3 ratio [-]", "@gear_3", 1.0, 1.8, 0.005),
    ("Powertrain", "Gear 4 ratio [-]", "@gear_4", 0.75, 1.4, 0.005),
    ("Powertrain", "Gear 5 ratio [-]", "@gear_5", 0.7, 1.2, 0.005),
    ("Powertrain", "Final drive ratio [-]", "driveline.final_drive", 3.0, 4.6, 0.01),
    ("Powertrain", "Engine torque, whole curve [%]", "@torque_scale", 70, 140, 1),
    ("Powertrain", "Rev limit [rpm]", "engine.rev_limit", 5500, 7800, 50),
    ("Powertrain", "LSD locking [%] (0 = open)", "driveline.lsd_lock_ratio", 0, 75, 5),
    ("Powertrain", "Driveline efficiency [-]", "driveline.efficiency", 0.80, 0.97, 0.01),
    ("Driver", "Downshift under braking at or below [rpm]", "driver.downshift_rpm", 4000, 7000, 100),
    ("Driver", "Shortest time in a gear [s]", "driver.min_gear_time", 0, 4, 0.1),
    ("Driver", "Pace [% of the grip limit]", "driver.pace", 70, 100, 1),
    ("Brakes", "Max pedal force (hard stop) [N]", "driver.max_pedal_force", 150, 700, 10),
    ("Brakes", "Pedal ratio [-]", "brakes.pedal_ratio", 3.0, 6.0, 0.1),
    ("Brakes", "Booster ratio [-] (1 = no servo)", "brakes.booster_ratio", 1.0, 4.0, 0.1),
    ("Brakes", "Master cylinder Ø [mm]", "brakes.master_cyl_d", 19.0, 26.0, 0.01),
    ("Brakes", "Rear wheel cylinder Ø [mm]", "brakes.rear_wheel_cyl_d", 15.0, 25.0, 0.1),
    ("Brakes", "Front pad μ, cold [-]", "brakes.front_pad_mu", 0.25, 0.60, 0.01),
    ("Brake cooling", "Front disc mass [kg]", "brakes.front_thermal.mass", 3.0, 8.0, 0.1),
    ("Brake cooling", "Front cooling factor [-] (ducts, shields)", "brakes.front_thermal.cooling_factor", 0.5, 3.0, 0.05),
    ("Brake cooling", "Rear drum mass [kg]", "brakes.rear_thermal.mass", 3.0, 8.0, 0.1),
    ("Brake cooling", "Ambient temperature [°C]", "brakes.ambient_temperature", 0, 45, 1),
    ("Mass / tyres", "Kerb mass [kg]", "mass.kerb", 850, 1200, 5),
    ("Mass / tyres", "Front weight fraction (kerb) [-]", "mass.front_fraction_kerb", 0.46, 0.60, 0.005),
    ("Mass / tyres", "CoG height (kerb) [mm]", "mass.cog_height_kerb", 400, 600, 5),
    ("Mass / tyres", "Driver mass [kg]", "mass.driver", 0, 120, 1),
    ("Mass / tyres", "Wheel offset ET [mm]", "wheels.offset_ET", -10, 40, 1),
    ("Mass / tyres", "Tyre μ (lateral peak) [-]", "tyre:lateral.mu_y0", 0.6, 1.6, 0.01),
    ("Mass / tyres", "Tyre cornering stiffness pKy1 [-]", "tyre:lateral.pKy1", 6, 30, 0.5),
    ("Mass / tyres", "Tyre camber μ gain [-]", "tyre:lateral.camber_mu_gain", 0, 0.15, 0.005),
    ("Mass / tyres", "Tyre vertical stiffness [kN/m]", "tyre:geometry.vertical_stiffness", 120, 320, 5),
]

# group -> (setup tab, section heading)
TABS = {"Front geometry": ("Geometry", "Front axle"), "Rear geometry": ("Geometry", "Rear axle"),
        "Springs / ARB": ("Springs", "Springs & anti-roll bars"), "Dampers": ("Springs", "Dampers"),
        "Powertrain": ("Powertrain", "Gearbox, final drive & engine"), "Driver": ("Powertrain", "Shift points (driver)"),
        "Brakes": ("Brakes", "Hydraulics & pads"), "Brake cooling": ("Brakes", "Disc / drum heat"),
        "Mass / tyres": ("Mass", "Mass & tyres")}
# first entry of each one-of group (drop-down) = the car as it is
AS_IS = {"Brakes": "Solid discs 256 × 12.9, ATE ti calipers (as is)", "Brake pads": "Road pads (as is)",
         "Wheels & tyres": "Steel 5.5J × 13 ET18 (as is)", "Engine": "Twin Weber 40 DCOE, ~120 PS (as is)",
         "Gearbox": "Getrag 242 4-speed (as is)", "Final drive": "3.64 (as is)"}


def leaf(raw, path):
    d = raw
    for k in path.split("."):
        d = d[k]
    return d


def split_idx(path):
    m = re.match(r"(.*)\[(\d)\]$", path)
    return (m.group(1), int(m.group(2))) if m else (path, None)


def tyre_raw(raw):
    import yaml
    return yaml.safe_load(open(params.INPUTS / "tyres" / leaf(raw, "wheels.tyre_model")["value"], encoding="utf-8"))


def n_gears(raw):
    return len(leaf(raw, "driveline.gear_ratios")["value"])


def special_values(raw):
    """Values of the '@' sliders, derived from the parameter file."""
    hp = leaf(raw, "suspension.rear.hardpoints")
    o, i = np.array(hp["tsa_pivot_outer"]["value"], float), np.array(hp["tsa_pivot_inner"]["value"], float)
    k = i - o
    gears = list(leaf(raw, "driveline.gear_ratios")["value"])
    cpo = leaf(raw, "suspension.front.camber_plate_offset")["value"]
    out = {"@rear_sweep": float(np.degrees(np.arctan2(-k[0], -k[1]))),
           "@rear_dihedral": float(np.degrees(np.arctan2(k[2], np.hypot(k[0], k[1])))), "@rear_pivot_z": float(o[2]),
           "@camber_plate_in": -float(cpo[1]), "@caster_plate_back": -float(cpo[0]), "@torque_scale": 100.0}
    for n in range(1, 6):
        out[f"@gear_{n}"] = float(gears[n - 1]) if n <= len(gears) else 0.0
    return out


def base_value(raw, traw, path):
    if path.startswith("@"):
        return special_values(raw)[path]
    if path.startswith("tyre:"):
        return float(leaf(traw, path[5:])["value"])
    p, idx = split_idx(path)
    v = leaf(raw, p)["value"]
    return float(v[idx] if idx is not None else v)


def overrides(raw, values, bases):
    """Slider values -> (parameter overrides, tyre overrides). Only sliders moved away from their base count.
    values / bases: {path: value}; bases are the values for the chosen variants (shown rounded to 4 decimals)."""
    ov, tov, vec = {}, {}, {}
    changed = {p: v for p, v in values.items() if min(abs(v - bases[p]), abs(v - round(bases[p], 4))) > 1e-9}
    val = lambda p: values.get(p, bases[p])
    for path, v in changed.items():
        if path.startswith("tyre:"):
            tov[path[5:]] = v
        elif not path.startswith("@"):
            p, idx = split_idx(path)
            if idx is None:
                ov[p] = v
            else:
                vec.setdefault(p, list(leaf(raw, p)["value"]))[idx] = v
    ov.update(vec)
    if any(p in changed for p in ("@camber_plate_in", "@caster_plate_back")):
        ov["suspension.front.camber_plate_offset"] = [-val("@caster_plate_back"), -val("@camber_plate_in")]
    if any(p in changed for p in ("@rear_sweep", "@rear_dihedral", "@rear_pivot_z")):
        hp = leaf(raw, "suspension.rear.hardpoints")
        o = np.array(hp["tsa_pivot_outer"]["value"], float)
        i = np.array(hp["tsa_pivot_inner"]["value"], float)
        L = np.linalg.norm(i - o)
        sw, dh = np.radians(val("@rear_sweep")), np.radians(val("@rear_dihedral"))
        o[2] = val("@rear_pivot_z")
        i_new = o + L * np.array([-np.sin(sw) * np.cos(dh), -np.cos(sw) * np.cos(dh), np.sin(dh)])
        ov["suspension.rear.hardpoints.tsa_pivot_outer"] = [round(float(x), 2) for x in o]
        ov["suspension.rear.hardpoints.tsa_pivot_inner"] = [round(float(x), 2) for x in i_new]
    ng = n_gears(raw)
    if any(f"@gear_{n}" in changed for n in range(1, ng + 1)):
        ov["driveline.gear_ratios"] = [round(val(f"@gear_{n}"), 4) for n in range(1, ng + 1)]
    if "@torque_scale" in changed:
        k = val("@torque_scale") / 100
        ov["engine.torque"] = [round(t * k, 3) for t in leaf(raw, "engine.torque")["value"]]
    return ov, tov
