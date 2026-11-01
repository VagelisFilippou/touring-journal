"""Realistic display geometry: 3D primitives built from the kinematic points (display only).

Shapes and proportions follow the ETK drawings (31/2 front axle carrier + control arm, 31/6 strut,
33/27 rear axle carrier, trailing arm) and known part sizes (256 mm disc, 250 mm drum, 115 mm mean
spring diameter, 185 mm tyre). Hardpoints come from the model; part *shapes* between them are
indicative.

Primitive = dict(kind, ...) in metres, body frame:
  tube   p0, p1, r          cylinder (also discs: short tube along the disc normal)
  sphere c, r
  box    c, h (half sizes)  axis-aligned
  poly   pts (N,3)          filled plate
  line   pts (N,3), lw      polyline (e.g. coil helix)
Each has: color, alpha (default 1), part (name for legends/toggles).
"""
from __future__ import annotations

import numpy as np

COL = dict(strut="#2c5d8a", rod="#9aa7b4", spring="#2e8b57", rubber="#3b3b3b", steel="#7f8c8d",
           disc="#b0b6bb", caliper="#c0392b", arm="#34495e", steer="#a04000", arb="#7d3c98",
           tyre="#262626", rim="#c8ccd0", body="#95a5a6", drum="#6e7377", diff="#5d6d7e",
           shaft="#566573", damper="#d68910", frame="#839192")

EZ = np.array([0.0, 0.0, 1.0])


def _n(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-15)


def tube(p0, p1, r, color, part="", alpha=1.0):
    return dict(kind="tube", p0=np.asarray(p0, float), p1=np.asarray(p1, float), r=r, color=COL.get(color, color),
                part=part, alpha=alpha)


def disk(c, n, r, t, color, part="", alpha=1.0):
    n = _n(n)
    c = np.asarray(c, float)
    return tube(c - n * t / 2, c + n * t / 2, r, color, part, alpha)


def sphere(c, r, color, part=""):
    return dict(kind="sphere", c=np.asarray(c, float), r=r, color=COL.get(color, color), part=part, alpha=1.0)


def box(c, h, color, part="", alpha=1.0):
    return dict(kind="box", c=np.asarray(c, float), h=np.asarray(h, float), color=COL.get(color, color), part=part, alpha=alpha)


def plate(pts, color, part="", alpha=1.0):
    return dict(kind="poly", pts=np.asarray(pts, float), color=COL.get(color, color), part=part, alpha=alpha)


def line(pts, color, lw=1.5, part=""):
    return dict(kind="line", pts=np.asarray(pts, float), color=COL.get(color, color), lw=lw, part=part, alpha=1.0)


def helix(p0, p1, R, coils, wire_r, color="spring", part="spring"):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = _n(p1 - p0)
    e1 = _n(np.cross(ax, [1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.cross(ax, [0, 1.0, 0]))
    e2 = np.cross(ax, e1)
    t = np.linspace(0, 1, int(coils * 24) + 1)
    ang = 2 * np.pi * coils * t
    pts = p0 + np.outer(t, p1 - p0) + R * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))
    d = line(pts, color, lw=2.2, part=part)
    d["wire_r"] = wire_r
    return d


def forward_dir(a):
    """Unit vector in the wheel plane pointing forward (+x)."""
    fw = np.cross(a, EZ)
    fw = _n(fw)
    return fw if fw[0] >= 0 else -fw


def up_dir(a):
    return _n(EZ - a[2] * a)


def inboard(a, wc):
    """Spin-axis direction pointing towards the car centre."""
    return a if a[1] * wc[1] < 0 else -a


# ----------------------------------------------------------------------------------------
def wheel(P, R0, brake="disc"):
    a, wc, hub = P["a"], P["wc"], P["hub"]
    ib = inboard(a, wc)
    prims = [disk(wc, a, R0, 0.165, "tyre", "tyre"),                 # tread band
             disk(wc, a, R0 - 0.014, 0.187, "tyre", "tyre"),         # sidewall bulge
             disk(wc, a, 0.168, 0.145, "rim", "rim"),                # 13" rim
             disk(hub, a, 0.07, 0.012, "steel", "hub")]
    if brake == "disc":
        dc = hub + ib * 0.035
        prims.append(disk(dc, a, 0.128, 0.0129, "disc", "brake"))
        cal = dc - forward_dir(a) * 0.098 + up_dir(a) * 0.02         # caliper behind the axle (indicative)
        prims.append(disk(cal, a, 0.04, 0.075, "caliper", "brake"))
    else:
        prims.append(disk(hub + ib * 0.04, a, 0.125, 0.055, "drum", "brake"))
    return prims


def front_corner(P, arb_d):
    u = P["axis_u"]
    foot, top = P["strut_foot"], P["strut_top"]
    tube_top = foot + u * 0.40
    pr = [
        tube(foot - u * 0.03, tube_top, 0.027, "strut", "strut"),
        tube(tube_top, top - u * 0.03, 0.011, "rod", "strut"),
        tube(top - u * 0.16, top - u * 0.09, 0.022, "rubber", "strut"),          # bump stop
        disk(P["perch"], u, 0.074, 0.006, "steel", "strut"),
        disk(top - u * 0.035, u, 0.066, 0.008, "steel", "strut"),
        disk(top, u, 0.06, 0.03, "rubber", "top mount"),
        disk(top + u * 0.025, u, 0.09, 0.008, "body", "top mount", alpha=.6),
        helix(P["perch"] + u * 0.006, top - u * 0.04, 0.0575, 6.5, 0.0059),
        tube(foot, P["hub"], 0.021, "strut", "knuckle"),                          # spindle
        tube(foot + u * 0.02, P["tro"], 0.012, "steer", "steering"),              # steering arm
        tube(P["tri"], P["tro"], 0.011, "steer", "steering"),
        sphere(P["tro"], 0.02, "steer"), sphere(P["tri"], 0.02, "steer"),
        tube(P["lca_inner"], P["bj"], 0.017, "arm", "lower arm"),
        sphere(P["bj"], 0.026, "steel", "lower arm"),
        disk(P["lca_inner"], [1, 0, 0], 0.032, 0.06, "rubber", "lower arm"),
        tube(P["bj"], foot, 0.016, "strut", "knuckle"),
    ]
    mid = 0.5 * (P["ts_arm"] + P["ts_body"]) + np.r_[0, 0, -0.03]                 # bent tension strut (ETK 31/2 #11)
    pr += [tube(P["ts_arm"], mid, 0.011, "arm", "tension strut"), tube(mid, P["ts_body"], 0.011, "arm", "tension strut"),
           disk(P["ts_body"], P["ts_body"] - mid, 0.036, 0.055, "rubber", "tension strut")]
    if arb_d > 0:
        pr.append(tube(P["arb"], P["arb"] + np.r_[0, 0, 0.07], 0.006, "arb", "ARB"))
    return pr


def front_axle_common(L, R, arb_d, hp_lca_z):
    pr = [tube(L["tri"], R["tri"], 0.013, "steer", "steering")]                   # centre link
    box_c = L["tri"] + np.r_[0.03, -0.02, 0.06]
    pr.append(box(box_c, [0.05, 0.045, 0.06], "steer", "steering"))               # steering box (LHD)
    pr.append(box(R["tri"] + np.r_[0.03, 0.02, 0.05], [0.03, 0.03, 0.05], "steer", "steering"))  # idler
    if arb_d > 0:
        a, b = L["arb"] + np.r_[0, 0, 0.07], R["arb"] + np.r_[0, 0, 0.07]
        f = 0.17
        pts = [a, a + np.r_[f, -0.03, 0.02], b + np.r_[f, 0.03, 0.02], b]
        pr += [tube(pts[i], pts[i + 1], arb_d / 2, "arb", "ARB") for i in range(3)]
    # front axle carrier (crossmember, ETK 31/2): beam + engine-mount towers + arms to A mounts
    z0 = hp_lca_z
    pr += [box([0.01, 0, z0 + 0.02], [0.055, 0.40, 0.045], "frame", "crossmember"),
           box([0.0, 0.21, z0 + 0.13], [0.05, 0.05, 0.09], "frame", "crossmember"),
           box([0.0, -0.21, z0 + 0.13], [0.05, 0.05, 0.09], "frame", "crossmember")]
    for s in (1, -1):
        # beam end rises to the rail; carrier end plate runs along the rail between bolt A (x +220)
        # and the two rear bolts (x ~0) — MAN-FRAME 251.8 / 224.5
        pr.append(tube([0.02, s * 0.37, z0 + 0.03], [0.02, s * 0.3405, 0.20], 0.03, "frame", "crossmember"))
        pr.append(box([0.11, s * 0.3405, 0.205], [0.125, 0.035, 0.015], "frame", "crossmember"))
        # front rail: visible in MAN-FRAME plan from x ≈ +465 to -170 (centre y ≈ 358, ~80 wide), then it
        # diverges under the floor to about (-970, ±490)
        pr.append(box([0.1475, s * 0.358, 0.255], [0.3175, 0.04, 0.045], "frame", "rails"))
        pr.append(tube([-0.17, s * 0.358, 0.25], [-0.97, s * 0.49, 0.22], 0.04, "frame", "rails"))
    return pr


def rear_corner(P, arb_d):
    a, wc, hub = P["a"], P["wc"], P["hub"]
    ib = inboard(a, wc)
    fw = forward_dir(a)
    hub_in = hub + ib * 0.07
    sl = P["spring_lower"]
    po, pi = P["pivot_outer"], P["pivot_inner"]
    arm = [po, hub_in + fw * 0.09 + np.r_[0, 0, -0.03], hub_in - fw * 0.07 + np.r_[0, 0, -0.04],
           sl + np.r_[-0.06, 0, -0.02], sl + np.r_[0.02, -np.sign(wc[1]) * 0.07, -0.02], pi]
    pr = [plate(arm, "arm", "trailing arm", alpha=.85)]
    pr += [tube(arm[i], arm[(i + 1) % len(arm)], 0.012, "arm", "trailing arm") for i in range(len(arm))]
    k = pi - po
    pr += [disk(po, k, 0.034, 0.05, "rubber", "trailing arm"), disk(pi, k, 0.034, 0.05, "rubber", "trailing arm"),
           disk(sl, EZ, 0.072, 0.008, "steel", "spring"),
           disk(P["spring_upper"], EZ, 0.075, 0.012, "body", "spring", alpha=.6),
           helix(sl + EZ * 0.008, P["spring_upper"] - EZ * 0.01, 0.0575, 6.3, 0.0066)]
    dl, du = P["damper_lower"], P["damper_upper"]
    pr += [tube(dl, dl + 0.58 * (du - dl), 0.024, "damper", "damper"), tube(dl + 0.58 * (du - dl), du, 0.008, "rod", "damper"),
           sphere(dl, 0.018, "rubber"), disk(du, du - dl, 0.04, 0.03, "rubber", "damper")]
    diff_out = np.array([wc[0], np.sign(wc[1]) * 0.13, 0.285])
    pr += [tube(diff_out, hub_in, 0.014, "shaft", "halfshaft"), sphere(diff_out, 0.04, "shaft"), sphere(hub_in, 0.04, "shaft")]
    if arb_d > 0:
        pr.append(tube(P["arb"], P["arb"] + np.r_[0, 0, 0.08], 0.006, "arb", "ARB"))
    return pr


def rear_axle_common(L, R, arb_d, sub_mount):
    xw = L["wc"][0]
    pr = [box([xw, 0, 0.285], [0.15, 0.12, 0.12], "diff", "diff"),
          tube([xw + 0.12, 0, 0.30], [xw + 0.36, 0, 0.32], 0.06, "diff", "diff"),
          disk([xw - 0.15, 0, 0.285], [1, 0, 0], 0.12, 0.02, "diff", "diff")]
    centre = 0.5 * (L["pivot_inner"] + R["pivot_inner"]) + np.r_[-0.05, 0, -0.01]
    for P, s in ((L, 1), (R, -1)):
        sm = np.asarray(sub_mount) * [1, s, 1]
        pr += [tube(sm, P["pivot_outer"], 0.035, "frame", "rear axle carrier"),
               tube(P["pivot_outer"], P["pivot_inner"], 0.035, "frame", "rear axle carrier"),
               tube(P["pivot_inner"], centre, 0.035, "frame", "rear axle carrier"),
               disk(sm, EZ, 0.045, 0.06, "rubber", "rear axle carrier")]
    if arb_d > 0:
        a, b = L["arb"] + np.r_[0, 0, 0.08], R["arb"] + np.r_[0, 0, 0.08]
        pts = [a, a + np.r_[-0.12, -0.05, 0.03], b + np.r_[-0.12, 0.05, 0.03], b]
        pr += [tube(pts[i], pts[i + 1], arb_d / 2, "arb", "ARB") for i in range(3)]
    return pr


# ----------------------------------------------------------------------------------------
# 2D projection
VIEWS = {  # (horizontal axis index, vertical axis index, depth axis index, depth sign: + = nearer viewer)
    "front": (1, 2, 0, 1),     # viewer in front of the car
    "rear": (1, 2, 0, -1),     # viewer behind
    "side": (0, 2, 1, 1),      # viewer on the left (+y) side
    "plan": (0, 1, 2, 1),      # viewer above
}


def _circle_pts(c, n, r, k=18):
    n = _n(n)
    e1 = _n(np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0]))
    e2 = np.cross(n, e1)
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return c + r * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))


def _hull(p2):
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(p2)
        return p2[h.vertices]
    except Exception:
        return p2


def project(prim, view):
    """-> (kind2d, data2d, depth) with data in metres."""
    h, v, d, sg = VIEWS[view]
    k = prim["kind"]
    if k == "tube":
        n = prim["p1"] - prim["p0"]
        pts = np.vstack([_circle_pts(prim["p0"], n, prim["r"]), _circle_pts(prim["p1"], n, prim["r"])])
        return "fill", _hull(pts[:, [h, v]]), sg * pts[:, d].mean()
    if k == "sphere":
        t = np.linspace(0, 2 * np.pi, 20)
        c = prim["c"]
        return "fill", np.c_[c[h] + prim["r"] * np.cos(t), c[v] + prim["r"] * np.sin(t)], sg * c[d]
    if k == "box":
        c, hh = prim["c"], prim["h"]
        corners = np.array([[c[0] + i * hh[0], c[1] + j * hh[1], c[2] + l * hh[2]] for i in (-1, 1) for j in (-1, 1) for l in (-1, 1)])
        return "fill", _hull(corners[:, [h, v]]), sg * c[d]
    if k == "poly":
        return "fill", prim["pts"][:, [h, v]], sg * prim["pts"][:, d].mean()
    return "line", prim["pts"][:, [h, v]], sg * prim["pts"][:, d].mean()


def render2d(ax, prims, view, scale=1e3, edge=True):
    items = [(project(p, view), p) for p in prims]
    items.sort(key=lambda it: it[0][2])                  # far first
    for z, ((kind, xy, depth), p) in enumerate(items):
        xy = xy * scale
        if kind == "fill":
            ax.fill(xy[:, 0], xy[:, 1], color=p["color"], alpha=p["alpha"], zorder=2 + z * 0.01,
                    ec="#1b1b1b" if edge else "none", lw=0.3)
        else:
            ax.plot(xy[:, 0], xy[:, 1], color=p["color"], lw=p.get("lw", 1.5), zorder=2 + z * 0.01)


def render3d(ax, prims, scale=1e3):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for p in prims:
        k = p["kind"]
        if k == "tube":
            n = p["p1"] - p["p0"]
            L = np.linalg.norm(n)
            if L < 0.6 * p["r"]:      # disc-like: draw two rims
                for c in (p["p0"], p["p1"]):
                    pts = _circle_pts(c, n, p["r"], 36)
                    pts = np.vstack([pts, pts[:1]]) * scale
                    ax.plot(*pts.T, color=p["color"], lw=0.8)
            else:
                pts = np.vstack([p["p0"], p["p1"]]) * scale
                ax.plot(*pts.T, color=p["color"], lw=max(1.0, p["r"] * 160), solid_capstyle="round")
        elif k == "sphere":
            ax.scatter(*(p["c"] * scale), color=p["color"], s=max(6, p["r"] * 900))
        elif k == "box":
            c, h = p["c"], p["h"]
            for i in (-1, 1):
                for j in (-1, 1):
                    for axis in range(3):
                        a0 = c.copy(); a1 = c.copy()
                        o = [i, j]
                        others = [q for q in range(3) if q != axis]
                        for q, s in zip(others, o):
                            a0[q] += s * h[q]; a1[q] += s * h[q]
                        a0[axis] -= h[axis]; a1[axis] += h[axis]
                        ax.plot(*(np.vstack([a0, a1]) * scale).T, color=p["color"], lw=0.8)
        elif k == "poly":
            ax.add_collection3d(Poly3DCollection([p["pts"] * scale], facecolor=p["color"], alpha=0.5 * p["alpha"], edgecolor="k", lw=.3))
        else:
            ax.plot(*(p["pts"] * scale).T, color=p["color"], lw=p.get("lw", 1.5))


def car_prims(pose_pts, p, R0, include=("front", "rear")):
    """All realistic primitives for the axles in `include` (pose_pts from the GUI)."""
    pr = []
    s = p.suspension
    if "front" in include:
        L, R, *_ = pose_pts["front"]
        for P in (L, R):
            pr += wheel(P, R0, "disc") + front_corner(P, s.front.arb.diameter)
        pr += front_axle_common(L, R, s.front.arb.diameter, s.front.hardpoints.lca_inner[2])
    if "rear" in include:
        L, R, *_ = pose_pts["rear"]
        for P in (L, R):
            pr += wheel(P, R0, "drum") + rear_corner(P, s.rear.arb.diameter)
        pr += rear_axle_common(L, R, s.rear.arb.diameter, s.rear.hardpoints.subframe_mount)
    return pr
