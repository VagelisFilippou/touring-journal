"""Driving lines within the road width: centre, own lane (keep right), minimum-curvature racing line.

The line is the road centreline shifted sideways by an offset n(s) (left +). For the lane and racing
lines the offsets minimise the summed squared curvature of the shifted line (second differences of the
points, a bounded linear least-squares problem), within the allowed band:
  centre : n = 0
  lane   : keep right — the car stays inside the right-hand lane (Greece drives on the right)
  full   : the whole road width (closed road, e.g. a hillclimb)
Minimum curvature is close to minimum time for a low-power car on a twisty road; it ignores braking
points and late apexes. Road width is not in our road files yet (OSM 'width' tags are rare), so it is
a parameter (default 6 m, a typical two-lane mountain road).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .roads import R_EARTH, Track, build_track

LINES = {"centre": "Centre of the road", "lane": "Own lane (keep right)", "full": "Racing line (full road width)"}
CAR_WIDTH = 1.59


def _frame(track: Track, ds: float):
    s = np.arange(0, track.s[-1], ds)
    x, y = np.interp(s, track.s, track.xy[:, 0]), np.interp(s, track.s, track.xy[:, 1])
    tx, ty = np.gradient(x, ds), np.gradient(y, ds)
    nrm = np.hypot(tx, ty)
    N = np.c_[-ty / nrm, tx / nrm]                                   # left normal
    return s, np.c_[x, y], N


def bounds(kind: str, width: float, margin: float = 0.3):
    half = width / 2 - CAR_WIDTH / 2 - margin
    if kind == "full":
        return -half, half
    if kind == "lane":
        lo, hi = -half, -(CAR_WIDTH / 2 + margin)
        return (lo, hi) if hi > lo else (-width / 4, -width / 4)
    return 0.0, 0.0


def offsets(track: Track, kind: str, width: float = 6.0, margin: float = 0.3, ds: float = 4.0, smooth: float = 0.02):
    """Sideways offsets n (m, left +) along track.s for the chosen line."""
    lo, hi = bounds(kind, width, margin)
    if kind == "centre" or hi - lo < 1e-6:
        return np.full_like(track.s, lo)
    s, C, N = _frame(track, ds)
    n = len(s)
    D2 = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    D1 = sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))
    # ends: keep the offset steady over the first/last ~20 m (start line / flying finish), otherwise the free
    # end lets the line swing across the road and creates a fake bend at the finish
    w_end = np.zeros(n - 1)
    w_end[:5] = w_end[-5:] = 3.0
    A = sparse.vstack([D2 @ sparse.diags(N[:, 0]), D2 @ sparse.diags(N[:, 1]), smooth * D1, sparse.diags(w_end) @ D1]).tocsr()
    b = -np.r_[D2 @ C[:, 0], D2 @ C[:, 1], np.zeros(n - 1), np.zeros(n - 1)]
    x = _fista(A, b, lo, hi)
    return np.interp(track.s, s, x)


def _fista(A, b, lo, hi, iters=4000, tol=1e-7):
    """Box-constrained least squares min ||A x - b||² by accelerated projected gradient (fast, no deps)."""
    AtA = (A.T @ A).tocsr()
    Atb = A.T @ b
    v = np.random.default_rng(0).standard_normal(AtA.shape[0])
    for _ in range(30):                                              # Lipschitz constant (power iteration)
        v = AtA @ v
        v /= np.linalg.norm(v)
    Lc = float(v @ (AtA @ v)) * 1.05
    x = np.clip(np.zeros(AtA.shape[0]), lo, hi)
    y, t = x.copy(), 1.0
    for _ in range(iters):
        xn = np.clip(y - (AtA @ y - Atb) / Lc, lo, hi)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = xn + (t - 1) / tn * (xn - x)
        if np.max(np.abs(xn - x)) < tol:
            x = xn
            break
        x, t = xn, tn
    return x


def _to_latlon(track: Track, xy):
    lat = track.lat0 + np.degrees(xy[:, 1] / R_EARTH)
    lon = track.lon0 + np.degrees(xy[:, 0] / (R_EARTH * np.cos(np.radians(track.lat0))))
    return np.c_[lat, lon]


def normals(track: Track):
    tx, ty = np.gradient(track.xy[:, 0], track.s), np.gradient(track.xy[:, 1], track.s)
    nrm = np.hypot(tx, ty)
    return np.c_[-ty / nrm, tx / nrm]


def line_track(track: Track, kind: str, width: float = 6.0, margin: float = 0.3):
    """-> (Track along the line, offsets, line xy in the centreline's local frame)."""
    if kind == "centre":                                               # identical to the road itself
        return track, np.zeros_like(track.s), track.xy
    n = offsets(track, kind, width, margin)
    xy = track.xy + normals(track) * n[:, None]
    ll = _to_latlon(track, xy)
    pts = np.c_[ll, track.elev]
    tr = build_track(pts, name=f"{track.name}:{kind}", smooth_m=2.0)    # already smooth; just re-parameterise
    return tr, n, xy


def edges(track: Track, width: float):
    N = normals(track)
    return track.xy + N * width / 2, track.xy - N * width / 2


def key_points(track_line: Track, res, base: Track):
    """Gear changes and braking points along a simulated line, with positions in the base road's x/y frame.
    res = lapsim Result on track_line."""
    from .roads import _xy
    xy = _xy(track_line.latlon, base.lat0, base.lon0)
    g = np.asarray(res.gear)
    shifts = []
    for i in np.where(np.diff(g) != 0)[0] + 1:
        shifts.append({"s": float(track_line.s[i]), "from": int(g[i - 1]), "to": int(g[i]), "v": float(res.v[i] * 3.6),
                       "latlon": track_line.latlon[i].round(6).tolist(), "xy": xy[i].round(2).tolist()})
    brakes = []
    lim = np.asarray(res.limit)
    on = lim == "brake"
    starts = np.where(on[1:] & ~on[:-1])[0] + 1
    for i in starts:
        j = i
        while j < len(on) - 1 and on[j]:
            j += 1
        k = i + int(np.argmin(res.v[i:j + 1]))                      # slowest point after braking (the apex)
        brakes.append({"s": float(track_line.s[i]), "v_in": float(res.v[i] * 3.6), "v_min": float(res.v[k] * 3.6),
                       "s_apex": float(track_line.s[k]), "latlon": track_line.latlon[i].round(6).tolist(),
                       "xy": xy[i].round(2).tolist(), "apex_xy": xy[k].round(2).tolist(), "gear": int(g[k])})
    return shifts, brakes
