"""Roads for the lap/route simulation: geometry from OpenStreetMap (via OSRM routing), elevation from
Open-Meteo (Copernicus DEM, 90 m). Road files live in inputs/roads/*.yaml (lat/lon/elevation points +
provenance), so simulations run offline and are reproducible.

Track model for the simulation: distance s (m), curvature kappa (1/m, + left), gradient (rise/run),
elevation (m), resampled every `ds` metres after smoothing (OSM polylines are coarse; raw point-to-point
curvature is very noisy, so the line is smoothed with a fitting spline before differentiating).
"""
from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROADS = Path(__file__).resolve().parent.parent / "inputs" / "roads"
UA = {"User-Agent": "BMW1600-vehicle-model/0.1 (personal research)"}
R_EARTH = 6371000.0


# ---------------------------------------------------------------- online sources
def osrm_route(waypoints):
    """waypoints [(lat, lon), ...] -> list of (lat, lon) along the roads (OSRM public demo, OSM data)."""
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in waypoints)
    url = f"https://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson&continue_straight=true"
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
    if d.get("code") != "Ok":
        raise RuntimeError(f"OSRM: {d.get('code')} {d.get('message', '')}")
    return [(lat, lon) for lon, lat in d["routes"][0]["geometry"]["coordinates"]]


def elevations(latlon):
    """Open-Meteo elevation API (Copernicus DEM GLO-90), 100 points per request."""
    out = []
    for i in range(0, len(latlon), 100):
        chunk = latlon[i:i + 100]
        q = urllib.parse.urlencode({"latitude": ",".join(f"{a:.6f}" for a, _ in chunk),
                                    "longitude": ",".join(f"{b:.6f}" for _, b in chunk)})
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.open-meteo.com/v1/elevation?{q}", headers=UA), timeout=60))
        out += d["elevation"]
    return out


def densify(latlon, step=10.0):
    """Insert points so no segment is longer than `step` m (elevation sampling)."""
    out = [latlon[0]]
    for (a0, b0), (a1, b1) in zip(latlon[:-1], latlon[1:]):
        x, y = _xy(np.array([[a0, b0], [a1, b1]]), a0, b0)[1]
        n = max(1, int(math.hypot(x, y) // step))
        out += [(a0 + (a1 - a0) * k / n, b0 + (b1 - b0) * k / n) for k in range(1, n + 1)]
    return out


def fetch_road(name, waypoints, description="", note=""):
    """Route through waypoints, add elevations, save inputs/roads/<name>.yaml."""
    pts = densify(osrm_route(waypoints), 10.0)
    ele = elevations(pts)
    data = {"meta": {"name": name, "description": description, "note": note,
                     "geometry": "OpenStreetMap via OSRM (router.project-osrm.org), ODbL",
                     "elevation": "Open-Meteo elevation API (Copernicus DEM GLO-90)",
                     "waypoints": [[round(a, 6), round(b, 6)] for a, b in waypoints]},
            "points": [[round(a, 6), round(b, 6), round(float(e), 1)] for (a, b), e in zip(pts, ele)]}
    ROADS.mkdir(exist_ok=True)
    (ROADS / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=200), encoding="utf-8")
    return data


# ---------------------------------------------------------------- track
def _xy(latlon, lat0, lon0):
    lat, lon = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    x = (lon - math.radians(lon0)) * math.cos(math.radians(lat0)) * R_EARTH
    y = (lat - math.radians(lat0)) * R_EARTH
    return np.c_[x, y]


@dataclass
class Track:
    name: str
    s: np.ndarray
    kappa: np.ndarray
    grade: np.ndarray
    elev: np.ndarray
    xy: np.ndarray
    latlon: np.ndarray

    @property
    def length(self):
        return float(self.s[-1])


def build_track(latlon_ele, name="road", ds=2.0, smooth_m=12.0, elev_smooth_m=150.0):
    """latlon_ele: array (N,3) lat, lon, elevation -> Track (curvature from a smoothing spline)."""
    from scipy.interpolate import splev, splprep
    from scipy.ndimage import gaussian_filter1d
    P = np.asarray(latlon_ele, float)
    xy = _xy(P[:, :2], P[0, 0], P[0, 1])
    keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 0.5]      # drop duplicate points
    xy, P = xy[keep], P[keep]
    s_raw = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    # resample uniformly, then smooth x/y with a Gaussian of `smooth_m` metres before differentiating
    s = np.arange(0, s_raw[-1], ds)
    x, y = np.interp(s, s_raw, xy[:, 0]), np.interp(s, s_raw, xy[:, 1])
    sig = smooth_m / ds
    xs, ys = gaussian_filter1d(x, sig, mode="nearest"), gaussian_filter1d(y, sig, mode="nearest")
    dx, dy = np.gradient(xs, ds), np.gradient(ys, ds)
    ddx, ddy = np.gradient(dx, ds), np.gradient(dy, ds)
    kappa = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    z = gaussian_filter1d(np.interp(s, s_raw, P[:, 2]), elev_smooth_m / ds, mode="nearest")
    grade = np.gradient(z, ds)
    lat = np.interp(s, s_raw, P[:, 0])
    lon = np.interp(s, s_raw, P[:, 1])
    return Track(name, s, kappa, grade, z, np.c_[xs, ys], np.c_[lat, lon])


def load_road(name, **kw):
    d = yaml.safe_load((ROADS / f"{name}.yaml").read_text(encoding="utf-8"))
    return build_track(d["points"], name=name, **kw), d["meta"]


def straight(length=1000.0, grade=0.0, name="straight"):
    s = np.arange(0, length + 1, 2.0)
    return Track(name, s, np.zeros_like(s), np.full_like(s, grade), s * grade, np.c_[s, 0 * s], np.zeros((len(s), 2)))
