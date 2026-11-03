"""Parameter loading for the BMW 02 vehicle dynamics model.

Input files are YAML. Every physical value is a *parameter leaf*:

    wheelbase:
      value: 2500        # number or [x, y, z] list
      unit: mm           # see UNITS below; '-' for dimensionless
      src: [HOM-5331]    # source ids, defined in inputs/sources.yaml
      conf: A            # A factory doc, B credible secondary, C forum/estimate, D placeholder
      note: free text

Variants (inputs/variants/*.yaml) only list what changes, by dotted path:

    meta: {name: lowered_30, base: baseline.yaml, description: ...}
    set:
      suspension.front.spring.rate: {value: 35, unit: N/mm, src: [...], conf: C}
      chassis.ride_height_drop.front: 30      # bare number = keep unit/src of base
    variants: [other_variant.yaml]            # optional, applied first

load() returns a Params object: attribute access to SI floats/arrays
(p.chassis.wheelbase -> 2.5), plus p.leaf('chassis.wheelbase') for metadata.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

INPUTS = Path(__file__).resolve().parent.parent / "inputs"

# unit -> factor to SI (m, kg, s, N, rad, W)
LBF, IN = 4.4482216, 0.0254
UNITS = {
    "-": 1.0, "%": 0.01, "ratio": 1.0,
    "m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": IN,
    "m2": 1.0,
    "kg": 1.0, "lb": 0.45359237,
    "N": 1.0, "kN": 1e3, "kgf": 9.80665, "lbf": LBF,
    "N/m": 1.0, "N/mm": 1e3, "kN/m": 1e3, "lbf/in": LBF / IN, "kgf/mm": 9.80665e3,
    "Nm": 1.0, "kgfm": 9.80665, "Nm/rad": 1.0, "Nm/deg": 180 / np.pi,
    "Ns/m": 1.0, "Ns/mm": 1e3,
    "rad": 1.0, "deg": np.pi / 180, "arcmin": np.pi / 180 / 60,
    "1/rad": 1.0, "1/deg": 180 / np.pi,
    "rpm": 2 * np.pi / 60, "rad/s": 1.0,
    "m/s": 1.0, "km/h": 1 / 3.6,
    "W": 1.0, "kW": 1e3, "PS": 735.49875, "hp": 745.69987,
    "Pa": 1.0, "kPa": 1e3, "bar": 1e5, "psi": 6894.757,
    "kgm2": 1.0, "L": 1e-3, "cc": 1e-6, "kg/m3": 1.0, "s": 1.0, "Hz": 1.0,
    "degC": 1.0, "W/m2K": 1.0, "J/kgK": 1.0,          # temperatures stay in °C (brakes)
}
CONF_LEVELS = ("A", "B", "C", "D")


def _is_leaf(d) -> bool:
    return isinstance(d, dict) and "value" in d


def _to_si(leaf: dict, path: str):
    unit = leaf.get("unit", "-")
    if unit not in UNITS:
        raise ValueError(f"{path}: unknown unit '{unit}'")
    v = leaf["value"]
    if v is None:
        return None
    if isinstance(v, str):  # categorical parameter (e.g. tyre file name, diff type)
        return v
    arr = np.asarray(v, dtype=float) * UNITS[unit]
    return float(arr) if arr.ndim == 0 else arr


class Node:
    """Attribute view on a nested dict of SI values."""

    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, k):
        try:
            v = self._d[k]
        except KeyError:
            raise AttributeError(k) from None
        return Node(v) if isinstance(v, dict) else v

    def __getitem__(self, k):
        return self.__getattr__(k)

    def __contains__(self, k):
        return k in self._d

    def keys(self):
        return self._d.keys()

    def __repr__(self):
        return f"Node({list(self._d)})"


class Params(Node):
    def __init__(self, raw: dict, name: str, applied: list[str]):
        self.raw = raw
        self.name = name
        self.applied = applied
        super().__init__(self._convert(raw, ""))

    def _convert(self, d, path):
        out = {}
        for k, v in d.items():
            p = f"{path}.{k}" if path else k
            if _is_leaf(v):
                out[k] = _to_si(v, p)
            elif isinstance(v, dict):
                out[k] = self._convert(v, p)
            else:
                out[k] = v  # plain structural value (strings, lists of names)
        return out

    def leaf(self, path: str) -> dict:
        d = self.raw
        for k in path.split("."):
            d = d[k]
        return d

    def leaves(self):
        """Yield (path, leaf) for every parameter, for reporting."""
        def walk(d, path):
            for k, v in d.items():
                p = f"{path}.{k}" if path else k
                if _is_leaf(v):
                    yield p, v
                elif isinstance(v, dict):
                    yield from walk(v, p)
        yield from walk(self.raw, "")


_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)   # libyaml when available (~6x faster); Pyodide falls back


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_LOADER) or {}


def _set_path(d: dict, path: str, new, origin: str):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    old = d.get(keys[-1])
    if _is_leaf(new):
        leaf = dict(new)
    elif _is_leaf(old):  # bare value: keep unit, mark as variant-set
        leaf = dict(old, value=new)
    else:
        raise ValueError(f"{origin}: '{path}' is new, give a full leaf with unit/src/conf")
    leaf["set_by"] = origin
    d[keys[-1]] = leaf


def load(*variants: str, base: str | None = None, overrides: dict | None = None) -> Params:
    """Load baseline (or `base`), apply variant files in order, then `overrides`
    ({dotted.path: value-or-leaf}, e.g. for sensitivity studies)."""
    applied: list[str] = []

    def apply(raw, vfile):
        vpath = INPUTS / "variants" / vfile
        v = _read(vpath)
        for sub in v.get("variants", []):
            apply(raw, sub)
        for path, val in (v.get("set") or {}).items():
            _set_path(raw, path, val, vfile)
        applied.append(vfile)

    first = _read(INPUTS / "variants" / variants[0]) if variants else {}
    base = base or first.get("meta", {}).get("base", "baseline.yaml")
    raw = copy.deepcopy(_read(INPUTS / base))
    for vf in variants:
        apply(raw, vf)
    for path, val in (overrides or {}).items():
        _set_path(raw, path, val, "override")
    name = "+".join([Path(base).stem] + [Path(v).stem for v in variants] + (["override"] if overrides else []))
    p = Params(raw, name, applied)
    validate(p)
    return p


GROUP_ORDER = ["Chassis", "Wheels & tyres", "Brakes", "Brake pads", "Engine", "Gearbox", "Final drive", "Combinations", "My setups"]
EXCLUSIVE_GROUPS = {"Engine", "Gearbox", "Final drive", "Wheels & tyres", "Brakes", "Brake pads"}


_CATALOG = {}


def variant_catalog() -> list[dict]:
    """All variant files with their group and label (meta.group / meta.label); unknown -> 'My setups'.
    Cached; re-read only when a variant file is added, removed or changed (the GUI asks on every recompute)."""
    files = sorted((INPUTS / "variants").glob("*.yaml"))
    key = tuple((f.name, f.stat().st_mtime_ns) for f in files)
    if key not in _CATALOG:
        _CATALOG.clear()
        _CATALOG[key] = _variant_catalog(files)
    return [dict(v) for v in _CATALOG[key]]


def _variant_catalog(files) -> list[dict]:
    out = []
    for f in files:
        meta = (_read(f).get("meta") or {})
        g = meta.get("group", "My setups")
        out.append({"file": f.name, "group": g, "label": meta.get("label", f.stem.replace("_", " ")),
                    "description": meta.get("description", ""), "exclusive": g in EXCLUSIVE_GROUPS})
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    return sorted(out, key=lambda d: (order.get(d["group"], 99), d["label"]))


def load_tyre(fname: str, overrides: dict | None = None) -> Params:
    """Load a tyre parameter file from inputs/tyres (same leaf format), optional overrides."""
    raw = _read(INPUTS / "tyres" / fname)
    for path, val in (overrides or {}).items():
        _set_path(raw, path, val, "override")
    t = Params(raw, Path(fname).stem, [])
    validate(t)
    return t


def validate(p: Params):
    srcs = _read(INPUTS / "sources.yaml").get("sources", {})
    problems = []
    for path, leaf in p.leaves():
        if leaf.get("conf") not in CONF_LEVELS:
            problems.append(f"{path}: conf must be one of {CONF_LEVELS}")
        for s in leaf.get("src") or []:
            if s not in srcs:
                problems.append(f"{path}: unknown source id '{s}'")
    if problems:
        raise ValueError("Parameter file problems:\n  " + "\n  ".join(problems))


def report(p: Params, only_conf: str | None = None) -> str:
    """Plain-text table of all parameters (path, value, unit, conf, src)."""
    rows = []
    for path, leaf in p.leaves():
        if only_conf and leaf.get("conf") != only_conf:
            continue
        rows.append(f"{path:55s} {str(leaf['value']):>24s} {leaf.get('unit','-'):8s} "
                    f"{leaf.get('conf','?')}  {','.join(leaf.get('src') or [])}"
                    + (f"  [{leaf['set_by']}]" if "set_by" in leaf else ""))
    return "\n".join(rows)


if __name__ == "__main__":
    import sys
    print(report(load(*sys.argv[1:])))
