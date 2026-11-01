"""Tyre model: 'MF-lite' (Magic Formula 5.2 subset). See inputs/tyres/*.yaml for the equations.

Sign convention here is *magnitude-based* for steady state: alpha > 0 gives Fy > 0 toward the
turn centre. Camber `gamma_in` > 0 means the wheel top leans toward the turn centre.
To be replaced by a full MF 6.1 evaluator when a proper parameter set is available.
"""
from __future__ import annotations

import numpy as np

from .params import load_tyre


class Tyre:
    def __init__(self, fname: str, overrides: dict | None = None):
        t = load_tyre(fname, overrides)
        self.t = t
        L = t.lateral
        self.Fz0, self.mu0, self.dmu = L.Fz0, L.mu_y0, L.dmu_y
        self.C, self.E, self.pKy1, self.pKy2 = L.C, L.E, L.pKy1, L.pKy2
        self.cg = L.camber_stiffness_per_load
        self.mu_cam_gain, self.cam_opt = L.camber_mu_gain, L.camber_opt
        self.kz = t.geometry.vertical_stiffness
        self.R0 = t.geometry.unloaded_radius

    def mu_y(self, Fz, gamma_in=0.0):
        dfz = (Fz - self.Fz0) / self.Fz0
        g = gamma_in / self.cam_opt
        cam = 1 + self.mu_cam_gain * (2 * g - g * g)        # parabola: max gain at cam_opt
        return self.mu0 * (1 + self.dmu * dfz) * cam

    def Kya(self, Fz):
        return self.pKy1 * self.Fz0 * np.sin(2 * np.arctan(Fz / (self.pKy2 * self.Fz0)))

    def Fy(self, alpha, Fz, gamma_in=0.0):
        Fz = np.maximum(Fz, 1.0)
        D = self.mu_y(Fz, gamma_in) * Fz
        K = self.Kya(Fz)
        B = K / (self.C * D)
        a = alpha + self.cg * Fz * gamma_in / K               # camber as equivalent slip
        x = B * a
        return D * np.sin(self.C * np.arctan(x - self.E * (x - np.arctan(x))))

    def alpha_peak(self, Fz, gamma_in=0.0):
        """Slip angle at peak Fy (numerical)."""
        al = np.linspace(0, 0.35, 701)
        return al[np.argmax(self.Fy(al, Fz, gamma_in))]
