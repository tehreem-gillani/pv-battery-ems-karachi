"""
PV module and array, single-diode model with an explicit (Lambert W) solution.

    I = Iph - I0*(exp((V + I*Rs)/a) - 1) - (V + I*Rs)/Rsh,   a = n*Ns*Vt

Solving for I with the Lambert W function avoids an iterative solver at every
time step. W(A*exp(B)) is evaluated as wrightomega(ln A + B) so the large
exponent never has to be formed.

Module parameters resemble a 550 W, 144 half-cut-cell mono-Si module
(electrically 72 cells in series). They were tuned so that the STC values land
close to a typical datasheet (Isc 14.0 A, Voc 49.8 V, Pmp about 550 W). They
are not a fit to one specific product.
"""
import numpy as np
from scipy.special import wrightomega

K_B = 1.380649e-23
Q = 1.602176634e-19
EG = 1.12          # eV, silicon
T_STC = 298.15


class Module:
    def __init__(self, isc=14.0, voc=49.8, ns=72, n=1.0, rs=0.20, rsh=450.0,
                 alpha_isc=0.00048, noct=45.0):
        self.isc, self.voc, self.ns, self.n = isc, voc, ns, n
        self.rs, self.rsh, self.alpha, self.noct = rs, rsh, alpha_isc, noct
        a = n * ns * K_B * T_STC / Q
        self.i0_stc = (isc - voc / rsh) / np.expm1(voc / a)

    def params(self, g, tc):
        t = np.asarray(tc, float) + 273.15
        a = self.n * self.ns * K_B * t / Q
        iph = self.isc * (np.asarray(g, float) / 1000.0) * (1 + self.alpha * (t - T_STC))
        i0 = self.i0_stc * (t / T_STC) ** 3 * np.exp(Q * EG / (self.n * K_B) * (1 / T_STC - 1 / t))
        return iph, i0, a

    def current(self, v, g, tc):
        """Module current at terminal voltage v (vectorised)."""
        iph, i0, a = self.params(g, tc)
        rs, rsh = self.rs, self.rsh
        v = np.asarray(v, float)
        lnA = np.log(rs * i0 * rsh / (a * (rs + rsh)))
        B = rsh * (rs * iph + rs * i0 + v) / (a * (rs + rsh))
        w = np.real(wrightomega(lnA + B))
        i = (rsh * (iph + i0) - v) / (rs + rsh) - (a / rs) * w
        return i

    def cell_temp(self, g, t_amb):
        """NOCT model: cell runs (NOCT-20)/800 K hotter per W/m^2."""
        return np.asarray(t_amb, float) + (self.noct - 20.0) / 800.0 * np.asarray(g, float)


class Array:
    """n_series modules per string, n_strings in parallel, uniform irradiance."""

    def __init__(self, module=None, n_series=5, n_strings=2):
        self.m = module or Module()
        self.ns, self.np = n_series, n_strings

    def current(self, v, g, tc):
        i = self.m.current(np.asarray(v, float) / self.ns, g, tc) * self.np
        return np.maximum(i, 0.0)

    def voc(self, g, tc):
        # bisection on the explicit current, good enough and robust
        lo, hi = 0.0, self.ns * self.m.voc * 1.3
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.current(mid, g, tc) > 0:
                lo = mid
            else:
                hi = mid
        return lo

    def mpp(self, g, tc, points=600):
        """Maximum power point by dense search (used as the reference that the
        trackers are graded against)."""
        if g <= 1.0:
            return 0.0, 0.0
        v = np.linspace(0, self.voc(g, tc), points)
        p = v * self.current(v, g, tc)
        k = int(np.argmax(p))
        # refine around the coarse peak
        v2 = np.linspace(v[max(k - 1, 0)], v[min(k + 1, points - 1)], 200)
        p2 = v2 * self.current(v2, g, tc)
        k2 = int(np.argmax(p2))
        return float(v2[k2]), float(p2[k2])

    @property
    def p_stc(self):
        return self.mpp(1000.0, 25.0)[1]
