"""
LFP battery pack, first-order Thevenin (1RC) equivalent circuit [Chen 2006, Plett 2015].

    v_t = OCV(SOC) - R0*i - v1
    dv1/dt = -v1/(R1*C1) + i/C1
    dSOC/dt = -i / (3600*Q_Ah)          (i > 0 means discharge)

Pack: 64 cells in series, 50 Ah -> 204.8 V nominal, 10.24 kWh.
The cell OCV curve is a representative LFP curve (flat plateau near 3.3 V);
R0, R1 and tau are typical values for a 50 Ah prismatic cell, not a fit to
one product.
"""
import numpy as np

SOC_PTS = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
OCV_PTS = np.array([2.80, 3.10, 3.20, 3.25, 3.28, 3.29, 3.30, 3.31, 3.32, 3.33, 3.35, 3.40, 3.55])


class Pack:
    def __init__(self, n_series=64, q_ah=50.0, r0_cell=1.0e-3, r1_cell=0.6e-3,
                 tau1=40.0, soc0=0.6):
        self.ns, self.q = n_series, q_ah
        self.r0 = r0_cell * n_series
        self.r1 = r1_cell * n_series
        self.c1 = tau1 / self.r1
        self.soc, self.v1 = soc0, 0.0

    @property
    def e_nom_kwh(self):
        return self.ns * 3.2 * self.q / 1000.0

    def ocv(self, soc=None):
        s = self.soc if soc is None else soc
        return self.ns * np.interp(s, SOC_PTS, OCV_PTS)

    def v_terminal(self, i):
        return self.ocv() - self.r0 * i - self.v1

    def step(self, i, dt):
        self.v1 += (-self.v1 / (self.r1 * self.c1) + i / self.c1) * dt
        self.soc -= i * dt / (3600.0 * self.q)
        self.soc = float(np.clip(self.soc, 0.0, 1.0))

    def round_trip_efficiency(self, p_kw, soc_lo=0.2, soc_hi=0.9, dt=1.0):
        """Charge from soc_lo to soc_hi and back at constant power p_kw,
        return energy out / energy in (terminal side)."""
        self.soc, self.v1 = soc_lo, 0.0
        e_in = 0.0
        while self.soc < soc_hi:
            v = self.v_terminal(0.0)
            i = -p_kw * 1000 / max(v, 1.0)
            v = self.v_terminal(i)
            e_in += -v * i * dt
            self.step(i, dt)
        self.v1 = 0.0
        e_out = 0.0
        while self.soc > soc_lo:
            v = self.v_terminal(0.0)
            i = p_kw * 1000 / max(v, 1.0)
            v = self.v_terminal(i)
            e_out += v * i * dt
            self.step(i, dt)
        return e_out / e_in
