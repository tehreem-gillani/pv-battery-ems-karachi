"""
MPPT algorithms acting on the duty cycle of a boost converter.

With the DC bus held at V_dc, the boost input voltage is v_pv = (1 - d) * V_dc,
so raising d lowers the PV voltage. Every tracker below returns the next duty
cycle from the latest measured PV voltage and current.
"""
import numpy as np


class Tracker:
    name = "base"

    def __init__(self, d0=0.55, d_min=0.05, d_max=0.95):
        self.d = d0
        self.d_min, self.d_max = d_min, d_max
        self.v_prev = None
        self.i_prev = None

    def _clip(self):
        self.d = float(np.clip(self.d, self.d_min, self.d_max))
        return self.d


class PerturbObserve(Tracker):
    """Classic fixed-step P&O [Femia 2005]."""

    def __init__(self, step=0.004, **kw):
        super().__init__(**kw)
        self.step = step
        self.direction = -1          # -1 means lower d, i.e. raise V
        self.name = f"P&O (dd={step})"

    def update(self, v, i):
        p = v * i
        if self.v_prev is not None:
            if p < self.v_prev * self.i_prev:
                self.direction *= -1
        self.v_prev, self.i_prev = v, i
        self.d += self.direction * self.step
        return self._clip()


class IncCond(Tracker):
    """Incremental conductance with fixed step [Esram & Chapman 2007].
    At the MPP dP/dV = 0, which is the same as dI/dV = -I/V."""

    def __init__(self, step=0.004, eps=0.002, **kw):
        super().__init__(**kw)
        self.step, self.eps = step, eps
        self.name = f"IncCond (dd={step})"

    def _direction(self, v, i):
        dv, di = v - self.v_prev, i - self.i_prev
        if abs(dv) < 1e-6:
            if abs(di) < 1e-6:
                return 0
            return 1 if di > 0 else -1           # +1: raise V
        g = di / dv + i / v
        if abs(g) < self.eps:
            return 0
        return 1 if g > 0 else -1

    def _size(self, v, i):
        return self.step

    def update(self, v, i):
        if self.v_prev is None:
            self.v_prev, self.i_prev = v, i
            self.d -= self.step
            return self._clip()
        s = self._direction(v, i)
        size = self._size(v, i)
        self.v_prev_old, self.i_prev_old = self.v_prev, self.i_prev
        self.v_prev, self.i_prev = v, i
        self.d -= s * size                    # raising V means lowering d
        return self._clip()


class VariableStepIncCond(IncCond):
    """IncCond with step proportional to |dP/dV| [Liu et al. 2008]:
    big steps far from the MPP, small steps near it."""

    def __init__(self, n_gain=2.0e-4, step_min=0.0005, step_max=0.02, **kw):
        super().__init__(step=step_min, **kw)
        self.n_gain, self.step_min, self.step_max = n_gain, step_min, step_max
        self.name = "Var-step IncCond"

    def _size(self, v, i):
        dv = v - self.v_prev
        if abs(dv) < 1e-6:
            return self.step_min
        dp_dv = (v * i - self.v_prev * self.i_prev) / dv
        return float(np.clip(self.n_gain * abs(dp_dv), self.step_min, self.step_max))
