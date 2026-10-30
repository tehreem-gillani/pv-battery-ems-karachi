"""
Stage 3: single-phase hybrid inverter, 5 kW, 230 V AC.

Loss model with a no-load term and a resistive (I^2 R) term:

    P_dc = P_ac + p0 + k * P_ac^2

p0 = 25 W and k are set so that efficiency is 96.5 % at rated power, which
puts the peak (where p0 = k*P^2) near 2 kW at about 97.6 %. These are typical
figures for a transformerless string inverter, not one product's datasheet.
The 400 V bus is above the 325 V peak of 230 V RMS, so a full bridge can make
the AC voltage without a further boost stage.
"""
import numpy as np

P_RATED = 5000.0
P0 = 25.0
K = (P_RATED / 0.965 - P_RATED - P0) / P_RATED ** 2


def dc_input(p_ac):
    p_ac = np.asarray(p_ac, float)
    return np.where(p_ac > 0, p_ac + P0 + K * p_ac ** 2, 0.0)


def efficiency(p_ac):
    p_ac = np.asarray(p_ac, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(p_ac > 0, p_ac / dc_input(p_ac), 0.0)
    return float(e) if e.ndim == 0 else e


def ac_output(p_dc):
    """Inverse of dc_input: AC power that a DC input p_dc can deliver."""
    p_dc = np.asarray(p_dc, float)
    disc = 1 + 4 * K * (p_dc - P0)
    out = np.where(p_dc > P0, (-1 + np.sqrt(np.maximum(disc, 0))) / (2 * K), 0.0)
    return out


def euro_efficiency():
    """European weighted efficiency (weights at 5..100 % of rated power)."""
    w = {0.05: 0.03, 0.10: 0.06, 0.20: 0.13, 0.30: 0.10, 0.50: 0.48, 1.00: 0.20}
    return sum(wt * efficiency(f * P_RATED) for f, wt in w.items())


def summary():
    return {"p0_W": P0, "k_per_W": K,
            "eta_peak_pct": round(100 * efficiency(np.sqrt(P0 / K)), 2),
            "p_at_peak_W": round(float(np.sqrt(P0 / K)), 0),
            "eta_rated_pct": round(100 * efficiency(P_RATED), 2),
            "eta_euro_pct": round(100 * euro_efficiency(), 2)}
