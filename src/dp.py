"""
Perfect-foresight benchmark by dynamic programming.

Given the whole year of PV, load and outages in advance, choose the battery
power in every hour to minimise

    J = sum_k [ grid_import_k + lam * unserved_k ]

lam is the price of one kWh lost in an outage relative to one kWh bought from
the grid. Sweeping lam traces the lowest grid import any controller could reach
for a given amount of unserved outage energy (a Pareto frontier). A real
controller cannot know the future exactly, so it cannot do better (up to the
discretisation error noted below).

Method: backward value iteration on a 0.5 % SOC grid (171 points, 10-95 %)
with linear interpolation of the value function, so the battery power itself
is continuous. In each hour the candidate actions are 21 levels of charging
from PV (0 to the PV / power / room limit) and 40 levels of discharging (up to
the amount that covers the whole load, limited by power and stored energy).
Charging from PV while the house imports is allowed, since a hybrid inverter
can do that. Physics and limits match ems.simulate. As for the rules, the SOC at the
end of the year is free.

Because of the grids this is an approximation of the true optimum. Halving
both grids changes the 2 h/day optimum by about 0.1 %, which is the
resolution to keep in mind when reading the gaps.
"""
import numpy as np
from inverter import P0, K, P_RATED
import ems

DS = 0.005
C_CH = np.linspace(0.0, 1.0, 21)
C_DIS = np.linspace(0.0, 1.0, 41)[1:]     # 40 discharge levels


def _ac_out(p_dc):
    p_dc = np.asarray(p_dc, float)
    return np.where(p_dc > P0, (-1.0 + np.sqrt(np.maximum(1.0 + 4.0 * K * (p_dc - P0), 0.0))) / (2.0 * K), 0.0)


def _dc_in(p_ac):
    p_ac = np.asarray(p_ac, float)
    return np.where(p_ac > 0, p_ac + P0 + K * p_ac ** 2, 0.0)


def _candidates(soc, pv, ld, e_bat, eb):
    """Battery charge (DC from bus) and discharge (DC to bus) candidates, shape (n_soc, A)."""
    room = np.maximum((ems.SOC_MAX - soc) * e_bat / eb, 0.0)
    ch_max = np.minimum(np.minimum(pv, ems.P_BAT_MAX), room)
    avail = np.maximum(np.minimum((soc - ems.SOC_MIN) * e_bat * eb, ems.P_BAT_MAX * eb), 0.0)
    need = max(float(_dc_in(min(ld, P_RATED))) - pv, 0.0)
    dis_max = np.minimum(avail, need)
    ch = np.concatenate([C_CH[None, :] * ch_max[:, None], np.zeros((len(soc), len(C_DIS)))], axis=1)
    dis = np.concatenate([np.zeros((len(soc), len(C_CH))), C_DIS[None, :] * dis_max[:, None]], axis=1)
    return ch, dis


def _step(soc, ch, dis, pv, ld, ok, lam, e_bat, eb):
    dc_net = pv - ch + dis
    ac = np.minimum(_ac_out(np.maximum(dc_net, 0.0)), P_RATED)
    short = ld - np.minimum(ac, ld)
    cost = short if ok else lam * short
    nxt = soc[:, None] + (ch * eb - dis / eb) / e_bat
    return cost, np.clip(nxt, ems.SOC_MIN, ems.SOC_MAX), short


def solve(pv_bus, load, actual, lam, e_bat_kwh=10.24, eta_bat=0.994, soc0=0.5):
    e_bat = e_bat_kwh * 1000.0
    eb = ems.ETA_BC * eta_bat
    grid = np.arange(ems.SOC_MIN, ems.SOC_MAX + 1e-9, DS)
    n = len(load)
    V = np.zeros((n + 1, len(grid)))
    for k in range(n - 1, -1, -1):
        pv, ld, ok = pv_bus[k], load[k], not actual[k]
        ch, dis = _candidates(grid, pv, ld, e_bat, eb)
        cost, nxt, _ = _step(grid, ch, dis, pv, ld, ok, lam, e_bat, eb)
        tot = cost + np.interp(nxt, grid, V[k + 1])
        V[k] = tot.min(axis=1)
    soc = np.array([soc0])
    imp = uns = 0.0
    soc_path = np.zeros(n)
    for k in range(n):
        pv, ld, ok = pv_bus[k], load[k], not actual[k]
        ch, dis = _candidates(soc, pv, ld, e_bat, eb)
        cost, nxt, short = _step(soc, ch, dis, pv, ld, ok, lam, e_bat, eb)
        a = int(np.argmin(cost[0] + np.interp(nxt[0], grid, V[k + 1])))
        if ok:
            imp += short[0, a]
        else:
            uns += short[0, a]
        soc = nxt[:, a]
        soc_path[k] = soc[0]
    return {"lam": lam, "grid_import_kWh": imp / 1000.0, "unserved_kWh": uns / 1000.0, "soc": soc_path}
