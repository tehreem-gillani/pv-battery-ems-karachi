"""
Hourly energy management over a full year.

Topology (hybrid inverter, DC-coupled battery):
    PV -> boost (MPPT) -> 400 V bus -> 5 kW inverter -> house AC bus <-> grid
                            |
                     bidirectional DC-DC -> LFP pack

The controller sees the *published* load-shedding schedule (`known`); the grid
actually goes down according to `actual`, which can differ (extended blocks,
unscheduled faults). The battery is charged only from PV.

Policies (`policy` dict):
  {"kind": "grid"}                 no PV, no battery
  {"kind": "pv"}                   PV with a grid-tied inverter, no battery; it
                                   must disconnect in outages (anti-islanding)
  {"kind": "fixed", "r": 0.10}     greedy self-consumption above a fixed backup
                                   reserve r (what commercial hybrid inverters
                                   offer as a user setting); r = SOC_MIN is pure
                                   greedy self-consumption
  {"kind": "sched"}                ablation: floor = sum of forecast load in the
                                   scheduled outages of the next HORIZON hours
  {"kind": "sched_robust"}         ablation: "sched" + ROBUST_H h standing reserve
  {"kind": "sched_eb"}             proposed: look-ahead energy-balance reserve.
                                   Walking the next 24 h backwards, the
                                   battery must hold enough for each scheduled
                                   outage minus the PV surplus it can still
                                   collect before that outage. Load and PV are
                                   forecast from the same hour of the previous
                                   7 days (PV derated to 80 %); only data from
                                   earlier days is used.
  {"kind": "sched_eb_robust"}      "sched_eb" that also covers a 1 h extension
                                   after every scheduled block, plus an optional
                                   standing reserve of `standing_h` hours of load
  "margin" and "derate" override FC_MARGIN and PV_FC_DERATE per policy.
In every policy the reserve may be used once the grid is actually down.
"""
import math
import numpy as np
from inverter import P0, K, P_RATED

ETA_BC = 0.97            # bidirectional DC-DC, each direction
P_BAT_MAX = 5000.0       # converter power limit, W (bus side when charging, cell side when discharging)
SOC_MIN, SOC_MAX = 0.10, 0.95
FC_MARGIN, HORIZON = 1.2, 12
FC_DEFAULT = 1500.0      # W, forecast used on day 1 before any history exists
ETA_INV_TYP = 0.965
ROBUST_H = 2
HORIZON_EB = 24
PV_FC_DERATE = 0.8      # PV forecast taken at 80 % of the 7-day same-hour mean

BLOCKS = {0: [], 2: [6, 20], 4: [6, 12, 20, 23], 6: [5, 6, 12, 15, 20, 23],
          8: [5, 6, 9, 12, 15, 18, 20, 23]}


def dc_in(p_ac):
    return p_ac + P0 + K * p_ac * p_ac if p_ac > 0 else 0.0


def ac_out(p_dc):
    if p_dc <= P0:
        return 0.0
    return (-1.0 + math.sqrt(1.0 + 4.0 * K * (p_dc - P0))) / (2.0 * K)


def schedule(pkt_hour, hours_per_day=2):
    """Assumed published load-shedding blocks (PKT clock hour starts)."""
    return np.isin(pkt_hour, BLOCKS[hours_per_day])


def realise(known, rng, p_ext=0.0, rate_unsched=0.0, dur=(1, 3), ext_h=1):
    """Actual outages: each scheduled block is extended by ext_h hours with
    probability p_ext; unscheduled outages arrive as a Poisson process with
    rate_unsched per day, uniform start hour, duration uniform in dur."""
    actual = known.copy()
    n = len(known)
    ends = np.flatnonzero(known[:-1] & ~known[1:]) + 1          # first hour after a block
    ext = ends[rng.random(len(ends)) < p_ext]
    for e in ext:
        actual[e:min(e + ext_h, n)] = True
    n_days = n // 24
    counts = rng.poisson(rate_unsched, n_days)
    for d, c in enumerate(counts):
        for _ in range(c):
            s = d * 24 + rng.integers(0, 24)
            L = rng.integers(dur[0], dur[1] + 1)
            actual[s:min(s + L, n)] = True
    return actual


def daily_profile(x, days_back=7):
    """prof[d, h] = mean of x at hour h over the days_back days before day d.
    A forecast made at any hour of day d uses only prof[d], i.e. only data
    from earlier days, so it is strictly causal."""
    days = x.reshape(-1, 24)
    cs = np.vstack([np.zeros((1, 24)), np.cumsum(days, axis=0)])
    prof = np.empty_like(days)
    for d in range(len(days)):
        lo = max(0, d - days_back)
        prof[d] = (cs[d] - cs[lo]) / (d - lo) if d > lo else np.nan
    return prof


def _forecast_matrix(prof, n, h, default):
    """F[k, i] = forecast made at hour k for hour k+1+i (i < h)."""
    k = np.arange(n)
    j = k[:, None] + 1 + np.arange(h)[None, :]
    day_k = np.minimum(k // 24, len(prof) - 1)
    f = prof[day_k[:, None], j % 24]
    return np.where(np.isnan(f), default, f), j < n


def floor_profile(policy, load, known, e_bat, eta_bat, pv_bus=None):
    n = len(load)
    kind = policy["kind"]
    if kind == "fixed":
        return np.full(n, policy.get("r", SOC_MIN))
    eb = ETA_BC * eta_bat
    to_soc = FC_MARGIN / (e_bat * eta_bat * ETA_BC * ETA_INV_TYP)
    L, valid = _forecast_matrix(daily_profile(load), n, HORIZON_EB, FC_DEFAULT)
    j = np.arange(n)[:, None] + 1 + np.arange(HORIZON_EB)[None, :]
    jc = np.minimum(j, n - 1)
    out_j = known[jc] & valid
    if kind in ("sched", "sched_robust"):
        # ablation: reserve = sum of forecast outage load in the next HORIZON hours
        need = np.where(out_j[:, :HORIZON], L[:, :HORIZON], 0.0).sum(axis=1)
        f = SOC_MIN + to_soc * need
        if kind == "sched_robust":
            f = f + to_soc * L[:, :ROBUST_H].sum(axis=1)
        return np.minimum(f, SOC_MAX)
    # energy-balance reserve (proposed): walk the next 24 h backwards,
    #   E = max(0, E + need_j - charge_j)
    # need_j: forecast outage deficit (load minus PV, both forecast), cell side
    # charge_j: forecast PV surplus the battery can take in hours with grid
    P, _ = _forecast_matrix(daily_profile(pv_bus), n, HORIZON_EB, 0.0)
    P = policy.get("derate", PV_FC_DERATE) * P
    ext = np.zeros_like(out_j)
    if kind == "sched_eb_robust":
        # also cover a 1 h extension right after every scheduled block
        ext[:, 1:] = out_j[:, :-1] & ~out_j[:, 1:]
    outage_like = out_j | ext
    deficit = np.maximum(L - P * ETA_INV_TYP, 0.0) / (eb * ETA_INV_TYP)
    need = np.where(outage_like, deficit, 0.0)
    charge = np.where(~outage_like & valid, np.minimum(np.maximum(P - L / ETA_INV_TYP, 0.0), P_BAT_MAX) * eb, 0.0)
    E = np.zeros(n)
    for i in range(HORIZON_EB - 1, -1, -1):
        E = np.maximum(0.0, E + need[:, i] - charge[:, i])
    sh = policy.get("standing_h", 0)
    if kind == "sched_eb_robust" and sh > 0:
        E = E + L[:, :sh].sum(axis=1) / (eb * ETA_INV_TYP)      # standing reserve
    margin = policy.get("margin", FC_MARGIN)
    return np.minimum(SOC_MIN + margin * E / e_bat, SOC_MAX)


def simulate(pv_bus, load, actual, policy, known=None, e_bat_kwh=10.24, eta_bat=0.994, soc0=0.5):
    n = len(load)
    known = actual if known is None else known
    kind = policy["kind"]
    e_bat = e_bat_kwh * 1000.0
    has_bat = kind in ("fixed", "sched", "sched_robust", "sched_eb", "sched_eb_robust") and e_bat > 0
    has_pv = kind != "grid"
    floor = floor_profile(policy, load, known, e_bat, eta_bat, pv_bus) if has_bat else np.zeros(n)
    soc = soc0
    keys = ["pv_used", "ac_pv", "ac_bat", "grid_imp", "grid_exp", "unserved",
            "curtailed", "pv_to_bat", "e_ch", "e_dis", "soc", "i_bat_A"]
    rec = {k: np.zeros(n) for k in keys}
    eb = ETA_BC * eta_bat
    pv_arr = pv_bus if has_pv else np.zeros(n)
    for k in range(n):
        ld, pv = load[k], pv_arr[k]
        grid_ok = not actual[k]
        inverter_on = has_pv and (grid_ok or has_bat)

        ac_pv = min(ld, ac_out(pv), P_RATED) if inverter_on else 0.0
        dc_pv_load = dc_in(ac_pv)
        surplus = (pv - dc_pv_load) if inverter_on else 0.0

        e_ch = p_in = 0.0
        if has_bat and surplus > 0:
            room = (SOC_MAX - soc) * e_bat
            p_in = min(surplus, P_BAT_MAX, room / eb)
            e_ch = p_in * eb
            soc += e_ch / e_bat
            surplus -= p_in

        ac_exp = 0.0
        if surplus > 0 and grid_ok and inverter_on:
            ac_tot = min(ac_out(dc_pv_load + surplus), P_RATED)
            ac_exp = max(ac_tot - ac_pv, 0.0)
            surplus = max(dc_pv_load + surplus - dc_in(ac_tot), 0.0)   # clipping
        curtailed = max(surplus, 0.0) if inverter_on else pv

        deficit = ld - ac_pv
        ac_bat = e_dis = 0.0
        if has_bat and deficit > 0:
            fl = SOC_MIN if not grid_ok else floor[k]
            if soc > fl:
                dc_avail = min((soc - fl) * e_bat * eb, P_BAT_MAX * eb)
                ac_room = P_RATED - ac_pv - ac_exp
                ac_bat = max(min(deficit, ac_out(dc_pv_load + dc_avail) - ac_pv, ac_room), 0.0)
                dc_used = dc_in(ac_pv + ac_bat) - dc_pv_load
                e_dis = dc_used / eb
                soc -= e_dis / e_bat
                deficit -= ac_bat
        rec["pv_used"][k] = dc_pv_load
        rec["ac_pv"][k] = ac_pv
        rec["ac_bat"][k] = ac_bat
        rec["grid_imp"][k] = deficit if grid_ok else 0.0
        rec["grid_exp"][k] = ac_exp
        rec["unserved"][k] = 0.0 if grid_ok else deficit
        rec["curtailed"][k] = curtailed
        rec["pv_to_bat"][k] = p_in
        rec["e_ch"][k] = e_ch
        rec["e_dis"][k] = e_dis
        rec["soc"][k] = soc
        rec["i_bat_A"][k] = (e_dis - e_ch) / 204.8          # mean cell-side current, A
    rec["outage"] = actual
    rec["known"] = known
    rec["floor"] = floor
    return rec


def kpis(rec, load, pv_bus, e_bat_kwh):
    kwh = lambda x: float(np.sum(x)) / 1000.0
    L = kwh(load)
    out = rec["outage"]
    usable = (SOC_MAX - SOC_MIN) * e_bat_kwh
    pv = kwh(pv_bus)
    d = {"load_kWh": L,
         "pv_bus_kWh": pv,
         "grid_import_kWh": kwh(rec["grid_imp"]),
         "grid_export_kWh": kwh(rec["grid_exp"]),
         "unserved_kWh": kwh(rec["unserved"]),
         "curtailed_kWh": kwh(rec["curtailed"]),
         "self_sufficiency_pct": 100 * kwh(rec["ac_pv"] + rec["ac_bat"]) / L,
         "outage_energy_served_pct": (100 * (1 - kwh(rec["unserved"]) / kwh(load[out]))
                                      if out.any() else 100.0),
         "outage_hours_fully_served_pct": (100 * np.mean(rec["unserved"][out] < 1.0)
                                           if out.any() else 100.0),
         "battery_full_cycles": (kwh(rec["e_dis"]) / usable) if usable > 0 else 0.0,
         "self_consumption_pct": (100 * (kwh(rec["pv_used"]) + kwh(rec["pv_to_bat"])) / pv) if pv > 0 else 0.0}
    return {k: round(float(v), 2) + 0.0 for k, v in d.items()}
