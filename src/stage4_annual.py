"""
Stage 4: one year, hourly, with NASA POWER weather for Karachi.

Main results use 2025. 2023 and 2024 are run the same way to check that the
conclusions do not depend on one year's weather.

Policies compared (see ems.py):
  S0 grid only | S1 PV, no battery | S2 greedy self-consumption (10 % floor)
  S3 fixed backup reserve, tuned: the smallest fixed reserve (5 % steps) that
     serves >= 99 % of outage energy in the main year. This is the best a
     user could do with the fixed-reserve setting of today's inverters, and it
     is tuned with hindsight on the test year, which favours S3.
  S4 naive schedule-aware reserve (sum of the next 12 h of outage load)
  S5 look-ahead energy-balance reserve (proposed)
  S6 S5 that also covers a 1 h extension after each scheduled block
The two parameters of S5/S6 (forecast margin, PV derate) are calibrated on
2023 only; 2024 and 2025 are held out.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import style
import solar
import load as loadmod
import ems
from pv import Array

YEARS = [2023, 2024, 2025]
MAIN_YEAR = 2025
SOILING, OTHER_DC = 0.03, 0.02       # assumed: dust on modules; mismatch + DC wiring
ETA_BOOST = 0.98
LABEL = {"S0": "S0 grid only", "S1": "S1 PV, no battery", "S2": "S2 greedy",
         "S3": "S3 fixed reserve (tuned)", "S4": "S4 naive schedule reserve",
         "S5": "S5 energy-balance reserve", "S6": "S6 S5 + extension cover"}
CAL_YEAR = 2023
MARGINS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
DERATES = [1.0, 0.9, 0.8, 0.7]
R_GRID = np.round(np.arange(0.10, 0.951, 0.05), 2)


def mpp_table(arr):
    gs = np.arange(0, 1301, 10.0)
    ts = np.arange(-5, 81, 2.5)
    p = np.array([[arr.mpp(g, t)[1] if g > 0 else 0.0 for t in ts] for g in gs])
    return RegularGridInterpolator((gs, ts), p, bounds_error=False, fill_value=None)


def year_inputs(year, table, eta_mppt, **load_kw):
    w = solar.load_nasa_power(f"../data/nasa_power_karachi_{year}_hourly_LST.csv")
    poa = solar.plane_of_array(w)
    tc = solar.cell_temperature(poa["poa"], w["t_amb"], w["wind"])
    g_eff = poa["poa_eff"] * (1 - SOILING)
    p_mpp = np.maximum(table(np.c_[np.clip(g_eff, 0, 1300), np.clip(tc, -5, 80)]), 0.0)
    p_array = p_mpp * (1 - OTHER_DC) * eta_mppt
    pv_bus = p_array * ETA_BOOST
    ld = loadmod.household_load(w, **load_kw)
    return {"w": w, "poa": poa, "tc": tc, "p_array": p_array, "pv_bus": pv_bus,
            "load": ld, "pkt": solar.pkt_hour(w)}


def policies(r_star, cal):
    m, dr = cal["margin"], cal["derate"]
    return {"S0": {"kind": "grid"}, "S1": {"kind": "pv"}, "S2": {"kind": "fixed", "r": ems.SOC_MIN},
            "S3": {"kind": "fixed", "r": r_star}, "S4": {"kind": "sched"},
            "S5": {"kind": "sched_eb", "margin": m, "derate": dr},
            "S6": {"kind": "sched_eb_robust", "margin": m, "derate": dr}}


def calibrate(d, eta_bat, target=99.0):
    """Pick (margin, derate) on the calibration year: lowest mean grid import
    over 2, 4, 6 and 8 h/day of load-shedding, subject to >= target % outage
    energy served in every one of those cases."""
    best = None
    table = []
    for m in MARGINS:
        for dr in DERATES:
            covs, imps = [], []
            for h in [2, 4, 6, 8]:
                kn = ems.schedule(d["pkt"], h)
                k = ems.kpis(ems.simulate(d["pv_bus"], d["load"], kn,
                                          {"kind": "sched_eb", "margin": m, "derate": dr},
                                          eta_bat=eta_bat), d["load"], d["pv_bus"], 10.24)
                covs.append(k["outage_energy_served_pct"]); imps.append(k["grid_import_kWh"])
            row = {"margin": m, "derate": dr, "min_coverage": min(covs), "mean_import": float(np.mean(imps))}
            table.append(row)
            if row["min_coverage"] >= target and (best is None or row["mean_import"] < best["mean_import"]):
                best = row
    return best, table


def tune_fixed_reserve(d, known, eta_bat, e_bat=10.24, target=99.0):
    for r in R_GRID:
        k = ems.kpis(ems.simulate(d["pv_bus"], d["load"], known, {"kind": "fixed", "r": r},
                                  e_bat_kwh=e_bat, eta_bat=eta_bat), d["load"], d["pv_bus"], e_bat)
        if k["outage_energy_served_pct"] >= target:
            return float(r)
    return float(R_GRID[-1])


def run(outdir_fig="../figures", outdir_res="../results"):
    s1 = json.load(open(f"{outdir_res}/stage1.json"))
    s2 = json.load(open(f"{outdir_res}/stage2.json"))
    eta_mppt = s1["eta_mppt_for_annual"]
    eta_bat = float(np.sqrt(s2["round_trip_eff_pct"]["2.5 kW"] / 100))   # one direction
    arr = Array()
    table = mpp_table(arr)
    data = {y: year_inputs(y, table, eta_mppt) for y in YEARS}
    d = data[MAIN_YEAR]
    known2 = ems.schedule(d["pkt"], 2)
    r_star = tune_fixed_reserve(d, known2, eta_bat)
    cal, cal_table = calibrate(data[CAL_YEAR], eta_bat)
    POL = policies(r_star, cal)
    res = {"assumptions": {"pv_kWp": round(arr.p_stc / 1000, 2), "tilt_deg": 24,
                           "azimuth": "south", "soiling": SOILING, "other_dc_losses": OTHER_DC,
                           "eta_mppt": eta_mppt, "eta_boost": ETA_BOOST, "eta_bc": ems.ETA_BC,
                           "eta_bat_one_way": round(eta_bat, 4), "battery_kWh": 10.24,
                           "soc_min": ems.SOC_MIN, "soc_max": ems.SOC_MAX,
                           "S3_fixed_reserve": r_star,
                           "outage_blocks_pkt": {h: ems.BLOCKS[h] for h in ems.BLOCKS}},
           "eta_bat_one_way": eta_bat, "calibration": cal, "calibration_table": cal_table}

    res["resource"] = {}
    for y, dy in data.items():
        poa = dy["poa"]["poa"].sum() / 1000
        e = dy["pv_bus"].sum() / 1000
        res["resource"][y] = {"ghi_kWh_m2": round(dy["w"]["ghi"].sum() / 1000, 1),
                              "poa_kWh_m2": round(poa, 1), "pv_bus_kWh": round(e, 0),
                              "specific_yield_kWh_kWp": round(e / (arr.p_stc / 1000), 0),
                              "performance_ratio_dc": round(e / (poa * arr.p_stc / 1000), 3),
                              "load_kWh": round(dy["load"].sum() / 1000, 0)}

    # fixed-reserve sweep (main year, 2 h/day)
    res["fixed_reserve_sweep"] = {}
    for r in R_GRID:
        k = ems.kpis(ems.simulate(d["pv_bus"], d["load"], known2, {"kind": "fixed", "r": r},
                                  eta_bat=eta_bat), d["load"], d["pv_bus"], 10.24)
        res["fixed_reserve_sweep"][f"{r:.2f}"] = k

    # all policies, all years, 2 h/day; main year also 4 h/day
    res["strategies"] = {}
    recs = {}
    for y, dy in data.items():
        kn = ems.schedule(dy["pkt"], 2)
        res["strategies"][y] = {}
        for s, pol in POL.items():
            rec = ems.simulate(dy["pv_bus"], dy["load"], kn, pol, eta_bat=eta_bat)
            res["strategies"][y][s] = ems.kpis(rec, dy["load"], dy["pv_bus"], 10.24)
            if y == MAIN_YEAR:
                recs[s] = rec
    kn4 = ems.schedule(d["pkt"], 4)
    res["outage_4h"] = {s: ems.kpis(ems.simulate(d["pv_bus"], d["load"], kn4, pol, eta_bat=eta_bat),
                                    d["load"], d["pv_bus"], 10.24) for s, pol in POL.items()}

    # battery size sweep
    sizes = [2.5, 5.0, 7.5, 10.24, 15.0, 20.0]
    res["battery_sweep"] = {}
    for s in ["S2", "S3", "S5"]:
        res["battery_sweep"][s] = {}
        for e in sizes:
            k = ems.kpis(ems.simulate(d["pv_bus"], d["load"], known2, POL[s], e_bat_kwh=e,
                                      eta_bat=eta_bat), d["load"], d["pv_bus"], e)
            res["battery_sweep"][s][f"{e}"] = k

    # ---------------- figures
    w = d["w"]
    months = np.arange(1, 13)
    pv_m = [d["pv_bus"][w["month"] == m].sum() / 1000 for m in months]
    ld_m = [d["load"][w["month"] == m].sum() / 1000 for m in months]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.bar(months - 0.2, pv_m, 0.4, label="PV at DC bus")
    ax.bar(months + 0.2, ld_m, 0.4, label="House load")
    ax.set_xticks(months); ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_ylabel("Energy (kWh/month)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{outdir_fig}/s4_monthly.png", dpi=200); plt.close(fig)

    for tag, (mo, dy_) in {"june": (6, 12), "jan": (1, 15)}.items():
        idx = np.flatnonzero((w["month"] == mo) & (w["day"] >= dy_) & (w["day"] < dy_ + 2))
        h = np.arange(len(idx)) + 1          # PKT clock hour = HR + 1
        fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
        ax[0].plot(h, d["pv_bus"][idx] / 1000, label="PV")
        ax[0].plot(h, d["load"][idx] / 1000, label="Load")
        ax[0].plot(h, recs["S5"]["grid_imp"][idx] / 1000, label="Grid import (S5)")
        for k in idx[recs["S5"]["outage"][idx]]:
            for a in ax:
                a.axvspan(k - idx[0] + 0.5, k - idx[0] + 1.5, color="0.85", lw=0)
        ax[0].set_ylabel("Power (kW)"); ax[0].set_ylim(0, 5.8)
        ax[0].legend(fontsize=6, ncol=3, loc="upper center")
        ax[1].plot(h, 100 * recs["S2"]["soc"][idx], lw=2.2, alpha=0.45, label="S2 greedy")
        ax[1].plot(h, 100 * recs["S3"]["soc"][idx], lw=0.9, label=f"S3 fixed {int(100*r_star)} %")
        ax[1].plot(h, 100 * recs["S5"]["soc"][idx], lw=0.9, label="S5")
        ax[1].plot(h, 100 * recs["S5"]["floor"][idx], ls=":", c="k", lw=0.8, label="S5 reserve")
        ax[1].set_ylabel("SOC (%)"); ax[1].set_xlabel("Hour (PKT), two days; grey = load-shedding")
        ax[1].legend(fontsize=6, ncol=2); ax[1].set_xticks(range(0, 49, 6))
        fig.tight_layout(); fig.savefig(f"{outdir_fig}/s4_days_{tag}.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for s in ["S2", "S3", "S5"]:
        bs = res["battery_sweep"][s]
        ax[0].plot(sizes, [bs[f"{e}"]["self_sufficiency_pct"] for e in sizes], marker="o", ms=3, label=LABEL[s])
        ax[1].plot(sizes, [bs[f"{e}"]["outage_energy_served_pct"] for e in sizes], marker="o", ms=3, label=LABEL[s])
    ax[0].set_xlabel("Battery capacity (kWh)"); ax[0].set_ylabel("Self-sufficiency (%)")
    ax[1].set_xlabel("Battery capacity (kWh)"); ax[1].set_ylabel("Outage energy served (%)")
    ax[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{outdir_fig}/s4_battery_sweep.png", dpi=200); plt.close(fig)

    with open(f"{outdir_res}/stage4.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps({"r_star": r["assumptions"]["S3_fixed_reserve"], "main": r["strategies"][MAIN_YEAR]}, indent=1))
