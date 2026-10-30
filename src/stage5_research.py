"""
Stage 5: the research questions.

Q1  How far is each causal rule from the best any controller could do?
    -> perfect-foresight DP frontier (grid import vs unserved outage energy)
       with the fixed-reserve curve and S2, S4, S5, S6 on the same axes.
Q2  How does the answer change with the number of load-shedding hours?
    -> 0, 2, 4, 6, 8 h/day; for each, the fixed reserve is re-tuned with
       hindsight (favouring it) and compared with S4 and the DP optimum.
Q3  What if outages do not follow the published schedule, and the load is
    not what the model assumed?
    -> Monte Carlo: blocks extended by 1 h with probability 0.2, unscheduled
       outages at 0.15 per day (1-3 h), AC load x U(0.8, 1.25), base load
       x U(0.85, 1.15), new noise seed. 100 samples. Plus a sweep of the
       unscheduled-outage rate.
Q4  Which policy wears the battery more?
    -> LFP cycle-ageing model of Wang et al. (2011), Arrhenius in temperature,
       power law in Ah throughput, applied hour by hour.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

import style
import ems
import dp
import stage4_annual as s4
from pv import Array

N_MC = 100
LAMS = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 3.0, 10.0]
LAMS_H = [10.0, 1.05, 1.02, 1.0, 0.95]

# Wang et al., J. Power Sources 196 (2011): Q_loss[%] = B exp(-(31700 - 370.3 C)/(R T)) Ah^0.55,
# B = 31630 at C/2 (our C-rates are all below 0.5C). Ah is for their 2.2 Ah cell, so the
# throughput of our 50 Ah cell is scaled to 2.2 Ah-equivalents.
R_GAS, B_WANG, Z_WANG, CELL_AH_WANG = 8.314, 31630.0, 0.55, 2.2
T_BAT_RISE = 3.0          # battery room sits ~3 K above outdoor air (assumption)


def capacity_fade(rec, t_amb, q_ah=50.0):
    """Capacity loss (%) after one year, plus years to 20 % fade at the same duty."""
    i = np.abs(rec["i_bat_A"])                      # mean current per hour, A (pack = cell, 1p)
    c_rate = i / q_ah
    t_k = t_amb + T_BAT_RISE + 273.15
    ea = 31700.0 - 370.3 * c_rate
    # throughput only counts discharge, as in the cycle definition of the model
    ah_dis = np.where(rec["i_bat_A"] > 0, i, 0.0) * (CELL_AH_WANG / q_ah)
    w = (B_WANG * np.exp(-ea / (R_GAS * t_k))) ** (1.0 / Z_WANG)
    x = float(np.sum(w * ah_dis))                  # state variable of the power law
    q1 = x ** Z_WANG
    years_20 = (20.0 / q1) ** (1.0 / Z_WANG) if q1 > 0 else float("inf")
    return round(q1, 2), round(years_20, 1)


def run(outdir_fig="../figures", outdir_res="../results"):
    s1 = json.load(open(f"{outdir_res}/stage1.json"))
    r4 = json.load(open(f"{outdir_res}/stage4.json"))
    eta_mppt = s1["eta_mppt_for_annual"]
    eta_bat = r4["eta_bat_one_way"]
    r_star = r4["assumptions"]["S3_fixed_reserve"]
    POL = s4.policies(r_star, r4["calibration"])
    table = s4.mpp_table(Array())
    d = s4.year_inputs(s4.MAIN_YEAR, table, eta_mppt)
    pv, ld, pkt = d["pv_bus"], d["load"], d["pkt"]
    res = {}

    def kp(rec, load=ld, pvb=pv):
        return ems.kpis(rec, load, pvb, 10.24)

    # ---------------- Q1: frontier
    known = ems.schedule(pkt, 2)
    front = []
    for lam in LAMS:
        r = dp.solve(pv, ld, known, lam, eta_bat=eta_bat)
        front.append({"lam": lam, "grid_import_kWh": round(r["grid_import_kWh"], 1),
                      "unserved_kWh": round(r["unserved_kWh"], 2)})
    res["dp_frontier"] = front
    pts = {}
    for name in ["S2", "S3", "S4", "S5", "S6"]:
        k = kp(ems.simulate(pv, ld, known, POL[name], eta_bat=eta_bat))
        pts[name] = {"grid_import_kWh": k["grid_import_kWh"], "unserved_kWh": k["unserved_kWh"]}
    res["policy_points"] = pts
    fx = r4["fixed_reserve_sweep"]
    # DP import at (about) the unserved level each policy reaches: interpolate the frontier
    fr_u = np.array([f["unserved_kWh"] for f in front])[::-1]
    fr_i = np.array([f["grid_import_kWh"] for f in front])[::-1]
    order = np.argsort(fr_u)
    res["gap_to_dp_pct"] = {n: round(float(100 * (p["grid_import_kWh"] / np.interp(p["unserved_kWh"], fr_u[order], fr_i[order]) - 1)), 1)
                            for n, p in pts.items()}

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot([f["unserved_kWh"] for f in front], [f["grid_import_kWh"] for f in front], "k-", lw=1.2,
            label="DP optimum (perfect foresight)")
    ax.plot([v["unserved_kWh"] for v in fx.values()], [v["grid_import_kWh"] for v in fx.values()],
            "o-", ms=2.5, lw=0.8, c="tab:gray", label="Fixed reserve 10-95 %")
    for r_lab in ["0.10", "0.25", "0.35"]:
        v = fx[r_lab]
        ax.annotate(f"{int(float(r_lab)*100)} %", (v["unserved_kWh"], v["grid_import_kWh"]),
                    textcoords="offset points", xytext=(4, 2), fontsize=6, color="tab:gray")
    for n, mk in [("S4", "v"), ("S5", "s"), ("S6", "^")]:
        ax.plot(pts[n]["unserved_kWh"], pts[n]["grid_import_kWh"], mk, ms=5, label=s4.LABEL[n])
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("Unserved outage energy (kWh/yr)"); ax.set_ylabel("Grid import (kWh/yr)")
    ax.set_ylim(300, 1700); ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(f"{outdir_fig}/s5_frontier.png", dpi=200); plt.close(fig)

    # ---------------- Q2: outage hours
    sweep = {}
    for h in [0, 2, 4, 6, 8]:
        kn = ems.schedule(pkt, h)
        rs = s4.tune_fixed_reserve(d, kn, eta_bat) if h > 0 else ems.SOC_MIN
        row = {"fixed_r_tuned": rs}
        for name, pol in [("S2", POL["S2"]), ("S3", {"kind": "fixed", "r": rs}),
                          ("S4", POL["S4"]), ("S5", POL["S5"]), ("S6", POL["S6"])]:
            k = kp(ems.simulate(pv, ld, kn, pol, eta_bat=eta_bat))
            row[name] = {x: k[x] for x in ["grid_import_kWh", "unserved_kWh", "self_sufficiency_pct",
                                           "outage_energy_served_pct"]}
        fr = [dp.solve(pv, ld, kn, lam, eta_bat=eta_bat) for lam in ([10.0] if h == 0 else LAMS_H)]
        opt = fr[0]
        row["DP"] = {"grid_import_kWh": round(opt["grid_import_kWh"], 1), "unserved_kWh": round(opt["unserved_kWh"], 2)}
        fu = np.array([f["unserved_kWh"] for f in fr]); fi = np.array([f["grid_import_kWh"] for f in fr])
        o = np.argsort(fu)
        for name in ["S3", "S4", "S5", "S6"]:
            p = row[name]
            row[name]["gap_to_dp_pct"] = round(float(100 * (p["grid_import_kWh"] / np.interp(p["unserved_kWh"], fu[o], fi[o]) - 1)), 1)
        sweep[h] = row
    res["outage_hours_sweep"] = sweep
    hs = list(sweep)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot(hs, [sweep[h]["DP"]["grid_import_kWh"] for h in hs], "k-", lw=1.2, label="DP optimum")
    ax.plot(hs, [sweep[h]["S3"]["grid_import_kWh"] for h in hs], "o-", ms=3, label="S3 fixed reserve, tuned")
    ax.plot(hs, [sweep[h]["S4"]["grid_import_kWh"] for h in hs], "v-", ms=3, label="S4 naive schedule reserve")
    ax.plot(hs, [sweep[h]["S5"]["grid_import_kWh"] for h in hs], "s-", ms=3, label="S5 energy-balance reserve")
    ax.plot(hs, [sweep[h]["S6"]["grid_import_kWh"] for h in hs], "^-", ms=3, label="S6 S5 + extension cover")
    ax.plot(hs, [sweep[h]["S2"]["grid_import_kWh"] for h in hs], "--", c="tab:gray", lw=0.8, label="S2 greedy (misses outages)")
    ax.set_xlabel("Scheduled load-shedding (h/day)"); ax.set_ylabel("Grid import (kWh/yr)")
    ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(f"{outdir_fig}/s5_outage_hours.png", dpi=200); plt.close(fig)

    # ---------------- Q3: Monte Carlo
    pols = {n: POL[n] for n in ["S2", "S3", "S4", "S5", "S6"]}
    r_mc_grid = np.round(np.arange(0.25, 0.651, 0.05), 2)

    def monte_carlo(ext_h, seed, with_dp):
        rng = np.random.default_rng(seed)
        mc = {n: {"cov": [], "imp": [], "uns": []} for n in pols}
        mc_fix = {r: {"cov": [], "imp": []} for r in r_mc_grid}
        dpi, dpu = [], []
        for i in range(N_MC):
            ac_s, base_s = rng.uniform(0.8, 1.25), rng.uniform(0.85, 1.15)
            di = s4.year_inputs(s4.MAIN_YEAR, table, eta_mppt, seed=100 + i, ac_scale=ac_s, base_scale=base_s)
            kn = ems.schedule(di["pkt"], 2)
            act = ems.realise(kn, rng, p_ext=0.2, rate_unsched=0.15, ext_h=ext_h)
            for n, pol in pols.items():
                k = ems.kpis(ems.simulate(di["pv_bus"], di["load"], act, pol, known=kn, eta_bat=eta_bat),
                             di["load"], di["pv_bus"], 10.24)
                mc[n]["cov"].append(k["outage_energy_served_pct"])
                mc[n]["imp"].append(k["grid_import_kWh"])
                mc[n]["uns"].append(k["unserved_kWh"])
            for r in r_mc_grid:
                k = ems.kpis(ems.simulate(di["pv_bus"], di["load"], act, {"kind": "fixed", "r": r}, known=kn,
                                          eta_bat=eta_bat), di["load"], di["pv_bus"], 10.24)
                mc_fix[r]["cov"].append(k["outage_energy_served_pct"]); mc_fix[r]["imp"].append(k["grid_import_kWh"])
            if with_dp and i < 30:      # perfect-foresight reference on the first 30 samples
                o = dp.solve(di["pv_bus"], di["load"], act, 10.0, eta_bat=eta_bat)
                dpi.append(o["grid_import_kWh"]); dpu.append(o["unserved_kWh"])
        out = {n: {"coverage_mean": round(float(np.mean(v["cov"])), 2),
                   "coverage_p5": round(float(np.percentile(v["cov"], 5)), 2),
                   "import_mean": round(float(np.mean(v["imp"])), 0),
                   "unserved_mean": round(float(np.mean(v["uns"])), 1)} for n, v in mc.items()}
        # fixed reserve re-tuned with hindsight on these samples to match S6's mean coverage
        target = out["S6"]["coverage_mean"]
        ok = [r for r in r_mc_grid if np.mean(mc_fix[r]["cov"]) >= target]
        if ok:
            r = ok[0]
            out["S3_retuned"] = {"r": float(r), "coverage_mean": round(float(np.mean(mc_fix[r]["cov"])), 2),
                                 "coverage_p5": round(float(np.percentile(mc_fix[r]["cov"], 5)), 2),
                                 "import_mean": round(float(np.mean(mc_fix[r]["imp"])), 0)}
        else:
            out["S3_retuned"] = {"r": None}
        if with_dp:
            out["DP_first30"] = {"import_mean": round(float(np.mean(dpi)), 0),
                                 "unserved_mean": round(float(np.mean(dpu)), 2)}
        return out, mc

    res["monte_carlo"], mc = monte_carlo(ext_h=1, seed=2025, with_dp=True)
    res["monte_carlo_ext2h"], _ = monte_carlo(ext_h=2, seed=2026, with_dp=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.boxplot([mc[n]["cov"] for n in pols], tick_labels=list(pols), widths=0.5,
               flierprops={"markersize": 2})
    ax.set_ylabel("Outage energy served (%)")
    ax.set_xlabel("Policy (100 samples: extended blocks, unscheduled outages, load error)")
    fig.tight_layout(); fig.savefig(f"{outdir_fig}/s5_monte_carlo.png", dpi=200); plt.close(fig)

    # unscheduled-outage rate sweep (30 samples each, same load)
    rate_sw = {}
    for rate in [0.0, 0.1, 0.2, 0.4]:
        rr = np.random.default_rng(7)
        rows = {n: [] for n in pols}
        imps = {n: [] for n in pols}
        for i in range(30):
            act = ems.realise(known, rr, p_ext=0.2, rate_unsched=rate)
            for n, pol in pols.items():
                k = kp(ems.simulate(pv, ld, act, pol, known=known, eta_bat=eta_bat))
                rows[n].append(k["outage_energy_served_pct"]); imps[n].append(k["grid_import_kWh"])
        rate_sw[rate] = {n: {"coverage_mean": round(float(np.mean(rows[n])), 2),
                             "import_mean": round(float(np.mean(imps[n])), 0)} for n in pols}
    res["unscheduled_rate_sweep"] = rate_sw

    # ---------------- Q4: ageing
    ageing = {}
    for n, pol in pols.items():
        rec = ems.simulate(pv, ld, known, pol, eta_bat=eta_bat)
        q1, y20 = capacity_fade(rec, d["w"]["t_amb"])
        ageing[n] = {"fade_pct_year1": q1, "years_to_20pct": y20,
                     "full_cycles": kp(rec)["battery_full_cycles"]}
    res["ageing"] = ageing

    with open(f"{outdir_res}/stage5.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
