"""Run every stage in order and write results/summary.txt.

    cd src && python run_all.py
"""
import json
import time

import style  # noqa: F401  (sets the plot style)
import stage1_mppt
import stage2_dcbus
import inverter
import stage4_annual
import stage5_research

t0 = time.time()
r1 = stage1_mppt.run()
print(f"stage 1 done ({time.time() - t0:.0f} s)")
r2 = stage2_dcbus.run()
print(f"stage 2 done ({time.time() - t0:.0f} s)")
r3 = inverter.summary()
json.dump(r3, open("../results/stage3.json", "w"), indent=2)
r4 = stage4_annual.run()
print(f"stage 4 done ({time.time() - t0:.0f} s)")
r5 = stage5_research.run()
print(f"stage 5 done ({time.time() - t0:.0f} s)")

y = str(stage4_annual.MAIN_YEAR)
lines = []
a = r1["array"]
lines.append(f"PV array: {a['p_stc_W']/1000:.2f} kWp, Vmp {a['vmp_stc_V']} V, Voc {a['voc_stc_V']} V at STC")
lines.append(f"Boost settling after one duty step (ms) vs W/m2: {r1['boost_settling_ms_vs_G']} (MPPT period 100 ms); dynamic vs quasi-static P&O at 200 W/m2: {r1['validation_200Wm2_PandO_pct']}")
lines.append("MPPT tracking efficiency (%):")
for test, row in r1["mppt_efficiency_pct"].items():
    lines.append(f"  {test:22s} " + "  ".join(f"{k}: {v}" for k, v in row.items()))
lines.append("Start-up time to first reach 99 % of Pmpp (s): " +
             ", ".join(f"{k}: {v}" for k, v in r1["startup_time_to_first_99pct_s"].items()))
lines.append("")
for key, lab in [("pi_only", "PI only"), ("with_ff", "PI + power feed-forward")]:
    ev = r2[key]["events"]
    lines.append(f"DC bus, {lab}: range {r2[key]['v_min_V']}..{r2[key]['v_max_V']} V; "
                 + "; ".join(f"t={e['t_event_s']}s peak {e['peak_dev_V']} V, settle {e['settle_1pct_ms']} ms" for e in ev))
lines.append(f"Switched vs averaged DC-bus model: {r2['switched_check']}")
lines.append(f"Battery round trip (cells, 20-90 % SOC): {r2['round_trip_eff_pct']}")
lines.append(f"Inverter: peak {r3['eta_peak_pct']} % at {r3['p_at_peak_W']:.0f} W, rated {r3['eta_rated_pct']} %, "
             f"European {r3['eta_euro_pct']} %")
lines.append("")
for yy, res in r4["resource"].items():
    lines.append(f"{yy}: GHI {res['ghi_kWh_m2']} kWh/m2, POA {res['poa_kWh_m2']}, PV at bus {res['pv_bus_kWh']:.0f} kWh "
                 f"({res['specific_yield_kWh_kWp']:.0f} kWh/kWp), load {res['load_kWh']:.0f} kWh")
lines.append("")
lines.append(f"S3 fixed reserve tuned on {y}: {r4['assumptions']['S3_fixed_reserve']}; "
             f"S5/S6 calibrated on {stage4_annual.CAL_YEAR}: {r4['calibration']}")
cols = ["grid_import_kWh", "grid_export_kWh", "unserved_kWh",
        "self_sufficiency_pct", "outage_energy_served_pct", "battery_full_cycles"]
for yy in r4["strategies"]:
    lines.append(f"Strategies {yy}: " + " | ".join(cols))
    for s, k in r4["strategies"][yy].items():
        lines.append(f"  {s}: " + " | ".join(f"{k[c]:.1f}" for c in cols))
lines.append("Strategies, 4 h/day load-shedding (main year):")
for s, k in r4["outage_4h"].items():
    lines.append(f"  {s}: " + " | ".join(f"{k[c]:.1f}" for c in cols))
lines.append("")
lines.append("DP frontier (lam, import kWh, unserved kWh): " +
             "; ".join(f"{f['lam']}: {f['grid_import_kWh']}, {f['unserved_kWh']}" for f in r5["dp_frontier"]))
lines.append(f"Gap of each policy to the DP frontier (% more import at the same unserved energy): {r5['gap_to_dp_pct']}")
lines.append("Outage-hours sweep (import kWh / outage energy served %):")
for h, v in r5["outage_hours_sweep"].items():
    lines.append(f"  {h} h/day, S3 r={v['fixed_r_tuned']}: " + " | ".join(
        f"{n} {v[n]['grid_import_kWh']:.0f} / {v[n].get('outage_energy_served_pct', float('nan'))}"
        + (f" (gap {v[n]['gap_to_dp_pct']} %)" if 'gap_to_dp_pct' in v[n] else "") for n in ["S2", "S3", "S4", "S5", "S6"])
        + f" | DP {v['DP']['grid_import_kWh']:.0f} (unserved {v['DP']['unserved_kWh']})")
lines.append(f"Monte Carlo, 1 h extensions (100 samples): {r5['monte_carlo']}")
lines.append(f"Monte Carlo, 2 h extensions (100 samples): {r5['monte_carlo_ext2h']}")
lines.append(f"Unscheduled-outage rate sweep: {r5['unscheduled_rate_sweep']}")
lines.append(f"Ageing (Wang 2011 cycle model): {r5['ageing']}")
open("../results/summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
