"""
Stage 1: PV array + boost converter + MPPT.

Part (a) runs the averaged dynamic model of the boost converter to check how
fast the PV voltage settles after a duty-cycle step, at irradiances from 100 to
1000 W/m^2. Settling is slowest in low light (the array's incremental
resistance is high, so it damps the LC input filter less), so the MPPT period
is set above the 100 W/m^2 settling time. Then the long ramp tests in part (b)
can use the quasi-static relation v_pv = (1 - d) * V_dc. As a direct check,
P&O is also run on the full dynamic model at 200 W/m^2 and compared with the
quasi-static result.

Part (b) is an EN 50530-style dynamic test: irradiance ramps between 300 and
1000 W/m^2 at several slopes, plus a slow low-light ramp, and each tracker is
graded by its tracking efficiency  eta = integral(P) / integral(P_mpp).
"""
import json
import numpy as np
import matplotlib.pyplot as plt

from pv import Array
from mppt import PerturbObserve, IncCond, VariableStepIncCond
import style

V_DC = 400.0        # DC bus, held by the battery converter (stage 2)
T_MPPT = 0.10       # MPPT update period, s (10 Hz); must exceed the settling time [Femia 2005]
L_B, C_PV, R_L = 3e-3, 220e-6, 0.10   # boost inductor, input capacitor, inductor ESR
T_AMB = 30.0
NOISE_V, NOISE_I = 0.10, 0.03          # sensor noise (1 sigma), V and A

arr = Array()


def cell_t(g):
    return arr.m.cell_temp(g, T_AMB)


# ---------------------------------------------------------------- part (a)
def dynamic_check(tracker, g=1000.0, t_end=1.2, dt=2e-6, seed=1):
    rng = np.random.default_rng(seed)
    n = int(t_end / dt)
    v, il = 0.6 * V_DC, 0.0
    d = tracker.d
    tc = cell_t(g)
    log_t, log_v, log_i, log_d = [], [], [], []
    next_mppt = T_MPPT
    for k in range(n):
        t = k * dt
        ipv = float(arr.current(v, g, tc))
        dv = (ipv - il) / C_PV
        dil = (v - (1 - d) * V_DC - R_L * il) / L_B
        v += dv * dt
        il = max(il + dil * dt, 0.0)            # diode blocks reverse current
        if t >= next_mppt:
            d = tracker.update(v + rng.normal(0, NOISE_V), ipv + rng.normal(0, NOISE_I))
            next_mppt += T_MPPT
        if k % 50 == 0:
            log_t.append(t); log_v.append(v); log_i.append(ipv); log_d.append(d)
    return np.array(log_t), np.array(log_v), np.array(log_i), np.array(log_d)


def settling_time(dt_step=0.004, g=1000.0, dt=1e-6):
    """Time for v_pv to settle within 2 % of the voltage change after one duty
    step near the MPP."""
    tc = cell_t(g)
    vmp, _ = arr.mpp(g, tc)
    from scipy.optimize import brentq
    # steady state of the averaged model: v = (1-d) V_dc + R_L * i_pv(v)
    def v_ss(d):
        return brentq(lambda x: x - (1 - d) * V_DC - R_L * float(arr.current(x, g, tc)), 1.0, 300.0)
    d0 = brentq(lambda d: v_ss(d) - vmp, 0.45, 0.7)
    v = vmp
    il = float(arr.current(v, g, tc))
    d1 = d0 + dt_step
    v_target = v_ss(d1)
    dv_total = abs(v - v_target)
    t, last_out = 0.0, 0.0
    while t < 0.3:
        ipv = float(arr.current(v, g, tc))
        v += (ipv - il) / C_PV * dt
        il = max(il + (v - (1 - d1) * V_DC - R_L * il) / L_B * dt, 0.0)
        t += dt
        if abs(v - v_target) > 0.02 * dv_total:
            last_out = t
    return last_out


# ---------------------------------------------------------------- part (b)
def ramp_profile(slope, g_lo=300.0, g_hi=1000.0, dwell=10.0):
    t_ramp = (g_hi - g_lo) / slope
    seg = [(dwell, g_lo, g_lo), (t_ramp, g_lo, g_hi), (dwell, g_hi, g_hi),
           (t_ramp, g_hi, g_lo), (dwell, g_lo, g_lo)]
    gs = []
    for dur, a, b in seg:
        n = max(int(round(dur / T_MPPT)), 1)
        gs.extend(np.linspace(a, b, n, endpoint=False))
    return np.array(gs)


_mpp_cache = {}


def p_mpp(g):
    key = round(float(g), 1)
    if key not in _mpp_cache:
        _mpp_cache[key] = arr.mpp(key, cell_t(key))[1]
    return _mpp_cache[key]


_voc_cache = {}


def v_oc(g):
    key = round(float(g), 1)
    if key not in _voc_cache:
        _voc_cache[key] = arr.voc(key, cell_t(key))
    return _voc_cache[key]


def quasi_static(tracker, gs, seed=2, noise=(NOISE_V, NOISE_I)):
    rng = np.random.default_rng(seed)
    p, pm, vv = np.zeros(len(gs)), np.zeros(len(gs)), np.zeros(len(gs))
    for k, g in enumerate(gs):
        # above Voc the boost diode stops conducting and the array sits at Voc
        v = min((1 - tracker.d) * V_DC, v_oc(g))
        i = float(arr.current(v, g, cell_t(g)))
        p[k], vv[k] = v * i, v
        pm[k] = p_mpp(g)
        tracker.update(v + rng.normal(0, noise[0]), i + rng.normal(0, noise[1]))
    return p, pm, vv


def make_trackers():
    return [PerturbObserve(step=0.004), PerturbObserve(step=0.010),
            IncCond(step=0.004), VariableStepIncCond()]


def run(outdir_fig="../figures", outdir_res="../results"):
    res = {}
    vmp, pmp = arr.mpp(1000.0, 25.0)
    res["array"] = {"p_stc_W": round(pmp, 1), "vmp_stc_V": round(vmp, 1),
                    "voc_stc_V": round(arr.voc(1000, 25), 1),
                    "isc_stc_A": round(float(arr.current(0, 1000, 25)), 2)}

    # I-V and P-V curves
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for g in [200, 400, 600, 800, 1000]:
        tc = cell_t(g)
        v = np.linspace(0, arr.voc(g, tc), 300)
        i = arr.current(v, g, tc)
        ax[0].plot(v, i, label=f"{g}")
        ax[1].plot(v, v * i / 1000)
    ax[0].set_xlabel("Array voltage (V)"); ax[0].set_ylabel("Current (A)")
    ax[1].set_xlabel("Array voltage (V)"); ax[1].set_ylabel("Power (kW)")
    ax[0].legend(title="W/m$^2$", fontsize=7, title_fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{outdir_fig}/s1_iv_pv.png", dpi=200); plt.close(fig)

    # (a) dynamic validation
    ts = settling_time()
    res["boost_settling_ms"] = round(ts * 1000, 2)
    res["boost_settling_ms_vs_G"] = {int(g): round(settling_time(g=g) * 1000, 1)
                                     for g in [100, 200, 500, 1000]}
    # same tracker, same irradiance: full averaged model vs quasi-static
    g_val, t_val = 200.0, 12.0
    tr = PerturbObserve(step=0.004)
    tr.d = 1 - arr.mpp(g_val, cell_t(g_val))[0] / V_DC + 0.02
    td, vd, idd, _ = dynamic_check(tr, g=g_val, t_end=t_val, dt=4e-6)
    pm200 = arr.mpp(g_val, cell_t(g_val))[1]
    keep = td > 2.0
    eta_dyn = float(np.mean(vd[keep] * idd[keep]) / pm200)
    tr = PerturbObserve(step=0.004)
    tr.d = 1 - arr.mpp(g_val, cell_t(g_val))[0] / V_DC + 0.02
    p_qs, pm_qs, _ = quasi_static(tr, np.full(int(t_val / T_MPPT), g_val))
    k0 = int(2.0 / T_MPPT)
    eta_qs = float(p_qs[k0:].sum() / pm_qs[k0:].sum())
    res["validation_200Wm2_PandO_pct"] = {"averaged_dynamic": round(100 * eta_dyn, 2),
                                          "quasi_static": round(100 * eta_qs, 2)}
    t, v, i, d = dynamic_check(PerturbObserve(step=0.004))
    vmp_op, _ = arr.mpp(1000.0, cell_t(1000.0))
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot(t, v, lw=0.8, label="$v_{pv}$ (averaged model)")
    ax.axhline(vmp_op, ls="--", c="k", lw=0.8, label="$V_{mpp}$")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("PV voltage (V)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{outdir_fig}/s1_dynamic_startup.png", dpi=200); plt.close(fig)

    # (b) ramp tests
    slopes = [10, 30, 50, 100]
    table = {}
    for s in slopes:
        gs = ramp_profile(s)
        row = {}
        for tr in make_trackers():
            p, pm, _ = quasi_static(tr, gs)
            row[tr.name] = round(100 * p.sum() / pm.sum(), 2)
        table[f"{s} W/m2/s"] = row
    # slow low-light ramp 100 -> 500 -> 100 at 1 W/m^2/s
    gs = ramp_profile(1.0, 100.0, 500.0, dwell=30.0)
    row = {}
    for tr in make_trackers():
        p, pm, _ = quasi_static(tr, gs)
        row[tr.name] = round(100 * p.sum() / pm.sum(), 2)
    table["low light 1 W/m2/s"] = row
    # static efficiency at 1000 and 200 W/m^2 (60 s each)
    for g in [1000.0, 200.0]:
        gs = np.full(int(60 / T_MPPT), g)
        row = {}
        for tr in make_trackers():
            p, pm, _ = quasi_static(tr, gs)
            k0 = int(10 / T_MPPT)   # skip the start-up
            row[tr.name] = round(100 * p[k0:].sum() / pm[k0:].sum(), 2)
        table[f"static {int(g)} W/m2"] = row
    res["mppt_efficiency_pct"] = table

    # start-up: from d = 0.46 (about 95 % of Voc) at 1000 W/m^2,
    # time until power first reaches 99 % of P_mpp
    startup = {}
    gs = np.full(int(20 / T_MPPT), 1000.0)
    for tr in make_trackers():
        tr.d = 0.46
        p, pm, _ = quasi_static(tr, gs)
        hit = np.flatnonzero(p >= 0.99 * pm)
        startup[tr.name] = round(hit[0] * T_MPPT, 2) if len(hit) else None
    res["startup_time_to_first_99pct_s"] = startup

    # sensor-noise sensitivity at 200 W/m^2 (static, 60 s)
    sweep = {}
    gs = np.full(int(60 / T_MPPT), 200.0)
    k0 = int(10 / T_MPPT)
    for nv in [0.0, 0.05, 0.10, 0.20, 0.40]:
        row = {}
        for mk in [lambda: PerturbObserve(step=0.004), lambda: IncCond(step=0.004),
                   lambda: VariableStepIncCond()]:
            effs = []
            for seed in range(5):              # average over 5 noise realisations
                tr = mk()
                p, pm, _ = quasi_static(tr, gs, seed=seed, noise=(nv, nv * 0.3))
                effs.append(p[k0:].sum() / pm[k0:].sum())
            row[tr.name] = round(100 * float(np.mean(effs)), 2)
        sweep[f"{nv:.2f} V"] = row
    res["noise_sweep_200Wm2_pct"] = sweep
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    xs = [float(k.split()[0]) for k in sweep]
    for name in next(iter(sweep.values())):
        ax.plot(xs, [sweep[k][name] for k in sweep], marker="o", ms=3, label=name)
    ax.set_xlabel("Voltage noise, 1$\\sigma$ (V); current noise 0.3 A per V")
    ax.set_ylabel("Tracking efficiency (%)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{outdir_fig}/s1_noise_sweep.png", dpi=200); plt.close(fig)

    # figure: power during the 50 W/m^2/s ramp, P&O vs IncCond
    gs = ramp_profile(50)
    tt = np.arange(len(gs)) * T_MPPT
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    _, pm, _ = quasi_static(PerturbObserve(step=0.004), gs)
    ax.plot(tt, pm / 1000, c="k", lw=1.0, label="$P_{mpp}$")
    for tr in [PerturbObserve(step=0.004), VariableStepIncCond()]:
        p, _, _ = quasi_static(tr, gs)
        ax.plot(tt, p / 1000, lw=0.8, label=tr.name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("PV power (kW)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{outdir_fig}/s1_ramp_power.png", dpi=200); plt.close(fig)

    # figure: operating voltage on the rising ramp, shows the P&O drift
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    vm = np.array([arr.mpp(g, cell_t(g))[0] for g in gs])
    ax.plot(tt, vm, c="k", lw=1.0, label="$V_{mpp}$")
    for tr in [PerturbObserve(step=0.004), VariableStepIncCond()]:
        _, _, vv = quasi_static(tr, gs)
        ax.plot(tt, vv, lw=0.8, label=tr.name)
    ax.set_xlim(8, 26); ax.set_ylim(150, 240)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("PV voltage (V)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{outdir_fig}/s1_ramp_voltage.png", dpi=200); plt.close(fig)

    # the tracker with the best mean over all tests feeds the annual model
    names = list(next(iter(table.values())).keys())
    means = {n: float(np.mean([table[k][n] for k in table])) for n in names}
    best = max(means, key=means.get)
    res["mean_efficiency_pct"] = {n: round(v, 2) for n, v in means.items()}
    res["tracker_for_annual"] = best
    res["eta_mppt_for_annual"] = round(means[best] / 100, 4)
    with open(f"{outdir_res}/stage1.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
